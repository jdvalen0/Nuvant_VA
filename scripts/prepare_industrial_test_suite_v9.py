import cv2
import os
import numpy as np
import shutil
import random
from pathlib import Path

# --- CONFIGURACIÓN V9 (ZERO-PITY) ---
SOURCE_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/images (2)")
OUTPUT_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/INDUSTRIAL_TEST_SUITE_V9")

# LISTAS BLANCAS QUIRÚRGICAS (Verificadas 1-a-1)
SOURCES = {
    # 1. ANILLOS (9 Archivos)
    # EXCLUSION: IMG_2393 (Cuero Arrugado)
    "REF_01_Anillos": {
        "type": "img_list",
        "files": [
            "IMG_2385.JPEG", "IMG_2386.JPEG", "IMG_2387.JPEG", "IMG_2388.JPEG",
            "IMG_2389.JPEG", "IMG_2390.JPEG", "IMG_2391.JPEG", "IMG_2392.JPEG",
            # SKIP 2393
            "IMG_2394.JPEG"
        ]
    },
    
    # 2. TRAMA FINA (13 Archivos)
    # EXCLUSION: IMG_2410 (Puntos Blancos)
    # EXCLUSION: IMG_2416+ (Lisa/Cuero)
    "REF_02_Trama": {
        "type": "img_list",
        "files": [
            "IMG_2402.JPEG", "IMG_2403.JPEG", "IMG_2404.JPEG", "IMG_2405.JPEG",
            "IMG_2406.JPEG", "IMG_2407.JPEG", "IMG_2408.JPEG", "IMG_2409.JPEG",
            # SKIP 2410
            "IMG_2411.JPEG", "IMG_2412.JPEG", "IMG_2413.JPEG", "IMG_2414.JPEG", "IMG_2415.JPEG"
        ]
    },

    # 3. PIEL A (RUGOSA): 05:01 - 05:50 (Estable)
    "REF_03_Piel_Rugosa": {
        "type": "frame_range_explicit",
        "start": "frame_20260115-050100",
        "end": "frame_20260115-055000"
    },

    # 4. PIEL B (SUAVE): 06:20 - 06:40 (Estable)
    "REF_04_Piel_Suave": {
        "type": "frame_range_explicit",
        "start": "frame_20260115-062000",
        "end": "frame_20260115-064000"
    },

    # 5. PIEL C (INTERMEDIA): 06:50 - 07:00 (Ajuste previo mantenido)
    "REF_05_Piel_Intermedia": {
        "type": "frame_range_explicit",
        "start": "frame_20260115-065000",
        "end": "frame_20260115-070000"
    },

    # 6. GEOMETRICA: 12:50 - 14:50 (Inicio seguro)
    "REF_06_Geometrica": {
        "type": "frame_range_explicit",
        "start": "frame_20260115-125000",
        "end": "frame_20260115-145000"
    },

    # 7. RADIAL: 17:35 - 23:59 (Estable)
    "REF_07_Radial": {
        "type": "frame_range_explicit",
        "start": "frame_20260115-173500",
        "end": "frame_20260115-235959"
    }
}

def apply_augmentation(img, forceful=False):
    augs = [img.copy()]
    augs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augs.append(cv2.rotate(img, cv2.ROTATE_180))
    augs.append(cv2.flip(img, 1))
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2].astype(np.int16)
    
    shifts = [10, -10] if not forceful else [20, -20, 30] 
    for s in shifts:
        v_mod = np.clip(v + s, 0, 255).astype(np.uint8)
        new_hsv = hsv.copy()
        new_hsv[:,:,2] = v_mod
        augs.append(cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR))
        
    return augs

def apply_industrial_defect(image, defect_type='stain'):
    img = image.copy()
    h, w = img.shape[:2]
    
    if defect_type == 'hole':
        center = (random.randint(50,w-50), random.randint(50,h-50))
        cv2.circle(img, center, random.randint(15,35), (10,10,10), -1)
        
    elif defect_type == 'stain_dark':
        mask = np.zeros((h,w), dtype=np.uint8)
        center = (random.randint(50,w-50), random.randint(50,h-50))
        cv2.circle(mask, center, random.randint(30,70), 255, -1)
        mask = cv2.GaussianBlur(mask, (31,31), 0)
        overlay = img.copy()
        overlay = (overlay * 0.5).astype(np.uint8) 
        alpha = (mask / 255.0)[:,:,np.newaxis]
        img = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
    return img

def main():
    print(f"--- ☢️ Generando V9 ZERO-PITY ---")
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    all_frames = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith("frame_")])

    for ref_name, cfg in SOURCES.items():
        print(f"📦 Procesando {ref_name}...")
        ref_path = OUTPUT_DIR / ref_name
        for sub in ["01_Entrenamiento_NORMAL", "02_Validacion_BUENAS", "03_Validacion_ADULTERADAS"]:
            (ref_path / sub).mkdir(parents=True, exist_ok=True)
            
        raw_imgs = []
        
        if cfg["type"] == "img_list":
            for fname in cfg["files"]:
                p = SOURCE_DIR / fname
                if p.exists():
                    i = cv2.imread(str(p))
                    if i is not None: raw_imgs.append(i)
                else:
                    print(f"   ❌ ERROR FATAL: No existe {fname}")
                    return # Stop immediately if a whitelist file is missing
                    
        elif cfg["type"] == "frame_range_explicit":
            candidates = [f for f in all_frames if f >= cfg["start"] and f <= cfg["end"]]
            print(f"   - Frames en rango: {len(candidates)}")
            if len(candidates) > 0:
                step = max(1, len(candidates) // 50)
                selection = candidates[::step]
                for f in selection:
                    i = cv2.imread(str(SOURCE_DIR / f))
                    if i is not None: raw_imgs.append(i)

        if not raw_imgs:
            print(f"❌ ERROR CRÍTICO: No hay imágenes para {ref_name}")
            continue

        print(f"   - Fuente Certificada: {len(raw_imgs)} imágenes.")

        # Generación (30 Train / 15 Val OK / 15 Val Fail)
        count = 0
        while count < 30:
            base = random.choice(raw_imgs)
            aug = random.choice(apply_augmentation(base, forceful=True))
            cv2.imwrite(str(ref_path / "01_Entrenamiento_NORMAL" / f"train_{count:02d}.png"), aug)
            count += 1
        
        count = 0
        while count < 15:
            base = random.choice(raw_imgs)
            aug = random.choice([base, cv2.flip(base, 1)])
            cv2.imwrite(str(ref_path / "02_Validacion_BUENAS" / f"val_ok_{count:02d}.png"), aug)
            count += 1
            
        count = 0
        while count < 15:
            base = random.choice(raw_imgs)
            aug = base.copy()
            bad = apply_industrial_defect(aug, random.choice(['stain_dark', 'hole']))
            cv2.imwrite(str(ref_path / "03_Validacion_ADULTERADAS" / f"val_fail_{count:02d}.png"), bad)
            count += 1
            
    print("\n✅ V9 ZERO-PITY COMPLETADO.")

if __name__ == "__main__":
    main()
