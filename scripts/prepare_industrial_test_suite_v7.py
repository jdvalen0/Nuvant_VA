import cv2
import os
import numpy as np
import shutil
import random
from pathlib import Path

# --- CONFIGURACIÓN "V7 - RE-CLASSIFICATION" ---
SOURCE_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/images (2)")
OUTPUT_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/INDUSTRIAL_TEST_SUITE_V7")

# LISTAS BLANCAS EXPLICITAS (Visualmente Verificadas)

SOURCES = {
    "REF_01_Anillos": {
        "type": "img_list",
        "files": [f"IMG_{i}.JPEG" for i in range(2385, 2395)] 
    },
    # UNIFICACIÓN: Trama Fina + Lisa = "Sintetico Gris"
    # Ambos son la misma tela gris semitexturizada.
    "REF_02_Sintetico_Gris": {
        "type": "img_list",
        # 2402-2415 (Trama) + 2416-2426 (Lisa)
        # Excluyendo basura (2395-2401)
        "files": [f"IMG_{i}.JPEG" for i in range(2402, 2427)]
    },
    # SPLIT PIEL NEGRA: 3 Referencias
    "REF_03_Piel_Rugosa": { # Contrast ~2.5
        "type": "frame_range_explicit",
        "start": "frame_20260115-050100", 
        "end": "frame_20260115-055000"
    },
    "REF_04_Piel_Suave": { # Contrast ~0.8 (Smoother)
        "type": "frame_range_explicit",
        "start": "frame_20260115-062000",
        "end": "frame_20260115-064000"
    },
    "REF_05_Piel_Intermedia": { # Contrast ~3.8 -> 0.7 (Mixed/Transitional but valid)
         # Using tight window around 07:05 - 07:20
        "type": "frame_range_explicit",
        "start": "frame_20260115-070000",
        "end": "frame_20260115-072000"
    },
    "REF_06_Geometrica": {
        "type": "frame_range_explicit",
        "start": "frame_20260115-123000",
        "end": "frame_20260115-145000"
    },
    "REF_07_Radial": {
        "type": "frame_range_explicit",
        "start": "frame_20260115-173000",
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
    shifts = [20, -20, 35, -35] if forceful else [15, -15]
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
        cv2.circle(img, center, random.randint(10,30), (5,5,5), -1)
        
    elif defect_type == 'stain_dark':
        mask = np.zeros((h,w), dtype=np.uint8)
        center = (random.randint(50,w-50), random.randint(50,h-50))
        cv2.circle(mask, center, random.randint(20,60), 255, -1)
        mask = cv2.GaussianBlur(mask, (51,51), 0)
        overlay = img.copy()
        overlay = (overlay * 0.6).astype(np.uint8)
        alpha = (mask / 255.0)[:,:,np.newaxis]
        img = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
    return img

def main():
    print(f"--- 🚨 Generando V7 RE-CLASSIFICATION ---")
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
                    
        elif cfg["type"] == "frame_range_explicit":
            candidates = [f for f in all_frames if f >= cfg["start"] and f <= cfg["end"]]
            print(f"   - Frames candidatos: {len(candidates)}")
            # Muestreo adaptativo (max 40)
            step = max(1, len(candidates) // 40)
            selection = candidates[::step]
            for f in selection:
                i = cv2.imread(str(SOURCE_DIR / f))
                if i is not None: raw_imgs.append(i)

        if not raw_imgs:
            print(f"❌ ERROR CRÍTICO: No hay imágenes para {ref_name}")
            continue

        print(f"   - Fuente limpia: {len(raw_imgs)} imágenes.")

        # Generación (30 Train / 15 Val OK / 15 Val Fail)
        # ... (Logica Estandar) ...
        count = 0
        while count < 30:
            base = random.choice(raw_imgs)
            for aug in apply_augmentation(base, forceful=True):
                if count >= 30: break
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
            bad = apply_industrial_defect(aug, 'stain_dark')
            cv2.imwrite(str(ref_path / "03_Validacion_ADULTERADAS" / f"val_fail_{count:02d}.png"), bad)
            count += 1
            
    print("\n✅ V7 COMPLETADO. Nueva Taxonomía Aplicada.")

if __name__ == "__main__":
    main()
