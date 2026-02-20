import cv2
import os
import numpy as np
import shutil
import random
from pathlib import Path

# --- CONFIGURACIÓN V10 (ATOMIC) ---
SOURCE_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/images (2)")
OUTPUT_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/INDUSTRIAL_TEST_SUITE_V10")

# LISTAS ATÓMICAS (VERIFICADAS VISUALMENTE UNA POR UNA)
SOURCES = {
    # 1. ANILLOS (9 Archivos)
    "REF_01_Anillos": {
        "type": "atomic_list",
        "files": [
            "IMG_2385.JPEG", "IMG_2386.JPEG", "IMG_2387.JPEG", "IMG_2388.JPEG",
            "IMG_2389.JPEG", "IMG_2390.JPEG", "IMG_2391.JPEG", "IMG_2392.JPEG",
            "IMG_2394.JPEG" # EXCLUIDA 2393 (Cuero)
        ]
    },
    
    # 2. TRAMA FINA (13 Archivos)
    "REF_02_Trama": {
        "type": "atomic_list",
        "files": [
            "IMG_2402.JPEG", "IMG_2403.JPEG", "IMG_2404.JPEG", "IMG_2405.JPEG",
            "IMG_2406.JPEG", "IMG_2407.JPEG", "IMG_2408.JPEG", "IMG_2409.JPEG",
            # EXCLUIDA 2410 (Defecto)
            "IMG_2411.JPEG", "IMG_2412.JPEG", "IMG_2413.JPEG", "IMG_2414.JPEG", 
            "IMG_2415.JPEG"
        ]
    },

    # 3. PIEL A (RUGOSA)
    "REF_03_Piel_Rugosa": {
        "type": "atomic_list",
        "files": [
            "frame_20260115-053120.png",
            "frame_20260115-053320.png",
            "frame_20260115-053520.png",
            "frame_20260115-053720.png",
            "frame_20260115-053920.png"
        ]
    },

    # 4. PIEL B (SUAVE)
    "REF_04_Piel_Suave": {
        "type": "atomic_list",
        "files": [
            "frame_20260115-063122.png",
            "frame_20260115-063321.png",
            "frame_20260115-063521.png",
            "frame_20260115-063722.png",
            "frame_20260115-063922.png"
        ]
    },

    # 5. PIEL C (INTERMEDIA) - CORREGIDA (Era negra/Radial)
    "REF_05_Piel_Intermedia": {
        "type": "atomic_list",
        "files": [
            "frame_20260115-055125.png",
            "frame_20260115-055320.png",
            "frame_20260115-055520.png",
            "frame_20260115-055720.png",
            "frame_20260115-055921.png"
        ]
    },

    # 6. GEOMETRICA
    "REF_06_Geometrica": {
        "type": "atomic_list",
        "files": [
            "frame_20260115-133134.png",
            "frame_20260115-133334.png",
            "frame_20260115-133534.png",
            "frame_20260115-133734.png",
            "frame_20260115-133934.png"
        ]
    },

    # 7. RADIAL - CORREGIDA (Era negra/Geometrica al final)
    "REF_07_Radial": {
        "type": "atomic_list",
        "files": [
            "frame_20260115-174142.png",
            "frame_20260115-174342.png",
            "frame_20260115-174542.png",
            "frame_20260115-174742.png",
            "frame_20260115-174943.png"
        ]
    }
}

def apply_augmentation(img, forceful=False):
    augs = [img.copy()]
    augs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augs.append(cv2.rotate(img, cv2.ROTATE_180))
    augs.append(cv2.flip(img, 1))
    
    # Solo cambios de valor (brillo), no matriz de color
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
    print(f"--- ☢️ Generando V10 ATOMIC SUITE ---")
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    for ref_name, cfg in SOURCES.items():
        print(f"📦 Procesando {ref_name}...")
        ref_path = OUTPUT_DIR / ref_name
        for sub in ["01_Entrenamiento_NORMAL", "02_Validacion_BUENAS", "03_Validacion_ADULTERADAS"]:
            (ref_path / sub).mkdir(parents=True, exist_ok=True)
            
        raw_imgs = []
        
        # LÓGICA ATÓMICA: Cargar SOLO los archivos listados.
        if cfg["type"] == "atomic_list":
            for fname in cfg["files"]:
                p = SOURCE_DIR / fname
                if p.exists():
                    i = cv2.imread(str(p))
                    if i is not None: raw_imgs.append(i)
                    else: print(f"   ⚠️ Leído fallido: {fname}")
                else:
                    print(f"   ❌ ERROR FATAL: No existe {fname}")
                    return # Stop immediately

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
            
    print("\n✅ V10 ATOMIC COMPLETADO.")

if __name__ == "__main__":
    main()
