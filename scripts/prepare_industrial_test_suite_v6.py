import cv2
import os
import numpy as np
import shutil
import random
from pathlib import Path

# --- CONFIGURACIÓN "ZERO-TOLERANCE" V6 (FINAL VERIFIED) ---
SOURCE_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/images (2)")
OUTPUT_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/INDUSTRIAL_TEST_SUITE_V6")

# LISTAS BLANCAS EXPLICITAS (Visualmente Verificadas una por una)
SOURCES = {
    "REF_01_Anillos": {
        "type": "img_list",
        # 2385-2394 confirmados visualmente.
        "files": [f"IMG_{i}.JPEG" for i in range(2385, 2395)] 
    },
    "REF_02_Trama_Fina": {
        "type": "img_list",
        # REVISION MANUAL EXHAUSTIVA:
        # 2395-2401: BASURA (Cuero/Arrugas). DESCARTADOS.
        # 2402-2415: TRAMA LIMPIA (Grid). CONFIRMADOS.
        # 2416: Start of LISA.
        "files": [f"IMG_{i}.JPEG" for i in range(2402, 2416)]
    },
    "REF_03_Lisa": {
        "type": "img_list",
        # 2416 en adelante es Lisa.
        "files": [f"IMG_{i}.JPEG" for i in range(2416, 2426)]
    },
    "REF_04_Piel_Negra": {
        "type": "frame_range_explicit",
        # VALIDACION VISUAL:
        # 05:01 (frame_20260115-050119) -> Piel Negra.
        # 07:21 (frame_20260115-072123) -> Piel Negra.
        "start": "frame_20260115-050100",
        "end": "frame_20260115-072959"
    },
    "REF_05_Geometrica": {
        "type": "frame_range_explicit",
        # VALIDACION PREVIA: 12:30 - 14:50 seguro.
        "start": "frame_20260115-123000",
        "end": "frame_20260115-145000"
    },
    "REF_06_Radial": {
        "type": "frame_range_explicit",
        # VALIDACION PREVIA: 17:30 seguro.
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

    elif defect_type == 'scratch':
        pt1 = (random.randint(20,w-20), random.randint(20,h-20))
        pt2 = (pt1[0] + random.randint(-80,80), pt1[1] + random.randint(-20,20))
        cv2.line(img, pt1, pt2, (200,200,200), 2)
        
    return img

def main():
    print(f"--- 🛡️ Generando V6 ZERO-TOLERANCE (FINAL) ---")
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    all_frames = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith("frame_")])

    for ref_name, cfg in SOURCES.items():
        print(f"📦 Procesando {ref_name}...")
        ref_path = OUTPUT_DIR / ref_name
        for sub in ["01_Entrenamiento_NORMAL", "02_Validacion_BUENAS", "03_Validacion_ADULTERADAS"]:
            (ref_path / sub).mkdir(parents=True, exist_ok=True)
            
        # Collect Raw Images
        raw_imgs = []
        
        if cfg["type"] == "img_list":
            for fname in cfg["files"]:
                p = SOURCE_DIR / fname
                if p.exists():
                    i = cv2.imread(str(p))
                    if i is not None: raw_imgs.append(i)
                else:
                    print(f"   ⚠️ Falta archivo: {fname}")
                    
        elif cfg["type"] == "frame_range_explicit":
            candidates = [f for f in all_frames if f >= cfg["start"] and f <= cfg["end"]]
            print(f"   - Frames candidatos verificados: {len(candidates)}")
            # Sample every 3rd frame
            selection = candidates[::3][:60]
            for f in selection:
                i = cv2.imread(str(SOURCE_DIR / f))
                if i is not None: raw_imgs.append(i)

        if not raw_imgs:
            print(f"❌ ERROR CRÍTICO: No hay imágenes para {ref_name}")
            continue

        print(f"   - Fuente limpia: {len(raw_imgs)} imágenes base.")

        # 1. Entrenamiento (30)
        count = 0
        while count < 30:
            base = random.choice(raw_imgs)
            for aug in apply_augmentation(base, forceful=True):
                if count >= 30: break
                cv2.imwrite(str(ref_path / "01_Entrenamiento_NORMAL" / f"train_{count:02d}.png"), aug)
                count += 1
        
        # 2. Validación Buenas (15) - LEVE
        count = 0
        while count < 15:
            base = random.choice(raw_imgs)
            aug = random.choice([base, cv2.flip(base, 1), cv2.rotate(base, cv2.ROTATE_180)]) 
            cv2.imwrite(str(ref_path / "02_Validacion_BUENAS" / f"val_ok_{count:02d}.png"), aug)
            count += 1
            
        # 3. Validación Adulterada (15)
        count = 0
        while count < 15:
            base = random.choice(raw_imgs)
            aug = base.copy()
            defect = random.choice(['hole', 'stain_dark', 'scratch'])
            bad = apply_industrial_defect(aug, defect)
            cv2.imwrite(str(ref_path / "03_Validacion_ADULTERADAS" / f"val_fail_{count:02d}.png"), bad)
            count += 1
            
    print("\n✅ V6 COMPLETADO. Verificación manual requerida.")

if __name__ == "__main__":
    main()
