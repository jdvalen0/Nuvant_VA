import cv2
import os
import numpy as np
import shutil
import random
from pathlib import Path

# --- CONFIGURACIÓN ---
SOURCE_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/images (2)")
OUTPUT_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/INDUSTRIAL_TEST_SUITE_V4")

# Definición de Referencias Maestro
REFERENCES = {
    "REF_01_Anillos": {"type": "fixed", "range": range(2385, 2394)},
    "REF_02_Trama_Fina": {"type": "fixed", "range": range(2394, 2416)},
    "REF_03_Lisa": {"type": "fixed", "range": range(2416, 2426)},
    "REF_04_Piel_Negra": {"type": "cluster", "mean": (5, 40), "std": (5, 100), "rugosity": (150, 600)},
    "REF_05_Geometrica": {"type": "cluster", "mean": (60, 160), "std": (30, 80), "rugosity": (300, 1200)},
    "REF_06_Radial": {"type": "cluster", "mean": (50, 150), "std": (30, 70), "rugosity": (1200, 4000)}
}

def apply_augmentation(img):
    augs = [img.copy()]
    augs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augs.append(cv2.rotate(img, cv2.ROTATE_180))
    augs.append(cv2.flip(img, 1))
    # Brillo +/-
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    for shift in [30, -30]:
        h_mod = hsv.copy()
        h_mod[:,:,2] = np.clip(v.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        augs.append(cv2.cvtColor(h_mod, cv2.COLOR_HSV2BGR))
    return augs

def apply_industrial_defect(image, defect_type='stain'):
    img = image.copy()
    h, w = img.shape[:2]
    if defect_type == 'hole':
        cv2.circle(img, (random.randint(100,w-100), random.randint(100,h-100)), random.randint(30,60), (10,10,10), -1)
    elif defect_type == 'grease':
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.ellipse(mask, (random.randint(100,w-100), random.randint(100,h-100)), (random.randint(50,100), random.randint(30,60)), random.randint(0,180), 0, 180, 255, -1)
        img[mask == 255] = (img[mask == 255] * 0.3).astype(np.uint8)
    elif defect_type == 'broken_yarn':
        y = random.randint(100,h-100)
        cv2.line(img, (random.randint(50,w-300), y), (random.randint(300,w-50), y + random.randint(-5,5)), (250,250,250), 4)
    return img

def main():
    print(f"--- 🚀 Generando Test Suite MAESTRO V4 ---")
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    # 1. Escaneo preliminar de frames para clustering veloz
    frame_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith("frame_")])
    frame_stats = []
    print("Pre-escaneando frames...")
    for f in frame_files[::5]: # Muestreo para velocidad
        img = cv2.imread(str(SOURCE_DIR / f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img_s = cv2.resize(img, (256, 256))
        m, s = np.mean(img_s), np.std(img_s)
        r = cv2.Laplacian(img_s, cv2.CV_64F).var()
        frame_stats.append({"file": f, "mean": m, "std": s, "rugosity": r})

    for ref_name, cfg in REFERENCES.items():
        print(f"\n📦 Fabricando {ref_name}...")
        ref_path = OUTPUT_DIR / ref_name
        paths = {k: ref_path / v for k, v in {"train": "01_Entrenamiento_NORMAL", "val_ok": "02_Validacion_BUENAS", "val_fail": "03_Validacion_ADULTERADAS"}.items()}
        for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
        
        # Recolectar base
        src_imgs = []
        if cfg["type"] == "fixed":
            for i in cfg["range"]:
                img = cv2.imread(str(SOURCE_DIR / f"IMG_{i}.JPEG"))
                if img is not None: src_imgs.append(img)
        else:
            m_min, m_max = cfg["mean"]
            s_min, s_max = cfg["std"]
            r_min, r_max = cfg["rugosity"]
            matches = [s["file"] for s in frame_stats if m_min < s["mean"] < m_max and s_min < s["std"] < s_max and r_min < s["rugosity"] < r_max]
            print(f" - Encontrados {len(matches)} frames candidatos.")
            for f in matches[:40]: # Limitar originales
                img = cv2.imread(str(SOURCE_DIR / f))
                if img is not None: src_imgs.append(img)
        
        if not src_imgs:
            print(f" ⚠️ No hay muestras para {ref_name}")
            continue
            
        # Entrenamiento (N=30)
        train_pool = []
        while len(train_pool) < 30:
            base = random.choice(src_imgs)
            for aug in apply_augmentation(base):
                train_pool.append(aug)
                if len(train_pool) >= 30: break
        for i, img in enumerate(train_pool[:30]): cv2.imwrite(str(paths["train"] / f"train_{i:02d}.png"), img)
        
        # Validación OK (N=15)
        for i in range(15):
            base = random.choice(src_imgs)
            cv2.imwrite(str(paths["val_ok"] / f"val_ok_{i:02d}.png"), random.choice(apply_augmentation(base)))
            
        # Validación Adulterada (N=15)
        defects = ['hole', 'grease', 'broken_yarn']
        for i in range(15):
            base = random.choice(src_imgs)
            bad = apply_industrial_defect(random.choice(apply_augmentation(base)), random.choice(defects))
            cv2.imwrite(str(paths["val_fail"] / f"val_fail_{i:02d}.png"), bad)
            
        print(f" ✅ Completado: 30 train, 15 ok, 15 fail.")

    print(f"\n--- ✅ PROCESO TERMINADO: INDUSTRIAL_TEST_SUITE_V4 LISTO ---")

if __name__ == "__main__":
    main()
