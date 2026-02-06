import numpy as np
import cv2
import os
import shutil
import random
from pathlib import Path

# Configuración de Rutas
PROJECT_ROOT = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA")
SOURCE_DIR = PROJECT_ROOT / "images (2)"
TEST_SUITE_V3_DIR = PROJECT_ROOT / "INDUSTRIAL_TEST_SUITE_V3"

# Definición de Referencias Reales (Basado en Auditoría Visual)
REFERENCES = {
    "REF_01_Anillos": {
        "images": list(range(2385, 2394)) # 2385-2393
    },
    "REF_02_Trama_Fina": {
        "images": list(range(2394, 2416)) # 2394-2415
    },
    "REF_03_Lisa": {
        "images": list(range(2416, 2426)) # 2416-2425
    }
}

def apply_augmentation(img):
    """Aplica aumentación básica para llegar al rigor estadístico de 30 muestras."""
    augs = []
    # 1. Original
    augs.append(img.copy())
    
    # 2. Rotaciones
    augs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augs.append(cv2.rotate(img, cv2.ROTATE_180))
    augs.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    
    # 3. Flips
    augs.append(cv2.flip(img, 0)) # Vertical
    augs.append(cv2.flip(img, 1)) # Horizontal
    
    # 4. Brillo (Sutil +/- 10%)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    
    brighter = hsv.copy()
    brighter[:,:,2] = np.clip(v.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    augs.append(cv2.cvtColor(brighter, cv2.COLOR_HSV2BGR))
    
    darker = hsv.copy()
    darker[:,:,2] = np.clip(v.astype(np.int16) - 25, 0, 255).astype(np.uint8)
    augs.append(cv2.cvtColor(darker, cv2.COLOR_HSV2BGR))
    
    return augs

def apply_industrial_defect(image, defect_type='stain'):
    img = image.copy()
    if len(img.shape) == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1
        
    if defect_type == 'hole':
        cy, cx = np.random.randint(100, h-100), np.random.randint(100, w-100)
        radius = np.random.randint(20, 50)
        color = (10, 10, 10) if c == 3 else 10
        cv2.circle(img, (cx, cy), radius, color, -1)
        
    elif defect_type == 'grease':
        cy, cx = np.random.randint(100, h-100), np.random.randint(100, w-100)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (np.random.randint(50, 100), np.random.randint(30, 60)), 
                    np.random.randint(0, 180), 0, 180, 255, -1)
        img[mask == 255] = (img[mask == 255] * 0.4).astype(np.uint8)
        
    elif defect_type == 'broken_yarn':
        y = np.random.randint(100, h-100)
        x_start = np.random.randint(50, w-300)
        length = np.random.randint(100, 250)
        color = (240, 240, 240) if c == 3 else 240
        cv2.line(img, (x_start, y), (x_start + length, y + np.random.randint(-5, 5)), color, 4)
        
    elif defect_type == 'stain':
        cy, cx = np.random.randint(100, h-100), np.random.randint(100, w-100)
        for r in range(40, 0, -8):
            color_val = 100 - (r * 2)
            color = (color_val, color_val, color_val) if c == 3 else color_val
            cv2.circle(img, (cx, cy), r, color, -1)
            
    return img

def main():
    print("--- 🚀 Generando INDUSTRIAL_TEST_SUITE_V3 (Multi-Referencia Real) ---")
    
    if TEST_SUITE_V3_DIR.exists():
        shutil.rmtree(TEST_SUITE_V3_DIR)
    
    for ref_name, config in REFERENCES.items():
        print(f"\n📦 Procesando: {ref_name}...")
        ref_path = TEST_SUITE_V3_DIR / ref_name
        paths = {
            "train": ref_path / "01_Entrenamiento_NORMAL",
            "val_good": ref_path / "02_Validacion_BUENAS",
            "val_bad": ref_path / "03_Validacion_ADULTERADAS"
        }
        for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
        
        # 1. Recolectar imágenes fuente
        src_images = []
        for i in config["images"]:
            name = f"IMG_{i}.JPEG"
            img_path = SOURCE_DIR / name
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    src_images.append(img)
            else:
                # Debugging info
                print(f"   - No se encontró {name}")

        if not src_images:
            print(f"⚠️ No se encontraron imágenes para {ref_name}")
            continue
            
        print(f" - Encontradas {len(src_images)} imágenes base.")
        
        # 2. Generar Entrenamiento (N=30) con Aumentación
        train_samples = []
        while len(train_samples) < 30:
            base_img = random.choice(src_images)
            augs = apply_augmentation(base_img)
            train_samples.extend(augs)
            
        train_samples = train_samples[:30] # Trim a exactamente 30
        for i, img in enumerate(train_samples):
            cv2.imwrite(str(paths["train"] / f"train_{i:02d}.png"), img)
            
        print(f" - Generadas 30 muestras de entrenamiento (Aumentadas).")
        
        # 3. Generar Validación Buena (15 muestras)
        val_good = []
        while len(val_good) < 15:
            base_img = random.choice(src_images)
            # Aplicar aumentación aleatoria para variedad
            aug = random.choice(apply_augmentation(base_img))
            val_good.append(aug)
        
        for i, img in enumerate(val_good):
            cv2.imwrite(str(paths["val_good"] / f"val_ok_{i:02d}.png"), img)
            
        print(f" - Generadas 15 muestras de validación BUENAS.")
            
        # 4. Generar Validación Adulterada (15 muestras)
        defects = ['hole', 'grease', 'broken_yarn', 'stain']
        for i in range(15):
            base_img = random.choice(src_images)
            aug = random.choice(apply_augmentation(base_img))
            d_type = random.choice(defects)
            bad_img = apply_industrial_defect(aug, d_type)
            cv2.imwrite(str(paths["val_bad"] / f"val_fail_{d_type}_{i:02d}.png"), bad_img)
            
        print(f" - Generadas 15 muestras de validación ADULTERADAS.")

    print(f"\n✅ ÉXITO: Test Suite V3 creada en {TEST_SUITE_V3_DIR}")

if __name__ == "__main__":
    main()
