import numpy as np
import cv2
import os
import shutil
import random
from pathlib import Path

# Configuración de Rutas
PROJECT_ROOT = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA")
SOURCE_DIR = PROJECT_ROOT / "images (2)"
TEST_SUITE_V2_DIR = PROJECT_ROOT / "INDUSTRIAL_TEST_SUITE_V2"

# Importar lógica de adulteración del script anterior para consistencia
def apply_industrial_defect(image, defect_type='stain'):
    img = image.copy()
    h, w = img.shape[:2] # Manejar color o gris
    
    if defect_type == 'hole':
        cy, cx = np.random.randint(100, h-100), np.random.randint(100, w-100)
        radius = np.random.randint(20, 40)
        cv2.circle(img, (cx, cy), radius, (10, 10, 10), -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
    elif defect_type == 'grease':
        cy, cx = np.random.randint(100, h-100), np.random.randint(100, w-100)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (np.random.randint(40, 80), np.random.randint(20, 50)), 
                    np.random.randint(0, 180), 0, 180, 255, -1)
        # Atenuación realista (oscurecimiento)
        if len(img.shape) == 3:
            img[mask == 255] = (img[mask == 255] * 0.3).astype(np.uint8)
        else:
            img[mask == 255] = (img[mask == 255] * 0.3).astype(np.uint8)
        
    elif defect_type == 'broken_yarn':
        y = np.random.randint(100, h-100)
        x_start = np.random.randint(50, w-300)
        length = np.random.randint(100, 250)
        color = (250, 250, 250) if len(img.shape) == 3 else 250
        cv2.line(img, (x_start, y), (x_start + length, y + np.random.randint(-5, 5)), color, 3)
        
    elif defect_type == 'stain':
        cy, cx = np.random.randint(100, h-100), np.random.randint(100, w-100)
        for r in range(30, 0, -5):
            color_val = 80 - (r * 2)
            color = (color_val, color_val, color_val) if len(img.shape) == 3 else color_val
            cv2.circle(img, (cx, cy), r, color, -1)
            
    return img

def main():
    if not SOURCE_DIR.exists():
        print(f"❌ Error: No se encuentra la carpeta origen {SOURCE_DIR}")
        return

    # Limpiar directorios previos
    if TEST_SUITE_V2_DIR.exists():
        shutil.rmtree(TEST_SUITE_V2_DIR)
        
    paths = {
        "train": TEST_SUITE_V2_DIR / "01_Entrenamiento_NORMAL",
        "val_good": TEST_SUITE_V2_DIR / "02_Validacion_BUENAS",
        "val_bad": TEST_SUITE_V2_DIR / "03_Validacion_ADULTERADAS"
    }
    
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # Listar imágenes reales
    all_images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Filtrar imágenes "negras" o demasiado oscuras (especialmente los frames)
    valid_images = []
    print("Analizando imágenes reales por calidad...")
    for img_name in all_images:
        img_path = SOURCE_DIR / img_name
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Si el promedio de píxeles es > 5, la consideramos válida
        if np.mean(img) > 5:
            valid_images.append(img_name)
            
    if len(valid_images) < 60:
        print(f"⚠️ Atención: Solo hay {len(valid_images)} imágenes con contenido real. Usando todas.")
        needed = 60
    else:
        # Mezclar para que el entrenamiento sea variado
        random.shuffle(valid_images)
        needed = 60

    print(f"--- Iniciando Generación de Test Suite V2 (Real) ---")
    
    # Repartir imágenes
    n_train = min(len(valid_images) // 2, 30)
    n_val = (len(valid_images) - n_train) // 2
    
    train_pool = valid_images[:n_train]
    val_good_pool = valid_images[n_train:n_train + n_val]
    val_bad_pool = valid_images[n_train + n_val:n_train + 2*n_val]

    # 1. Entrenamiento
    print(f"(1/3) Copiando {len(train_pool)} muestras para entrenamiento...")
    for i, name in enumerate(train_pool):
        shutil.copy(SOURCE_DIR / name, paths["train"] / f"real_train_{i:02d}.png")
        
    # 2. Validación Buenas
    print(f"(2/3) Copiando {len(val_good_pool)} muestras para validación BUENAS...")
    for i, name in enumerate(val_good_pool):
        shutil.copy(SOURCE_DIR / name, paths["val_good"] / f"real_val_ok_{i:02d}.png")
        
    # 3. Validación Adulteradas
    print(f"(3/3) Generando {len(val_bad_pool)} muestras ADULTERADAS con defectos industriales...")
    defects = ['hole', 'grease', 'broken_yarn', 'stain']
    for i, name in enumerate(val_bad_pool):
        img = cv2.imread(str(SOURCE_DIR / name))
        d_type = np.random.choice(defects)
        bad_img = apply_industrial_defect(img, d_type)
        cv2.imwrite(str(paths["val_bad"] / f"real_defect_{d_type}_{i:02d}.png"), bad_img)
        
    print(f"\n✅ EXITOSO: Test Suite V2 creada en:\n{TEST_SUITE_V2_DIR}")
    print(f"Estructura:")
    print(f" - Entrenamiento (Real): {len(train_pool)} imágenes")
    print(f" - Validación OK (Real): {len(val_good_pool)} imágenes")
    print(f" - Validación FAIL (Real Adulterado): {len(val_bad_pool)} imágenes")

if __name__ == "__main__":
    main()
