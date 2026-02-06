import numpy as np
import cv2
import os
import shutil
from pathlib import Path

# Configuración de Rutas
PROJECT_ROOT = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA")
TEST_SUITE_DIR = PROJECT_ROOT / "INDUSTRIAL_TEST_SUITE"

def generate_fabric_texture(size=(512, 512), weave_density=40, randomness=0.2):
    """
    Genera una textura de tela con variaciones estadísticas realistas.
    """
    h, w = size
    x = np.linspace(0, weave_density * np.pi, w)
    y = np.linspace(0, weave_density * np.pi, h)
    xv, yv = np.meshgrid(x, y)
    
    # Base pattern (sinusoidal weave)
    # Agregamos pequeñas variaciones de fase para que no sean idénticas
    phase_x = np.random.uniform(0, 2*np.pi)
    phase_y = np.random.uniform(0, 2*np.pi)
    z = np.sin(xv + phase_x) + np.sin(yv + phase_y)
    
    # Escalar a 0-255
    textura = ((z + 2) / 4 * 255).astype(np.uint8)
    
    # Agregar Ruido Gaussiano (Variación de sensor/iluminación)
    noise = np.random.normal(0, 10, size).astype(np.int16)
    textura = np.clip(textura + noise, 0, 255).astype(np.uint8)
    
    # Ajustar brillo aleatoriamente
    brightness = np.random.uniform(0.85, 1.15)
    textura = (textura * brightness).clip(0, 255).astype(np.uint8)
    
    return textura

def apply_industrial_defect(image, defect_type='stain'):
    """
    Aplica 'adulteración' industrial controlada a una imagen de tela.
    """
    img = image.copy()
    h, w = img.shape
    
    if defect_type == 'hole':
        # Agujero: círculo negro con bordes irregulares
        cy, cx = np.random.randint(100, h-100), np.random.randint(100, w-100)
        radius = np.random.randint(10, 25)
        cv2.circle(img, (cx, cy), radius, (10, 10, 10), -1)
        # Blur en bordes para realismo
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    elif defect_type == 'grease':
        # Mancha de grasa: parche oscuro irregular
        cy, cx = np.random.randint(100, h-100), np.random.randint(100, w-100)
        mask = np.zeros_like(img)
        cv2.ellipse(mask, (cx, cy), (np.random.randint(20, 50), np.random.randint(10, 30)), 
                    np.random.randint(0, 180), 0, 180, 255, -1)
        img = np.where(mask == 255, (img * 0.4).astype(np.uint8), img)
        
    elif defect_type == 'broken_yarn':
        # Hilo roto / Trama interrumpida: línea brillante u oscura
        y = np.random.randint(100, h-100)
        x_start = np.random.randint(50, w-200)
        length = np.random.randint(50, 150)
        cv2.line(img, (x_start, y), (x_start + length, y + np.random.randint(-2, 2)), (20, 20, 20), 2)
        
    elif defect_type == 'stain':
        # Mancha típica (ej: tinta): parche con bordes difusos
        cy, cx = np.random.randint(100, h-100), np.random.randint(100, w-100)
        for r in range(15, 0, -1):
            color = 120 - (r * 3)
            cv2.circle(img, (cx, cy), r, (color), -1)
            
    return img

def main():
    # Limpiar directorios previos
    if TEST_SUITE_DIR.exists():
        shutil.rmtree(TEST_SUITE_DIR)
        
    # Crear estructura de carpetas
    paths = {
        "train": TEST_SUITE_DIR / "01_Entrenamiento_NORMAL",
        "val_good": TEST_SUITE_DIR / "02_Validacion_BUENAS",
        "val_bad": TEST_SUITE_DIR / "03_Validacion_ADULTERADAS"
    }
    
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    print(f"--- Iniciando Generación de Test Suite Industrial ---")
    
    # 1. Generar Entrenamiento (N=30 como solicita el estándar estadístico)
    print(f"(1/3) Generando {30} muestras de entrenamiento NORMAL...")
    for i in range(30):
        img = generate_fabric_texture()
        cv2.imwrite(str(paths["train"] / f"sample_{i:02d}.png"), img)
        
    # 2. Generar Validación Buenas (N=15)
    print(f"(2/3) Generando {15} muestras de validación BUENAS...")
    for i in range(15):
        img = generate_fabric_texture()
        cv2.imwrite(str(paths["val_good"] / f"val_ok_{i:02d}.png"), img)
        
    # 3. Generar Validación Adulteradas (N=15)
    print(f"(3/3) Generando {15} muestras ADULTERADAS con defectos industriales...")
    defects = ['hole', 'grease', 'broken_yarn', 'stain']
    for i in range(15):
        d_type = np.random.choice(defects)
        img = generate_fabric_texture()
        bad_img = apply_industrial_defect(img, d_type)
        cv2.imwrite(str(paths["val_bad"] / f"defect_{d_type}_{i:02d}.png"), bad_img)
        
    print(f"\n✅ EXITOSO: Test Suite creada en:\n{TEST_SUITE_DIR}")
    print(f"Estructura:")
    print(f" - entrenamiento: 30 imágenes")
    print(f" - val_good: 15 imágenes")
    print(f" - val_bad: 15 imágenes")

if __name__ == "__main__":
    main()
