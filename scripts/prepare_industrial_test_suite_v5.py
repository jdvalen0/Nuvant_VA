import cv2
import os
import numpy as np
import shutil
import random
from pathlib import Path

# --- CONFIGURACIÓN FORENSE V5 ---
SOURCE_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/images (2)")
OUTPUT_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/INDUSTRIAL_TEST_SUITE_V5")

# Definición de Rangos de Tiempo (Timestamp en nombre de archivo)
# Formato: frame_YYYYMMDD-HHMMSS.png
FRAME_RANGES = {
    "REF_04_Piel_Negra": ("frame_20260115-045500", "frame_20260115-073000"), # Core Seguro (Madrugada)
    "REF_05_Geometrica": ("frame_20260115-121500", "frame_20260115-150000"), # Core Seguro (Mediodía)
    "REF_06_Radial":     ("frame_20260115-172100", "frame_20260115-235959")  # Core Seguro (Tarde/Noche)
}

# Definición de Fotos Específicas (Para evitar sucias)
IMG_SOURCES = {
    "REF_01_Anillos": range(2385, 2394),
    "REF_02_Trama_Fina": [2395, 2396, 2397], # Solo las más limpias
    "REF_03_Lisa": range(2416, 2426)
}

def apply_augmentation(img, forceful=False):
    augs = [img.copy()]
    # Rotaciones
    augs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augs.append(cv2.rotate(img, cv2.ROTATE_180))
    # Espejo
    augs.append(cv2.flip(img, 1))
    
    # Brillo / Contraste
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2].astype(np.int16)
    
    shifts = [20, -20, 40, -40] if forceful else [20, -20]
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
        # Agujero negro/oscuro
        cv2.circle(img, (random.randint(50,w-50), random.randint(50,h-50)), random.randint(15,40), (5,5,5), -1)
        
    elif defect_type == 'grease':
        # Mancha aceite (oscurecer región)
        mask = np.zeros((h,w), dtype=np.uint8)
        center = (random.randint(50,w-50), random.randint(50,h-50))
        axes = (random.randint(30,80), random.randint(20,50))
        angle = random.randint(0,180)
        cv2.ellipse(mask, center, axes, angle, 0, 180, 255, -1)
        
        # Mezclar
        overlay = img.copy()
        overlay[mask==255] = (overlay[mask==255] * 0.4).astype(np.uint8)
        img = overlay
        
    elif defect_type == 'thread':
        # Hilo suelto (blanco o gris)
        pt1 = (random.randint(20,w-20), random.randint(20,h-20))
        pt2 = (pt1[0] + random.randint(-50,50), pt1[1] + random.randint(-50,50))
        cv2.line(img, pt1, pt2, (200,200,200), 2)
        
    return img

def main():
    print(f"--- 🕵️ Generando Test Suite FORENSE V5 ---")
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    # --- PROCESAR FOTOS (IMG_*) ---
    for ref_name, sources in IMG_SOURCES.items():
        print(f"📦 Procesando {ref_name} (Origen Limpio)...")
        ref_path = OUTPUT_DIR / ref_name
        for sub in ["01_Entrenamiento_NORMAL", "02_Validacion_BUENAS", "03_Validacion_ADULTERADAS"]:
            (ref_path / sub).mkdir(parents=True, exist_ok=True)

        base_imgs = []
        for s in sources:
            fname = f"IMG_{s}.JPEG"
            img = cv2.imread(str(SOURCE_DIR / fname))
            if img is not None: base_imgs.append(img)
            
        if not base_imgs:
            print(f"⚠️ Error: No imagenes para {ref_name}")
            continue

        # Generar Sets
        # Train: 30 (Augmentado)
        count = 0
        while count < 30:
            src = random.choice(base_imgs)
            for aug in apply_augmentation(src, forceful=True):
                if count >= 30: break
                cv2.imwrite(str(ref_path / "01_Entrenamiento_NORMAL" / f"train_{count:02d}.png"), aug)
                count += 1
                
        # Val OK: 15 (Augmentado de LIMPIAS)
        count = 0
        while count < 15:
            src = random.choice(base_imgs) # Usar las mismas limpias base
            aug = random.choice(apply_augmentation(src))
            cv2.imwrite(str(ref_path / "02_Validacion_BUENAS" / f"val_ok_{count:02d}.png"), aug)
            count += 1
            
        # Val Fail: 15
        count = 0
        while count < 15:
            src = random.choice(base_imgs)
            aug = random.choice(apply_augmentation(src))
            defect = random.choice(['hole', 'grease', 'thread'])
            bad = apply_industrial_defect(aug, defect)
            cv2.imwrite(str(ref_path / "03_Validacion_ADULTERADAS" / f"val_fail_{count:02d}.png"), bad)
            count += 1
            
    # --- PROCESAR FRAMES (Video) ---
    all_frames = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith("frame_")])
    
    for ref_name, (start_f, end_f) in FRAME_RANGES.items():
        print(f"🎥 Procesando {ref_name} (Rango: {start_f} -> {end_f})...")
        ref_path = OUTPUT_DIR / ref_name
        for sub in ["01_Entrenamiento_NORMAL", "02_Validacion_BUENAS", "03_Validacion_ADULTERADAS"]:
            (ref_path / sub).mkdir(parents=True, exist_ok=True)
            
        # Seleccionar frames en rango
        candidates = [f for f in all_frames if f >= start_f and f <= end_f]
        print(f"   - Candidatos en rango temporal: {len(candidates)}")
        
        # Muestreo disperso para variedad (cada 5 frames)
        selection = candidates[::5]
        if len(selection) < 5: selection = candidates # Fallback
        
        imgs = []
        for f in selection[:50]: # Max 50 base
            i = cv2.imread(str(SOURCE_DIR / f))
            if i is not None: imgs.append(i)
            
        if not imgs:
            print(f"⚠️ Error: No frames para {ref_name}")
            continue

        # Generar Sets (Misma logica)
        count = 0
        while count < 30:
            src = random.choice(imgs)
            for aug in apply_augmentation(src, forceful=True):
                if count >= 30: break
                cv2.imwrite(str(ref_path / "01_Entrenamiento_NORMAL" / f"train_{count:02d}.png"), aug)
                count += 1
                
        count = 0
        while count < 15:
            src = random.choice(imgs)
            aug = random.choice(apply_augmentation(src))
            cv2.imwrite(str(ref_path / "02_Validacion_BUENAS" / f"val_ok_{count:02d}.png"), aug)
            count += 1

        count = 0
        while count < 15:
            src = random.choice(imgs)
            aug = random.choice(apply_augmentation(src))
            bad = apply_industrial_defect(aug, random.choice(['hole', 'grease', 'thread']))
            cv2.imwrite(str(ref_path / "03_Validacion_ADULTERADAS" / f"val_fail_{count:02d}.png"), bad)
            count += 1

    print("\n✅ GENERACIÓN V5 COMPLETADA - Pureza Garantizada")

if __name__ == "__main__":
    main()
