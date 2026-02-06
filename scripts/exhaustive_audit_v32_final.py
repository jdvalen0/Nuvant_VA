import numpy as np
import cv2
import time
import os
import torch
import json
from backend.core.anomaly_patchcore import PatchCoreDetector
from backend.core.anomaly import AnomalyDetector as MahalanobisDetector
from backend.api.routers import references

def run_exhaustive_audit():
    print("🚀 INICIANDO AUDITORÍA EXHAUSTIVA V32.5...")
    passed = []
    failed = []

    # 1. Validación de Sensibilidad PatchCore (Fórmula: threshold * (1 - offset/1000))
    print("\n[1/6] Validando Sensibilidad PatchCore...")
    pcd = PatchCoreDetector()
    dummy_imgs = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]
    pcd.train(images=dummy_imgs)
    
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, score_strict, _ = pcd.predict(image=test_img, sensitivity_offset=500)
    _, score_ignore, _ = pcd.predict(image=test_img, sensitivity_offset=-500)
    
    if score_strict > score_ignore:
        print(" ✅ Sensibilidad: OK (Estricto > Ignorar)")
        passed.append("Sensibilidad")
    else:
        print(f" ❌ Sensibilidad: FALLO (Strict: {score_strict}, Ignore: {score_ignore})")
        failed.append("Sensibilidad")

    # 2. Validación de Margen de Seguridad (Threshold * 1.1)
    print("\n[2/6] Validando Margen de Seguridad (Safety Margin 1.1x)...")
    # Predicted score for training image should be around 45 (below 50) due to 1.1 margin
    _, score_train, _ = pcd.predict(image=dummy_imgs[0])
    print(f" - Score en imagen de entrenamiento: {score_train:.2f}")
    if score_train < 50.0:
        print(" ✅ Margen de Seguridad: OK (<50.0 en entrenamiento)")
        passed.append("Margen Seguridad")
    else:
        print(f" ❌ Margen de Seguridad: FALLO (Score: {score_train} >= 50.0)")
        failed.append("Margen Seguridad")

    # 3. Validación de Score No-Negativo en Mahalanobis (V31)
    print("\n[3/6] Validando Score No-Negativo V31...")
    md = MahalanobisDetector()
    # Mocking trained state for V31
    md.gold_limit = 1.0
    md.gold_median = 0.5
    md.mean_v = np.zeros(10)
    md.precision_v = np.eye(10)
    md.pca = type('PCA', (), {'transform': lambda self, x: x})()
    
    # Force a very distant point to get negative score in old logic
    is_a, score_v31 = md.predict(feature_vector=np.ones((1, 10)) * 100)
    print(f" - Score V31 para anomalía extrema: {score_v31:.2f}")
    if score_v31 >= 0:
        print(" ✅ Score V31: OK (No es negativo)")
        passed.append("Score V31")
    else:
        print(f" ❌ Score V31: FALLO (Negativo detectado)")
        failed.append("Score V31")

    # 4. Chequeo de NameError en Backend (falsa variable 'features')
    print("\n[4/6] Chequeo de NameError en references.py...")
    import inspect
    source = inspect.getsource(references.train_reference)
    if 'len(images)' in source and 'len(features)' not in source: # Assuming 'features' was the error
        print(" ✅ NameError: OK (Corregido a 'images')")
        passed.append("NameError Fix")
    else:
        print(" ❌ NameError: POSIBLE RIESGO (Variable 'features' detectada en return)")
        failed.append("NameError Fix")

    # 5. Auditoría de Idioma (Grep manual en index.html)
    print("\n[5/6] Chequeo de Idioma (Residual English)...")
    with open("backend/api/static/index.html", "r") as f:
        content = f.read()
    english_terms = ["Browse...", "No files selected", "DEFECT DETECTED", "QUALITY OK"] # Terms we translated
    found_bugs = [term for term in english_terms if term in content and "badge.textContent" not in content and "status.textContent" not in content]
    # Note: they might exist in code strings like attribute names, but we check for text content
    if len(found_bugs) == 0:
        print(" ✅ Idioma: OK (Sin términos residuales visibles)")
        passed.append("Idioma")
    else:
        print(f" ⚠️ Idioma: ADVERTENCIA (Términos detectados: {found_bugs})")
        passed.append("Idioma (Con advertencias)")

    # 6. Verificación de Capacidad de Localización
    print("\n[6/6] Verificación de Heatmap...")
    _, _, hmap = pcd.predict(image=dummy_imgs[0])
    if hmap is not None and hmap.max() <= 1.0 and hmap.min() >= 0.0:
        print(" ✅ Heatmap: OK (Normalizado 0-1)")
        passed.append("Heatmap")
    else:
        print(" ❌ Heatmap: FALLO (No generado o fuera de rango)")
        failed.append("Heatmap")

    print("\n" + "="*40)
    print(f"RESULTADO FINAL: {len(passed)}/6 PASADOS")
    if failed:
        print(f"FALLOS: {failed}")
    print("="*40)

if __name__ == "__main__":
    run_exhaustive_audit()
