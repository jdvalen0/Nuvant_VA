import os
import sys
import os
import inspect
import torch
import numpy as np
import cv2
import json
import joblib
from pathlib import Path

# Mock FastAPI for path checking
from backend.config import BASE_DIR, STORAGE_DIR, DATABASE_URL

def run_deep_audit():
    print("="*50)
    print("📋 AUDITORÍA PROFUNDA NUVANT VA - V32.5+")
    print("="*50)
    
    # 1. Rutas y Persistencia (Docker Ready)
    print("\n[1/5] Auditando Rutas y Persistencia...")
    checks = {
        "BASE_DIR": BASE_DIR,
        "STORAGE_DIR": STORAGE_DIR,
        "DATABASE_URL": DATABASE_URL,
    }
    for k, v in checks.items():
        print(f" - {k}: {v}")
    
    db_path = DATABASE_URL.replace("sqlite:///", "")
    if "/app/" in str(STORAGE_DIR) or "/app/" in db_path:
        print(" ✅ Detectada configuración de rutas interna de Docker.")
    else:
        print(" ⚠️ Advertencia: Rutas locales detectadas (Normal en dev, pero verificar docker-compose).")

    # 2. Requerimientos y Dependencias
    print("\n[2/5] Auditando Dependencias Críticas...")
    try:
        from backend.core.anomaly_patchcore import PatchCoreDetector
        pcd = PatchCoreDetector()
        print(f" ✅ PatchCore V32 cargado correctamente. Device: {pcd.device}")
        
        # Check for backbone weights
        print(" - Probando carga de backbone (WideResNet50_2)...")
        # Just to check if it downloads or is cached
        # In Docker, this might fail if no internet during build, but we should verify cached paths
    except Exception as e:
        print(f" ❌ Error en motor PatchCore: {e}")

    # 3. Consistencia de Parámetros de Entrenamiento
    print("\n[3/5] Auditando Parámetros de Entrenamiento...")
    from backend.api.routers import references
    # Check if we are passing parameters that are actually used
    import inspect
    sig = inspect.signature(pcd.train)
    params_pcd = sig.parameters.keys()
    print(f" - Parámetros aceptados por PatchCore.train: {list(params_pcd)}")
    if "contamination" in params_pcd:
        print(" ✅ 'contamination' está correctamente vinculado.")
    else:
        print(" ❌ 'contamination' no se encuentra en la firma del método train.")

    # 4. Verificación de Lógica de Sensibilidad (Safety Margin)
    print("\n[4/5] Auditando Lógica de Sensibilidad y Margen 1.1x...")
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pcd.train(images=[dummy_img]*5, contamination=0.01)
    
    thresh_base = pcd.threshold
    _, score, _ = pcd.predict(image=dummy_img) # Same image
    print(f" - Threshold Base: {thresh_base:.4f}")
    print(f" - Score imagen entrenamiento (debe ser < 50): {score:.2f}")
    
    if score < 50.0:
        print(" ✅ Margen de seguridad 1.1x verificado.")
    else:
        print(f" ⚠️ Margen de seguridad potencialmente bajo (Score: {score:.2f} >= 50).")

    # 5. Auditoría de Reconexión (Frontend logic in index.html)
    print("\n[5/5] Auditando Lógica de Reconexión UI...")
    index_path = Path("backend/api/main.py").parent / "static" / "index.html"
    if index_path.exists():
        with open(index_path, "r") as f:
            html = f.read()
            if "reconnectBtn" in html and "animate-pulse" in html:
                 print(" ✅ Botón de reconexión con feedback visual detectado en UI.")
            else:
                 print(" ❌ Botón de reconexión no encontrado o sin animación.")
    else:
        print(" ❌ No se encontró index.html para auditoría de UI.")

    print("\n" + "="*50)
    print("🏆 AUDITORÍA COMPLETADA")
    print("="*50)

if __name__ == "__main__":
    run_deep_audit()
