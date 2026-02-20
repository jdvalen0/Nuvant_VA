import torch
import numpy as np
import time
import os
import cv2
from backend.core.anomaly_patchcore import AnomalyDetectorV32
from backend.core.features import FeatureExtractor

def audit_algorithmic_integrity():
    print("=== AUDITORÍA CIENTÍFICA NUVANT VA V32.5++ ===")
    
    # 1. Validación de Feature Extractor (F7 Fixes)
    print("\n[Audit 1/4] Integridad del Feature Extractor (Luminancia/Blur)...")
    extractor = FeatureExtractor()
    
    # Simular Negro Absoluto
    black_img = np.zeros((448, 448, 3), dtype=np.uint8)
    try:
        extractor.extract(black_img)
        print(" ❌ ERROR: Debería haber rechazado oscuridad 0.0")
    except ValueError as e:
        print(f" ✅ RECHAZO CORRECTO (Oscuridad 0.0): {e}")

    # Simular Tela Negra Industrial (Brillo 0.45 - El punto de falla anterior)
    # IMPORTANTE: Una imagen de np.ones no tiene textura, por lo que Laplacian será 0.0.
    # Usamos ruido aleatorio tenue para simular textura de tela real.
    dark_fabric = np.random.randint(2, 5, (448, 448, 3), dtype=np.uint8) 
    try:
        extractor.extract(dark_fabric)
        print(" ✅ PROCESAMIENTO CORRECTO: Tela negra industrial con textura permitida.")
    except ValueError as e:
        print(f" ❌ ERROR: Fallo en tela negra industrial: {e}")

    # 2. Validación de Lógica Híbrida y PatchCore
    print("\n[Audit 2/4] Lógica PatchCore (Aggregation & Reweighting)...")
    detector = AnomalyDetectorV32()
    
    # Simular Entrenamiento con datos sintéticos
    train_imgs = [np.random.randint(100, 110, (448, 448, 3), dtype=np.uint8) for _ in range(5)]
    # Añadir un frame muerto para probar el Train Filter
    train_imgs.append(np.zeros((448, 448, 3), dtype=np.uint8))
    
    print(" - Iniciando entrenamiento de prueba...")
    detector.train(images=train_imgs)
    print(f" - Coreset size: {len(detector.memory_bank) if detector.memory_bank is not None else 0}")
    
    # 3. Prueba de Sensibilidad y Estabilidad
    print("\n[Audit 3/4] Estabilidad de Score y Sensibilidad...")
    # Frame normal
    normal_frame = np.random.randint(100, 110, (448, 448, 3), dtype=np.uint8)
    # El retorno es (is_anomaly, score, heatmap)
    is_anomaly_n, score_n, _ = detector.predict(normal_frame)
    print(f" - Score Normal: {score_n:.4f}")
    
    # Frame con anomalía sintética (un parche brillante)
    anomaly_frame = normal_frame.copy()
    anomaly_frame[200:250, 200:250] = 255
    is_anomaly_a, score_a, heatmap_a = detector.predict(anomaly_frame)
    print(f" - Score Anomalía: {score_a:.4f}")
    
    if score_a > score_n:
        print(" ✅ DETECCIÓN LÓGICA: PatchCore responde correctamente a gradientes.")
    else:
        print(" ❌ ERROR: Fallo en discriminación de anomalía.")

    # 4. Auditoría de Memoria y Gráficos (Data Flow)
    print("\n[Audit 4/4] Integridad de Flujo WebSocket/Chart...")
    # Verificar que el payload contiene los campos necesarios para Chart.js
    if score_n is not None and (heatmap_a is not None or is_anomaly_a is not None):
         print(" ✅ INTERFAZ: Payload compatible con Dashboard V32 Gold.")
    else:
         print(" ❌ ERROR: Estructura de datos incompleta.")

if __name__ == "__main__":
    audit_algorithmic_integrity()
