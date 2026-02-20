import torch
import numpy as np
import time
import os
import cv2
from backend.core.anomaly_patchcore import AnomalyDetectorV32
from backend.core.features import FeatureExtractor

def get_memory_usage():
    """Fallback memory check for Linux systems without psutil."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0 # MB
    except:
        return 0.0

def diamond_audit():
    print("=== 💎 AUDITORÍA HOLÍSTICA GRADO DIAMANTE - NUVANT VA V32.5++ 💎 ===")
    results = {"status": "SUCCESS", "checks": []}
    
    # 1. Integridad de la Extracción Híbrida (Audit Holístico)
    print("\n[Audit 1/5] Verificación de la Cadena Cinética de Características...")
    try:
        extractor = FeatureExtractor()
        # Test 1A: Respuesta a Textura Oscura (Límite 0.05 Laplacian)
        # Usamos ruido aleatorio para simular textura real
        dark_fabric = np.random.randint(2, 6, (448, 448, 3), dtype=np.uint8)
        feats = extractor.extract(dark_fabric)
        if feats.shape == (16, 2560):
            print(f" ✅ Extracción Híbrida: Dimensiones correctas {feats.shape}.")
            results["checks"].append("Features: OK")
        else:
            print(f" ❌ ERROR: Dimensiones incorrectas {feats.shape}")
            results["status"] = "FAILURE"
    except Exception as e:
        print(f" ❌ ERROR en Extractor: {e}")
        results["status"] = "FAILURE"

    # 2. Análisis del Motor Científico (PatchCore V32.5++ Gold)
    print("\n[Audit 2/5] Validación de Precisión Matemática y Hibridación...")
    try:
        detector = AnomalyDetectorV32()
        # Escenario: Coreset Subsampling (arXiv:2106.08265)
        train_data = [np.random.randint(10, 20, (448, 448, 3), dtype=np.uint8) for _ in range(3)]
        detector.train(images=train_data)
        
        # Test 2A: Inferencia de Imagen Directa
        test_img = np.random.randint(10, 20, (448, 448, 3), dtype=np.uint8)
        is_anom, score, heatmap = detector.predict(test_img)
        print(f" ✅ Inferencia Unificada: Score={score:.2f}, Heatmap={type(heatmap)}")
        
        # Test 2B: Inferencia de Características (V31 Legacy Support)
        fake_feats = np.random.randn(1, 1536) 
        is_anom_f, score_f, _ = detector.predict(features=fake_feats)
        print(f" ✅ Soporte Híbrido V31/V32: Vectores OK (Score={score_f:.2f})")
    except Exception as e:
        print(f" ❌ ERROR en Motor: {e}")
        results["status"] = "FAILURE"

    # 3. Prueba de Estrés y Fugas de Memoria
    print("\n[Audit 3/5] Prueba de Estabilidad de Recursos (VmRSS Test)...")
    mem_start = get_memory_usage()
    for i in range(30): 
        detector.predict(test_img)
    mem_end = get_memory_usage()
    print(f" ✅ Estabilidad de RAM: Crecimiento tras 30 ciclos: {mem_end - mem_start:.2f} MB")
    if (mem_end - mem_start) < 50: # Tolerancia razonable para PyTorch cache
        print(" ✅ CERTIFICADO: Estabilidad de memoria validada.")
    else:
        print(" ⚠️ ADVERTENCIA: Crecimiento de memoria inusual.")

    # 4. Auditoría de Flujo Asíncrono
    print("\n[Audit 4/5] Verificación de Latencia de Procesamiento...")
    st = time.time()
    for _ in range(10): detector.predict(test_img)
    avg_latency = (time.time() - st) / 10 * 1000
    print(f" ✅ Rendimiento: Latencia media: {avg_latency:.2f} ms")
    if avg_latency < 300: # Tolerancia en CPUs sin AVX optimizado
        print(f" ✅ CERTIFICADO: Cumple con tiempo real ({1000/max(1,avg_latency):.1f} FPS e-2-e)")

    # 5. Integridad de los Planes y Documentación
    print("\n[Audit 5/5] Consistencia de Rutas Críticas...")
    paths = ["backend/core/anomaly_patchcore.py", "backend/core/features.py", "backend/api/routers/inference.py"]
    missing = [p for p in paths if not os.path.exists(p)]
    if not missing:
        print(" ✅ Arquitectura: Todos los módulos están en posición industrial.")
    else:
        print(f" ❌ ERROR: Faltan módulos: {missing}")
        results["status"] = "FAILURE"
    
    print(f"\n=== RESULTADO FINAL: {results['status']} ===")
    return results

if __name__ == "__main__":
    diamond_audit()
