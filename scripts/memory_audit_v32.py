import torch
import numpy as np
import psutil
import os
import time
from backend.core.anomaly_patchcore import AnomalyDetectorV32

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def run_memory_audit():
    print("=== AUDITORÍA DE MEMORIA NUVANT VA V32.5++ ===")
    
    # 1. Base Memory
    mem_base = get_process_memory()
    print(f"[1] Memoria Base (Sistema Operativo + Python): {mem_base:.2f} MB")
    
    # 2. Backbone Loading
    start_time = time.time()
    detector = AnomalyDetectorV32()
    load_time = time.time() - start_time
    mem_after_backbone = get_process_memory()
    print(f"[2] Memoria tras cargar WideResNet-50-2: {mem_after_backbone:.2f} MB")
    print(f"    - Incremento Backbone: {mem_after_backbone - mem_base:.2f} MB")
    print(f"    - Tiempo de carga: {load_time:.2f} s")
    
    # 3. Memory Bank Simulation (Coreset)
    # PatchCore typical features are (N, 1536)
    # A coreset of 5000 vectors is common
    features_sim = np.random.randn(5000, 1536).astype(np.float32)
    detector.memory_bank = features_sim
    detector.is_trained = True
    mem_after_bank = get_process_memory()
    print(f"[3] Memoria con Memory Bank (5000 parches x 1536 dims): {mem_after_bank:.2f} MB")
    print(f"    - Incremento Memory Bank: {mem_after_bank - mem_after_backbone:.2f} MB")
    
    # 4. Inference Simulation
    img_sim = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    start_inf = time.time()
    detector.predict(img_sim)
    inf_time = time.time() - start_inf
    mem_final = get_process_memory()
    print(f"[4] Memoria tras Primera Inferencia: {mem_final:.2f} MB")
    print(f"    - Tiempo Inferencia (CPU): {inf_time:.2f} s")
    
    print("\n=== RESUMEN DE INVESTIGACIÓN ===")
    print(f"Consumo Total Estimado por Referencia: ~{mem_after_bank - mem_after_backbone:.2f} MB")
    print(f"Consumo Estático (Arquitectura): {mem_after_backbone:.2f} MB")

if __name__ == "__main__":
    run_memory_audit()
