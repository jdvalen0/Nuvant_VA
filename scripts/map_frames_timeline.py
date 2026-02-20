import cv2
import os
import json
from pathlib import Path

SOURCE_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/images (2)")
frame_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith("frame_") and f.endswith(".png")])

print(f"Total frame files: {len(frame_files)}")

# Muestrear secuencialmente para hacer un 'mapa de tiempo'
# Asumimos que el video muestra una tela por un tiempo, luego cambia.
samples = []
step = 100 # Revisar cada 100 frames (~40 muestras total)

print("--- MAPA DE TIEMPO DE FRAMES ---")
for i in range(0, len(frame_files), step):
    f = frame_files[i]
    print(f"Index {i}: {f}")
