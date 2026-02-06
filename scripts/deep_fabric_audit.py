import cv2
import os
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import json

SOURCE_DIR = Path("/home/juan-david-valencia/Escritorio/Nuvant_VA/images (2)")
frame_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith("frame_") and f.endswith(".png")])

print(f"--- 🧵 Iniciando Auditoría Profunda de {len(frame_files)} Frames ---")

data = []
filenames = []

# Procesamiento rápido de todos los frames
for i, f in enumerate(frame_files):
    img_path = SOURCE_DIR / f
    # Leer en gris y escala reducida para velocidad
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    
    img_small = cv2.resize(img, (256, 256))
    
    mean_val = np.mean(img_small)
    std_val = np.std(img_small)
    # Varianza Laplaciana (mide bordes/rugosidad)
    laplacian_var = cv2.Laplacian(img_small, cv2.CV_64F).var()
    
    # Filtro básico: descartar negros totales o blancos totales sin textura
    if mean_val < 5 or (mean_val > 250 and std_val < 5):
        continue
        
    data.append([mean_val, std_val, laplacian_var])
    filenames.append(f)
    
    if i % 500 == 0:
        print(f" Auditados {i}/{len(frame_files)}...")

if not data:
    print("❌ No se encontraron imágenes válidas con textura.")
    exit()

X = np.array(data)

# Normalización simple para clustering
X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

# Intentar encontrar 6 clusters (pueden ser menos si son repetitivas)
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_norm)

clusters = {}
for idx, label in enumerate(labels):
    l = int(label)
    if l not in clusters:
        clusters[l] = []
    clusters[l].append({
        "file": filenames[idx],
        "stats": {
            "mean": float(data[idx][0]),
            "std": float(data[idx][1]),
            "rugosity": float(data[idx][2])
        }
    })

print(f"\n📊 Resumen de Clústeres (Potenciales Referencias):")
report = []
for label, items in clusters.items():
    # Ordenar por rugosidad descendente dentro del cluster para ver el "mejor" ejemplo
    items.sort(key=lambda x: x["stats"]["rugosity"], reverse=True)
    example = items[0]
    avg_mean = np.mean([x["stats"]["mean"] for x in items])
    avg_rug = np.mean([x["stats"]["rugosity"] for x in items])
    
    print(f"Cluster {label}: {len(items)} archivos. Promedio Luz: {avg_mean:.1f}, Rugosidad: {avg_rug:.1f}")
    print(f"   Ejemplo meta: {example['file']}")
    
    report.append({
        "cluster": label,
        "count": len(items),
        "avg_mean": avg_mean,
        "avg_rugosity": avg_rug,
        "example": example["file"]
    })

# Guardar reporte para consulta rápida del agente
with open("fabric_audit_report.json", "w") as f:
    json.dump(report, f, indent=4)

print(f"\n✅ Auditoría completa. Reporte guardado en fabric_audit_report.json")
