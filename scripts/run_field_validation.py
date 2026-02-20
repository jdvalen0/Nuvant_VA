#!/usr/bin/env python3
"""
Script de Ejecución de Validación de Campo
Ejecuta el sistema contra el dataset sintético y genera reporte de métricas.
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Agregar backend al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from core.anomaly_patchcore import AnomalyDetectorV32

def load_ground_truth(gt_path):
    """Carga el archivo de ground truth."""
    with open(gt_path, 'r') as f:
        return json.load(f)

def run_validation(reference_id, dataset_dir, ground_truth_path, output_report):
    """Ejecuta la validación completa."""
    
    # Cargar ground truth
    print("📄 Cargando ground truth...")
    gt_data = load_ground_truth(ground_truth_path)
    total_images = gt_data['total_images']
    adulterations = gt_data['data']
    
    print(f"📊 Total de imágenes a validar: {total_images}")
    print(f"📸 Imágenes base: {gt_data['base_images']}")
    print(f"🔬 Adulteraciones por imagen: {gt_data['adulterations_per_image']}")
    
    # Cargar modelo
    print(f"\n🤖 Cargando modelo de referencia ID={reference_id}...")
    model_path = Path(f"backend/data/models/reference_{reference_id}_model.pkl")
    
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        print("💡 Asegúrate de haber entrenado la referencia primero.")
        return 1
    
    detector = AnomalyDetectorV32()
    import joblib
    model_data = joblib.load(model_path)
    detector.memory_bank = model_data['memory_bank']
    detector.threshold = model_data['threshold']
    detector.is_trained = True
    
    print(f"✅ Modelo cargado. Threshold: {detector.threshold:.4f}")
    
    # Ejecutar inferencia
    print(f"\n🔍 Ejecutando inferencia en {total_images} imágenes...")
    
    results = []
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    latencies = []
    
    for i, item in enumerate(adulterations, 1):
        img_path = Path(dataset_dir) / item['filename']
        
        if not img_path.exists():
            print(f"⚠️ Imagen no encontrada: {img_path}")
            continue
        
        # Cargar imagen
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        # Medir latencia
        start_time = time.time()
        is_defect, score, heatmap = detector.predict(image=img_array)
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        # Comparar con ground truth
        expected = item['expected_detection']
        
        if expected and is_defect:
            true_positives += 1
            result_type = 'TP'
        elif expected and not is_defect:
            false_negatives += 1
            result_type = 'FN'
        elif not expected and is_defect:
            false_positives += 1
            result_type = 'FP'
        else:
            true_negatives += 1
            result_type = 'TN'
        
        results.append({
            'filename': item['filename'],
            'defect_type': item['defect_type'],
            'severity': item['severity'],
            'expected': expected,
            'predicted': is_defect,
            'score': float(score),
            'result_type': result_type,
            'latency_ms': latency_ms
        })
        
        # Progreso
        if i % 10 == 0:
            print(f"  Procesadas: {i}/{total_images} ({i*100//total_images}%)")
    
    # Calcular métricas
    print("\n📊 Calculando métricas...")
    
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    
    # Generar reporte
    report = {
        'summary': {
            'total_images': total_images,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency
        },
        'failures': [r for r in results if r['result_type'] in ['FP', 'FN']],
        'all_results': results
    }
    
    # Guardar reporte
    with open(output_report, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Imprimir resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE VALIDACIÓN")
    print("="*60)
    print(f"Total de imágenes: {total_images}")
    print(f"Verdaderos Positivos (TP): {true_positives}")
    print(f"Falsos Positivos (FP): {false_positives}")
    print(f"Verdaderos Negativos (TN): {true_negatives}")
    print(f"Falsos Negativos (FN): {false_negatives}")
    print(f"\n📈 Métricas:")
    print(f"  Recall (Sensibilidad): {recall*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  F1-Score: {f1_score*100:.1f}%")
    print(f"\n⏱️ Rendimiento:")
    print(f"  Latencia Promedio: {avg_latency:.1f}ms")
    print(f"  Latencia Máxima: {max_latency:.1f}ms")
    
    # Decisión GO/NO-GO
    print(f"\n{'='*60}")
    if recall >= 0.95 and precision >= 0.97:
        print("✅ DECISIÓN: GO - Sistema aprobado para producción")
    else:
        print("❌ DECISIÓN: NO-GO - Sistema requiere ajustes")
        if recall < 0.95:
            print(f"   Razón: Recall ({recall*100:.1f}%) < 95%")
        if precision < 0.97:
            print(f"   Razón: Precision ({precision*100:.1f}%) < 97%")
    print("="*60)
    
    print(f"\n📄 Reporte completo guardado en: {output_report}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Ejecutar validación de campo')
    parser.add_argument('--reference-id', type=int, required=True, help='ID de la referencia entrenada')
    parser.add_argument('--dataset', required=True, help='Directorio con imágenes adulteradas')
    parser.add_argument('--ground-truth', required=True, help='Archivo ground_truth.json')
    parser.add_argument('--output-report', default='validation_report.json', help='Archivo de reporte de salida')
    
    args = parser.parse_args()
    
    return run_validation(
        args.reference_id,
        args.dataset,
        args.ground_truth,
        args.output_report
    )

if __name__ == '__main__':
    sys.exit(main())
