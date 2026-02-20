#!/usr/bin/env python3
"""
Script de Generación de Dataset de Validación Sintética
Genera adulteraciones controladas de imágenes limpias para validar el sistema.
"""
import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from pathlib import Path

def create_pixel_alteration(img_array, size):
    """Altera un parche de píxeles en el centro de la imagen."""
    h, w = img_array.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Crear defecto (píxeles blancos)
    half_size = size // 2
    x1, y1 = max(0, cx - half_size), max(0, cy - half_size)
    x2, y2 = min(w, cx + half_size), min(h, cy + half_size)
    
    img_defect = img_array.copy()
    img_defect[y1:y2, x1:x2] = 255  # Blanco puro
    
    return img_defect, (x1, y1, x2 - x1, y2 - y1)

def add_gaussian_noise(img_array, sigma):
    """Añade ruido Gaussiano."""
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(img_array, density):
    """Añade ruido Salt-and-Pepper."""
    noisy = img_array.copy()
    num_salt = int(density * img_array.size * 0.5)
    num_pepper = int(density * img_array.size * 0.5)
    
    # Salt (blanco)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_array.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    
    # Pepper (negro)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img_array.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    
    return noisy

def adjust_brightness(img_pil, factor):
    """Ajusta el brillo de la imagen."""
    enhancer = ImageEnhance.Brightness(img_pil)
    return enhancer.enhance(factor)

def compress_jpeg(img_pil, quality):
    """Comprime la imagen con JPEG."""
    import io
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def rotate_image(img_array, angle):
    """Rota la imagen."""
    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_array, M, (w, h), borderValue=(0, 0, 0))
    return rotated

def generate_adulterations(base_image_path, output_dir):
    """Genera todas las adulteraciones para una imagen base."""
    base_name = Path(base_image_path).stem
    img_pil = Image.open(base_image_path).convert('RGB')
    img_array = np.array(img_pil)
    
    adulterations = []
    
    # 1. Adulteraciones de Píxeles (4 variantes)
    for size, label in [(1, '1px'), (3, '3x3'), (10, '10x10'), (50, '50x50')]:
        img_defect, bbox = create_pixel_alteration(img_array, size)
        output_path = output_dir / f"{base_name}_pixel_{label}.jpg"
        Image.fromarray(img_defect).save(output_path, quality=95)
        
        adulterations.append({
            'filename': output_path.name,
            'base_image': Path(base_image_path).name,
            'defect_type': 'pixel_alteration',
            'severity': label,
            'expected_detection': size >= 3,  # Solo 3x3 o mayor debería detectarse
            'defect_location': bbox
        })
    
    # 2. Ruido Gaussiano (2 variantes)
    for sigma, label in [(10, 'sigma10'), (20, 'sigma20')]:
        img_noisy = add_gaussian_noise(img_array, sigma)
        output_path = output_dir / f"{base_name}_noise_gaussian_{label}.jpg"
        Image.fromarray(img_noisy).save(output_path, quality=95)
        
        adulterations.append({
            'filename': output_path.name,
            'base_image': Path(base_image_path).name,
            'defect_type': 'gaussian_noise',
            'severity': label,
            'expected_detection': False,  # Ruido no debería detectarse como defecto
            'defect_location': None
        })
    
    # 3. Ruido Salt-Pepper (2 variantes)
    for density, label in [(0.01, 'density001'), (0.05, 'density005')]:
        img_noisy = add_salt_pepper_noise(img_array, density)
        output_path = output_dir / f"{base_name}_noise_saltpepper_{label}.jpg"
        Image.fromarray(img_noisy).save(output_path, quality=95)
        
        adulterations.append({
            'filename': output_path.name,
            'base_image': Path(base_image_path).name,
            'defect_type': 'salt_pepper_noise',
            'severity': label,
            'expected_detection': density >= 0.05,  # Solo alta densidad debería detectarse
            'defect_location': None
        })
    
    # 4. Variaciones de Iluminación (4 variantes)
    for factor, label in [(0.7, 'sub30'), (0.5, 'sub50'), (1.3, 'over30'), (1.5, 'over50')]:
        img_bright = adjust_brightness(img_pil, factor)
        output_path = output_dir / f"{base_name}_lighting_{label}.jpg"
        img_bright.save(output_path, quality=95)
        
        adulterations.append({
            'filename': output_path.name,
            'base_image': Path(base_image_path).name,
            'defect_type': 'lighting_variation',
            'severity': label,
            'expected_detection': False,  # Iluminación no debería detectarse como defecto
            'defect_location': None
        })
    
    # 5. Compresión JPEG (2 variantes)
    for quality, label in [(30, 'q30'), (10, 'q10')]:
        img_compressed = compress_jpeg(img_pil, quality)
        output_path = output_dir / f"{base_name}_compression_{label}.jpg"
        img_compressed.save(output_path, quality=quality)
        
        adulterations.append({
            'filename': output_path.name,
            'base_image': Path(base_image_path).name,
            'defect_type': 'jpeg_compression',
            'severity': label,
            'expected_detection': False,  # Compresión no debería detectarse
            'defect_location': None
        })
    
    # 6. Transformaciones Geométricas (3 variantes)
    # Rotación
    img_rotated = rotate_image(img_array, 5)
    output_path = output_dir / f"{base_name}_geometric_rotate5.jpg"
    Image.fromarray(img_rotated).save(output_path, quality=95)
    adulterations.append({
        'filename': output_path.name,
        'base_image': Path(base_image_path).name,
        'defect_type': 'geometric_rotation',
        'severity': 'rotate5',
        'expected_detection': False,
        'defect_location': None
    })
    
    # Escalado
    h, w = img_array.shape[:2]
    img_scaled = cv2.resize(img_array, (int(w * 0.9), int(h * 0.9)))
    img_scaled = cv2.resize(img_scaled, (w, h))  # Volver al tamaño original
    output_path = output_dir / f"{base_name}_geometric_scale90.jpg"
    Image.fromarray(img_scaled).save(output_path, quality=95)
    adulterations.append({
        'filename': output_path.name,
        'base_image': Path(base_image_path).name,
        'defect_type': 'geometric_scale',
        'severity': 'scale90',
        'expected_detection': False,
        'defect_location': None
    })
    
    return adulterations

def main():
    parser = argparse.ArgumentParser(description='Generar dataset de validación sintética')
    parser.add_argument('--base-images', required=True, help='Directorio con imágenes limpias')
    parser.add_argument('--output', required=True, help='Directorio de salida')
    parser.add_argument('--num-samples', type=int, default=5, help='Número de imágenes base a usar')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar imágenes
    image_files = list(base_dir.rglob('*.jpg')) + list(base_dir.rglob('*.png')) + list(base_dir.rglob('*.jpeg'))
    
    if len(image_files) == 0:
        print(f"❌ No se encontraron imágenes en {base_dir}")
        return 1
    
    # Seleccionar muestra
    selected = image_files[:args.num_samples]
    print(f"📸 Procesando {len(selected)} imágenes base...")
    
    all_adulterations = []
    for i, img_path in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {img_path.name}...")
        adulterations = generate_adulterations(img_path, output_dir)
        all_adulterations.extend(adulterations)
    
    # Guardar ground truth
    gt_path = output_dir / 'ground_truth.json'
    with open(gt_path, 'w') as f:
        json.dump({
            'total_images': len(all_adulterations),
            'base_images': len(selected),
            'adulterations_per_image': len(all_adulterations) // len(selected),
            'data': all_adulterations
        }, f, indent=2)
    
    print(f"\n✅ Generadas {len(all_adulterations)} imágenes adulteradas")
    print(f"📄 Ground truth guardado en: {gt_path}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
