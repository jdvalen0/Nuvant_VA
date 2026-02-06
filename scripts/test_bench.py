import cv2
import numpy as np
import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.core.features import FeatureExtractor
from backend.core.anomaly import AnomalyDetector
from backend.config import STORAGE_DIR

def apply_pixel_defect(img, x, y, size=5):
    """Simulates a micro-hole or impurity by setting a small square to 0 (black)."""
    adulterated = img.copy()
    adulterated[y:y+size, x:x+size] = 0
    return adulterated

def apply_stain(img, x, y, radius=15):
    """Simulates an oil stain or dirt using alpha-blending a dark circle."""
    adulterated = img.copy()
    overlay = adulterated.copy()
    cv2.circle(overlay, (x, y), radius, (20, 20, 20), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, adulterated, 1 - alpha, 0, adulterated)
    return adulterated

def apply_noise_patch(img, x, y, size=20):
    """Simulates localized texture degradation with Gaussian noise."""
    adulterated = img.copy()
    patch = adulterated[y:y+size, x:x+size]
    noise = np.random.normal(0, 50, patch.shape).astype(np.uint8)
    adulterated[y:y+size, x:x+size] = cv2.add(patch, noise)
    return adulterated

def run_test_bench(ref_id, sample_count=10):
    print(f"\n🚀 STARTING TEST BENCH FOR REFERENCE {ref_id}")
    print("-" * 50)
    
    ref_path = Path(STORAGE_DIR) / str(ref_id)
    model_path = ref_path / "model.pkl"
    samples_path = ref_path / "samples"
    
    if not model_path.exists():
        print(f"❌ Error: Model for reference {ref_id} not found.")
        return

    # 1. Load Model
    detector = AnomalyDetector()
    detector.load(model_path)
    extractor = FeatureExtractor()
    
    # 2. Get samples
    all_samples = [f for f in samples_path.iterdir() if f.is_file()]
    if not all_samples:
        print("❌ Error: No samples found.")
        return
        
    test_samples = all_samples[:sample_count]
    print(f"Selected {len(test_samples)} samples for testing.")
    
    # 3. Preparation
    results = []
    
    for i, img_path in enumerate(test_samples):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # A. Clean Inference
        feat_clean = extractor.extract(img)
        is_anom_clean, score_clean = detector.predict(feat_clean)
        
        # B. Adulteration 1: Pixel (Small) - Testing with standard and high sensitivity
        img_pixel = apply_pixel_defect(img, 100, 100, size=5)
        feat_pixel = extractor.extract(img_pixel)
        _, score_pixel = detector.predict(feat_pixel)
        is_anom_pixel_high, score_pixel_high = detector.predict(feat_pixel, sensitivity_offset=500) # Test high sensitivity
        
        # C. Adulteration 2: Stain (Medium)
        img_stain = apply_stain(img, 150, 50, radius=20)
        feat_stain = extractor.extract(img_stain)
        _, score_stain = detector.predict(feat_stain)
        
        # D. Adulteration 3: Noise (Texture)
        img_noise = apply_noise_patch(img, 50, 180, size=30)
        feat_noise = extractor.extract(img_noise)
        _, score_noise = detector.predict(feat_noise)
        
        results.append({
            "sample": img_path.name,
            "clean": float(score_clean),
            "pixel_defect": float(score_pixel),
            "pixel_defect_high_sens": float(score_pixel_high),
            "pixel_detected_high": bool(is_anom_pixel_high),
            "stain_defect": float(score_stain),
            "noise_defect": float(score_noise)
        })
        print(f"Sample {i+1}/{len(test_samples)} processed.")

    # 4. Report
    print("\n📊 PRECISION REPORT (Sensitivity Test)")
    print(f"{'Sample':<30} | {'Clean':>10} | {'Pixel(Std)':>10} | {'Pixel(+500)':>10} | {'Stain':>10}")
    print("-" * 100)
    
    pixel_std_detected = 0
    pixel_high_detected = 0
    
    for res in results:
        print(f"{res['sample']:<30} | {res['clean']:>10.2f} | {res['pixel_defect']:>10.2f} | {res['pixel_defect_high_sens']:>10.2f} | {res['stain_defect']:>10.2f}")
        if res['pixel_defect'] < 0: pixel_std_detected += 1
        if res['pixel_detected_high']: pixel_high_detected += 1
        
    print("-" * 100)
    print(f"Pixel Detection (Standard): {pixel_std_detected}/{len(results)}")
    print(f"Pixel Detection (High Sens): {pixel_high_detected}/{len(results)} ✅")

    # 5. Strict Model Test (Retraining)
    print("\n🔬 TESTING RIGOROUS RETRAINING (PCA=0.60, Contam=0.05)")
    strict_detector = AnomalyDetector(contamination=0.05, pca_variance=0.60)
    
    # Collect all features for retraining
    all_features = []
    for img_path in samples_path.iterdir():
        if not img_path.is_file(): continue
        img = cv2.imread(str(img_path))
        if img is None: continue
        all_features.append(extractor.extract(img))
    
    strict_detector.train(np.array(all_features).squeeze())
    
    strict_pixel_detected = 0
    for img_path in test_samples:
        img = cv2.imread(str(img_path))
        img_pixel = apply_pixel_defect(img, 100, 100, size=5)
        _, sc = strict_detector.predict(extractor.extract(img_pixel))
        if sc < 0: strict_pixel_detected += 1
    
    print(f"Pixel Detection (Strict Model): {strict_pixel_detected}/{len(results)} {'✅' if strict_pixel_detected > 0 else '❌'}")

    # Save results
    output_path = Path("scripts/tests/results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=int, default=1, help="Reference ID to test")
    parser.add_argument("--count", type=int, default=10, help="Number of samples")
    args = parser.parse_args()
    
    run_test_bench(args.ref, args.count)
