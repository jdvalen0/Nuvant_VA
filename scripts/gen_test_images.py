import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Reuse functions from test_bench
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.test_bench import apply_pixel_defect, apply_stain, apply_noise_patch

def generate_visual_test_samples(ref_id=1):
    from backend.config import STORAGE_DIR
    ref_path = Path(STORAGE_DIR) / str(ref_id)
    samples_path = ref_path / "samples"
    output_dir = Path("scripts/tests/visual_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample = next(samples_path.iterdir())
    img = cv2.imread(str(sample))
    
    # Large pixel defect
    v1 = apply_pixel_defect(img, 100, 100, size=20)
    cv2.imwrite(str(output_dir / "test_defect_pixel.png"), v1)
    
    # Large stain
    v2 = apply_stain(img, 120, 120, radius=40)
    cv2.imwrite(str(output_dir / "test_defect_stain.png"), v2)
    
    # Heavy noise
    v3 = apply_noise_patch(img, 50, 50, size=80)
    cv2.imwrite(str(output_dir / "test_defect_noise.png"), v3)
    
    print(f"✅ Visual test samples generated in {output_dir}")

if __name__ == "__main__":
    generate_visual_test_samples()
