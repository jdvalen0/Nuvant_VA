import numpy as np
import cv2
import os

def generate_fabric_pattern(size=(512, 512), frequency=50, noise_level=20):
    """Generates a synthetic woven fabric pattern."""
    x = np.linspace(0, 10 * np.pi, size[1])
    y = np.linspace(0, 10 * np.pi, size[0])
    xv, yv = np.meshgrid(x, y)
    
    # Weave pattern: Sinusoidal grid
    z = np.sin(xv * frequency) + np.sin(yv * frequency)
    
    # Normalize to 0-255
    pattern = ((z + 2) / 4 * 255).astype(np.uint8)
    
    # Add noise to simulate real world imperfections
    noise = np.random.normal(0, noise_level, size).astype(np.int16)
    noisy_pattern = np.clip(pattern + noise, 0, 255).astype(np.uint8)
    
    return noisy_pattern

def add_defect(image, type='hole'):
    """Adds a synthetic defect to the image."""
    img = image.copy()
    h, w = img.shape
    
    if type == 'hole':
        # Dark circular spot
        center = (np.random.randint(50, w-50), np.random.randint(50, h-50))
        cv2.circle(img, center, 15, (20, 20, 20), -1)
    elif type == 'stain':
        # Brighter/Darker patch
        x = np.random.randint(50, w-50)
        y = np.random.randint(50, h-50)
        roi = img[y:y+40, x:x+40].astype(np.float32)
        roi = roi * 0.5 # Darken
        img[y:y+40, x:x+40] = roi.astype(np.uint8)
    elif type == 'tear':
        # Linear cut
        pt1 = (np.random.randint(50, w-50), np.random.randint(50, h-50))
        pt2 = (pt1[0] + np.random.randint(-20, 20), pt1[1] + np.random.randint(20, 50))
        cv2.line(img, pt1, pt2, (0, 0, 0), 2)
        
    return img

if __name__ == "__main__":
    output_dir = "/home/juan-david-valencia/Escritorio/Nuvant_VA/images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating training images (Normal)...")
    for i in range(20):
        img = generate_fabric_pattern(noise_level=15)
        # Add slight variations
        img = (img * np.random.uniform(0.9, 1.1)).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f"{output_dir}/train_normal_{i}.png", img)
        
    print("Generating test images (Defects)...")
    for i, d_type in enumerate(['hole', 'stain', 'tear']):
        img = generate_fabric_pattern(noise_level=15)
        defective_img = add_defect(img, d_type)
        cv2.imwrite(f"{output_dir}/test_defect_{d_type}_{i}.png", defective_img)
        
    print("Done. Check folder: " + output_dir)
