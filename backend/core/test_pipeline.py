import glob
import cv2
import numpy as np
from features import FeatureExtractor
from anomaly import AnomalyDetector

def main():
    print("Initializing Core Pipeline Test...")
    
    # 1. Setup
    extractor = FeatureExtractor()
    detector = AnomalyDetector(method='ocsvm', nu=0.1) # Expect ~10% false positives in training
    
    # 2. Load Training Data
    train_files = glob.glob("/home/juan-david-valencia/Escritorio/Nuvant_VA/images/train_normal_*.png")
    if not train_files:
        print("ERROR: No training images found. Run dummy_data.py first.")
        return

    print(f"Loading {len(train_files)} training images...")
    train_features = []
    for f in train_files:
        img = cv2.imread(f)
        feats = extractor.extract(img)
        train_features.append(feats)
    
    train_features = np.array(train_features)
    print(f"Training Feature Matrix shape: {train_features.shape}")
    
    # 3. Train Model
    print("Training Anomaly Detector...")
    detector.train(train_features)
    
    # 4. Test on Defects
    test_files = glob.glob("/home/juan-david-valencia/Escritorio/Nuvant_VA/images/test_defect_*.png")
    print(f"\nTesting on {len(test_files)} defective images...")
    
    correct_detections = 0
    for f in test_files:
        img = cv2.imread(f)
        feats = extractor.extract(img)
        is_anomaly, score = detector.predict(feats)
        
        status = "DEFECT" if is_anomaly else "NORMAL"
        print(f"File: {f.split('/')[-1]} -> Prediction: {status} (Score: {score:.4f})")
        
        if is_anomaly:
            correct_detections += 1
            
    print(f"\nDetection Rate on Defects: {correct_detections}/{len(test_files)}")
    
    # 5. False Positive Test
    print("\nTesting on Training Data (Sanity Check)...")
    fp = 0
    for i in range(min(5, len(train_features))):
        is_anomaly, score = detector.predict(train_features[i])
        print(f"Train Img {i} -> Prediction: {'DEFECT' if is_anomaly else 'NORMAL'} (Score: {score:.4f})")
        if is_anomaly: fp += 1
        
    print(f"False Positives on known normal: {fp}/5")

if __name__ == "__main__":
    main()
