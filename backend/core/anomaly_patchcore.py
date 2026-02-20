"""
Nuvant VA - PatchCore Anomaly Detection Engine V32
Production-Ready Implementation with Localization

Based on: "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)

Mathematical Foundation:
- Feature Extraction: Mid-level CNN features (layers 2-3) to capture textures
- Memory Bank: Coreset subsampling via k-Center-Greedy algorithm
- Anomaly Score: Nearest-neighbor distance with density re-weighting
- Localization: Per-patch scores aggregated into heatmap
"""

import numpy as np
import os
import torch
import torch.nn as nn
import cv2
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from torchvision import transforms, models
from PIL import Image


class FeatureExtractorBackbone(nn.Module):
    """
    Feature extractor using pre-trained CNN backbone.
    Extracts mid-level features from specified layers.
    """
    
    def __init__(self, backbone_name: str = "wide_resnet50_2"):
        super().__init__()
        
        # Load pre-trained backbone
        if backbone_name == "wide_resnet50_2":
            backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        
        # Extract layers
        self.layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from multiple layers."""
        x = self.layer1(x)
        layer2_out = self.layer2(x)
        layer3_out = self.layer3(layer2_out)
        
        return {
            "layer2": layer2_out,
            "layer3": layer3_out
        }


class PatchCoreDetector:
    """
    PatchCore-based anomaly detector for fabric inspection.
    
    Features:
    - ~99% AUROC on industrial benchmarks
    - Pixel-level localization (heatmaps)
    - Coreset memory bank for efficient inference
    - Production-ready
    """
    
    def __init__(self, 
                 backbone: str = None,
                 coreset_sampling_ratio: float = None,
                 num_neighbors: int = None):
        """
        Initialize PatchCore detector.
        """
        # Load from arguments or environment variables
        self.backbone_name = backbone or os.getenv("PATCHCORE_BACKBONE", "wide_resnet50_2")
        self.coreset_sampling_ratio = coreset_sampling_ratio or float(os.getenv("PATCHCORE_CORESET_RATIO", "0.1"))
        self.num_neighbors = num_neighbors or int(os.getenv("PATCHCORE_NEIGHBORS", "9"))
        
        # Industrial Pre-processing Params
        # CROP_RATIO: 0.05 = crop 5% from each side to avoid fabric edges
        self.roi_crop = float(os.getenv("PATCHCORE_ROI_CROP", "0.05"))
        self.use_clahe = os.getenv("PATCHCORE_USE_CLAHE", "false").lower() == "true"
        
        # Model state
        self.feature_extractor: Optional[FeatureExtractorBackbone] = None
        self.memory_bank: Optional[np.ndarray] = None
        self.is_trained = False
        self.threshold = 0.5
        self.image_size = (224, 224)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._init_feature_extractor()
    
    def _apply_industrial_preprocess(self, pil_img: Image.Image) -> Image.Image:
        """Apply ROI crop and optional CLAHE."""
        # 1. ROI Crop (Remove edges where fabric might be irregular)
        if self.roi_crop > 0:
            w, h = pil_img.size
            left = w * self.roi_crop
            top = h * self.roi_crop
            right = w * (1 - self.roi_crop)
            bottom = h * (1 - self.roi_crop)
            pil_img = pil_img.crop((left, top, right, bottom))
            
        # 2. CLAHE (Better contrast, light invariance)
        if self.use_clahe:
            img_np = np.array(pil_img)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            pil_img = Image.fromarray(final_img)
            
        return pil_img

    def _init_feature_extractor(self):
        """Initialize the feature extraction backbone."""
        self.feature_extractor = FeatureExtractorBackbone(self.backbone_name)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
    
    def _extract_features(self, tensor: torch.Tensor) -> np.ndarray:
        """Extract and concatenate features from backbone layers."""
        with torch.no_grad():
            features = self.feature_extractor(tensor)
            
            # Get layer2 and layer3 features
            layer2 = features["layer2"]
            layer3 = features["layer3"]
            
            # Upsample layer3 to match layer2 size
            H2, W2 = layer2.shape[2], layer2.shape[3]
            layer3_up = torch.nn.functional.interpolate(
                layer3, size=(H2, W2), mode="bilinear", align_corners=False
            )
            
            combined = torch.cat([layer2, layer3_up], dim=1)
            
            # ========== NEIGHBORHOOD AGGREGATION (arXiv:2106.08265) ==========
            # Paper requirement: "Locally aware patch features by neighborhood aggregation"
            # We use AvgPool2d to mix local features, increasing robustness
            padding = 1
            stride = 1
            kernel_size = 3
            avg_pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
            combined = avg_pool(combined)
            # ===============================================================
            
            B, C, H, W = combined.shape
            features_flat = combined.permute(0, 2, 3, 1).reshape(-1, C)
            
            features_norm = torch.nn.functional.normalize(features_flat, p=2, dim=1)
            
            return features_norm.cpu().numpy(), (H, W)
    
    def _coreset_subsampling(self, features: np.ndarray, ratio: float) -> np.ndarray:
        """
        Greedy coreset subsampling using k-Center algorithm.
        
        Selects a subset that maximally covers the feature space.
        """
        n_samples = max(100, min(int(len(features) * ratio), 5000))
        
        if len(features) <= n_samples:
            return features
        
        # k-Center Greedy
        selected_indices = [np.random.randint(len(features))]
        min_distances = np.full(len(features), np.inf)
        
        for _ in range(n_samples - 1):
            # Update minimum distances
            last_selected = features[selected_indices[-1]]
            distances = np.linalg.norm(features - last_selected, axis=1)
            min_distances = np.minimum(min_distances, distances)
            
            # Select point with maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return features[selected_indices]
    
    def train(self, 
              image_paths: list = None,
              images: list = None,
              contamination: float = 0.01) -> Dict[str, Any]:
        """
        Train PatchCore on normal (defect-free) images.
        
        Args:
            image_paths: List of paths to normal images
            images: List of numpy arrays (BGR format)
            contamination: Expected fraction of anomalies for threshold calibration
            
        Returns:
            Training statistics
        """
        print(f"[PatchCore V32] Training...")
        
        # Collect features from all training images
        all_features = []
        
        # Determine source
        if images is not None:
            source = images
            is_array = True
        elif image_paths is not None:
            source = image_paths
            is_array = False
        else:
            raise ValueError("Must provide either image_paths or images")
        
        print(f"[PatchCore V32] Processing {len(source)} training images...")
        
        for idx, item in enumerate(source):
            if is_array:
                img_bgr = item
                img_rgb = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
            else:
                img_bgr = cv2.imread(item)
                pil_img = Image.open(item).convert("RGB")
                
            # Skip "Dead/Total Dark" frames during training to avoid memory corruption
            if img_bgr is not None:
                gray_check = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                mean_b = np.mean(gray_check)
                if mean_b < 0.2:
                    print(f" [PatchCore V32] Skipping image {idx}: Too dark (mean_brightness={mean_b:.2f})")
                    continue
            
            # Apply ROI and CLAHE
            pil_img = self._apply_industrial_preprocess(pil_img)
            
            # Transform and add batch dimension
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Extract features
            features, _ = self._extract_features(tensor)
            all_features.append(features)
        
        # Combine all features
        all_features = np.vstack(all_features)
        print(f"[PatchCore V32] Total features: {all_features.shape}")
        
        # Coreset subsampling
        self.memory_bank = self._coreset_subsampling(all_features, self.coreset_sampling_ratio)
        print(f"[PatchCore V32] Coreset size: {self.memory_bank.shape}")
        
        # Calibrate threshold based on training data distances
        train_dists = self._compute_distances(all_features)
        percentile = 100.0 * (1.0 - contamination)
        
        # CORREGIDO CRÍTICO: Calibración MUY generosa para GARANTIZAR cero falsos positivos
        # El problema es que incluso con percentil 99, algunas imágenes pueden quedar fuera
        # Solución: Usar el MÁXIMO real de entrenamiento + margen generoso
        base_threshold = np.percentile(train_dists, percentile)
        max_train_dist = np.max(train_dists)
        median_train_dist = np.median(train_dists)
        
        # ESTRATEGIA: Usar el máximo entre:
        # 1. Máximo real + 10% (garantiza que TODAS las imágenes de entrenamiento queden dentro)
        # 2. Percentil 99 + 15% (margen adicional de seguridad)
        # 3. Mediana * 3 (fallback para casos extremos)
        # Esto es MUY generoso pero garantiza cero falsos positivos en training data
        safety_threshold = max(
            max_train_dist * 1.10,  # Máximo + 10% - GARANTIZA que todas las imágenes de entrenamiento queden dentro
            base_threshold * 1.15,   # Percentil + 15% - margen adicional
            median_train_dist * 3.0  # Fallback para casos extremos
        )
        
        self.threshold = safety_threshold
        
        print(f"[PatchCore V32] Threshold calibration:")
        print(f"  Percentile {percentile}% = {base_threshold:.4f}")
        print(f"  Max training dist = {max_train_dist:.4f}")
        print(f"  Median = {median_train_dist:.4f}")
        print(f"  Final threshold = {self.threshold:.4f} (Max*1.10={max_train_dist*1.10:.4f})")
        print(f"  Margin over max = {(self.threshold / max_train_dist - 1.0) * 100:.1f}%")
        
        self.is_trained = True
        
        print(f"[PatchCore V32] Trained. Threshold: {self.threshold:.4f}")
        
        return {
            "memory_bank_size": len(self.memory_bank),
            "threshold": self.threshold,
            "num_training_images": len(source)
        }
    
    def _compute_distances(self, features: np.ndarray) -> np.ndarray:
        """
        Compute nearest-neighbor distances to memory bank with Density Reweighting (arXiv:2106.08265).
        """
        if self.memory_bank is None:
            raise ValueError("Model not trained")
        
        # 1. Compute cosine similarities (L2 normalized)
        similarities = features @ self.memory_bank.T  # (N_patches, N_coreset)
        
        # 2. Get top-K neighbors
        k = min(self.num_neighbors, len(self.memory_bank))
        # partitioning is faster than sorting for just top-K
        top_k_indices = np.argpartition(-similarities, k-1, axis=1)[:, :k]
        
        # Extract top similarities for each patch
        rows = np.arange(features.shape[0])[:, None]
        top_similarities = similarities[rows, top_k_indices] # (N_patches, k)
        
        # Sort them within the top-K (optional but cleaner)
        top_similarities = -np.sort(-top_similarities, axis=1)
        
        # 3. Compute distances (1 - sim)
        top_distances = 1.0 - top_similarities
        
        # 4. Density Reweighting [Equation 6 in Paper]
        # s = (1 - exp(-max_dist)) * max_dist / weight
        # Simplified robust version: s = (1 - softmax_weight) * max_dist
        max_similarity = top_similarities[:, 0]
        max_dist = 1.0 - max_similarity
        
        # Softmax-like weighting
        weights = np.exp(top_similarities)
        weights_sum = np.sum(weights, axis=1)
        soft_max_weight = 1.0 - (weights[:, 0] / weights_sum)
        
        # Final reweighted score per patch
        reweighted_distances = soft_max_weight * max_dist
        
        return reweighted_distances
    
    def predict(self, 
                image: np.ndarray = None,
                image_path: str = None,
                sensitivity_offset: float = 0.0) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Predict if image contains anomaly and generate heatmap.
        
        Args:
            image: BGR numpy array
            image_path: Path to image file
            sensitivity_offset: Adjustment to threshold (negative = more sensitive)
            
        Returns:
            (is_anomaly, anomaly_score, heatmap)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Load/convert image
        if image is not None:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
        elif image_path is not None:
            pil_img = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("Must provide either image or image_path")
        
        original_size = pil_img.size  # (W, H)
        
        # Apply ROI and CLAHE
        pil_img = self._apply_industrial_preprocess(pil_img)
        
        # Transform
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Extract features
        features, (H, W) = self._extract_features(tensor)
        
        # Compute per-patch distances
        patch_distances = self._compute_distances(features)
        
        # Reshape to spatial map (this map corresponds to the CROPPED/processed image)
        distance_map = patch_distances.reshape(H, W)
        
        # Upscale distances to match the PRE-PROCESSED (cropped) size first
        cropped_w, cropped_h = pil_img.size
        heatmap_cropped = cv2.resize(distance_map.astype(np.float32), (cropped_w, cropped_h), interpolation=cv2.INTER_LINEAR)
        
        # 5. Gaussian Blur (per Paper arXiv:2106.08265, sigma=4)
        # Smooths the blocky patch predictions into a organic "thermal" map
        heatmap_cropped = cv2.GaussianBlur(heatmap_cropped, (0, 0), sigmaX=4, sigmaY=4)
        
        # Create full-size heatmap with zeros
        full_heatmap = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
        
        # Calculate ROI coordinates in the full image
        w, h = original_size
        left = int(w * self.roi_crop)
        top = int(h * self.roi_crop)
        
        # Paste the cropped heatmap into the middle of the full-size heatmap
        hh, ww = heatmap_cropped.shape
        full_heatmap[top:top+hh, left:left+ww] = heatmap_cropped[:min(hh, h-top), :min(ww, w-left)]
        
        # 6. Industrial Normalization (Threshold-Relative)
        # Old method (Min-Max) caused "Red Noise" on perfect fabrics.
        # New method (Corrected): Enhanced Visualization Logic
        if self.threshold > 0:
            # Normalize relative to threshold (0.5 = threshold)
            # We want to boost visibility of anomalies, so we don't clamp too early
            # factor = 1.0 means val == threshold -> 0.5 (Green/Yellow transition)
            heatmap_normalized = 0.5 * (full_heatmap / self.threshold)
            
            # Non-linear boost (Sqrt) to make faint anomalies more visible (Thermal Effect)
            # Safety Clip to prevent NaN on tiny negative float errors
            heatmap_normalized = np.clip(heatmap_normalized, 0, None)
            heatmap_normalized = np.power(heatmap_normalized, 0.7)
            
            heatmap_normalized = np.clip(heatmap_normalized, 0, 1)
        else:
            # Fallback
             heatmap_min, heatmap_max = full_heatmap.min(), full_heatmap.max()
             if heatmap_max > heatmap_min:
                 heatmap_normalized = (full_heatmap - heatmap_min) / (heatmap_max - heatmap_min)
             else:
                 heatmap_normalized = np.zeros_like(full_heatmap)
        
        # Compute overall score (max of patch distances)
        max_distance = np.max(patch_distances)
        
        # Apply sensitivity offset
        # Positive offset (Strict) -> Lower threshold -> More detections
        # Negative offset (Ignore) -> Higher threshold -> Less detections
        adjusted_threshold = self.threshold * (1.0 - sensitivity_offset / 1000.0)
        
        # Determine if anomaly
        is_anomaly = max_distance > adjusted_threshold
        
        # Normalize score to 0-100 scale (relative to adjusted threshold)
        # 50.0 means exactly at the threshold. > 50.0 is anomaly.
        score = min(100.0, (max_distance / (adjusted_threshold + 1e-6)) * 50.0)
        
        return is_anomaly, score, heatmap_normalized
    
    def save(self, path: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        save_dict = {
            "memory_bank": self.memory_bank,
            "threshold": self.threshold,
            "backbone": self.backbone_name,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
            "image_size": self.image_size,
            "version": "V32_PatchCore"
        }
        
        joblib.dump(save_dict, path)
        print(f"[PatchCore V32] Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model from disk."""
        save_dict = joblib.load(path)
        
        if save_dict.get("version") != "V32_PatchCore":
            print(f"[Warning] Loading model with version: {save_dict.get('version')}")
        
        self.memory_bank = save_dict["memory_bank"]
        self.threshold = save_dict["threshold"]
        self.backbone_name = save_dict.get("backbone", "wide_resnet50_2")
        self.image_size = save_dict.get("image_size", (224, 224))
        
        # Recreate feature extractor
        self._init_feature_extractor()
        
        self.is_trained = True
        print(f"[PatchCore V32] Model loaded. Memory bank: {self.memory_bank.shape}, Threshold: {self.threshold:.4f}")


# Compatibility wrapper for existing API
class AnomalyDetectorV32(PatchCoreDetector):
    """Backward-compatible wrapper that matches V31 API."""
    
    def train(self, features=None, contamination=0.01, pca_variance=None, images=None):
        """
        Train with V31-style API or new image-based API.
        """
        if images is not None:
            # New V32 API: train on images directly
            return super().train(images=images, contamination=contamination)
        elif features is not None:
            # V31 compatibility: use features directly as memory bank
            features = np.array(features)
            if features.ndim == 3:
                N, T, D = features.shape
                features = features.reshape(N * T, D)
            
            # L2 normalize
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features_norm = features / (norms + 1e-9)
            
            # Coreset subsampling
            self.memory_bank = self._coreset_subsampling(features_norm, self.coreset_sampling_ratio)
            
            # Calibrate threshold
            train_dists = self._compute_distances(features_norm)
            percentile = 100.0 * (1.0 - contamination)
            self.threshold = np.percentile(train_dists, percentile)
            
            self.is_trained = True
            print(f"[PatchCore V32] Trained (features mode). Memory: {self.memory_bank.shape}, Threshold: {self.threshold:.4f}")
            
            return {"memory_bank_size": len(self.memory_bank), "threshold": self.threshold}
        else:
            raise ValueError("Must provide features or images")
    
    def predict(self, features=None, image=None, sensitivity_offset=0.0):
        """
        Unified prediction API for V31 (features) and V32 (images).
        Includes auto-detection for positional arguments.
        """
        # AUTO-DETECTION: If 'features' contains a numpy image, redirect to 'image'
        if image is None and features is not None:
            if isinstance(features, np.ndarray) and features.ndim == 3:
                image = features
                features = None

        # Case 1: Image Processing (Superior PatchCore V32 Engine)
        if image is not None:
            is_anomaly, score, heatmap = super().predict(
                image=image, 
                sensitivity_offset=sensitivity_offset
            )
            return is_anomaly, score, heatmap
            
        # Case 2: External Features (V31 Compatibility)
        if features is not None:
            features = np.array(features)
            
            # Canonical flattening for PatchCore (N, D)
            if features.ndim == 3:
                N, T, D = features.shape
                features_flat = features.reshape(N * T, D)
            else:
                features_flat = features.reshape(-1, features.shape[-1])
            
            # Dimension Guard
            if self.memory_bank is not None:
                if features_flat.shape[1] != self.memory_bank.shape[1]:
                    # Likely a V31 feature vector (2560 dims) vs V32 model (1536 dims)
                    raise ValueError(f"Dim mismatch: expected {self.memory_bank.shape[1]}, got {features_flat.shape[1]}")
            
            # Predict
            distances = self._compute_distances(features_flat)
            max_distance = np.max(distances)
            
            # Apply sensitivity
            adj_threshold = self.threshold * (1.0 - sensitivity_offset / 1000.0)
            is_anomaly = max_distance > adj_threshold
            score = min(100.0, (max_distance / (adj_threshold + 1e-6)) * 50.0)
            
            return is_anomaly, score, None
            
        raise ValueError("Must provide features or image")
