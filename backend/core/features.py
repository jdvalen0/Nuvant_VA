import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        # Load pre-trained MobileNetV2 (efficient and robust)
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove the classification head (classifier) to get raw embeddings
        self.model.classifier = torch.nn.Identity()
        self.model.to(device)
        self.model.eval()

        # Standard ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image):
        """
        V21: 4x4 Tiling (16 Tiles) + Global Context.
        Total Precision: 16 Tiles * 1280 dims + 1280 Global = Extremely high defect focus.
        Output shape: (16, 2560)
        """
        if image is None:
            raise ValueError("Image is empty")

        # Color conversion (BGR -> RGB)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ========== FRAME QUALITY FILTER (H7 - CORREGIDO PARA TELAS NEGRAS Y CONSISTENCIA) ==========
        # Filtro mejorado: permite telas negras válidas, solo rechaza casos extremos
        # IMPORTANTE: Debe ser consistente con el filtro de entrenamiento de PatchCore V32 (mean_b < 0.2)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = np.mean(gray)
        
        # 1. Luminance Check (CORREGIDO: Consistente con entrenamiento PatchCore V32)
        # PatchCore V32 rechaza imágenes con mean_b < 0.2 durante entrenamiento
        # Para inferencia, usamos umbral ligeramente más permisivo (0.2) para ser consistente
        # pero aún rechazamos casos extremos de lens cap
        if mean_brightness < 0.2:  # Consistente con filtro de entrenamiento PatchCore V32
            raise ValueError(f"Frame quality rejected: Total darkness (brightness: {mean_brightness:.2f})")
        if mean_brightness > 250.0:  # Almost white (sobreexposición total)
            raise ValueError(f"Frame quality rejected: Overexposed (brightness: {mean_brightness:.2f})")
        
        # 2. Blur Detection (CORREGIDO: Más permisivo, solo rechaza blur real)
        # Solo rechazar si hay baja textura Y muy baja luminancia (probable lens cap/blur real)
        # NO rechazar telas negras con textura (aunque tengan baja varianza)
        # Umbral de blur muy bajo para permitir telas lisas, pero combinado con brightness muy bajo
        BLUR_THRESHOLD = 0.01  # Muy bajo para permitir telas lisas
        if laplacian_var < BLUR_THRESHOLD and mean_brightness < 0.5:
            # Solo rechazar si hay baja textura Y luminancia extremadamente baja (lens cap/blur real)
            # Esto permite telas negras válidas con brightness >= 0.5
            raise ValueError(f"Frame quality rejected: No texture/Blur (Var: {laplacian_var:.4f}, Brightness: {mean_brightness:.2f})")
        # ========== END FRAME QUALITY FILTER ==========
            
        h, w = image.shape[:2]
        # Force a multiple of 4 for clean tiling
        image = cv2.resize(image, (448, 448))
        h, w = 448, 448
             
        # 1. Global Context Embedding
        global_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            global_feat = self.model(global_tensor).cpu().numpy().flatten() 
            # L2 Normalization (Brightness Invariance)
            global_feat = global_feat / (np.linalg.norm(global_feat) + 1e-9)

        # 2. Local 4x4 Tiling
        step_h, step_w = h // 4, w // 4
        tiles = []
        for i in range(4):
            for j in range(4):
                y1, y2 = i * step_h, (i + 1) * step_h
                x1, x2 = j * step_w, (j + 1) * step_w
                tiles.append(image[y1:y2, x1:x2])
        
        batch_tensors = []
        for tile in tiles:
            batch_tensors.append(self.preprocess(tile))
            
        input_batch = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            tile_feats = self.model(input_batch).cpu().numpy() # (16, 1280)
            # L2 Normalization per tile
            for i in range(16):
                tile_feats[i] = tile_feats[i] / (np.linalg.norm(tile_feats[i]) + 1e-9)
        
        # 3. Hybrid Concatenation: (16, 1280) + (16, 1280) -> (16, 2560)
        hybrid_features = []
        for i in range(16):
            combined = np.concatenate([tile_feats[i], global_feat])
            # Re-normalize combined vector
            combined = combined / (np.linalg.norm(combined) + 1e-9)
            hybrid_features.append(combined)
            
        return np.array(hybrid_features)
