import numpy as np
import joblib
from sklearn.covariance import EllipticEnvelope, LedoitWolf  # H9: LedoitWolf moved to top
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

class AnomalyDetector:
    def __init__(self, method='robust_covariance', contamination=0.01, pca_variance=0.95):
        # We default to Robust Covariance (Mahalanobis) as it works best with Deep Features
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.contamination = contamination
        self.pca_variance = pca_variance
        # Threshold offset for sensitivity (defaults to 0, which is standard Mahalanobis boundary)
        self.threshold_offset = 0.0

    def train(self, features, contamination=0.01, pca_variance=None):
        """
        V31: Spectral Statistical Sovereign (CORREGIDO).
        Uses Mahalanobis Distance with Covariance Shrinkage (Ledoit-Wolf).
        This is the "Industrial Gold Standard" for patterned anomaly detection.
        
        FIX: Calibración corregida - usa distribución de peor-tile por imagen,
        no tiles individuales, para alinear con la decisión en predict().
        """
        X = np.array(features) # (N, 16, 2560)
        if X.ndim != 3:
            raise ValueError(f"Expected features with shape (N, T, D), got {X.shape}")
        N, T, D = X.shape
        X_all = X.reshape(N * T, D)
        
        # 1. Hyper-Vision PCA (0.99) - Essential for dimension reduction before Covariance
        pca_keep = pca_variance if pca_variance is not None else self.pca_variance
        self.pca = PCA(n_components=pca_keep)
        X_pca = self.pca.fit_transform(X_all)
        
        # 2. Spectral Modeling (The "Soul" of the Fabric)
        # Ledoit-Wolf is robust against noise and works with limited samples.
        self.cov_model = LedoitWolf()
        self.cov_model.fit(X_pca)
        
        self.mean_v = self.cov_model.location_
        self.precision_v = self.cov_model.precision_ # Inverse Covariance Matrix
        
        # 3. Statistical Boundary - CORREGIDO: Calibrado para peor-tile por imagen
        # dist = sqrt((x-mu)^T * Sigma^-1 * (x-mu))
        diff = X_pca - self.mean_v
        dists = np.sqrt(np.sum(np.dot(diff, self.precision_v) * diff, axis=1))
        
        # FIX CRÍTICO: Reconstruir distribución de peor-tile por imagen
        # Esto alinea la calibración con la decisión en predict() que usa max(dists)
        dists_by_image = dists.reshape(N, T)  # (N, 16)
        worst_per_image = np.max(dists_by_image, axis=1)  # (N,) - peor tile de cada imagen
        
        # Contamination determina el FPR objetivo en imágenes "buenas"
        percentile = 100.0 * (1.0 - (contamination or 0.01))
        self.gold_limit = np.percentile(worst_per_image, percentile) + 1e-6
        self.gold_median = float(np.median(worst_per_image))
        
        print(f"Model V31 (Spectral) trained. Limit: {self.gold_limit:.4f} (Target FPR: {contamination}, Calibrado por peor-tile)")

    def predict(self, feature_vector, sensitivity_offset=0.0):
        """
        V31 Prediction: Pure Statistical Mahalanobis (CORREGIDO).
        Detects "Geometric Spikes" that break the fabric's spectral signature.
        
        NOTA: sensitivity_offset se mantiene por compatibilidad, pero se recomienda
        usar contamination en train() para máximo rigor determinista.
        """
        if not hasattr(self, 'precision_v'):
            raise RuntimeError("Model not trained.")
        
        # Validar shape de entrada
        feature_vector = np.array(feature_vector)
        if feature_vector.ndim != 2 or feature_vector.shape[0] != 16:
            raise ValueError(f"Expected feature_vector with shape (16, D), got {feature_vector.shape}")
        
        # 1. Project
        X_test = self.pca.transform(feature_vector) # (16, D_pca)
        
        # 2. Mahalanobis Distance calculation
        diff = X_test - self.mean_v
        dists = np.sqrt(np.sum(np.dot(diff, self.precision_v) * diff, axis=1))
        
        # Worst tile error (decisión basada en peor tile)
        worst_dist = np.max(dists)
        
        # 3. Threshold con sensibilidad opcional (mantenido por compatibilidad)
        # Para máximo rigor, usar sensitivity_offset=0.0 y calibrar contamination en train()
        if sensitivity_offset >= 0:
            # 1.0 down to 0.5 (Strict)
            sens_p = 1.0 - (sensitivity_offset / 2000.0) 
        else:
            # 1.0 up to 5.0 (Loose)
            sens_p = 1.0 + (abs(sensitivity_offset) / 250.0)

        # Final Threshold (ahora calibrado correctamente con peor-tile)
        threshold = self.gold_limit * sens_p
        is_anomaly = worst_dist > threshold
        
        # 4. Quality Score [UI Standard] - CORREGIDO: Evita valores negativos
        # 100 = Muy cerca del centro estadístico. 0 = En el límite.
        # Usar relación inversa: mayor distancia = menor score
        if worst_dist <= threshold:
            # Dentro del límite: score positivo basado en distancia relativa
            quality_score = 100.0 * (1.0 - (worst_dist / (threshold + 1e-9)))
        else:
            # Fuera del límite: score negativo (defecto detectado)
            quality_score = -100.0 * ((worst_dist - threshold) / (threshold + 1e-9))
        
        # Normalizar a rango [0, 100] para UI (defectos muestran score bajo)
        quality_score_ui = max(0.0, min(100.0, 100.0 + quality_score))
        
        return is_anomaly, float(quality_score_ui)
        
    def save(self, path):
        state = {
            "pca": self.pca,
            "mean_v": self.mean_v,
            "precision_v": self.precision_v,
            "gold_limit": self.gold_limit,
            "gold_median": self.gold_median
        }
        joblib.dump(state, path)
        
    def load(self, path):
        state = joblib.load(path)
        self.pca = state["pca"]
        self.mean_v = state["mean_v"]
        self.precision_v = state["precision_v"]
        self.gold_limit = state["gold_limit"]
        self.gold_median = state["gold_median"]
