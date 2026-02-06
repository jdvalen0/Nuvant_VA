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
        V31: Spectral Statistical Sovereign.
        Uses Mahalanobis Distance with Covariance Shrinkage (Ledoit-Wolf).
        This is the "Industrial Gold Standard" for patterned anomaly detection.
        """
        X = np.array(features) # (N, 16, 2560)
        N, T, D = X.shape
        X_all = X.reshape(N * T, D)
        
        # 1. Hyper-Vision PCA (0.999) - Essential for dimension reduction before Covariance
        self.pca = PCA(n_components=0.99) # Reduced to 0.99 for better Covariance stability
        X_pca = self.pca.fit_transform(X_all)
        
        # 2. Spectral Modeling (The "Soul" of the Fabric)
        # Ledoit-Wolf is robust against noise and works with limited samples.
        self.cov_model = LedoitWolf()
        self.cov_model.fit(X_pca)
        
        self.mean_v = self.cov_model.location_
        self.precision_v = self.cov_model.precision_ # Inverse Covariance Matrix
        
        # 3. Statistical Boundary linking to UI Slider
        # dist = sqrt((x-mu)^T * Sigma^-1 * (x-mu))
        diff = X_pca - self.mean_v
        dists = np.sqrt(np.sum(np.dot(diff, self.precision_v) * diff, axis=1))
        
        # Slider 'contamination' (nu) determines the strictness. 
        # If user sets 0.01 (1%), we set limit at 99.0 percentile.
        percentile = 100.0 * (1.0 - (contamination or 0.01))
        self.gold_limit = np.percentile(dists, percentile) + 1e-6
        self.gold_median = np.median(dists)
        
        print(f"Model V31 (Spectral) trained. Limit: {self.gold_limit:.4f} (Rigor: {contamination})")

    def predict(self, feature_vector, sensitivity_offset=0.0):
        """
        V31 Prediction: Pure Statistical Mahalanobis.
        Detects "Geometric Spikes" that break the fabric's spectral signature.
        """
        if not hasattr(self, 'precision_v'):
            raise RuntimeError("Model not trained.")
        
        # 1. Project
        X_test = self.pca.transform(feature_vector) # (16, D_pca)
        
        # 2. Mahalanobis Distance calculation
        diff = X_test - self.mean_v
        dists = np.sqrt(np.sum(np.dot(diff, self.precision_v) * diff, axis=1))
        
        # Worst tile error
        worst_dist = np.max(dists)
        
        # 3. Sensitivity Scaling (V31 Balanced)
        # +1000 = Ultra Strict. 0 = Balanced. -1000 = Ultra Loose.
        if sensitivity_offset >= 0:
            # 1.0 down to 0.5 (Strict)
            sens_p = 1.0 - (sensitivity_offset / 2000.0) 
        else:
            # 1.0 up to 5.0 (Loose)
            sens_p = 1.0 + (abs(sensitivity_offset) / 250.0)

        # Final Threshold
        threshold = self.gold_limit * sens_p
        is_anomaly = worst_dist > threshold
        
        # 4. Quality Score [UI Standard]
        # 100 = Statistical Center. 0 = Boundary.
        quality_score = max(0.0, 100.0 * (1.1 - (worst_dist / (threshold + 1e-9))))
        
        return is_anomaly, min(quality_score, 100.0)
        
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
