# src/mlshield/detectors/models/isolation.py
"""Isolation Forest wrapper for GPU telemetry anomaly detection."""
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional
import pickle
from pathlib import Path


class GPUIsolationForest:
    """
    Isolation Forest for detecting anomalous GPU telemetry patterns.

    Trained on normal GPU metrics, flags unusual patterns that may indicate
    cryptojacking, unauthorized distillation, or resource abuse.
    """

    FEATURE_KEYS = [
        "DCGM_FI_DEV_GPU_UTIL",
        "DCGM_FI_DEV_MEM_COPY_UTIL",
        "DCGM_FI_DEV_FB_USED",
        "DCGM_FI_DEV_GPU_TEMP",
        "DCGM_FI_DEV_POWER_USAGE",
        "DCGM_FI_DEV_ENC_UTIL",
    ]

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.is_fitted = False
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    def extract_gpu_features(self, details: dict) -> Optional[np.ndarray]:
        """Extract GPU feature vector from event details. Uses 0.0 for missing optional metrics."""
        # Must have at least GPU_UTIL to count as a GPU event
        if "DCGM_FI_DEV_GPU_UTIL" not in details:
            return None
        values = []
        for key in self.FEATURE_KEYS:
            values.append(float(details.get(key, 0.0)))
        return np.array(values, dtype=np.float32)

    def fit(self, gpu_events: list[dict]):
        """Train on a list of normal GPU metric dicts."""
        features = []
        for details in gpu_events:
            feat = self.extract_gpu_features(details)
            if feat is not None:
                features.append(feat)

        if len(features) < 10:
            return  # Not enough data

        X = np.array(features)

        # Normalize
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0) + 1e-8
        X_normalized = (X - self._scaler_mean) / self._scaler_std

        self.model.fit(X_normalized)
        self.is_fitted = True

    def score(self, details: dict) -> float:
        """
        Score a single GPU metrics event.
        Returns anomaly score in [0, 1] where higher = more anomalous.
        """
        if not self.is_fitted:
            return 0.0

        feat = self.extract_gpu_features(details)
        if feat is None:
            return 0.0

        X = (feat - self._scaler_mean) / self._scaler_std
        X = X.reshape(1, -1)

        # IsolationForest.score_samples returns negative scores
        # More negative = more anomalous
        raw_score = self.model.score_samples(X)[0]

        # Convert to [0, 1] where 1 = most anomalous
        # Typical range is roughly [-0.7, 0.0] for normal, < -0.7 for anomalous
        anomaly_score = max(0.0, min(1.0, -raw_score))
        return float(anomaly_score)

    def predict(self, details: dict) -> bool:
        """Return True if anomalous."""
        if not self.is_fitted:
            return False

        feat = self.extract_gpu_features(details)
        if feat is None:
            return False

        X = (feat - self._scaler_mean) / self._scaler_std
        X = X.reshape(1, -1)

        return self.model.predict(X)[0] == -1

    def save(self, path: str):
        """Save model to disk."""
        data = {
            "model": self.model,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "is_fitted": self.is_fitted,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._scaler_mean = data["scaler_mean"]
        self._scaler_std = data["scaler_std"]
        self.is_fitted = data["is_fitted"]
