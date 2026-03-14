# src/mlshield/detectors/models/lstm_detector.py
import torch
import torch.nn as nn
import numpy as np


class TrajectoryLSTM(nn.Module):
    """LSTM for detecting anomalous event sequences in ML infrastructure."""

    def __init__(
        self,
        input_dim: int = 32,       # Feature dimension per event
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) -- sequence of event features
        Returns:
            anomaly_scores: (batch, 1) -- anomaly probability
        """
        lstm_out, _ = self.lstm(x)              # (batch, seq_len, hidden)
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )                                        # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)
        return self.classifier(context)


class EventFeaturizer:
    """Convert TrajectoryEvents into numeric feature vectors."""

    # Map action types to numeric categories
    ACTION_CATEGORIES = {
        "k8s_get": 0, "k8s_list": 1, "k8s_create": 2,
        "k8s_update": 3, "k8s_delete": 4, "k8s_exec": 5,
        "k8s_attach": 6, "k8s_portforward": 7,
        "gpu_metrics_snapshot": 8, "gpu_anomaly": 9,
        "network_egress": 10, "checkpoint_create": 11,
        "weight_access": 12, "unknown": 13,
    }

    def featurize(self, event) -> np.ndarray:
        """Convert a single event to a 32-dim feature vector."""
        features = np.zeros(32, dtype=np.float32)

        # Action category (one-hot, dims 0-13)
        action = event.action.lower() if hasattr(event, 'action') else str(event.get("action", "")).lower()
        for key, idx in self.ACTION_CATEGORIES.items():
            if key in action:
                features[idx] = 1.0
                break

        # Time features (dims 14-16)
        if hasattr(event, 'timestamp'):
            ts = event.timestamp
        else:
            from datetime import datetime
            ts_str = event.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(str(ts_str))
            except (ValueError, TypeError):
                ts = datetime(2024, 1, 1, 12, 0)

        features[14] = ts.hour / 24.0
        features[15] = ts.minute / 60.0
        features[16] = 1.0 if ts.weekday() >= 5 else 0.0  # Weekend

        # Security signals (dims 17-21)
        if hasattr(event, 'details'):
            details = event.details or {}
        else:
            details = event.get("details", {}) or {}

        features[17] = 1.0 if details.get("is_weight_access") else 0.0
        features[18] = min(details.get("z_score", 0) / 10.0, 1.0)
        features[19] = 1.0 if "exec" in action or "attach" in action else 0.0
        features[20] = 1.0 if details.get("response_code", 200) >= 400 else 0.0

        step = event.trajectory_step if hasattr(event, 'trajectory_step') else event.get("step", 0)
        features[21] = min(step / 100.0, 1.0)

        # GPU metrics (dims 22-27) -- if available
        features[22] = min(details.get("DCGM_FI_DEV_GPU_UTIL", 0) / 100, 1.0)
        features[23] = min(details.get("DCGM_FI_DEV_MEM_COPY_UTIL", 0) / 100, 1.0)
        features[24] = min(details.get("DCGM_FI_DEV_FB_USED", 0) / 80000, 1.0)
        features[25] = min(details.get("DCGM_FI_DEV_GPU_TEMP", 0) / 90, 1.0)
        features[26] = min(details.get("DCGM_FI_DEV_POWER_USAGE", 0) / 700, 1.0)
        features[27] = min(details.get("DCGM_FI_DEV_ENC_UTIL", 0) / 100, 1.0)

        # Source encoding (dims 28-31)
        if hasattr(event, 'source'):
            source_val = event.source.value if hasattr(event.source, 'value') else str(event.source)
        else:
            source_val = "k8s_audit"
        source_map = {"k8s_audit": 28, "dcgm_gpu": 29, "app_event": 30, "falco_alert": 31}
        src_idx = source_map.get(source_val, 30)
        features[src_idx] = 1.0

        return features

    def featurize_trajectory(self, events: list, max_len: int = 50) -> np.ndarray:
        """Convert a list of events to a (max_len, 32) feature matrix."""
        features = []
        for event in events[:max_len]:
            features.append(self.featurize(event))

        # Pad if shorter than max_len
        while len(features) < max_len:
            features.append(np.zeros(32, dtype=np.float32))

        return np.array(features, dtype=np.float32)
