# src/mlshield/detectors/layer2_ml.py
"""Layer 2: ML Anomaly Detector -- LSTM + Isolation Forest."""
import torch
import numpy as np
from typing import Optional
from pathlib import Path

from ..ingestion.event_bus import TrajectoryEvent
from .models.lstm_detector import TrajectoryLSTM, EventFeaturizer
from .models.isolation import GPUIsolationForest


class MLDetector:
    """
    Layer 2 of the cascaded detector.

    Combines LSTM sequence model (for trajectory-level anomaly detection)
    with Isolation Forest (for GPU telemetry anomalies).

    Only processes events escalated from Layer 1 (~5% of total).
    """

    def __init__(
        self,
        lstm_model_path: Optional[str] = None,
        isolation_model_path: Optional[str] = None,
        lstm_weight: float = 0.7,
        isolation_weight: float = 0.3,
        sequence_length: int = 50,
    ):
        self.featurizer = EventFeaturizer()
        self.lstm_weight = lstm_weight
        self.isolation_weight = isolation_weight
        self.sequence_length = sequence_length

        # LSTM model
        self.lstm = TrajectoryLSTM()
        self.lstm.eval()
        self._lstm_loaded = False
        if lstm_model_path and Path(lstm_model_path).exists():
            self.lstm.load_state_dict(torch.load(lstm_model_path, map_location="cpu", weights_only=True))
            self._lstm_loaded = True

        # Isolation Forest
        self.isolation_forest = GPUIsolationForest()
        if isolation_model_path and Path(isolation_model_path).exists():
            self.isolation_forest.load(isolation_model_path)

        # Event buffer per job for LSTM sequence scoring
        self._event_buffers: dict[str, list[np.ndarray]] = {}

    async def score(self, event: TrajectoryEvent) -> float:
        """
        Score an event for anomalousness.

        Returns a score in [0, 1] where higher = more anomalous.
        Combines LSTM trajectory score and Isolation Forest GPU score.
        """
        scores = []
        weights = []

        # LSTM sequence score
        lstm_score = self._score_lstm(event)
        if lstm_score is not None:
            scores.append(lstm_score)
            weights.append(self.lstm_weight)

        # Isolation Forest GPU score
        iso_score = self._score_isolation(event)
        if iso_score is not None:
            scores.append(iso_score)
            weights.append(self.isolation_weight)

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        combined = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return float(combined)

    def _score_lstm(self, event: TrajectoryEvent) -> Optional[float]:
        """Score using LSTM on the event's trajectory buffer."""
        if not self._lstm_loaded:
            return None

        job_id = event.job_id

        # Add featurized event to buffer
        feat = self.featurizer.featurize(event)
        if job_id not in self._event_buffers:
            self._event_buffers[job_id] = []
        self._event_buffers[job_id].append(feat)

        # Keep buffer at sequence_length
        if len(self._event_buffers[job_id]) > self.sequence_length:
            self._event_buffers[job_id] = self._event_buffers[job_id][-self.sequence_length:]

        # Need at least a few events for meaningful scoring
        if len(self._event_buffers[job_id]) < 5:
            return None

        # Prepare input tensor
        buffer = self._event_buffers[job_id]
        # Pad to sequence_length
        padded = buffer + [np.zeros(32, dtype=np.float32)] * (self.sequence_length - len(buffer))
        x = torch.tensor(np.array([padded]), dtype=torch.float32)

        with torch.no_grad():
            score = self.lstm(x).item()

        return score

    def _score_isolation(self, event: TrajectoryEvent) -> Optional[float]:
        """Score GPU metrics using Isolation Forest."""
        if not self.isolation_forest.is_fitted:
            return None

        if event.source.value != "dcgm_gpu":
            return None

        return self.isolation_forest.score(event.details)

    def clear_buffer(self, job_id: str):
        """Clear the event buffer for a completed job."""
        self._event_buffers.pop(job_id, None)

    def load_lstm(self, path: str):
        """Load a trained LSTM model."""
        self.lstm.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.lstm.eval()
        self._lstm_loaded = True

    def load_isolation_forest(self, path: str):
        """Load a trained Isolation Forest model."""
        self.isolation_forest.load(path)
