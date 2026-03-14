# tests/test_layer2.py
"""Tests for Layer 2 ML detectors and temporal metrics."""
import sys
import pytest
import numpy as np
import torch
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, "src")

from mlshield.ingestion.event_bus import TrajectoryEvent, EventSource
from mlshield.detectors.models.lstm_detector import TrajectoryLSTM, EventFeaturizer
from mlshield.detectors.models.isolation import GPUIsolationForest
from mlshield.metrics.temporal import TemporalMetrics


def make_event(**kwargs) -> TrajectoryEvent:
    defaults = {
        "event_id": "test",
        "timestamp": datetime(2024, 1, 15, 14, 30),
        "source": EventSource.K8S_AUDIT,
        "job_id": "test-job",
        "user": "admin",
        "action": "k8s_get",
        "resource": "pods/test",
        "details": {},
        "trajectory_step": 0,
    }
    defaults.update(kwargs)
    return TrajectoryEvent(**defaults)


class TestEventFeaturizer:
    """Test event featurization."""

    def test_output_shape(self):
        feat = EventFeaturizer()
        event = make_event()
        result = feat.featurize(event)
        assert result.shape == (32,)
        assert result.dtype == np.float32

    def test_action_encoding(self):
        feat = EventFeaturizer()

        event_get = make_event(action="k8s_get")
        result = feat.featurize(event_get)
        assert result[0] == 1.0  # k8s_get is index 0

        event_exec = make_event(action="k8s_exec")
        result = feat.featurize(event_exec)
        assert result[5] == 1.0  # k8s_exec is index 5

    def test_time_features(self):
        feat = EventFeaturizer()
        event = make_event(timestamp=datetime(2024, 1, 15, 14, 30))
        result = feat.featurize(event)
        assert abs(result[14] - 14 / 24.0) < 0.01
        assert abs(result[15] - 30 / 60.0) < 0.01
        assert result[16] == 0.0  # Monday is not weekend

    def test_weekend_flag(self):
        feat = EventFeaturizer()
        event = make_event(timestamp=datetime(2024, 1, 13, 10, 0))  # Saturday
        result = feat.featurize(event)
        assert result[16] == 1.0

    def test_security_signals(self):
        feat = EventFeaturizer()
        event = make_event(
            action="k8s_exec",
            details={"is_weight_access": True, "z_score": 5.0},
        )
        result = feat.featurize(event)
        assert result[17] == 1.0  # weight access
        assert result[18] == 0.5  # z_score 5.0 / 10.0
        assert result[19] == 1.0  # exec action

    def test_gpu_metrics(self):
        feat = EventFeaturizer()
        event = make_event(
            source=EventSource.DCGM_GPU,
            action="gpu_metrics_snapshot",
            details={
                "DCGM_FI_DEV_GPU_UTIL": 85.0,
                "DCGM_FI_DEV_MEM_COPY_UTIL": 70.0,
                "DCGM_FI_DEV_FB_USED": 60000,
                "DCGM_FI_DEV_GPU_TEMP": 65.0,
            },
        )
        result = feat.featurize(event)
        assert abs(result[22] - 0.85) < 0.01
        assert abs(result[23] - 0.70) < 0.01

    def test_source_encoding(self):
        feat = EventFeaturizer()

        event_k8s = make_event(source=EventSource.K8S_AUDIT)
        result = feat.featurize(event_k8s)
        assert result[28] == 1.0

        event_gpu = make_event(source=EventSource.DCGM_GPU)
        result = feat.featurize(event_gpu)
        assert result[29] == 1.0

    def test_trajectory_featurization(self):
        feat = EventFeaturizer()
        events = [
            make_event(action="k8s_get", trajectory_step=i)
            for i in range(10)
        ]
        result = feat.featurize_trajectory(events, max_len=50)
        assert result.shape == (50, 32)
        # Padding should be zeros
        assert np.all(result[10:] == 0)

    def test_trajectory_truncation(self):
        feat = EventFeaturizer()
        events = [make_event(trajectory_step=i) for i in range(100)]
        result = feat.featurize_trajectory(events, max_len=50)
        assert result.shape == (50, 32)

    def test_dict_events(self):
        """Test featurizing dict-based events (from benchmark)."""
        feat = EventFeaturizer()
        event = {
            "action": "k8s_exec",
            "timestamp": "2024-01-15T14:30:00",
            "details": {"is_weight_access": True},
            "step": 5,
        }
        result = feat.featurize(event)
        assert result.shape == (32,)
        assert result[5] == 1.0  # exec


class TestTrajectoryLSTM:
    """Test LSTM model architecture."""

    def test_forward_shape(self):
        model = TrajectoryLSTM(input_dim=32, hidden_dim=64)
        x = torch.randn(4, 50, 32)  # batch=4, seq_len=50, features=32
        output = model(x)
        assert output.shape == (4, 1)

    def test_output_range(self):
        model = TrajectoryLSTM()
        x = torch.randn(8, 50, 32)
        output = model(x)
        # Sigmoid output should be in [0, 1]
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_single_sample(self):
        model = TrajectoryLSTM()
        x = torch.randn(1, 50, 32)
        output = model(x)
        assert output.shape == (1, 1)

    def test_short_sequence(self):
        model = TrajectoryLSTM()
        x = torch.randn(1, 5, 32)  # Short sequence
        output = model(x)
        assert output.shape == (1, 1)

    def test_load_trained_model(self):
        """Test loading the trained model if it exists."""
        from pathlib import Path
        model_path = "benchmark/data/models/lstm_detector.pt"
        if Path(model_path).exists():
            model = TrajectoryLSTM()
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.eval()
            x = torch.randn(1, 50, 32)
            output = model(x)
            assert output.shape == (1, 1)


class TestGPUIsolationForest:
    """Test Isolation Forest wrapper."""

    def test_extract_features(self):
        iso = GPUIsolationForest()
        details = {
            "DCGM_FI_DEV_GPU_UTIL": 85.0,
            "DCGM_FI_DEV_MEM_COPY_UTIL": 70.0,
            "DCGM_FI_DEV_FB_USED": 60000,
            "DCGM_FI_DEV_GPU_TEMP": 65.0,
            "DCGM_FI_DEV_POWER_USAGE": 300,
            "DCGM_FI_DEV_ENC_UTIL": 10,
        }
        feat = iso.extract_gpu_features(details)
        assert feat is not None
        assert feat.shape == (6,)

    def test_extract_missing_gpu_util(self):
        iso = GPUIsolationForest()
        details = {"some_other_metric": 42}
        feat = iso.extract_gpu_features(details)
        assert feat is None

    def test_extract_partial_features(self):
        iso = GPUIsolationForest()
        details = {"DCGM_FI_DEV_GPU_UTIL": 85.0}
        feat = iso.extract_gpu_features(details)
        assert feat is not None  # Should work with partial data
        assert feat[0] == 85.0
        assert feat[1] == 0.0  # Missing = 0

    def test_unfitted_score(self):
        iso = GPUIsolationForest()
        assert iso.score({"DCGM_FI_DEV_GPU_UTIL": 85}) == 0.0

    def test_fit_and_score(self):
        iso = GPUIsolationForest()
        normal_events = [
            {
                "DCGM_FI_DEV_GPU_UTIL": np.random.normal(85, 5),
                "DCGM_FI_DEV_MEM_COPY_UTIL": np.random.normal(70, 8),
                "DCGM_FI_DEV_FB_USED": np.random.normal(60000, 5000),
                "DCGM_FI_DEV_GPU_TEMP": np.random.normal(65, 3),
                "DCGM_FI_DEV_POWER_USAGE": np.random.normal(300, 30),
                "DCGM_FI_DEV_ENC_UTIL": np.random.normal(10, 5),
            }
            for _ in range(100)
        ]
        iso.fit(normal_events)
        assert iso.is_fitted is True

        # Normal event should have low anomaly score
        normal_score = iso.score(normal_events[0])
        assert 0 <= normal_score <= 1

        # Anomalous event should have higher score
        anomalous = {
            "DCGM_FI_DEV_GPU_UTIL": 99.9,
            "DCGM_FI_DEV_MEM_COPY_UTIL": 99.0,
            "DCGM_FI_DEV_FB_USED": 79000,
            "DCGM_FI_DEV_GPU_TEMP": 90.0,
            "DCGM_FI_DEV_POWER_USAGE": 600,
            "DCGM_FI_DEV_ENC_UTIL": 95,
        }
        anomalous_score = iso.score(anomalous)
        assert anomalous_score > normal_score

    def test_save_and_load(self, tmp_path):
        iso = GPUIsolationForest()
        normal_events = [
            {"DCGM_FI_DEV_GPU_UTIL": np.random.normal(85, 5)}
            for _ in range(50)
        ]
        iso.fit(normal_events)

        save_path = str(tmp_path / "test_iso.pkl")
        iso.save(save_path)

        iso2 = GPUIsolationForest()
        iso2.load(save_path)
        assert iso2.is_fitted is True


class TestTemporalMetrics:
    """Test temporal security metrics."""

    @dataclass
    class MockDetectionResult:
        event: TrajectoryEvent
        detected_by_layer: int
        step_detected: int

    def test_empty_metrics(self):
        metrics = TemporalMetrics()
        summary = metrics.summary()
        assert summary["total_detections"] == 0
        assert summary["early_intervention_rate"] == 0.0

    def test_eir_perfect(self):
        metrics = TemporalMetrics()
        event = make_event(job_id="job-1")

        metrics.record_ground_truth_violation("job-1", step=10)
        result = self.MockDetectionResult(event=event, detected_by_layer=1, step_detected=10)
        metrics.record_detection(result)

        eir = metrics.early_intervention_rate(max_acceptable_gap=5)
        assert eir == 1.0

    def test_eir_late_detection(self):
        metrics = TemporalMetrics()
        event = make_event(job_id="job-1")

        metrics.record_ground_truth_violation("job-1", step=10)
        result = self.MockDetectionResult(event=event, detected_by_layer=2, step_detected=20)
        metrics.record_detection(result)

        eir = metrics.early_intervention_rate(max_acceptable_gap=5)
        assert eir == 0.0  # Gap of 10 > max_acceptable_gap of 5

    def test_detection_gap(self):
        metrics = TemporalMetrics()

        for i in range(5):
            event = make_event(job_id=f"job-{i}")
            metrics.record_ground_truth_violation(f"job-{i}", step=10)
            result = self.MockDetectionResult(
                event=event, detected_by_layer=1, step_detected=10 + i
            )
            metrics.record_detection(result)

        gap = metrics.detection_gap()
        assert gap["mean"] == 2.0  # avg(0,1,2,3,4) = 2
        assert gap["median"] == 2.0
        assert gap["max"] == 4

    def test_damage_prevented(self):
        metrics = TemporalMetrics()
        event = make_event(job_id="job-1")

        metrics.record_ground_truth_violation("job-1", step=10)
        result = self.MockDetectionResult(event=event, detected_by_layer=1, step_detected=10)
        metrics.record_detection(result)

        prevented = metrics.damage_prevented()
        assert prevented == 1.0  # Detected immediately = 100% prevented

    def test_damage_prevented_late(self):
        metrics = TemporalMetrics()
        event = make_event(job_id="job-1")

        metrics.record_ground_truth_violation("job-1", step=10)
        result = self.MockDetectionResult(event=event, detected_by_layer=2, step_detected=50)
        metrics.record_detection(result)

        prevented = metrics.damage_prevented()
        # remaining=50, total_window=90 -> 50/90 ≈ 0.556
        assert 0.5 < prevented < 0.6

    def test_summary(self):
        metrics = TemporalMetrics()
        event = make_event(job_id="job-1")

        metrics.record_ground_truth_violation("job-1", step=10)
        result = self.MockDetectionResult(event=event, detected_by_layer=1, step_detected=12)
        metrics.record_detection(result)

        summary = metrics.summary()
        assert summary["total_detections"] == 1
        assert summary["early_intervention_rate"] == 1.0
        assert summary["detection_gap"]["mean"] == 2.0
        assert summary["damage_prevented_pct"] > 0
        assert "layer_1" in summary["detections_by_layer"]

    def test_count_by_layer(self):
        metrics = TemporalMetrics()
        for layer in [1, 1, 1, 2, 3]:
            event = make_event(job_id=f"job-{layer}")
            result = self.MockDetectionResult(event=event, detected_by_layer=layer, step_detected=10)
            metrics.record_detection(result)

        summary = metrics.summary()
        assert summary["detections_by_layer"]["layer_1"] == 3
        assert summary["detections_by_layer"]["layer_2"] == 1
        assert summary["detections_by_layer"]["layer_3"] == 1
