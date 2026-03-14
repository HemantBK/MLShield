# tests/test_specs.py
"""Tests for the behavioral specification engine."""

import sys
import pytest
from datetime import datetime

sys.path.insert(0, "src")

from mlshield.ingestion.event_bus import TrajectoryEvent, EventSource
from mlshield.specs.spec_types import ViolationResult
from mlshield.specs.spec_validator import SpecValidator


@pytest.fixture
def validator():
    return SpecValidator(spec_path="configs/default_specs.yaml")


def make_event(**kwargs) -> TrajectoryEvent:
    """Helper to create test events."""
    defaults = {
        "event_id": "test",
        "timestamp": datetime(2024, 1, 1),
        "source": EventSource.K8S_AUDIT,
        "job_id": "test-job",
        "user": "admin",
        "action": "k8s_get",
        "resource": "pods/test",
        "details": {},
    }
    defaults.update(kwargs)
    return TrajectoryEvent(**defaults)


class TestSpecValidator:
    """Test spec validation logic."""

    def test_benign_event_passes(self, validator):
        event = make_event(
            action="k8s_get",
            resource="pods/data-loader",
            details={"path": "/data/training/batch_1.parquet"},
        )
        result = validator.validate_event(event)
        assert result.is_violation is False

    def test_denied_path_access_critical(self, validator):
        event = make_event(
            action="k8s_get",
            resource="/models/production/model-v3.pt",
        )
        result = validator.validate_event(event)
        assert result.is_violation is True
        assert result.severity == "critical"
        assert result.violation_type == "weight_access_outside_pipeline"

    def test_denied_secrets_path(self, validator):
        event = make_event(
            action="k8s_get",
            resource="/secrets/aws-key",
        )
        result = validator.validate_event(event)
        assert result.is_violation is True
        assert result.severity == "critical"

    def test_network_egress_to_s3(self, validator):
        event = make_event(
            action="network_egress",
            resource="pods/training-job",
            details={"destination": "attacker-bucket.s3.amazonaws.com"},
        )
        result = validator.validate_event(event)
        assert result.is_violation is True
        assert result.violation_type == "suspicious_egress"
        assert result.severity == "critical"

    def test_network_egress_to_azure(self, validator):
        event = make_event(
            action="network_egress",
            resource="pods/training-job",
            details={"destination": "storage.blob.core.windows.net"},
        )
        result = validator.validate_event(event)
        assert result.is_violation is True

    def test_allowed_network_passes(self, validator):
        event = make_event(
            action="network_egress",
            resource="pods/training-job",
            details={"destination": "api.internal.cluster"},
        )
        result = validator.validate_event(event)
        # Not in denied list, should pass
        assert result.is_violation is False

    def test_gpu_anomaly_high_z_score(self, validator):
        event = make_event(
            source=EventSource.DCGM_GPU,
            action="gpu_anomaly_dcgm_fi_dev_gpu_util",
            resource="gpu/0",
            details={"z_score": 6.0, "metric": "DCGM_FI_DEV_GPU_UTIL"},
        )
        result = validator.validate_event(event)
        assert result.is_violation is True
        assert result.violation_type == "unusual_gpu_pattern"
        assert result.severity == "high"

    def test_gpu_anomaly_medium_z_score(self, validator):
        event = make_event(
            source=EventSource.DCGM_GPU,
            action="gpu_anomaly_test",
            resource="gpu/0",
            details={"z_score": 3.5, "metric": "DCGM_FI_DEV_GPU_UTIL"},
        )
        result = validator.validate_event(event)
        assert result.is_violation is True
        assert result.severity == "medium"

    def test_gpu_normal_z_score_passes(self, validator):
        event = make_event(
            source=EventSource.DCGM_GPU,
            action="gpu_metrics_snapshot",
            resource="gpu/0",
            details={"z_score": 1.5},
        )
        result = validator.validate_event(event)
        assert result.is_violation is False

    def test_checkpoint_bad_format(self, validator):
        event = make_event(
            action="checkpoint_create",
            resource="checkpoints/model-epoch-5.onnx",
        )
        result = validator.validate_event(event)
        assert result.is_violation is True
        assert result.violation_type == "checkpoint_anomaly"

    def test_checkpoint_good_format(self, validator):
        event = make_event(
            action="checkpoint_create",
            resource="checkpoints/model-epoch-5.safetensors",
        )
        result = validator.validate_event(event)
        assert result.is_violation is False

    def test_nonexistent_spec_passes(self, validator):
        event = make_event()
        result = validator.validate_event(event, spec_name="nonexistent")
        assert result.is_violation is False

    def test_non_matching_action_passes(self, validator):
        event = make_event(
            action="health_check",
            resource="pods/health",
        )
        result = validator.validate_event(event)
        assert result.is_violation is False


class TestViolationResult:
    """Test ViolationResult dataclass."""

    def test_defaults(self):
        v = ViolationResult(is_violation=False)
        assert v.violation_type == ""
        assert v.severity == "info"
        assert v.event is None

    def test_full_violation(self):
        event = make_event()
        v = ViolationResult(
            is_violation=True,
            violation_type="suspicious_egress",
            severity="critical",
            description="Egress to attacker.com",
            spec_name="standard_training",
            event=event,
            step_number=5,
        )
        assert v.is_violation is True
        assert v.step_number == 5
        assert v.event.event_id == "test"
