# tests/test_cascade.py
"""Integration tests for the full cascaded detector."""
import sys
import pytest
from datetime import datetime

sys.path.insert(0, "src")

from mlshield.ingestion.event_bus import TrajectoryEvent, EventSource
from mlshield.specs.spec_validator import SpecValidator
from mlshield.specs.spec_types import ViolationResult
from mlshield.detectors.layer2_ml import MLDetector
from mlshield.detectors.layer3_llm import LLMJudge
from mlshield.detectors.cascade import CascadedDetector, DetectionResult


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


@pytest.fixture
def cascade():
    validator = SpecValidator(spec_path="configs/default_specs.yaml")
    ml_detector = MLDetector()  # No trained model -- will return 0.0
    llm_judge = LLMJudge()      # No API key -- will use fallback
    return CascadedDetector(
        spec_validator=validator,
        ml_detector=ml_detector,
        llm_judge=llm_judge,
    )


@pytest.fixture
def cascade_with_models():
    """Cascade with trained models loaded."""
    from pathlib import Path
    lstm_path = "benchmark/data/models/lstm_detector.pt"
    iso_path = "benchmark/data/models/isolation_forest.pkl"

    if not Path(lstm_path).exists():
        pytest.skip("Trained models not found")

    validator = SpecValidator(spec_path="configs/default_specs.yaml")
    ml_detector = MLDetector(
        lstm_model_path=lstm_path,
        isolation_model_path=iso_path,
    )
    llm_judge = LLMJudge()
    return CascadedDetector(
        spec_validator=validator,
        ml_detector=ml_detector,
        llm_judge=llm_judge,
    )


class TestCascadeLayer1:
    """Test that Layer 1 handles clear cases directly."""

    @pytest.mark.asyncio
    async def test_benign_cleared_by_layer1(self, cascade):
        event = make_event(
            action="k8s_get",
            resource="pods/health",
        )
        result = await cascade.evaluate(event)
        assert result.is_threat is False
        assert result.detected_by_layer == 1
        assert "Cleared by Layer 1" in result.description

    @pytest.mark.asyncio
    async def test_critical_violation_caught_by_layer1(self, cascade):
        event = make_event(
            action="k8s_get",
            resource="secrets/aws-credentials",
        )
        result = await cascade.evaluate(event)
        assert result.is_threat is True
        assert result.detected_by_layer == 1
        assert result.severity == "critical"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_egress_caught_by_layer1(self, cascade):
        event = make_event(
            action="network_egress",
            resource="pods/training-job",
            details={"destination": "evil.s3.amazonaws.com"},
        )
        result = await cascade.evaluate(event)
        assert result.is_threat is True
        assert result.detected_by_layer == 1
        assert result.threat_type == "suspicious_egress"

    @pytest.mark.asyncio
    async def test_production_model_caught(self, cascade):
        event = make_event(
            action="k8s_list",
            resource="persistentvolumeclaims/models-production",
            details={"is_weight_access": True},
        )
        result = await cascade.evaluate(event)
        assert result.is_threat is True
        assert result.detected_by_layer == 1


class TestCascadeStats:
    """Test cascade statistics tracking."""

    @pytest.mark.asyncio
    async def test_counters_increment(self, cascade):
        # Process a benign event
        event = make_event(action="k8s_get", resource="pods/health")
        await cascade.evaluate(event)

        assert cascade.total_events == 1
        assert cascade.layer1_cleared == 1

    @pytest.mark.asyncio
    async def test_get_cascade_stats(self, cascade):
        # Process some events
        for _ in range(10):
            event = make_event(action="k8s_get", resource="pods/health")
            await cascade.evaluate(event)

        stats = cascade.get_cascade_stats()
        assert stats["total_events"] == 10
        assert stats["layer1_cleared_pct"] == 100.0

    @pytest.mark.asyncio
    async def test_temporal_metrics_recorded(self, cascade):
        event = make_event(
            action="network_egress",
            resource="pods/training-job",
            details={"destination": "evil.s3.amazonaws.com"},
        )
        result = await cascade.evaluate(event)
        assert result.is_threat is True
        assert len(cascade.temporal_metrics.detections) == 1


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_creation(self):
        event = make_event()
        result = DetectionResult(
            event=event,
            is_threat=True,
            confidence=0.95,
            threat_type="suspicious_egress",
            severity="critical",
            description="Test detection",
            detected_by_layer=1,
            detection_latency_ms=0.015,
            step_detected=5,
        )
        assert result.is_threat is True
        assert result.explanation == ""

    def test_with_explanation(self):
        event = make_event()
        result = DetectionResult(
            event=event,
            is_threat=True,
            confidence=0.85,
            threat_type="weight_exfiltration",
            severity="critical",
            description="Weight theft detected",
            detected_by_layer=3,
            detection_latency_ms=1500.0,
            step_detected=25,
            explanation="The event sequence shows classic weight exfiltration pattern.",
        )
        assert result.detected_by_layer == 3
        assert "exfiltration" in result.explanation


class TestLLMJudgeFallback:
    """Test LLM Judge fallback behavior (no API key)."""

    @pytest.mark.asyncio
    async def test_fallback_benign(self):
        judge = LLMJudge()
        event = make_event(action="k8s_get", resource="pods/health")
        l1_result = ViolationResult(is_violation=False)
        result = await judge.judge(event, l1_result, l2_score=0.3)
        assert result["is_threat"] is False

    @pytest.mark.asyncio
    async def test_fallback_threat_high_score(self):
        judge = LLMJudge()
        event = make_event(
            action="network_egress",
            resource="pods/job",
            details={"destination": "attacker.com"},
        )
        l1_result = ViolationResult(
            is_violation=True,
            violation_type="suspicious_egress",
            severity="high",
        )
        result = await judge.judge(event, l1_result, l2_score=0.8)
        assert result["is_threat"] is True
        assert result["severity"] in ("critical", "high")
        assert result["threat_type"] == "weight_exfiltration"

    @pytest.mark.asyncio
    async def test_fallback_weight_access(self):
        judge = LLMJudge()
        event = make_event(
            action="k8s_get",
            resource="pvc/model.safetensors",
            details={"is_weight_access": True},
        )
        l1_result = ViolationResult(
            is_violation=True,
            violation_type="weight_access",
            severity="critical",
        )
        result = await judge.judge(event, l1_result, l2_score=0.75)
        assert result["is_threat"] is True
        assert "weight" in result["description"].lower()

    @pytest.mark.asyncio
    async def test_fallback_gpu_anomaly(self):
        judge = LLMJudge()
        event = make_event(
            source=EventSource.DCGM_GPU,
            action="gpu_anomaly",
            resource="gpu/0",
            details={"z_score": 6.5, "metric": "DCGM_FI_DEV_GPU_UTIL"},
        )
        l1_result = ViolationResult(
            is_violation=True,
            violation_type="unusual_gpu_pattern",
            severity="high",
        )
        result = await judge.judge(event, l1_result, l2_score=0.72)
        assert result["is_threat"] is True

    @pytest.mark.asyncio
    async def test_prompt_building(self):
        judge = LLMJudge()
        event = make_event(action="k8s_exec", details={"command": "ls"})
        l1_result = ViolationResult(is_violation=False)
        prompt = judge._build_prompt(event, l1_result, 0.5)
        assert "security analyst" in prompt
        assert "k8s_exec" in prompt
        assert "JSON object" in prompt


class TestCascadeWithModels:
    """Integration test with trained models."""

    @pytest.mark.asyncio
    async def test_benign_trajectory(self, cascade_with_models):
        """A full benign trajectory should not trigger alerts."""
        cascade = cascade_with_models
        alerts = 0
        for i in range(20):
            event = make_event(
                event_id=f"benign-{i}",
                job_id="normal-job",
                action="k8s_get",
                resource="pods/data-loader",
                details={"path": f"/data/training/batch_{i}.parquet"},
                trajectory_step=i,
            )
            result = await cascade.evaluate(event)
            if result.is_threat:
                alerts += 1
        assert alerts == 0

    @pytest.mark.asyncio
    async def test_attack_trajectory_detected(self, cascade_with_models):
        """An attack trajectory should trigger at least one alert."""
        cascade = cascade_with_models
        alerts = 0

        # Normal events first
        for i in range(15):
            event = make_event(
                event_id=f"normal-{i}",
                job_id="attack-job",
                action="k8s_get",
                resource="pods/data-loader",
                trajectory_step=i,
            )
            await cascade.evaluate(event)

        # Then attack events
        for i in range(15, 25):
            event = make_event(
                event_id=f"attack-{i}",
                job_id="attack-job",
                action="network_egress",
                resource="pods/training-job",
                details={"destination": "evil.s3.amazonaws.com", "bytes_sent": 5_000_000_000},
                trajectory_step=i,
            )
            result = await cascade.evaluate(event)
            if result.is_threat:
                alerts += 1

        assert alerts > 0
