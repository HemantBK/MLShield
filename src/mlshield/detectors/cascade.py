# src/mlshield/detectors/cascade.py
"""3-layer cascaded detector adapted from StepShield's HybridGuard."""

import time
from dataclasses import dataclass

from ..ingestion.event_bus import TrajectoryEvent
from ..specs.spec_validator import SpecValidator
from .layer1_rules import RuleEngine
from .layer2_ml import MLDetector
from .layer3_llm import LLMJudge
from ..metrics.temporal import TemporalMetrics


@dataclass
class DetectionResult:
    """Final output of the cascaded detector."""

    event: TrajectoryEvent
    is_threat: bool
    confidence: float  # 0.0 - 1.0
    threat_type: str
    severity: str
    description: str
    detected_by_layer: int  # 1, 2, or 3
    detection_latency_ms: float
    step_detected: int  # Which trajectory step
    explanation: str = ""  # Natural language (from Layer 3)


class CascadedDetector:
    """
    3-layer cascaded detector adapted from StepShield's HybridGuard.

    Layer 1 (Static Rules): Microsecond-latency hard policy checks.
                            Clears ~95% of benign events.
    Layer 2 (ML Model):     Millisecond-latency anomaly detection on
                            escalated events using LSTM + Isolation Forest.
    Layer 3 (LLM Judge):    Second-latency semantic analysis for the most
                            complex cases (<1% of events).
    """

    def __init__(
        self,
        spec_validator: SpecValidator,
        ml_detector: MLDetector,
        llm_judge: LLMJudge,
        layer2_threshold: float = 0.6,
        layer3_threshold: float = 0.8,
    ):
        self.layer1 = RuleEngine(spec_validator)
        self.layer2 = ml_detector
        self.layer3 = llm_judge
        self.layer2_threshold = layer2_threshold
        self.layer3_threshold = layer3_threshold
        self.temporal_metrics = TemporalMetrics()

        # Counters for cascade efficiency
        self.total_events = 0
        self.layer1_cleared = 0
        self.layer2_processed = 0
        self.layer3_processed = 0

    async def evaluate(self, event: TrajectoryEvent) -> DetectionResult:
        """Run event through the cascade."""
        self.total_events += 1
        start = time.perf_counter()

        # ---- Layer 1: Static Rules ----
        l1_result = self.layer1.check(event)
        elapsed = (time.perf_counter() - start) * 1000

        if l1_result.is_violation and l1_result.severity == "critical":
            # Hard violation -- immediate alert, skip other layers
            result = DetectionResult(
                event=event,
                is_threat=True,
                confidence=0.95,
                threat_type=l1_result.violation_type,
                severity=l1_result.severity,
                description=l1_result.description,
                detected_by_layer=1,
                detection_latency_ms=elapsed,
                step_detected=event.trajectory_step,
            )
            self.temporal_metrics.record_detection(result)
            return result

        if not l1_result.is_violation:
            # Benign -- check if rules are confident enough to clear
            self.layer1_cleared += 1
            return DetectionResult(
                event=event,
                is_threat=False,
                confidence=0.0,
                threat_type="",
                severity="info",
                description="Cleared by Layer 1",
                detected_by_layer=1,
                detection_latency_ms=elapsed,
                step_detected=event.trajectory_step,
            )

        # ---- Layer 2: ML Anomaly Detection ----
        self.layer2_processed += 1
        l2_score = await self.layer2.score(event)
        elapsed = (time.perf_counter() - start) * 1000

        if l2_score < self.layer2_threshold:
            return DetectionResult(
                event=event,
                is_threat=False,
                confidence=l2_score,
                threat_type="",
                severity="info",
                description=f"Cleared by Layer 2 (score={l2_score:.3f})",
                detected_by_layer=2,
                detection_latency_ms=elapsed,
                step_detected=event.trajectory_step,
            )

        if l2_score >= self.layer3_threshold:
            # High enough confidence to alert without LLM
            result = DetectionResult(
                event=event,
                is_threat=True,
                confidence=l2_score,
                threat_type=l1_result.violation_type or "ml_anomaly",
                severity="high",
                description=f"ML detector flagged (score={l2_score:.3f})",
                detected_by_layer=2,
                detection_latency_ms=elapsed,
                step_detected=event.trajectory_step,
            )
            self.temporal_metrics.record_detection(result)
            return result

        # ---- Layer 3: LLM Semantic Judge ----
        self.layer3_processed += 1
        l3_result = await self.layer3.judge(event, l1_result, l2_score)
        elapsed = (time.perf_counter() - start) * 1000

        result = DetectionResult(
            event=event,
            is_threat=l3_result["is_threat"],
            confidence=l3_result["confidence"],
            threat_type=l3_result.get("threat_type", "complex_anomaly"),
            severity=l3_result.get("severity", "medium"),
            description=l3_result.get("description", ""),
            detected_by_layer=3,
            detection_latency_ms=elapsed,
            step_detected=event.trajectory_step,
            explanation=l3_result.get("explanation", ""),
        )

        if result.is_threat:
            self.temporal_metrics.record_detection(result)
        return result

    def get_cascade_stats(self) -> dict:
        """Return cascade efficiency metrics."""
        return {
            "total_events": self.total_events,
            "layer1_cleared_pct": (
                self.layer1_cleared / max(self.total_events, 1) * 100
            ),
            "layer2_processed_pct": (
                self.layer2_processed / max(self.total_events, 1) * 100
            ),
            "layer3_processed_pct": (
                self.layer3_processed / max(self.total_events, 1) * 100
            ),
        }
