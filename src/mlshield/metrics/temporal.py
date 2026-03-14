# src/mlshield/metrics/temporal.py
"""Temporal security metrics adapted from StepShield."""
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TemporalMetrics:
    """
    Temporal security metrics adapted from StepShield.
    Measures WHEN threats are detected, not just WHETHER.
    """
    detections: list = field(default_factory=list)
    _job_first_violation: dict = field(default_factory=dict)
    _job_first_detection: dict = field(default_factory=dict)

    def record_detection(self, result):
        """Record a detection event for temporal analysis."""
        self.detections.append(result)
        job_id = result.event.job_id
        step = result.step_detected

        # Track first actual violation vs first detection per job
        if job_id not in self._job_first_detection:
            self._job_first_detection[job_id] = step

    def record_ground_truth_violation(self, job_id: str, step: int):
        """Record when a violation actually began (for benchmark evaluation)."""
        if job_id not in self._job_first_violation:
            self._job_first_violation[job_id] = step

    def early_intervention_rate(self, max_acceptable_gap: int = 5) -> float:
        """
        ML-EIR: What fraction of threats were detected within
        `max_acceptable_gap` steps of the actual violation?
        Adapted from StepShield's EIR metric.
        """
        if not self._job_first_violation:
            return 0.0

        early_count = 0
        total = 0

        for job_id, violation_step in self._job_first_violation.items():
            if job_id in self._job_first_detection:
                total += 1
                detection_step = self._job_first_detection[job_id]
                gap = detection_step - violation_step
                if gap <= max_acceptable_gap:
                    early_count += 1

        return early_count / max(total, 1)

    def detection_gap(self) -> dict:
        """Average and median steps between violation and detection."""
        import numpy as np
        gaps = []
        for job_id, violation_step in self._job_first_violation.items():
            if job_id in self._job_first_detection:
                gaps.append(self._job_first_detection[job_id] - violation_step)

        if not gaps:
            return {"mean": 0, "median": 0, "max": 0}
        return {
            "mean": float(np.mean(gaps)),
            "median": float(np.median(gaps)),
            "max": int(np.max(gaps)),
        }

    def damage_prevented(self) -> float:
        """
        Estimate % of potential damage prevented by early detection.
        Assumes linear damage accumulation -- earlier detection = more prevented.
        """
        if not self._job_first_violation:
            return 0.0

        prevented_scores = []
        max_trajectory_length = 100  # Assume 100-step trajectories

        for job_id, violation_step in self._job_first_violation.items():
            if job_id in self._job_first_detection:
                detection_step = self._job_first_detection[job_id]
                remaining = max_trajectory_length - detection_step
                total_damage_window = max_trajectory_length - violation_step
                if total_damage_window > 0:
                    prevented_scores.append(remaining / total_damage_window)

        return sum(prevented_scores) / max(len(prevented_scores), 1)

    def summary(self) -> dict:
        """Full temporal metrics summary."""
        return {
            "total_detections": len(self.detections),
            "early_intervention_rate": self.early_intervention_rate(),
            "detection_gap": self.detection_gap(),
            "damage_prevented_pct": self.damage_prevented() * 100,
            "detections_by_layer": self._count_by_layer(),
        }

    def _count_by_layer(self) -> dict:
        counts = defaultdict(int)
        for d in self.detections:
            counts[f"layer_{d.detected_by_layer}"] += 1
        return dict(counts)
