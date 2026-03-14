# src/mlshield/specs/spec_validator.py
import yaml
import re
from ..ingestion.event_bus import TrajectoryEvent
from .spec_types import ViolationResult


class SpecValidator:
    """Validates events against behavioral specifications."""

    def __init__(self, spec_path: str = "configs/default_specs.yaml"):
        with open(spec_path) as f:
            self.config = yaml.safe_load(f)
        self.specs = {s["name"]: s for s in self.config.get("specs", [])}

    def validate_event(
        self,
        event: TrajectoryEvent,
        spec_name: str = "standard_training",
    ) -> ViolationResult:
        """Check a single event against its behavioral spec."""
        spec = self.specs.get(spec_name)
        if not spec:
            return ViolationResult(is_violation=False)

        # Check each violation type
        checks = [
            self._check_data_access(event, spec),
            self._check_network_egress(event, spec),
            self._check_gpu_anomaly(event, spec),
            self._check_checkpoint_behavior(event, spec),
        ]

        # Return the most severe violation found
        violations = [c for c in checks if c.is_violation]
        if violations:
            violations.sort(
                key=lambda v: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    v.severity, 4
                )
            )
            return violations[0]

        return ViolationResult(is_violation=False)

    def _check_data_access(self, event: TrajectoryEvent, spec: dict) -> ViolationResult:
        """Check if data access violates allowed patterns."""
        if "k8s_get" not in event.action and "k8s_list" not in event.action:
            return ViolationResult(is_violation=False)

        allowed = spec.get("allowed_behaviors", {}).get("data_access", {})
        denied_paths = allowed.get("denied_paths", [])

        resource = event.resource.lower()

        for pattern in denied_paths:
            if self._match_glob(resource, pattern):
                return ViolationResult(
                    is_violation=True,
                    violation_type="weight_access_outside_pipeline",
                    severity="critical",
                    description=f"Access to denied path: {event.resource}",
                    spec_name=spec["name"],
                    event=event,
                    step_number=event.trajectory_step,
                )

        return ViolationResult(is_violation=False)

    def _check_network_egress(
        self, event: TrajectoryEvent, spec: dict
    ) -> ViolationResult:
        """Check for suspicious outbound network activity."""
        if "network" not in event.action:
            return ViolationResult(is_violation=False)

        denied = (
            spec.get("allowed_behaviors", {})
            .get("network", {})
            .get("denied_egress", [])
        )
        destination = event.details.get("destination", "")

        for pattern in denied:
            if self._match_glob(destination, pattern):
                return ViolationResult(
                    is_violation=True,
                    violation_type="suspicious_egress",
                    severity="critical",
                    description=f"Egress to denied destination: {destination}",
                    spec_name=spec["name"],
                    event=event,
                    step_number=event.trajectory_step,
                )

        return ViolationResult(is_violation=False)

    def _check_gpu_anomaly(self, event: TrajectoryEvent, spec: dict) -> ViolationResult:
        """Check GPU metrics against expected profile."""
        if event.source.value != "dcgm_gpu":
            return ViolationResult(is_violation=False)

        z_score = event.details.get("z_score", 0)

        if z_score > 3.0:
            return ViolationResult(
                is_violation=True,
                violation_type="unusual_gpu_pattern",
                severity="high" if z_score > 5 else "medium",
                description=(
                    f"GPU metric {event.details.get('metric', 'unknown')} "
                    f"deviated {z_score:.1f}\u03c3 from baseline"
                ),
                spec_name=spec["name"],
                event=event,
                step_number=event.trajectory_step,
            )

        return ViolationResult(is_violation=False)

    def _check_checkpoint_behavior(
        self, event: TrajectoryEvent, spec: dict
    ) -> ViolationResult:
        """Check checkpoint creation patterns."""
        if "checkpoint" not in event.action.lower():
            return ViolationResult(is_violation=False)

        ckpt_spec = spec.get("allowed_behaviors", {}).get("checkpoints", {})
        allowed_formats = ckpt_spec.get("allowed_formats", [])

        resource = event.resource.lower()
        if allowed_formats and not any(resource.endswith(f) for f in allowed_formats):
            return ViolationResult(
                is_violation=True,
                violation_type="checkpoint_anomaly",
                severity="medium",
                description=f"Checkpoint in unexpected format: {event.resource}",
                spec_name=spec["name"],
                event=event,
                step_number=event.trajectory_step,
            )

        return ViolationResult(is_violation=False)

    @staticmethod
    def _match_glob(text: str, pattern: str) -> bool:
        """Simple glob matching (supports * wildcard)."""
        regex = pattern.replace(".", r"\.").replace("*", ".*")
        return bool(re.match(regex, text, re.IGNORECASE))
