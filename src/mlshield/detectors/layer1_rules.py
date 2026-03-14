# src/mlshield/detectors/layer1_rules.py
"""Layer 1: Static Rules Engine -- microsecond-latency hard policy checks."""
import re
from ..ingestion.event_bus import TrajectoryEvent, EventSource
from ..specs.spec_types import ViolationResult
from ..specs.spec_validator import SpecValidator


class RuleEngine:
    """
    Layer 1 of the cascaded detector.

    Fast, deterministic rule checks against behavioral specifications.
    Clears ~95% of benign events. Catches hard policy violations immediately.
    """

    def __init__(self, spec_validator: SpecValidator):
        self.spec_validator = spec_validator

        # Additional hardcoded rules beyond spec validation
        self._denied_resources = {
            "secrets/aws-credentials",
            "secrets/gcp-credentials",
            "secrets/azure-credentials",
            "secrets/docker-registry",
        }

        self._suspicious_commands = [
            "convert_to_onnx",
            "torch.jit.save",
            "onnx.export",
            "curl",
            "wget",
            "nc ",
            "ncat",
            "scp ",
            "rsync",
        ]

        self._denied_egress_patterns = [
            r".*\.s3\.amazonaws\.com",
            r".*\.blob\.core\.windows\.net",
            r".*huggingface\.co",
            r".*\.ngrok\.io",
            r".*pastebin\.com",
        ]

    def check(self, event: TrajectoryEvent, spec_name: str = "standard_training") -> ViolationResult:
        """
        Run all Layer 1 checks on an event.

        Returns the most severe violation found, or a non-violation result.
        """
        checks = [
            self._check_spec_violation(event, spec_name),
            self._check_credential_access(event),
            self._check_suspicious_exec(event),
            self._check_egress_rules(event),
            self._check_unauthorized_resource_access(event),
        ]

        violations = [c for c in checks if c.is_violation]
        if violations:
            violations.sort(
                key=lambda v: {"critical": 0, "high": 1, "medium": 2, "low": 3}
                .get(v.severity, 4)
            )
            return violations[0]

        return ViolationResult(is_violation=False)

    def _check_spec_violation(self, event: TrajectoryEvent, spec_name: str) -> ViolationResult:
        """Delegate to the spec validator."""
        return self.spec_validator.validate_event(event, spec_name)

    def _check_credential_access(self, event: TrajectoryEvent) -> ViolationResult:
        """Check for unauthorized access to credential secrets."""
        if event.action not in ("k8s_get", "k8s_list", "k8s_watch"):
            return ViolationResult(is_violation=False)

        resource = event.resource.lower()
        for denied in self._denied_resources:
            if denied in resource:
                return ViolationResult(
                    is_violation=True,
                    violation_type="credential_access",
                    severity="critical",
                    description=f"Access to credential secret: {event.resource}",
                    spec_name="hardcoded_rules",
                    event=event,
                    step_number=event.trajectory_step,
                )

        # Also catch any secrets access from non-system users
        if "secrets/" in resource and event.user and "system:" not in (event.user or ""):
            return ViolationResult(
                is_violation=True,
                violation_type="secret_access",
                severity="high",
                description=f"User {event.user} accessed secret: {event.resource}",
                spec_name="hardcoded_rules",
                event=event,
                step_number=event.trajectory_step,
            )

        return ViolationResult(is_violation=False)

    def _check_suspicious_exec(self, event: TrajectoryEvent) -> ViolationResult:
        """Check for suspicious command execution in pods."""
        if event.action != "k8s_exec":
            return ViolationResult(is_violation=False)

        command = event.details.get("command", "").lower()
        for suspicious in self._suspicious_commands:
            if suspicious in command:
                return ViolationResult(
                    is_violation=True,
                    violation_type="suspicious_exec",
                    severity="high",
                    description=f"Suspicious command executed: {command}",
                    spec_name="hardcoded_rules",
                    event=event,
                    step_number=event.trajectory_step,
                )

        # exec from external IP is always suspicious
        source_ip = event.details.get("source_ip", "")
        if source_ip and not source_ip.startswith(("10.", "172.", "192.168.")):
            return ViolationResult(
                is_violation=True,
                violation_type="external_exec",
                severity="critical",
                description=f"Pod exec from external IP: {source_ip}",
                spec_name="hardcoded_rules",
                event=event,
                step_number=event.trajectory_step,
            )

        return ViolationResult(is_violation=False)

    def _check_egress_rules(self, event: TrajectoryEvent) -> ViolationResult:
        """Check network egress against denied patterns."""
        if "network" not in event.action and "egress" not in event.action:
            return ViolationResult(is_violation=False)

        destination = event.details.get("destination", "")
        if not destination:
            return ViolationResult(is_violation=False)

        for pattern in self._denied_egress_patterns:
            if re.match(pattern, destination, re.IGNORECASE):
                return ViolationResult(
                    is_violation=True,
                    violation_type="suspicious_egress",
                    severity="critical",
                    description=f"Egress to denied destination: {destination}",
                    spec_name="hardcoded_rules",
                    event=event,
                    step_number=event.trajectory_step,
                )

        # Large transfers are suspicious
        bytes_sent = event.details.get("bytes_sent", 0)
        if bytes_sent > 1_000_000_000:  # > 1GB
            return ViolationResult(
                is_violation=True,
                violation_type="large_egress",
                severity="high",
                description=f"Large outbound transfer: {bytes_sent / 1e9:.1f}GB to {destination}",
                spec_name="hardcoded_rules",
                event=event,
                step_number=event.trajectory_step,
            )

        return ViolationResult(is_violation=False)

    def _check_unauthorized_resource_access(self, event: TrajectoryEvent) -> ViolationResult:
        """Check for access to production models or restricted resources."""
        resource = event.resource.lower()
        details = event.details or {}

        # Direct check for production model access
        path = details.get("path", "").lower()
        if "/models/production" in path or "/models/production" in resource:
            return ViolationResult(
                is_violation=True,
                violation_type="weight_access_outside_pipeline",
                severity="critical",
                description=f"Access to production model: {event.resource}",
                spec_name="hardcoded_rules",
                event=event,
                step_number=event.trajectory_step,
            )

        # Models-production in resource name
        if "models-production" in resource:
            return ViolationResult(
                is_violation=True,
                violation_type="weight_access_outside_pipeline",
                severity="critical",
                description=f"Access to production models: {event.resource}",
                spec_name="hardcoded_rules",
                event=event,
                step_number=event.trajectory_step,
            )

        return ViolationResult(is_violation=False)
