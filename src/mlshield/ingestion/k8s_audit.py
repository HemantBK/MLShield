# src/mlshield/ingestion/k8s_audit.py
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional
from datetime import datetime
from .event_bus import TrajectoryEvent, EventSource

# ML-security-relevant K8s audit events
ML_RELEVANT_RESOURCES = {
    "pods", "jobs", "secrets", "configmaps",
    "persistentvolumeclaims", "services", "deployments",
    "serviceaccounts", "roles", "rolebindings",
}

# Actions that matter for weight security
SECURITY_RELEVANT_VERBS = {
    "get", "list", "watch", "create", "update",
    "patch", "delete", "exec", "attach", "portforward",
}


class K8sAuditIngester:
    """Ingests Kubernetes audit logs and normalizes to TrajectoryEvents."""

    def __init__(self, log_source: str = "/var/log/kubernetes/audit.log"):
        self.log_source = log_source
        self._checkpoint_patterns = [
            "checkpoint", "model", ".pt", ".pth", ".bin",
            ".safetensors", ".onnx", "weights",
        ]

    async def stream_events(self) -> AsyncGenerator[TrajectoryEvent, None]:
        """Stream audit events from the K8s audit log."""
        async for line in self._tail_log():
            event = self._parse_audit_event(line)
            if event and self._is_ml_relevant(event):
                yield self._to_trajectory_event(event)

    def _parse_audit_event(self, raw: str) -> Optional[dict]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _is_ml_relevant(self, event: dict) -> bool:
        """Filter to ML-security-relevant events."""
        resource = event.get("objectRef", {}).get("resource", "")
        verb = event.get("verb", "")
        return (
            resource in ML_RELEVANT_RESOURCES
            and verb in SECURITY_RELEVANT_VERBS
        )

    def _to_trajectory_event(self, audit: dict) -> TrajectoryEvent:
        """Convert K8s audit event to unified TrajectoryEvent."""
        obj_ref = audit.get("objectRef", {})
        user_info = audit.get("user", {})

        # Detect weight-related access
        resource_name = obj_ref.get("name", "")
        is_weight_access = any(
            p in resource_name.lower() for p in self._checkpoint_patterns
        )

        return TrajectoryEvent(
            event_id=audit.get("auditID", ""),
            timestamp=datetime.fromisoformat(
                audit.get("requestReceivedTimestamp", "")
                .replace("Z", "+00:00")
            ),
            source=EventSource.K8S_AUDIT,
            job_id=self._extract_job_id(audit),
            user=user_info.get("username"),
            action=f"k8s_{audit.get('verb', 'unknown')}",
            resource=f"{obj_ref.get('resource', '')}/{resource_name}",
            details={
                "namespace": obj_ref.get("namespace", ""),
                "api_group": obj_ref.get("apiGroup", ""),
                "response_code": audit.get("responseStatus", {}).get("code"),
                "source_ip": (audit.get("sourceIPs") or ["unknown"])[0],
                "is_weight_access": is_weight_access,
                "user_agent": audit.get("userAgent", ""),
            },
        )

    def _extract_job_id(self, audit: dict) -> str:
        """Extract a meaningful job identifier from the audit event."""
        obj_ref = audit.get("objectRef", {})
        ns = obj_ref.get("namespace", "default")
        name = obj_ref.get("name", "unknown")
        return f"{ns}/{name}"

    async def _tail_log(self) -> AsyncGenerator[str, None]:
        """Tail the audit log file (like tail -f)."""
        path = Path(self.log_source)
        if not path.exists():
            # Demo mode: generate synthetic events
            from benchmark.scenarios.normal_training import generate_normal_events
            for event in generate_normal_events():
                yield json.dumps(event)
                await asyncio.sleep(0.1)
            return

        with open(path) as f:
            f.seek(0, 2)  # Go to end
            while True:
                line = f.readline()
                if line:
                    yield line.strip()
                else:
                    await asyncio.sleep(0.5)
