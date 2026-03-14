# src/mlshield/ingestion/event_bus.py
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional


class EventSource(Enum):
    K8S_AUDIT = "k8s_audit"
    DCGM_GPU = "dcgm_gpu"
    APP_EVENT = "app_event"
    FALCO_ALERT = "falco_alert"


class EventSeverity(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TrajectoryEvent:
    """Unified event format -- all data sources normalize to this."""

    event_id: str
    timestamp: datetime
    source: EventSource
    job_id: str  # K8s job/pod identifier
    user: Optional[str]  # User or service account
    action: str  # What happened (e.g., "read_checkpoint", "gpu_util_spike")
    resource: str  # What was acted on (e.g., "model-v3.pt", "gpu-0")
    details: dict = field(default_factory=dict)  # Source-specific metadata
    trajectory_step: int = 0  # Step number within this job's trajectory


@dataclass
class Trajectory:
    """Ordered sequence of events for a single ML job."""

    job_id: str
    events: list[TrajectoryEvent] = field(default_factory=list)
    start_time: Optional[datetime] = None
    job_type: str = "unknown"  # training, inference, data_pipeline
    spec_name: Optional[str] = None  # Which behavioral spec applies

    def add_event(self, event: TrajectoryEvent):
        event.trajectory_step = len(self.events)
        self.events.append(event)
        if not self.start_time:
            self.start_time = event.timestamp
