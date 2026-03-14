# src/mlshield/specs/spec_types.py
from dataclasses import dataclass
from typing import Optional
from ..ingestion.event_bus import TrajectoryEvent


@dataclass
class ViolationResult:
    """Result of checking an event against a spec."""
    is_violation: bool
    violation_type: str = ""
    severity: str = "info"
    description: str = ""
    spec_name: str = ""
    event: Optional[TrajectoryEvent] = None
    step_number: int = 0
