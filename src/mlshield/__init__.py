"""MLShield: ML-Infrastructure-Aware Anomaly Detection for Weight Exfiltration."""

__version__ = "0.1.0"

from .ingestion.event_bus import TrajectoryEvent, EventSource
from .detectors.cascade import CascadedDetector, DetectionResult
from .utils.config import MLShieldConfig, load_config

__all__ = [
    "TrajectoryEvent",
    "EventSource",
    "CascadedDetector",
    "DetectionResult",
    "MLShieldConfig",
    "load_config",
]
