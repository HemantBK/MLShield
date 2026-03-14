# src/mlshield/ingestion/base.py
"""Abstract ingestion interface for MLShield data sources."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from .event_bus import TrajectoryEvent


class BaseIngester(ABC):
    """Abstract base class for all data source ingesters."""

    @abstractmethod
    async def stream_events(self) -> AsyncGenerator[TrajectoryEvent, None]:
        """Stream events from the data source as TrajectoryEvents."""
        yield  # pragma: no cover

    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this data source."""
        ...
