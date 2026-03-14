# src/mlshield/ingestion/app_events.py
"""Application-level event ingester for MLShield."""
import json
import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
from .event_bus import TrajectoryEvent, EventSource


class AppEventIngester:
    """Ingests application-level events (checkpoint saves, data access, etc.)."""

    def __init__(self, redis_url: Optional[str] = None, channel: str = "mlshield:app_events"):
        self.redis_url = redis_url
        self.channel = channel
        self._redis = None

    async def stream_events(self) -> AsyncGenerator[TrajectoryEvent, None]:
        """Stream application events from Redis pub/sub or direct input."""
        if self.redis_url:
            async for event in self._stream_from_redis():
                yield event
        else:
            # Fallback: no events in standalone mode
            while True:
                await asyncio.sleep(10)

    async def _stream_from_redis(self) -> AsyncGenerator[TrajectoryEvent, None]:
        """Stream events from Redis Streams."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self.redis_url)
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(self.channel)

            async for message in pubsub.listen():
                if message["type"] == "message":
                    event = self._parse_app_event(message["data"])
                    if event:
                        yield event
        except Exception:
            # Redis not available, wait and retry
            await asyncio.sleep(5)

    def _parse_app_event(self, raw: bytes | str) -> Optional[TrajectoryEvent]:
        """Parse an application event into a TrajectoryEvent."""
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)

            return TrajectoryEvent(
                event_id=data.get("event_id", f"app-{datetime.now(timezone.utc).timestamp()}"),
                timestamp=datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(timezone.utc),
                source=EventSource.APP_EVENT,
                job_id=data.get("job_id", "unknown"),
                user=data.get("user"),
                action=data.get("action", "app_event"),
                resource=data.get("resource", ""),
                details=data.get("details", {}),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def ingest_direct(self, data: dict) -> Optional[TrajectoryEvent]:
        """Directly ingest an application event dict (for testing/demo)."""
        return self._parse_app_event(json.dumps(data))
