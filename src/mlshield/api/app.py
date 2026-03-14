# src/mlshield/api/app.py
"""FastAPI application for MLShield monitoring server."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import hmac
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..utils.config import load_config, MLShieldConfig
from ..utils.logging import setup_logging
from ..ingestion.event_bus import TrajectoryEvent, EventSource
from ..specs.spec_validator import SpecValidator
from ..detectors.layer2_ml import MLDetector
from ..detectors.layer3_llm import LLMJudge
from ..detectors.cascade import CascadedDetector, DetectionResult
from ..metrics.prometheus import (
    EVENTS_PROCESSED,
    THREATS_DETECTED,
    DETECTION_LATENCY,
    EARLY_INTERVENTION_RATE,
    CASCADE_EFFICIENCY,
)


# ---- Pydantic request/response models ----

class EventRequest(BaseModel):
    """Incoming event for analysis."""
    event_id: Optional[str] = None
    timestamp: Optional[str] = None
    source: str = "k8s_audit"
    job_id: str = "unknown"
    user: Optional[str] = None
    action: str
    resource: str
    details: dict = Field(default_factory=dict)
    trajectory_step: int = 0


class DetectionResponse(BaseModel):
    """Detection result returned by the API."""
    event_id: str
    is_threat: bool
    confidence: float
    threat_type: str
    severity: str
    description: str
    detected_by_layer: int
    detection_latency_ms: float
    explanation: str = ""


class BatchEventRequest(BaseModel):
    """Multiple events for batch analysis."""
    events: list[EventRequest]


class BatchDetectionResponse(BaseModel):
    """Batch detection results."""
    results: list[DetectionResponse]
    total_events: int
    threats_found: int


class CascadeStatsResponse(BaseModel):
    """Cascade efficiency statistics."""
    total_events: int
    layer1_cleared_pct: float
    layer2_processed_pct: float
    layer3_processed_pct: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    total_events_processed: int
    total_threats_detected: int


# ---- Global state ----

class AppState:
    """Holds the cascade detector and connection manager."""
    cascade: Optional[CascadedDetector] = None
    config: Optional[MLShieldConfig] = None
    start_time: Optional[datetime] = None
    alert_history: list[dict] = []
    max_alert_history: int = 1000
    ws_clients: list[WebSocket] = []


state = AppState()


# ---- WebSocket connection manager ----

class ConnectionManager:
    """Manage WebSocket connections for real-time alert streaming."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send alert to all connected WebSocket clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


ws_manager = ConnectionManager()

# ---- Rate limiter ----
limiter = Limiter(key_func=get_remote_address)


# ---- API Key authentication ----
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Verify API key if configured. Skips auth when no key is set."""
    if state.config and state.config.api_key:
        if not api_key or not hmac.compare_digest(api_key, state.config.api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key or ""


# ---- App lifecycle ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the cascade detector on startup."""
    import os

    config = load_config(os.environ.get("MLSHIELD_CONFIG"))
    logger = setup_logging(level=config.log_level, json_output=config.log_json)

    # Build cascade
    validator = SpecValidator(spec_path=config.spec_path)

    # Try to load trained models
    lstm_path = os.environ.get("MLSHIELD_LSTM_MODEL", "benchmark/data/models/lstm_detector.pt")
    iso_path = os.environ.get("MLSHIELD_ISO_MODEL", "benchmark/data/models/isolation_forest.pkl")

    ml_detector = MLDetector(
        lstm_model_path=lstm_path if Path(lstm_path).exists() else None,
        isolation_model_path=iso_path if Path(iso_path).exists() else None,
    )
    llm_judge = LLMJudge(
        api_key=config.llm_api_key,
        model=config.llm_model,
    )
    cascade = CascadedDetector(
        spec_validator=validator,
        ml_detector=ml_detector,
        llm_judge=llm_judge,
        layer2_threshold=config.layer2_threshold,
        layer3_threshold=config.layer3_threshold,
    )

    state.cascade = cascade
    state.config = config
    state.start_time = datetime.now(timezone.utc)
    state.alert_history = []

    logger.info(
        "MLShield initialized",
        lstm_loaded=ml_detector._lstm_loaded,
        iso_loaded=ml_detector.isolation_forest.is_fitted,
    )

    yield

    # Cleanup
    state.cascade = None


# ---- FastAPI app ----

app = FastAPI(
    title="MLShield",
    description="ML-Infrastructure-Aware Anomaly Detection for Weight Exfiltration",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# ---- Helper functions ----

def event_request_to_trajectory(req: EventRequest) -> TrajectoryEvent:
    """Convert an API event request to a TrajectoryEvent."""
    source_map = {
        "k8s_audit": EventSource.K8S_AUDIT,
        "dcgm_gpu": EventSource.DCGM_GPU,
        "app_event": EventSource.APP_EVENT,
        "falco_alert": EventSource.FALCO_ALERT,
    }
    return TrajectoryEvent(
        event_id=req.event_id or f"api-{datetime.now(timezone.utc).timestamp()}",
        timestamp=datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.now(timezone.utc),
        source=source_map.get(req.source, EventSource.K8S_AUDIT),
        job_id=req.job_id,
        user=req.user,
        action=req.action,
        resource=req.resource,
        details=req.details,
        trajectory_step=req.trajectory_step,
    )


def detection_to_response(result: DetectionResult) -> DetectionResponse:
    """Convert a DetectionResult to an API response."""
    return DetectionResponse(
        event_id=result.event.event_id,
        is_threat=result.is_threat,
        confidence=result.confidence,
        threat_type=result.threat_type,
        severity=result.severity,
        description=result.description,
        detected_by_layer=result.detected_by_layer,
        detection_latency_ms=result.detection_latency_ms,
        explanation=result.explanation,
    )


def detection_to_dict(result: DetectionResult) -> dict:
    """Convert a DetectionResult to a dict for WebSocket/history."""
    return {
        "event_id": result.event.event_id,
        "timestamp": result.event.timestamp.isoformat(),
        "is_threat": result.is_threat,
        "confidence": result.confidence,
        "threat_type": result.threat_type,
        "severity": result.severity,
        "description": result.description,
        "detected_by_layer": result.detected_by_layer,
        "detection_latency_ms": result.detection_latency_ms,
        "explanation": result.explanation,
        "job_id": result.event.job_id,
        "action": result.event.action,
        "resource": result.event.resource,
    }


# ---- API Endpoints ----

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if state.cascade is None:
        raise HTTPException(status_code=503, detail="Cascade not initialized")

    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()
    threats = sum(1 for a in state.alert_history if a.get("is_threat"))

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=uptime,
        total_events_processed=state.cascade.total_events,
        total_threats_detected=threats,
    )


@app.post("/api/v1/events", response_model=DetectionResponse)
@limiter.limit("120/minute")
async def submit_event(request: Request, req: EventRequest, _key: str = Depends(verify_api_key)):
    """Submit a single event for analysis."""
    if state.cascade is None:
        raise HTTPException(status_code=503, detail="Cascade not initialized")

    event = event_request_to_trajectory(req)
    result = await state.cascade.evaluate(event)

    # Update Prometheus metrics
    EVENTS_PROCESSED.labels(layer=str(result.detected_by_layer)).inc()
    DETECTION_LATENCY.observe(result.detection_latency_ms)

    if result.is_threat:
        THREATS_DETECTED.labels(type=result.threat_type, severity=result.severity).inc()

        # Store in alert history
        alert_dict = detection_to_dict(result)
        state.alert_history.append(alert_dict)
        if len(state.alert_history) > state.max_alert_history:
            state.alert_history = state.alert_history[-state.max_alert_history:]

        # Broadcast to WebSocket clients
        await ws_manager.broadcast(alert_dict)

    return detection_to_response(result)


@app.post("/api/v1/events/batch", response_model=BatchDetectionResponse)
@limiter.limit("30/minute")
async def submit_batch(request: Request, req: BatchEventRequest, _key: str = Depends(verify_api_key)):
    """Submit a batch of events for analysis."""
    if state.cascade is None:
        raise HTTPException(status_code=503, detail="Cascade not initialized")

    results = []
    threats_found = 0

    for event_req in req.events:
        event = event_request_to_trajectory(event_req)
        result = await state.cascade.evaluate(event)

        EVENTS_PROCESSED.labels(layer=str(result.detected_by_layer)).inc()
        DETECTION_LATENCY.observe(result.detection_latency_ms)

        if result.is_threat:
            threats_found += 1
            THREATS_DETECTED.labels(type=result.threat_type, severity=result.severity).inc()
            alert_dict = detection_to_dict(result)
            state.alert_history.append(alert_dict)
            await ws_manager.broadcast(alert_dict)

        results.append(detection_to_response(result))

    # Trim history
    if len(state.alert_history) > state.max_alert_history:
        state.alert_history = state.alert_history[-state.max_alert_history:]

    return BatchDetectionResponse(
        results=results,
        total_events=len(results),
        threats_found=threats_found,
    )


@app.get("/api/v1/stats", response_model=CascadeStatsResponse)
async def get_stats(_key: str = Depends(verify_api_key)):
    """Get cascade efficiency statistics."""
    if state.cascade is None:
        raise HTTPException(status_code=503, detail="Cascade not initialized")

    stats = state.cascade.get_cascade_stats()
    return CascadeStatsResponse(**stats)


@app.get("/api/v1/stats/temporal")
async def get_temporal_stats(_key: str = Depends(verify_api_key)):
    """Get temporal security metrics."""
    if state.cascade is None:
        raise HTTPException(status_code=503, detail="Cascade not initialized")

    return state.cascade.temporal_metrics.summary()


@app.get("/api/v1/alerts")
async def get_alerts(
    limit: int = Query(default=50, le=500),
    severity: Optional[str] = Query(default=None),
    threat_type: Optional[str] = Query(default=None),
    _key: str = Depends(verify_api_key),
):
    """Get recent alerts with optional filtering."""
    alerts = list(reversed(state.alert_history))

    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]
    if threat_type:
        alerts = [a for a in alerts if a.get("threat_type") == threat_type]

    return {"alerts": alerts[:limit], "total": len(alerts)}


@app.get("/api/v1/alerts/summary")
async def get_alerts_summary(_key: str = Depends(verify_api_key)):
    """Get summary of alerts by type and severity."""
    from collections import Counter

    by_type = Counter(a.get("threat_type", "unknown") for a in state.alert_history)
    by_severity = Counter(a.get("severity", "unknown") for a in state.alert_history)
    by_layer = Counter(a.get("detected_by_layer", 0) for a in state.alert_history)

    return {
        "total_alerts": len(state.alert_history),
        "by_type": dict(by_type),
        "by_severity": dict(by_severity),
        "by_layer": {f"layer_{k}": v for k, v in by_layer.items()},
    }


# ---- WebSocket endpoint ----

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """Real-time alert streaming via WebSocket."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Client can send "ping" to keep alive
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ---- Dashboard ----

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard."""
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    if template_path.exists():
        return template_path.read_text()
    return HTMLResponse(content="<h1>MLShield Dashboard</h1><p>Template not found.</p>")


# ---- Prometheus metrics endpoint ----

@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    # Update gauges
    if state.cascade:
        stats = state.cascade.get_cascade_stats()
        CASCADE_EFFICIENCY.set(stats["layer1_cleared_pct"])
        eir = state.cascade.temporal_metrics.early_intervention_rate()
        EARLY_INTERVENTION_RATE.set(eir)

    return JSONResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
