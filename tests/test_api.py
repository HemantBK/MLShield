# tests/test_api.py
"""Tests for the FastAPI REST API."""
import sys
import pytest
from datetime import datetime, timezone

sys.path.insert(0, "src")

from httpx import AsyncClient, ASGITransport
from mlshield.api.app import app, state
from mlshield.specs.spec_validator import SpecValidator
from mlshield.detectors.layer2_ml import MLDetector
from mlshield.detectors.layer3_llm import LLMJudge
from mlshield.detectors.cascade import CascadedDetector


@pytest.fixture(autouse=True)
async def setup_cascade():
    """Initialize cascade detector for tests."""
    validator = SpecValidator(spec_path="configs/default_specs.yaml")
    ml_detector = MLDetector()
    llm_judge = LLMJudge()
    cascade = CascadedDetector(
        spec_validator=validator,
        ml_detector=ml_detector,
        llm_judge=llm_judge,
    )
    state.cascade = cascade
    state.start_time = datetime.now(timezone.utc)
    state.alert_history = []
    yield
    state.cascade = None


@pytest.fixture
async def client():
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Test /health endpoint."""

    async def test_health_check(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "uptime_seconds" in data
        assert "total_events_processed" in data

    async def test_health_has_correct_fields(self, client):
        response = await client.get("/health")
        data = response.json()
        assert set(data.keys()) == {
            "status", "version", "uptime_seconds",
            "total_events_processed", "total_threats_detected",
        }


class TestEventSubmission:
    """Test POST /api/v1/events endpoint."""

    async def test_submit_benign_event(self, client):
        response = await client.post("/api/v1/events", json={
            "action": "k8s_get",
            "resource": "pods/health",
            "job_id": "test-job",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["is_threat"] is False
        assert data["detected_by_layer"] == 1
        assert "detection_latency_ms" in data

    async def test_submit_threat_event(self, client):
        response = await client.post("/api/v1/events", json={
            "action": "k8s_get",
            "resource": "secrets/aws-credentials",
            "job_id": "test-job",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["is_threat"] is True
        assert data["severity"] == "critical"
        assert data["detected_by_layer"] == 1

    async def test_submit_egress_event(self, client):
        response = await client.post("/api/v1/events", json={
            "action": "network_egress",
            "resource": "pods/training-job",
            "job_id": "test-job",
            "details": {"destination": "evil.s3.amazonaws.com"},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["is_threat"] is True
        assert data["threat_type"] == "suspicious_egress"

    async def test_submit_with_all_fields(self, client):
        response = await client.post("/api/v1/events", json={
            "event_id": "test-evt-123",
            "timestamp": "2024-01-15T14:30:00",
            "source": "k8s_audit",
            "job_id": "my-job",
            "user": "admin",
            "action": "k8s_get",
            "resource": "pods/test",
            "details": {"path": "/data/test.csv"},
            "trajectory_step": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["event_id"] == "test-evt-123"

    async def test_submit_minimal_event(self, client):
        response = await client.post("/api/v1/events", json={
            "action": "k8s_list",
            "resource": "pods",
        })
        assert response.status_code == 200

    async def test_submit_invalid_event(self, client):
        response = await client.post("/api/v1/events", json={
            "resource": "pods/test",
            # Missing required 'action' field
        })
        assert response.status_code == 422


class TestBatchSubmission:
    """Test POST /api/v1/events/batch endpoint."""

    async def test_submit_batch(self, client):
        events = [
            {"action": "k8s_get", "resource": "pods/health", "job_id": "j1"},
            {"action": "k8s_get", "resource": "secrets/aws-credentials", "job_id": "j2"},
            {"action": "k8s_get", "resource": "pods/data-loader", "job_id": "j3"},
        ]
        response = await client.post("/api/v1/events/batch", json={"events": events})
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 3
        assert data["threats_found"] == 1
        assert len(data["results"]) == 3

    async def test_submit_empty_batch(self, client):
        response = await client.post("/api/v1/events/batch", json={"events": []})
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 0
        assert data["threats_found"] == 0

    async def test_batch_all_threats(self, client):
        events = [
            {"action": "k8s_get", "resource": "secrets/aws-credentials", "job_id": "j1"},
            {"action": "network_egress", "resource": "pods/job", "job_id": "j2",
             "details": {"destination": "evil.s3.amazonaws.com"}},
        ]
        response = await client.post("/api/v1/events/batch", json={"events": events})
        assert response.status_code == 200
        data = response.json()
        assert data["threats_found"] == 2


class TestStatsEndpoint:
    """Test /api/v1/stats endpoints."""

    async def test_get_stats_empty(self, client):
        response = await client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 0
        assert data["layer1_cleared_pct"] == 0.0

    async def test_get_stats_after_events(self, client):
        # Submit some events first
        for _ in range(5):
            await client.post("/api/v1/events", json={
                "action": "k8s_get",
                "resource": "pods/health",
                "job_id": "test",
            })

        response = await client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 5
        assert data["layer1_cleared_pct"] == 100.0

    async def test_get_temporal_stats(self, client):
        response = await client.get("/api/v1/stats/temporal")
        assert response.status_code == 200
        data = response.json()
        assert "total_detections" in data
        assert "early_intervention_rate" in data


class TestAlertsEndpoint:
    """Test /api/v1/alerts endpoints."""

    async def test_get_alerts_empty(self, client):
        response = await client.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        assert data["alerts"] == []
        assert data["total"] == 0

    async def test_get_alerts_after_threats(self, client):
        # Generate some threats
        await client.post("/api/v1/events", json={
            "action": "k8s_get",
            "resource": "secrets/aws-credentials",
            "job_id": "test",
        })
        await client.post("/api/v1/events", json={
            "action": "network_egress",
            "resource": "pods/job",
            "job_id": "test",
            "details": {"destination": "evil.s3.amazonaws.com"},
        })

        response = await client.get("/api/v1/alerts")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["alerts"]) == 2

    async def test_get_alerts_with_limit(self, client):
        # Generate threats using known-caught resources
        known_threats = [
            {"action": "k8s_get", "resource": "secrets/aws-credentials", "job_id": "t1"},
            {"action": "network_egress", "resource": "pods/job", "job_id": "t2",
             "details": {"destination": "evil.s3.amazonaws.com"}},
            {"action": "k8s_get", "resource": "secrets/gcp-credentials", "job_id": "t3"},
            {"action": "k8s_get", "resource": "persistentvolumeclaims/models-production",
             "job_id": "t4", "details": {"is_weight_access": True}},
            {"action": "k8s_get", "resource": "secrets/docker-registry", "job_id": "t5"},
        ]
        for event in known_threats:
            await client.post("/api/v1/events", json=event)

        response = await client.get("/api/v1/alerts?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["alerts"]) == 2

    async def test_get_alerts_filter_severity(self, client):
        await client.post("/api/v1/events", json={
            "action": "k8s_get",
            "resource": "secrets/aws-credentials",
            "job_id": "test",
        })

        response = await client.get("/api/v1/alerts?severity=critical")
        assert response.status_code == 200
        data = response.json()
        assert all(a["severity"] == "critical" for a in data["alerts"])

    async def test_get_alerts_summary(self, client):
        # Generate some threats
        await client.post("/api/v1/events", json={
            "action": "k8s_get",
            "resource": "secrets/aws-credentials",
            "job_id": "test",
        })

        response = await client.get("/api/v1/alerts/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_alerts" in data
        assert "by_type" in data
        assert "by_severity" in data
        assert "by_layer" in data


class TestDashboard:
    """Test dashboard endpoint."""

    async def test_dashboard_serves_html(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        assert "MLShield" in response.text
        assert "text/html" in response.headers["content-type"]


class TestPrometheusMetrics:
    """Test /metrics endpoint."""

    async def test_metrics_endpoint(self, client):
        response = await client.get("/metrics")
        assert response.status_code == 200
        body = response.json()
        assert "mlshield" in body
