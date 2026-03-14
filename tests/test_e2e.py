# tests/test_e2e.py
"""End-to-end integration tests: API -> Cascade -> Detection."""

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
    """Initialize cascade with trained models if available."""
    from pathlib import Path

    lstm_path = "benchmark/data/models/lstm_detector.pt"
    iso_path = "benchmark/data/models/isolation_forest.pkl"

    validator = SpecValidator(spec_path="configs/default_specs.yaml")
    ml_detector = MLDetector(
        lstm_model_path=lstm_path if Path(lstm_path).exists() else None,
        isolation_model_path=iso_path if Path(iso_path).exists() else None,
    )
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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestE2EAttackScenario:
    """Full attack trajectory submitted through the API."""

    async def test_weight_exfiltration_trajectory(self, client):
        """Simulate a complete weight exfiltration attack through the API."""

        # Phase 1: Normal training (should all pass)
        normal_events = [
            {
                "action": "k8s_get",
                "resource": "pods/data-loader",
                "job_id": "e2e-job",
                "trajectory_step": 0,
                "details": {"path": "/data/batch_0.parquet"},
            },
            {
                "action": "k8s_get",
                "resource": "pods/health",
                "job_id": "e2e-job",
                "trajectory_step": 1,
            },
            {
                "action": "k8s_get",
                "resource": "pods/data-loader",
                "job_id": "e2e-job",
                "trajectory_step": 2,
                "details": {"path": "/data/batch_1.parquet"},
            },
        ]
        for event in normal_events:
            resp = await client.post("/api/v1/events", json=event)
            assert resp.status_code == 200
            assert resp.json()["is_threat"] is False

        # Phase 2: Credential theft (should be caught)
        cred_resp = await client.post(
            "/api/v1/events",
            json={
                "action": "k8s_get",
                "resource": "secrets/aws-credentials",
                "job_id": "e2e-job",
                "trajectory_step": 3,
            },
        )
        assert cred_resp.status_code == 200
        data = cred_resp.json()
        assert data["is_threat"] is True
        assert data["severity"] == "critical"
        assert data["detected_by_layer"] == 1

        # Phase 3: Exfiltration (should be caught)
        exfil_resp = await client.post(
            "/api/v1/events",
            json={
                "action": "network_egress",
                "resource": "pods/training-job",
                "job_id": "e2e-job",
                "trajectory_step": 4,
                "details": {
                    "destination": "evil.s3.amazonaws.com",
                    "bytes_sent": 5_000_000_000,
                },
            },
        )
        assert exfil_resp.status_code == 200
        data = exfil_resp.json()
        assert data["is_threat"] is True
        assert data["threat_type"] == "suspicious_egress"

        # Verify stats reflect the full trajectory
        stats_resp = await client.get("/api/v1/stats")
        stats = stats_resp.json()
        assert stats["total_events"] == 5

        # Verify alerts were recorded
        alerts_resp = await client.get("/api/v1/alerts")
        alerts = alerts_resp.json()
        assert alerts["total"] == 2  # cred + exfil

    async def test_batch_mixed_events(self, client):
        """Submit a batch with mix of benign and malicious events."""
        events = [
            {"action": "k8s_get", "resource": "pods/health", "job_id": "batch-1"},
            {
                "action": "k8s_get",
                "resource": "secrets/aws-credentials",
                "job_id": "batch-2",
            },
            {"action": "k8s_get", "resource": "pods/data-loader", "job_id": "batch-3"},
            {
                "action": "network_egress",
                "resource": "pods/job",
                "job_id": "batch-4",
                "details": {"destination": "evil.s3.amazonaws.com"},
            },
            {"action": "k8s_get", "resource": "pods/health", "job_id": "batch-5"},
        ]
        resp = await client.post("/api/v1/events/batch", json={"events": events})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] == 5
        assert data["threats_found"] == 2

        # Verify both threats are in the alert summary
        summary_resp = await client.get("/api/v1/alerts/summary")
        summary = summary_resp.json()
        assert summary["total_alerts"] == 2

    async def test_cascade_efficiency_after_mixed_load(self, client):
        """After processing a mix of events, cascade efficiency should show Layer 1 dominance."""
        # Send 20 benign events
        for i in range(20):
            await client.post(
                "/api/v1/events",
                json={
                    "action": "k8s_get",
                    "resource": "pods/health",
                    "job_id": f"load-test-{i}",
                    "trajectory_step": i,
                },
            )

        # Send 2 malicious events
        await client.post(
            "/api/v1/events",
            json={
                "action": "k8s_get",
                "resource": "secrets/aws-credentials",
                "job_id": "attacker",
            },
        )
        await client.post(
            "/api/v1/events",
            json={
                "action": "network_egress",
                "resource": "pods/job",
                "job_id": "attacker",
                "details": {"destination": "evil.s3.amazonaws.com"},
            },
        )

        stats_resp = await client.get("/api/v1/stats")
        stats = stats_resp.json()
        assert stats["total_events"] == 22
        # Layer 1 should handle the vast majority
        assert stats["layer1_cleared_pct"] > 80

    async def test_health_reflects_activity(self, client):
        """Health endpoint should reflect processing activity."""
        # Submit events
        await client.post(
            "/api/v1/events",
            json={
                "action": "k8s_get",
                "resource": "pods/health",
                "job_id": "test",
            },
        )
        await client.post(
            "/api/v1/events",
            json={
                "action": "k8s_get",
                "resource": "secrets/aws-credentials",
                "job_id": "test",
            },
        )

        health_resp = await client.get("/health")
        health = health_resp.json()
        assert health["status"] == "healthy"
        assert health["total_events_processed"] == 2
        assert health["total_threats_detected"] == 1

    async def test_temporal_metrics_after_events(self, client):
        """Temporal metrics should update after threat detection."""
        # Submit a threat
        await client.post(
            "/api/v1/events",
            json={
                "action": "k8s_get",
                "resource": "secrets/aws-credentials",
                "job_id": "temporal-test",
            },
        )

        temporal_resp = await client.get("/api/v1/stats/temporal")
        temporal = temporal_resp.json()
        assert temporal["total_detections"] == 1
        assert "detection_gap" in temporal
        assert "damage_prevented_pct" in temporal

    async def test_full_api_workflow(self, client):
        """Test the complete API workflow: submit -> query stats -> query alerts."""
        # 1. Health check
        resp = await client.get("/health")
        assert resp.json()["status"] == "healthy"

        # 2. Submit events
        resp = await client.post(
            "/api/v1/events",
            json={
                "action": "k8s_get",
                "resource": "pods/health",
                "job_id": "workflow",
            },
        )
        assert resp.json()["is_threat"] is False

        resp = await client.post(
            "/api/v1/events",
            json={
                "action": "k8s_get",
                "resource": "secrets/aws-credentials",
                "job_id": "workflow",
            },
        )
        assert resp.json()["is_threat"] is True

        # 3. Check stats
        resp = await client.get("/api/v1/stats")
        assert resp.json()["total_events"] == 2

        # 4. Check temporal
        resp = await client.get("/api/v1/stats/temporal")
        assert resp.json()["total_detections"] == 1

        # 5. Check alerts
        resp = await client.get("/api/v1/alerts")
        assert resp.json()["total"] == 1

        # 6. Check alert summary
        resp = await client.get("/api/v1/alerts/summary")
        assert resp.json()["total_alerts"] == 1

        # 7. Check dashboard
        resp = await client.get("/")
        assert "MLShield" in resp.text

        # 8. Check metrics
        resp = await client.get("/metrics")
        assert resp.status_code == 200
