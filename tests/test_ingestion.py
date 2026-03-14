# tests/test_ingestion.py
"""Tests for the ingestion layer."""
import sys
import json
import pytest
from datetime import datetime

sys.path.insert(0, "src")

from mlshield.ingestion.event_bus import (
    EventSource,
    EventSeverity,
    TrajectoryEvent,
    Trajectory,
)
from mlshield.ingestion.k8s_audit import K8sAuditIngester
from mlshield.ingestion.app_events import AppEventIngester


class TestEventBus:
    """Test core data models."""

    def test_event_source_values(self):
        assert EventSource.K8S_AUDIT.value == "k8s_audit"
        assert EventSource.DCGM_GPU.value == "dcgm_gpu"
        assert EventSource.APP_EVENT.value == "app_event"
        assert EventSource.FALCO_ALERT.value == "falco_alert"

    def test_event_severity_values(self):
        assert EventSeverity.INFO.value == "info"
        assert EventSeverity.CRITICAL.value == "critical"

    def test_trajectory_event_creation(self):
        event = TrajectoryEvent(
            event_id="test-1",
            timestamp=datetime(2024, 1, 1),
            source=EventSource.K8S_AUDIT,
            job_id="ml-training/job-1",
            user="admin",
            action="k8s_get",
            resource="pods/training-job",
        )
        assert event.event_id == "test-1"
        assert event.source == EventSource.K8S_AUDIT
        assert event.trajectory_step == 0
        assert event.details == {}

    def test_trajectory_event_with_details(self):
        event = TrajectoryEvent(
            event_id="test-2",
            timestamp=datetime(2024, 1, 1),
            source=EventSource.DCGM_GPU,
            job_id="gpu-0",
            user=None,
            action="gpu_metrics_snapshot",
            resource="gpu/0",
            details={"DCGM_FI_DEV_GPU_UTIL": 85.0, "z_score": 1.2},
        )
        assert event.details["DCGM_FI_DEV_GPU_UTIL"] == 85.0
        assert event.user is None

    def test_trajectory_add_events(self):
        traj = Trajectory(job_id="test-job")
        assert len(traj.events) == 0
        assert traj.start_time is None

        e1 = TrajectoryEvent(
            event_id="e1",
            timestamp=datetime(2024, 1, 1, 10, 0),
            source=EventSource.K8S_AUDIT,
            job_id="test-job",
            user="admin",
            action="k8s_get",
            resource="pods/test",
        )
        traj.add_event(e1)

        assert len(traj.events) == 1
        assert traj.events[0].trajectory_step == 0
        assert traj.start_time == datetime(2024, 1, 1, 10, 0)

        e2 = TrajectoryEvent(
            event_id="e2",
            timestamp=datetime(2024, 1, 1, 10, 1),
            source=EventSource.K8S_AUDIT,
            job_id="test-job",
            user="admin",
            action="k8s_list",
            resource="pods/test",
        )
        traj.add_event(e2)

        assert len(traj.events) == 2
        assert traj.events[1].trajectory_step == 1
        # start_time should not change
        assert traj.start_time == datetime(2024, 1, 1, 10, 0)

    def test_trajectory_job_type_default(self):
        traj = Trajectory(job_id="test")
        assert traj.job_type == "unknown"
        assert traj.spec_name is None


class TestK8sAuditIngester:
    """Test K8s audit log ingester."""

    def test_parse_valid_json(self):
        ingester = K8sAuditIngester()
        result = ingester._parse_audit_event('{"verb": "get", "objectRef": {"resource": "pods"}}')
        assert result is not None
        assert result["verb"] == "get"

    def test_parse_invalid_json(self):
        ingester = K8sAuditIngester()
        result = ingester._parse_audit_event("not json")
        assert result is None

    def test_ml_relevant_filter(self):
        ingester = K8sAuditIngester()
        # Relevant event
        assert ingester._is_ml_relevant({
            "objectRef": {"resource": "pods"},
            "verb": "get",
        }) is True

        # Irrelevant resource
        assert ingester._is_ml_relevant({
            "objectRef": {"resource": "events"},
            "verb": "get",
        }) is False

        # Irrelevant verb
        assert ingester._is_ml_relevant({
            "objectRef": {"resource": "pods"},
            "verb": "proxy",
        }) is False

    def test_weight_access_detection(self):
        ingester = K8sAuditIngester()
        audit = {
            "auditID": "test-id",
            "requestReceivedTimestamp": "2024-01-01T10:00:00Z",
            "verb": "get",
            "user": {"username": "admin"},
            "objectRef": {
                "resource": "persistentvolumeclaims",
                "name": "frontier-model-v3.safetensors",
                "namespace": "ml-training",
            },
            "responseStatus": {"code": 200},
            "sourceIPs": ["10.0.0.1"],
            "userAgent": "kubectl",
        }
        event = ingester._to_trajectory_event(audit)
        assert event.details["is_weight_access"] is True
        assert event.source == EventSource.K8S_AUDIT
        assert "safetensors" in event.resource

    def test_non_weight_access(self):
        ingester = K8sAuditIngester()
        audit = {
            "auditID": "test-id",
            "requestReceivedTimestamp": "2024-01-01T10:00:00Z",
            "verb": "get",
            "user": {"username": "admin"},
            "objectRef": {
                "resource": "pods",
                "name": "data-loader",
                "namespace": "ml-training",
            },
            "responseStatus": {"code": 200},
            "sourceIPs": ["10.0.0.1"],
            "userAgent": "kubelet",
        }
        event = ingester._to_trajectory_event(audit)
        assert event.details["is_weight_access"] is False


class TestAppEventIngester:
    """Test application event ingester."""

    def test_ingest_direct(self):
        ingester = AppEventIngester()
        event = ingester.ingest_direct({
            "event_id": "app-1",
            "timestamp": "2024-01-01T10:00:00",
            "job_id": "training-job-1",
            "user": "ml-pipeline",
            "action": "checkpoint_create",
            "resource": "checkpoint-10.safetensors",
            "details": {"size_gb": 15.5},
        })
        assert event is not None
        assert event.event_id == "app-1"
        assert event.source == EventSource.APP_EVENT
        assert event.details["size_gb"] == 15.5

    def test_ingest_invalid(self):
        ingester = AppEventIngester()
        event = ingester._parse_app_event("not valid json")
        assert event is None

    def test_ingest_minimal(self):
        ingester = AppEventIngester()
        event = ingester.ingest_direct({
            "action": "health_check",
        })
        assert event is not None
        assert event.action == "health_check"
        assert event.job_id == "unknown"
