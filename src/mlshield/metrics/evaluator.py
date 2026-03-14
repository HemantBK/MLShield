# src/mlshield/metrics/evaluator.py
"""Benchmark evaluation utilities for the full cascade."""
import json
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

from ..ingestion.event_bus import TrajectoryEvent, EventSource
from ..specs.spec_validator import SpecValidator
from ..detectors.layer2_ml import MLDetector
from ..detectors.layer3_llm import LLMJudge
from ..detectors.cascade import CascadedDetector


def event_from_benchmark(raw: dict, step: int) -> TrajectoryEvent:
    """Convert a benchmark event dict to a TrajectoryEvent."""
    source_map = {
        "gpu_metrics_snapshot": EventSource.DCGM_GPU,
        "gpu_anomaly": EventSource.DCGM_GPU,
    }
    action = raw.get("action", "k8s_get")
    source = source_map.get(action, EventSource.K8S_AUDIT)

    return TrajectoryEvent(
        event_id=f"{raw.get('job_id', 'unknown')}-step-{step}",
        timestamp=datetime.fromisoformat(raw["timestamp"])
        if "timestamp" in raw
        else datetime.now(timezone.utc),
        source=source,
        job_id=raw.get("job_id", "unknown"),
        user=raw.get("user"),
        action=action,
        resource=raw.get("resource", ""),
        details=raw.get("details", {}),
        trajectory_step=step,
    )


async def evaluate_cascade(
    benchmark_path: str = "benchmark/data/mlshield_benchmark_v1.json",
    lstm_model_path: str = "benchmark/data/models/lstm_detector.pt",
    iso_model_path: str = "benchmark/data/models/isolation_forest.pkl",
):
    """Evaluate the full cascaded detector on the benchmark dataset."""
    # Load benchmark
    with open(benchmark_path) as f:
        dataset = json.load(f)

    # Build cascade
    validator = SpecValidator(spec_path="configs/default_specs.yaml")
    ml_detector = MLDetector(
        lstm_model_path=lstm_model_path,
        isolation_model_path=iso_model_path,
    )
    llm_judge = LLMJudge()  # Will use fallback (no API key in eval)
    cascade = CascadedDetector(
        spec_validator=validator,
        ml_detector=ml_detector,
        llm_judge=llm_judge,
    )

    stats = {
        "total_trajectories": len(dataset),
        "total_events": 0,
        "total_malicious_events": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 0,
        "detections_by_layer": defaultdict(int),
        "detections_by_type": defaultdict(int),
        "latency_ms": [],
        "trajectories_with_detection": 0,
        "malicious_trajectories": 0,
    }

    for traj_data in dataset:
        label = traj_data["label"]
        is_malicious_traj = label != "benign"
        if is_malicious_traj:
            stats["malicious_trajectories"] += 1

        # Register ground truth for temporal metrics
        attack_start = traj_data.get("attack_start_step")
        if attack_start is not None:
            cascade.temporal_metrics.record_ground_truth_violation(
                traj_data["job_id"], attack_start
            )

        traj_detected = False
        ml_detector.clear_buffer(traj_data["job_id"])

        for step, event_data in enumerate(traj_data["events"]):
            stats["total_events"] += 1
            event = event_from_benchmark(event_data, step)
            is_malicious_event = event_data.get("is_malicious", False)
            if is_malicious_event:
                stats["total_malicious_events"] += 1

            result = await cascade.evaluate(event)
            stats["latency_ms"].append(result.detection_latency_ms)

            if result.is_threat:
                stats["detections_by_layer"][f"layer_{result.detected_by_layer}"] += 1
                stats["detections_by_type"][result.threat_type] += 1
                if is_malicious_event:
                    stats["true_positives"] += 1
                    traj_detected = True
                else:
                    stats["false_positives"] += 1
            else:
                if is_malicious_event:
                    stats["false_negatives"] += 1
                else:
                    stats["true_negatives"] += 1

        if traj_detected:
            stats["trajectories_with_detection"] += 1

    # Compute derived metrics
    tp = stats["true_positives"]
    fp = stats["false_positives"]
    fn = stats["false_negatives"]
    tn = stats["true_negatives"]

    stats["precision"] = tp / max(tp + fp, 1)
    stats["recall"] = tp / max(tp + fn, 1)
    stats["f1"] = (
        2 * stats["precision"] * stats["recall"]
        / max(stats["precision"] + stats["recall"], 1e-9)
    )
    stats["accuracy"] = (tp + tn) / max(tp + fp + fn + tn, 1)

    latencies = stats["latency_ms"]
    stats["latency_p50_ms"] = float(np.percentile(latencies, 50))
    stats["latency_p95_ms"] = float(np.percentile(latencies, 95))
    stats["latency_p99_ms"] = float(np.percentile(latencies, 99))

    # Cascade stats
    stats["cascade_stats"] = cascade.get_cascade_stats()

    # Temporal metrics
    stats["temporal_metrics"] = cascade.temporal_metrics.summary()

    # Cleanup for serialization
    stats["detections_by_layer"] = dict(stats["detections_by_layer"])
    stats["detections_by_type"] = dict(stats["detections_by_type"])
    del stats["latency_ms"]

    return stats
