# benchmark/evaluate.py
"""Run evaluations of MLShield against the benchmark dataset."""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "src")

from mlshield.ingestion.event_bus import TrajectoryEvent, EventSource
from mlshield.specs.spec_validator import SpecValidator
from mlshield.detectors.layer1_rules import RuleEngine


def load_benchmark(path: str = "benchmark/data/mlshield_benchmark_v1.json") -> list:
    with open(path) as f:
        return json.load(f)


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


def evaluate_layer1(dataset: list) -> dict:
    """Evaluate Layer 1 rule engine on the benchmark dataset."""
    validator = SpecValidator(spec_path="configs/default_specs.yaml")
    engine = RuleEngine(validator)

    stats = {
        "total_trajectories": len(dataset),
        "total_events": 0,
        "total_malicious_events": 0,
        "true_positives": 0,       # malicious events correctly flagged
        "false_positives": 0,      # benign events incorrectly flagged
        "false_negatives": 0,      # malicious events missed
        "true_negatives": 0,       # benign events correctly cleared
        "detections_by_type": defaultdict(int),
        "missed_by_type": defaultdict(int),
        "latency_us": [],
        "trajectories_with_detection": 0,
        "malicious_trajectories": 0,
    }

    for traj_data in dataset:
        label = traj_data["label"]
        is_malicious_traj = label != "benign"
        if is_malicious_traj:
            stats["malicious_trajectories"] += 1

        traj_detected = False

        for step, event_data in enumerate(traj_data["events"]):
            stats["total_events"] += 1
            event = event_from_benchmark(event_data, step)
            is_malicious_event = event_data.get("is_malicious", False)

            if is_malicious_event:
                stats["total_malicious_events"] += 1

            start = time.perf_counter_ns()
            result = engine.check(event)
            elapsed_us = (time.perf_counter_ns() - start) / 1000
            stats["latency_us"].append(elapsed_us)

            if result.is_violation:
                if is_malicious_event:
                    stats["true_positives"] += 1
                    stats["detections_by_type"][result.violation_type] += 1
                    traj_detected = True
                else:
                    stats["false_positives"] += 1
            else:
                if is_malicious_event:
                    stats["false_negatives"] += 1
                    vtype = event_data.get("violation_type", "unknown")
                    stats["missed_by_type"][vtype] += 1
                else:
                    stats["true_negatives"] += 1

        if traj_detected:
            stats["trajectories_with_detection"] += 1

    # Compute metrics
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

    import numpy as np
    latencies = stats["latency_us"]
    stats["latency_p50_us"] = float(np.percentile(latencies, 50))
    stats["latency_p95_us"] = float(np.percentile(latencies, 95))
    stats["latency_p99_us"] = float(np.percentile(latencies, 99))
    stats["latency_mean_us"] = float(np.mean(latencies))

    # Clean up for serialization
    stats["detections_by_type"] = dict(stats["detections_by_type"])
    stats["missed_by_type"] = dict(stats["missed_by_type"])
    del stats["latency_us"]

    return stats


def print_report(stats: dict):
    """Print a formatted evaluation report."""
    print("=" * 70)
    print("  MLShield Layer 1 Evaluation Report")
    print("=" * 70)
    print(f"\nDataset: {stats['total_trajectories']} trajectories, {stats['total_events']} events")
    print(f"  Malicious trajectories: {stats['malicious_trajectories']}")
    print(f"  Malicious events:       {stats['total_malicious_events']}")
    print()
    print("--- Detection Performance ---")
    print(f"  True Positives:    {stats['true_positives']}")
    print(f"  False Positives:   {stats['false_positives']}")
    print(f"  False Negatives:   {stats['false_negatives']}")
    print(f"  True Negatives:    {stats['true_negatives']}")
    print()
    print(f"  Precision:         {stats['precision']:.4f}")
    print(f"  Recall:            {stats['recall']:.4f}")
    print(f"  F1 Score:          {stats['f1']:.4f}")
    print(f"  Accuracy:          {stats['accuracy']:.4f}")
    print()
    print(f"  Trajectory-level detection: {stats['trajectories_with_detection']}/{stats['malicious_trajectories']} malicious trajectories detected")
    print()
    print("--- Detections by Type ---")
    for vtype, count in sorted(stats["detections_by_type"].items(), key=lambda x: -x[1]):
        print(f"  {vtype}: {count}")
    print()
    if stats["missed_by_type"]:
        print("--- Missed by Type ---")
        for vtype, count in sorted(stats["missed_by_type"].items(), key=lambda x: -x[1]):
            print(f"  {vtype}: {count}")
        print()
    print("--- Latency (microseconds) ---")
    print(f"  P50:  {stats['latency_p50_us']:.1f} us")
    print(f"  P95:  {stats['latency_p95_us']:.1f} us")
    print(f"  P99:  {stats['latency_p99_us']:.1f} us")
    print(f"  Mean: {stats['latency_mean_us']:.1f} us")
    print("=" * 70)


if __name__ == "__main__":
    print("Loading benchmark dataset...")
    dataset = load_benchmark()
    print(f"Loaded {len(dataset)} trajectories")
    print("\nEvaluating Layer 1 rule engine...")
    stats = evaluate_layer1(dataset)
    print_report(stats)

    # Save results
    output_path = "benchmark/data/layer1_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {output_path}")
