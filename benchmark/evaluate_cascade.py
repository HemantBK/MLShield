# benchmark/evaluate_cascade.py
"""Run full cascade evaluation against the benchmark dataset."""
import json
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, "src")

from mlshield.metrics.evaluator import evaluate_cascade


def print_cascade_report(stats: dict):
    """Print a formatted cascade evaluation report."""
    print("=" * 70)
    print("  MLShield Full Cascade Evaluation Report")
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
    print("--- Cascade Efficiency ---")
    cs = stats["cascade_stats"]
    print(f"  Total events processed: {cs['total_events']}")
    print(f"  Layer 1 cleared:        {cs['layer1_cleared_pct']:.1f}%")
    print(f"  Layer 2 processed:      {cs['layer2_processed_pct']:.1f}%")
    print(f"  Layer 3 processed:      {cs['layer3_processed_pct']:.1f}%")
    print()
    print("--- Detections by Layer ---")
    for layer, count in sorted(stats["detections_by_layer"].items()):
        print(f"  {layer}: {count}")
    print()
    print("--- Detections by Type ---")
    for dtype, count in sorted(stats["detections_by_type"].items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count}")
    print()
    print("--- Latency (milliseconds) ---")
    print(f"  P50:  {stats['latency_p50_ms']:.3f} ms")
    print(f"  P95:  {stats['latency_p95_ms']:.3f} ms")
    print(f"  P99:  {stats['latency_p99_ms']:.3f} ms")
    print()
    print("--- Temporal Metrics ---")
    tm = stats["temporal_metrics"]
    print(f"  Total detections:          {tm['total_detections']}")
    print(f"  Early Intervention Rate:   {tm['early_intervention_rate']:.2%}")
    dg = tm["detection_gap"]
    print(f"  Detection Gap (mean):      {dg['mean']:.1f} steps")
    print(f"  Detection Gap (median):    {dg['median']:.1f} steps")
    print(f"  Detection Gap (max):       {dg['max']} steps")
    print(f"  Damage Prevented:          {tm['damage_prevented_pct']:.1f}%")
    if tm.get("detections_by_layer"):
        print(f"  Detections by layer:       {tm['detections_by_layer']}")
    print("=" * 70)


async def main():
    # Check if models exist
    lstm_path = "benchmark/data/models/lstm_detector.pt"
    iso_path = "benchmark/data/models/isolation_forest.pkl"

    if not Path(lstm_path).exists():
        print("ERROR: Trained models not found. Run benchmark/train_lstm.py first.")
        sys.exit(1)

    print("Loading benchmark dataset and trained models...")
    print("Running full cascade evaluation (this may take a minute)...")
    print()

    stats = await evaluate_cascade(
        benchmark_path="benchmark/data/mlshield_benchmark_v1.json",
        lstm_model_path=lstm_path,
        iso_model_path=iso_path,
    )

    print_cascade_report(stats)

    # Save results
    output_path = "benchmark/data/cascade_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
