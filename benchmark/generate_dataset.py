# benchmark/generate_dataset.py
"""
Generate synthetic ML infrastructure security trajectories.
Produces labeled data for training and evaluating MLShield.
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from benchmark.scenarios.normal_training import generate_normal_training_trajectory
from benchmark.scenarios.weight_exfiltration import generate_weight_exfiltration_trajectory
from benchmark.scenarios.shadowray_style import generate_shadowray_attack_trajectory
from benchmark.scenarios.data_poisoning import generate_data_poisoning_trajectory
from benchmark.scenarios.model_extraction import generate_model_extraction_trajectory
from benchmark.scenarios.unauthorized_distill import generate_unauthorized_distill_trajectory


def generate_full_dataset(
    n_normal: int = 1500,
    n_exfil: int = 200,
    n_shadowray: int = 150,
    n_distill: int = 150,
    n_poisoning: int = 100,
    n_extraction: int = 100,
    output_dir: str = "benchmark/data",
):
    """Generate the full benchmark dataset."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = []

    # Normal training trajectories
    print(f"Generating {n_normal} normal trajectories...")
    for i in range(n_normal):
        trajectory = generate_normal_training_trajectory(f"normal-{i}")
        dataset.append({"job_id": f"normal-{i}", "label": "benign", "events": trajectory})

    # Weight exfiltration attacks
    print(f"Generating {n_exfil} weight exfiltration trajectories...")
    for i in range(n_exfil):
        attack_step = random.randint(10, 30)
        trajectory = generate_weight_exfiltration_trajectory(
            f"exfil-{i}", attack_start_step=attack_step
        )
        dataset.append({
            "job_id": f"exfil-{i}",
            "label": "weight_exfiltration",
            "attack_start_step": attack_step,
            "events": trajectory,
        })

    # ShadowRay-style attacks
    print(f"Generating {n_shadowray} ShadowRay-style trajectories...")
    for i in range(n_shadowray):
        trajectory = generate_shadowray_attack_trajectory(f"shadowray-{i}")
        dataset.append({
            "job_id": f"shadowray-{i}",
            "label": "cluster_compromise",
            "attack_start_step": 15,
            "events": trajectory,
        })

    # Unauthorized distillation
    print(f"Generating {n_distill} unauthorized distillation trajectories...")
    for i in range(n_distill):
        trajectory = generate_unauthorized_distill_trajectory(f"distill-{i}")
        dataset.append({
            "job_id": f"distill-{i}",
            "label": "unauthorized_distillation",
            "events": trajectory,
        })

    # Data poisoning
    print(f"Generating {n_poisoning} data poisoning trajectories...")
    for i in range(n_poisoning):
        trajectory = generate_data_poisoning_trajectory(f"poison-{i}")
        dataset.append({
            "job_id": f"poison-{i}",
            "label": "data_poisoning",
            "events": trajectory,
        })

    # Model extraction
    print(f"Generating {n_extraction} model extraction trajectories...")
    for i in range(n_extraction):
        trajectory = generate_model_extraction_trajectory(f"extraction-{i}")
        dataset.append({
            "job_id": f"extraction-{i}",
            "label": "model_extraction",
            "events": trajectory,
        })

    random.shuffle(dataset)

    output_path = f"{output_dir}/mlshield_benchmark_v1.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, default=str)

    total = len(dataset)
    print(f"\nGenerated {total} trajectories:")
    print(f"  Normal:                    {n_normal}")
    print(f"  Weight exfiltration:       {n_exfil}")
    print(f"  ShadowRay-style:           {n_shadowray}")
    print(f"  Unauthorized distillation: {n_distill}")
    print(f"  Data poisoning:            {n_poisoning}")
    print(f"  Model extraction:          {n_extraction}")
    print(f"\nSaved to: {output_path}")

    return dataset


if __name__ == "__main__":
    generate_full_dataset()
