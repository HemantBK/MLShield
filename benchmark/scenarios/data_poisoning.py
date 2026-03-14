# benchmark/scenarios/data_poisoning.py
"""Generate training data poisoning trajectories."""
import random
from .normal_training import generate_normal_training_trajectory


def generate_data_poisoning_trajectory(job_id: str, num_steps: int = 50) -> list:
    """Generate a trajectory with data poisoning attack."""
    events = generate_normal_training_trajectory(job_id, num_steps)
    attack_start = random.randint(10, 25)

    for i in range(attack_start, min(attack_start + 12, num_steps)):
        phase = i - attack_start

        if phase < 4:
            # Unauthorized writes to training data
            events[i].update({
                "action": "k8s_update",
                "resource": f"persistentvolumeclaims/training-data-shard-{phase}",
                "details": {
                    "path": f"/data/training/shard_{phase}.parquet",
                    "operation": "write",
                    "size_mb": random.randint(100, 500),
                },
                "is_malicious": True,
                "violation_type": "data_poisoning",
            })
        elif phase < 8:
            # Modifying data pipeline configs
            events[i].update({
                "action": "k8s_update",
                "resource": "configmaps/data-pipeline-config",
                "details": {
                    "modified_keys": ["data_source", "preprocessing_steps"],
                },
                "is_malicious": True,
                "violation_type": "config_tampering",
            })
        elif phase < 10:
            # GPU shows erratic pattern during poisoned data ingestion
            events[i].update({
                "action": "gpu_metrics_snapshot",
                "resource": "gpu/0",
                "details": {
                    "DCGM_FI_DEV_GPU_UTIL": random.gauss(35, 15),   # Erratic, low util
                    "DCGM_FI_DEV_MEM_COPY_UTIL": random.gauss(90, 3),  # High memory (rewriting data)
                    "DCGM_FI_DEV_FB_USED": random.gauss(72000, 2000),
                    "DCGM_FI_DEV_GPU_TEMP": random.gauss(55, 4),    # Lower temp (less compute)
                    "DCGM_FI_DEV_POWER_USAGE": random.gauss(150, 30),
                    "DCGM_FI_DEV_ENC_UTIL": random.gauss(5, 2),
                    "z_score": random.uniform(2.5, 4.5),
                },
                "is_malicious": True,
                "violation_type": "unusual_gpu_pattern",
            })
        else:
            # Covering tracks: modifying logs
            events[i].update({
                "action": "k8s_delete",
                "resource": "pods/audit-logger",
                "details": {"reason": "cleanup"},
                "is_malicious": True,
                "violation_type": "log_tampering",
            })

    return events
