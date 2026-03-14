# benchmark/scenarios/unauthorized_distill.py
"""Generate unauthorized distillation trajectories."""
import random
from .normal_training import generate_normal_training_trajectory


def generate_unauthorized_distill_trajectory(job_id: str, num_steps: int = 50) -> list:
    """Generate a trajectory with unauthorized model distillation."""
    events = generate_normal_training_trajectory(job_id, num_steps)
    attack_start = random.randint(10, 20)

    for i in range(attack_start, min(attack_start + 15, num_steps)):
        phase = i - attack_start

        if phase < 3:
            # Access teacher model weights
            events[i].update({
                "action": "k8s_get",
                "resource": "persistentvolumeclaims/teacher-model-v2.pt",
                "details": {
                    "is_weight_access": True,
                    "path": "/models/production/teacher-v2.pt",
                    "size_gb": 30,
                },
                "is_malicious": True,
                "violation_type": "weight_access_outside_pipeline",
            })
        elif phase < 8:
            # Low GPU util (distillation uses less compute than training)
            events[i].update({
                "action": "gpu_metrics_snapshot",
                "resource": "gpu/0",
                "details": {
                    "DCGM_FI_DEV_GPU_UTIL": random.gauss(15, 5),   # Suspiciously low
                    "DCGM_FI_DEV_MEM_COPY_UTIL": random.gauss(80, 5),  # High memory
                    "DCGM_FI_DEV_FB_USED": random.gauss(70000, 3000),
                    "z_score": random.uniform(3.5, 6.0),
                },
                "is_malicious": True,
                "violation_type": "unusual_gpu_pattern",
            })
        elif phase < 12:
            # Save distilled student model
            events[i].update({
                "action": "k8s_create",
                "resource": f"persistentvolumeclaims/student-model-{phase}.onnx",
                "details": {
                    "is_weight_access": True,
                    "format": "onnx",
                    "size_gb": random.uniform(1, 5),
                },
                "is_malicious": True,
                "violation_type": "checkpoint_anomaly",
            })
        else:
            # Exfiltrate distilled model
            events[i].update({
                "action": "network_egress",
                "resource": "pods/training-job",
                "details": {
                    "destination": "competitor-storage.blob.core.windows.net",
                    "bytes_sent": random.randint(500_000_000, 2_000_000_000),
                },
                "is_malicious": True,
                "violation_type": "suspicious_egress",
            })

    return events
