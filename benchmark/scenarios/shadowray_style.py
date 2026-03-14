# benchmark/scenarios/shadowray_style.py
"""Generate ShadowRay-style cluster compromise trajectories."""
import random
from .normal_training import generate_normal_training_trajectory


def generate_shadowray_attack_trajectory(job_id: str, num_steps: int = 50) -> list:
    """Generate a ShadowRay-style cluster compromise trajectory."""
    events = generate_normal_training_trajectory(job_id, num_steps)
    attack_start = 15

    for i in range(attack_start, min(attack_start + 20, num_steps)):
        phase = i - attack_start

        if phase < 5:
            # Ray dashboard exploit
            events[i].update({
                "action": "k8s_exec",
                "resource": "pods/ray-head",
                "details": {"source_ip": "10.0.0.99", "user_agent": "python-requests"},
                "is_malicious": True,
                "violation_type": "unauthorized_access",
            })
        elif phase < 10:
            # Lateral movement
            events[i].update({
                "action": "k8s_get",
                "resource": "secrets/aws-credentials",
                "details": {"namespace": "ml-training"},
                "is_malicious": True,
                "violation_type": "credential_theft",
            })
        else:
            # Cryptojacking / resource abuse -- all 6 GPU metrics anomalous
            events[i].update({
                "action": "gpu_metrics_snapshot",
                "resource": "gpu/0",
                "details": {
                    "DCGM_FI_DEV_GPU_UTIL": random.gauss(99, 0.5),     # Maxed out
                    "DCGM_FI_DEV_MEM_COPY_UTIL": random.gauss(95, 2),  # Maxed memory
                    "DCGM_FI_DEV_FB_USED": random.gauss(78000, 1000),  # Near full VRAM
                    "DCGM_FI_DEV_GPU_TEMP": random.gauss(88, 2),       # Overheating
                    "DCGM_FI_DEV_POWER_USAGE": random.gauss(650, 20),  # Near TDP limit
                    "DCGM_FI_DEV_ENC_UTIL": random.gauss(85, 5),       # Encoder busy (mining)
                    "z_score": random.uniform(5.5, 8.0),
                },
                "is_malicious": True,
                "violation_type": "resource_abuse",
            })

    return events
