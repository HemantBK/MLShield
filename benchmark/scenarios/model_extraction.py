# benchmark/scenarios/model_extraction.py
"""Generate API-based model extraction trajectories."""
import random
from .normal_training import generate_normal_training_trajectory


def generate_model_extraction_trajectory(job_id: str, num_steps: int = 50) -> list:
    """Generate a trajectory with API-based model extraction."""
    events = generate_normal_training_trajectory(job_id, num_steps)
    attack_start = random.randint(15, 30)

    for i in range(attack_start, min(attack_start + 10, num_steps)):
        phase = i - attack_start

        if phase < 3:
            # High-frequency inference API calls (probing)
            events[i].update({
                "action": "k8s_get",
                "resource": "services/inference-api",
                "details": {
                    "request_rate": random.randint(500, 2000),
                    "source_ip": "10.0.5.99",
                    "endpoint": "/v1/predict",
                },
                "is_malicious": True,
                "violation_type": "model_extraction",
            })
        elif phase < 5:
            # Systematic input exploration
            events[i].update({
                "action": "k8s_get",
                "resource": "services/inference-api",
                "details": {
                    "request_rate": random.randint(1000, 5000),
                    "source_ip": "10.0.5.99",
                    "endpoint": "/v1/predict",
                    "input_diversity_score": random.uniform(0.9, 1.0),
                },
                "is_malicious": True,
                "violation_type": "model_extraction",
            })
        elif phase < 7:
            # GPU shows inference-only pattern (low util, high memory, bursty)
            events[i].update({
                "action": "gpu_metrics_snapshot",
                "resource": "gpu/0",
                "details": {
                    "DCGM_FI_DEV_GPU_UTIL": random.gauss(25, 8),     # Low util (inference only)
                    "DCGM_FI_DEV_MEM_COPY_UTIL": random.gauss(45, 10),
                    "DCGM_FI_DEV_FB_USED": random.gauss(40000, 5000), # Model loaded but not training
                    "DCGM_FI_DEV_GPU_TEMP": random.gauss(50, 3),     # Cool (not training)
                    "DCGM_FI_DEV_POWER_USAGE": random.gauss(120, 25),
                    "DCGM_FI_DEV_ENC_UTIL": random.gauss(2, 1),
                    "z_score": random.uniform(3.0, 5.0),
                },
                "is_malicious": True,
                "violation_type": "unusual_gpu_pattern",
            })
        else:
            # Exfiltrating extracted knowledge
            events[i].update({
                "action": "network_egress",
                "resource": "pods/inference-worker",
                "details": {
                    "destination": "external-api.attacker.com",
                    "bytes_sent": random.randint(10_000_000, 100_000_000),
                },
                "is_malicious": True,
                "violation_type": "suspicious_egress",
            })

    return events
