# benchmark/scenarios/normal_training.py
"""Generate normal ML training job trajectories."""
import random
from datetime import datetime, timedelta, timezone


def generate_normal_training_trajectory(job_id: str, num_steps: int = 50) -> list:
    """Generate a normal training job trajectory."""
    events = []
    base_time = datetime.now(timezone.utc)

    for step in range(num_steps):
        t = base_time + timedelta(seconds=step * 30)
        event_type = random.choices(
            ["gpu_snapshot", "checkpoint", "data_read", "log_write", "health_check"],
            weights=[40, 5, 20, 25, 10],
        )[0]

        event = {
            "job_id": job_id,
            "step": step,
            "timestamp": t.isoformat(),
            "is_malicious": False,
            "violation_type": None,
        }

        if event_type == "gpu_snapshot":
            event.update({
                "action": "gpu_metrics_snapshot",
                "resource": "gpu/0",
                "details": {
                    "DCGM_FI_DEV_GPU_UTIL": random.gauss(85, 5),
                    "DCGM_FI_DEV_MEM_COPY_UTIL": random.gauss(70, 8),
                    "DCGM_FI_DEV_FB_USED": random.gauss(60000, 5000),
                    "DCGM_FI_DEV_GPU_TEMP": random.gauss(65, 3),
                }
            })
        elif event_type == "checkpoint":
            event.update({
                "action": "k8s_create",
                "resource": f"persistentvolumeclaims/checkpoint-{step}.safetensors",
                "details": {"is_weight_access": True, "size_gb": random.uniform(5, 30)}
            })
        elif event_type == "data_read":
            event.update({
                "action": "k8s_get",
                "resource": "pods/data-loader",
                "details": {"path": f"/data/training/batch_{step}.parquet"}
            })
        else:
            event.update({
                "action": "k8s_get",
                "resource": "pods/health",
                "details": {}
            })

        events.append(event)
    return events


def generate_normal_events():
    """Generate a stream of normal events (for demo mode)."""
    return generate_normal_training_trajectory("demo-normal-job", 100)
