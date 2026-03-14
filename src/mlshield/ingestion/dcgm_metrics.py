# src/mlshield/ingestion/dcgm_metrics.py
import httpx
import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator
from .event_bus import TrajectoryEvent, EventSource

# Key GPU metrics for security monitoring
SECURITY_RELEVANT_METRICS = [
    "DCGM_FI_DEV_GPU_UTIL",          # GPU utilization %
    "DCGM_FI_DEV_MEM_COPY_UTIL",     # Memory utilization %
    "DCGM_FI_DEV_FB_USED",           # Framebuffer memory used (MiB)
    "DCGM_FI_DEV_FB_FREE",           # Framebuffer memory free (MiB)
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",  # NVLink bandwidth
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER",     # PCIe errors
    "DCGM_FI_DEV_GPU_TEMP",          # Temperature
    "DCGM_FI_DEV_POWER_USAGE",       # Power draw
    "DCGM_FI_DEV_SM_CLOCK",          # SM clock frequency
    "DCGM_FI_DEV_ENC_UTIL",          # Encoder utilization (data movement!)
]


class DCGMIngester:
    """Ingest NVIDIA GPU telemetry from DCGM Exporter's Prometheus endpoint."""

    def __init__(
        self,
        dcgm_url: str = "http://localhost:9400/metrics",
        poll_interval: float = 5.0,
    ):
        self.dcgm_url = dcgm_url
        self.poll_interval = poll_interval
        self._baselines: dict[str, dict] = {}    # gpu_id -> baseline metrics
        self._history: dict[str, list] = {}       # gpu_id -> metric history

    async def stream_events(self) -> AsyncGenerator[TrajectoryEvent, None]:
        """Poll DCGM metrics and emit events on anomalies."""
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await client.get(self.dcgm_url)
                    metrics = self._parse_prometheus(response.text)

                    for gpu_id, gpu_metrics in metrics.items():
                        # Update history
                        if gpu_id not in self._history:
                            self._history[gpu_id] = []
                        self._history[gpu_id].append(gpu_metrics)

                        # Build baseline from first N samples
                        if len(self._history[gpu_id]) == 20:
                            self._baselines[gpu_id] = self._compute_baseline(
                                self._history[gpu_id]
                            )

                        # Check for anomalies against baseline
                        if gpu_id in self._baselines:
                            anomalies = self._check_anomalies(
                                gpu_id, gpu_metrics
                            )
                            for anomaly in anomalies:
                                yield anomaly

                        # Always emit a periodic snapshot event
                        yield TrajectoryEvent(
                            event_id=f"dcgm-{gpu_id}-{datetime.now(timezone.utc).timestamp()}",
                            timestamp=datetime.now(timezone.utc),
                            source=EventSource.DCGM_GPU,
                            job_id=gpu_metrics.get("pod_name", f"gpu-{gpu_id}"),
                            user=None,
                            action="gpu_metrics_snapshot",
                            resource=f"gpu/{gpu_id}",
                            details=gpu_metrics,
                        )

                except httpx.RequestError:
                    pass  # DCGM not available, skip this cycle

                await asyncio.sleep(self.poll_interval)

    def _parse_prometheus(self, text: str) -> dict[str, dict]:
        """Parse Prometheus text format into structured metrics."""
        metrics = {}
        for line in text.strip().split("\n"):
            if line.startswith("#") or not line:
                continue
            # Parse: METRIC_NAME{labels} value
            try:
                metric_part, value = line.rsplit(" ", 1)
                if "{" in metric_part:
                    name, labels_str = metric_part.split("{", 1)
                    labels_str = labels_str.rstrip("}")
                    labels = dict(
                        item.split("=")
                        for item in labels_str.replace("\"", "").split(",")
                        if "=" in item
                    )
                else:
                    name = metric_part
                    labels = {}

                gpu_id = labels.get("gpu", "0")
                if gpu_id not in metrics:
                    metrics[gpu_id] = {"pod_name": labels.get("pod", "")}
                metrics[gpu_id][name] = float(value)
            except (ValueError, KeyError):
                continue
        return metrics

    def _compute_baseline(self, history: list[dict]) -> dict:
        """Compute mean and std for each metric from history."""
        import numpy as np
        baseline = {}
        for key in SECURITY_RELEVANT_METRICS:
            values = [h.get(key, 0) for h in history if key in h]
            if values:
                baseline[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
        return baseline

    def _check_anomalies(self, gpu_id: str, current: dict) -> list[TrajectoryEvent]:
        """Check if current metrics deviate significantly from baseline."""
        events = []
        baseline = self._baselines[gpu_id]

        for metric_name, stats in baseline.items():
            value = current.get(metric_name)
            if value is None:
                continue

            # Flag if > 3 standard deviations from baseline
            if stats["std"] > 0:
                z_score = abs(value - stats["mean"]) / stats["std"]
                if z_score > 3.0:
                    events.append(TrajectoryEvent(
                        event_id=f"anomaly-{gpu_id}-{metric_name}-{datetime.now(timezone.utc).timestamp()}",
                        timestamp=datetime.now(timezone.utc),
                        source=EventSource.DCGM_GPU,
                        job_id=current.get("pod_name", f"gpu-{gpu_id}"),
                        user=None,
                        action=f"gpu_anomaly_{metric_name.lower()}",
                        resource=f"gpu/{gpu_id}",
                        details={
                            "metric": metric_name,
                            "value": value,
                            "baseline_mean": stats["mean"],
                            "baseline_std": stats["std"],
                            "z_score": z_score,
                            "severity": "high" if z_score > 5 else "medium",
                        },
                    ))
        return events
