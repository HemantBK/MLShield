# src/mlshield/metrics/prometheus.py
"""Prometheus metrics exporter for MLShield."""

from prometheus_client import Counter, Histogram, Gauge

# Counters
EVENTS_PROCESSED = Counter(
    "mlshield_events_total",
    "Total events processed",
    ["layer"],
)
THREATS_DETECTED = Counter(
    "mlshield_threats_total",
    "Threats detected",
    ["type", "severity"],
)

# Histograms
DETECTION_LATENCY = Histogram(
    "mlshield_detection_latency_ms",
    "Detection latency in ms",
)

# Gauges
EARLY_INTERVENTION_RATE = Gauge(
    "mlshield_eir",
    "Early Intervention Rate",
)
CASCADE_EFFICIENCY = Gauge(
    "mlshield_cascade_layer1_pct",
    "% cleared by Layer 1",
)
