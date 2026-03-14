# MLShield Architecture

MLShield monitors Kubernetes-based ML training clusters for model weight exfiltration. It adapts StepShield's cascaded HybridGuard architecture from code-agent trajectories to infrastructure event sequences.

## System Overview

```
DATA SOURCES                 EVENT INGESTION         SPECIFICATION ENGINE
+--------------+             +----------------+      +------------------+
| K8s Audit    |------------>|                |      | YAML Behavioral  |
| Logs (JSON)  |             | Unified Event  |      | Specs per job    |
+--------------+             | Bus (Redis /   |----->| type             |
| NVIDIA DCGM  |------------>| In-memory)     |      +------------------+
| GPU Metrics  |             |                |              |
+--------------+             | Normalizes to  |              v
| Application  |------------>| TrajectoryEvent|      +------------------+
| Events       |             +----------------+      | CASCADED         |
+--------------+                    |                | DETECTOR         |
                                    v                | (HybridGuard)    |
                             +------+--------+       +------------------+
                             | Layer 1: Rules |  ~us |
                             | Clears ~95%    |----->|
                             +----------------+      |
                                    |                |
                             +------+--------+       |
                             | Layer 2: ML    |  ~ms |
                             | LSTM + IsoFor  |----->|
                             +----------------+      |
                                    |                |
                             +------+--------+       |
                             | Layer 3: LLM   |  ~s  |
                             | Semantic Judge |----->|
                             +----------------+      |
                                                     v
                                            +------------------+
                                            | TEMPORAL METRICS |
                                            | & ALERTING       |
                                            +------------------+
                                            | ML-EIR           |
                                            | Detection Gap    |
                                            | Damage Prevented |
                                            +------------------+
                                                     |
                                            +--------+--------+
                                            | FastAPI + WS     |
                                            | Prometheus /     |
                                            | metrics          |
                                            +------------------+
```

## Components

### 1. Data Sources & Ingestion (`src/mlshield/ingestion/`)

Three ingesters normalize heterogeneous infrastructure signals into a unified `TrajectoryEvent` format:

- **K8sAuditIngester** (`k8s_audit.py`): Streams Kubernetes audit logs, filtering for ML-relevant resources (pods, jobs, secrets, ConfigMaps, PVCs). Maps K8s verbs to MLShield actions.
- **DCGMIngester** (`dcgm_metrics.py`): Polls NVIDIA DCGM Prometheus endpoint at configurable intervals. Tracks GPU baselines and emits events on anomalous telemetry spikes.
- **AppEventIngester** (`app_events.py`): Subscribes to Redis Pub/Sub for application-level events. Supports direct injection for testing.

All events are normalized into `TrajectoryEvent` (defined in `event_bus.py`) which contains: event_id, timestamp, source, job_id, user, action, resource, details, and trajectory_step.

### 2. Specification Engine (`src/mlshield/specs/`)

YAML-defined behavioral specifications describe what ML jobs are allowed to do:

- **SpecParser** (`spec_parser.py`): Loads specs from YAML, matches jobs by labels, namespace patterns, and job type.
- **SpecValidator** (`spec_validator.py`): Validates events against specs: data access paths, network egress rules, GPU utilization profiles, checkpoint behavior.
- **Spec Types** (`spec_types.py`): `ViolationResult` dataclass returned by all validation checks.

Default specs are in `configs/default_specs.yaml`. Custom rules live in `configs/rules/`.

### 3. Cascaded Detector (`src/mlshield/detectors/`)

The core detection engine uses a 3-layer cascade inspired by StepShield's HybridGuard:

**Layer 1 -- Static Rules Engine** (`layer1_rules.py`)
- Microsecond latency
- Hard policy violations (credential access, denied resources)
- Suspicious command patterns (convert_to_onnx, curl, wget, nc)
- Network egress against denied patterns (S3, Azure Blob, HuggingFace)
- Clears ~95% of benign events

**Layer 2 -- ML Anomaly Detector** (`layer2_ml.py`)
- Millisecond latency
- Only processes events escalated from Layer 1 (~5%)
- **TrajectoryLSTM** (`models/lstm_detector.py`): 2-layer LSTM with attention mechanism. 32-dim event features, trained on benchmark trajectories.
- **GPUIsolationForest** (`models/isolation.py`): Isolation Forest on 6 GPU telemetry features.
- Weighted composite scoring: 0.7 LSTM + 0.3 Isolation Forest

**Layer 3 -- LLM Semantic Judge** (`layer3_llm.py`)
- Second-scale latency
- Only invoked for medium-confidence events (<1% of total)
- Calls Anthropic Claude API with full event context
- Provides natural language threat explanations
- Falls back to heuristic scoring when API is unavailable

The **CascadedDetector** (`cascade.py`) orchestrates all three layers and tracks cascade efficiency statistics.

### 4. Temporal Metrics (`src/mlshield/metrics/`)

Security metrics adapted from StepShield that measure *when* threats are detected, not just *whether*:

- **ML-EIR (Early Intervention Rate)**: Fraction of threats detected within N steps of the actual violation
- **Detection Gap**: Steps between violation start and detection (mean, median, max)
- **Damage Prevented**: Estimated % of exfiltration stopped by early detection
- **Prometheus exporter** (`prometheus.py`): Events processed, threats detected, detection latency, EIR, cascade efficiency

### 5. API & Dashboard (`src/mlshield/api/`)

FastAPI server providing:

- `POST /api/v1/events` -- Submit single event for analysis
- `POST /api/v1/events/batch` -- Submit event batch
- `GET /api/v1/stats` -- Cascade efficiency statistics
- `GET /api/v1/stats/temporal` -- Temporal security metrics
- `GET /api/v1/alerts` -- Recent alerts with filtering
- `GET /api/v1/alerts/summary` -- Alert summary by type/severity/layer
- `WS /ws/alerts` -- Real-time WebSocket alert streaming
- `GET /metrics` -- Prometheus metrics endpoint
- `GET /` -- HTML monitoring dashboard

### 6. Benchmark (`benchmark/`)

Synthetic dataset generator with 6 attack scenarios:
- Normal training (baseline)
- Weight exfiltration (4-phase: recon, staging, conversion, exfil)
- Unauthorized distillation
- Data poisoning
- Model extraction
- ShadowRay-style cluster compromise

Generated dataset: 2,200 trajectories (1,500 normal + 700 attack).

## Configuration

MLShield uses a layered configuration approach:

1. **YAML config file** (via `MLSHIELD_CONFIG` env var)
2. **Environment variable overrides** (e.g., `MLSHIELD_REDIS_URL`, `ANTHROPIC_API_KEY`)
3. **Behavioral specs** (separate YAML files in `configs/`)

See `src/mlshield/utils/config.py` for all configuration options.

## Deployment

- **Docker Compose**: `docker-compose.yaml` for local development (MLShield + Redis + Prometheus)
- **Kubernetes**: Raw manifests in `deploy/kubernetes/`
- **Helm**: Chart in `deploy/helm/mlshield/` for production deployments
