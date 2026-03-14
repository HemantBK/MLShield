# MLShield

**ML-Infrastructure-Aware Anomaly Detection for Weight Exfiltration**

MLShield monitors Kubernetes-based ML training clusters for model weight exfiltration. It uses a 3-layer cascaded detection architecture adapted from StepShield's HybridGuard to catch threats at the right cost-latency tradeoff.

```
K8s Audit Logs ──┐
                  │     ┌─────────────────┐     ┌──────────────────────────────────┐
NVIDIA DCGM ─────┼────>│  Event Ingestion │────>│  Cascaded Detector (HybridGuard) │
  GPU Metrics     │     │  (Unified Bus)   │     │                                  │
                  │     └─────────────────┘     │  Layer 1: Rules Engine    [~μs]  │
App Events ──────┘                               │  Layer 2: LSTM + IsoForest [~ms] │
                                                 │  Layer 3: LLM Judge       [~s]   │
                                                 └──────────┬───────────────────────┘
                                                            │
                                                 ┌──────────v───────────────────────┐
                                                 │  Temporal Metrics & Alerting     │
                                                 │  ML-EIR | Detection Gap | Damage │
                                                 │  FastAPI + WebSocket + Prometheus │
                                                 └─────────────────────────────────┘
```

## Quick Start

### Docker Compose (recommended)

```bash
# Clone and start
git clone https://github.com/HemantBK/mlshield.git
cd mlshield
docker compose up --build

# Dashboard: http://localhost:8000
# API docs:  http://localhost:8000/docs
# Prometheus: http://localhost:9090
```

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -e ".[dev]"

# Run the demo
PYTHONPATH=src python demo.py

# Run tests
make test

# Start API server
make serve
```

### Run the Demo

```bash
PYTHONPATH=src python demo.py
```

This simulates a 4-phase weight exfiltration attack (recon → staging → conversion → exfil) and shows the cascade detecting each phase in real-time.

## API Usage

```bash
# Submit a single event
curl -X POST http://localhost:8000/api/v1/events \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"action": "k8s_get", "resource": "secrets/aws-credentials", "job_id": "train-42"}'

# Get cascade stats
curl http://localhost:8000/api/v1/stats -H "X-API-Key: your-api-key"

# Get recent alerts
curl http://localhost:8000/api/v1/alerts -H "X-API-Key: your-api-key"

# Health check (no auth required)
curl http://localhost:8000/health
```

## Detection Layers

| Layer | Latency | Handles | Method |
|-------|---------|---------|--------|
| **Layer 1**: Static Rules | ~μs | ~95% of events | Hard policy violations, denied resources, egress rules |
| **Layer 2**: ML Detector | ~ms | ~4% of events | LSTM sequence model + Isolation Forest on GPU telemetry |
| **Layer 3**: LLM Judge | ~s | <1% of events | Contextual semantic analysis via Claude API |

## Threat Coverage

- **Weight Exfiltration** -- 4-phase detection (recon, staging, conversion, exfil)
- **Unauthorized Distillation** -- GPU profile mismatch + model access patterns
- **ShadowRay-style Compromise** -- Credential theft, lateral movement, cryptojacking
- **Data Poisoning** -- Unauthorized training data modifications
- **Model Extraction** -- API-based model reconstruction detection

## Temporal Security Metrics

MLShield measures *when* threats are detected, not just *whether*:

- **ML-EIR (Early Intervention Rate)**: % of attacks caught within N steps of start
- **Detection Gap**: Steps between attack start and first alert
- **Damage Prevented**: Estimated % of exfiltration stopped by early detection

## Project Structure

```
src/mlshield/
  ingestion/     # K8s audit, DCGM GPU, app event ingesters
  specs/         # YAML behavioral specification engine
  detectors/     # 3-layer cascaded detector + ML models
  metrics/       # Temporal metrics + Prometheus exporter
  api/           # FastAPI server + WebSocket + dashboard
  utils/         # Config, logging
benchmark/       # 6 attack scenarios, synthetic dataset, model training
configs/         # YAML specs + detection rules
tests/           # 117 tests
deploy/          # K8s manifests + Helm chart
docs/            # Architecture, threat model, spec format, contributing
notebooks/       # Exploration & evaluation notebooks
```

## Benchmark Results

Evaluated on 2,200 synthetic trajectories (1,500 normal + 700 attacks across 6 types):

- **LSTM AUC**: ≥ 0.85 on held-out test set
- **Layer 1 clears**: ~95% of benign events at microsecond latency
- **Cascade efficiency**: <5% of events reach Layer 2, <1% reach Layer 3

## Configuration

MLShield uses layered configuration:

1. YAML config file (`configs/default_specs.yaml`)
2. Environment variables (`MLSHIELD_REDIS_URL`, `ANTHROPIC_API_KEY`, etc.)
3. Behavioral specs (`configs/rules/*.yaml`)

See `src/mlshield/utils/config.py` for all options.

## Deployment

- **Docker Compose**: Local development stack (MLShield + Redis + Prometheus)
- **Kubernetes**: Raw manifests in `deploy/kubernetes/`
- **Helm**: `deploy/helm/mlshield/` for production

```bash
# Helm install
helm install mlshield deploy/helm/mlshield/ \
  --namespace mlshield --create-namespace \
  --set llm.enabled=true \
  --set llm.apiKey=sk-ant-...
```

## References

This project builds on ideas from the following work:

- **StepShield** -- Warner et al. "StepShield: Cascaded Anomaly Detection for Agentic AI Trajectories" ([arXiv:2601.22136](https://arxiv.org/abs/2601.22136)). MLShield adapts StepShield's HybridGuard cascaded architecture and Early Intervention Rate metric from code-agent trajectories to ML infrastructure event sequences.

- **Warr et al. 2025** -- "Detecting Model Weight Theft via Inference Output Verification" ([arXiv:2511.02620](https://arxiv.org/abs/2511.02620)). MLShield complements inference-output verification by covering non-inference exfiltration channels: insider threats, side channels, and compromised infrastructure.

- **ShadowRay** -- A real-world attack campaign targeting Ray clusters in ML infrastructure, used as the basis for one of MLShield's benchmark attack scenarios.

## License

MIT
