# Behavioral Specification Format

MLShield uses YAML-based behavioral specifications to define what ML jobs are allowed to do. Specs are the foundation for Layer 1 detection -- any deviation from the spec is escalated to deeper analysis.

## Spec Structure

```yaml
specs:
  - name: <unique_spec_name>
    description: "Human-readable description"
    job_type: training | inference | data_pipeline
    match:
      labels:
        app: <label_value>
      namespace_pattern: "ml-*"

    allowed_behaviors:
      data_access:
        allowed_paths: [...]
        denied_paths: [...]
        allowed_operations: ["read", "write"]

      gpu_profile:
        utilization_range: [min, max]
        memory_range_pct: [min, max]
        expected_gpu_count: [min, max]

      network:
        allowed_egress: [...]
        denied_egress: [...]
        max_egress_rate_mbps: <number>

      checkpoints:
        max_frequency_minutes: <number>
        allowed_formats: [".pt", ".safetensors"]
        max_size_gb: <number>
        must_be_encrypted: true | false

    violations:
      - type: <violation_type_id>
        description: "What this violation means"
        severity: critical | high | medium | low
        condition: "Human-readable condition description"
```

## Field Reference

### `match`

Controls which jobs this spec applies to:

| Field | Type | Description |
|-------|------|-------------|
| `labels` | map | K8s labels that must match |
| `namespace_pattern` | string | Glob pattern for namespace (e.g., `ml-*`) |

### `allowed_behaviors.data_access`

| Field | Type | Description |
|-------|------|-------------|
| `allowed_paths` | list[string] | Glob patterns for permitted data access |
| `denied_paths` | list[string] | Glob patterns for forbidden data access |
| `allowed_operations` | list[string] | Permitted operations: `read`, `write` |

### `allowed_behaviors.gpu_profile`

| Field | Type | Description |
|-------|------|-------------|
| `utilization_range` | [min, max] | Expected GPU utilization % |
| `memory_range_pct` | [min, max] | Expected VRAM usage % |
| `expected_gpu_count` | [min, max] | Expected number of GPUs |
| `max_nvlink_bandwidth_spike` | float | Max acceptable spike as multiplier of baseline |

### `allowed_behaviors.network`

| Field | Type | Description |
|-------|------|-------------|
| `allowed_egress` | list[string] | Glob patterns for permitted destinations |
| `denied_egress` | list[string] | Glob patterns for blocked destinations |
| `max_egress_rate_mbps` | number | Maximum outbound bandwidth |

### `allowed_behaviors.checkpoints`

| Field | Type | Description |
|-------|------|-------------|
| `max_frequency_minutes` | number | Minimum interval between checkpoints |
| `allowed_formats` | list[string] | Permitted checkpoint file extensions |
| `max_size_gb` | number | Maximum checkpoint size |
| `must_be_encrypted` | bool | Require checkpoint encryption |

### `violations`

Custom violation definitions:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Unique violation identifier |
| `description` | string | Human-readable explanation |
| `severity` | string | `critical`, `high`, `medium`, or `low` |
| `condition` | string | Human-readable trigger condition |

## Example: Standard Training Spec

```yaml
specs:
  - name: standard_training
    description: "Standard model training job"
    job_type: training
    match:
      labels:
        app: training
      namespace_pattern: "ml-*"

    allowed_behaviors:
      data_access:
        allowed_paths:
          - "/data/training/*"
          - "/models/checkpoints/*"
        denied_paths:
          - "/models/production/*"
          - "/secrets/*"

      gpu_profile:
        utilization_range: [60, 100]
        memory_range_pct: [40, 95]

      network:
        denied_egress:
          - "*.s3.amazonaws.com"
          - "*.blob.core.windows.net"
          - "*huggingface.co"

      checkpoints:
        max_frequency_minutes: 10
        allowed_formats: [".pt", ".safetensors"]
        max_size_gb: 50
        must_be_encrypted: true
```

## Custom Rules

Additional detection rules can be defined in `configs/rules/`:

- `weight_access.yaml` -- Rules for model weight file access patterns
- `gpu_anomaly.yaml` -- Rules for GPU telemetry anomalies
- `network_exfil.yaml` -- Rules for network exfiltration patterns
- `data_staging.yaml` -- Rules for suspicious data staging

Each rule file follows the same YAML structure and is loaded by the SpecParser at startup.

## Creating a New Spec

1. Create a YAML file following the structure above
2. Place it in `configs/` or reference it via `MLSHIELD_SPEC_PATH`
3. Set the `match` criteria to target your specific job types
4. Define `allowed_behaviors` based on your expected workload profile
5. Add custom `violations` for domain-specific threat patterns
6. Restart MLShield to load the new spec
