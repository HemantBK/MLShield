# MLShield Threat Model

## Overview

MLShield defends against threats targeting ML model weights in Kubernetes-based training clusters. The primary threat is model weight exfiltration -- the unauthorized extraction of trained model parameters that represent significant intellectual property and competitive advantage.

## Threat Taxonomy

### T1: Weight Exfiltration (Primary Threat)

**Description**: Attacker steals trained model weights from production storage.

**Attack Phases**:
1. **Reconnaissance**: List/enumerate model storage (PVCs, S3 buckets)
2. **Staging**: Access production weight files, copy to writable location
3. **Conversion**: Convert weights to portable format (ONNX, TorchScript)
4. **Exfiltration**: Upload to external storage (S3, Azure Blob, HuggingFace)

**Detection Layers**:
- Layer 1: Catches direct access to `/models/production/`, denied egress destinations, suspicious exec (convert_to_onnx, curl)
- Layer 2: LSTM detects abnormal trajectory pattern (recon -> staging -> conversion -> exfil sequence)
- Layer 3: LLM provides contextual analysis of multi-step exfiltration campaigns

**Severity**: Critical

### T2: Unauthorized Distillation

**Description**: Attacker trains a smaller student model by querying the production model, extracting knowledge without copying weights directly.

**Indicators**:
- Low GPU utilization for a declared training job (inference patterns instead)
- Repeated API calls to model serving endpoints
- Small model checkpoints inconsistent with declared architecture

**Detection Layers**:
- Layer 1: GPU profile mismatch (util < 20% for training job)
- Layer 2: Isolation Forest detects abnormal GPU telemetry pattern
- Layer 3: Contextual analysis of job behavior vs declared purpose

**Severity**: High

### T3: Data Poisoning

**Description**: Attacker modifies training data to inject backdoors or degrade model performance.

**Indicators**:
- Unauthorized writes to training data volumes
- Tampering with data pipeline logs
- Unusual data access patterns outside normal training schedule

**Detection Layers**:
- Layer 1: Write access to training data from unauthorized pods
- Layer 2: Trajectory anomaly (data modification outside training window)

**Severity**: High

### T4: Model Extraction via API

**Description**: Attacker reconstructs the model by systematically querying its predictions.

**Indicators**:
- High-volume inference API requests from single source
- Systematic input patterns (grid search over input space)
- Requests for confidence scores or logits (not just labels)

**Detection Layers**:
- Layer 1: Egress rate limits exceeded
- Layer 2: Unusual query patterns detected by LSTM

**Severity**: High

### T5: ShadowRay-Style Cluster Compromise

**Description**: Based on the real-world ShadowRay attack -- attacker compromises the ML cluster infrastructure to gain persistent access.

**Indicators**:
- Access to cloud credentials (AWS, GCP, Azure)
- Unusual pod exec from external IPs
- Cryptojacking (GPU utilization without corresponding training jobs)
- Credential theft and lateral movement

**Detection Layers**:
- Layer 1: Credential access, external IP exec (critical violations)
- Layer 2: GPU anomaly detection (compute without legitimate workload)
- Layer 3: Multi-signal correlation for complex compromise scenarios

**Severity**: Critical

### T6: Supply Chain Attacks on Model Dependencies

**Description**: Compromised packages in model training pipeline inject malicious code.

**Indicators**:
- Unexpected network egress during `pip install` or dependency resolution
- Modified checksums for training scripts
- Unusual process spawning during training initialization

**Detection Layers**:
- Layer 1: Denied egress during dependency installation
- Layer 2: Trajectory anomaly during job startup phase

**Severity**: Medium

## Trust Boundaries

```
+------------------------------------------+
|  Kubernetes Cluster                       |
|  +------------------------------------+  |
|  |  ML Training Namespace             |  |
|  |  +--------+  +--------+           |  |
|  |  | Train  |  | Train  |           |  |
|  |  | Pod 1  |  | Pod 2  |           |  |
|  |  +---+----+  +---+----+           |  |
|  |      |           |                 |  |
|  |  +---v-----------v----+           |  |
|  |  | Shared Model PVC   | <-- T1    |  |
|  |  +--------------------+           |  |
|  +------------------------------------+  |
|                                          |
|  +------------------------------------+  |
|  |  MLShield Namespace                |  |
|  |  +--------+  +--------+           |  |
|  |  | API    |  | Redis  |           |  |
|  |  | Server |  | Bus    |           |  |
|  |  +--------+  +--------+           |  |
|  +------------------------------------+  |
+------------------------------------------+
           |
    Trust Boundary (Cluster -> External)
           |
    +------v------+
    | External    | <-- Exfil destination
    | Cloud (S3,  |
    | Azure, HF)  |
    +--------------+
```

## Key Assumptions

1. Kubernetes audit logging is enabled and MLShield has read access
2. NVIDIA DCGM exporter is deployed on GPU nodes
3. Network policies allow MLShield to receive events from the ML namespace
4. Attacker has compromised a training pod (not the cluster control plane)
5. MLShield itself runs in a separate namespace with minimal RBAC permissions

## Metrics for Evaluation

- **ML-EIR**: Fraction of attacks detected within 5 steps of first malicious action
- **Detection Gap**: Average steps between attack start and detection
- **Damage Prevented**: Estimated % of exfiltration stopped
- **False Positive Rate**: Benign events incorrectly flagged
- **Cascade Efficiency**: % of events resolved at each layer
