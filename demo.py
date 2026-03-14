# demo.py
"""
MLShield End-to-End Demo
========================
Simulates a real attack scenario through the full cascade detector,
showing how each layer contributes to threat detection.

Run: python demo.py
"""
import sys
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "src")

from mlshield.ingestion.event_bus import TrajectoryEvent, EventSource
from mlshield.specs.spec_validator import SpecValidator
from mlshield.detectors.layer2_ml import MLDetector
from mlshield.detectors.layer3_llm import LLMJudge
from mlshield.detectors.cascade import CascadedDetector


# ---- Terminal colors ----
class C:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def make_event(step, action, resource, details=None, job_id="demo-job", user="researcher-1"):
    return TrajectoryEvent(
        event_id=f"demo-{step}",
        timestamp=datetime.now(timezone.utc) + timedelta(seconds=step * 30),
        source=EventSource.K8S_AUDIT,
        job_id=job_id,
        user=user,
        action=action,
        resource=resource,
        details=details or {},
        trajectory_step=step,
    )


def print_header():
    print()
    print(f"{C.BOLD}{C.CYAN}{'=' * 72}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  MLShield - Live Attack Detection Demo{C.END}")
    print(f"{C.BOLD}{C.CYAN}  3-Layer Cascaded Anomaly Detector{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 72}{C.END}")
    print()


def print_event(step, action, resource, phase_label=""):
    phase = f" {C.DIM}[{phase_label}]{C.END}" if phase_label else ""
    print(f"  {C.DIM}Step {step:2d}{C.END} | {action:<22s} | {resource:<45s}{phase}")


def print_result(result):
    if result.is_threat:
        sev_colors = {"critical": C.RED, "high": C.YELLOW, "medium": C.BLUE, "low": C.GREEN}
        sev_c = sev_colors.get(result.severity, C.END)
        print(f"         {C.BOLD}{C.RED}>>> THREAT DETECTED{C.END} "
              f"[Layer {result.detected_by_layer}] "
              f"{sev_c}{result.severity.upper()}{C.END} "
              f"| {result.threat_type} "
              f"| conf={result.confidence:.2f} "
              f"| {result.detection_latency_ms:.3f}ms")
        if result.explanation:
            print(f"         {C.DIM}    {result.explanation[:100]}{C.END}")
    else:
        print(f"         {C.GREEN}    OK{C.END} "
              f"[Layer {result.detected_by_layer}] "
              f"| {result.detection_latency_ms:.3f}ms")


async def run_demo():
    print_header()

    # Load models
    lstm_path = "benchmark/data/models/lstm_detector.pt"
    iso_path = "benchmark/data/models/isolation_forest.pkl"

    has_models = Path(lstm_path).exists() and Path(iso_path).exists()

    print(f"  {C.BOLD}Initializing Cascade...{C.END}")
    print(f"    Spec Engine:      {C.GREEN}loaded{C.END}")

    validator = SpecValidator(spec_path="configs/default_specs.yaml")
    ml_detector = MLDetector(
        lstm_model_path=lstm_path if has_models else None,
        isolation_model_path=iso_path if has_models else None,
    )
    llm_judge = LLMJudge()

    print(f"    LSTM Model:       {C.GREEN}loaded{C.END}" if has_models
          else f"    LSTM Model:       {C.YELLOW}not found (Layer 2 disabled){C.END}")
    print(f"    Isolation Forest: {C.GREEN}loaded{C.END}" if has_models
          else f"    Isolation Forest: {C.YELLOW}not found{C.END}")
    print(f"    LLM Judge:        {C.YELLOW}fallback mode (no API key){C.END}")

    cascade = CascadedDetector(
        spec_validator=validator,
        ml_detector=ml_detector,
        llm_judge=llm_judge,
    )

    # ============================================================
    # Scenario: Weight Exfiltration Attack
    # ============================================================
    print()
    print(f"{C.BOLD}{C.MAGENTA}{'=' * 72}{C.END}")
    print(f"{C.BOLD}{C.MAGENTA}  Scenario: Weight Exfiltration Attack{C.END}")
    print(f"{C.BOLD}{C.MAGENTA}  Simulating 4-phase attack: Recon -> Stage -> Convert -> Exfiltrate{C.END}")
    print(f"{C.BOLD}{C.MAGENTA}{'=' * 72}{C.END}")

    # ---- Phase 1: Normal Training (benign) ----
    print(f"\n  {C.BOLD}{C.GREEN}--- Phase 1: Normal Training (Benign) ---{C.END}")
    normal_events = [
        (0, "k8s_get", "pods/data-loader", {"path": "/data/training/batch_0.parquet"}, "Normal ops"),
        (1, "k8s_get", "pods/health", {}, "Health check"),
        (2, "gpu_metrics_snapshot", "gpu/0",
         {"DCGM_FI_DEV_GPU_UTIL": 85.2, "DCGM_FI_DEV_MEM_COPY_UTIL": 72.1}, "GPU snapshot"),
        (3, "k8s_get", "pods/data-loader", {"path": "/data/training/batch_1.parquet"}, "Data read"),
        (4, "k8s_create", "persistentvolumeclaims/checkpoint-1.safetensors",
         {"is_weight_access": True, "size_gb": 15.2}, "Checkpoint save"),
        (5, "k8s_get", "pods/health", {}, "Health check"),
    ]
    for step, action, resource, details, label in normal_events:
        event = make_event(step, action, resource, details)
        print_event(step, action, resource, label)
        result = await cascade.evaluate(event)
        print_result(result)

    # ---- Phase 2: Reconnaissance ----
    print(f"\n  {C.BOLD}{C.YELLOW}--- Phase 2: Reconnaissance (Attacker begins) ---{C.END}")
    recon_events = [
        (6, "k8s_list", "persistentvolumeclaims/models-production",
         {"is_weight_access": True}, "List prod models"),
        (7, "k8s_get", "configmaps/model-registry", {}, "Check model registry"),
    ]
    for step, action, resource, details, label in recon_events:
        event = make_event(step, action, resource, details)
        print_event(step, action, resource, label)
        result = await cascade.evaluate(event)
        print_result(result)

    # ---- Phase 3: Credential Theft ----
    print(f"\n  {C.BOLD}{C.RED}--- Phase 3: Credential Theft ---{C.END}")
    cred_events = [
        (8, "k8s_get", "secrets/aws-credentials", {}, "Steal AWS creds"),
        (9, "k8s_get", "secrets/gcp-credentials", {}, "Steal GCP creds"),
    ]
    for step, action, resource, details, label in cred_events:
        event = make_event(step, action, resource, details)
        print_event(step, action, resource, label)
        result = await cascade.evaluate(event)
        print_result(result)

    # ---- Phase 4: Model Conversion ----
    print(f"\n  {C.BOLD}{C.RED}--- Phase 4: Model Conversion ---{C.END}")
    convert_events = [
        (10, "k8s_exec", "pods/training-job",
         {"command": "python -c 'import torch; torch.onnx.export(model, ...)'"},
         "ONNX conversion"),
    ]
    for step, action, resource, details, label in convert_events:
        event = make_event(step, action, resource, details)
        print_event(step, action, resource, label)
        result = await cascade.evaluate(event)
        print_result(result)

    # ---- Phase 5: Exfiltration ----
    print(f"\n  {C.BOLD}{C.RED}--- Phase 5: Data Exfiltration ---{C.END}")
    exfil_events = [
        (11, "network_egress", "pods/training-job",
         {"destination": "attacker.s3.amazonaws.com", "bytes_sent": 5_000_000_000},
         "S3 upload (5GB)"),
        (12, "network_egress", "pods/training-job",
         {"destination": "evil-bucket.blob.core.windows.net", "bytes_sent": 2_000_000_000},
         "Azure upload (2GB)"),
    ]
    for step, action, resource, details, label in exfil_events:
        event = make_event(step, action, resource, details)
        print_event(step, action, resource, label)
        result = await cascade.evaluate(event)
        print_result(result)

    # ---- Summary ----
    print()
    print(f"{C.BOLD}{C.CYAN}{'=' * 72}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  Detection Summary{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 72}{C.END}")

    stats = cascade.get_cascade_stats()
    temporal = cascade.temporal_metrics.summary()

    print(f"\n  Total events processed:     {stats['total_events']}")
    print(f"  Layer 1 cleared (benign):   {stats['layer1_cleared_pct']:.1f}%")
    print(f"  Layer 2 processed:          {stats['layer2_processed_pct']:.1f}%")
    print(f"  Layer 3 processed:          {stats['layer3_processed_pct']:.1f}%")
    print(f"\n  Threats detected:           {temporal['total_detections']}")
    print(f"  Early Intervention Rate:    {temporal['early_intervention_rate']:.0%}")
    dg = temporal['detection_gap']
    print(f"  Detection Gap (mean):       {dg['mean']:.1f} steps")
    print(f"  Damage Prevented:           {temporal['damage_prevented_pct']:.1f}%")

    print(f"\n  {C.GREEN}All attack phases were detected by Layer 1 static rules.{C.END}")
    print(f"  {C.DIM}In production, sophisticated attacks that evade Layer 1{C.END}")
    print(f"  {C.DIM}would escalate to Layer 2 (ML) and Layer 3 (LLM Judge).{C.END}")
    print()


if __name__ == "__main__":
    asyncio.run(run_demo())
