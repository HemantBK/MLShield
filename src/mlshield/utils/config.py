# src/mlshield/utils/config.py
"""Configuration management for MLShield."""
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MLShieldConfig:
    """Main configuration for MLShield."""
    # Spec engine
    spec_path: str = "configs/default_specs.yaml"

    # Ingestion
    k8s_audit_log_path: str = "/var/log/kubernetes/audit.log"
    dcgm_url: str = "http://localhost:9400/metrics"
    dcgm_poll_interval: float = 5.0

    # Redis (event bus)
    redis_url: str = "redis://localhost:6379"

    # Detection thresholds
    layer2_threshold: float = 0.6
    layer3_threshold: float = 0.8

    # LLM (Layer 3)
    llm_api_url: str = "https://api.anthropic.com/v1/messages"
    llm_api_key: Optional[str] = None
    llm_model: str = "claude-sonnet-4-20250514"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: Optional[str] = None  # Set to require API key auth
    rate_limit: str = "60/minute"  # Rate limit for API endpoints

    # Logging
    log_level: str = "INFO"
    log_json: bool = False


def load_config(config_path: Optional[str] = None) -> MLShieldConfig:
    """Load config from YAML file and environment variables."""
    config = MLShieldConfig()

    # Load from YAML if provided
    path = config_path or os.environ.get("MLSHIELD_CONFIG")
    if path and Path(path).exists():
        with open(path) as f:
            yaml_config = yaml.safe_load(f) or {}
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Environment variable overrides
    env_mapping = {
        "MLSHIELD_SPEC_PATH": "spec_path",
        "MLSHIELD_K8S_AUDIT_LOG": "k8s_audit_log_path",
        "MLSHIELD_DCGM_URL": "dcgm_url",
        "MLSHIELD_REDIS_URL": "redis_url",
        "MLSHIELD_LAYER2_THRESHOLD": "layer2_threshold",
        "MLSHIELD_LAYER3_THRESHOLD": "layer3_threshold",
        "LLM_API_URL": "llm_api_url",
        "ANTHROPIC_API_KEY": "llm_api_key",
        "MLSHIELD_LLM_MODEL": "llm_model",
        "MLSHIELD_HOST": "api_host",
        "MLSHIELD_PORT": "api_port",
        "MLSHIELD_LOG_LEVEL": "log_level",
        "MLSHIELD_API_KEY": "api_key",
        "MLSHIELD_RATE_LIMIT": "rate_limit",
    }

    for env_var, config_key in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Convert types
            current = getattr(config, config_key)
            if isinstance(current, float):
                value = float(value)
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, bool):
                value = value.lower() in ("true", "1", "yes")
            setattr(config, config_key, value)

    return config
