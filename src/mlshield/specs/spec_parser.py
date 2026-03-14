# src/mlshield/specs/spec_parser.py
import yaml
from pathlib import Path
from typing import Optional


class SpecParser:
    """Parses YAML behavioral specification files."""

    def __init__(self, spec_path: str = "configs/default_specs.yaml"):
        self.spec_path = Path(spec_path)
        self.specs: dict = {}
        self._load()

    def _load(self) -> None:
        """Load specs from YAML file."""
        if not self.spec_path.exists():
            return
        with open(self.spec_path) as f:
            config = yaml.safe_load(f)
        self.specs = {s["name"]: s for s in config.get("specs", [])}

    def get_spec(self, name: str) -> Optional[dict]:
        """Get a spec by name."""
        return self.specs.get(name)

    def list_specs(self) -> list[str]:
        """List all available spec names."""
        return list(self.specs.keys())

    def match_spec(
        self, job_type: str, labels: dict = None, namespace: str = ""
    ) -> Optional[dict]:
        """Find the best matching spec for a job."""
        for spec in self.specs.values():
            match = spec.get("match", {})

            # Check job type
            if spec.get("job_type") and spec["job_type"] != job_type:
                continue

            # Check namespace pattern
            ns_pattern = match.get("namespace_pattern", "")
            if ns_pattern and namespace:
                import re

                regex = ns_pattern.replace("*", ".*")
                if not re.match(regex, namespace):
                    continue

            # Check labels
            match_labels = match.get("labels", {})
            if match_labels and labels:
                if not all(labels.get(k) == v for k, v in match_labels.items()):
                    continue

            return spec

        return None
