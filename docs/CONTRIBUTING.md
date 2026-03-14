# Contributing to MLShield

## Development Setup

### Prerequisites

- Python 3.10+
- Redis (for event bus, optional for testing)
- NVIDIA GPU + DCGM (optional, for GPU metric ingestion)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mlshield.git
cd mlshield

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_cascade.py

# Run with verbose output
pytest -v
```

### Running Locally

```bash
# Start with Docker Compose (recommended)
docker compose up --build

# Or run directly
PYTHONPATH=src python -m uvicorn mlshield.api.app:app --reload --port 8000
```

## Project Structure

```
src/mlshield/
  ingestion/     # Data source connectors
  specs/         # Behavioral specification engine
  detectors/     # 3-layer cascaded detector
  metrics/       # Temporal security metrics
  api/           # FastAPI server + dashboard
  utils/         # Config, logging
benchmark/       # Synthetic dataset + evaluation
configs/         # YAML specs and rules
tests/           # Test suite
deploy/          # K8s manifests + Helm chart
```

## Code Style

- **Formatter**: Black (line length 88)
- **Linter**: Ruff
- **Type checking**: mypy

```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Adding a New Detection Rule

1. Add the rule logic to `src/mlshield/detectors/layer1_rules.py`
2. Add corresponding spec entries in `configs/rules/`
3. Add a test scenario in `benchmark/scenarios/`
4. Write tests in `tests/test_detectors.py`

## Adding a New Data Source

1. Create a new ingester in `src/mlshield/ingestion/` inheriting from `BaseIngester`
2. Implement the `ingest()` method that yields `TrajectoryEvent` objects
3. Add the new `EventSource` enum value in `event_bus.py`
4. Wire it into the API startup in `src/mlshield/api/app.py`
5. Write tests in `tests/test_ingestion.py`

## Adding a New Attack Scenario

1. Create a new scenario generator in `benchmark/scenarios/`
2. Follow the pattern: generate normal trajectory, then inject malicious events
3. Include `is_malicious: True` and `violation_type` in malicious event details
4. Add the scenario to `benchmark/generate_dataset.py`
5. Regenerate the benchmark dataset and re-run evaluations

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass: `pytest`
4. Format code: `black src/ tests/`
5. Submit a PR with a clear description of changes

## Reporting Issues

Please include:
- MLShield version
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output
