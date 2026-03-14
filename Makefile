.PHONY: help test lint format typecheck serve demo benchmark train docker docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test: ## Run all tests
	PYTHONPATH=src python -m pytest tests/ -v

test-fast: ## Run tests without verbose output
	PYTHONPATH=src python -m pytest tests/ -q

lint: ## Run linter (ruff)
	ruff check src/ tests/ benchmark/

format: ## Format code (black + ruff)
	black src/ tests/ benchmark/
	ruff check --fix src/ tests/ benchmark/

typecheck: ## Run type checker (mypy)
	PYTHONPATH=src mypy src/mlshield/

serve: ## Start the API server locally
	PYTHONPATH=src python -m uvicorn mlshield.api.app:app --reload --host 0.0.0.0 --port 8000

demo: ## Run the attack detection demo
	PYTHONPATH=src python demo.py

benchmark: ## Generate the synthetic benchmark dataset
	PYTHONPATH=. python benchmark/generate_dataset.py

train: ## Train LSTM + Isolation Forest models
	PYTHONPATH=. python benchmark/train_lstm.py

evaluate: ## Run cascade evaluation on benchmark
	PYTHONPATH=. python benchmark/evaluate_cascade.py

docker: ## Build Docker image
	docker build -t mlshield:latest .

docker-up: ## Start full stack with Docker Compose
	docker compose up --build -d

docker-down: ## Stop Docker Compose stack
	docker compose down

docker-test: ## Run integration tests against Docker stack
	docker compose up --build -d
	sleep 5
	bash tests/integration/test_docker.sh
	docker compose down

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
