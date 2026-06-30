.PHONY: help setup test lint format serve-dev serve run-default run-example docs clean build

PYTHON ?= .venv/bin/python
PYTHONPATH := src
export PYTHONPATH

help: ## Show this help
	@echo "OpenAlgo Quant Research Engine — Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# === Setup ===

setup: ## Full dev environment setup
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip || .venv/Scripts/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]" || .venv/Scripts/pip install numpy pandas pydantic pydantic-settings pyyaml fastapi uvicorn typer jinja2 httpx pytest pytest-cov ruff
	mkdir -p data/cache data/runs data/exports
	@echo "Setup complete. Activate: source .venv/bin/activate"

install: ## Install dependencies only
	pip install numpy pandas pydantic pydantic-settings pyyaml fastapi uvicorn typer jinja2 httpx pytest pytest-cov ruff

# === Testing ===

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-quick: ## Run tests without verbose output
	$(PYTHON) -m pytest tests/ -q

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=src/quant_engine --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-generation: ## Run generation tests only
	$(PYTHON) -m pytest tests/test_generation/ -v

test-backtest: ## Run backtest tests only
	$(PYTHON) -m pytest tests/test_backtest/ -v

test-validation: ## Run validation tests only
	$(PYTHON) -m pytest tests/test_validation/ -v

test-pipeline: ## Run integration tests only
	$(PYTHON) -m pytest tests/test_pipeline/ -v

# === Linting ===

lint: ## Check code style (ruff)
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code (ruff)
	ruff check --fix src/ tests/
	ruff format src/ tests/

# === Running ===

serve: ## Start API server (production)
	$(PYTHON) -m quant_engine serve --host 0.0.0.0 --port 8000

serve-dev: ## Start API server (dev mode with reload)
	$(PYTHON) -m quant_engine serve --host 127.0.0.1 --port 8000 --debug

run-default: ## Run research with default config
	$(PYTHON) -m quant_engine run config/default_research.yaml --debug

run-example: ## Run research with example config
	$(PYTHON) -m quant_engine run config/example_research.yaml --debug

# Support running with direct argument: make run config/example_research.yaml or make run example_research.yaml
ifeq ($(firstword $(MAKECMDGOALS)),run)
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

run: ## Run research with provided config
	@CONFIG_FILE="$(RUN_ARGS)"; \
	if [ -z "$$CONFIG_FILE" ]; then \
		CONFIG_FILE="$(config)"; \
	fi; \
	if [ -z "$$CONFIG_FILE" ]; then \
		echo "Usage: make run <config-name-or-path.yaml>"; \
		echo "Example: make run sbin_intraday_ml.yaml"; \
		exit 1; \
	fi; \
	if [ ! -f "$$CONFIG_FILE" ] && [ -f "config/$$CONFIG_FILE" ]; then \
		CONFIG_FILE="config/$$CONFIG_FILE"; \
	fi; \
	$(PYTHON) -m quant_engine run $$CONFIG_FILE

list-runs: ## List all completed research runs
	$(PYTHON) -m quant_engine list

# === Build ===

build: ## Build distributable package
	$(PYTHON) -m build

check-build: ## Verify package builds correctly
	$(PYTHON) -m build
	pip install dist/*.whl --force-reinstall
	$(PYTHON) -c "import quant_engine; print(f'v{quant_engine.__version__}')"

# === Docs ===

docs: ## Build documentation site (requires mkdocs)
	$(PYTHON) -m pip install mkdocs mkdocs-material -q
	$(PYTHON) scripts/generate_dashboard.py
	.venv/bin/mkdocs build

docs-serve: ## Serve docs locally with hot reload
	$(PYTHON) -m pip install mkdocs mkdocs-material -q
	.venv/bin/mkdocs serve

# === Cleanup ===

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache htmlcov .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

clean-data: ## Remove all research run data (DESTRUCTIVE)
	rm -rf data/runs/* data/cache/* data/exports/*
	@echo "Data directories cleared."

# === Bot Generation ===

generate-bot: ## Generate a bot script (e.g. make generate-bot symbol=SBIN exchange=NSE interval=5m strategy=MyStrategy whatsapp=919876543210)
	@SYMBOL="$(symbol)"; \
	if [ -z "$$SYMBOL" ]; then SYMBOL="$(symbols)"; fi; \
	if [ -z "$$SYMBOL" ]; then SYMBOL="$(Symbols)"; fi; \
	if [ -z "$$SYMBOL" ]; then \
		echo "Usage: make generate-bot symbol=<SYMBOL> [exchange=<EXCHANGE>] [interval=<INTERVAL>] [strategy=<STRATEGY>] [whatsapp=<NUMBERS>]"; \
		exit 1; \
	fi; \
	EXCHANGE="$(exchange)"; \
	if [ -z "$$EXCHANGE" ]; then EXCHANGE="$(Exchange)"; fi; \
	INTERVAL="$(interval)"; \
	if [ -z "$$INTERVAL" ]; then INTERVAL="$(Interval)"; fi; \
	STRATEGY="$(strategy)"; \
	if [ -z "$$STRATEGY" ]; then STRATEGY="$(strategy_name)"; fi; \
	if [ -z "$$STRATEGY" ]; then STRATEGY="$(stragery)"; fi; \
	WHATSAPP="$(whatsapp)"; \
	if [ -z "$$WHATSAPP" ]; then WHATSAPP="$(whatsapp_numbers)"; fi; \
	if [ -z "$$WHATSAPP" ]; then WHATSAPP="$(whatsapp_number)"; fi; \
	if [ -z "$$WHATSAPP" ]; then WHATSAPP="$(whatsapp_nums)"; fi; \
	ARGS="$$SYMBOL"; \
	if [ -n "$$EXCHANGE" ]; then ARGS="$$ARGS --exchange $$EXCHANGE"; fi; \
	if [ -n "$$INTERVAL" ]; then ARGS="$$ARGS --interval $$INTERVAL"; fi; \
	if [ -n "$$STRATEGY" ]; then ARGS="$$ARGS --strategy $$STRATEGY"; fi; \
	if [ -n "$$WHATSAPP" ]; then ARGS="$$ARGS --whatsapp $$WHATSAPP"; fi; \
	$(PYTHON) scripts/generate_bot.py $$ARGS
