# Developer Guide

## Prerequisites

- Python 3.11+ (3.12 recommended)
- Git
- Make (optional, see below)
  - **Windows:** `choco install make` or `winget install GnuWin32.Make`
  - **Ubuntu/Debian:** `sudo apt install make`
  - **macOS:** Already installed with Xcode CLI tools

## Development Setup

### Quick Start (one command)

```bash
# Windows (PowerShell)
.\scripts\setup-dev.ps1

# Linux/macOS
./scripts/setup-dev.sh

# Or using Make
make setup
```

### Manual Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd openalgo-quant-engine

# 2. Create virtual environment
python -m venv .venv

# Activate (choose your OS)
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows CMD
.venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Verify installation
python -c "import quant_engine; print(quant_engine.__version__)"
python -m pytest tests/ -v
```

## Project Layout

```
openalgo-quant-engine/
├── src/quant_engine/          # Source code (installed as package)
│   ├── models/                # Data models (StrategyGenome, Results)
│   ├── data/                  # OpenAlgo API client + cache
│   ├── generation/            # Strategy generation (grammar, indicators, patterns)
│   ├── backtest/              # Backtest engine + metrics
│   ├── validation/            # Walk-forward, OOS, Monte Carlo, stress
│   ├── evolution/             # Mutation, crossover, population
│   ├── ranking/               # Pareto, weighted scoring, portfolios
│   ├── export/                # Signal script generation (Jinja2)
│   ├── pipeline/              # Orchestrator (connects all stages)
│   ├── storage/               # CSV backend (protocol-based)
│   ├── parallel/              # Process pool + progress tracking
│   └── api/                   # FastAPI routes + schemas
├── tests/                     # pytest test suite
├── config/                    # YAML research configs
├── docs/                      # Documentation
├── scripts/                   # Dev setup + utility scripts
├── .github/workflows/         # CI/CD pipelines
└── data/                      # Runtime output (gitignored)
```

## Development Workflow

### Running Tests

```bash
# All tests
make test

# Or directly
PYTHONPATH=src python -m pytest tests/ -v

# Specific module
PYTHONPATH=src python -m pytest tests/test_generation/ -v

# With coverage
PYTHONPATH=src python -m pytest tests/ --cov=src/quant_engine --cov-report=html
open htmlcov/index.html
```

### Linting

```bash
# Check
make lint

# Auto-fix
make format
```

### Running Locally

```bash
# CLI mode
make run-default

# API server (dev mode with hot reload)
make serve-dev

# Run with custom config
PYTHONPATH=src python -m quant_engine run config/example_research.yaml --debug
```

### Building Documentation

```bash
make docs

# Preview locally
cd docs/_site && python -m http.server 8080
```

## Adding a New Indicator

1. Add the enum value to `models/strategy.py`:
```python
class IndicatorType(str, Enum):
    ...
    MY_INDICATOR = "my_indicator"
```

2. Implement the computation in `generation/indicators.py`:
```python
def _my_indicator(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 14))
    # Your computation here
    return result
```

3. Register it:
```python
INDICATOR_FUNCTIONS[IndicatorType.MY_INDICATOR] = _my_indicator

INDICATOR_PARAM_RANGES[IndicatorType.MY_INDICATOR] = {
    "period": (5, 50, 5),  # (min, max, step)
}
```

4. Add to a category:
```python
INDICATOR_CATEGORIES["momentum"].append(IndicatorType.MY_INDICATOR)
```

5. Write a test in `tests/test_generation/test_strategy_generation.py`:
```python
def test_my_indicator(self, sample_df):
    result = compute_indicator(sample_df, IndicatorType.MY_INDICATOR, {"period": 14}, PriceSource.CLOSE)
    assert len(result) == len(sample_df)
```

## Adding a New Pattern

1. Implement in `generation/patterns.py`:
```python
def _my_pattern(df: pd.DataFrame, params: dict) -> pd.Series:
    # Return boolean Series
    return (some_condition).fillna(False)
```

2. Register:
```python
PATTERN_FUNCTIONS["my_pattern"] = _my_pattern
PATTERN_PARAM_RANGES["my_pattern"] = {"param_name": (min, max, step)}
```

## Adding a New Validation Engine

1. Create `validation/my_validator.py`:
```python
class MyValidator:
    def validate(self, strategy, data) -> dict:
        # Return dict with score keys
        return {"my_score": 75.0, "my_detail": "..."}
```

2. Wire into `pipeline/orchestrator.py` at the appropriate stage.

3. Add rejection criteria to `config.py` if needed.

## Adding a New Storage Backend

1. Implement the `StorageBackend` protocol in `storage/my_backend.py`:
```python
class MyStorage:
    def init_run(self, run_id, config): ...
    def save_rejections(self, run_id, rejections): ...
    # ... all protocol methods
```

2. Add factory logic in config to select backend based on `output.storage` value.

## Code Style

- **Line length:** 100 characters
- **Formatter:** Ruff (auto-formatted)
- **Type hints:** Required for all public functions
- **Docstrings:** Only for non-obvious public classes/functions
- **Comments:** Only for WHY, never for WHAT
- **Imports:** Sorted by ruff (isort-compatible)

## Commit Message Convention

```
feat: add new indicator XYZ
fix: correct Sharpe calculation for edge case
refactor: extract common condition evaluation
test: add Monte Carlo edge case tests
docs: update developer guide
```

## CI Pipeline

The GitHub Actions pipeline runs on every push/PR:

1. **Lint** — Ruff check + format verification
2. **Test** — Unit tests on Python 3.11+3.12, Ubuntu+Windows
3. **Integration** — Full pipeline test with synthetic data
4. **API Smoke** — FastAPI endpoint verification
5. **Build** — Package builds and imports correctly

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENALGO_API_KEY` | OpenAlgo API authentication | (none) |
| `OPENALGO_HOST` | OpenAlgo server URL | `http://127.0.0.1:5000` |
| `QUANT_ENGINE_WORKERS` | Override CPU auto-detection | `cpu_count - 1` |
| `QUANT_ENGINE_LOG_LEVEL` | Logging level | `INFO` |

## Debugging Tips

### Strategy produces no trades
- Check if indicators have enough warmup bars (need at least `period` bars)
- Check if the condition combination is too restrictive (AND of many conditions)
- Use `--debug` flag to see indicator values at signal points

### Backtest results differ from expected
- Verify cost model: commission + slippage applied twice (entry + exit)
- Check forced_exit_time for intraday strategies
- Look at trade_logs CSV for individual trade details

### Pipeline runs out of memory
- Reduce `generation.target_count`
- Increase `execution.chunk_size` (fewer concurrent items)
- Use `generate_batch()` instead of `generate()` for streaming

### Tests fail on Windows
- Ensure `spawn` multiprocessing context (already default)
- Path separators handled by `pathlib.Path`
- Timestamp parsing may differ — use `pd.to_datetime()` explicitly
