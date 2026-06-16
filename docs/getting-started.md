# Getting Started

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | 3.12 recommended |
| OpenAlgo | Latest | For data retrieval only |
| Make | Any | Optional (Windows: `choco install make`) |

## Installation

### Option 1: Using setup script (recommended)

```bash
# Clone
git clone <repo-url>
cd openalgo-quant-engine

# Linux/macOS
./scripts/setup-dev.sh

# Windows PowerShell
.\scripts\setup-dev.ps1
```

### Option 2: Using Make

```bash
make setup
```

### Option 3: Manual

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install numpy pandas pydantic pydantic-settings pyyaml \
    fastapi uvicorn typer jinja2 httpx pytest pytest-cov ruff
```

## Verify Installation

```bash
# Check import
PYTHONPATH=src python -c "import quant_engine; print(quant_engine.__version__)"

# Run tests
make test
```

## Configure OpenAlgo Connection

Set your API key as an environment variable:

```bash
# Linux/macOS
export OPENALGO_API_KEY="your-key-here"

# Windows
set OPENALGO_API_KEY=your-key-here
```

Or set it directly in your YAML config:

```yaml
data:
  openalgo:
    api_key: "your-key-here"
```

## Next Steps

- [Quick Start Guide](quick-start.md) — Run your first research
- [Configuration Guide](guides/configuration.md) — Customize settings
- [Developer Guide](developer-guide.md) — Contribute to the engine
