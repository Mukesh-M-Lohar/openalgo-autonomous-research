#!/usr/bin/env bash
set -euo pipefail

echo "=== OpenAlgo Quant Research Engine — Dev Setup ==="
echo ""

# Check Python version
PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" &> /dev/null; then
    PYTHON=python
fi

PY_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

echo "Python: $PYTHON ($PY_VERSION)"

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]); then
    echo "ERROR: Python 3.11+ required. Found $PY_VERSION"
    exit 1
fi

# Create virtual environment
echo ""
echo "1. Creating virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    echo "   Created .venv/"
else
    echo "   .venv/ already exists, skipping"
fi

# Activate
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

# Install dependencies
echo ""
echo "2. Installing dependencies..."
pip install --upgrade pip -q
pip install -e ".[dev]" 2>/dev/null || pip install \
    numpy pandas pydantic pydantic-settings pyyaml \
    fastapi uvicorn typer jinja2 httpx \
    pytest pytest-cov ruff -q

echo "   Done."

# Verify
echo ""
echo "3. Verifying installation..."
$PYTHON -c "import quant_engine; print(f'   quant_engine v{quant_engine.__version__}')"

# Run tests
echo ""
echo "4. Running tests..."
PYTHONPATH=src $PYTHON -m pytest tests/ -q --tb=line

# Create data directories
echo ""
echo "5. Creating data directories..."
mkdir -p data/cache data/runs data/exports
echo "   Created data/{cache,runs,exports}/"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate your environment:"
echo "  source .venv/bin/activate    (Linux/macOS)"
echo "  .venv\\Scripts\\activate       (Windows)"
echo ""
echo "Quick commands:"
echo "  make test          — Run tests"
echo "  make lint          — Check code style"
echo "  make serve-dev     — Start API server (dev mode)"
echo "  make run-default   — Run with default config"
