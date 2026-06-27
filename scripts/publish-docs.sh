#!/usr/bin/env bash
set -euo pipefail

echo "=== Publishing docs to GitHub Pages ==="
echo ""

# Install mkdocs if needed
if ! command -v mkdocs &> /dev/null; then
    echo "Installing mkdocs..."
    pip install mkdocs mkdocs-material pymdown-extensions -q
fi

# Build
echo "Building documentation..."
python scripts/generate_dashboard.py
mkdocs build --strict

echo ""
echo "Docs built to site/"
echo ""
echo "To deploy:"
echo "  Option 1: Push to main (GitHub Actions auto-deploys)"
echo "  Option 2: mkdocs gh-deploy --force"
echo ""

read -p "Deploy now with gh-deploy? (y/N) " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    mkdocs gh-deploy --force
    echo "Deployed!"
fi
