#!/usr/bin/env python3
"""
CLI entrypoint script to compile quantitative research runs and update the interactive dashboard.
"""

import sys
from pathlib import Path

# Add src/ to python path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_engine.storage.dashboard import generate_all_dashboards


def main():
    runs_dir = Path(__file__).parent.parent / "data" / "runs"
    generate_all_dashboards(runs_dir)


if __name__ == "__main__":
    main()
