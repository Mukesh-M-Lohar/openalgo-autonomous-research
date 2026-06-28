"""
loop.py — Autonomous autoresearch loop runner.

This script implements the Karpathy autoresearch loop for backtesting.
It calls prepare.py in a subprocess, reads the SCORE, and logs results.

The agent (AI) does NOT run this file — this is a utility you run manually
to watch/log iterations. The agent reads program.md for instructions.

Usage:
    python .autoresearch/loop.py                     # watch mode (prints scores)
    python .autoresearch/loop.py --max-iters 20     # stop after 20 iterations
    python .autoresearch/loop.py --baseline          # just run baseline once
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTORESEARCH_DIR = Path(__file__).parent
LOG_FILE = AUTORESEARCH_DIR / "experiment_log.jsonl"
STRATEGY_FILE = AUTORESEARCH_DIR / "strategy.py"
STRATEGY_BACKUP = AUTORESEARCH_DIR / "strategy.py.bak"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def run_prepare(
    symbol: str,
    exchange: str,
    tf: str,
    start: str,
    end: str,
    timeout: int = 120,
) -> tuple[float, dict]:
    """
    Run prepare.py as a subprocess and parse the SCORE from stdout.
    Returns (score, metrics_dict).
    """
    cmd = [
        sys.executable,
        str(AUTORESEARCH_DIR / "prepare.py"),
        "--symbol",
        symbol,
        "--exchange",
        exchange,
        "--tf",
        tf,
        "--start",
        start,
        "--end",
        end,
        "--json",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )
        output = result.stdout.strip()

        # Parse SCORE from last line
        score = 0.0
        for line in reversed(output.splitlines()):
            if line.startswith("SCORE:"):
                score = float(line.split(":", 1)[1].strip())
                break

        # Parse JSON metrics (second-to-last block)
        metrics = {}
        try:
            # find the JSON blob
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                metrics = json.loads(output[json_start:json_end])
        except json.JSONDecodeError:
            pass

        if result.returncode != 0 and not metrics:
            print(f"  [stderr] {result.stderr.strip()[:300]}", file=sys.stderr)

        return score, metrics

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] prepare.py exceeded {timeout}s", file=sys.stderr)
        return 0.0, {"error": "timeout"}
    except Exception as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        return 0.0, {"error": str(e)}


def log_iteration(iteration: int, score: float, metrics: dict, note: str = ""):
    """Append one iteration result to the JSONL log."""
    entry = {
        "iteration": iteration,
        "timestamp": datetime.utcnow().isoformat(),
        "score": score,
        "note": note,
        **metrics,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def backup_strategy():
    """Save current strategy.py to strategy.py.bak."""
    shutil.copy2(STRATEGY_FILE, STRATEGY_BACKUP)


def restore_strategy():
    """Restore strategy.py from the last backup."""
    if STRATEGY_BACKUP.exists():
        shutil.copy2(STRATEGY_BACKUP, STRATEGY_FILE)
        print("  ↩  Reverted strategy.py to last backup")


def print_bar(iteration: int, score: float, best: float, metrics: dict):
    trades = metrics.get("trades", "?")
    sharpe = metrics.get("sharpe", "?")
    dd = metrics.get("max_drawdown_pct", "?")
    delta = score - best
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
    tag = "★ NEW BEST" if delta > 0 else ""
    print(
        f"  [{iteration:>3}] score={score:.4f}  {arrow}{abs(delta):.4f}  "
        f"| trades={trades}  sharpe={sharpe}  dd={dd}%  {tag}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Autoresearch loop watcher")
    parser.add_argument("--symbol", default="SBIN", help="Symbol")
    parser.add_argument("--exchange", default="NSE", help="Exchange")
    parser.add_argument("--tf", default="D", help="Timeframe")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--max-iters", type=int, default=0, help="0=unlimited")
    parser.add_argument("--baseline", action="store_true", help="Run baseline only")
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Seconds to wait between polls (default: 5)"
    )
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print("  AUTORESEARCH BACKTEST LOOP")
    print(f"  Symbol: {args.symbol}/{args.exchange}  TF: {args.tf}")
    print(f"  Period: {args.start} -> {args.end}")
    print(f"{'═' * 60}")
    print(f"  Log  : {LOG_FILE}")
    print(f"{'─' * 60}")

    # ── Baseline ────────────────────────────────────────────────────────────────
    print("  Running baseline evaluation...")
    score, metrics = run_prepare(args.symbol, args.exchange, args.tf, args.start, args.end)
    best_score = score
    best_iteration = 0
    log_iteration(0, score, metrics, note="baseline")
    print_bar(0, score, 0.0, metrics)

    if args.baseline:
        print(f"\n  Baseline score: {score}")
        return

    # ── Watch loop ──────────────────────────────────────────────────────────────
    print(f"\n  Watching for strategy.py changes every {args.interval}s...")
    print("  The AI agent should now modify .autoresearch/strategy.py")
    print("  This loop will score each version automatically.\n")

    last_mtime = STRATEGY_FILE.stat().st_mtime if STRATEGY_FILE.exists() else 0
    iteration = 0

    try:
        while True:
            time.sleep(args.interval)

            if not STRATEGY_FILE.exists():
                continue

            mtime = STRATEGY_FILE.stat().st_mtime
            if mtime <= last_mtime:
                continue  # no change

            iteration += 1
            last_mtime = mtime
            print(f"\n  Detected strategy.py change — running evaluation #{iteration}...")
            backup_strategy()

            score, metrics = run_prepare(args.symbol, args.exchange, args.tf, args.start, args.end)

            if score > best_score:
                best_score = score
                best_iteration = iteration
                log_iteration(iteration, score, metrics, note="improved")
                print_bar(iteration, score, best_score, metrics)
                print(f"  Best so far: {best_score:.4f} (iteration {best_iteration})")
            else:
                log_iteration(iteration, score, metrics, note="reverted")
                print_bar(iteration, score, best_score, metrics)
                restore_strategy()

            if args.max_iters > 0 and iteration >= args.max_iters:
                print(f"\n  Reached max iterations ({args.max_iters}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n\n  Loop stopped by user.")
        print(f"  Best score: {best_score:.4f} achieved at iteration {best_iteration}")
        print(f"  Full log: {LOG_FILE}")


if __name__ == "__main__":
    main()
