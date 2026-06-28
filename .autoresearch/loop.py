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

# Recursive watcher protection
ignore_changes = False

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


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
    global ignore_changes
    ignore_changes = True
    try:
        if STRATEGY_BACKUP.exists():
            shutil.copy2(STRATEGY_BACKUP, STRATEGY_FILE)
            print("  ↩  Reverted strategy.py to last backup")
    finally:
        time.sleep(0.1)
        ignore_changes = False


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
    parser.add_argument("--symbol", default="SBIN", help="Symbol(s), comma-separated")
    parser.add_argument("--exchange", default="NSE", help="Exchange")
    parser.add_argument("--tf", default="D", help="Timeframe(s), comma-separated")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2026-06-25", help="End date")
    parser.add_argument("--max-iters", type=int, default=0, help="0=unlimited")
    parser.add_argument("--baseline", action="store_true", help="Run baseline only")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds to wait between polls if watchdog not used (default: 2)",
    )
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print("  AUTORESEARCH BACKTEST LOOP")
    print(f"  Symbols: {args.symbol}  Exchange: {args.exchange}  TFs: {args.tf}")
    print(f"  Period : {args.start} -> {args.end}")
    print(f"{'═' * 60}")
    print(f"  Log    : {LOG_FILE}")
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

    # Watch logic state
    state = {
        "best_score": best_score,
        "best_iteration": best_iteration,
        "iteration": 0,
        "last_mtime": STRATEGY_FILE.stat().st_mtime if STRATEGY_FILE.exists() else 0,
    }

    def evaluate_change():
        global ignore_changes
        if ignore_changes:
            return

        state["iteration"] += 1
        iter_num = state["iteration"]
        print(f"\n  Detected strategy.py change — running evaluation #{iter_num}...")
        backup_strategy()

        new_score, new_metrics = run_prepare(
            args.symbol, args.exchange, args.tf, args.start, args.end
        )

        if new_score > state["best_score"]:
            delta = new_score - state["best_score"]
            state["best_score"] = new_score
            state["best_iteration"] = iter_num
            log_iteration(iter_num, new_score, new_metrics, note="improved")
            print_bar(iter_num, new_score, new_score, new_metrics)
            print(f"  Best so far: {new_score:.4f} (iteration {iter_num})")
        else:
            log_iteration(iter_num, new_score, new_metrics, note="reverted")
            print_bar(iter_num, new_score, state["best_score"], new_metrics)
            restore_strategy()

        if args.max_iters > 0 and iter_num >= args.max_iters:
            print(f"\n  Reached max iterations ({args.max_iters}). Stopping.")
            sys.exit(0)

    # ── Watch Mode Selection ───────────────────────────────────────────────────
    if HAS_WATCHDOG:
        print("\n  [Watchdog] Active - watching strategy.py for instant changes...")
        print("  The AI agent should now modify .autoresearch/strategy.py\n")

        class StrategyHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if ignore_changes:
                    return
                # Only trigger on target strategy.py file
                if Path(event.src_path).resolve() == STRATEGY_FILE.resolve():
                    # Debounce duplicate filesystem events
                    time.sleep(0.05)
                    evaluate_change()

        event_handler = StrategyHandler()
        observer = Observer()
        observer.schedule(event_handler, path=str(AUTORESEARCH_DIR), recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            observer.stop()
            observer.join()
    else:
        print(f"\n  [Polling] Active - polling strategy.py every {args.interval}s...")
        print("  The AI agent should now modify .autoresearch/strategy.py\n")

        try:
            while True:
                time.sleep(args.interval)

                if not STRATEGY_FILE.exists():
                    continue

                mtime = STRATEGY_FILE.stat().st_mtime
                if mtime <= state["last_mtime"]:
                    continue  # no change

                state["last_mtime"] = mtime
                evaluate_change()

        except KeyboardInterrupt:
            pass

    print("\n\n  Loop stopped.")
    print(
        f"  Best score: {state['best_score']:.4f} achieved at iteration {state['best_iteration']}"
    )
    print(f"  Full log: {LOG_FILE}")


if __name__ == "__main__":
    main()
