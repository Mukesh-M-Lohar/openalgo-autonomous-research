import json
import os
import random
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUTORESEARCH_DIR = "/root/openalgo-autonomous-research/autoresearch"
PARAMS_FILE = os.path.join(AUTORESEARCH_DIR, "params.json")
METRICS_FILE = os.path.join(AUTORESEARCH_DIR, "metrics.json")
RESULTS_FILE = os.path.join(AUTORESEARCH_DIR, "results.tsv")
BEST_STRATEGY_FILE = os.path.join(AUTORESEARCH_DIR, "best_strategy.json")
PLOT_PATH = os.path.join(AUTORESEARCH_DIR, "best_equity_curve.png")


def propose_parameters(best_params=None):
    if best_params is None:
        # Starting config with highest probability setup: ORB + VWAP
        return {
            "strategy_name": "orb_vwap",
            "rsi_period": 14,
            "rsi_lower": 35,
            "ema_trend": 100,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "st_period": 10,
            "st_mult": 3.0,
            "tp": 0.02,
            "sl": 0.01,
            "allocation": 2.0,
            "use_atr_stop": True,
            "atr_mult": 2.5,
        }

    new_params = best_params.copy()

    # Mutate strategy type (focusing heavily on high-probability index setups: ORB, VWAP Pullback, Supertrend Pullback)
    if random.random() < 0.25:
        new_params["strategy_name"] = random.choice(
            ["orb_vwap", "vwap_pullback", "supertrend_ema_pullback", "bb_reversion", "macd_trend"]
        )

    mutation_choice = random.choice(
        [
            "rsi_lower",
            "tp",
            "sl",
            "allocation",
            "use_atr_stop",
            "atr_mult",
            "macd_fast_slow",
            "bb_std",
            "st_params",
        ]
    )

    if mutation_choice == "rsi_lower":
        new_params["rsi_lower"] = random.choice([25, 30, 35, 40])
    elif mutation_choice == "tp":
        new_params["tp"] = round(random.choice([0.005, 0.01, 0.015, 0.02, 0.025, 0.03]), 3)
    elif mutation_choice == "sl":
        new_params["sl"] = round(random.choice([0.005, 0.01, 0.015, 0.02]), 3)
    elif mutation_choice == "allocation":
        new_params["allocation"] = round(random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0]), 1)
    elif mutation_choice == "use_atr_stop":
        new_params["use_atr_stop"] = not new_params["use_atr_stop"]
    elif mutation_choice == "atr_mult":
        new_params["atr_mult"] = round(random.choice([1.5, 2.0, 2.5, 3.0]), 1)
    elif mutation_choice == "st_params":
        new_params["st_period"] = random.choice([7, 10, 14])
        new_params["st_mult"] = round(random.choice([1.5, 2.0, 2.5, 3.0, 4.0]), 1)
    elif mutation_choice == "macd_fast_slow":
        fast = random.choice([8, 12, 15])
        new_params["macd_fast"] = fast
        new_params["macd_slow"] = fast + random.choice([10, 14, 18])
    elif mutation_choice == "bb_std":
        new_params["bb_std"] = round(random.choice([1.5, 2.0, 2.5]), 1)

    return new_params


def run_loop(iterations=10):
    print("=" * 80)
    print(f"      OPENALGO INDEX OPTIONS AUTORESEARCH LOOP ({iterations} iterations)")
    print("=" * 80)

    best_params = None
    best_score = -float("inf")

    # Load previous best
    if os.path.exists(BEST_STRATEGY_FILE):
        try:
            with open(BEST_STRATEGY_FILE, "r") as f:
                best_data = json.load(f)
                best_params = best_data["params"]
                best_score = best_data["score"]
                print(f"Loaded previous best from disk: Score = {best_score:.4f}")
        except Exception:
            pass

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(
                "timestamp\tstrategy\tparameters\tsharpe\tdaily_return_pct\tmax_dd_pct\ttrades\tscore\tstatus\n"
            )

    for step in range(1, iterations + 1):
        print(f"\n[Step {step}/{iterations}] Mutating parameters...")
        proposed = propose_parameters(best_params)
        print(
            f"  Proposed: {proposed['strategy_name']} | TP={proposed['tp'] * 100:.1f}%, SL={proposed['sl'] * 100:.1f}%, Alloc={proposed['allocation']}x, ATR_Stop={proposed['use_atr_stop']}"
        )

        # Write to params.json
        with open(PARAMS_FILE, "w") as f:
            json.dump(proposed, f, indent=4)

        # Execute train.py
        try:
            subprocess.run(["python3", "train.py"], cwd=AUTORESEARCH_DIR, check=True)
        except Exception as e:
            print(f"  Execution failed: {e}")
            continue

        # Read metrics.json
        if not os.path.exists(METRICS_FILE):
            print("  Error: metrics.json was not generated.")
            continue

        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

        sharpe = metrics["sharpe"]
        daily_ret = metrics["daily_return_avg_pct"]
        max_dd = metrics["max_dd_pct"]
        trades = metrics["total_trades"]

        # Evaluate fitness: maximize daily return and Sharpe, penalize large drawdown
        score = daily_ret * 10.0 + sharpe * 0.5
        if max_dd < -20.0:
            score -= 5.0

        status = "REJECTED"
        if score > best_score:
            best_score = score
            best_params = proposed
            status = "ACCEPTED"
            print(
                f"  🔥 NEW BEST! Score: {score:.4f} | Sharpe: {sharpe:.2f} | Daily Ret: {daily_ret:.4f}% | Max DD: {max_dd:.2f}%"
            )

            # Save best strategy
            best_meta = {
                "score": best_score,
                "params": best_params,
                "metrics": {
                    "sharpe": sharpe,
                    "daily_return_avg_pct": daily_ret,
                    "max_dd_pct": max_dd,
                    "total_trades": trades,
                },
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(BEST_STRATEGY_FILE, "w") as f:
                json.dump(best_meta, f, indent=4)

            # Plot equity curve
            if "equity_curve" in metrics and "dates" in metrics:
                plt.figure(figsize=(10, 5))
                dates = pd.to_datetime(metrics["dates"])
                plt.plot(
                    dates,
                    100000.0 * (np.array(metrics["equity_curve"]) / metrics["equity_curve"][0]),
                    color="#22c55e",
                    lw=2,
                    label=f"Best Portfolio ({best_params['strategy_name'].upper()})",
                )
                plt.title(
                    f"Best Autonomous Portfolio Equity (Daily Return: {daily_ret:.3f}%)",
                    fontweight="bold",
                )
                plt.xlabel("Date")
                plt.ylabel("Equity (Base 100,000)")
                plt.grid(True, linestyle=":", alpha=0.6)
                plt.legend(loc="upper left")
                plt.tight_layout()
                plt.savefig(PLOT_PATH, dpi=300)
                plt.close()
        else:
            print(f"  Rejected: Score {score:.4f} <= Best Score {best_score:.4f}")

        # Log to results.tsv
        with open(RESULTS_FILE, "a") as f:
            param_str = json.dumps(proposed).replace("\t", " ")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"{ts}\t{proposed['strategy_name']}\t{param_str}\t{sharpe:.4f}\t{daily_ret:.4f}\t{max_dd:.4f}\t{trades}\t{score:.4f}\t{status}\n"
            )

    print("\n" + "=" * 80)
    print("      RESEARCH CYCLE COMPLETED!")
    print(f"      Winning configuration saved to: {BEST_STRATEGY_FILE}")
    print(f"      Full mutation logs written to: {RESULTS_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    run_loop(30)
