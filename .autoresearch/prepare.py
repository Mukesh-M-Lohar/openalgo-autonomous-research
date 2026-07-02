"""
prepare.py — READ-ONLY measurement tool for the autoresearch backtest loop.

This file is the EVALUATOR. The agent must NEVER modify this file.
It fetches OHLCV data, runs the strategy defined in strategy.py,
and returns a single objective score (higher = better).

Karpathy analogy:
  - prepare.py  ←→  evaluate_bpb()  — produces the number, never modified
  - strategy.py ←→  train.py        — what the agent mutates each iteration
  - program.md  ←→  loop instructions for the agent

Usage:
    python .autoresearch/prepare.py
    python .autoresearch/prepare.py --symbol SBIN --exchange NSE --tf D --start 2022-01-01 --end 2024-12-31
    python .autoresearch/prepare.py --json

Output (stdout, last line always):
    SCORE: <float>   ← the objective metric (composite score, higher = better)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Add src to path so quant_engine is importable ──────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_engine.backtest.metrics import compute_metrics  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("autoresearch.prepare")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA FETCHING  (read-only — never touched by agent)
# ══════════════════════════════════════════════════════════════════════════════


def fetch_data(symbol: str, exchange: str, tf: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data.

    Priority order:
      1. Local CSV cache  →  data/cache/{SYMBOL}_{EXCHANGE}_{TF}.csv
      2. OpenAlgo REST API (requires .env with OPENALGO_API_KEY & OPENALGO_HOST)
      3. Raise clear error if neither is available

    CSV format expected (with header):
        timestamp,open,high,low,close,volume
        2022-01-03,100,110,95,108,100000
    """

    # ── Try local CSV first ────────────────────────────────────────────────────
    tf_map = {"D": "D", "1d": "D", "15m": "15m", "5m": "5m", "1h": "1h", "60m": "1h"}
    tf_key = tf_map.get(tf, tf)

    csv_candidates = [
        ROOT / "data" / "cache" / f"{symbol}_{exchange}_{tf_key}.csv",
        ROOT / "data" / "cache" / f"{symbol}_{exchange}_{tf}.csv",
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.loc[start:end]
            if not df.empty:
                logger.info(f"Loaded {len(df)} bars from CSV: {csv_path}")
                return df

    # ── Try OpenAlgo API ───────────────────────────────────────────────────────
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")

        import os

        from quant_engine.config import OpenAlgoConfig
        from quant_engine.data.client import OpenAlgoClient

        cfg = OpenAlgoConfig(
            host=os.environ.get("OPENALGO_HOST", "http://127.0.0.1:5000"),
            api_key=os.environ.get("OPENALGO_API_KEY", ""),
            source=os.environ.get("OPENALGO_SOURCE", "db"),
        )
        with OpenAlgoClient(cfg, cache_dir=ROOT / "data" / "cache") as client:
            df = client.fetch_history(symbol, exchange, tf, start, end)
        if not df.empty:
            logger.info(f"Fetched {len(df)} bars via OpenAlgo API")
            return df
    except Exception as e:
        logger.warning(f"OpenAlgo API fetch failed: {e}")

    raise FileNotFoundError(
        f"No data found for {symbol}/{exchange}/{tf} [{start} -> {end}].\n"
        f"Tip: place a CSV at data/cache/{symbol}_{exchange}_{tf_key}.csv\n"
        f"     OR set OPENALGO_API_KEY + OPENALGO_HOST in .env"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCORING  (read-only — never touched by agent)
# ══════════════════════════════════════════════════════════════════════════════


def compute_score(result) -> float:
    """
    Composite objective score — higher is better.

    Formula (weights sum to 1.0):
        score = sharpe    * 0.35
              + sortino   * 0.20
              + calmar    * 0.15
              + pf_capped * 0.15
              + win_rate  * 0.10
              + cagr_norm * 0.05
              - dd_penalty

    All sub-scores are clipped to reasonable ranges before weighting to
    prevent any single outlier metric from dominating.

    Returns 0.0 if there are fewer than MIN_TRADES trades.
    """
    min_trades = 5

    if result is None or result.total_trades < min_trades:
        return 0.0

    sharpe = max(0.0, min(result.sharpe, 5.0))
    sortino = max(0.0, min(result.sortino, 7.0))
    calmar = max(0.0, min(result.calmar, 5.0))
    pf = max(0.0, min(result.profit_factor, 4.0))
    win_rate = max(0.0, min(result.win_rate, 1.0))
    cagr_norm = max(0.0, result.cagr / 100.0)  # % -> ratio

    # Penalty for drawdown > 25%
    dd_pct = result.max_drawdown_pct
    dd_penalty = max(0.0, (dd_pct - 25.0) / 100.0)

    score = (
        sharpe * 0.35
        + sortino * 0.20
        + calmar * 0.15
        + pf * 0.15
        + win_rate * 0.10
        + cagr_norm * 0.05
        - dd_penalty
    )
    return round(score, 6)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRADE SIMULATOR  (aligned with engine exits)
# ══════════════════════════════════════════════════════════════════════════════


def _simulate_trades(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_pct: float = 0.03,
    slippage_pct: float = 0.01,
    atr_series: pd.Series | None = None,
    atr_mult_sl: float = 0.0,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    trailing_stop_pct: float | None = None,
    max_hold_bars: int | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Simple bar-by-bar trade simulator supporting long/short entries, stops and exits.
    """
    trades = []
    equity = initial_capital
    equity_points = []

    position: str | None = None  # "long", "SELL", or None
    entry_price = 0.0
    entry_bar: int | None = None
    max_price_since_entry = 0.0
    min_price_since_entry = 0.0

    price_arr = price_df.reset_index()
    sig_aligned = signals_df.reindex(price_df.index, fill_value=False)

    for i in range(len(price_arr)):
        row = price_df.iloc[i]
        sig = sig_aligned.iloc[i]

        entry_sig = bool(sig.get("entry", False))
        exit_sig = bool(sig.get("exit", False))
        short_entry_sig = bool(sig.get("short_entry", False))
        short_exit_sig = bool(sig.get("short_exit", False))

        if position is not None:
            bars_held = i - entry_bar
            current_high = float(row["high"])
            current_low = float(row["low"])
            current_close = float(row["close"])

            exit_price_val = None
            exit_reason = ""

            if position == "long":
                max_price_since_entry = max(max_price_since_entry, current_high)

                # Check stop loss pct
                if stop_loss_pct is not None:
                    sl_price = entry_price * (1 - stop_loss_pct / 100)
                    if current_low <= sl_price:
                        exit_price_val = sl_price
                        exit_reason = "stop_loss"

                # Check ATR stop loss
                if exit_price_val is None and atr_mult_sl > 0 and atr_series is not None:
                    atr_val = atr_series.iloc[entry_bar]
                    sl_price = entry_price - atr_mult_sl * atr_val
                    if current_low <= sl_price:
                        exit_price_val = sl_price
                        exit_reason = "atr_stop_loss"

                # Check take profit pct
                if exit_price_val is None and take_profit_pct is not None:
                    tp_price = entry_price * (1 + take_profit_pct / 100)
                    if current_high >= tp_price:
                        exit_price_val = tp_price
                        exit_reason = "take_profit"

                # Check trailing stop
                if exit_price_val is None and trailing_stop_pct is not None:
                    trail_price = max_price_since_entry * (1 - trailing_stop_pct / 100)
                    if current_low <= trail_price:
                        exit_price_val = trail_price
                        exit_reason = "trailing_stop"

                # Check max hold bars
                if exit_price_val is None and max_hold_bars is not None:
                    if bars_held >= max_hold_bars:
                        exit_price_val = current_close
                        exit_reason = "max_hold"

                # Check standard exit signal or final bar
                if exit_price_val is None and (exit_sig or i == len(price_arr) - 1):
                    exit_price_val = float(row["open"]) * (1 - slippage_pct / 100)
                    exit_reason = "signal" if exit_sig else "end_of_data"

                if exit_price_val is not None:
                    if exit_reason in [
                        "stop_loss",
                        "atr_stop_loss",
                        "take_profit",
                        "trailing_stop",
                    ]:
                        exit_price_val = exit_price_val * (1 - slippage_pct / 100)

                    pnl_pct = (exit_price_val / entry_price - 1) * 100 - (
                        commission_pct + slippage_pct
                    ) * 2
                    pnl_abs = equity * pnl_pct / 100
                    equity = max(equity + pnl_abs, 0.0)
                    trades.append(
                        {
                            "entry_bar": entry_bar,
                            "exit_bar": i,
                            "bars_held": bars_held,
                            "pnl_pct": round(pnl_pct, 6),
                            "pnl_abs": round(pnl_abs, 2),
                            "entry_price": entry_price,
                            "exit_price": exit_price_val,
                            "exit_reason": exit_reason,
                            "direction": "long",
                        }
                    )
                    position = None

            elif position == "SELL":
                min_price_since_entry = min(min_price_since_entry, current_low)

                # Check stop loss pct (short stop loss is above entry)
                if stop_loss_pct is not None:
                    sl_price = entry_price * (1 + stop_loss_pct / 100)
                    if current_high >= sl_price:
                        exit_price_val = sl_price
                        exit_reason = "stop_loss"

                # Check ATR stop loss (short stop loss is above entry)
                if exit_price_val is None and atr_mult_sl > 0 and atr_series is not None:
                    atr_val = atr_series.iloc[entry_bar]
                    sl_price = entry_price + atr_mult_sl * atr_val
                    if current_high >= sl_price:
                        exit_price_val = sl_price
                        exit_reason = "atr_stop_loss"

                # Check take profit pct (short take profit is below entry)
                if exit_price_val is None and take_profit_pct is not None:
                    tp_price = entry_price * (1 - take_profit_pct / 100)
                    if current_low <= tp_price:
                        exit_price_val = tp_price
                        exit_reason = "take_profit"

                # Check trailing stop (short trailing stop is above min price since entry)
                if exit_price_val is None and trailing_stop_pct is not None:
                    trail_price = min_price_since_entry * (1 + trailing_stop_pct / 100)
                    if current_high >= trail_price:
                        exit_price_val = trail_price
                        exit_reason = "trailing_stop"

                # Check max hold bars
                if exit_price_val is None and max_hold_bars is not None:
                    if bars_held >= max_hold_bars:
                        exit_price_val = current_close
                        exit_reason = "max_hold"

                # Check standard exit signal or final bar
                if exit_price_val is None and (short_exit_sig or i == len(price_arr) - 1):
                    exit_price_val = float(row["open"]) * (1 + slippage_pct / 100)
                    exit_reason = "signal" if short_exit_sig else "end_of_data"

                if exit_price_val is not None:
                    if exit_reason in [
                        "stop_loss",
                        "atr_stop_loss",
                        "take_profit",
                        "trailing_stop",
                    ]:
                        exit_price_val = exit_price_val * (1 + slippage_pct / 100)

                    # Short profit formula: (1 - exit/entry)
                    pnl_pct = (1 - exit_price_val / entry_price) * 100 - (
                        commission_pct + slippage_pct
                    ) * 2
                    pnl_abs = equity * pnl_pct / 100
                    equity = max(equity + pnl_abs, 0.0)
                    trades.append(
                        {
                            "entry_bar": entry_bar,
                            "exit_bar": i,
                            "bars_held": bars_held,
                            "pnl_pct": round(pnl_pct, 6),
                            "pnl_abs": round(pnl_abs, 2),
                            "entry_price": entry_price,
                            "exit_price": exit_price_val,
                            "exit_reason": exit_reason,
                            "direction": "SELL",
                        }
                    )
                    position = None

        if position is None and i < len(price_arr) - 1:
            if entry_sig:
                entry_price = float(row["open"]) * (1 + (commission_pct + slippage_pct) / 100)
                entry_bar = i
                max_price_since_entry = float(row["high"])
                position = "long"
            elif short_entry_sig:
                entry_price = float(row["open"]) * (1 - (commission_pct + slippage_pct) / 100)
                entry_bar = i
                min_price_since_entry = float(row["low"])
                position = "SELL"

        equity_points.append({"equity": equity})

    equity_curve = pd.DataFrame(equity_points, index=price_df.index)
    return trades, equity_curve


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def run_evaluation(symbol: str, exchange: str, tf: str, start: str, end: str) -> dict:
    """
    Full evaluation pipeline:
      1. Fetch OHLCV data
      2. Import generate_signals() from .autoresearch/strategy.py
      3. Simulate trades incorporating stops
      4. Compute all performance metrics
    """

    df = fetch_data(symbol, exchange, tf, start, end)

    # ── Dynamically import the strategy module ─────────────────────────────────
    import importlib.util

    strategy_path = Path(__file__).parent / "strategy.py"
    if not strategy_path.exists():
        raise FileNotFoundError(
            f"strategy.py not found at {strategy_path}. The agent must create/modify this file."
        )

    spec = importlib.util.spec_from_file_location("autoresearch_strategy", strategy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "generate_signals"):
        raise AttributeError(
            "strategy.py must define generate_signals(df: pd.DataFrame) -> pd.DataFrame"
        )

    # generate_signals(df) → pd.DataFrame with boolean columns: entry, exit
    signals_df = mod.generate_signals(df.copy())

    if signals_df is None or signals_df.empty:
        return {"score": 0.0, "trades": 0, "error": "generate_signals returned empty/None"}

    # Extract risk variables if defined
    atr_mult_sl = getattr(mod, "ATR_MULT_SL", 0.0)
    stop_loss_pct = getattr(mod, "STOP_LOSS_PCT", None)
    take_profit_pct = getattr(mod, "TAKE_PROFIT_PCT", None)
    trailing_stop_pct = getattr(mod, "TRAILING_STOP_PCT", None)
    max_hold_bars = getattr(mod, "MAX_HOLD_BARS", None)

    atr_series = None
    if atr_mult_sl > 0:
        if hasattr(mod, "_atr"):
            atr_series = mod._atr(df)
        else:
            hl = df["high"] - df["low"]
            hpc = (df["high"] - df["close"].shift(1)).abs()
            lpc = (df["low"] - df["close"].shift(1)).abs()
            tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
            atr_series = tr.ewm(com=13, min_periods=14).mean()

    # ── Simulate trades ────────────────────────────────────────────────────────
    trades, equity_curve = _simulate_trades(
        df,
        signals_df,
        atr_series=atr_series,
        atr_mult_sl=atr_mult_sl,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
        max_hold_bars=max_hold_bars,
    )

    if not trades:
        return {"score": 0.0, "trades": 0, "error": "no trades generated by current strategy"}

    # ── Compute all metrics ────────────────────────────────────────────────────
    result = compute_metrics(
        strategy_id="autoresearch",
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=100_000.0,
        total_bars=len(df),
    )

    score = compute_score(result)

    return {
        "score": score,
        "trades": result.total_trades,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "calmar": result.calmar,
        "profit_factor": result.profit_factor,
        "win_rate": result.win_rate,
        "cagr": result.cagr,
        "max_drawdown_pct": result.max_drawdown_pct,
        "net_profit_pct": result.net_profit_pct,
        "avg_hold_bars": result.avg_hold_bars,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Autoresearch: evaluate the current strategy.py")
    parser.add_argument("--symbol", default="SBIN", help="Trading symbol(s), comma-separated")
    parser.add_argument("--exchange", default="NSE", help="Exchange (NSE/BSE/MCX)")
    parser.add_argument("--tf", default="D", help="Timeframe(s), comma-separated")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-06-25", help="End date YYYY-MM-DD")
    parser.add_argument("--json", action="store_true", help="Output averaged metrics as JSON")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
    tfs = [t.strip() for t in args.tf.split(",") if t.strip()]

    all_metrics = []
    errors = []

    for s in symbols:
        for t in tfs:
            try:
                metrics = run_evaluation(s, args.exchange, t, args.start, args.end)
                all_metrics.append(metrics)
            except Exception as e:
                errors.append(f"{s}/{t}: {e}")

    if not all_metrics:
        if errors:
            print("ERRORS during evaluation:", file=sys.stderr)
            for err in errors:
                print(f"  {err}", file=sys.stderr)
        print("SCORE: 0.0")
        sys.exit(1)

    # Average metrics
    avg_metrics = {}
    for k in all_metrics[0].keys():
        if k in ["error"]:
            continue
        vals = [m[k] for m in all_metrics if k in m and not isinstance(m[k], str)]
        if vals:
            avg_metrics[k] = round(float(np.mean(vals)), 4)

    if args.json:
        print(json.dumps(avg_metrics, indent=2))
    else:
        print(f"\n{'─' * 52}")
        print(f"  Symbols   : {', '.join(symbols)}/{args.exchange}")
        print(f"  TFs       : {', '.join(tfs)}")
        print(f"  Period    : {args.start} -> {args.end}")
        print(f"{'─' * 52}")
        for k, v in avg_metrics.items():
            if k == "score":
                continue
            print(f"  {k:<22} : {v}")
        print(f"{'─' * 52}")

    # ALWAYS print SCORE: <float> as the very last line
    print(f"SCORE: {avg_metrics.get('score', 0.0)}")


if __name__ == "__main__":
    main()
