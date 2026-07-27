import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from dotenv import find_dotenv, load_dotenv
from openalgo import api, ta

# Add scripts directory to path to import scanner modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from scripts.supertrend_scanner.ss_scanner import (
    TOUCH_PCT,
    build_indicators,
    detect_supertrend_touches,
    fetch_history,
)

# --- Config ---
load_dotenv(find_dotenv(), override=False)

SYMBOL = "NIFTY"
EXCHANGE = "NSE_INDEX"
INTERVAL = "5m"
HTF_INTERVAL = "15m"

# Out-of-Sample Period
START_DATE = "2021-01-01"
END_DATE = "2026-07-21"

INIT_CASH = 1_000_000
FEES = 0.0002  # Indian Index Futures (approx 0.02% total per leg including STT/Exch)
FIXED_FEES = 20  # Rs 20 per order
ALLOCATION = 1.0

# --- Fetch Data ---
client = api(
    api_key=os.getenv("OPENALGO_API_KEY"),
    host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
)

print(f"Fetching {SYMBOL} data for {START_DATE} to {END_DATE}...")
df_ltf = fetch_history(client, SYMBOL, EXCHANGE, INTERVAL, START_DATE, END_DATE)
df_htf = fetch_history(client, SYMBOL, EXCHANGE, HTF_INTERVAL, START_DATE, END_DATE)

print("Computing 5m indicators...")
ind_ltf = build_indicators(df_ltf)

print("Computing 15m indicators...")
ind_htf = build_indicators(df_htf)
ind_htf.columns = [f"htf_{c}" for c in ind_htf.columns]

# Forward fill HTF onto LTF
print("Merging timeframes...")
ind_htf_aligned = ind_htf.reindex(ind_ltf.index, method="ffill")
df_merged = ind_ltf.join(ind_htf_aligned)

print("Detecting touches and applying ML rules...")
# 1. Base rule: Price touches 5m Supertrend
touches = detect_supertrend_touches(df_merged, TOUCH_PCT)

# Let's align touches back to the main dataframe
df_merged["touch_type"] = None
df_merged.loc[touches.index, "touch_type"] = touches["signal_type"]

# 2. Filter 1: HTF ADX must be trending (>25)
htf_trend_strong = df_merged["htf_adx"] > 25

# 3. Filter 2: Elasticity (The True "Rubber Band" Rule)
# A) The trend must have stretched significantly before the pullback.
# We define "stretch" as the total size of the trend (high to low) being at least 3x the ATR.
trend_size = df_merged["fib_swing_high"] - df_merged["fib_swing_low"]
is_stretched = trend_size > (3.0 * df_merged["atr"])

# B) The pullback must be deep (at least 50% retracement of the trend).
# In an uptrend (st_trend == 1), pullback depth = (high - close) / trend_size
# In a downtrend (st_trend == -1), pullback depth = (close - low) / trend_size
pullback_depth = np.where(
    df_merged["st_trend"] == 1,
    (df_merged["fib_swing_high"] - df_merged["close"]) / trend_size.replace(0, np.nan),
    (df_merged["close"] - df_merged["fib_swing_low"]) / trend_size.replace(0, np.nan),
)
is_deep_pullback = pullback_depth >= 0.5

# Create BUY and SELL signals
buy_raw = (
    (df_merged["touch_type"] == "BUY_TOUCH") & htf_trend_strong & is_stretched & is_deep_pullback
)
sell_raw = (
    (df_merged["touch_type"] == "SELL_TOUCH") & htf_trend_strong & is_stretched & is_deep_pullback
)
# Exit when the 5m supertrend flips against us
exits_raw = df_merged["st_trend"] == -1

# Clean signals
entries = ta.exrem(buy_raw.fillna(False), exits_raw.fillna(False))
exits = ta.exrem(exits_raw.fillna(False), buy_raw.fillna(False))

print(f"Total Entries: {entries.sum()}")
print(f"Total Exits: {exits.sum()}")

# --- Backtest ---
print("Running VectorBT Portfolio...")
pf = vbt.Portfolio.from_signals(
    df_merged["close"],
    entries,
    exits,
    init_cash=INIT_CASH,
    size=ALLOCATION,
    size_type="percent",
    fees=FEES,
    fixed_fees=FIXED_FEES,
    direction="longonly",
    freq="5m",
    sl_stop=0.005,  # 0.5% Stop Loss
    sl_trail=True,  # Make it a Trailing Stop Loss to let winners run
)

# --- Benchmark ---
bench_close = df_merged["close"]
pf_bench = vbt.Portfolio.from_holding(bench_close, init_cash=INIT_CASH, fees=FEES, freq="5m")

# --- Results ---
# Strategy vs Benchmark comparison
comparison = pd.DataFrame(
    {
        "Strategy": [
            f"{pf.total_return() * 100:.2f}%",
            f"{pf.sharpe_ratio():.2f}",
            f"{pf.sortino_ratio():.2f}",
            f"{pf.max_drawdown() * 100:.2f}%",
            f"{pf.trades.win_rate() * 100:.1f}%",
            f"{pf.trades.count()}",
            f"{pf.trades.profit_factor():.2f}",
        ],
        f"Benchmark ({SYMBOL})": [
            f"{pf_bench.total_return() * 100:.2f}%",
            f"{pf_bench.sharpe_ratio():.2f}",
            f"{pf_bench.sortino_ratio():.2f}",
            f"{pf_bench.max_drawdown() * 100:.2f}%",
            "-",
            "-",
            "-",
        ],
    },
    index=[
        "Total Return",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Win Rate",
        "Total Trades",
        "Profit Factor",
    ],
)

print("\n--- OOS BACKTEST (2021-2023) ---")
print(comparison.to_string())

# --- Explain ---
print(
    f"\n* Total Return: {pf.total_return() * 100:.2f}% vs Benchmark {pf_bench.total_return() * 100:.2f}%"
)
print(f"* Max Drawdown: {pf.max_drawdown() * 100:.2f}%")
print(
    f"  -> On Rs {INIT_CASH:,}, worst temporary loss = Rs {abs(pf.max_drawdown()) * INIT_CASH:,.0f}"
)
