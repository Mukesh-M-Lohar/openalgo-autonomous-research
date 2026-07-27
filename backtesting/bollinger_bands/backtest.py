import os
import sys
from pathlib import Path

import pandas as pd
import vectorbt as vbt
from dotenv import find_dotenv, load_dotenv
from openalgo import api, ta

# Add scripts directory to path to import scanner modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
# ruff: noqa: E402
from scripts.supertrend_scanner.indicators import bollinger_bands
from scripts.supertrend_scanner.ss_scanner import fetch_history

# --- Config ---
load_dotenv(find_dotenv(), override=False)

SYMBOL = "NIFTY"
EXCHANGE = "NSE_INDEX"
INTERVAL = "5m"

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
df = fetch_history(client, SYMBOL, EXCHANGE, INTERVAL, START_DATE, END_DATE)

print("Computing Bollinger Bands...")
bb_mid, bb_upper, bb_lower, bb_pctb, bb_bw = bollinger_bands(df["close"], period=20, num_std=2.0)
df["bb_lower"] = bb_lower
df["bb_upper"] = bb_upper

print("Applying Bollinger Band Touch Logic...")
# Buy on touch of lower band
buy_raw = df["low"] <= df["bb_lower"]
# Sell (Exit) on touch of upper band
sell_raw = df["high"] >= df["bb_upper"]

# Clean signals
entries = ta.exrem(buy_raw.fillna(False), sell_raw.fillna(False))
exits = ta.exrem(sell_raw.fillna(False), buy_raw.fillna(False))

print(f"Total Entries: {entries.sum()}")
print(f"Total Exits: {exits.sum()}")

# --- Backtest ---
print("Running VectorBT Portfolio...")
pf = vbt.Portfolio.from_signals(
    df["close"],
    entries,
    exits,
    init_cash=INIT_CASH,
    size=ALLOCATION,
    size_type="percent",
    fees=FEES,
    fixed_fees=FIXED_FEES,
    direction="longonly",
    freq="5m",
    sl_stop=0.02,  # Wide 2% stop loss to prevent catastrophic trend moves from blowing up the account
)

# --- Benchmark ---
bench_close = df["close"]
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

print(f"\n--- BB MEAN REVERSION BACKTEST ({START_DATE} to {END_DATE}) ---")
print(comparison.to_string())

# --- Explain ---
print(
    f"\n* Total Return: {pf.total_return() * 100:.2f}% vs Benchmark {pf_bench.total_return() * 100:.2f}%"
)
print(f"* Max Drawdown: {pf.max_drawdown() * 100:.2f}%")
print(
    f"  -> On Rs {INIT_CASH:,}, worst temporary loss = Rs {abs(pf.max_drawdown()) * INIT_CASH:,.0f}"
)
