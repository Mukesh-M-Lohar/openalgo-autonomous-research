import json
import os

import numpy as np
import pandas as pd

# Re-use indicator & backtest logic from train.py
from train import backtest_intraday, load_data

BEST_STRATEGY_FILE = "/root/openalgo-autonomous-research/autoresearch/best_strategy.json"

SYMBOLS = [
    ("PROTEAN", "NSE", False),
    ("ZEEL", "NSE", True),
    ("BAHETI-SM", "NSE", False),
    ("CDSL", "NSE", True),
    ("ANGELONE", "NSE", True),
    ("SCI", "NSE", False),
    ("ACUTAAS", "NSE", False),
    ("SAMMAANCAP", "NSE", False),
    ("CLEAN", "NSE", False),
    ("COPPER31JUL26FUT", "MCX", True),
    ("COPPER31AUG26FUT", "MCX", True),
    ("MCX", "NSE", True),
    ("SBIN", "NSE", True),
    ("BSE", "NSE", False),
    ("NIFTY", "NSE_INDEX", True),
    ("BANKNIFTY", "NSE_INDEX", True),
]


def main():
    if not os.path.exists(BEST_STRATEGY_FILE):
        print("best_strategy.json not found")
        return

    with open(BEST_STRATEGY_FILE, "r") as f:
        best_data = json.load(f)

    params = best_data["params"]
    print(f"Running individual analysis using parameters: {params}\n")

    print(
        f"{'Asset':<20} | {'Trades':<6} | {'Sharpe':<7} | {'Total PnL':<10} | {'Avg Daily':<10} | {'Max DD':<7}"
    )
    print("-" * 75)

    for sym, exch, allow_short in SYMBOLS:
        df = load_data(sym, exch)
        if df.empty:
            continue

        df_test = df.loc["2026-03-01":]
        trades, eq = backtest_intraday(df_test, params, allow_short)

        if len(eq) < 2:
            continue

        df_eq = pd.DataFrame(eq, columns=["equity"])
        df_eq.index.name = None
        if not isinstance(df_eq.index, pd.DatetimeIndex):
            df_eq.index = pd.to_datetime(df_eq.index)
        df_eq["date"] = df_eq.index.date
        daily_eq = df_eq.groupby("date")["equity"].last()

        total_pnl = (daily_eq.iloc[-1] - daily_eq.iloc[0]) / daily_eq.iloc[0] * 100
        daily_returns = daily_eq.pct_change().dropna()
        avg_daily = daily_returns.mean() * 100
        sharpe = (
            np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if daily_returns.std() > 0
            else 0
        )
        roll_max = daily_eq.cummax()
        drawdown = (daily_eq - roll_max) / roll_max
        max_dd = drawdown.min() * 100

        print(
            f"{sym + ':' + exch:<20} | {len(trades):<6} | {sharpe:<7.2f} | {total_pnl:<9.2f}% | {avg_daily:<9.4f}% | {max_dd:<6.2f}%"
        )


if __name__ == "__main__":
    main()
