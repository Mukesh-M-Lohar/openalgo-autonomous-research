import json
import os

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
    print(f"{'Asset':<20} | {'Total Trades':<12} | {'Longs':<8} | {'Shorts':<8}")
    print("-" * 55)

    for sym, exch, allow_short in SYMBOLS:
        df = load_data(sym, exch)
        if df.empty:
            continue

        df_test = df.loc["2026-03-01":]
        trades, _ = backtest_intraday(df_test, params, allow_short)

        longs = sum(1 for t in trades if t["direction"] == "LONG")
        shorts = sum(1 for t in trades if t["direction"] == "SHORT")

        print(f"{sym + ':' + exch:<20} | {len(trades):<12} | {longs:<8} | {shorts:<8}")


if __name__ == "__main__":
    main()
