import os
import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openalgo import api

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# pyrefly: ignore [missing-import]
from data_cache import CachedDataFetcher

# --- CONFIGURATION ---
UNDERLYING_SYMBOL = "BANKNIFTY"
UNDERLYING_EXCHANGE = "NSE_INDEX"
START_DATE = "2026-07-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
STRIKE_PADDING = 1500  # Expand the month's High/Low by this many points

OUTPUT_DIR = "/root/openalgo-autonomous-research/supertrend_touch_output"
CACHE_DIR = os.path.join(OUTPUT_DIR, ".cache")


def get_client():
    load_dotenv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"
        )
    )
    API_KEY = os.getenv("OPENALGO_API_KEY")
    HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
    if not API_KEY:
        raise ValueError("OPENALGO_API_KEY not found in .env")
    return api(api_key=API_KEY, host=HOST)


def main():
    print("==================================================")
    print("UNIVERSAL OPTIONS DATA PRE-FETCHER (1m)")
    print("==================================================")

    os.makedirs(CACHE_DIR, exist_ok=True)
    client = get_client()
    fetcher = CachedDataFetcher(client, cache_dir=CACHE_DIR)

    # 1. Get Underlying Month High/Low
    print(f"Fetching {UNDERLYING_SYMBOL} (D) to determine strike boundaries...")
    df_daily = fetcher.fetch_history(
        UNDERLYING_SYMBOL, UNDERLYING_EXCHANGE, "D", START_DATE, END_DATE
    )

    if df_daily.empty:
        print("Failed to fetch underlying data.")
        return

    month_high = df_daily["high"].max()
    month_low = df_daily["low"].min()

    min_strike = int((month_low - STRIKE_PADDING) // 100 * 100)
    max_strike = int((month_high + STRIKE_PADDING) // 100 * 100) + 100

    print(f"Month Low: {month_low:.2f} | Month High: {month_high:.2f}")
    print(f"Target Strikes: {min_strike} to {max_strike}")

    # 2. Get Instruments
    print("Fetching Instruments Master...")
    try:
        instruments = client.instruments()
        inst_df = pd.DataFrame(instruments)
    except Exception as e:
        print(f"Error fetching instruments: {e}")
        return

    # 3. Filter Options
    # Only NFO options for the underlying, with expiry in the backtest window (or slightly after)
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE) + pd.Timedelta(days=45)

    if "expiry" in inst_df.columns:
        inst_df["expiry_date"] = pd.to_datetime(inst_df["expiry"], errors="coerce")
    else:
        print("No expiry column found in instruments.")
        return

    opt_df = inst_df[
        (inst_df["exchange"] == "NFO")
        & (inst_df["name"] == UNDERLYING_SYMBOL)
        & (inst_df["expiry_date"] >= start_dt)
        & (inst_df["expiry_date"] <= end_dt)
        & (inst_df["strike"] >= min_strike)
        & (inst_df["strike"] <= max_strike)
    ]

    symbols_to_fetch = opt_df["symbol"].unique()
    total_symbols = len(symbols_to_fetch)
    print(f"\nIdentified {total_symbols} unique option contracts to fetch.")

    # 4. Fetch and Cache 1m Data
    print("\nStarting bulk fetch of 1m data...")
    for i, symbol in enumerate(symbols_to_fetch):
        print(f"[{i + 1}/{total_symbols}] Fetching {symbol} (1m)...")
        # By calling fetcher, it automatically checks cache and hits API if needed
        # We fetch for the whole month. If the option expires early, the API just returns what it has.
        _ = fetcher.fetch_history(symbol, "NFO", "1m", START_DATE, END_DATE)

    print("\n==================================================")
    print("ALL OPTION DATA PRE-FETCHED AND CACHED SUCCESSFULLY!")
    print("You can now run your grid optimizer entirely offline.")
    print("==================================================")


if __name__ == "__main__":
    main()
