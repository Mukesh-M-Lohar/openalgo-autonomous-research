import os

# We need the indicators from the scanner
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openalgo import api

sys.path.append("/root/openalgo-autonomous-research/scripts/supertrend_scanner")
from indicators import supertrend

load_dotenv("/root/openalgo-autonomous-research/.env")
API_KEY = os.getenv("OPENALGO_API_KEY")
HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")

client = api(api_key=API_KEY, host=HOST)

print("Fetching instruments...")
instruments = client.instruments(exchange="NFO")
bn_opts = instruments[
    (instruments["name"] == "BANKNIFTY") & (instruments["instrumenttype"].isin(["CE", "PE"]))
].copy()
bn_opts["expiry_date"] = pd.to_datetime(bn_opts["expiry"])

print("Loading touches...")
df = pd.read_csv(
    "/root/openalgo-autonomous-research/supertrend_touch_output/supertrend_touches_ALL.csv"
)
df["timestamp"] = pd.to_datetime(df["timestamp"])
# Filter for July BANKNIFTY
df_july = df[
    (df["symbol"] == "BANKNIFTY")
    & (df["timestamp"] >= "2026-07-01")
    & (df["timestamp"] < "2026-08-01")
].copy()

print(f"Found {len(df_july)} touches in July for BANKNIFTY.")

results = []

for idx, row in df_july.iterrows():
    touch_time = row["timestamp"]
    touch_date = touch_time.date()
    price = row["close"]
    signal = row["signal_type"]

    # Calculate target strike
    if signal == "SELL_TOUCH":
        strike = round((price - 500) / 100) * 100
        opt_type = "PE"
    else:
        strike = round((price + 500) / 100) * 100
        opt_type = "CE"

    # Find nearest expiry >= touch_date
    valid_opts = bn_opts[
        (bn_opts["strike"] == strike)
        & (bn_opts["instrumenttype"] == opt_type)
        & (bn_opts["expiry_date"].dt.date >= touch_date)
    ]
    if valid_opts.empty:
        print(f"No option found for {strike} {opt_type} on {touch_date}")
        continue

    # Get the nearest expiry
    nearest_opt = valid_opts.sort_values("expiry_date").iloc[0]
    opt_symbol = nearest_opt["symbol"]

    # Progress log
    current = df_july.index.get_loc(idx) + 1
    total = len(df_july)
    print(
        f"[{current}/{total}] {row['timestamp']} | Touch: {signal} @ {price} | Fetching Option: {opt_symbol}..."
    )

    # Fetch option history from touch date to +2 days (to get full day)
    start_str = touch_date.strftime("%Y-%m-%d")
    end_str = (touch_date + timedelta(days=2)).strftime("%Y-%m-%d")

    try:
        opt_history = client.history(
            symbol=opt_symbol, exchange="NFO", interval="5m", start_date=start_str, end_date=end_str
        )
    except Exception as e:
        print(f"Error fetching history for {opt_symbol}: {e}")
        continue

    if isinstance(opt_history, dict):  # Error returned
        print(f"API Error for {opt_symbol}: {opt_history}")
        continue

    # Normalize columns and index
    opt_history.columns = [c.lower() for c in opt_history.columns]
    ts_col = next(
        (c for c in ["timestamp", "datetime", "date", "time"] if c in opt_history.columns), None
    )
    if ts_col is not None:
        opt_history[ts_col] = pd.to_datetime(opt_history[ts_col])
        opt_history = opt_history.set_index(ts_col)

    # Calculate Supertrend on the option
    if len(opt_history) > 10:  # Need enough bars for supertrend
        opt_history = supertrend(opt_history, period=10, multiplier=3.0)
    else:
        continue

    # Find the specific bar at touch_time
    if touch_time not in opt_history.index:
        # Try to find nearest
        continue

    entry_bar = opt_history.loc[touch_time]
    entry_price = entry_bar["close"]
    opt_st_trend = entry_bar.get("st_trend", np.nan)

    # Let's find what happens for the rest of the day
    rest_of_day = opt_history[
        (opt_history.index > touch_time) & (opt_history.index.date == touch_date)
    ]
    if not rest_of_day.empty:
        max_favorable = rest_of_day["high"].max()
        max_adverse = rest_of_day["low"].min()
        close_of_day = rest_of_day["close"].iloc[-1]
    else:
        max_favorable = entry_price
        max_adverse = entry_price
        close_of_day = entry_price

    results.append(
        {
            "Touch Time": touch_time,
            "Underlying Signal": signal,
            "Underlying Price": price,
            "Option Symbol": opt_symbol,
            "Option Strike": strike,
            "Option Type": opt_type,
            "Entry Price": entry_price,
            "Option ST Trend": "UP (1)"
            if opt_st_trend == 1
            else "DOWN (-1)"
            if opt_st_trend == -1
            else "Unknown",
            "Max Favorable (High)": max_favorable,
            "Max Adverse (Low)": max_adverse,
            "EOD Close": close_of_day,
            "Max Potential Profit %": ((max_favorable / entry_price) - 1) * 100
            if entry_price > 0
            else 0,
        }
    )

res_df = pd.DataFrame(results)
res_df.to_csv(
    "/root/openalgo-autonomous-research/supertrend_touch_output/july_options_analysis.csv",
    index=False,
)
print(
    f"Analysis complete! Saved {len(res_df)} option trades to /root/openalgo-autonomous-research/supertrend_touch_output/july_options_analysis.csv"
)

# Print a preview
pd.set_option("display.max_columns", None)
print("\nPreview of first 5 trades:")
print(res_df.head())
