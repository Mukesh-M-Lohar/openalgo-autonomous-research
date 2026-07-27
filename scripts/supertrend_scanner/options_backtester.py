import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

# Import existing indicators and helper scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pyrefly: ignore [missing-import]
from data_cache import CachedDataFetcher
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from indicators import atr, detect_supertrend_touches, supertrend
from openalgo import api

# =========================================================================
# CONFIGURATION
# =========================================================================
# --- UNDERLYING PARAMS (same names as live bot) ---
UNDERLYING_SYMBOL = "BANKNIFTY"
EXCHANGE = "NSE_INDEX"
TIMEFRAME = "5m"
ST_PERIOD = 10
ST_MULT = 3.0
TOUCH_PCT = 0.15  # How close to band to count as touch (%)

# --- OPTION PARAMS (same names as live bot) ---
STRIKE_OFFSET = 500  # 500 points ITM
STRIKE_STEP = 100  # Strike interval (100 for BANKNIFTY, 50 for NIFTY)
AVOID_0DTE = True  # Skip same-day expiry contracts
OPTION_TIMEFRAME = "3m"  # Backtester-only: for option ST analysis
OPTION_ST_PERIOD = 10
OPTION_ST_MULT = 3.0

# --- BACKTEST WINDOW ---
START_DATE = "2026-07-01"
END_DATE = "2026-07-17"

# --- EXIT RULES (same names as live bot) ---
TAKE_PROFIT_PCT = None  # e.g. 10.0 for 10%
TRAIL_SL_PCT = 10.0  # Trailing stop loss percentage (10.0 for 10%)
EXIT_ON_ST_FLIP = True  # Exit trade if Underlying Index Supertrend flips direction

# --- TIME OF DAY FILTER (IST) ---
ALLOWED_START_HOUR = 11  # 11:00 AM IST start
ALLOWED_END_HOUR = 14  # 02:00 PM IST end

OUTPUT_DIR = "/root/openalgo-autonomous-research/supertrend_touch_output"

# =========================================================================


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


# pyrefly: ignore [missing-import]


def main(print_logs=True, df_underlying_prefetched=None, opts_master_prefetched=None):
    if print_logs:
        print("Initializing Options Backtester...")
    client = get_client()
    fetcher = CachedDataFetcher(client, cache_dir=os.path.join(OUTPUT_DIR, ".cache"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Fetch Underlying Data
    if df_underlying_prefetched is not None:
        df_underlying = df_underlying_prefetched.copy()
    else:
        if print_logs:
            print(f"Fetching {UNDERLYING_SYMBOL} ({TIMEFRAME}) from {START_DATE} to {END_DATE}...")
        df_underlying = fetcher.fetch_history(
            UNDERLYING_SYMBOL, EXCHANGE, TIMEFRAME, START_DATE, END_DATE
        )

    if df_underlying.empty:
        print("Failed to fetch underlying data. Exiting.")
        return

    print("Calculating Underlying Supertrend...")
    df_underlying["atr"] = atr(df_underlying, period=ST_PERIOD)
    df_underlying = supertrend(df_underlying, period=ST_PERIOD, multiplier=ST_MULT)

    touches = detect_supertrend_touches(df_underlying, touch_pct=TOUCH_PCT)

    print(f"Found {len(touches)} touches.")
    if len(touches) == 0:
        return

    # 2. Fetch Options Instrument List
    print("Fetching Instruments Master...")
    instruments = client.instruments(exchange="NFO")
    opts_master = instruments[
        (instruments["name"] == UNDERLYING_SYMBOL)
        & (instruments["instrumenttype"].isin(["CE", "PE"]))
    ].copy()
    opts_master["expiry_date"] = pd.to_datetime(opts_master["expiry"])

    # 3. Process each touch
    results = []

    for idx, row in touches.iterrows():
        touch_time = idx
        touch_date = touch_time.date()
        price = row["close"]
        signal = row["signal_type"]
        current_idx = touches.index.get_loc(idx) + 1

        # Time of day filter
        if ALLOWED_START_HOUR is not None and ALLOWED_END_HOUR is not None:
            if not (ALLOWED_START_HOUR <= touch_time.hour <= ALLOWED_END_HOUR):
                print(
                    f"[{current_idx}/{len(touches)}] {touch_time} | Skipped (Outside allowed hours)"
                )
                continue

        print(f"[{current_idx}/{len(touches)}] {touch_time} | {signal} @ {price:.2f} | ", end="")

        # Determine Strike — ITM selection matching live bot
        base_strike = int(round(price / STRIKE_STEP)) * STRIKE_STEP
        if signal == "SELL_TOUCH":
            strike = base_strike + STRIKE_OFFSET  # PE ITM
            opt_type = "PE"
        else:
            strike = base_strike - STRIKE_OFFSET  # CE ITM
            opt_type = "CE"

        # Liquidity window: target strike and +/- one step (matches live bot)
        strikes_to_check = [strike - STRIKE_STEP, strike, strike + STRIKE_STEP]

        # Find Nearest Expiry (with AVOID_0DTE support)
        if AVOID_0DTE:
            expiry_filter = opts_master["expiry_date"].dt.date > touch_date
        else:
            expiry_filter = opts_master["expiry_date"].dt.date >= touch_date

        valid_opts = opts_master[
            (opts_master["strike"].isin(strikes_to_check))
            & (opts_master["instrumenttype"] == opt_type)
            & expiry_filter
        ]

        if valid_opts.empty:
            print(f"No option found for {strike} {opt_type}")
            continue

        # Pick nearest expiry, then try each candidate strike for available history
        nearest_expiry = valid_opts["expiry_date"].min()
        candidates = valid_opts[valid_opts["expiry_date"] == nearest_expiry]
        # Prefer the target strike first, then neighbors
        candidates = candidates.copy()
        candidates["strike_dist"] = (candidates["strike"] - strike).abs()
        candidates = candidates.sort_values("strike_dist")

        opt_symbol = candidates.iloc[0]["symbol"]
        print(f"{opt_symbol}...", end="", flush=True)

        # Fetch Option History
        start_str = touch_date.strftime("%Y-%m-%d")
        end_str = (touch_date + timedelta(days=2)).strftime("%Y-%m-%d")

        opt_history = fetcher.fetch_history(opt_symbol, "NFO", OPTION_TIMEFRAME, start_str, end_str)
        if opt_history.empty:
            print("Failed history fetch.")
            continue

        # Calculate Option Supertrend (for analysis — live bot doesn't use this for decisions)
        if len(opt_history) > OPTION_ST_PERIOD:
            opt_history = supertrend(
                opt_history, period=OPTION_ST_PERIOD, multiplier=OPTION_ST_MULT
            )
        else:
            print("Not enough bars for ST.")
            continue

        # Extract trade details
        valid_entry_bars = opt_history[opt_history.index >= pd.Timestamp(touch_time)]
        if valid_entry_bars.empty:
            print("No valid option bar after touch.")
            continue

        entry_time = valid_entry_bars.index[0]
        entry_bar = valid_entry_bars.iloc[0]
        entry_price = entry_bar["close"]
        opt_st = entry_bar.get("st_trend", np.nan)

        # 🚨 NEW RULE: Option ST must be UP (1) to enter the trade
        if opt_st != 1:
            print(f"Skipping {opt_symbol} at {touch_time} — Option ST is not UP (Trend: {opt_st})")
            continue

        # Simulate Intraday Exit (TP / SL / EOD)
        eod_bars = opt_history[
            (opt_history.index > entry_time) & (opt_history.index.date == touch_date)
        ]

        exit_price = entry_price
        exit_reason = "NO_DATA"
        max_fav = entry_price
        max_adv = entry_price

        if not eod_bars.empty:
            tp_price = (
                entry_price * (1 + (TAKE_PROFIT_PCT / 100.0)) if TAKE_PROFIT_PCT else float("inf")
            )
            initial_sl = entry_price * (1 - (TRAIL_SL_PCT / 100.0)) if TRAIL_SL_PCT else 0.0
            sl_price = initial_sl

            und_trends = df_underlying["st_trend"].reindex(eod_bars.index, method="ffill")

            for idx, bar in eod_bars.iterrows():
                high = bar["high"]
                low = bar["low"]

                # 1. Update maximums FIRST (matches live bot order)
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)

                # 2. Update Trailing Stop Loss based on new high
                if TRAIL_SL_PCT:
                    new_sl = max_fav * (1 - (TRAIL_SL_PCT / 100.0))
                    if new_sl > sl_price:
                        sl_price = new_sl

                # 3. Check SL hit (after trail update, matching live bot)
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = "TRAIL_SL_HIT" if sl_price > initial_sl else "SL_HIT"
                    break

                # 4. Check Take Profit
                if high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP_HIT"
                    break

                # 5. Check Underlying Index Supertrend Flip
                if EXIT_ON_ST_FLIP:
                    und_st_trend = und_trends.loc[idx] if idx in und_trends.index else np.nan
                    if signal == "BUY_TOUCH" and und_st_trend == -1:
                        exit_price = bar["close"]
                        exit_reason = "INDEX_ST_FLIP"
                        break
                    elif signal == "SELL_TOUCH" and und_st_trend == 1:
                        exit_price = bar["close"]
                        exit_reason = "INDEX_ST_FLIP"
                        break
            else:
                # Loop finished without hitting SL or TP, exit at EOD
                exit_price = eod_bars["close"].iloc[-1]
                exit_reason = "EOD"
                max_fav = max(max_fav, eod_bars["high"].max())
                max_adv = min(max_adv, eod_bars["low"].min())

        results.append(
            {
                "Touch Time": touch_time,
                "Underlying Signal": signal,
                "Underlying Price": price,
                "Option Symbol": opt_symbol,
                "Option Strike": strike,
                "Option Type": opt_type,
                "Entry Price": entry_price,
                "Option ST Trend": "UP" if opt_st == 1 else "DOWN" if opt_st == -1 else "UNKNOWN",
                "Max Favorable": max_fav,
                "Max Adverse": max_adv,
                "Exit Reason": exit_reason,
                "Exit Price": exit_price,
                "Max Profit %": ((max_fav / entry_price) - 1) * 100 if entry_price > 0 else 0,
                "Max Drawdown %": ((max_adv / entry_price) - 1) * 100 if entry_price > 0 else 0,
                "Return %": ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0,
            }
        )
        print("Done.")

    if not results:
        print("No trades generated.")
        return

    # 4. Generate Report
    res_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "backtest_trades.csv")
    res_df.to_csv(out_path, index=False)

    print("\n" + "=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Total Trades Simulated: {len(res_df)}")

    winners = res_df[res_df["Return %"] > 0]
    win_rate = (len(winners) / len(res_df)) * 100

    total_return_sum = res_df["Return %"].sum()
    total_return_comp = (np.prod(1 + res_df["Return %"] / 100) - 1) * 100

    print(f"Overall Win Rate: {win_rate:.2f}%")
    print(f"Average Max Profit %: {res_df['Max Profit %'].mean():.2f}%")
    print(f"Average Max Drawdown %: {res_df['Max Drawdown %'].mean():.2f}%")
    print(f"Average Return %: {res_df['Return %'].mean():.2f}%")
    print(f"Total Cumulative Return % (Sum): {total_return_sum:.2f}%")
    print(f"Total Compounded Return %: {total_return_comp:.2f}%")

    print("\nBy Option Type:")
    ce = res_df[res_df["Option Type"] == "CE"]
    if len(ce):
        ce_win = len(ce[ce["Return %"] > 0]) / len(ce) * 100
        print(
            f"  CE Trades (BUY Touches): {len(ce)} | Win Rate: {ce_win:.1f}% | Avg Return: {ce['Return %'].mean():.2f}%"
        )

    pe = res_df[res_df["Option Type"] == "PE"]
    if len(pe):
        pe_win = len(pe[pe["Return %"] > 0]) / len(pe) * 100
        print(
            f"  PE Trades (SELL Touches): {len(pe)} | Win Rate: {pe_win:.1f}% | Avg Return: {pe['Return %'].mean():.2f}%"
        )

    print(f"\nDetailed trade logs saved to: {out_path}")
    return res_df


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
