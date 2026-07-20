import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from indicators import build_indicators, detect_supertrend_touches
from openalgo import api

# =========================================================================
# CONFIG -- edit these
# =========================================================================

load_dotenv()
OPENALGO_API_KEY = os.getenv("OPENALGO_API_KEY")
OPENALGO_HOST = "http://127.0.0.1:5000"  # local host by default

# Instruments to scan. Adjust exchange to match how they're listed in your
# OpenAlgo broker mapping -- NSE_INDEX for spot index, or NFO for futures
# (e.g. {"symbol": "NIFTY28AUG25FUT", "exchange": "NFO"}).
SYMBOLS = [
    {"symbol": "NIFTY", "exchange": "NSE_INDEX"},
    {"symbol": "BANKNIFTY", "exchange": "NSE_INDEX"},
]

INTERVAL = "5m"  # see client.interval() for supported values
START_DATE = "2026-01-01"
END_DATE = "2026-07-19"

# Supertrend
ST_PERIOD = 10
ST_MULTIPLIER = 3.0

# "Near-touch" threshold: close within this % distance of the active band
TOUCH_PCT = 0.15

# EMA / SMA periods -- broad set covering intraday scalping through swing
EMA_PERIODS = (5, 9, 13, 21, 34, 50, 100, 200)
SMA_PERIODS = (20, 50, 100, 200)

RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_PERIOD, BB_STD = 20, 2.0
STOCH_K, STOCH_D, STOCH_SMOOTH = 14, 3, 3
ADX_PERIOD = 14
VOL_AVG_PERIOD = 20

# India VIX (daily-only). Forward-filled onto intraday bars: for intraday
# timeframes the VIX value shown is the most recent daily close known at
# that point in time (no lookahead).
INCLUDE_VIX = True
VIX_SYMBOL = "INDIAVIX"
VIX_EXCHANGE = "NSE_INDEX"

OUTPUT_DIR = "./supertrend_touch_output"

# =========================================================================


def fetch_history(client, symbol, exchange, interval, start_date, end_date):
    df = client.history(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
    )
    df.columns = [c.lower() for c in df.columns]

    # Normalize the timestamp column into a DatetimeIndex
    ts_col = next((c for c in ["timestamp", "datetime", "date", "time"] if c in df.columns), None)
    if ts_col is not None:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected OHLCV columns from OpenAlgo response: {missing}")
    return df


def fetch_vix(client, start_date, end_date):
    try:
        vix = client.history(
            symbol=VIX_SYMBOL,
            exchange=VIX_EXCHANGE,
            interval="D",
            start_date=start_date,
            end_date=end_date,
        )
        vix.columns = [c.lower() for c in vix.columns]
        ts_col = next(
            (c for c in ["timestamp", "datetime", "date", "time"] if c in vix.columns), None
        )
        if ts_col is not None:
            vix[ts_col] = pd.to_datetime(vix[ts_col])
            vix = vix.set_index(ts_col)
        vix = vix.sort_index()
        return vix["close"].rename("india_vix")
    except Exception as e:
        print(f"  [warn] Could not fetch India VIX: {e}")
        return None


def attach_vix(df: pd.DataFrame, vix_daily: pd.Series) -> pd.DataFrame:
    """Forward-fill daily VIX onto intraday index, no lookahead bias."""
    if vix_daily is None:
        df["india_vix"] = np.nan
        return df

    vix_daily = vix_daily.copy()
    vix_daily.index = pd.to_datetime(vix_daily.index).normalize()

    df = df.copy()
    df["_date"] = pd.to_datetime(df.index).normalize()

    # reindex vix to cover the full date range, forward-fill for weekends/holidays
    full_dates = pd.date_range(vix_daily.index.min(), df["_date"].max(), freq="D")
    vix_ff = vix_daily.reindex(full_dates).ffill()

    df["india_vix"] = df["_date"].map(vix_ff)
    df = df.drop(columns=["_date"])
    return df


def scan_symbol(client, sym_cfg, vix_daily):
    symbol, exchange = sym_cfg["symbol"], sym_cfg["exchange"]
    print(f"\nFetching {symbol} ({exchange}), interval={INTERVAL} ...")
    df = fetch_history(client, symbol, exchange, INTERVAL, START_DATE, END_DATE)
    print(f"  {len(df)} bars fetched: {df.index.min()} -> {df.index.max()}")

    if INCLUDE_VIX:
        df = attach_vix(df, vix_daily)

    print("  Computing indicators ...")
    ind = build_indicators(
        df,
        st_period=ST_PERIOD,
        st_mult=ST_MULTIPLIER,
        ema_periods=EMA_PERIODS,
        sma_periods=SMA_PERIODS,
        rsi_period=RSI_PERIOD,
        atr_period=ATR_PERIOD,
        macd_fast=MACD_FAST,
        macd_slow=MACD_SLOW,
        macd_signal=MACD_SIGNAL,
        bb_period=BB_PERIOD,
        bb_std=BB_STD,
        stoch_k=STOCH_K,
        stoch_d=STOCH_D,
        stoch_smooth=STOCH_SMOOTH,
        adx_period=ADX_PERIOD,
        vol_avg_period=VOL_AVG_PERIOD,
    )

    touches = detect_supertrend_touches(ind, touch_pct=TOUCH_PCT)
    touches.insert(0, "symbol", symbol)
    touches.index.name = "timestamp"

    print(
        f"  Touch events found: {len(touches)} "
        f"(BUY: {(touches['signal_type'] == 'BUY_TOUCH').sum()}, "
        f"SELL: {(touches['signal_type'] == 'SELL_TOUCH').sum()})"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"supertrend_touches_{symbol}.csv")
    touches.to_csv(out_path)
    print(f"  Saved -> {out_path}")
    return touches


def main():
    client = api(api_key=OPENALGO_API_KEY, host=OPENALGO_HOST)

    vix_daily = None
    if INCLUDE_VIX:
        print("Fetching India VIX (daily) ...")
        vix_daily = fetch_vix(client, START_DATE, END_DATE)

    all_touches = []
    for sym_cfg in SYMBOLS:
        touches = scan_symbol(client, sym_cfg, vix_daily)
        all_touches.append(touches)

    combined = pd.concat(all_touches).sort_index()
    combined_path = os.path.join(OUTPUT_DIR, "supertrend_touches_ALL.csv")
    combined.to_csv(combined_path)
    print(f"\nCombined file -> {combined_path}  ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
