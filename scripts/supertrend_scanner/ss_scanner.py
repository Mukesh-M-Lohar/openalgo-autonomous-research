"""
Supertrend Band-Touch Scanner (single-file version)
=====================================================
Fetches OHLCV data from your local OpenAlgo instance for NIFTY / BANKNIFTY,
computes Supertrend + a full indicator stack (RSI, EMAs, SMAs, ATR, MACD,
Bollinger Bands, Stochastic, ADX, VWAP, volume ratio, OI delta if available,
India VIX), detects every bar where price comes within TOUCH_PCT% of the
active Supertrend band, and tracks what happened after each touch (bounce
size, trend reversal point). Outputs one CSV per symbol + a combined CSV.

SETUP
-----
1. pip install openalgo pandas numpy python-dotenv --break-system-packages
2. Create a .env file in the same folder as this script:
       OPENALGO_API_KEY=your_actual_key
3. Edit the CONFIG section below (symbols, dates, interval, thresholds).
4. Run:  python3 supertrend_scanner.py

Docs referenced: https://docs.openalgo.in/api-documentation/v1/data-api/history
"""

import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openalgo import api

# =========================================================================
# INDICATOR + TOUCH-DETECTION LOGIC
# =========================================================================


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ATR, ADX)."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return wilder_smooth(tr, period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close: pd.Series, period=20, num_std=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return mid, upper, lower, pct_b, bandwidth


def stochastic(df: pd.DataFrame, k_period=14, d_period=3, smooth=3):
    low_n = df["low"].rolling(k_period).min()
    high_n = df["high"].rolling(k_period).max()
    raw_k = 100 * (df["close"] - low_n) / (high_n - low_n)
    k = raw_k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d


def adx(df: pd.DataFrame, period: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr_atr = atr(df, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / tr_atr
    minus_di = 100 * wilder_smooth(minus_dm, period) / tr_atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = wilder_smooth(dx, period)
    return adx_val, plus_di, minus_di


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """Session VWAP, resets each calendar day. Requires a DatetimeIndex."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tpv = typical * df["volume"]
    day = df.index.date
    cum_tpv = pd.Series(tpv, index=df.index).groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Returns df with columns: st_upperband, st_lowerband, st_trend (1=up/green, -1=down/red),
    st_line (the plotted supertrend line itself).
    """
    high, low, close = df["high"], df["low"], df["close"]
    hl2 = (high + low) / 2
    tr_atr = atr(df, period)

    basic_upper = hl2 + multiplier * tr_atr
    basic_lower = hl2 - multiplier * tr_atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(1, index=df.index)  # start assuming uptrend

    for i in range(1, len(df)):
        prev_final_upper = final_upper.iloc[i - 1]
        prev_final_lower = final_lower.iloc[i - 1]

        # If the previous band is still NaN (ATR warm-up period), just seed
        # with the current basic band instead of propagating NaN forever.
        if pd.isna(prev_final_upper):
            final_upper.iloc[i] = basic_upper.iloc[i]
        elif basic_upper.iloc[i] < prev_final_upper or close.iloc[i - 1] > prev_final_upper:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_final_upper

        if pd.isna(prev_final_lower):
            final_lower.iloc[i] = basic_lower.iloc[i]
        elif basic_lower.iloc[i] > prev_final_lower or close.iloc[i - 1] < prev_final_lower:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_final_lower

        # trend (guard against NaN bands during warm-up)
        if pd.isna(prev_final_upper) or pd.isna(prev_final_lower):
            trend.iloc[i] = trend.iloc[i - 1]
        elif close.iloc[i] > prev_final_upper:
            trend.iloc[i] = 1
        elif close.iloc[i] < prev_final_lower:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
            # lock the band that isn't in use to the previous value (standard supertrend behavior)
            if trend.iloc[i] == 1 and final_lower.iloc[i] < prev_final_lower:
                final_lower.iloc[i] = prev_final_lower
            if trend.iloc[i] == -1 and final_upper.iloc[i] > prev_final_upper:
                final_upper.iloc[i] = prev_final_upper

    st_line = np.where(trend == 1, final_lower, final_upper)

    out = df.copy()
    out["st_upperband"] = final_upper
    out["st_lowerband"] = final_lower
    out["st_trend"] = trend  # 1 = green/up, -1 = red/down
    out["st_line"] = st_line
    return out


# ----------------------------------------------------------------------
# Full indicator pipeline
# ----------------------------------------------------------------------


def build_indicators(
    df: pd.DataFrame,
    st_period: int = 10,
    st_mult: float = 3.0,
    ema_periods=(5, 9, 13, 21, 34, 50, 100, 200),
    sma_periods=(20, 50, 100, 200),
    rsi_period: int = 14,
    atr_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    stoch_k: int = 14,
    stoch_d: int = 3,
    stoch_smooth: int = 3,
    adx_period: int = 14,
    vol_avg_period: int = 20,
) -> pd.DataFrame:
    df = df.sort_index().copy()

    # Supertrend
    df = supertrend(df, st_period, st_mult)

    # RSI
    df["rsi"] = rsi(df["close"], rsi_period)

    # EMAs / SMAs
    for p in ema_periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    for p in sma_periods:
        df[f"sma_{p}"] = df["close"].rolling(p).mean()

    # ATR
    df["atr"] = atr(df, atr_period)

    # MACD
    macd_line, signal_line, hist = macd(df["close"], macd_fast, macd_slow, macd_signal)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    # Bollinger Bands
    bb_mid, bb_upper, bb_lower, bb_pctb, bb_bw = bollinger_bands(df["close"], bb_period, bb_std)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_pctb"] = bb_pctb
    df["bb_bandwidth"] = bb_bw

    # Stochastic
    stoch_k_val, stoch_d_val = stochastic(df, stoch_k, stoch_d, stoch_smooth)
    df["stoch_k"] = stoch_k_val
    df["stoch_d"] = stoch_d_val

    # ADX / DI
    adx_val, plus_di, minus_di = adx(df, adx_period)
    df["adx"] = adx_val
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # VWAP (session, needs datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        df["vwap"] = vwap_session(df)
    else:
        df["vwap"] = np.nan

    # Volume analytics
    df["vol_sma"] = df["volume"].rolling(vol_avg_period).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"].replace(0, np.nan)

    # OI delta (only if OI column present - e.g. futures data)
    if "oi" in df.columns:
        df["oi_change"] = df["oi"].diff()

    return df


def detect_supertrend_touches(df: pd.DataFrame, touch_pct: float) -> pd.DataFrame:
    """
    For every bar, checks whether close is within `touch_pct` % of the active
    supertrend band (lower band when trend is green/up, upper band when
    trend is red/down). Returns only the rows that qualify as a "touch",
    tagged with signal_type BUY_TOUCH / SELL_TOUCH.
    """
    d = df.copy()

    is_up = d["st_trend"] == 1
    is_down = d["st_trend"] == -1

    band = np.where(is_up, d["st_lowerband"], d["st_upperband"])
    dist_pct = (d["close"] - band).abs() / d["close"] * 100

    d["active_band"] = band
    d["band_distance_pct"] = dist_pct

    touched = dist_pct <= touch_pct
    d["signal_type"] = np.where(
        touched & is_up, "BUY_TOUCH", np.where(touched & is_down, "SELL_TOUCH", None)
    )

    return d[d["signal_type"].notna()].copy()


def evaluate_touch_outcomes(
    full_df: pd.DataFrame, touches: pd.DataFrame, bounce_threshold_points: float
) -> pd.DataFrame:
    """
    For every touch event, walks forward bar-by-bar until the Supertrend
    trend actually reverses (or data runs out), and records:

      - peak_bounce_points   : best favorable move (in points) reached
                                before the trend reversed
      - peak_bounce_timestamp: when that peak occurred
      - points_at_reversal   : signed points move from touch close to the
                                close on the bar where trend flipped
                                (favorable = positive)
      - reversal_timestamp   : timestamp of the trend-flip bar
      - bars_to_reversal     : number of bars from touch to reversal
      - bounce_threshold_points : the X used for this symbol
      - bounce_threshold_hit : whether peak_bounce_points reached X
                                before the reversal happened
      - bars_to_threshold_hit: how many bars it took to hit X (NaN if never)

    "Favorable" direction is up for BUY_TOUCH, down for SELL_TOUCH.
    No lookahead into indicator values is introduced -- this only
    labels what price/trend actually did afterwards, for evaluation.
    """
    full_df = full_df.sort_index()
    close = full_df["close"].values
    trend = full_df["st_trend"].values
    index = full_df.index

    pos_map = {ts: i for i, ts in enumerate(index)}

    records = []
    for ts, row in touches.iterrows():
        pos = pos_map.get(ts)
        if pos is None:
            records.append({})
            continue

        direction = 1 if row["signal_type"] == "BUY_TOUCH" else -1
        touch_close = close[pos]
        touch_trend = trend[pos]

        peak_bounce = 0.0
        peak_ts = pd.NaT
        reversal_ts = pd.NaT
        reversal_close = np.nan
        bars_to_reversal = np.nan
        threshold_hit = False
        bars_to_threshold = np.nan

        j = pos + 1
        n = len(full_df)
        while j < n:
            move = (close[j] - touch_close) * direction
            if move > peak_bounce:
                peak_bounce = move
                peak_ts = index[j]
            if not threshold_hit and peak_bounce >= bounce_threshold_points:
                threshold_hit = True
                bars_to_threshold = j - pos
            if trend[j] != touch_trend:
                reversal_ts = index[j]
                reversal_close = close[j]
                bars_to_reversal = j - pos
                break
            j += 1

        points_at_reversal = (
            (reversal_close - touch_close) * direction if not pd.isna(reversal_close) else np.nan
        )

        records.append(
            {
                "peak_bounce_points": round(peak_bounce, 4),
                "peak_bounce_timestamp": peak_ts,
                "points_at_reversal": points_at_reversal,
                "reversal_timestamp": reversal_ts,
                "bars_to_reversal": bars_to_reversal,
                "bounce_threshold_points": bounce_threshold_points,
                "bounce_threshold_hit": threshold_hit,
                "bars_to_threshold_hit": bars_to_threshold,
            }
        )

    outcome_df = pd.DataFrame(records, index=touches.index)
    return outcome_df


# =========================================================================
# CONFIG + DATA FETCHING + SCAN LOGIC
# =========================================================================


# =========================================================================
# CONFIG -- edit these
# =========================================================================

load_dotenv()
OPENALGO_API_KEY = os.getenv("OPENALGO_API_KEY")
OPENALGO_HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")  # local host by default

if not OPENALGO_API_KEY:
    raise RuntimeError(
        "OPENALGO_API_KEY not found. Create a .env file next to this script "
        "with a line: OPENALGO_API_KEY=your_api_key_here"
    )

# Instruments to scan. Adjust exchange to match how they're listed in your
# OpenAlgo broker mapping -- NSE_INDEX for spot index, or NFO for futures
# (e.g. {"symbol": "NIFTY28AUG25FUT", "exchange": "NFO"}).
SYMBOLS = [
    {"symbol": "NIFTY", "exchange": "NSE_INDEX"},
    {"symbol": "BANKNIFTY", "exchange": "NSE_INDEX"},
]


INTERVAL = sys.argv[1] if len(sys.argv) > 1 else "5m"
START_DATE = "2026-01-01"
END_DATE = "2026-07-19"

# Supertrend
ST_PERIOD = 10
ST_MULTIPLIER = 3.0

# "Near-touch" threshold: close within this % distance of the active band
TOUCH_PCT = 0.15

# Bounce tracking: after a touch, how many points of favorable move (in the
# direction of the trend at the time of touch) counts as a "bounce" before
# checking whether the trend then reversed. Set per symbol since NIFTY and
# BANKNIFTY move on very different point scales.
BOUNCE_POINTS_BY_SYMBOL = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
}
DEFAULT_BOUNCE_POINTS = 50  # used if a symbol isn't listed above

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
    """Forward-fill daily VIX onto intraday index, no lookahead bias.

    OpenAlgo can return tz-aware timestamps (e.g. Asia/Kolkata) for one
    series and tz-naive for another depending on interval/broker. Since we
    only need the calendar date for this join, both sides are normalized
    to tz-naive dates first to avoid a tz-mismatch error.
    """
    if vix_daily is None:
        df["india_vix"] = np.nan
        return df

    vix_daily = vix_daily.copy()
    vix_index = pd.to_datetime(vix_daily.index)
    if vix_index.tz is not None:
        vix_index = vix_index.tz_localize(None)
    vix_daily.index = vix_index.normalize()

    df = df.copy()
    df_index = pd.to_datetime(df.index)
    if df_index.tz is not None:
        df_index = df_index.tz_localize(None)
    df["_date"] = df_index.normalize()

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

    bounce_points = BOUNCE_POINTS_BY_SYMBOL.get(symbol, DEFAULT_BOUNCE_POINTS)
    outcomes = evaluate_touch_outcomes(ind, touches, bounce_threshold_points=bounce_points)
    touches = touches.join(outcomes)

    print(
        f"  Touch events found: {len(touches)} "
        f"(BUY: {(touches['signal_type'] == 'BUY_TOUCH').sum()}, "
        f"SELL: {(touches['signal_type'] == 'SELL_TOUCH').sum()})"
    )
    if len(touches):
        hit_rate = touches["bounce_threshold_hit"].mean() * 100
        print(
            f"  Bounce threshold ({bounce_points} pts) hit before reversal: {hit_rate:.1f}% of touches"
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
    combined_path = os.path.join(OUTPUT_DIR, f"supertrend_touches_ALL_{INTERVAL}.csv")
    combined.to_csv(combined_path)
    print(f"\nCombined file -> {combined_path}  ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
