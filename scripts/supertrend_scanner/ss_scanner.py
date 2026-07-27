"""
Supertrend Band-Touch Scanner (single-file version)
=====================================================
Fetches OHLCV data from your local OpenAlgo instance for NIFTY / BANKNIFTY,
computes Supertrend + a full indicator stack (RSI, EMAs, SMAs, ATR, MACD,
Bollinger Bands, Stochastic, ADX, VWAP, CCI, volume ratio, OI delta if
available, India VIX), detects every bar where price comes within TOUCH_PCT%
of the active Supertrend band, and tracks what happened after each touch
(bounce size, trend reversal point).

Additional analytics per bar:
  - Supertrend touch count (which touch # within the current trend segment)
  - Fibonacci retracement/extension levels (anchored per trend segment)
  - Price–indicator divergence detection (RSI, MACD, Stoch, ADX, OBV, CCI)
  - Candlestick pattern recognition (Marubozu, Spinning Top, Doji, Hammer,
    Hanging Man, Shooting Star, Inverted Hammer, Engulfing, Piercing,
    Dark Cloud Cover, Harami, Morning Star, Evening Star)

Outputs one CSV per symbol + a combined CSV.

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


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical - sma_tp) / (0.015 * mad.replace(0, np.nan))


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


# ----------------------------------------------------------------------
# Candlestick pattern recognition (Zerodha Varsity Ch.5-10)
# ----------------------------------------------------------------------


def _prior_trend(close: pd.Series, lookback: int = 5) -> pd.Series:
    """Simple trend detection: +1 uptrend, -1 downtrend, 0 flat.
    Uses EMA(5) slope over `lookback` bars."""
    ema5 = close.ewm(span=5, adjust=False).mean()
    slope = ema5.diff(lookback)
    return np.sign(slope)


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect single-candle and multi-candle patterns per Zerodha Varsity.
    Adds columns: candle_pattern, candle_signal, candle_body_pct.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan)
    body_pct = body / candle_range * 100  # body as % of range
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    is_bullish = c > o
    is_bearish = c < o
    range_pct = candle_range / c * 100  # range as % of close

    trend = _prior_trend(c)
    prev_trend = trend.shift(1)  # trend before this candle

    # Tolerance for "approximately equal"
    tol = 0.005  # 0.5%

    pattern = pd.Series("", index=df.index)
    signal = pd.Series("", index=df.index)

    # ---- Single-candle patterns ----

    # Bullish Marubozu: Open ≈ Low, Close ≈ High, body > 1% range
    bull_maru = ((o - l).abs() <= tol * c) & ((h - c).abs() <= tol * c) & (range_pct > 1)
    pattern = pattern.where(~bull_maru, "BULLISH_MARUBOZU")
    signal = signal.where(~bull_maru, "BULLISH")

    # Bearish Marubozu: Open ≈ High, Close ≈ Low, body > 1% range
    bear_maru = ((h - o).abs() <= tol * c) & ((c - l).abs() <= tol * c) & (range_pct > 1)
    pattern = pattern.where(~bear_maru, "BEARISH_MARUBOZU")
    signal = signal.where(~bear_maru, "BEARISH")

    # Doji: body < 5% of range
    doji = (body_pct < 5) & (candle_range.notna())
    pattern = pattern.where(~((pattern == "") & doji), "DOJI")
    signal = signal.where(~((signal == "") & doji), "NEUTRAL")

    # Spinning Top: body < 30% of range, both shadows > body
    spinning = (body_pct < 30) & (body_pct >= 5) & (upper_shadow > body) & (lower_shadow > body)
    pattern = pattern.where(~((pattern == "") & spinning), "SPINNING_TOP")
    signal = signal.where(~((signal == "") & spinning), "NEUTRAL")

    # Hammer: lower_shadow >= 2*body, upper_shadow <= 0.1*range, downtrend prior
    hammer_shape = (
        (lower_shadow >= 2 * body) & (upper_shadow <= 0.1 * candle_range) & (body_pct >= 5)
    )
    hammer = hammer_shape & (prev_trend == -1)
    pattern = pattern.where(~((pattern == "") & hammer), "HAMMER")
    signal = signal.where(~((signal == "") & hammer), "BULLISH")

    # Hanging Man: same shape as Hammer, but uptrend prior
    hanging = hammer_shape & (prev_trend == 1)
    pattern = pattern.where(~((pattern == "") & hanging), "HANGING_MAN")
    signal = signal.where(~((signal == "") & hanging), "BEARISH")

    # Shooting Star: upper_shadow >= 2*body, lower_shadow <= 0.1*range, uptrend prior
    star_shape = (upper_shadow >= 2 * body) & (lower_shadow <= 0.1 * candle_range) & (body_pct >= 5)
    shooting = star_shape & (prev_trend == 1)
    pattern = pattern.where(~((pattern == "") & shooting), "SHOOTING_STAR")
    signal = signal.where(~((signal == "") & shooting), "BEARISH")

    # Inverted Hammer: same shape as Shooting Star, but downtrend prior
    inv_hammer = star_shape & (prev_trend == -1)
    pattern = pattern.where(~((pattern == "") & inv_hammer), "INVERTED_HAMMER")
    signal = signal.where(~((signal == "") & inv_hammer), "BULLISH")

    # ---- Multi-candle patterns (2-candle) ----

    prev_o = o.shift(1)
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)
    prev_body = (prev_c - prev_o).abs()
    prev_is_bullish = prev_c > prev_o
    prev_is_bearish = prev_c < prev_o

    # Bullish Engulfing: prev bearish, current bullish, current body engulfs prev body
    bull_engulf = (
        prev_is_bearish
        & is_bullish
        & (c > prev_o)
        & (o < prev_c)
        & (body > prev_body)
        & (prev_trend == -1)
    )
    pattern = pattern.where(~((pattern == "") & bull_engulf), "BULLISH_ENGULFING")
    signal = signal.where(~((signal == "") & bull_engulf), "BULLISH")

    # Bearish Engulfing: prev bullish, current bearish, current body engulfs prev body
    bear_engulf = (
        prev_is_bullish
        & is_bearish
        & (c < prev_o)
        & (o > prev_c)
        & (body > prev_body)
        & (prev_trend == 1)
    )
    pattern = pattern.where(~((pattern == "") & bear_engulf), "BEARISH_ENGULFING")
    signal = signal.where(~((signal == "") & bear_engulf), "BEARISH")

    # Piercing Pattern: prev bearish, current opens below prev low,
    #   closes above midpoint of prev body, downtrend
    prev_mid = (prev_o + prev_c) / 2
    piercing = (
        prev_is_bearish
        & is_bullish
        & (o < prev_l)
        & (c > prev_mid)
        & (c < prev_o)
        & (prev_trend == -1)
    )
    pattern = pattern.where(~((pattern == "") & piercing), "PIERCING")
    signal = signal.where(~((signal == "") & piercing), "BULLISH")

    # Dark Cloud Cover: prev bullish, current opens above prev high,
    #   closes below midpoint of prev body, uptrend
    prev_mid_bull = (prev_o + prev_c) / 2
    dark_cloud = (
        prev_is_bullish
        & is_bearish
        & (o > prev_h)
        & (c < prev_mid_bull)
        & (c > prev_o)
        & (prev_trend == 1)
    )
    pattern = pattern.where(~((pattern == "") & dark_cloud), "DARK_CLOUD_COVER")
    signal = signal.where(~((signal == "") & dark_cloud), "BEARISH")

    # Bullish Harami: prev bearish long body, current small bullish body inside prev
    bull_harami = (
        prev_is_bearish
        & is_bullish
        & (o > prev_c)
        & (c < prev_o)
        & (body < prev_body * 0.5)
        & (prev_trend == -1)
    )
    pattern = pattern.where(~((pattern == "") & bull_harami), "BULLISH_HARAMI")
    signal = signal.where(~((signal == "") & bull_harami), "BULLISH")

    # Bearish Harami: prev bullish long body, current small bearish body inside prev
    bear_harami = (
        prev_is_bullish
        & is_bearish
        & (o < prev_c)
        & (c > prev_o)
        & (body < prev_body * 0.5)
        & (prev_trend == 1)
    )
    pattern = pattern.where(~((pattern == "") & bear_harami), "BEARISH_HARAMI")
    signal = signal.where(~((signal == "") & bear_harami), "BEARISH")

    # ---- Multi-candle patterns (3-candle) ----

    prev2_o = o.shift(2)
    prev2_c = c.shift(2)
    prev2_is_bearish = prev2_c < prev2_o
    prev2_is_bullish = prev2_c > prev2_o
    prev_body_pct = prev_body / (prev_h - prev_l).replace(0, np.nan) * 100

    # Morning Star: Day1 long bearish, Day2 small body (gap down), Day3 bullish closes above Day1 mid
    day1_mid = (prev2_o + prev2_c) / 2
    morning_star = (
        prev2_is_bearish
        & (prev_body_pct < 30)  # Day2 is small body (star)
        & is_bullish
        & (c > day1_mid)  # Day3 closes above Day1 midpoint
        & (prev_trend.shift(2) == -1)  # downtrend context
    )
    pattern = pattern.where(~((pattern == "") & morning_star), "MORNING_STAR")
    signal = signal.where(~((signal == "") & morning_star), "BULLISH")

    # Evening Star: Day1 long bullish, Day2 small body (gap up), Day3 bearish closes below Day1 mid
    evening_star = (
        prev2_is_bullish
        & (prev_body_pct < 30)  # Day2 is small body (star)
        & is_bearish
        & (c < day1_mid)  # Day3 closes below Day1 midpoint
        & (prev_trend.shift(2) == 1)  # uptrend context
    )
    pattern = pattern.where(~((pattern == "") & evening_star), "EVENING_STAR")
    signal = signal.where(~((signal == "") & evening_star), "BEARISH")

    # Replace empty strings with None for cleaner CSV output
    pattern = pattern.replace("", None)
    signal = signal.replace("", None)

    df = df.copy()
    df["candle_pattern"] = pattern
    df["candle_signal"] = signal
    df["candle_body_pct"] = body_pct.round(2)
    return df


# ----------------------------------------------------------------------
# Divergence detection (price vs indicators)
# ----------------------------------------------------------------------


def _find_pivots(series: pd.Series, order: int = 5) -> tuple:
    """
    Find local pivot highs and pivot lows.
    `order` = number of bars on each side to confirm a pivot.
    Returns (pivot_high_indices, pivot_high_values,
             pivot_low_indices, pivot_low_values).
    """
    vals = series.values
    n = len(vals)
    ph_idx, ph_val = [], []
    pl_idx, pl_val = [], []

    for i in range(order, n - order):
        if np.isnan(vals[i]):
            continue
        # Pivot high: vals[i] is greater than all neighbors
        if all(
            vals[i] > vals[i - j] for j in range(1, order + 1) if not np.isnan(vals[i - j])
        ) and all(vals[i] > vals[i + j] for j in range(1, order + 1) if not np.isnan(vals[i + j])):
            ph_idx.append(i)
            ph_val.append(vals[i])
        # Pivot low: vals[i] is less than all neighbors
        if all(
            vals[i] < vals[i - j] for j in range(1, order + 1) if not np.isnan(vals[i - j])
        ) and all(vals[i] < vals[i + j] for j in range(1, order + 1) if not np.isnan(vals[i + j])):
            pl_idx.append(i)
            pl_val.append(vals[i])

    return ph_idx, ph_val, pl_idx, pl_val


def _detect_single_divergence(
    price: pd.Series, indicator: pd.Series, pivot_order: int = 5, lookback: int = 14
) -> tuple:
    """
    Detect regular and hidden divergences between price and an indicator.
    Returns (bull_div_series, bear_div_series) as boolean pd.Series.
    """
    n = len(price)
    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)

    # Price pivots
    p_ph_idx, p_ph_val, p_pl_idx, p_pl_val = _find_pivots(price, pivot_order)
    # Indicator pivots
    i_ph_idx, i_ph_val, i_pl_idx, i_pl_val = _find_pivots(indicator, pivot_order)

    # Regular Bullish: price lower low, indicator higher low
    for k in range(1, len(p_pl_idx)):
        if p_pl_val[k] < p_pl_val[k - 1]:  # price made lower low
            # Find nearest indicator pivot low near this price pivot
            pi = p_pl_idx[k]
            candidates = [(j, v) for j, v in zip(i_pl_idx, i_pl_val) if abs(j - pi) <= lookback]
            prev_candidates = [
                (j, v) for j, v in zip(i_pl_idx, i_pl_val) if abs(j - p_pl_idx[k - 1]) <= lookback
            ]
            if candidates and prev_candidates:
                curr_ind = min(candidates, key=lambda x: abs(x[0] - pi))[1]
                prev_ind = min(prev_candidates, key=lambda x: abs(x[0] - p_pl_idx[k - 1]))[1]
                if curr_ind > prev_ind:  # indicator higher low
                    bull_div[pi] = True

    # Regular Bearish: price higher high, indicator lower high
    for k in range(1, len(p_ph_idx)):
        if p_ph_val[k] > p_ph_val[k - 1]:  # price made higher high
            pi = p_ph_idx[k]
            candidates = [(j, v) for j, v in zip(i_ph_idx, i_ph_val) if abs(j - pi) <= lookback]
            prev_candidates = [
                (j, v) for j, v in zip(i_ph_idx, i_ph_val) if abs(j - p_ph_idx[k - 1]) <= lookback
            ]
            if candidates and prev_candidates:
                curr_ind = min(candidates, key=lambda x: abs(x[0] - pi))[1]
                prev_ind = min(prev_candidates, key=lambda x: abs(x[0] - p_ph_idx[k - 1]))[1]
                if curr_ind < prev_ind:  # indicator lower high
                    bear_div[pi] = True

    # Hidden Bullish: price higher low, indicator lower low
    for k in range(1, len(p_pl_idx)):
        if p_pl_val[k] > p_pl_val[k - 1]:  # price higher low
            pi = p_pl_idx[k]
            candidates = [(j, v) for j, v in zip(i_pl_idx, i_pl_val) if abs(j - pi) <= lookback]
            prev_candidates = [
                (j, v) for j, v in zip(i_pl_idx, i_pl_val) if abs(j - p_pl_idx[k - 1]) <= lookback
            ]
            if candidates and prev_candidates:
                curr_ind = min(candidates, key=lambda x: abs(x[0] - pi))[1]
                prev_ind = min(prev_candidates, key=lambda x: abs(x[0] - p_pl_idx[k - 1]))[1]
                if curr_ind < prev_ind:  # indicator lower low
                    bull_div[pi] = True

    # Hidden Bearish: price lower high, indicator higher high
    for k in range(1, len(p_ph_idx)):
        if p_ph_val[k] < p_ph_val[k - 1]:  # price lower high
            pi = p_ph_idx[k]
            candidates = [(j, v) for j, v in zip(i_ph_idx, i_ph_val) if abs(j - pi) <= lookback]
            prev_candidates = [
                (j, v) for j, v in zip(i_ph_idx, i_ph_val) if abs(j - p_ph_idx[k - 1]) <= lookback
            ]
            if candidates and prev_candidates:
                curr_ind = min(candidates, key=lambda x: abs(x[0] - pi))[1]
                prev_ind = min(prev_candidates, key=lambda x: abs(x[0] - p_ph_idx[k - 1]))[1]
                if curr_ind > prev_ind:  # indicator higher high
                    bear_div[pi] = True

    return (
        pd.Series(bull_div, index=price.index),
        pd.Series(bear_div, index=price.index),
    )


def detect_divergences(df: pd.DataFrame, pivot_order: int = 5, lookback: int = 14) -> pd.DataFrame:
    """
    Detect divergences between price and multiple indicators.
    Adds boolean columns div_<ind>_bull / div_<ind>_bear plus a summary.
    """
    df = df.copy()
    price = df["close"]

    indicators = {
        "rsi": df.get("rsi"),
        "macd": df.get("macd_hist"),
        "stoch": df.get("stoch_k"),
        "adx": df.get("adx"),
        "cci": df.get("cci"),
    }
    # OBV only if we have volume
    if "obv" in df.columns:
        indicators["obv"] = df["obv"]

    summary_parts = []
    for name, ind_series in indicators.items():
        if ind_series is None or ind_series.isna().all():
            df[f"div_{name}_bull"] = False
            df[f"div_{name}_bear"] = False
            continue
        bull, bear = _detect_single_divergence(price, ind_series, pivot_order, lookback)
        df[f"div_{name}_bull"] = bull
        df[f"div_{name}_bear"] = bear
        summary_parts.append((name, bull, bear))

    # Build summary string per bar
    def _build_summary(row):
        parts = []
        for name, _, _ in summary_parts:
            if row.get(f"div_{name}_bull", False):
                parts.append(f"{name.upper()}_BULL")
            if row.get(f"div_{name}_bear", False):
                parts.append(f"{name.upper()}_BEAR")
        return ",".join(parts) if parts else None

    df["divergence_summary"] = df.apply(_build_summary, axis=1)
    return df


# ----------------------------------------------------------------------
# Fibonacci retracement / extension levels
# ----------------------------------------------------------------------


def compute_fibonacci_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Fibonacci retracement and extension levels anchored to each
    supertrend trend segment's swing high/low.
    """
    df = df.copy()
    trend = df["st_trend"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    ext_ratios = [1.272, 1.618, 2.0, 2.618]

    # Output arrays
    fib_swing_high = np.full(n, np.nan)
    fib_swing_low = np.full(n, np.nan)
    fib_levels = {r: np.full(n, np.nan) for r in fib_ratios + ext_ratios}
    fib_nearest = [None] * n
    fib_dist_pct = np.full(n, np.nan)

    # Track segment
    seg_start = 0
    seg_high = high[0] if not np.isnan(high[0]) else 0
    seg_low = low[0] if not np.isnan(low[0]) else 0

    for i in range(n):
        # Detect trend change → reset segment
        if i > 0 and trend[i] != trend[i - 1]:
            seg_start = i
            seg_high = high[i]
            seg_low = low[i]
        else:
            if not np.isnan(high[i]):
                seg_high = max(seg_high, high[i])
            if not np.isnan(low[i]):
                seg_low = min(seg_low, low[i])

        fib_swing_high[i] = seg_high
        fib_swing_low[i] = seg_low

        diff = seg_high - seg_low
        if diff <= 0 or np.isnan(diff):
            continue

        # In uptrend: retracement from high down; in downtrend: from low up
        if trend[i] == 1:  # uptrend — retracement is measured from high downward
            for r in fib_ratios:
                fib_levels[r][i] = seg_high - r * diff
            for r in ext_ratios:
                fib_levels[r][i] = seg_high + (r - 1.0) * diff
        else:  # downtrend — retracement is measured from low upward
            for r in fib_ratios:
                fib_levels[r][i] = seg_low + r * diff
            for r in ext_ratios:
                fib_levels[r][i] = seg_low - (r - 1.0) * diff

        # Find nearest Fib level to current close
        c = close[i]
        if np.isnan(c):
            continue
        best_r, best_dist = None, float("inf")
        for r in fib_ratios + ext_ratios:
            lv = fib_levels[r][i]
            if np.isnan(lv):
                continue
            d = abs(c - lv)
            if d < best_dist:
                best_dist = d
                best_r = r
        if best_r is not None:
            fib_nearest[i] = best_r
            fib_dist_pct[i] = best_dist / c * 100 if c != 0 else np.nan

    df["fib_swing_high"] = fib_swing_high
    df["fib_swing_low"] = fib_swing_low
    for r in fib_ratios + ext_ratios:
        df[f"fib_{r}"] = np.round(fib_levels[r], 2)
    df["fib_nearest_level"] = fib_nearest
    df["fib_distance_pct"] = np.round(fib_dist_pct, 4)
    return df


# ----------------------------------------------------------------------
# Touch counting within supertrend trend segments
# ----------------------------------------------------------------------


def compute_touch_number(df: pd.DataFrame, touch_pct: float) -> pd.DataFrame:
    """
    Walk through bars and count which touch # this is within the current
    supertrend trend segment.
    Adds: trend_touch_number, trend_start_idx, bars_since_trend_start.
    """
    df = df.copy()
    trend = df["st_trend"].values
    close = df["close"].values
    lower = df["st_lowerband"].values
    upper = df["st_upperband"].values
    n = len(df)

    touch_num = np.zeros(n, dtype=int)
    trend_start_arr = np.zeros(n, dtype=int)
    bars_since = np.zeros(n, dtype=int)

    seg_start = 0
    counter = 0

    for i in range(n):
        if i > 0 and trend[i] != trend[i - 1]:
            seg_start = i
            counter = 0

        bars_since[i] = i - seg_start
        trend_start_arr[i] = seg_start

        # Check if this bar is a touch
        if trend[i] == 1:  # uptrend, active band = lower
            band = lower[i]
        else:
            band = upper[i]

        if not np.isnan(band) and not np.isnan(close[i]) and close[i] != 0:
            dist = abs(close[i] - band) / close[i] * 100
            if dist <= touch_pct:
                counter += 1
                touch_num[i] = counter

    df["trend_touch_number"] = touch_num
    df["trend_start_idx"] = trend_start_arr
    df["bars_since_trend_start"] = bars_since
    return df


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
    df["rsi_sma"] = df["rsi"].rolling(rsi_period).mean()

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

    # OBV
    df["obv"] = obv(df["close"], df["volume"])

    # CCI
    df["cci"] = cci(df, period=20)

    # OI delta (only if OI column present - e.g. futures data)
    if "oi" in df.columns:
        df["oi_change"] = df["oi"].diff()

    # Candlestick pattern recognition
    df = detect_candlestick_patterns(df)

    # Divergence detection
    df = detect_divergences(df, pivot_order=5, lookback=14)

    # Fibonacci levels (requires st_trend)
    df = compute_fibonacci_levels(df)

    # Fractal zones
    df = compute_fractal_zones(df)

    return df


def compute_fractal_zones(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    close = df["close"].values
    atr_val = df["atr"].values
    trend = df["st_trend"].values
    lowerband = df["st_lowerband"].values
    upperband = df["st_upperband"].values
    volume = df["volume"].values

    # 6-period moving average of volume for fractal confirmation
    vol_ma = df["volume"].rolling(6).mean().values

    n = len(df)

    res_zone_high = np.nan
    res_zone_low = np.nan
    res_rejection_ratio = np.nan
    res_zone_width_pct = np.nan
    res_retest_count = 0
    res_in_retest = False

    sup_zone_high = np.nan
    sup_zone_low = np.nan
    sup_rejection_ratio = np.nan
    sup_zone_width_pct = np.nan
    sup_retest_count = 0
    sup_in_retest = False

    nearest_zone_price = np.full(n, np.nan)
    nearest_zone_type = np.full(n, None, dtype=object)
    zone_distance_pct = np.full(n, np.nan)
    zone_strength_score = np.full(n, np.nan)

    for i in range(n):
        ab = lowerband[i] if trend[i] == 1 else upperband[i]

        # 1. Update retests for existing zones up to current bar
        if not np.isnan(res_zone_low):
            approach_threshold = res_zone_low * (1 - 0.0015)
            if close[i] > res_zone_high:
                res_in_retest = False
            else:
                if high[i] >= approach_threshold:
                    if not res_in_retest:
                        res_retest_count += 1
                        res_in_retest = True
                else:
                    res_in_retest = False

        if not np.isnan(sup_zone_high):
            approach_threshold = sup_zone_high * (1 + 0.0015)
            if close[i] < sup_zone_low:
                sup_in_retest = False
            else:
                if low[i] <= approach_threshold:
                    if not sup_in_retest:
                        sup_retest_count += 1
                        sup_in_retest = True
                else:
                    sup_in_retest = False

        # 2. Check for new fractals formed at i-3 (requires 6 bars context)
        if i >= 5:
            p = i - 3
            # Swing High
            if (
                high[p] > high[p - 1]
                and high[p - 1] > high[p - 2]
                and high[p + 1] < high[p]
                and high[p + 2] < high[p + 1]
                and volume[p] > vol_ma[p]
            ):
                res_zone_high = high[p]
                res_zone_low = max(open_[p], close[p])
                res_rejection_ratio = (high[p] - low[p]) / atr_val[p] if atr_val[p] else 0
                zw_pct = abs(res_zone_high - res_zone_low) / res_zone_high
                res_zone_width_pct = max(zw_pct, 1e-6)
                res_retest_count = 0
                res_in_retest = False

            # Swing Low
            if (
                low[p] < low[p - 1]
                and low[p - 1] < low[p - 2]
                and low[p + 1] > low[p]
                and low[p + 2] > low[p + 1]
                and volume[p] > vol_ma[p]
            ):
                sup_zone_low = low[p]
                sup_zone_high = min(open_[p], close[p])
                sup_rejection_ratio = (high[p] - low[p]) / atr_val[p] if atr_val[p] else 0
                zw_pct = abs(sup_zone_high - sup_zone_low) / sup_zone_low
                sup_zone_width_pct = max(zw_pct, 1e-6)
                sup_retest_count = 0
                sup_in_retest = False

        # 3. Calculate distance and assign nearest zone attributes for current bar
        if not np.isnan(ab):
            dist_res = abs(ab - res_zone_high) if not np.isnan(res_zone_high) else np.inf
            dist_sup = abs(ab - sup_zone_low) if not np.isnan(sup_zone_low) else np.inf

            if np.isinf(dist_res) and np.isinf(dist_sup):
                continue

            if dist_res <= dist_sup:
                nearest_zone_price[i] = res_zone_high
                nearest_zone_type[i] = "Resistance"
                zone_distance_pct[i] = dist_res / ab * 100
                zone_strength_score[i] = (
                    res_rejection_ratio * (1 / res_zone_width_pct) * (1 + 0.1 * res_retest_count)
                )
            else:
                nearest_zone_price[i] = sup_zone_low
                nearest_zone_type[i] = "Support"
                zone_distance_pct[i] = dist_sup / ab * 100
                zone_strength_score[i] = (
                    sup_rejection_ratio * (1 / sup_zone_width_pct) * (1 + 0.1 * sup_retest_count)
                )

    df["nearest_fractal_zone_price"] = np.round(nearest_zone_price, 2)
    df["nearest_fractal_zone_type"] = nearest_zone_type
    df["fractal_zone_distance_pct"] = np.round(zone_distance_pct, 4)
    df["fractal_zone_strength_score"] = np.round(zone_strength_score, 4)

    # Calculate quartiles for buckets across the full series
    scores = df["fractal_zone_strength_score"].dropna()
    df["fractal_zone_strength_bucket"] = None
    if len(scores) > 3:
        try:
            buckets = pd.qcut(scores, 4, labels=["weak", "medium", "strong", "very_strong"])
        except ValueError:
            buckets = pd.qcut(
                scores.rank(method="first"), 4, labels=["weak", "medium", "strong", "very_strong"]
            )
        df.loc[scores.index, "fractal_zone_strength_bucket"] = buckets

    return df


def detect_supertrend_touches(df: pd.DataFrame, touch_pct: float) -> pd.DataFrame:
    """
    For every bar, checks whether close is within `touch_pct` % of the active
    supertrend band (lower band when trend is green/up, upper band when
    trend is red/down). Returns only the rows that qualify as a "touch",
    tagged with signal_type BUY_TOUCH / SELL_TOUCH.

    Also computes the touch number within the current supertrend trend segment.
    """
    # First compute touch numbers on the full dataframe
    d = compute_touch_number(df, touch_pct)

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

        bounce_pct = round(peak_bounce / touch_close * 100, 4) if touch_close != 0 else 0.0

        records.append(
            {
                "bounced": threshold_hit,
                "bounce_pct": bounce_pct,
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
    {"symbol": "RELIANCE", "exchange": "NSE"},
]


INTERVAL = sys.argv[1] if len(sys.argv) > 1 else "5m"
HTF_INTERVAL = sys.argv[2] if len(sys.argv) > 2 else None
START_DATE = "2024-01-01"
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
    "BANKNIFTY": 300,
    "RELIANCE": 10,
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

    if HTF_INTERVAL:
        print(f"  Fetching HTF {symbol} ({exchange}), interval={HTF_INTERVAL} ...")
        df_htf = fetch_history(client, symbol, exchange, HTF_INTERVAL, START_DATE, END_DATE)
        print(f"    {len(df_htf)} HTF bars fetched: {df_htf.index.min()} -> {df_htf.index.max()}")

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

    if HTF_INTERVAL:
        print("  Computing HTF indicators and merging ...")
        ind_htf = build_indicators(
            df_htf,
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

        # Prefix HTF columns to avoid collision
        ind_htf.columns = [f"htf_{c}" for c in ind_htf.columns]

        # Forward fill merge HTF onto LTF
        ind_htf_aligned = ind_htf.reindex(ind.index, method="ffill")
        ind = ind.join(ind_htf_aligned)

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
