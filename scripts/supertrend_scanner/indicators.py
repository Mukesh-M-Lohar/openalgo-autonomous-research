"""
Core indicator calculations + Supertrend band-touch detection.
Pure pandas/numpy, no external TA library dependency (keeps it portable
on any machine without needing talib compiled).
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Basic building blocks
# ----------------------------------------------------------------------


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

    # OI delta (only if OI column present - e.g. futures data)
    if "oi" in df.columns:
        df["oi_change"] = df["oi"].diff()

    return df


def detect_supertrend_touches(df: pd.DataFrame, touch_pct: float) -> pd.DataFrame:
    """
    For every bar, checks whether close is within `touch_pct` % of the PREVIOUS
    bar's active supertrend band (lower band when trend is green/up, upper band
    when trend is red/down). Uses shifted values to avoid lookahead bias —
    you only know the ST level after the prior bar completes.
    Returns only the rows that qualify as a "touch",
    tagged with signal_type BUY_TOUCH / SELL_TOUCH.
    """
    d = df.copy()

    # Use PREVIOUS bar's ST values to avoid lookahead bias
    prev_trend = d["st_trend"].shift(1)
    prev_lower = d["st_lowerband"].shift(1)
    prev_upper = d["st_upperband"].shift(1)

    is_up = prev_trend == 1
    is_down = prev_trend == -1

    band = np.where(is_up, prev_lower, prev_upper)
    dist_pct = (d["close"] - band).abs() / d["close"] * 100

    d["active_band"] = band
    d["band_distance_pct"] = dist_pct

    touched = dist_pct <= touch_pct
    d["signal_type"] = np.where(
        touched & is_up, "BUY_TOUCH", np.where(touched & is_down, "SELL_TOUCH", None)
    )

    return d[d["signal_type"].notna()].copy()
