"""Price pattern detection library — 11 pattern types."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_pattern(df: pd.DataFrame, pattern_name: str, params: dict | None = None) -> pd.Series:
    """Detect a price pattern, returning boolean Series."""
    fn = PATTERN_FUNCTIONS.get(pattern_name)
    if fn is None:
        raise ValueError(f"Unknown pattern: {pattern_name}")
    return fn(df, params or {})


def _opening_range_breakout(df: pd.DataFrame, params: dict) -> pd.Series:
    """Opening range breakout — price breaks above first N bars' high."""
    n_bars = int(params.get("n_bars", 3))
    result = pd.Series(False, index=df.index)
    opening_high = df["high"].rolling(n_bars).max().shift(1)
    result = df["close"] > opening_high
    return result.fillna(False)


def _gap_up(df: pd.DataFrame, params: dict) -> pd.Series:
    """Gap up — today's open > yesterday's high."""
    min_gap_pct = float(params.get("min_gap_pct", 0.5))
    prev_high = df["high"].shift(1)
    gap_pct = (df["open"] - prev_high) / prev_high * 100
    return (gap_pct > min_gap_pct).fillna(False)


def _gap_down(df: pd.DataFrame, params: dict) -> pd.Series:
    """Gap down — today's open < yesterday's low."""
    min_gap_pct = float(params.get("min_gap_pct", 0.5))
    prev_low = df["low"].shift(1)
    gap_pct = (prev_low - df["open"]) / prev_low * 100
    return (gap_pct > min_gap_pct).fillna(False)


def _inside_bar(df: pd.DataFrame, params: dict) -> pd.Series:
    """Inside bar — current bar's range is within previous bar's range."""
    return ((df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))).fillna(False)


def _outside_bar(df: pd.DataFrame, params: dict) -> pd.Series:
    """Outside bar — current bar engulfs previous bar's range."""
    return ((df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))).fillna(False)


def _engulfing_bullish(df: pd.DataFrame, params: dict) -> pd.Series:
    """Bullish engulfing — current green candle engulfs previous red candle."""
    prev_red = df["close"].shift(1) < df["open"].shift(1)
    curr_green = df["close"] > df["open"]
    engulfs = (df["close"] > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))
    return (prev_red & curr_green & engulfs).fillna(False)


def _engulfing_bearish(df: pd.DataFrame, params: dict) -> pd.Series:
    """Bearish engulfing — current red candle engulfs previous green candle."""
    prev_green = df["close"].shift(1) > df["open"].shift(1)
    curr_red = df["close"] < df["open"]
    engulfs = (df["open"] > df["close"].shift(1)) & (df["close"] < df["open"].shift(1))
    return (prev_green & curr_red & engulfs).fillna(False)


def _pin_bar_bullish(df: pd.DataFrame, params: dict) -> pd.Series:
    """Bullish pin bar — long lower wick, small body at top."""
    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]
    lower_wick = pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]
    body_ratio = body / total_range.replace(0, np.nan)
    wick_ratio = lower_wick / total_range.replace(0, np.nan)
    return ((body_ratio < 0.35) & (wick_ratio > 0.6)).fillna(False)


def _pin_bar_bearish(df: pd.DataFrame, params: dict) -> pd.Series:
    """Bearish pin bar — long upper wick, small body at bottom."""
    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]
    upper_wick = df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
    body_ratio = body / total_range.replace(0, np.nan)
    wick_ratio = upper_wick / total_range.replace(0, np.nan)
    return ((body_ratio < 0.35) & (wick_ratio > 0.6)).fillna(False)


def _higher_high_higher_low(df: pd.DataFrame, params: dict) -> pd.Series:
    """Higher high + higher low pattern (uptrend continuation)."""
    lookback = int(params.get("lookback", 2))
    hh = df["high"] > df["high"].shift(lookback)
    hl = df["low"] > df["low"].shift(lookback)
    return (hh & hl).fillna(False)


def _lower_high_lower_low(df: pd.DataFrame, params: dict) -> pd.Series:
    """Lower high + lower low pattern (downtrend continuation)."""
    lookback = int(params.get("lookback", 2))
    lh = df["high"] < df["high"].shift(lookback)
    ll = df["low"] < df["low"].shift(lookback)
    return (lh & ll).fillna(False)


PATTERN_FUNCTIONS: dict[str, callable] = {
    "opening_range_breakout": _opening_range_breakout,
    "gap_up": _gap_up,
    "gap_down": _gap_down,
    "inside_bar": _inside_bar,
    "outside_bar": _outside_bar,
    "engulfing_bullish": _engulfing_bullish,
    "engulfing_bearish": _engulfing_bearish,
    "pin_bar_bullish": _pin_bar_bullish,
    "pin_bar_bearish": _pin_bar_bearish,
    "higher_high_higher_low": _higher_high_higher_low,
    "lower_high_lower_low": _lower_high_lower_low,
}

PATTERN_PARAM_RANGES: dict[str, dict[str, tuple[float, float, float]]] = {
    "opening_range_breakout": {"n_bars": (2, 5, 1)},
    "gap_up": {"min_gap_pct": (0.3, 2.0, 0.3)},
    "gap_down": {"min_gap_pct": (0.3, 2.0, 0.3)},
    "inside_bar": {},
    "outside_bar": {},
    "engulfing_bullish": {},
    "engulfing_bearish": {},
    "pin_bar_bullish": {},
    "pin_bar_bearish": {},
    "higher_high_higher_low": {"lookback": (1, 5, 1)},
    "lower_high_lower_low": {"lookback": (1, 5, 1)},
}
