"""Indicator computation library — all 19+ supported technical indicators.

Each function takes a DataFrame with OHLCV columns and returns a Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.models.strategy import IndicatorType, PriceSource


def get_source_series(df: pd.DataFrame, source: PriceSource) -> pd.Series:
    """Extract the price source series from OHLCV DataFrame."""
    if source == PriceSource.CLOSE:
        return df["close"]
    elif source == PriceSource.OPEN:
        return df["open"]
    elif source == PriceSource.HIGH:
        return df["high"]
    elif source == PriceSource.LOW:
        return df["low"]
    elif source == PriceSource.VOLUME:
        return df["volume"]
    elif source == PriceSource.HL2:
        return (df["high"] + df["low"]) / 2
    elif source == PriceSource.HLC3:
        return (df["high"] + df["low"] + df["close"]) / 3
    elif source == PriceSource.OHLC4:
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    return df["close"]


def compute_indicator(
    df: pd.DataFrame, indicator_type: IndicatorType, params: dict, source: PriceSource
) -> pd.Series:
    """Compute any indicator given type, parameters, and source."""
    src = get_source_series(df, source)
    fn = INDICATOR_FUNCTIONS.get(indicator_type)
    if fn is None:
        raise ValueError(f"Unknown indicator: {indicator_type}")
    return fn(df, src, params)


# --- Individual indicator implementations ---


def _sma(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    return src.rolling(window=period).mean()


def _ema(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    return src.ewm(span=period, adjust=False).mean()


def _wma(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    weights = np.arange(1, period + 1, dtype=float)
    return src.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _vwma(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    vol = df["volume"]
    return (src * vol).rolling(period).sum() / vol.rolling(period).sum()


def _rsi(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 14))
    delta = src.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    fast = int(params.get("fast_period", 12))
    slow = int(params.get("slow_period", 26))
    ema_fast = src.ewm(span=fast, adjust=False).mean()
    ema_slow = src.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _macd_signal(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    int(params.get("fast_period", 12))
    int(params.get("slow_period", 26))
    signal_period = int(params.get("signal_period", 9))
    macd_line = _macd(df, src, params)
    return macd_line.ewm(span=signal_period, adjust=False).mean()


def _macd_hist(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    macd_line = _macd(df, src, params)
    signal = _macd_signal(df, src, params)
    return macd_line - signal


def _adx(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 14))
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _atr(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 14))
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _bbands_upper(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    std_dev = float(params.get("std_dev", 2.0))
    mid = src.rolling(period).mean()
    std = src.rolling(period).std()
    return mid + std_dev * std


def _bbands_middle(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    return src.rolling(period).mean()


def _bbands_lower(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    std_dev = float(params.get("std_dev", 2.0))
    mid = src.rolling(period).mean()
    std = src.rolling(period).std()
    return mid - std_dev * std


def _keltner_upper(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    multiplier = float(params.get("multiplier", 1.5))
    mid = src.ewm(span=period, adjust=False).mean()
    atr = _atr(df, src, {"period": period})
    return mid + multiplier * atr


def _keltner_lower(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    multiplier = float(params.get("multiplier", 1.5))
    mid = src.ewm(span=period, adjust=False).mean()
    atr = _atr(df, src, {"period": period})
    return mid - multiplier * atr


def _donchian_upper(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    return df["high"].rolling(period).max()


def _donchian_lower(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    return df["low"].rolling(period).min()


def _supertrend(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 10))
    multiplier = float(params.get("multiplier", 3.0))
    hl2 = (df["high"] + df["low"]) / 2
    atr = _atr(df, src, {"period": period})

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]

    return supertrend


def _stoch_k(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    k_period = int(params.get("k_period", 14))
    smooth_k = int(params.get("smooth_k", 3))
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    fast_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    return fast_k.rolling(smooth_k).mean()


def _stoch_d(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    d_period = int(params.get("d_period", 3))
    k = _stoch_k(df, src, params)
    return k.rolling(d_period).mean()


def _cci(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)


def _roc(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 12))
    return ((src - src.shift(period)) / src.shift(period).replace(0, np.nan)) * 100


def _momentum(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 10))
    return src - src.shift(period)


def _vwap(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _obv(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    sign = np.sign(df["close"].diff())
    return (sign * df["volume"]).cumsum()


def _volume_sma(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 20))
    return df["volume"].rolling(period).mean()


def _price(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    return src


# --- Registry ---

INDICATOR_FUNCTIONS: dict = {
    IndicatorType.SMA: _sma,
    IndicatorType.EMA: _ema,
    IndicatorType.WMA: _wma,
    IndicatorType.VWMA: _vwma,
    IndicatorType.RSI: _rsi,
    IndicatorType.MACD: _macd,
    IndicatorType.MACD_SIGNAL: _macd_signal,
    IndicatorType.MACD_HIST: _macd_hist,
    IndicatorType.ADX: _adx,
    IndicatorType.ATR: _atr,
    IndicatorType.BBANDS_UPPER: _bbands_upper,
    IndicatorType.BBANDS_MIDDLE: _bbands_middle,
    IndicatorType.BBANDS_LOWER: _bbands_lower,
    IndicatorType.KELTNER_UPPER: _keltner_upper,
    IndicatorType.KELTNER_LOWER: _keltner_lower,
    IndicatorType.DONCHIAN_UPPER: _donchian_upper,
    IndicatorType.DONCHIAN_LOWER: _donchian_lower,
    IndicatorType.SUPERTREND: _supertrend,
    IndicatorType.STOCH_K: _stoch_k,
    IndicatorType.STOCH_D: _stoch_d,
    IndicatorType.CCI: _cci,
    IndicatorType.ROC: _roc,
    IndicatorType.MOMENTUM: _momentum,
    IndicatorType.VWAP: _vwap,
    IndicatorType.OBV: _obv,
    IndicatorType.VOLUME_SMA: _volume_sma,
    IndicatorType.PRICE: _price,
}


# --- Parameter ranges for generation ---

INDICATOR_PARAM_RANGES: dict[IndicatorType, dict[str, tuple[float, float, float]]] = {
    # (min, max, step)
    IndicatorType.SMA: {"period": (5, 200, 5)},
    IndicatorType.EMA: {"period": (5, 200, 5)},
    IndicatorType.WMA: {"period": (5, 200, 5)},
    IndicatorType.VWMA: {"period": (5, 50, 5)},
    IndicatorType.RSI: {"period": (7, 28, 7)},
    IndicatorType.MACD: {
        "fast_period": (8, 16, 2),
        "slow_period": (20, 30, 2),
        "signal_period": (7, 12, 1),
    },
    IndicatorType.MACD_SIGNAL: {
        "fast_period": (8, 16, 2),
        "slow_period": (20, 30, 2),
        "signal_period": (7, 12, 1),
    },
    IndicatorType.MACD_HIST: {
        "fast_period": (8, 16, 2),
        "slow_period": (20, 30, 2),
        "signal_period": (7, 12, 1),
    },
    IndicatorType.ADX: {"period": (7, 28, 7)},
    IndicatorType.ATR: {"period": (7, 28, 7)},
    IndicatorType.BBANDS_UPPER: {"period": (10, 30, 5), "std_dev": (1.5, 3.0, 0.5)},
    IndicatorType.BBANDS_MIDDLE: {"period": (10, 30, 5)},
    IndicatorType.BBANDS_LOWER: {"period": (10, 30, 5), "std_dev": (1.5, 3.0, 0.5)},
    IndicatorType.KELTNER_UPPER: {"period": (10, 30, 5), "multiplier": (1.0, 3.0, 0.5)},
    IndicatorType.KELTNER_LOWER: {"period": (10, 30, 5), "multiplier": (1.0, 3.0, 0.5)},
    IndicatorType.DONCHIAN_UPPER: {"period": (10, 50, 5)},
    IndicatorType.DONCHIAN_LOWER: {"period": (10, 50, 5)},
    IndicatorType.SUPERTREND: {"period": (7, 14, 1), "multiplier": (1.5, 4.0, 0.5)},
    IndicatorType.STOCH_K: {"k_period": (7, 21, 7), "smooth_k": (3, 5, 1)},
    IndicatorType.STOCH_D: {"k_period": (7, 21, 7), "smooth_k": (3, 5, 1), "d_period": (3, 5, 1)},
    IndicatorType.CCI: {"period": (10, 30, 5)},
    IndicatorType.ROC: {"period": (5, 20, 5)},
    IndicatorType.MOMENTUM: {"period": (5, 20, 5)},
    IndicatorType.VWAP: {},
    IndicatorType.OBV: {},
    IndicatorType.VOLUME_SMA: {"period": (10, 50, 10)},
    IndicatorType.PRICE: {},
}

# Categorization for generation filtering
INDICATOR_CATEGORIES: dict[str, list[IndicatorType]] = {
    "trend": [
        IndicatorType.SMA,
        IndicatorType.EMA,
        IndicatorType.WMA,
        IndicatorType.MACD,
        IndicatorType.MACD_SIGNAL,
        IndicatorType.MACD_HIST,
        IndicatorType.ADX,
        IndicatorType.SUPERTREND,
    ],
    "momentum": [
        IndicatorType.RSI,
        IndicatorType.STOCH_K,
        IndicatorType.STOCH_D,
        IndicatorType.CCI,
        IndicatorType.ROC,
        IndicatorType.MOMENTUM,
    ],
    "volatility": [
        IndicatorType.ATR,
        IndicatorType.BBANDS_UPPER,
        IndicatorType.BBANDS_LOWER,
        IndicatorType.BBANDS_MIDDLE,
        IndicatorType.KELTNER_UPPER,
        IndicatorType.KELTNER_LOWER,
        IndicatorType.DONCHIAN_UPPER,
        IndicatorType.DONCHIAN_LOWER,
    ],
    "volume": [
        IndicatorType.VWMA,
        IndicatorType.VWAP,
        IndicatorType.OBV,
        IndicatorType.VOLUME_SMA,
    ],
}
