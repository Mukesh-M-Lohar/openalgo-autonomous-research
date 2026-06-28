"""
strategy.py — THE THING BEING OPTIMIZED.

The agent (AI) modifies ONLY this file each iteration.
prepare.py calls generate_signals(df) and measures the result.

════════════════════════════════════════════════════════════
CONTRACT (DO NOT CHANGE THIS DOCSTRING):
  Input:  df — pd.DataFrame with columns: open, high, low, close, volume
              index is a DatetimeIndex (daily or intraday bars)
  Output: pd.DataFrame with same index and at least two boolean columns:
              entry : True on bars where we should go long
              exit  : True on bars where we should exit the long
════════════════════════════════════════════════════════════

Current iteration: 0  (baseline — simple SMA crossover)
Best score so far: N/A

Iteration log:
  #0 | score=N/A | baseline: 20/50 SMA crossover, RSI filter

The agent should:
  1. Read this file and the current score
  2. Think of ONE change that might improve the score
  3. Implement the change
  4. Let prepare.py measure the result
  5. Keep the change if score improved, revert if it did not
  6. Update the Iteration log above with the new score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# HELPER INDICATORS  (feel free to add/modify helpers)
# ══════════════════════════════════════════════════════════════════════════════


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS  (agent tunes these each iteration)
# ══════════════════════════════════════════════════════════════════════════════

FAST_PERIOD = 10  # fast SMA period
SLOW_PERIOD = 30  # slow SMA period
RSI_PERIOD = 14  # RSI lookback
RSI_ENTRY_MAX = 60  # enter only when RSI is below this (not overbought)
RSI_EXIT = 70  # exit when RSI exceeds this (overbought)
ATR_MULT_SL = 2.0  # stop-loss = entry - ATR_MULT_SL * ATR  (0 = disabled)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION — agent may change the logic inside, but NOT the signature
# ══════════════════════════════════════════════════════════════════════════════


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entry and exit signals.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.

    Returns:
        DataFrame with same index, columns:
            entry  (bool)  — go long on this bar's OPEN
            exit   (bool)  — close long on this bar's OPEN
    """

    close = df["close"]

    # ── Indicators ─────────────────────────────────────────────────────────────
    fast_ma = _sma(close, FAST_PERIOD)
    slow_ma = _sma(close, SLOW_PERIOD)
    rsi = _rsi(close, RSI_PERIOD)
    atr = _atr(df)

    # ── Entry conditions ────────────────────────────────────────────────────────
    # Primary: fast MA crossed above slow MA (golden cross)
    ma_cross_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))

    # Filter: RSI not overbought at entry time
    rsi_not_overbought = rsi < RSI_ENTRY_MAX

    entry = ma_cross_up & rsi_not_overbought

    # ── Exit conditions ─────────────────────────────────────────────────────────
    # Primary: fast MA crossed below slow MA (death cross)
    ma_cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    # Secondary: RSI overbought
    rsi_overbought = rsi > RSI_EXIT

    exit_sig = ma_cross_down | rsi_overbought

    # ── Build output DataFrame ──────────────────────────────────────────────────
    signals = pd.DataFrame(index=df.index)
    signals["entry"] = entry.fillna(False)
    signals["exit"] = exit_sig.fillna(False)

    return signals
