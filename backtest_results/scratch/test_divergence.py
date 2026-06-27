import os

import httpx
import numpy as np
import pandas as pd


def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_rsi_divergence(
    df, rsi_period=14, pivot_lookback=2, max_bars=35, oversold=40, overbought=60
):
    rsi = compute_rsi(df, rsi_period)
    df = df.copy()
    df["rsi"] = rsi

    bull_signals = pd.Series(False, index=df.index)
    bear_signals = pd.Series(False, index=df.index)

    # Store indices of peaks and troughs
    troughs = []
    peaks = []

    rsi_vals = df["rsi"].values
    low_vals = df["low"].values
    high_vals = df["high"].values

    for i in range(pivot_lookback * 2, len(df)):
        # Check if i - pivot_lookback is a trough
        is_trough = True
        is_peak = True
        for offset in range(1, pivot_lookback + 1):
            if (
                rsi_vals[i - pivot_lookback] >= rsi_vals[i - pivot_lookback - offset]
                or rsi_vals[i - pivot_lookback] >= rsi_vals[i - pivot_lookback + offset]
            ):
                is_trough = False
            if (
                rsi_vals[i - pivot_lookback] <= rsi_vals[i - pivot_lookback - offset]
                or rsi_vals[i - pivot_lookback] <= rsi_vals[i - pivot_lookback + offset]
            ):
                is_peak = False

        if is_trough:
            trough_idx = i - pivot_lookback
            troughs.append(trough_idx)
            # Compare with previous troughs
            if len(troughs) >= 2:
                # Find a trough within max_bars range
                for prev_trough_idx in reversed(troughs[:-1]):
                    if trough_idx - prev_trough_idx > max_bars:
                        break
                    # Bullish divergence check
                    # Price makes lower low, RSI makes higher low
                    if (
                        low_vals[trough_idx] < low_vals[prev_trough_idx]
                        and rsi_vals[trough_idx] > rsi_vals[prev_trough_idx]
                    ):
                        if rsi_vals[trough_idx] < oversold:
                            bull_signals.iloc[i] = True
                            break

        if is_peak:
            peak_idx = i - pivot_lookback
            peaks.append(peak_idx)
            # Compare with previous peaks
            if len(peaks) >= 2:
                for prev_peak_idx in reversed(peaks[:-1]):
                    if peak_idx - prev_peak_idx > max_bars:
                        break
                    # Bearish divergence check
                    # Price makes higher high, RSI makes lower high
                    if (
                        high_vals[peak_idx] > high_vals[prev_peak_idx]
                        and rsi_vals[peak_idx] < rsi_vals[prev_peak_idx]
                    ):
                        if rsi_vals[peak_idx] > overbought:
                            bear_signals.iloc[i] = True
                            break

    return bull_signals, bear_signals


# Fetch cached data and run
host = "http://127.0.0.1:5000"
api_key = os.environ.get("OPENALGO_API_KEY", "test")
symbol = "COPPER31JUL26FUT"
exchange = "MCX"
intervals = ["5m", "15m", "1h"]

for interval in intervals:
    payload = {
        "apikey": api_key,
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "start_date": "2025-01-01",
        "end_date": "2026-06-27",
        "source": "db",
    }
    response = httpx.post(f"{host}/api/v1/history", json=payload)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp").sort_index()

        bull, bear = detect_rsi_divergence(df)
        print(f"Interval {interval}: Bullish signals: {bull.sum()}, Bearish signals: {bear.sum()}")
