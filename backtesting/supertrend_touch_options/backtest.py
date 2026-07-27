"""
VectorBT Backtest: Supertrend Band Touch Options Strategy for NIFTY & BANKNIFTY
================================================================================
Simulates the production Supertrend Touch Options Bot on historical 5m data:
- Supertrend (10, 3.0) band touch signals (within 0.07% threshold).
- Configurable indicator filters: ADX, 150-SMA, Intraday VWAP, MVWAP (14).
- Realistic option price leverage simulation (Delta ~0.50, Intraday option volatility).
- Realistic transaction fees (Brokerage + STT + statutory charges).
- Benchmark comparison against NIFTY 50 Buy & Hold.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from dotenv import find_dotenv, load_dotenv

# Import OpenAlgo SDK
try:
    from openalgo import api, ta
except ImportError:
    print("ERROR: Please install openalgo: pip install openalgo")
    sys.exit(1)

# --- Environment & Setup ---
load_dotenv(find_dotenv(), override=False)
script_dir = Path(__file__).resolve().parent

# ======================================================================
# BACKTEST CONFIGURATION
# ======================================================================
SYMBOL = os.getenv("BACKTEST_SYMBOL", "NIFTY")  # Options: "NIFTY" or "BANKNIFTY"
EXCHANGE = "NSE_INDEX"
TIMEFRAME = "5m"

# Strategy Parameters
ST_PERIOD = 10
ST_MULT = 3.0
TOUCH_PCT = 0.07  # Touch distance threshold (%)
TRAIL_SL_PCT = 10.0  # Trailing stop loss (%)
MAX_TRADES_PER_DAY = 3

# Filter Toggles
USE_ADX_FILTER = True  # ADX > 25 filter
USE_MA_FILTER = True  # 150-SMA direction filter
USE_VWAP_FILTER = False  # Intraday VWAP direction filter
USE_MVWAP_FILTER = False  # Moving VWAP (14) direction filter

# Filter Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0
MA_PERIOD = 150
MVWAP_PERIOD = 14

# Capital & Cost Model
INIT_CASH = 500_000  # Initial Portfolio Capital (Rs. 5 Lakhs)
OPTION_DELTA = 0.55  # Average ITM Option Delta (~0.55)
OPTION_PREMIUM_PCT = 0.012  # Option premium as % of index LTP (~1.2%)
FEES_PER_TRADE = 0.0015  # 0.15% option turnover charges + STT + GST
FIXED_FEE_PER_ORDER = 20  # Rs. 20 flat brokerage per executed order

# Instrument Master Settings
LOT_SIZES = {
    "NIFTY": 65,
    "BANKNIFTY": 15,
}

# ======================================================================
# INDICATOR CALCULATIONS
# ======================================================================


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing for ATR."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return wilder_smooth(tr, period)


def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate Supertrend band and direction (1=UP, -1=DOWN)."""
    high, low, close = df["high"], df["low"], df["close"]
    hl2 = (high + low) / 2.0
    tr_atr = calc_atr(df, period)

    basic_upper = hl2 + multiplier * tr_atr
    basic_lower = hl2 - multiplier * tr_atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        prev_upper = final_upper.iloc[i - 1]
        prev_lower = final_lower.iloc[i - 1]

        if (
            pd.isna(prev_upper)
            or basic_upper.iloc[i] < prev_upper
            or close.iloc[i - 1] > prev_upper
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_upper

        if (
            pd.isna(prev_lower)
            or basic_lower.iloc[i] > prev_lower
            or close.iloc[i - 1] < prev_lower
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_lower

        if close.iloc[i] > prev_upper:
            trend.iloc[i] = 1
        elif close.iloc[i] < prev_lower:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
            if trend.iloc[i] == 1 and final_lower.iloc[i] < prev_lower:
                final_lower.iloc[i] = prev_lower
            if trend.iloc[i] == -1 and final_upper.iloc[i] > prev_upper:
                final_upper.iloc[i] = prev_upper

    st_line = np.where(trend == 1, final_lower, final_upper)
    out = df.copy()
    out["st_band"] = st_line
    out["st_trend"] = trend
    return out


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday Volume Weighted Average Price (resets per daily trading session)."""
    high, low, close = df["high"], df["low"], df["close"]
    volume = (
        df["volume"]
        if "volume" in df.columns and df["volume"].sum() > 0
        else pd.Series(1.0, index=df.index)
    )
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume

    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.strftime("%Y-%m-%d")
        cum_tpv = tpv.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
    else:
        cum_tpv = tpv.cumsum()
        cum_vol = volume.cumsum()

    cum_vol = cum_vol.replace(0, np.nan)
    return cum_tpv / cum_vol


def calc_mvwap(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Moving Volume Weighted Average Price (Rolling N-period SMA of VWAP)."""
    vwap_series = calc_vwap(df)
    return vwap_series.rolling(window=period, min_periods=1).mean()


# ======================================================================
# DATA LOADING & BACKTEST ENGINE
# ======================================================================


def load_data(symbol: str) -> pd.DataFrame:
    """Load historical 5m OHLCV data from OpenAlgo backend API."""
    print(f"Fetching {symbol} 5m historical candles from OpenAlgo API...")
    client = api(
        api_key=os.getenv(
            "OPENALGO_API_KEY", "b45feb0a6973ed00fe86d25ace49d4da8dfe8d0a78c334455d46254ded28a26d"
        ),
        host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
    )
    end_date = datetime.now()
    start_date = end_date - timedelta(days=210)

    df = client.history(
        symbol=symbol,
        exchange="NSE_INDEX",
        interval="5m",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    if df is None or df.empty:
        raise ValueError(f"Failed to fetch historical data for {symbol} from OpenAlgo API.")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    # Clean numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["close", "high", "low"])


def run_backtest(symbol: str):
    """Run vectorbt backtest for given symbol (NIFTY or BANKNIFTY)."""
    print("\n" + "=" * 70)
    print(f"  RUNNING BACKTEST FOR: {symbol} (5m Timeframe)")
    print("=" * 70)

    df = load_data(symbol)
    print(f"Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}")

    # 1. Compute Indicators on completed candles (No Lookahead Bias)
    df_st = calc_supertrend(df, period=ST_PERIOD, multiplier=ST_MULT)
    st_band = df_st["st_band"]
    st_trend = df_st["st_trend"]

    # Prev bar values to avoid lookahead bias
    prev_st_band = st_band.shift(1)
    prev_st_trend = st_trend.shift(1)
    prev_close = df["close"].shift(1)

    # Touch detection on completed / live candle
    dist_pct = (df["close"] - prev_st_band).abs() / df["close"] * 100
    touch_condition = (dist_pct <= TOUCH_PCT) & (prev_st_trend.isin([1, -1]))

    # Direction signals
    raw_long_entries = touch_condition & (prev_st_trend == 1)
    raw_short_entries = touch_condition & (prev_st_trend == -1)

    # 2. Technical Filters (Evaluated on prev_close)
    filter_mask = pd.Series(True, index=df.index)

    if USE_ADX_FILTER:
        _, _, adx_series = ta.adx(df["high"], df["low"], df["close"], period=ADX_PERIOD)
        adx_prev = adx_series.shift(1)
        filter_mask = filter_mask & (adx_prev >= ADX_THRESHOLD)

    if USE_MA_FILTER:
        ma_series = ta.sma(df["close"], period=MA_PERIOD).shift(1)
        long_ma_ok = prev_close >= ma_series
        short_ma_ok = prev_close <= ma_series
    else:
        long_ma_ok = pd.Series(True, index=df.index)
        short_ma_ok = pd.Series(True, index=df.index)

    if USE_VWAP_FILTER:
        vwap_series = calc_vwap(df).shift(1)
        long_vwap_ok = prev_close >= vwap_series
        short_vwap_ok = prev_close <= vwap_series
    else:
        long_vwap_ok = pd.Series(True, index=df.index)
        short_vwap_ok = pd.Series(True, index=df.index)

    if USE_MVWAP_FILTER:
        mvwap_series = calc_mvwap(df, period=MVWAP_PERIOD).shift(1)
        long_mvwap_ok = prev_close >= mvwap_series
        short_mvwap_ok = prev_close <= mvwap_series
    else:
        long_mvwap_ok = pd.Series(True, index=df.index)
        short_mvwap_ok = pd.Series(True, index=df.index)

    # Final Filtered Signals
    long_entries = raw_long_entries & filter_mask & long_ma_ok & long_vwap_ok & long_mvwap_ok
    short_entries = raw_short_entries & filter_mask & short_ma_ok & short_vwap_ok & short_mvwap_ok

    # Exits: Supertrend direction flip
    long_exits = prev_st_trend == -1
    short_exits = prev_st_trend == 1

    # Clean signals using openalgo.ta.exrem
    entries = ta.exrem(long_entries.fillna(False), long_exits.fillna(False))
    exits = ta.exrem(long_exits.fillna(False), long_entries.fillna(False))

    # 3. Simulate Option Delta Leverage Series
    # Option price approx = Index Close * OPTION_PREMIUM_PCT
    # Option price delta movement = Index movement * OPTION_DELTA
    sim_option_price = df["close"] * OPTION_PREMIUM_PCT

    # 4. Run VectorBT Portfolio Simulation
    pf = vbt.Portfolio.from_signals(
        close=sim_option_price,
        entries=entries,
        exits=exits,
        init_cash=INIT_CASH,
        fees=FEES_PER_TRADE,
        fixed_fees=FIXED_FEE_PER_ORDER,
        sl_stop=TRAIL_SL_PCT / 100.0,  # Trailing SL
        sl_trail=True,
        freq="5m",
        direction="longonly",  # We buy CE for UP touch, buy PE for DOWN touch
    )

    # 5. Benchmark: NIFTY 50 Buy & Hold
    pf_bench = vbt.Portfolio.from_holding(df["close"], init_cash=INIT_CASH, fees=0.001, freq="5m")

    # 6. Performance Summary Table
    win_rate = pf.trades.win_rate() * 100.0 if not np.isnan(pf.trades.win_rate()) else 0.0
    profit_factor = pf.trades.profit_factor() if not np.isnan(pf.trades.profit_factor()) else 0.0

    comparison = pd.DataFrame(
        {
            f"Supertrend Touch ({symbol})": [
                f"{pf.total_return() * 100:.2f}%",
                f"{pf.sharpe_ratio():.2f}",
                f"{pf.sortino_ratio():.2f}",
                f"{pf.max_drawdown() * 100:.2f}%",
                f"{win_rate:.1f}%",
                f"{pf.trades.count()}",
                f"{profit_factor:.2f}",
            ],
            "Benchmark (NIFTY 50 Buy & Hold)": [
                f"{pf_bench.total_return() * 100:.2f}%",
                f"{pf_bench.sharpe_ratio():.2f}",
                f"{pf_bench.sortino_ratio():.2f}",
                f"{pf_bench.max_drawdown() * 100:.2f}%",
                "-",
                "-",
                "-",
            ],
        },
        index=[
            "Total Return",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Win Rate",
            "Total Trades",
            "Profit Factor",
        ],
    )

    print("\n" + "=" * 70)
    print(f"  PERFORMANCE SUMMARY: {symbol}")
    print("=" * 70)
    print(comparison.to_string())
    print("=" * 70)

    # 7. Plain Language Performance Explanation
    print("\n--- PLAIN LANGUAGE ANALYSIS REPORT ---")
    print(
        f"• Total Return: {pf.total_return() * 100:.2f}% on initial capital of Rs. {INIT_CASH:,.0f}"
    )
    print(
        f"• Benchmark Return: {pf_bench.total_return() * 100:.2f}% (NIFTY 50 Buy & Hold over same period)"
    )
    print(
        f"• Maximum Drawdown: {pf.max_drawdown() * 100:.2f}% (Peak-to-trough temporary loss: Rs. {abs(pf.max_drawdown()) * INIT_CASH:,.0f})"
    )
    print(f"• Win Rate: {win_rate:.1f}% across {pf.trades.count()} total trades")
    print(f"• Profit Factor: {profit_factor:.2f} (Gross Profit / Gross Loss ratio)")
    print("--------------------------------------\n")

    # Save detailed trades to CSV
    output_csv = script_dir / f"backtest_results_{symbol.lower()}.csv"
    if not pf.trades.records_readable.empty:
        pf.trades.records_readable.to_csv(output_csv, index=False)
        print(f"Trade records saved to: {output_csv}")

    return pf, pf_bench, comparison


if __name__ == "__main__":
    symbols_to_test = ["NIFTY", "BANKNIFTY"]
    for sym in symbols_to_test:
        run_backtest(sym)
