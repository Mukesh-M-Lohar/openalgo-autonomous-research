import sys
from pathlib import Path

import pandas as pd
import vectorbt as vbt
from openalgo import ta

# Set base path
BASE_DIR = Path("/root/openalgo-autonomous-research")
sys.path.append(str(BASE_DIR))


def load_dataset(symbol="SBIN", exchange="NSE", interval="D"):
    """
    Loads historical dataset from cache or returns empty DataFrame.
    """
    if interval == "15m":
        filename = f"{symbol}_{exchange}_15m.csv"
        filepath = BASE_DIR / "data" / "cache_15m" / filename
    else:
        filename = f"{symbol}_{exchange}_D.csv"
        filepath = BASE_DIR / "data" / "cache" / filename

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    date_col = [c for c in df.columns if c.lower() in ["timestamp", "date", "datetime"]][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]
    return df


def run_ma_ribbon_stoch_backtest(
    df,
    symbol="SBIN",
    interval="D",
    initial_cash=1_000_000,
    ema_fast_len=9,
    ema_mid_len=30,
    ema_slow_len=100,
    stoch_k_len=140,
    stoch_smooth_k=10,
    stoch_d_len=30,
    stoch_mode="state",
):
    """
    Backtests MA Ribbon + Stochastic Strategy
    - EMA 9, EMA 30, EMA 100
    - Stochastic (140, 10, 30)

    Modes of Stochastic condition:
    - 'state': stoch_k > stoch_d for Long, stoch_k < stoch_d for Short at time of EMA cross
    - 'crossover': stoch_k crossed above stoch_d within last 10 bars
    - 'state_filtered': stoch_k > stoch_d AND stoch_k > 20 for Long
    """
    if df.empty or len(df) < max(ema_slow_len, stoch_k_len + stoch_d_len):
        print("Data frame too small for indicators.")
        return None

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_p = df["open"].values

    # Indicators via openalgo.ta
    ema9 = pd.Series(ta.ema(close, ema_fast_len), index=df.index)
    ema30 = pd.Series(ta.ema(close, ema_mid_len), index=df.index)
    ema100 = pd.Series(ta.ema(close, ema_slow_len), index=df.index)

    k_arr, d_arr = ta.stochastic(
        high, low, close, k_period=stoch_k_len, smooth_k=stoch_smooth_k, d_period=stoch_d_len
    )
    stoch_k = pd.Series(k_arr, index=df.index)
    stoch_d = pd.Series(d_arr, index=df.index)

    # EMA Crossovers
    ema9_cross_above_30 = ta.crossover(ema9, ema30)
    ema9_cross_below_30 = ta.crossunder(ema9, ema30)

    # Stochastic Buy / Sell conditions
    if stoch_mode == "state":
        stoch_buy = stoch_k > stoch_d
        stoch_sell = stoch_k < stoch_d
    elif stoch_mode == "crossover":
        # Stoch K crossed above D in past 10 bars
        stoch_cross_up = ta.crossover(stoch_k, stoch_d)
        stoch_cross_down = ta.crossunder(stoch_k, stoch_d)
        stoch_buy = stoch_cross_up.rolling(10).max() == 1
        stoch_sell = stoch_cross_down.rolling(10).max() == 1
    elif stoch_mode == "state_filtered":
        stoch_buy = (stoch_k > stoch_d) & (stoch_k > 20)
        stoch_sell = (stoch_k < stoch_d) & (stoch_k < 80)
    else:
        stoch_buy = stoch_k > stoch_d
        stoch_sell = stoch_k < stoch_d

    # Signals
    # Long Entry: Stoch Buy active AND EMA 9 crosses above EMA 30
    long_entry_raw = (stoch_buy & ema9_cross_above_30).fillna(False)
    # Long Exit: Close below 30 EMA
    long_exit_raw = (df["close"] < ema30).fillna(False)

    # Short Entry: Stoch Sell active AND EMA 9 crosses below EMA 30
    short_entry_raw = (stoch_sell & ema9_cross_below_30).fillna(False)
    # Short Exit: Close above 30 EMA
    short_exit_raw = (df["close"] > ema30).fillna(False)

    # Exrem cleaning to prevent redundant signals
    long_entries = ta.exrem(long_entry_raw, long_exit_raw)
    long_exits = ta.exrem(long_exit_raw, long_entry_raw)

    short_entries = ta.exrem(short_entry_raw, short_exit_raw)
    short_exits = ta.exrem(short_exit_raw, short_entry_raw)

    # Fees (Indian Market equity delivery standard: 0.111% fee + Rs 20 fixed fee per order)
    fees = 0.00111
    fixed_fees = 20.0
    freq = "1D" if interval == "D" else "15m"

    # VectorBT Portfolio (Long-only)
    pf_long = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=long_entries,
        exits=long_exits,
        init_cash=initial_cash,
        fees=fees,
        fixed_fees=fixed_fees,
        size=0.95,
        size_type="percent",
        direction="longonly",
        freq=freq,
    )

    # VectorBT Portfolio (Long + Short)
    pf_bi = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=initial_cash,
        fees=fees,
        fixed_fees=fixed_fees,
        size=0.95,
        size_type="percent",
        direction="both",
        freq=freq,
    )

    return {
        "df": df,
        "ema9": ema9,
        "ema30": ema30,
        "ema100": ema100,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "pf_long": pf_long,
        "pf_bi": pf_bi,
    }


if __name__ == "__main__":
    df = load_dataset("SBIN", "NSE", "D")
    res = run_ma_ribbon_stoch_backtest(df, symbol="SBIN", interval="D")
    if res:
        print("=== SBIN Daily Backtest Stats (Long Only) ===")
        print(res["pf_long"].stats())
        print("\n=== SBIN Daily Backtest Stats (Long + Short) ===")
        print(res["pf_bi"].stats())
