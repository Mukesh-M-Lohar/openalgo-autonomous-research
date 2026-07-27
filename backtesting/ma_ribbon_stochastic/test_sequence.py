import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from openalgo import ta

BASE_DIR = Path("/root/openalgo-autonomous-research")
sys.path.append(str(BASE_DIR / "backtesting" / "ma_ribbon_stochastic"))
from backtest import load_data


def generate_sequence_signals(
    df,
    ema_fast_len=9,
    ema_mid_len=30,
    ema_slow_len=100,
    stoch_k_len=140,
    stoch_smooth_k=10,
    stoch_d_len=30,
):
    """
    Generates signals based on exact sequential trigger:
    1. Stochastic %K crosses above %D (Fresh Stoch Buy)
    2. Then NEXT EMA 9 cross above EMA 30 triggers Buy Entry (not pre-existing state)
    3. Reset Stoch setup after entry or on opposing Stoch cross.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    ema9 = pd.Series(ta.ema(close, ema_fast_len), index=df.index)
    ema30 = pd.Series(ta.ema(close, ema_mid_len), index=df.index)

    k_arr, d_arr = ta.stochastic(
        high, low, close, k_period=stoch_k_len, smooth_k=stoch_smooth_k, d_period=stoch_d_len
    )
    stoch_k = pd.Series(k_arr, index=df.index)
    stoch_d = pd.Series(d_arr, index=df.index)

    stoch_cross_up = np.asarray(ta.crossover(stoch_k, stoch_d), dtype=bool)
    stoch_cross_down = np.asarray(ta.crossunder(stoch_k, stoch_d), dtype=bool)

    ema9_cross_up = np.asarray(ta.crossover(ema9, ema30), dtype=bool)
    ema9_cross_down = np.asarray(ta.crossunder(ema9, ema30), dtype=bool)

    n = len(df)
    long_entries = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)

    stoch_buy_active = False
    stoch_sell_active = False

    for i in range(n):
        if stoch_cross_up[i]:
            stoch_buy_active = True
            stoch_sell_active = False
        elif stoch_cross_down[i]:
            stoch_sell_active = True
            stoch_buy_active = False

        if ema9_cross_up[i] and stoch_buy_active:
            long_entries[i] = True
            stoch_buy_active = (
                False  # Consume setup so only NEXT stoch cross can trigger next trade
            )

        if ema9_cross_down[i] and stoch_sell_active:
            short_entries[i] = True
            stoch_sell_active = False

    long_entries_s = pd.Series(long_entries, index=df.index)
    short_entries_s = pd.Series(short_entries, index=df.index)

    long_exits = (df["close"] < ema30).fillna(False)
    short_exits = (df["close"] > ema30).fillna(False)

    long_entries_clean = ta.exrem(long_entries_s, long_exits)
    long_exits_clean = ta.exrem(long_exits, long_entries_s)

    short_entries_clean = ta.exrem(short_entries_s, short_exits)
    short_exits_clean = ta.exrem(short_exits, short_entries_s)

    return {
        "long_entries": long_entries_clean,
        "long_exits": long_exits_clean,
        "short_entries": short_entries_clean,
        "short_exits": short_exits_clean,
        "ema9": ema9,
        "ema30": ema30,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
    }


if __name__ == "__main__":
    for sym, ex, tf in [
        ("NIFTY", "NSE_INDEX", "5m"),
        ("BANKNIFTY", "NSE_INDEX", "5m"),
        ("BSE", "NSE", "15m"),
    ]:
        df = load_data(sym, ex, tf)
        if df is not None:
            sig = generate_sequence_signals(df)
            print(f"=== {sym} ({tf}) Sequential Signals ===")
            print("Long Entries count:", sig["long_entries"].sum())
            print("Short Entries count:", sig["short_entries"].sum())

            pf_bi = vbt.Portfolio.from_signals(
                close=df["close"],
                entries=sig["long_entries"],
                exits=sig["long_exits"],
                short_entries=sig["short_entries"],
                short_exits=sig["short_exits"],
                init_cash=1_000_000,
                fees=0.00111,
                fixed_fees=20,
                size=950000,
                size_type="value",
            )
            print(
                f"Total Return: {pf_bi.total_return() * 100:.2f}% | Win Rate: {pf_bi.trades.win_rate() * 100:.1f}% | Trades: {pf_bi.trades.count()}"
            )
            print("-" * 50)
