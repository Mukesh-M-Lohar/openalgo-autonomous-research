import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from openalgo import ta

BASE_DIR = Path("/root/openalgo-autonomous-research")
sys.path.append(str(BASE_DIR))

OUTPUT_DIR = BASE_DIR / "backtesting" / "ma_ribbon_stochastic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(symbol="SBIN", exchange="NSE", timeframe="D"):
    """Loads historical CSV dataset from data cache or OpenAlgo API."""
    if timeframe == "5m":
        filepath = BASE_DIR / "data" / "cache_5m" / f"{symbol}_{exchange}_5m.csv"
    elif timeframe == "15m":
        filepath = BASE_DIR / "data" / "cache_15m" / f"{symbol}_{exchange}_15m.csv"
    else:
        filepath = BASE_DIR / "data" / "cache" / f"{symbol}_{exchange}_D.csv"

    if not filepath.exists():
        try:
            from dotenv import find_dotenv, load_dotenv
            from openalgo import api

            load_dotenv(find_dotenv())
            api_key = os.getenv("OPENALGO_API_KEY")
            host = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
            client = api(api_key=api_key, host=host)
            res_data = client.history(
                symbol=symbol,
                exchange=exchange,
                interval=timeframe,
                start_date="2024-01-01",
                end_date="2026-07-22",
            )
            if isinstance(res_data, dict) and res_data.get("status") == "success":
                df = pd.DataFrame(res_data["data"])
            elif isinstance(res_data, pd.DataFrame):
                df = res_data
            else:
                df = None

            if df is not None and not df.empty:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(filepath, index=False)
            else:
                return None
        except Exception as e:
            print(f"Error fetching from OpenAlgo API for {symbol} ({timeframe}): {e}")
            return None

    df = pd.read_csv(filepath)
    date_cols = [
        c for c in df.columns if c.lower() in ["timestamp", "date", "datetime", "index", "time"]
    ]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    else:
        df.index = pd.to_datetime(df.index)

    df.columns = [c.lower() for c in df.columns]

    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df


def run_backtest(df, symbol="SBIN", timeframe="D", init_cash=1_000_000, stoch_variant="sequence"):
    """
    Runs MA Ribbon + Stochastic Backtest.
    Parameters:
    - EMA lengths: 9, 30, 100
    - Stochastic: 140, 10, 30 (%K length, %K smoothing, %D smoothing)
    - Sequential Logic ('sequence'):
      1. Stochastic %K crosses above %D (Fresh Stoch Buy)
      2. Then NEXT EMA 9 cross above EMA 30 triggers Buy Entry.
      3. Long Exit: Close below EMA 30.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema9 = pd.Series(ta.ema(close.values, 9), index=df.index)
    ema30 = pd.Series(ta.ema(close.values, 30), index=df.index)
    ema100 = pd.Series(ta.ema(close.values, 100), index=df.index)

    k_arr, d_arr = ta.stochastic(
        high.values, low.values, close.values, k_period=140, smooth_k=10, d_period=30
    )
    stoch_k = pd.Series(k_arr, index=df.index)
    stoch_d = pd.Series(d_arr, index=df.index)

    stoch_cross_up = np.asarray(ta.crossover(stoch_k, stoch_d), dtype=bool)
    stoch_cross_down = np.asarray(ta.crossunder(stoch_k, stoch_d), dtype=bool)

    ema9_cross_up = np.asarray(ta.crossover(ema9, ema30), dtype=bool)
    ema9_cross_down = np.asarray(ta.crossunder(ema9, ema30), dtype=bool)

    n = len(df)

    if stoch_variant == "sequence":
        long_entry_arr = np.zeros(n, dtype=bool)
        short_entry_arr = np.zeros(n, dtype=bool)

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
                long_entry_arr[i] = True
                stoch_buy_active = False  # Consume trigger

            if ema9_cross_down[i] and stoch_sell_active:
                short_entry_arr[i] = True
                stoch_sell_active = False

        long_entry_raw = pd.Series(long_entry_arr, index=df.index)
        short_entry_raw = pd.Series(short_entry_arr, index=df.index)
    else:
        # State-based
        stoch_buy = stoch_k > stoch_d
        stoch_sell = stoch_k < stoch_d
        long_entry_raw = (stoch_buy & pd.Series(ema9_cross_up, index=df.index)).fillna(False)
        short_entry_raw = (stoch_sell & pd.Series(ema9_cross_down, index=df.index)).fillna(False)

    long_exit_raw = (close < ema30).fillna(False)
    short_exit_raw = (close > ema30).fillna(False)

    # Signal cleaning via openalgo.ta.exrem
    long_entries = ta.exrem(long_entry_raw, long_exit_raw)
    long_exits = ta.exrem(long_exit_raw, long_entry_raw)

    short_entries = ta.exrem(short_entry_raw, short_exit_raw)
    short_exits = ta.exrem(short_exit_raw, short_entry_raw)

    fees = 0.00111
    fixed_fees = 20.0
    freq = "1D" if timeframe == "D" else ("5m" if timeframe == "5m" else "15m")

    pf_long = vbt.Portfolio.from_signals(
        close=close,
        entries=long_entries,
        exits=long_exits,
        init_cash=init_cash,
        fees=fees,
        fixed_fees=fixed_fees,
        size=0.95,
        size_type="percent",
        direction="longonly",
        min_size=1,
        size_granularity=1,
        freq=freq,
    )

    pf_short = vbt.Portfolio.from_signals(
        close=close,
        entries=short_entries,
        exits=short_exits,
        init_cash=init_cash,
        fees=fees,
        fixed_fees=fixed_fees,
        size=0.95,
        size_type="percent",
        direction="shortonly",
        min_size=1,
        size_granularity=1,
        freq=freq,
    )

    pf_bi = vbt.Portfolio.from_signals(
        close=close,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=init_cash,
        fees=fees,
        fixed_fees=fixed_fees,
        size=init_cash * 0.95,
        size_type="value",
        min_size=1,
        size_granularity=1,
        freq=freq,
    )

    return {
        "df": df,
        "symbol": symbol,
        "timeframe": timeframe,
        "variant": stoch_variant,
        "close": close,
        "ema9": ema9,
        "ema30": ema30,
        "ema100": ema100,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "long_entries": long_entries,
        "long_exits": long_exits,
        "short_entries": short_entries,
        "short_exits": short_exits,
        "pf_long": pf_long,
        "pf_short": pf_short,
        "pf_bi": pf_bi,
    }


def generate_summary(res, df_bench=None):
    """Generates strategy summary metric table."""
    pf_long = res["pf_long"]
    pf_short = res["pf_short"]
    pf_bi = res["pf_bi"]
    timeframe = res["timeframe"]

    if df_bench is not None and not df_bench.empty:
        freq = "1D" if timeframe == "D" else ("5m" if timeframe == "5m" else "15m")
        bench_close = df_bench["close"].reindex(res["close"].index).ffill().bfill()
        pf_bench = vbt.Portfolio.from_holding(
            bench_close, init_cash=1_000_000, fees=0.00111, freq=freq
        )
        bench_return = f"{pf_bench.total_return() * 100:.2f}%"
        bench_sharpe = f"{pf_bench.sharpe_ratio():.2f}"
        bench_mdd = f"{pf_bench.max_drawdown() * 100:.2f}%"
    else:
        bench_return = "N/A"
        bench_sharpe = "N/A"
        bench_mdd = "N/A"

    metrics = {
        "Metric": [
            "Total Return (%)",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown (%)",
            "Win Rate (%)",
            "Total Trades",
            "Profit Factor",
            "Expectancy (INR)",
        ],
        "Long Only": [
            f"{pf_long.total_return() * 100:.2f}%",
            f"{pf_long.sharpe_ratio():.2f}",
            f"{pf_long.sortino_ratio():.2f}",
            f"{pf_long.max_drawdown() * 100:.2f}%",
            f"{pf_long.trades.win_rate() * 100:.1f}%",
            f"{pf_long.trades.count()}",
            f"{pf_long.trades.profit_factor():.2f}",
            f"Rs {pf_long.trades.expectancy():,.2f}" if pf_long.trades.count() > 0 else "Rs 0.00",
        ],
        "Short Only": [
            f"{pf_short.total_return() * 100:.2f}%",
            f"{pf_short.sharpe_ratio():.2f}",
            f"{pf_short.sortino_ratio():.2f}",
            f"{pf_short.max_drawdown() * 100:.2f}%",
            f"{pf_short.trades.win_rate() * 100:.1f}%",
            f"{pf_short.trades.count()}",
            f"{pf_short.trades.profit_factor():.2f}",
            f"Rs {pf_short.trades.expectancy():,.2f}" if pf_short.trades.count() > 0 else "Rs 0.00",
        ],
        "Long + Short": [
            f"{pf_bi.total_return() * 100:.2f}%",
            f"{pf_bi.sharpe_ratio():.2f}",
            f"{pf_bi.sortino_ratio():.2f}",
            f"{pf_bi.max_drawdown() * 100:.2f}%",
            f"{pf_bi.trades.win_rate() * 100:.1f}%",
            f"{pf_bi.trades.count()}",
            f"{pf_bi.trades.profit_factor():.2f}",
            f"Rs {pf_bi.trades.expectancy():,.2f}" if pf_bi.trades.count() > 0 else "Rs 0.00",
        ],
        "NIFTY Benchmark": [bench_return, bench_sharpe, "-", bench_mdd, "-", "-", "-", "-"],
    }

    return pd.DataFrame(metrics)
