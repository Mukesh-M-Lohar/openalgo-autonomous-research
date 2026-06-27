import json
import os

import numpy as np
import pandas as pd

DATA_DIR = "/root/openalgo-autonomous-research/autoresearch/data"
PARAMS_FILE = "/root/openalgo-autonomous-research/autoresearch/params.json"
METRICS_FILE = "/root/openalgo-autonomous-research/autoresearch/metrics.json"

SYMBOLS = [
    ("PROTEAN", "NSE", False),
    ("ZEEL", "NSE", True),
    ("BAHETI-SM", "NSE", False),
    ("CDSL", "NSE", True),
    ("ANGELONE", "NSE", True),
    ("SCI", "NSE", False),
    ("ACUTAAS", "NSE", False),
    ("SAMMAANCAP", "NSE", False),
    ("CLEAN", "NSE", False),
    ("COPPER31JUL26FUT", "MCX", True),
    ("COPPER31AUG26FUT", "MCX", True),
    ("MCX", "NSE", True),
    ("SBIN", "NSE", True),
    ("BSE", "NSE", False),
    ("NIFTY", "NSE_INDEX", True),
    ("BANKNIFTY", "NSE_INDEX", True),
]


def load_data(symbol, exchange):
    path = os.path.join(DATA_DIR, f"{symbol}_{exchange}_15m.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    return pd.DataFrame()


def compute_supertrend(df, period=10, multiplier=3.0):
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df))

    close = df["close"].values
    upper = upperband.values.copy()
    lower = lowerband.values.copy()

    for i in range(1, len(df)):
        # If previous direction was bullish (1)
        if direction[i - 1] == 1:
            if close[i] < lower[i - 1]:
                direction[i] = -1  # reversal
            else:
                direction[i] = 1
                # trail stop (cannot decrease lowerband)
                if lower[i] < lower[i - 1]:
                    lower[i] = lower[i - 1]
        # If previous direction was bearish (-1)
        elif direction[i - 1] == -1:
            if close[i] > upper[i - 1]:
                direction[i] = 1  # reversal
            else:
                direction[i] = -1
                # trail stop (cannot increase upperband)
                if upper[i] > upper[i - 1]:
                    upper[i] = upper[i - 1]
        else:
            # Initial direction based on close vs midpoint
            if close[i] > hl2.iloc[i]:
                direction[i] = 1
            else:
                direction[i] = -1

        supertrend[i] = lower[i] if direction[i] == 1 else upper[i]

    return supertrend, direction


def compute_indicators(df, params):
    df = df.copy()

    # RSI
    rsi_p = params.get("rsi_period", 14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_p, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # EMAs
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=params.get("ema_trend", 100), adjust=False).mean()

    # MACD
    df["macd_fast"] = df["close"].ewm(span=params.get("macd_fast", 12), adjust=False).mean()
    df["macd_slow"] = df["close"].ewm(span=params.get("macd_slow", 26), adjust=False).mean()
    df["macd"] = df["macd_fast"] - df["macd_slow"]
    df["macd_signal"] = df["macd"].ewm(span=params.get("macd_signal", 9), adjust=False).mean()

    # Bollinger Bands
    bb_p = params.get("bb_period", 20)
    bb_std = params.get("bb_std", 2.0)
    df["bb_middle"] = df["close"].rolling(bb_p).mean()
    std = df["close"].rolling(bb_p).std()
    df["bb_upper"] = df["bb_middle"] + bb_std * std
    df["bb_lower"] = df["bb_middle"] - bb_std * std

    # VWAP
    df["date"] = df.index.date
    df["pv"] = df["close"] * df["volume"]
    cum_pv = df.groupby("date")["pv"].cumsum()
    cum_vol = df.groupby("date")["volume"].cumsum()
    df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Supertrend
    st_period = params.get("st_period", 10)
    st_mult = params.get("st_mult", 3.0)
    supertrend, direction = compute_supertrend(df, st_period, st_mult)
    df["supertrend"] = supertrend
    df["supertrend_dir"] = direction

    # ORB (Opening Range Breakout) - First 15m candle of the day
    first_candles = df.groupby("date").first()
    df["orb_high"] = df["date"].map(first_candles["high"])
    df["orb_low"] = df["date"].map(first_candles["low"])

    return df


def backtest_intraday(df, params, allow_short=False, commission_pct=0.03, slippage_pct=0.02):
    df = compute_indicators(df, params)
    df = df.dropna(
        subset=[
            "rsi",
            "ema_trend",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "vwap",
            "atr",
            "supertrend",
            "orb_high",
        ]
    )
    if len(df) < 50:
        return [], pd.Series(100000.0, index=df.index)

    close_vals = df["close"].values
    open_vals = df["open"].values
    high_vals = df["high"].values
    low_vals = df["low"].values
    timestamps = df.index

    # Indicators
    rsi = df["rsi"].values
    ema_20 = df["ema_20"].values
    ema_50 = df["ema_50"].values
    ema_trend = df["ema_trend"].values
    macd = df["macd"].values
    macd_sig = df["macd_signal"].values
    bb_upper = df["bb_upper"].values
    bb_lower = df["bb_lower"].values
    vwap = df["vwap"].values
    atr = df["atr"].values
    supertrend = df["supertrend"].values
    st_dir = df["supertrend_dir"].values
    orb_high = df["orb_high"].values
    orb_low = df["orb_low"].values

    # Strategy parameters
    strat_name = params.get("strategy_name", "orb_vwap")
    tp_pct = params.get("tp", 0.015)
    sl_pct = params.get("sl", 0.01)
    alloc = params.get("allocation", 1.0)
    atr_mult = params.get("atr_mult", 2.0)

    position = 0
    entry_price = 0.0
    entry_idx = 0
    trades = []

    capital = 100000.0
    equity = np.full(len(df), capital)

    for i in range(1, len(df) - 1):
        time_idx = timestamps[i]
        is_exit_time = time_idx.hour == 15 and time_idx.minute == 15

        if position != 0:
            tp_price = entry_price * (1 + tp_pct) if position == 1 else entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 - sl_pct) if position == 1 else entry_price * (1 + sl_pct)

            if params.get("use_atr_stop", False):
                sl_price = (
                    entry_price - atr_mult * atr[entry_idx]
                    if position == 1
                    else entry_price + atr_mult * atr[entry_idx]
                )

            exit_triggered = False
            exit_price = close_vals[i]
            reason = "END_OF_DAY"

            if position == 1:
                if low_vals[i] <= sl_price:
                    exit_triggered = True
                    exit_price = sl_price
                    reason = "STOP_LOSS"
                elif high_vals[i] >= tp_price:
                    exit_triggered = True
                    exit_price = tp_price
                    reason = "TAKE_PROFIT"
            else:
                if high_vals[i] >= sl_price:
                    exit_triggered = True
                    exit_price = sl_price
                    reason = "STOP_LOSS"
                elif low_vals[i] <= tp_price:
                    exit_triggered = True
                    exit_price = tp_price
                    reason = "TAKE_PROFIT"

            if not exit_triggered and is_exit_time:
                exit_triggered = True
                exit_price = close_vals[i]
                reason = "FORCED_INTRADAY_EXIT"

            if exit_triggered:
                raw_pnl = (
                    (exit_price - entry_price) / entry_price
                    if position == 1
                    else (entry_price - exit_price) / entry_price
                )
                net_pnl = (raw_pnl * alloc) - (commission_pct + slippage_pct) * 2 / 100
                trade_profit = capital * net_pnl
                capital += trade_profit

                trades.append(
                    {
                        "entry_time": timestamps[entry_idx],
                        "exit_time": time_idx,
                        "direction": "LONG" if position == 1 else "SHORT",
                        "pnl_pct": net_pnl * 100,
                        "reason": reason,
                    }
                )
                equity[i:] = capital
                position = 0
            else:
                paper_pnl = (
                    (close_vals[i] - entry_price) / entry_price
                    if position == 1
                    else (entry_price - close_vals[i]) / entry_price
                )
                equity[i] = capital * (1 + paper_pnl * alloc)
        else:
            # We are flat, look for entry (restricted to after the opening range and before 15:00)
            if (
                not is_exit_time
                and time_idx.hour < 15
                and not (time_idx.hour == 9 and time_idx.minute == 15)
            ):
                signal = 0
                if strat_name == "orb_vwap":
                    # Long: Close crosses above ORB High AND price is above VWAP
                    if (
                        close_vals[i] > orb_high[i]
                        and close_vals[i] > vwap[i]
                        and close_vals[i - 1] <= orb_high[i - 1]
                    ):
                        signal = 1
                    # Short: Close crosses below ORB Low AND price is below VWAP
                    elif (
                        allow_short
                        and close_vals[i] < orb_low[i]
                        and close_vals[i] < vwap[i]
                        and close_vals[i - 1] >= orb_low[i - 1]
                    ):
                        signal = -1
                elif strat_name == "vwap_pullback":
                    # Pullback to VWAP: Low touches VWAP (or close to it) while price remains above VWAP (sloping up)
                    vwap_slope_up = vwap[i] > vwap[i - 1]
                    vwap_slope_down = vwap[i] < vwap[i - 1]
                    if close_vals[i] > vwap[i] and low_vals[i] <= vwap[i] * 1.002 and vwap_slope_up:
                        signal = 1
                    elif (
                        allow_short
                        and close_vals[i] < vwap[i]
                        and high_vals[i] >= vwap[i] * 0.998
                        and vwap_slope_down
                    ):
                        signal = -1
                elif strat_name == "supertrend_ema_pullback":
                    # Supertrend green, 20 EMA > 50 EMA, pullback to 20 EMA, RSI > 55
                    trend_bullish = st_dir[i] == 1 and ema_20[i] > ema_50[i]
                    trend_bearish = st_dir[i] == -1 and ema_20[i] < ema_50[i]
                    if trend_bullish and low_vals[i] <= ema_20[i] <= high_vals[i] and rsi[i] > 55:
                        signal = 1
                    elif (
                        allow_short
                        and trend_bearish
                        and low_vals[i] <= ema_20[i] <= high_vals[i]
                        and rsi[i] < 45
                    ):
                        signal = -1
                elif strat_name == "vwap_rsi":
                    if close_vals[i] < vwap[i] and rsi[i] < params.get("rsi_lower", 35):
                        signal = 1
                    elif (
                        allow_short
                        and close_vals[i] > vwap[i]
                        and rsi[i] > (100 - params.get("rsi_lower", 35))
                    ):
                        signal = -1
                elif strat_name == "macd_trend":
                    if (
                        close_vals[i] > ema_trend[i]
                        and macd[i] > macd_sig[i]
                        and macd[i - 1] <= macd_sig[i - 1]
                    ):
                        signal = 1
                    elif (
                        allow_short
                        and close_vals[i] < ema_trend[i]
                        and macd[i] < macd_sig[i]
                        and macd[i - 1] >= macd_sig[i - 1]
                    ):
                        signal = -1
                elif strat_name == "bb_reversion":
                    if close_vals[i] < bb_lower[i] and rsi[i] < params.get("rsi_lower", 35):
                        signal = 1
                    elif (
                        allow_short
                        and close_vals[i] > bb_upper[i]
                        and rsi[i] > (100 - params.get("rsi_lower", 35))
                    ):
                        signal = -1

                if signal != 0:
                    position = signal
                    entry_price = open_vals[i + 1]
                    entry_idx = i + 1
                    equity[i + 1] = capital

    return trades, pd.Series(equity, index=df.index)


def run_training():
    if not os.path.exists(PARAMS_FILE):
        print("Error: params.json not found.")
        return

    with open(PARAMS_FILE, "r") as f:
        params = json.load(f)

    portfolio_eqs = []
    total_trades_count = 0

    for sym, exch, allow_short in SYMBOLS:
        df = load_data(sym, exch)
        if df.empty:
            continue

        # Evaluate on the validation period (e.g. 2026-03-01 to 2026-06-27)
        df_test = df.loc["2026-03-01":]
        trades, eq = backtest_intraday(df_test, params, allow_short)

        df_eq = pd.DataFrame(eq, columns=["equity"])
        df_eq.index.name = None
        if not isinstance(df_eq.index, pd.DatetimeIndex):
            df_eq.index = pd.to_datetime(df_eq.index)
        df_eq["date"] = df_eq.index.date
        daily_eq = df_eq.groupby("date")["equity"].last()

        portfolio_eqs.append(daily_eq)
        total_trades_count += len(trades)

    if not portfolio_eqs or total_trades_count < 5:
        metrics = {
            "sharpe": -10.0,
            "daily_return_avg_pct": -10.0,
            "max_dd_pct": -100.0,
            "total_trades": 0,
        }
    else:
        port_df = pd.DataFrame(portfolio_eqs).T.sort_index().ffill().bfill()
        port_equity = port_df.mean(axis=1)

        daily_returns = port_equity.pct_change().dropna()
        avg_daily_ret = daily_returns.mean() * 100
        sharpe = (
            np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if daily_returns.std() > 0
            else 0
        )
        roll_max = port_equity.cummax()
        drawdown = (port_equity - roll_max) / roll_max
        max_dd = drawdown.min() * 100

        metrics = {
            "sharpe": float(sharpe),
            "daily_return_avg_pct": float(avg_daily_ret),
            "max_dd_pct": float(max_dd),
            "total_trades": int(total_trades_count),
            "equity_curve": port_equity.tolist(),
            "dates": [str(d) for d in port_equity.index],
        }

    # Write output metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    print(
        f"Evaluation complete. Sharpe={metrics['sharpe']:.4f} | Daily Ret={metrics['daily_return_avg_pct']:.4f}% | Trades={metrics['total_trades']}"
    )


if __name__ == "__main__":
    run_training()
