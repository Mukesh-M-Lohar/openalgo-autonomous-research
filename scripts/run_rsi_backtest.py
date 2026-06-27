import os
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
artifact_dir = Path("/root/.gemini/antigravity-cli/brain/6e9f0eb7-c7e7-47a3-98ad-4b7e70b45678")
artifact_dir.mkdir(parents=True, exist_ok=True)


# 1. Fetching historical data
def fetch_data(symbol, exchange, interval, start_date, end_date):
    host = "http://127.0.0.1:5000"
    api_key = os.environ.get("OPENALGO_API_KEY", "test")

    payload = {
        "apikey": api_key,
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "source": "db",
    }

    url = f"{host}/api/v1/history"
    try:
        response = httpx.post(url, json=payload, timeout=20.0)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            df = pd.DataFrame(data["data"])
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df = df.set_index("timestamp").sort_index()
                return df
    except Exception as e:
        print(f"Error fetching data for {interval}: {e}")
    return pd.DataFrame()


# 2. RSI Calculation
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# 3. Divergence Detection
def detect_rsi_divergence(
    df, rsi_period=14, pivot_lookback=2, max_bars=35, oversold=40, overbought=60
):
    rsi = compute_rsi(df, rsi_period)
    df = df.copy()
    df["rsi"] = rsi

    bull_signals = pd.Series(False, index=df.index)
    bear_signals = pd.Series(False, index=df.index)

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
            if len(troughs) >= 2:
                for prev_trough_idx in reversed(troughs[:-1]):
                    if trough_idx - prev_trough_idx > max_bars:
                        break
                    # Bullish divergence: price LL, RSI HL
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
            if len(peaks) >= 2:
                for prev_peak_idx in reversed(peaks[:-1]):
                    if peak_idx - prev_peak_idx > max_bars:
                        break
                    # Bearish divergence: price HH, RSI LH
                    if (
                        high_vals[peak_idx] > high_vals[prev_peak_idx]
                        and rsi_vals[peak_idx] < rsi_vals[prev_peak_idx]
                    ):
                        if rsi_vals[peak_idx] > overbought:
                            bear_signals.iloc[i] = True
                            break

    return bull_signals, bear_signals, rsi


# 4. Simulation Engine
def run_backtest(
    df,
    bull_signals,
    bear_signals,
    stop_loss_pct=0.5,
    take_profit_pct=1.0,
    commission_pct=0.03,
    slippage_pct=0.02,
    initial_capital=100000.0,
):
    trades = []
    position = None  # None, 'long', 'short'
    entry_price = 0.0
    entry_time = None
    max_price_since_entry = 0.0
    min_price_since_entry = 0.0
    bars_held = 0

    # Pre-calculate last bar of the day
    df = df.copy()
    df["date"] = df.index.date
    df["is_last_bar"] = df["date"] != df["date"].shift(-1)

    close_vals = df["close"].values
    high_vals = df["high"].values
    low_vals = df["low"].values
    timestamps = df.index
    is_last_bar = df["is_last_bar"].values

    equity = pd.Series(initial_capital, index=df.index)
    current_capital = initial_capital

    for i in range(1, len(df)):
        # If position is open, update equity based on paper value
        # Update capital with realized trades

        if position == "long":
            bars_held += 1
            max_price_since_entry = max(max_price_since_entry, high_vals[i])

            exit_price = None
            exit_reason = ""

            # Check Stop Loss
            if stop_loss_pct is not None:
                sl_price = entry_price * (1 - stop_loss_pct / 100)
                if low_vals[i] <= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"

            # Check Take Profit
            if exit_price is None and take_profit_pct is not None:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if high_vals[i] >= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"

            # Check opposite signal
            if exit_price is None and bear_signals.iloc[i]:
                exit_price = close_vals[i]
                exit_reason = "opposite_signal"

            # Check end of day forced exit
            if exit_price is None and is_last_bar[i]:
                exit_price = close_vals[i]
                exit_reason = "forced_intraday_exit"

            if exit_price is not None:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                cost_pct = (commission_pct + slippage_pct) * 2
                net_pnl_pct = pnl_pct - cost_pct
                pnl = current_capital * (net_pnl_pct / 100)
                current_capital += pnl

                trades.append(
                    {
                        "type": "long",
                        "entry_time": entry_time,
                        "exit_time": timestamps[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": net_pnl_pct,
                        "bars_held": bars_held,
                        "exit_reason": exit_reason,
                    }
                )
                position = None

        elif position == "short":
            bars_held += 1
            min_price_since_entry = min(min_price_since_entry, low_vals[i])

            exit_price = None
            exit_reason = ""

            # Check Stop Loss
            if stop_loss_pct is not None:
                sl_price = entry_price * (1 + stop_loss_pct / 100)
                if high_vals[i] >= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"

            # Check Take Profit
            if exit_price is None and take_profit_pct is not None:
                tp_price = entry_price * (1 - take_profit_pct / 100)
                if low_vals[i] <= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"

            # Check opposite signal
            if exit_price is None and bull_signals.iloc[i]:
                exit_price = close_vals[i]
                exit_reason = "opposite_signal"

            # Check end of day forced exit
            if exit_price is None and is_last_bar[i]:
                exit_price = close_vals[i]
                exit_reason = "forced_intraday_exit"

            if exit_price is not None:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                cost_pct = (commission_pct + slippage_pct) * 2
                net_pnl_pct = pnl_pct - cost_pct
                pnl = current_capital * (net_pnl_pct / 100)
                current_capital += pnl

                trades.append(
                    {
                        "type": "short",
                        "entry_time": entry_time,
                        "exit_time": timestamps[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": net_pnl_pct,
                        "bars_held": bars_held,
                        "exit_reason": exit_reason,
                    }
                )
                position = None

        # If no position open, check entries (don't enter on the last bar of the day)
        if position is None and not is_last_bar[i]:
            # Also don't enter in the last 30 minutes of the session for intraday
            hour = timestamps[i].hour
            minute = timestamps[i].minute
            # MCX closes at 17:55 UTC (23:25 IST)
            # 17:25 UTC is 30 mins before close
            is_near_close = (hour == 17 and minute >= 25) or (hour > 17)

            if not is_near_close:
                if bull_signals.iloc[i]:
                    position = "long"
                    entry_price = close_vals[i]
                    entry_time = timestamps[i]
                    max_price_since_entry = high_vals[i]
                    bars_held = 0
                elif bear_signals.iloc[i]:
                    position = "short"
                    entry_price = close_vals[i]
                    entry_time = timestamps[i]
                    min_price_since_entry = low_vals[i]
                    bars_held = 0

        equity.iloc[i] = current_capital

    return trades, equity


# 5. Performance Metrics Calculator
def calculate_performance_metrics(trades, equity, initial_capital=100000.0):
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "net_profit_pct": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
        }

    df_trades = pd.DataFrame(trades)

    total_trades = len(df_trades)
    wins = df_trades[df_trades["pnl_pct"] > 0]
    losses = df_trades[df_trades["pnl_pct"] <= 0]

    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    net_profit = equity.iloc[-1] - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100

    gross_profit = wins["pnl_pct"].sum()
    gross_loss = abs(losses["pnl_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Drawdown calculation
    roll_max = equity.cummax()
    drawdowns = (equity - roll_max) / roll_max * 100
    max_dd = abs(drawdowns.min())

    # Sharpe ratio (daily returns simplified)
    daily_equity = equity.resample("D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "net_profit_pct": net_profit_pct,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "avg_trade_pnl": df_trades["pnl_pct"].mean(),
        "long_trades": len(df_trades[df_trades["type"] == "long"]),
        "short_trades": len(df_trades[df_trades["type"] == "short"]),
    }


# 6. Main execution
def main():
    print("Starting RSI Divergence Backtest...")
    symbol = "COPPER31JUL26FUT"
    exchange = "MCX"
    start_date = "2026-06-01"
    end_date = "2026-06-27"
    intervals = ["5m", "15m", "1h"]

    # We will test multiple SL/TP combinations to find the best settings for each timeframe
    sl_tp_combinations = [
        {"sl": 0.3, "tp": 0.6},
        {"sl": 0.5, "tp": 1.0},
        {"sl": 0.5, "tp": 1.5},
        {"sl": 0.8, "tp": 1.6},
        {"sl": 1.0, "tp": 2.0},
        {"sl": None, "tp": None},  # Opposite signal exits only
    ]

    best_results = {}
    timeframe_data = {}

    plt.figure(figsize=(12, 7))

    for interval in intervals:
        print(f"\n--- Processing interval: {interval} ---")
        df = fetch_data(symbol, exchange, interval, start_date, end_date)
        if df.empty:
            print(f"No data for {interval}")
            continue

        timeframe_data[interval] = df

        # Detect signals (default parameters)
        # Using oversold = 40, overbought = 60 to capture meaningful swing regions
        bull, bear, rsi = detect_rsi_divergence(df, oversold=40, overbought=60)

        # Grid search SL/TP
        best_sharpe = -999
        best_opt = None
        best_trades = None
        best_equity = None

        for combo in sl_tp_combinations:
            trades, equity = run_backtest(
                df, bull, bear, stop_loss_pct=combo["sl"], take_profit_pct=combo["tp"]
            )
            metrics = calculate_performance_metrics(trades, equity)

            # Select best based on net profit and Sharpe ratio
            score = metrics["net_profit_pct"]
            if metrics["total_trades"] >= 3:  # Must have at least 3 trades to be representative
                if score > best_sharpe:
                    best_sharpe = score
                    best_opt = combo
                    best_metrics = metrics
                    best_trades = trades
                    best_equity = equity

        # If no configuration found or no trades, fall back to default SL=0.5, TP=1.0
        if best_opt is None:
            best_opt = {"sl": 0.5, "tp": 1.0}
            best_trades, best_equity = run_backtest(
                df, bull, bear, stop_loss_pct=0.5, take_profit_pct=1.0
            )
            best_metrics = calculate_performance_metrics(best_trades, best_equity)

        print(f"Best SL/TP Settings: SL={best_opt['sl']}%, TP={best_opt['tp']}%")
        print(
            f"Total Trades: {best_metrics['total_trades']} (Longs: {best_metrics['long_trades']}, Shorts: {best_metrics['short_trades']})"
        )
        print(f"Win Rate: {best_metrics['win_rate']:.2f}%")
        print(f"Net Profit: {best_metrics['net_profit_pct']:.2f}%")
        print(f"Max Drawdown: {best_metrics['max_drawdown']:.2f}%")
        print(f"Profit Factor: {best_metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")

        best_results[interval] = {
            "metrics": best_metrics,
            "opt": best_opt,
            "trades": best_trades,
            "equity": best_equity,
        }

        # Save trade log to CSV
        if best_trades:
            df_trades = pd.DataFrame(best_trades)
            csv_path = artifact_dir / f"trades_{interval}.csv"
            df_trades.to_csv(csv_path, index=False)
            print(f"Saved trade log to {csv_path.name}")

        # Plot equity curve
        plt.plot(
            best_equity.index,
            (best_equity / 100000.0 - 1) * 100,
            label=f"{interval} (SL={best_opt['sl']}%, TP={best_opt['tp']}%)",
        )

    plt.title(
        f"RSI Divergence Strategy Backtest on {symbol} (MCX)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()

    plot_path = artifact_dir / "equity_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved combined equity curve plot to {plot_path}")

    # Save overall summary JSON
    summary = {}
    for interval, res in best_results.items():
        summary[interval] = {
            "settings": {"stop_loss_pct": res["opt"]["sl"], "take_profit_pct": res["opt"]["tp"]},
            "metrics": {
                "total_trades": res["metrics"]["total_trades"],
                "long_trades": res["metrics"]["long_trades"],
                "short_trades": res["metrics"]["short_trades"],
                "win_rate_pct": float(res["metrics"]["win_rate"]),
                "net_profit_pct": float(res["metrics"]["net_profit_pct"]),
                "max_drawdown_pct": float(res["metrics"]["max_drawdown"]),
                "profit_factor": float(res["metrics"]["profit_factor"])
                if res["metrics"]["profit_factor"] != float("inf")
                else "inf",
                "sharpe_ratio": float(res["metrics"]["sharpe_ratio"]),
            },
        }

    import json

    with open(artifact_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved backtest summary to {artifact_dir / 'backtest_summary.json'}")


if __name__ == "__main__":
    main()
