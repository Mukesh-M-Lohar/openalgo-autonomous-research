import json
import os
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
artifact_dir = Path("/root/.gemini/antigravity-cli/brain/6e9f0eb7-c7e7-47a3-98ad-4b7e70b45678")
artifact_dir.mkdir(parents=True, exist_ok=True)


# Reuse fetch and compute functions
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
    try:
        response = httpx.post(f"{host}/api/v1/history", json=payload, timeout=20.0)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                df = pd.DataFrame(data["data"])
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    return df.set_index("timestamp").sort_index()
    except Exception as e:
        print(f"Error fetching data: {e}")
    return pd.DataFrame()


def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detect_rsi_divergence(df, rsi, pivot_lookback=2, max_bars=35, oversold=40, overbought=60):
    bull_signals = pd.Series(False, index=df.index)
    bear_signals = pd.Series(False, index=df.index)

    troughs = []
    peaks = []

    rsi_vals = rsi.values
    low_vals = df["low"].values
    high_vals = df["high"].values

    for i in range(pivot_lookback * 2, len(df)):
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
                    if (
                        high_vals[peak_idx] > high_vals[prev_peak_idx]
                        and rsi_vals[peak_idx] < rsi_vals[prev_peak_idx]
                    ):
                        if rsi_vals[peak_idx] > overbought:
                            bear_signals.iloc[i] = True
                            break

    return bull_signals, bear_signals


def run_backtest(
    df,
    bull_signals,
    bear_signals,
    stop_loss_pct,
    take_profit_pct,
    commission_pct=0.03,
    slippage_pct=0.02,
    initial_capital=100000.0,
):
    trades = []
    position = None
    entry_price = 0.0
    entry_time = None
    max_price_since_entry = 0.0
    min_price_since_entry = 0.0
    bars_held = 0

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
        if position == "long":
            bars_held += 1
            max_price_since_entry = max(max_price_since_entry, high_vals[i])

            exit_price = None
            exit_reason = ""

            if stop_loss_pct is not None:
                sl_price = entry_price * (1 - stop_loss_pct / 100)
                if low_vals[i] <= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"

            if exit_price is None and take_profit_pct is not None:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if high_vals[i] >= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"

            if exit_price is None and bear_signals.iloc[i]:
                exit_price = close_vals[i]
                exit_reason = "opposite_signal"

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

        elif position == "SELL":
            bars_held += 1
            min_price_since_entry = min(min_price_since_entry, low_vals[i])

            exit_price = None
            exit_reason = ""

            if stop_loss_pct is not None:
                sl_price = entry_price * (1 + stop_loss_pct / 100)
                if high_vals[i] >= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"

            if exit_price is None and take_profit_pct is not None:
                tp_price = entry_price * (1 - take_profit_pct / 100)
                if low_vals[i] <= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"

            if exit_price is None and bull_signals.iloc[i]:
                exit_price = close_vals[i]
                exit_reason = "opposite_signal"

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
                        "type": "SELL",
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

        if position is None and not is_last_bar[i]:
            hour = timestamps[i].hour
            minute = timestamps[i].minute
            is_near_close = (hour == 17 and minute >= 25) or (hour > 17)

            if not is_near_close:
                if bull_signals.iloc[i]:
                    position = "long"
                    entry_price = close_vals[i]
                    entry_time = timestamps[i]
                    max_price_since_entry = high_vals[i]
                    bars_held = 0
                elif bear_signals.iloc[i]:
                    position = "SELL"
                    entry_price = close_vals[i]
                    entry_time = timestamps[i]
                    min_price_since_entry = low_vals[i]
                    bars_held = 0

        equity.iloc[i] = current_capital

    return trades, equity


def calculate_metrics(trades, equity, initial_capital=100000.0):
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_profit_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }
    df_trades = pd.DataFrame(trades)
    total_trades = len(df_trades)
    wins = df_trades[df_trades["pnl_pct"] > 0]
    losses = df_trades[df_trades["pnl_pct"] <= 0]
    win_rate = (len(wins) / total_trades) * 100
    net_profit_pct = (equity.iloc[-1] - initial_capital) / initial_capital * 100

    gross_profit = wins["pnl_pct"].sum()
    gross_loss = abs(losses["pnl_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    roll_max = equity.cummax()
    drawdowns = (equity - roll_max) / roll_max * 100
    max_dd = abs(drawdowns.min())

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
    }


def main():
    print("Running Global Parameter Optimization...")
    symbol = "COPPER31JUL26FUT"
    exchange = "MCX"
    start_date = "2026-06-01"
    end_date = "2026-06-27"
    intervals = ["5m", "15m", "1h"]

    # Define Parameter Grid
    rsi_periods = [14, 21]
    oversold_levels = [35, 40, 45]
    overbought_levels = [65, 60, 55]
    pivot_lookbacks = [2, 3]
    max_bars_list = [25, 35, 50]

    sl_options = [0.3, 0.5, 0.8, 1.2, None]
    tp_options = [0.6, 1.0, 1.6, 2.4, None]

    optimization_results = {}

    for interval in intervals:
        df = fetch_data(symbol, exchange, interval, start_date, end_date)
        if df.empty:
            continue

        print(f"\nOptimizing interval: {interval}")

        # Precompute RSIs to save time
        rsi_cache = {}
        for p in rsi_periods:
            rsi_cache[p] = compute_rsi(df, p)

        best_cfg = None
        best_metrics = None
        best_equity = None
        best_trades = None

        count = 0
        total_combos = (
            len(rsi_periods)
            * len(oversold_levels)
            * len(pivot_lookbacks)
            * len(max_bars_list)
            * len(sl_options)
            * len(tp_options)
        )

        for r_p in rsi_periods:
            rsi = rsi_cache[r_p]
            for os_lvl in oversold_levels:
                ob_lvl = 100 - os_lvl  # Keep symmetric
                for p_lb in pivot_lookbacks:
                    for m_b in max_bars_list:
                        # Detect signals for this indicator setup
                        bull, bear = detect_rsi_divergence(df, rsi, p_lb, m_b, os_lvl, ob_lvl)

                        if bull.sum() == 0 and bear.sum() == 0:
                            continue

                        # Evaluate all SL/TP settings
                        for sl in sl_options:
                            for tp in tp_options:
                                # Ensure we don't do None/None unless we want to, and check that if one is None, both are None or we have SL/TP
                                if (sl is None and tp is not None) or (
                                    sl is not None and tp is None
                                ):
                                    continue

                                count += 1
                                trades, equity = run_backtest(df, bull, bear, sl, tp)
                                metrics = calculate_metrics(trades, equity)

                                # We want a setting with a reasonable number of trades (e.g. >= 5 trades)
                                if metrics["total_trades"] >= 5:
                                    # Compare by Net Profit
                                    if (
                                        best_metrics is None
                                        or metrics["net_profit_pct"]
                                        > best_metrics["net_profit_pct"]
                                    ):
                                        best_cfg = {
                                            "rsi_period": r_p,
                                            "oversold": os_lvl,
                                            "overbought": ob_lvl,
                                            "pivot_lookback": p_lb,
                                            "max_bars": m_b,
                                            "sl": sl,
                                            "tp": tp,
                                        }
                                        best_metrics = metrics
                                        best_equity = equity
                                        best_trades = trades

        print(f"Evaluated {count} valid parameter combinations.")
        if best_cfg:
            print(
                "FOUND A PROFITABLE CONFIGURATION!"
                if best_metrics["net_profit_pct"] > 0
                else "NO PROFITABLE CONFIG FOUND (Showing least negative)."
            )
            print(f"Best parameters: {best_cfg}")
            print(
                f"Trades: {best_metrics['total_trades']}, Win Rate: {best_metrics['win_rate']:.2f}%, Net Profit: {best_metrics['net_profit_pct']:.2f}%, Max DD: {best_metrics['max_drawdown']:.2f}%"
            )

            optimization_results[interval] = {
                "config": best_cfg,
                "metrics": best_metrics,
                "trades": best_trades,
                "equity_curve": best_equity,
            }

            # Save optimized trades
            df_opt_trades = pd.DataFrame(best_trades)
            df_opt_trades.to_csv(artifact_dir / f"opt_trades_{interval}.csv", index=False)
        else:
            print("No configuration met trade threshold constraint.")

    # Plot optimized equity curves
    plt.figure(figsize=(12, 7))
    for interval, res in optimization_results.items():
        eq = res["equity_curve"]
        cfg = res["config"]
        lbl = f"{interval}: RSI {cfg['rsi_period']}, OS {cfg['oversold']}, SL {cfg['sl']}%, TP {cfg['tp']}% (Profit: {res['metrics']['net_profit_pct']:.2f}%)"
        plt.plot(eq.index, (eq / 100000.0 - 1) * 100, label=lbl)

    plt.title(
        "Optimized RSI Divergence Strategy on COPPER31JUL26FUT (MCX)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()

    plt.savefig(artifact_dir / "opt_equity_curves.png", dpi=300)
    plt.close()

    # Save opt summary
    opt_summary = {}
    for interval, res in optimization_results.items():
        opt_summary[interval] = {
            "config": res["config"],
            "metrics": {
                "total_trades": res["metrics"]["total_trades"],
                "win_rate_pct": float(res["metrics"]["win_rate"]),
                "net_profit_pct": float(res["metrics"]["net_profit_pct"]),
                "max_drawdown_pct": float(res["metrics"]["max_drawdown"]),
                "profit_factor": float(res["metrics"]["profit_factor"])
                if res["metrics"]["profit_factor"] != float("inf")
                else "inf",
                "sharpe_ratio": float(res["metrics"]["sharpe_ratio"]),
                "avg_trade_pnl_pct": float(res["metrics"]["avg_trade_pnl"]),
            },
        }

    with open(artifact_dir / "opt_backtest_summary.json", "w") as f:
        json.dump(opt_summary, f, indent=4)

    print("\nOptimization run completed and saved.")


if __name__ == "__main__":
    main()
