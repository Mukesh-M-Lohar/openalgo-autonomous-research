import argparse
import json
import os
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
workspace_dir = Path("/root/openalgo-autonomous-research")
artifact_dir = Path("/root/.gemini/antigravity-cli/brain/6e9f0eb7-c7e7-47a3-98ad-4b7e70b45678")


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
        response = httpx.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            df = pd.DataFrame(data["data"])
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                return df.set_index("timestamp").sort_index()
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


# 4. Simulation Engine (Exchange Aware)
def run_backtest(
    df,
    bull_signals,
    bear_signals,
    exchange,
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

    # Configure exchange-aware session end times (UTC)
    if exchange.upper() == "MCX":
        # MCX closes at 17:55 UTC (23:25 IST)
        near_close_hour = 17
        near_close_minute = 25
    else:
        # NSE closes at 10:00 UTC (15:30 IST)
        near_close_hour = 9
        near_close_minute = 30

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
            is_near_close = (hour == near_close_hour and minute >= near_close_minute) or (
                hour > near_close_hour
            )

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


# 5. Performance Metrics Calculator
def calculate_metrics(trades, equity, initial_capital=100000.0):
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_profit_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "avg_trade_pnl": 0.0,
            "long_trades": 0,
            "short_trades": 0,
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
        "long_trades": len(df_trades[df_trades["type"] == "long"]),
        "short_trades": len(df_trades[df_trades["type"] == "SELL"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backtest and Optimize RSI Divergence on Stocks/Futures"
    )
    parser.add_argument("--symbol", type=str, required=True, help="Symbol name (e.g. SAMMAANCAP)")
    parser.add_argument("--exchange", type=str, required=True, help="Exchange name (e.g. NSE)")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2026-06-27", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    symbol = args.symbol
    exchange = args.exchange.upper()
    start_date = args.start
    end_date = args.end

    output_dir = workspace_dir / "backtest_results" / symbol.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy also to brain artifacts directory
    brain_out_dir = artifact_dir / symbol.lower()
    brain_out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running RSI Divergence Backtest for {symbol} ({exchange}) from {start_date} to {end_date}..."
    )
    intervals = ["5m", "15m", "1h"]

    # Grid search ranges
    rsi_periods = [14, 21]
    oversold_levels = [30, 35, 40]
    pivot_lookbacks = [2, 3]
    max_bars_list = [20, 35, 50]

    sl_options = [0.3, 0.5, 0.8, 1.2, None]
    tp_options = [0.6, 1.0, 1.6, 2.4, None]

    # Store results
    default_results = {}
    optimized_results = {}

    # Plot setup
    plt.figure(figsize=(12, 7))

    for interval in intervals:
        df = fetch_data(symbol, exchange, interval, start_date, end_date)
        if df.empty:
            print(f"Skipping {interval} - no data.")
            continue

        print(f"\nProcessing timeframe: {interval} ({len(df)} bars)")

        # 1. RUN DEFAULT SETUP FIRST
        default_rsi = compute_rsi(df, period=14)
        bull_def, bear_def = detect_rsi_divergence(
            df, default_rsi, pivot_lookback=2, max_bars=35, oversold=40, overbought=60
        )
        # Select SL/TP based on default
        def_sl = 0.5
        def_tp = 1.0
        trades_def, equity_def = run_backtest(df, bull_def, bear_def, exchange, def_sl, def_tp)
        metrics_def = calculate_metrics(trades_def, equity_def)
        default_results[interval] = {
            "metrics": metrics_def,
            "settings": {"sl": def_sl, "tp": def_tp},
            "equity": equity_def,
        }

        # Save default trades
        if trades_def:
            pd.DataFrame(trades_def).to_csv(
                output_dir / f"default_trades_{interval}.csv", index=False
            )
            pd.DataFrame(trades_def).to_csv(
                brain_out_dir / f"default_trades_{interval}.csv", index=False
            )

        # 2. RUN PARAMETER OPTIMIZATION
        # Precompute RSIs
        rsi_cache = {}
        for p in rsi_periods:
            rsi_cache[p] = compute_rsi(df, p)

        best_cfg = None
        best_metrics = None
        best_equity = None
        best_trades = None

        for r_p in rsi_periods:
            rsi = rsi_cache[r_p]
            for os_lvl in oversold_levels:
                ob_lvl = 100 - os_lvl
                for p_lb in pivot_lookbacks:
                    for m_b in max_bars_list:
                        bull, bear = detect_rsi_divergence(df, rsi, p_lb, m_b, os_lvl, ob_lvl)

                        if bull.sum() == 0 and bear.sum() == 0:
                            continue

                        for sl in sl_options:
                            for tp in tp_options:
                                if (sl is None and tp is not None) or (
                                    sl is not None and tp is None
                                ):
                                    continue

                                trades, equity = run_backtest(df, bull, bear, exchange, sl, tp)
                                metrics = calculate_metrics(trades, equity)

                                # Require at least 5 trades to filter out single lucky trades
                                if metrics["total_trades"] >= 5:
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

        # If no configuration found meeting the trade count threshold, try with lower trade threshold
        if best_cfg is None:
            for r_p in rsi_periods:
                rsi = rsi_cache[r_p]
                for os_lvl in oversold_levels:
                    ob_lvl = 100 - os_lvl
                    for p_lb in pivot_lookbacks:
                        for m_b in max_bars_list:
                            bull, bear = detect_rsi_divergence(df, rsi, p_lb, m_b, os_lvl, ob_lvl)
                            for sl in sl_options:
                                for tp in tp_options:
                                    if (sl is None and tp is not None) or (
                                        sl is not None and tp is None
                                    ):
                                        continue
                                    trades, equity = run_backtest(df, bull, bear, exchange, sl, tp)
                                    metrics = calculate_metrics(trades, equity)
                                    if metrics["total_trades"] >= 2:
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

        if best_cfg:
            print(f"  Optimized parameters: {best_cfg}")
            print(
                f"  Trades: {best_metrics['total_trades']}, Win Rate: {best_metrics['win_rate']:.2f}%, Net Profit: {best_metrics['net_profit_pct']:.2f}%"
            )
            optimized_results[interval] = {
                "config": best_cfg,
                "metrics": best_metrics,
                "equity": best_equity,
                "trades": best_trades,
            }

            # Save optimized trades
            if best_trades:
                pd.DataFrame(best_trades).to_csv(
                    output_dir / f"opt_trades_{interval}.csv", index=False
                )
                pd.DataFrame(best_trades).to_csv(
                    brain_out_dir / f"opt_trades_{interval}.csv", index=False
                )

            # Plot optimized equity curve
            plt.plot(
                best_equity.index,
                (best_equity / 100000.0 - 1) * 100,
                label=f"{interval}: Return {best_metrics['net_profit_pct']:.2f}% (SL {best_cfg['sl']}%, TP {best_cfg['tp']}%)",
            )
        else:
            print("  No suitable optimized parameter configuration found.")

    # Save chart
    plt.title(
        f"Optimized RSI Divergence Equity Curves: {symbol} ({exchange})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()

    plt.savefig(output_dir / "opt_equity_curves.png", dpi=300)
    plt.savefig(brain_out_dir / "opt_equity_curves.png", dpi=300)
    plt.close()

    # Save default chart
    plt.figure(figsize=(12, 7))
    for interval, res in default_results.items():
        eq = res["equity"]
        plt.plot(
            eq.index,
            (eq / 100000.0 - 1) * 100,
            label=f"{interval} default (Return {res['metrics']['net_profit_pct']:.2f}%)",
        )
    plt.title(
        f"Default RSI Divergence Equity Curves: {symbol} ({exchange})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "default_equity_curves.png", dpi=300)
    plt.savefig(brain_out_dir / "default_equity_curves.png", dpi=300)
    plt.close()

    # Save summary JSON files
    summary = {"symbol": symbol, "exchange": exchange, "default": {}, "optimized": {}}

    for interval, res in default_results.items():
        summary["default"][interval] = {
            "settings": res["settings"],
            "metrics": {
                "total_trades": res["metrics"]["total_trades"],
                "win_rate_pct": float(res["metrics"]["win_rate"]),
                "net_profit_pct": float(res["metrics"]["net_profit_pct"]),
                "max_drawdown_pct": float(res["metrics"]["max_drawdown"]),
                "profit_factor": float(res["metrics"]["profit_factor"])
                if res["metrics"]["profit_factor"] != float("inf")
                else "inf",
                "sharpe_ratio": float(res["metrics"]["sharpe_ratio"]),
            },
        }

    for interval, res in optimized_results.items():
        summary["optimized"][interval] = {
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
            },
        }

    with open(output_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    with open(brain_out_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Write a clean markdown report inside backtest_results/symbol/
    report_content = rf"""# RSI Divergence Backtest Report: {symbol} ({exchange})

This report presents the backtesting results of the **RSI Divergence Strategy** applied to `{symbol}` on the `{exchange}` exchange. The backtest runs cover intraday trading (MIS) from `{start_date}` to `{end_date}`.

---

## Strategy & Market Hours
- **Execution**: Signals are triggered causally (no lookahead) using pivot confirmation rules.
- **Intraday Exits (MIS)**: All positions are closed at the last bar of the trading session (NSE closes at 10:00 UTC / 3:30 PM IST). No trades are allowed to be entered after 09:30 UTC (3:00 PM IST).
- **Transaction Costs**: Includes a $0.03\%$ commission and $0.02\%$ slippage per side (total $0.10\%$ roundtrip cost).

---

## 1. Default Parameters Backtest
- **Parameters**: RSI 14, Oversold = 40, Overbought = 60, Pivot Lookback = 2, Max Window = 35.
- **Stop Loss / Take Profit**: SL = 0.5%, TP = 1.0%

### Default Performance Summary
"""

    # Add default table
    report_content += "| Timeframe | SL | TP | Total Trades | Win Rate (%) | Net Profit (%) | Max Drawdown (%) | Profit Factor | Sharpe Ratio |\n"
    report_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for interval, res in default_results.items():
        m = res["metrics"]
        s = res["settings"]
        report_content += f"| **{interval}** | {s['sl']}% | {s['tp']}% | {m['total_trades']} | {m['win_rate']:.2f}% | **{m['net_profit_pct']:.2f}%** | {m['max_drawdown']:.2f}% | {m['profit_factor']:.2f} | {m['sharpe_ratio']:.2f} |\n"

    report_content += """
### Default Strategy Equity Curves
![Default Strategy Equity Curves](default_equity_curves.png)

---

## 2. Optimized Parameters Backtest
To find the best configuration, we performed a multi-parameter grid search across RSI periods, overbought/oversold levels, pivot strengths, and SL/TP combinations.

### Optimized Performance Summary
"""

    report_content += "| Timeframe | Best Parameters | Total Trades | Win Rate (%) | Net Profit (%) | Max Drawdown (%) | Profit Factor | Sharpe Ratio |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for interval, res in optimized_results.items():
        m = res["metrics"]
        cfg = res["config"]
        params_str = f"RSI {cfg['rsi_period']}, OS {cfg['oversold']}, Lookback {cfg['pivot_lookback']}, SL {cfg['sl']}%, TP {cfg['tp']}%"
        report_content += f"| **{interval}** | {params_str} | {m['total_trades']} | {m['win_rate']:.2f}% | **{m['net_profit_pct']:.2f}%** | {m['max_drawdown']:.2f}% | {m['profit_factor']:.2f} | {m['sharpe_ratio']:.2f} |\n"

    report_content += rf"""
### Optimized Strategy Equity Curves
![Optimized Strategy Equity Curves](opt_equity_curves.png)

---

## Key Insights
1. **Timeframe Behavior**: On stock equities like `{symbol}`, higher timeframes tend to filter out high-frequency noise but lead to fewer opportunities.
2. **Transaction Drag**: On smaller timeframes (like 5m), transaction fees ($0.10\%$ roundtrip) degrade performance significantly over large trade counts. Tightening the entry requirements is critical.
"""

    with open(output_dir / "rsi_divergence_backtest_report.md", "w") as f:
        f.write(report_content)
    with open(brain_out_dir / "rsi_divergence_backtest_report.md", "w") as f:
        f.write(report_content)

    print(f"\nBacktesting completed successfully. Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
