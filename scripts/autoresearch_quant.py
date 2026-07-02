import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

API_KEY = "b45feb0a6973ed00fe86d25ace49d4da8dfe8d0a78c334455d46254ded28a26d"
API_HOST = "http://127.0.0.1:5000"
CACHE_DIR = "/root/openalgo-autonomous-research/data/cache_15m"
RESULTS_FILE = "/root/openalgo-autonomous-research/results.tsv"
BEST_STRATEGY_FILE = "/root/openalgo-autonomous-research/best_strategy.json"
PLOT_PATH = "/root/openalgo-autonomous-research/best_equity_curve.png"

os.makedirs(CACHE_DIR, exist_ok=True)

SYMBOLS = [
    ("PROTEAN", "NSE", False),
    ("ZEEL", "NSE", False),
    ("BAHETI-SM", "NSE", False),
    ("CDSL", "NSE", False),
    ("ANGELONE", "NSE", False),
    ("SCI", "NSE", False),
    ("ACUTAAS", "NSE", False),
    ("SAMMAANCAP", "NSE", False),
    ("CLEAN", "NSE", False),
    ("COPPER31JUL26FUT", "MCX", True),
    ("COPPER31AUG26FUT", "MCX", True),
    ("MCX", "NSE", False),
    ("SBIN", "NSE", False),
    ("BSE", "NSE", False),
    ("NIFTY", "NSE_INDEX", True),
    ("BANKNIFTY", "NSE_INDEX", True),
]


# 1. Fetching Helper
def get_15m_data(symbol, exchange):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{exchange}_15m.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df
    return pd.DataFrame()


# 2. Indicators calculations
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

    # EMA Trend
    ema_p = params.get("ema_trend", 100)
    df["ema_trend"] = df["close"].ewm(span=ema_p, adjust=False).mean()

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

    return df


# 3. Flexible Strategy Simulator
def simulate_strategy(df, params, allow_short=False, commission_pct=0.03, slippage_pct=0.02):
    df = compute_indicators(df, params)
    df = df.dropna(
        subset=["rsi", "ema_trend", "macd", "macd_signal", "bb_upper", "bb_lower", "vwap", "atr"]
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
    ema_trend = df["ema_trend"].values
    macd = df["macd"].values
    macd_sig = df["macd_signal"].values
    bb_upper = df["bb_upper"].values
    bb_lower = df["bb_lower"].values
    vwap = df["vwap"].values
    atr = df["atr"].values

    # Strategy Rules
    strat_name = params.get("strategy_name", "vwap_rsi")
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
            # Check exit
            tp_price = entry_price * (1 + tp_pct) if position == 1 else entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 - sl_pct) if position == 1 else entry_price * (1 + sl_pct)

            # If using ATR stop
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
                        "direction": "LONG" if position == 1 else "SELL",
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
            if not is_exit_time and time_idx.hour < 15:
                signal = 0
                if strat_name == "vwap_rsi":
                    # Buy when close < VWAP and RSI < rsi_lower
                    if close_vals[i] < vwap[i] and rsi[i] < params.get("rsi_lower", 35):
                        signal = 1
                    elif (
                        allow_short
                        and close_vals[i] > vwap[i]
                        and rsi[i] > (100 - params.get("rsi_lower", 35))
                    ):
                        signal = -1
                elif strat_name == "macd_trend":
                    # Buy when close > EMA and MACD crosses above Signal
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
                    # Buy when close below BB lower and RSI oversold
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


# 4. Performance Evaluator
def evaluate_portfolio(params):
    portfolio_eqs = []
    total_trades_count = 0

    for sym, exch, allow_short in SYMBOLS:
        df = get_15m_data(sym, exch)
        if df.empty:
            continue
        # Use Test period (from 2026-03-01 to 2026-06-27) for optimization evaluation
        df_test = df.loc["2026-03-01":]
        trades, eq = simulate_strategy(df_test, params, allow_short)

        # Group by day
        df_eq = pd.DataFrame(eq, columns=["equity"])
        df_eq.index.name = None
        if not isinstance(df_eq.index, pd.DatetimeIndex):
            df_eq.index = pd.to_datetime(df_eq.index)
        df_eq["date"] = df_eq.index.date
        daily_eq = df_eq.groupby("date")["equity"].last()

        portfolio_eqs.append(daily_eq)
        total_trades_count += len(trades)

    if not portfolio_eqs or total_trades_count < 10:
        return None

    port_df = pd.DataFrame(portfolio_eqs).T.sort_index().ffill().bfill()
    port_equity = port_df.mean(axis=1)

    daily_returns = port_equity.pct_change().dropna()
    avg_daily_ret = daily_returns.mean() * 100
    sharpe = (
        np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    )
    roll_max = port_equity.cummax()
    drawdown = (port_equity - roll_max) / roll_max
    max_dd = drawdown.min() * 100

    return {
        "sharpe": sharpe,
        "daily_return_avg_pct": avg_daily_ret,
        "max_dd_pct": max_dd,
        "total_trades": total_trades_count,
        "equity_curve": port_equity,
    }


# 5. Propose Mutations (Autonomous Proposal)
def propose_next_parameters(best_params=None):
    if best_params is None:
        # Default starting point
        return {
            "strategy_name": "vwap_rsi",
            "rsi_period": 14,
            "rsi_lower": 35,
            "ema_trend": 100,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "tp": 0.015,
            "sl": 0.01,
            "allocation": 2.0,
            "use_atr_stop": False,
            "atr_mult": 2.0,
        }

    new_params = best_params.copy()

    # Mutate strategy type
    if random.random() < 0.2:
        new_params["strategy_name"] = random.choice(["vwap_rsi", "macd_trend", "bb_reversion"])

    # Mutate parameters
    mutation_choice = random.choice(
        [
            "rsi_lower",
            "tp",
            "sl",
            "allocation",
            "use_atr_stop",
            "atr_mult",
            "macd_fast_slow",
            "bb_std",
        ]
    )

    if mutation_choice == "rsi_lower":
        new_params["rsi_lower"] = random.choice([25, 30, 35, 40])
    elif mutation_choice == "tp":
        new_params["tp"] = round(random.choice([0.005, 0.01, 0.015, 0.02, 0.025]), 3)
    elif mutation_choice == "sl":
        new_params["sl"] = round(random.choice([0.005, 0.01, 0.015, 0.02]), 3)
    elif mutation_choice == "allocation":
        new_params["allocation"] = round(random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0]), 1)
    elif mutation_choice == "use_atr_stop":
        new_params["use_atr_stop"] = not new_params["use_atr_stop"]
    elif mutation_choice == "atr_mult":
        new_params["atr_mult"] = round(random.choice([1.5, 2.0, 2.5, 3.0]), 1)
    elif mutation_choice == "macd_fast_slow":
        fast = random.choice([8, 12, 15])
        new_params["macd_fast"] = fast
        new_params["macd_slow"] = fast + random.choice([10, 14, 18])
    elif mutation_choice == "bb_std":
        new_params["bb_std"] = round(random.choice([1.5, 2.0, 2.5]), 1)

    return new_params


# 6. Autonomous Loop Runner
def run_autonomous_research(iterations=15):
    print("=" * 85)
    print(f"      OPENALGO AUTONOMOUS QUANT RESEARCH LOOP ({iterations} Iterations)")
    print("=" * 85)

    # Load previous best if exists
    best_params = None
    best_score = -float("inf")

    if os.path.exists(BEST_STRATEGY_FILE):
        try:
            with open(BEST_STRATEGY_FILE, "r") as f:
                best_data = json.load(f)
                best_params = best_data["params"]
                best_score = best_data["score"]
                print(f"Loaded existing best strategy from disk: Score = {best_score:.4f}")
        except Exception:
            pass

    # Setup results.tsv headers if not exists
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(
                "timestamp\tstrategy\tparameters\tsharpe\tdaily_return_pct\tmax_dd_pct\ttrades\tscore\tstatus\n"
            )

    for step in range(1, iterations + 1):
        print(f"\n[Iteration {step}/{iterations}] Proposing mutation...")
        proposed = propose_next_parameters(best_params)
        print(
            f"Proposed: {proposed['strategy_name']} | TP={proposed['tp'] * 100:.1f}%, SL={proposed['sl'] * 100:.1f}%, Alloc={proposed['allocation']}x, ATR_Stop={proposed['use_atr_stop']}"
        )

        # Backtest & Evaluate
        metrics = evaluate_portfolio(proposed)

        if metrics is None:
            print("  Skipped: insufficient trades or failed evaluation.")
            continue

        sharpe = metrics["sharpe"]
        daily_ret = metrics["daily_return_avg_pct"]
        max_dd = metrics["max_dd_pct"]
        trades = metrics["total_trades"]

        # Scoring function: maximize daily return and Sharpe, penalize large drawdown
        score = daily_ret * 10.0 + sharpe * 0.5
        if max_dd < -20.0:
            score -= 5.0

        status = "REJECTED"
        if score > best_score:
            best_score = score
            best_params = proposed
            status = "ACCEPTED"
            print(
                f"  🔥 NEW BEST! Score: {score:.4f} | Sharpe: {sharpe:.2f} | Daily Ret: {daily_ret:.4f}% | Max DD: {max_dd:.2f}%"
            )

            # Save best strategy metadata
            best_strategy_meta = {
                "score": best_score,
                "params": best_params,
                "metrics": {
                    "sharpe": sharpe,
                    "daily_return_avg_pct": daily_ret,
                    "max_dd_pct": max_dd,
                    "total_trades": trades,
                },
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(BEST_STRATEGY_FILE, "w") as f:
                json.dump(best_strategy_meta, f, indent=4)

            # Plot best equity curve
            plt.figure(figsize=(10, 5))
            plt.plot(
                metrics["equity_curve"].index,
                100000.0 * (metrics["equity_curve"] / metrics["equity_curve"].iloc[0]),
                color="#22c55e",
                lw=2,
                label=f"Best Portfolio ({best_params['strategy_name'].upper()})",
            )
            plt.title(
                f"Best Autonomous Portfolio Equity (Daily Return: {daily_ret:.3f}%)",
                fontweight="bold",
            )
            plt.xlabel("Date")
            plt.ylabel("Equity (Base 100,000)")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(PLOT_PATH, dpi=300)
            plt.close()
        else:
            print(f"  Rejected: Score {score:.4f} <= Best Score {best_score:.4f}")

        # Log to results.tsv
        with open(RESULTS_FILE, "a") as f:
            param_str = json.dumps(proposed).replace("\t", " ")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"{ts}\t{proposed['strategy_name']}\t{param_str}\t{sharpe:.4f}\t{daily_ret:.4f}\t{max_dd:.4f}\t{trades}\t{score:.4f}\t{status}\n"
            )

    print("\n" + "=" * 85)
    print("      AUTONOMOUS RESEARCH COMPLETED!")
    print(f"      Best Strategy parameters saved to: {BEST_STRATEGY_FILE}")
    print(f"      Full research history logged in: {RESULTS_FILE}")
    print("=" * 85)


if __name__ == "__main__":
    run_autonomous_research(20)
