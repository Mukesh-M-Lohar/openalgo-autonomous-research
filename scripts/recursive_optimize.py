"""
Recursive Strategy Optimizer for MCX (NSE Exchange) - FAST VECTORIZED VERSION
----------------------------------------------------------------------------
Optimizes for:
1. Maximum average profit per day
2. Maximum Drawdown strictly <= 3.0%
3. Minimal trading frequency (fewer trades per day)
"""

import pandas as pd
from rsi_divergence_backtest import compute_rsi, detect_divergences, load_data

# ==============================================================================
# FAST BACKTEST WRAPPERS WITH PARAMETERS
# ==============================================================================


def backtest_swing(df_daily: pd.DataFrame, w: int, sl_pct: float, tp_pct: float) -> dict:
    df = detect_divergences(df_daily, w=w)

    capital = 100000.0
    initial_capital = capital
    position = 0
    entry_price = 0.0
    trades = []

    for i in range(len(df) - 1):
        date = df.index[i]
        next_date = df.index[i + 1]
        next_open = df["open"].iloc[i + 1]

        if position > 0:
            low_p = df["low"].iloc[i]
            high_p = df["high"].iloc[i]

            sl_price = entry_price * (1.0 - sl_pct / 100.0) if sl_pct > 0 else 0.0
            tp_price = entry_price * (1.0 + tp_pct / 100.0) if tp_pct > 0 else float("inf")

            exit_triggered = False
            exit_price = 0.0
            exit_reason = ""

            if sl_pct > 0 and low_p <= sl_price:
                exit_triggered = True
                exit_price = sl_price
                exit_reason = "STOP_LOSS"
            elif tp_pct > 0 and high_p >= tp_price:
                exit_triggered = True
                exit_price = tp_price
                exit_reason = "TAKE_PROFIT"
            elif df["bearish_divergence"].iloc[i]:
                exit_triggered = True
                exit_price = next_open
                exit_reason = "BEARISH_DIVERGENCE"

            if exit_triggered:
                pnl = (exit_price - entry_price) / entry_price
                capital = capital * (1 + pnl)
                trades.append(
                    {
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl * 100,
                        "capital": capital,
                    }
                )
                position = 0

        elif position == 0:
            if df["bullish_divergence"].iloc[i]:
                position = 1
                entry_price = next_open

    if not trades:
        return {"net_profit_pct": 0.0, "total_trades": 0, "max_drawdown_pct": 0.0}

    df_trades = pd.DataFrame(trades)
    net_profit = (capital - initial_capital) / initial_capital * 100

    cap_series = pd.Series([initial_capital] + df_trades["capital"].tolist())
    cum_max = cap_series.cummax()
    drawdowns = (cap_series - cum_max) / cum_max
    max_dd = abs(drawdowns.min()) * 100

    return {
        "net_profit_pct": net_profit,
        "total_trades": len(df_trades),
        "max_drawdown_pct": max_dd,
    }


def backtest_intraday_fast(
    df_daily: pd.DataFrame,
    df_15m: pd.DataFrame,
    w: int,
    rsi_long: float,
    rsi_short: float,
    sl_pct: float = 1.0,
    allocation: float = 1.0,
) -> dict:
    df_div = detect_divergences(df_daily, w=w)

    df_div["bias"] = 0
    current_bias = 0
    for idx, row in df_div.iterrows():
        if row["bullish_divergence"]:
            current_bias = 1
        elif row["bearish_divergence"]:
            current_bias = -1
        df_div.loc[idx, "bias"] = current_bias

    df_15m = df_15m.copy()
    df_15m["rsi_15m"] = compute_rsi(df_15m["close"], 14)
    df_15m["date"] = df_15m.index.date
    daily_bias_map = df_div["bias"].to_dict()
    df_15m["bias"] = df_15m["date"].map(lambda d: daily_bias_map.get(pd.Timestamp(d), 0))

    capital = 100000.0
    initial_capital = capital
    trades = []

    active_days = df_15m[df_15m["bias"] != 0]["date"].unique()

    for date in active_days:
        day_df = df_15m[df_15m["date"] == date]
        bias = day_df["bias"].iloc[0]

        if bias == 1:
            entry_cond = day_df["rsi_15m"] < rsi_long
            if not entry_cond.any():
                continue
            entry_idx = entry_cond.idxmax()
            entry_price = day_df.loc[entry_idx, "close"]

            trade_window = day_df.loc[entry_idx:]
            exit_time = day_df.index[-1]
            exit_price = day_df.loc[exit_time, "close"]

            sl_price = entry_price * (1 - sl_pct / 100)
            lows = trade_window["low"]
            sl_hit_cond = lows <= sl_price
            if sl_hit_cond.any():
                exit_price = sl_price

            # Apply position sizing allocation factor to the trade PnL
            pnl = ((exit_price - entry_price) / entry_price) * allocation
            capital = capital * (1 + pnl)
            trades.append({"pnl_pct": pnl * 100, "capital": capital})

        elif bias == -1:
            entry_cond = day_df["rsi_15m"] > rsi_short
            if not entry_cond.any():
                continue
            entry_idx = entry_cond.idxmax()
            entry_price = day_df.loc[entry_idx, "close"]

            trade_window = day_df.loc[entry_idx:]
            exit_time = day_df.index[-1]
            exit_price = day_df.loc[exit_time, "close"]

            sl_price = entry_price * (1 + sl_pct / 100)
            highs = trade_window["high"]
            sl_hit_cond = highs >= sl_price
            if sl_hit_cond.any():
                exit_price = sl_price

            pnl = ((entry_price - exit_price) / entry_price) * allocation
            capital = capital * (1 + pnl)
            trades.append({"pnl_pct": pnl * 100, "capital": capital})

    if not trades:
        return {"net_profit_pct": 0.0, "total_trades": 0, "max_drawdown_pct": 0.0}

    df_trades = pd.DataFrame(trades)
    net_profit = (capital - initial_capital) / initial_capital * 100

    cap_series = pd.Series([initial_capital] + df_trades["capital"].tolist())
    cum_max = cap_series.cummax()
    drawdowns = (cap_series - cum_max) / cum_max
    max_dd = abs(drawdowns.min()) * 100

    return {
        "net_profit_pct": net_profit,
        "total_trades": len(df_trades),
        "max_drawdown_pct": max_dd,
    }


# ==============================================================================
# RECURSIVE RE-OPTIMIZATION
# ==============================================================================

if __name__ == "__main__":
    df_daily, df_15m = load_data("MCX", "NSE")
    total_days = len(df_daily)

    print("\n" + "=" * 70)
    print("      RECURSIVE OPTIMIZATION SWEEP (DRAWDOWN <= 3.0%)")
    print("=" * 70)

    # 1. Swing Strategy Optimization
    print("\nRunning Swing Sweep...")
    swing_results = []
    for w in range(3, 10):
        for sl in [1.0, 1.5, 2.0, 2.5]:
            for tp in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
                res = backtest_swing(df_daily, w, sl, tp)
                if res["max_drawdown_pct"] <= 3.0:
                    profit_per_day = res["net_profit_pct"] / total_days
                    trades_per_day = res["total_trades"] / total_days
                    swing_results.append(
                        {
                            "w": w,
                            "stop_loss_pct": sl,
                            "take_profit_pct": tp,
                            "net_profit_pct": res["net_profit_pct"],
                            "max_drawdown_pct": res["max_drawdown_pct"],
                            "total_trades": res["total_trades"],
                            "profit_per_day": profit_per_day,
                            "trades_per_day": trades_per_day,
                        }
                    )

    df_swing = pd.DataFrame(swing_results)
    if not df_swing.empty:
        df_swing = df_swing.sort_values(
            by=["profit_per_day", "trades_per_day"], ascending=[False, True]
        )
        print("\nTop 5 Optimized Swing Configurations (Drawdown <= 3.0%):")
        print(df_swing.head(5).to_string(index=False))
    else:
        print("\nNo Swing configurations satisfied the Max Drawdown <= 3.0% constraint.")

    # 2. Intraday Strategy Optimization
    print("\nRunning Intraday Sweep...")
    intraday_results = []
    # Narrowing grid slightly to run even faster
    for w in [4, 5, 7]:
        for rsi_long in [30, 35, 40]:
            for rsi_short in [60, 65, 70]:
                for sl in [0.5, 1.0]:
                    for alloc in [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                    ]:  # Capital allocation factor (position sizing)
                        res = backtest_intraday_fast(
                            df_daily, df_15m, w, rsi_long, rsi_short, sl, alloc
                        )
                        if res["max_drawdown_pct"] <= 3.0:
                            profit_per_day = res["net_profit_pct"] / total_days
                            trades_per_day = res["total_trades"] / total_days
                            intraday_results.append(
                                {
                                    "w": w,
                                    "rsi_long": rsi_long,
                                    "rsi_short": rsi_short,
                                    "intraday_sl": sl,
                                    "allocation": alloc,
                                    "net_profit_pct": res["net_profit_pct"],
                                    "max_drawdown_pct": res["max_drawdown_pct"],
                                    "total_trades": res["total_trades"],
                                    "profit_per_day": profit_per_day,
                                    "trades_per_day": trades_per_day,
                                }
                            )

    df_intra = pd.DataFrame(intraday_results)
    if not df_intra.empty:
        df_intra = df_intra.sort_values(
            by=["profit_per_day", "trades_per_day"], ascending=[False, True]
        )
        print("\nTop 5 Optimized Intraday Configurations (Drawdown <= 3.0%):")
        print(df_intra.head(5).to_string(index=False))
    else:
        print("\nNo Intraday configurations satisfied the Max Drawdown <= 3.0% constraint.")
