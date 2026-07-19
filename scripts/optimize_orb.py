import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# Paths
NIFTY_PATH = "/root/openalgo-autonomous-research/data/cache_15m/NIFTY_NSE_INDEX_15m.csv"
BANKNIFTY_PATH = "/root/openalgo-autonomous-research/data/cache_15m/BANKNIFTY_NSE_INDEX_15m.csv"

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    return df

def simulate_orb(df: pd.DataFrame, params: dict, commission_pct=0.03, slippage_pct=0.02) -> tuple:
    or_bars = params.get("or_bars", 1)  # 1 for 15m, 2 for 30m, 3 for 45m
    tp_mult = params.get("tp_mult", 1.0)
    sl_mult = params.get("sl_mult", 1.0)  # multiplier of OR range
    sl_type = params.get("sl_type", "opposite")  # "opposite", "midpoint", "multiplier"
    buffer_pct = params.get("buffer_pct", 0.0)  # breakout buffer
    cutoff_time = params.get("cutoff_time", "14:30")  # no entries after this time
    trade_direction = params.get("direction", "both")  # "both", "long", "short"
    
    cost_factor = (commission_pct + slippage_pct) / 100.0  # 0.05% per leg
    
    df = df.copy()
    df["date"] = df.index.date
    
    trades = []
    daily_returns = {}
    
    # Group by date and iterate day-by-day
    grouped = df.groupby("date")
    
    all_dates = sorted(grouped.groups.keys())
    capital = 100000.0
    equity_curve = []
    
    for date in all_dates:
        day_df = grouped.get_group(date).sort_index()
        if len(day_df) <= or_bars:
            daily_returns[date] = 0.0
            equity_curve.append(capital)
            continue
            
        # 1. Establish Opening Range
        or_df = day_df.iloc[:or_bars]
        or_high = or_df["high"].max()
        or_low = or_df["low"].min()
        or_range = or_high - or_low
        if or_range <= 0:
            daily_returns[date] = 0.0
            equity_curve.append(capital)
            continue
            
        # Breakout trigger levels
        long_trigger = or_high * (1.0 + buffer_pct / 100.0)
        short_trigger = or_low * (1.0 - buffer_pct / 100.0)
        
        # 2. Iterate through subsequent candles of the day
        trade_candles = day_df.iloc[or_bars:]
        
        position = 0  # 0: flat, 1: long, -1: short
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        trades_count = 0
        max_trades_per_day = 1
        
        day_pnl = 0.0
        
        for idx, row in trade_candles.iterrows():
            t = idx.time()
            hour, minute = t.hour, t.minute
            is_eod = (hour == 15 and minute == 15)
            
            # Check cutoff limit for entry
            cutoff_h, cutoff_m = map(int, cutoff_time.split(":"))
            can_enter = (hour < cutoff_h) or (hour == cutoff_h and minute < cutoff_m)
            
            # If flat, check entry triggers
            if position == 0 and trades_count < max_trades_per_day:
                if can_enter:
                    # check long breakout
                    high_val = row["high"]
                    low_val = row["low"]
                    open_val = row["open"]
                    
                    triggered_long = (high_val >= long_trigger) and (trade_direction in ["both", "long"])
                    triggered_short = (low_val <= short_trigger) and (trade_direction in ["both", "short"])
                    
                    if triggered_long and triggered_short:
                        # If both are triggered in same candle, take the one that is closer to open, or assume long
                        # For safety, let's just go long
                        triggered_short = False
                        
                    if triggered_long:
                        position = 1
                        trades_count += 1
                        entry_price = max(open_val, long_trigger)
                        
                        # SL configuration
                        if sl_type == "opposite":
                            sl_price = or_low
                        elif sl_type == "midpoint":
                            sl_price = (or_high + or_low) / 2.0
                        else:
                            sl_price = entry_price - sl_mult * or_range
                            
                        tp_price = entry_price + tp_mult * or_range
                        
                        # Check exit on same candle
                        if low_val <= sl_price:
                            pnl = (sl_price - entry_price) / entry_price - 2 * cost_factor
                            day_pnl += pnl
                            position = 0
                        elif high_val >= tp_price:
                            pnl = (tp_price - entry_price) / entry_price - 2 * cost_factor
                            day_pnl += pnl
                            position = 0
                            
                    elif triggered_short:
                        position = -1
                        trades_count += 1
                        entry_price = min(open_val, short_trigger)
                        
                        if sl_type == "opposite":
                            sl_price = or_high
                        elif sl_type == "midpoint":
                            sl_price = (or_high + or_low) / 2.0
                        else:
                            sl_price = entry_price + sl_mult * or_range
                            
                        tp_price = entry_price - tp_mult * or_range
                        
                        if high_val >= sl_price:
                            pnl = (entry_price - sl_price) / entry_price - 2 * cost_factor
                            day_pnl += pnl
                            position = 0
                        elif low_val <= tp_price:
                            pnl = (entry_price - tp_price) / entry_price - 2 * cost_factor
                            day_pnl += pnl
                            position = 0
                            
            # If in position, check exit triggers
            elif position != 0:
                high_val = row["high"]
                low_val = row["low"]
                close_val = row["close"]
                
                exit_triggered = False
                exit_price = close_val
                
                if position == 1:
                    if low_val <= sl_price:
                        exit_triggered = True
                        exit_price = sl_price
                    elif high_val >= tp_price:
                        exit_triggered = True
                        exit_price = tp_price
                elif position == -1:
                    if high_val >= sl_price:
                        exit_triggered = True
                        exit_price = sl_price
                    elif low_val <= tp_price:
                        exit_triggered = True
                        exit_price = tp_price
                        
                if not exit_triggered and is_eod:
                    exit_triggered = True
                    exit_price = close_val
                    
                if exit_triggered:
                    pnl = ((exit_price - entry_price) / entry_price if position == 1 else (entry_price - exit_price) / entry_price) - 2 * cost_factor
                    day_pnl += pnl
                    trades.append({
                        "date": date,
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl * 100.0
                    })
                    position = 0
                    
        daily_returns[date] = day_pnl
        capital = capital * (1.0 + day_pnl)
        equity_curve.append(capital)
        
    daily_ret_series = pd.Series(daily_returns)
    equity_series = pd.Series(equity_curve, index=all_dates)
    return trades, daily_ret_series, equity_series

def evaluate_orb_symbol(df: pd.DataFrame, params: dict) -> dict:
    # Split dates
    dates = df.index.date
    # Find indices for train, val, oos
    # Train: 2025-01-01 to 2025-10-31
    # Val: 2025-11-01 to 2026-02-28
    # OOS: 2026-03-01 to 2026-06-25
    
    df_train = df.loc["2025-01-01":"2025-10-31"]
    df_val = df.loc["2025-11-01":"2026-02-28"]
    df_oos = df.loc["2026-03-01":"2026-06-25"]
    
    t_trades, t_ret, t_eq = simulate_orb(df_train, params)
    v_trades, v_ret, v_eq = simulate_orb(df_val, params)
    o_trades, o_ret, o_eq = simulate_orb(df_oos, params)
    
    def get_metrics(trades, ret, eq):
        if not trades:
            return {"adr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "trades": 0, "win_rate": 0.0, "positive_days_pct": 0.0}
        adr = ret.mean() * 100.0
        std = ret.std()
        sharpe = (np.sqrt(252) * ret.mean() / std) if std > 0 else 0.0
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max
        max_dd = abs(dd.min()) * 100.0
        
        pnl_pcts = [t["pnl_pct"] for t in trades]
        win_rate = sum(1 for p in pnl_pcts if p > 0) / len(trades) * 100.0
        
        # positive days pct (excluding flat days if we want, or including all days)
        trading_days = len(ret)
        positive_days = sum(1 for r in ret if r > 0)
        positive_days_pct = (positive_days / trading_days) * 100.0
        
        return {
            "adr": adr,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "trades": len(trades),
            "win_rate": win_rate,
            "positive_days_pct": positive_days_pct
        }
        
    return {
        "train": get_metrics(t_trades, t_ret, t_eq),
        "val": get_metrics(v_trades, v_ret, v_eq),
        "oos": get_metrics(o_trades, o_ret, o_eq)
    }

if __name__ == "__main__":
    print("Loading NIFTY and BANKNIFTY data...")
    df_nifty = load_and_preprocess(NIFTY_PATH)
    df_bank = load_and_preprocess(BANKNIFTY_PATH)
    print("Data loaded successfully!")
    
    # Baseline ORB Parameters
    baseline_params = {
        "or_bars": 1,          # 15m OR
        "tp_mult": 1.0,         # 1.0x range TP
        "sl_mult": 1.0,         # 1.0x range SL
        "sl_type": "opposite",  # Stop Loss at opposite boundary
        "buffer_pct": 0.0,      # 0% buffer
        "cutoff_time": "14:30", # cut off entries at 14:30 IST
        "direction": "both"     # trade both directions
    }
    
    current_params = baseline_params.copy()
    
    print("\nEvaluating Baseline ORB...")
    n_metrics = evaluate_orb_symbol(df_nifty, current_params)
    b_metrics = evaluate_orb_symbol(df_bank, current_params)
    best_fitness = (n_metrics["train"]["adr"] + b_metrics["train"]["adr"]) / 2.0
    
    print("### BASELINE RESULTS")
    print(f"Parameters: {current_params}")
    print(f"NIFTY Train ADR: {n_metrics['train']['adr']:.4f}%, Sharpe: {n_metrics['train']['sharpe']:.2f}, MaxDD: {n_metrics['train']['max_dd']:.2f}%, WinRate: {n_metrics['train']['win_rate']:.1f}%, PosDays: {n_metrics['train']['positive_days_pct']:.1f}%, Trades: {n_metrics['train']['trades']}")
    print(f"NIFTY Val ADR: {n_metrics['val']['adr']:.4f}%, Sharpe: {n_metrics['val']['sharpe']:.2f}, MaxDD: {n_metrics['val']['max_dd']:.2f}%, WinRate: {n_metrics['val']['win_rate']:.1f}%, PosDays: {n_metrics['val']['positive_days_pct']:.1f}%, Trades: {n_metrics['val']['trades']}")
    print(f"BANKNIFTY Train ADR: {b_metrics['train']['adr']:.4f}%, Sharpe: {b_metrics['train']['sharpe']:.2f}, MaxDD: {b_metrics['train']['max_dd']:.2f}%, WinRate: {b_metrics['train']['win_rate']:.1f}%, PosDays: {b_metrics['train']['positive_days_pct']:.1f}%, Trades: {b_metrics['train']['trades']}")
    print(f"BANKNIFTY Val ADR: {b_metrics['val']['adr']:.4f}%, Sharpe: {b_metrics['val']['sharpe']:.2f}, MaxDD: {b_metrics['val']['max_dd']:.2f}%, WinRate: {b_metrics['val']['win_rate']:.1f}%, PosDays: {b_metrics['val']['positive_days_pct']:.1f}%, Trades: {b_metrics['val']['trades']}")
    print(f"Combined Fitness (Train ADR): {best_fitness:.4f}%")
    print("=" * 60)
    
    experiments = [
        {
            "name": "Iteration 1: Optimize OR Duration (30m vs 15m)",
            "changes": {"or_bars": 2},
            "reasoning": "A wider 30-minute range may reduce false breakouts, creating higher quality trades.",
            "next_experiment": "Optimize Take Profit Multiplier (1.5x)"
        },
        {
            "name": "Iteration 2: Optimize Take Profit Multiplier (1.5x vs 1.0x)",
            "changes": {"tp_mult": 1.5},
            "reasoning": "Increasing target level to 1.5x range width to capture larger breakout extensions.",
            "next_experiment": "Optimize Take Profit Multiplier (2.0x)"
        },
        {
            "name": "Iteration 3: Optimize Take Profit Multiplier (2.0x vs 1.5x)",
            "changes": {"tp_mult": 2.0},
            "reasoning": "Aim for a higher 2:1 reward-to-risk ratio using 2.0x range width TP.",
            "next_experiment": "Optimize Stop Loss Level (Midpoint SL)"
        },
        {
            "name": "Iteration 4: Optimize Stop Loss Level (Midpoint SL)",
            "changes": {"sl_type": "midpoint"},
            "reasoning": "Using the midpoint of the opening range as SL reduces risk per trade and improves average return.",
            "next_experiment": "Optimize breakout buffer (0.05% buffer)"
        },
        {
            "name": "Iteration 5: Optimize breakout buffer (0.05% buffer)",
            "changes": {"buffer_pct": 0.05},
            "reasoning": "Adding a tiny 0.05% price buffer to avoid triggers on noise breakouts.",
            "next_experiment": "Optimize entry cutoff time (13:30 cutoff)"
        },
        {
            "name": "Iteration 6: Optimize entry cutoff time (13:30 cutoff)",
            "changes": {"cutoff_time": "13:30"},
            "reasoning": "Prevent entering late in the session when volatility decreases and ranges shrink.",
            "next_experiment": "Test Long-only vs Short-only vs Both"
        },
        {
            "name": "Iteration 7: Test Short-only Direction filter",
            "changes": {"direction": "short"},
            "reasoning": "Evaluate if the breakout strategy is more robust only trading the short side.",
            "next_experiment": "Final evaluation"
        }
    ]
    
    for idx, exp in enumerate(experiments):
        print(f"\n### ITERATION {idx + 1}: {exp['name']}")
        
        # Propose changes
        test_params = current_params.copy()
        for k, v in exp["changes"].items():
            test_params[k] = v
            
        # Evaluate
        n_m = evaluate_orb_symbol(df_nifty, test_params)
        b_m = evaluate_orb_symbol(df_bank, test_params)
        
        test_fitness = (n_m["train"]["adr"] + b_m["train"]["adr"]) / 2.0
        
        # Overfitting/robustness check
        overfit = False
        overfit_reasons = []
        
        # Decay check if train is positive
        if n_m["train"]["adr"] > 0:
            if n_m["val"]["adr"] <= 0:
                overfit = True
                overfit_reasons.append(f"NIFTY Val ADR is non-positive ({n_m['val']['adr']:.4f}%) while Train ADR is positive.")
            elif n_m["val"]["adr"] < 0.4 * n_m["train"]["adr"]:
                overfit = True
                overfit_reasons.append(f"NIFTY Val ADR ({n_m['val']['adr']:.4f}%) decayed by more than 60% compared to Train ADR ({n_m['train']['adr']:.4f}%).")
                
        if b_m["train"]["adr"] > 0:
            if b_m["val"]["adr"] <= 0:
                overfit = True
                overfit_reasons.append(f"BANKNIFTY Val ADR is non-positive ({b_m['val']['adr']:.4f}%) while Train ADR is positive.")
            elif b_m["val"]["adr"] < 0.4 * b_m["train"]["adr"]:
                overfit = True
                overfit_reasons.append(f"BANKNIFTY Val ADR ({b_m['val']['adr']:.4f}%) decayed by more than 60% compared to Train ADR ({b_m['train']['adr']:.4f}%).")
                
        # Drawdown limit check
        if n_m["train"]["max_dd"] > 15.0 or n_m["val"]["max_dd"] > 15.0:
            overfit = True
            overfit_reasons.append(f"NIFTY drawdown exceeds 15% (Train: {n_m['train']['max_dd']:.2f}%, Val: {n_m['val']['max_dd']:.2f}%).")
        if b_m["train"]["max_dd"] > 15.0 or b_m["val"]["max_dd"] > 15.0:
            overfit = True
            overfit_reasons.append(f"BANKNIFTY drawdown exceeds 15% (Train: {b_m['train']['max_dd']:.2f}%, Val: {b_m['val']['max_dd']:.2f}%).")
            
        # Trade count limit check
        total_train_trades = n_m["train"]["trades"] + b_m["train"]["trades"]
        if total_train_trades < 10:
            overfit = True
            overfit_reasons.append(f"Insufficient trades: {total_train_trades} combined (minimum 10 required).")
            
        improvement = test_fitness > best_fitness
        
        # If train is negative and improving, that is also a valid optimization step!
        if improvement and not overfit:
            decision = "KEEP"
            old_fitness = best_fitness
            best_fitness = test_fitness
            current_params = test_params.copy()
            reasoning = f"Combined fitness improved from {old_fitness:.4f}% to {test_fitness:.4f}% and no overfitting/risk conditions were triggered."
        elif overfit:
            decision = "REVERT"
            reasoning = f"Rejected due to overfitting or risk constraint violations: {'; '.join(overfit_reasons)}."
        else:
            decision = "REVERT"
            reasoning = f"Combined fitness did not improve ({test_fitness:.4f}% <= {best_fitness:.4f}%)."
            
        # Print output in exactly the requested structure
        print(f"1. Parameters tested: {exp['changes']}")
        print("2. Metrics for NIFTY:")
        print(f"   - Train Period: ADR = {n_m['train']['adr']:.4f}%, Sharpe = {n_m['train']['sharpe']:.2f}, MaxDD = {n_m['train']['max_dd']:.2f}%, WinRate = {n_m['train']['win_rate']:.1f}%, PosDays = {n_m['train']['positive_days_pct']:.1f}%, Trades = {n_m['train']['trades']}")
        print(f"   - Validation Period: ADR = {n_m['val']['adr']:.4f}%, Sharpe = {n_m['val']['sharpe']:.2f}, MaxDD = {n_m['val']['max_dd']:.2f}%, WinRate = {n_m['val']['win_rate']:.1f}%, PosDays = {n_m['val']['positive_days_pct']:.1f}%, Trades = {n_m['val']['trades']}")
        print(f"   - Out-of-Sample Period: ADR = {n_m['oos']['adr']:.4f}%, Sharpe = {n_m['oos']['sharpe']:.2f}, MaxDD = {n_m['oos']['max_dd']:.2f}%, WinRate = {n_m['oos']['win_rate']:.1f}%, PosDays = {n_m['oos']['positive_days_pct']:.1f}%, Trades = {n_m['oos']['trades']}")
        print("3. Metrics for BANKNIFTY:")
        print(f"   - Train Period: ADR = {b_m['train']['adr']:.4f}%, Sharpe = {b_m['train']['sharpe']:.2f}, MaxDD = {b_m['train']['max_dd']:.2f}%, WinRate = {b_m['train']['win_rate']:.1f}%, PosDays = {b_m['train']['positive_days_pct']:.1f}%, Trades = {b_m['train']['trades']}")
        print(f"   - Validation Period: ADR = {b_m['val']['adr']:.4f}%, Sharpe = {b_m['val']['sharpe']:.2f}, MaxDD = {b_m['val']['max_dd']:.2f}%, WinRate = {b_m['val']['win_rate']:.1f}%, PosDays = {b_m['val']['positive_days_pct']:.1f}%, Trades = {b_m['val']['trades']}")
        print(f"   - Out-of-Sample Period: ADR = {b_m['oos']['adr']:.4f}%, Sharpe = {b_m['oos']['sharpe']:.2f}, MaxDD = {b_m['oos']['max_dd']:.2f}%, WinRate = {b_m['oos']['win_rate']:.1f}%, PosDays = {b_m['oos']['positive_days_pct']:.1f}%, Trades = {b_m['oos']['trades']}")
        print(f"4. Combined fitness score: {test_fitness:.4f}%")
        print(f"5. Reasoning: {reasoning}")
        print(f"6. Decision: {decision}")
        print(f"7. Next experiment: {exp['next_experiment']}")
        print("=" * 60)
        
    print("\n### OPTIMIZATION SUMMARY")
    print(f"Final Optimized Parameters: {current_params}")
    print(f"Final Combined Train Fitness: {best_fitness:.4f}%")
