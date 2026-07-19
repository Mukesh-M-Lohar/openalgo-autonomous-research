import os
import random
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

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def simulate_orb_fade(days_list: list, params: dict, commission_pct=0.03, slippage_pct=0.02) -> tuple:
    or_bars = params.get("or_bars", 1)
    tp_mult = params.get("tp_mult", 1.0)
    sl_mult = params.get("sl_mult", 1.0)
    sl_type = params.get("sl_type", "opposite")
    buffer_pct = params.get("buffer_pct", 0.0)
    cutoff_time = params.get("cutoff_time", "14:30")
    trade_direction = params.get("direction", "both")
    min_range_pct = params.get("min_range_pct", 0.0)
    max_range_pct = params.get("max_range_pct", 999.0)
    use_trend_filter = params.get("use_trend_filter", False)
    
    cost_factor = (commission_pct + slippage_pct) / 100.0
    cutoff_h, cutoff_m = map(int, cutoff_time.split(":"))
    
    capital = 100000.0
    equity_curve = []
    daily_returns = []
    trades = []
    
    for day_df in days_list:
        if len(day_df) <= or_bars:
            daily_returns.append(0.0)
            equity_curve.append(capital)
            continue
            
        highs = day_df["high"].values
        lows = day_df["low"].values
        opens = day_df["open"].values
        closes = day_df["close"].values
        times = day_df.index
        
        or_high = highs[:or_bars].max()
        or_low = lows[:or_bars].min()
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2.0
        
        if or_range <= 0:
            daily_returns.append(0.0)
            equity_curve.append(capital)
            continue
            
        range_pct = (or_range / or_mid) * 100.0
        if range_pct < min_range_pct or range_pct > max_range_pct:
            daily_returns.append(0.0)
            equity_curve.append(capital)
            continue
            
        long_trigger = or_low * (1.0 - buffer_pct / 100.0) # Fade downside breakout: buy OR low breach
        short_trigger = or_high * (1.0 + buffer_pct / 100.0) # Fade upside breakout: sell OR high breach
        
        position = 0
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        trades_count = 0
        day_pnl = 0.0
        
        emas = day_df["ema"].values if use_trend_filter else None
        
        for i in range(or_bars, len(day_df)):
            t = times[i]
            hour, minute = t.hour, t.minute
            is_eod = (hour == 15 and minute == 15)
            
            can_enter = (hour < cutoff_h) or (hour == cutoff_h and minute < cutoff_m)
            
            if position == 0 and trades_count < 1:
                if can_enter:
                    high_val = highs[i]
                    low_val = lows[i]
                    open_val = opens[i]
                    close_val = closes[i]
                    
                    trend_ok_long = True
                    trend_ok_short = True
                    if use_trend_filter:
                        ema_val = emas[i]
                        trend_ok_long = close_val > ema_val
                        trend_ok_short = close_val < ema_val
                        
                    # Fading downside breakout => LONG
                    triggered_long = (low_val <= long_trigger) and (trade_direction in ["both", "long"]) and trend_ok_long
                    # Fading upside breakout => SHORT
                    triggered_short = (high_val >= short_trigger) and (trade_direction in ["both", "short"]) and trend_ok_short
                    
                    if triggered_long and triggered_short:
                        triggered_short = False
                        
                    if triggered_long:
                        position = 1
                        trades_count += 1
                        entry_price = min(open_val, long_trigger)
                        
                        if sl_type == "opposite":
                            # Stop loss if price continues dropping past some distance
                            sl_price = entry_price - sl_mult * or_range
                        elif sl_type == "midpoint":
                            sl_price = entry_price - (sl_mult / 2.0) * or_range
                        else:
                            sl_price = entry_price - sl_mult * or_range
                            
                        # Profit target is midpoint or range high
                        tp_price = entry_price + tp_mult * or_range
                        
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
                        entry_price = max(open_val, short_trigger)
                        
                        if sl_type == "opposite":
                            sl_price = entry_price + sl_mult * or_range
                        elif sl_type == "midpoint":
                            sl_price = entry_price + (sl_mult / 2.0) * or_range
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
                            
            elif position != 0:
                high_val = highs[i]
                low_val = lows[i]
                close_val = closes[i]
                
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
                    trades.append(pnl)
                    position = 0
                    
        daily_returns.append(day_pnl)
        capital = capital * (1.0 + day_pnl)
        equity_curve.append(capital)
        
    ret_arr = np.array(daily_returns)
    eq_arr = np.array(equity_curve)
    return trades, ret_arr, eq_arr

def evaluate_metrics_fast(n_train, n_val, b_train, b_val, params: dict) -> dict:
    n_train_t, n_train_r, n_train_e = simulate_orb_fade(n_train, params)
    n_val_t, n_val_r, n_val_e = simulate_orb_fade(n_val, params)
    
    b_train_t, b_train_r, b_train_e = simulate_orb_fade(b_train, params)
    b_val_t, b_val_r, b_val_e = simulate_orb_fade(b_val, params)
    
    def get_adr_dd_trades(trades, ret, eq):
        if len(trades) == 0:
            return 0.0, 0.0, 0
        adr = ret.mean() * 100.0
        cum_max = np.maximum.accumulate(eq)
        dd = (eq - cum_max) / cum_max
        max_dd = abs(dd.min()) * 100.0
        return adr, max_dd, len(trades)
        
    n_tr_adr, n_tr_dd, n_tr_tc = get_adr_dd_trades(n_train_t, n_train_r, n_train_e)
    n_v_adr, n_v_dd, n_v_tc = get_adr_dd_trades(n_val_t, n_val_r, n_val_e)
    b_tr_adr, b_tr_dd, b_tr_tc = get_adr_dd_trades(b_train_t, b_train_r, b_train_e)
    b_v_adr, b_v_dd, b_v_tc = get_adr_dd_trades(b_val_t, b_val_r, b_val_e)
    
    return {
        "n_train_adr": n_tr_adr, "n_train_dd": n_tr_dd, "n_train_tc": n_tr_tc,
        "n_val_adr": n_v_adr, "n_val_dd": n_v_dd, "n_val_tc": n_v_tc,
        "b_train_adr": b_tr_adr, "b_train_dd": b_tr_dd, "b_train_tc": b_tr_tc,
        "b_val_adr": b_v_adr, "b_val_dd": b_v_dd, "b_val_tc": b_v_tc
    }

if __name__ == "__main__":
    print("Loading data...")
    df_nifty = load_and_preprocess(NIFTY_PATH)
    df_bank = load_and_preprocess(BANKNIFTY_PATH)
    
    df_nifty["ema"] = compute_ema(df_nifty["close"], 50)
    df_bank["ema"] = compute_ema(df_bank["close"], 50)
    
    train_n = df_nifty.loc["2025-01-01":"2025-10-31"]
    val_n = df_nifty.loc["2025-11-01":"2026-02-28"]
    train_b = df_bank.loc["2025-01-01":"2025-10-31"]
    val_b = df_bank.loc["2025-11-01":"2026-02-28"]
    
    print("Pre-grouping daily data...")
    n_train_days = [group.sort_index() for _, group in train_n.groupby(train_n.index.date)]
    n_val_days = [group.sort_index() for _, group in val_n.groupby(val_n.index.date)]
    b_train_days = [group.sort_index() for _, group in train_b.groupby(train_b.index.date)]
    b_val_days = [group.sort_index() for _, group in val_b.groupby(val_b.index.date)]
    
    print("Beginning ORB Fade Grid Search (3000 iterations)...")
    
    or_bars_options = [1, 2] 
    tp_mult_options = [0.2, 0.5, 1.0, 1.5, 2.0]
    sl_mult_options = [0.5, 1.0, 1.5, 2.0]
    sl_type_options = ["opposite", "midpoint", "multiplier"]
    buffer_pct_options = [0.0, 0.02, 0.05]
    direction_options = ["both", "long", "short"]
    min_range_pct_options = [0.0, 0.1, 0.2]
    max_range_pct_options = [0.8, 1.2, 2.0, 999.0]
    use_trend_filter_options = [True, False]
    cutoff_time_options = ["13:00", "14:00", "14:30"]
    
    results = []
    num_samples = 3000
    random.seed(42)
    
    for count in range(num_samples):
        params = {
            "or_bars": random.choice(or_bars_options),
            "tp_mult": random.choice(tp_mult_options),
            "sl_mult": random.choice(sl_mult_options),
            "sl_type": random.choice(sl_type_options),
            "buffer_pct": random.choice(buffer_pct_options),
            "direction": random.choice(direction_options),
            "min_range_pct": random.choice(min_range_pct_options),
            "max_range_pct": random.choice(max_range_pct_options),
            "use_trend_filter": random.choice(use_trend_filter_options),
            "cutoff_time": random.choice(cutoff_time_options)
        }
        
        m = evaluate_metrics_fast(n_train_days, n_val_days, b_train_days, b_val_days, params)
        
        if m["n_train_tc"] < 5 or m["b_train_tc"] < 5:
            continue
        if m["n_train_dd"] > 15.0 or m["n_val_dd"] > 15.0 or m["b_train_dd"] > 15.0 or m["b_val_dd"] > 15.0:
            continue
        if m["n_train_adr"] > 0:
            if m["n_val_adr"] <= 0 or m["n_val_adr"] < 0.4 * m["n_train_adr"]:
                continue
        if m["b_train_adr"] > 0:
            if m["b_val_adr"] <= 0 or m["b_val_adr"] < 0.4 * m["b_train_adr"]:
                continue
                
        combined_train = (m["n_train_adr"] + m["b_train_adr"]) / 2.0
        combined_val = (m["n_val_adr"] + m["b_val_adr"]) / 2.0
        
        results.append({
            "params": params,
            "metrics": m,
            "combined_train_adr": combined_train,
            "combined_val_adr": combined_val
        })
        
    print(f"Fade Search finished. Found {len(results)} valid configurations.")
    results_sorted = sorted(results, key=lambda x: x["combined_train_adr"], reverse=True)
    
    for idx, r in enumerate(results_sorted[:5]):
        print(f"\nRANK {idx+1} (Combined Train ADR: {r['combined_train_adr']:.4f}% | Combined Val ADR: {r['combined_val_adr']:.4f}%)")
        print(f"Parameters: {r['params']}")
        print(f"Metrics: {r['metrics']}")
