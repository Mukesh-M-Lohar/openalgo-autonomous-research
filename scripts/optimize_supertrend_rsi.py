import numpy as np
import pandas as pd

# Paths
NIFTY_PATH = "/root/openalgo-autonomous-research/data/cache_15m/NIFTY_NSE_INDEX_15m.csv"
BANKNIFTY_PATH = "/root/openalgo-autonomous-research/data/cache_15m/BANKNIFTY_NSE_INDEX_15m.csv"


# --- Indicator Helpers ---
def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    g = d.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    losses = (-d.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + g / losses.replace(0, np.nan))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    hl2 = (df["high"] + df["low"]) / 2
    atr = compute_atr(df, period)
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_upper_band = np.zeros(len(df))
    final_lower_band = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    direction = np.ones(len(df), dtype=int)

    close_vals = df["close"].values
    ub_vals = upper_band.values
    lb_vals = lower_band.values

    fub = 0.0
    flb = 0.0
    st = 0.0
    dir_val = 1

    for i in range(len(df)):
        if i == 0:
            fub = ub_vals[0] if not np.isnan(ub_vals[0]) else 0.0
            flb = lb_vals[0] if not np.isnan(lb_vals[0]) else 0.0
            st = fub
            dir_val = 1
            final_upper_band[0] = fub
            final_lower_band[0] = flb
            supertrend[0] = st
            direction[0] = dir_val
            continue

        # Upper band update
        if ub_vals[i] < fub or close_vals[i - 1] > fub:
            fub = ub_vals[i]
        # Lower band update
        if lb_vals[i] > flb or close_vals[i - 1] < flb:
            flb = lb_vals[i]

        # Direction changes
        if st == final_upper_band[i - 1]:
            if close_vals[i] > fub:
                dir_val = -1
                st = flb
            else:
                dir_val = 1
                st = fub
        else:
            if close_vals[i] < flb:
                dir_val = 1
                st = fub
            else:
                dir_val = -1
                st = flb

        final_upper_band[i] = fub
        final_lower_band[i] = flb
        supertrend[i] = st
        direction[i] = dir_val

    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

    tr_rma = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = (
        100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / tr_rma.replace(0, np.nan)
    )
    plus_di = plus_di.fillna(0.0)
    minus_di = (
        100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / tr_rma.replace(0, np.nan)
    )
    minus_di = minus_di.fillna(0.0)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    dx = dx.fillna(0.0)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx


# --- Data Preparation ---
def load_and_preprocess(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    # Convert UTC index to IST
    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    return df


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()

    # 1. Supertrend
    st_atr_len = params.get("st_atr_len", 10)
    st_mult = params.get("st_mult", 3.0)
    supertrend, direction = compute_supertrend(df, st_atr_len, st_mult)
    df["st_value"] = supertrend
    df["st_direction"] = direction

    # 2. RSI
    rsi_len = params.get("rsi_len", 14)
    rsi_sma_len = params.get("rsi_sma_len", 14)
    rsi = compute_rsi(df["close"], rsi_len)
    df["rsi"] = rsi
    df["rsi_sma"] = compute_sma(rsi, rsi_sma_len)

    # 3. ATR
    atr_len = params.get("atr_len", 14)
    df["atr"] = compute_atr(df, atr_len)

    # 4. ADX
    adx_len = params.get("adx_len", 14)
    df["adx"] = compute_adx(df, adx_len)

    # 5. EMA filter (if enabled)
    if params.get("use_ema_filter", False):
        ema_len = params.get("ema_len", 100)
        df["ema"] = compute_ema(df["close"], ema_len)

    # 6. VWAP filter (if enabled)
    if params.get("use_vwap_filter", False):
        df["date"] = df.index.date
        df["pv"] = df["close"] * df["volume"]
        cum_pv = df.groupby("date")["pv"].cumsum()
        cum_vol = df.groupby("date")["volume"].cumsum()
        df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)

    # 7. Volume filter (if enabled)
    if params.get("use_volume_filter", False):
        vol_sma_len = params.get("vol_sma_len", 20)
        df["vol_sma"] = df["volume"].rolling(vol_sma_len).mean()

    # 8. ATR expansion filter (if enabled)
    if params.get("use_atr_expansion_filter", False):
        atr_sma_len = params.get("atr_sma_len", 20)
        df["atr_sma"] = df["atr"].rolling(atr_sma_len).mean()

    # 9. HTF Filter (resample to 1 hour)
    if params.get("use_htf", False):
        df_1h = (
            df.resample("1h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        _, htf_dir = compute_supertrend(df_1h, st_atr_len, st_mult)
        df_1h["htf_dir"] = htf_dir

        # map back to 15m
        df["htf_direction"] = df.index.map(
            lambda t: df_1h["htf_dir"].asof(t) if not df_1h.empty else 1
        )
        df["htf_direction"] = df["htf_direction"].ffill().fillna(1)
    else:
        df["htf_direction"] = -1

    # Precompute signals to be fast
    rsi = df["rsi"].values
    rsi_sma = df["rsi_sma"].values
    adx = df["adx"].values
    htf_dir = df["htf_direction"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values
    st_direction = df["st_direction"].values
    atr = df["atr"].values

    long_sig = np.zeros(len(df), dtype=bool)
    short_sig = np.zeros(len(df), dtype=bool)

    rsi_bull_min = params.get("rsi_bull_min", 40.0)
    rsi_bull_max = params.get("rsi_bull_max", 50.0)
    rsi_bear_min = params.get("rsi_bear_min", 50.0)
    rsi_bear_max = params.get("rsi_bear_max", 60.0)
    use_rsi_zone = params.get("use_rsi_zone", True)
    use_adx = params.get("use_adx", True)
    adx_threshold = params.get("adx_threshold", 25.0)
    use_htf = params.get("use_htf", True)

    use_ema = params.get("use_ema_filter", False)
    use_vwap = params.get("use_vwap_filter", False)
    use_volume = params.get("use_volume_filter", False)
    use_atr_exp = params.get("use_atr_expansion_filter", False)

    for i in range(2, len(df)):
        # --- Common filters ---
        adx_ok = not use_adx or (adx[i] > adx_threshold)

        # --- Extra custom filters ---
        ema_ok_long = True
        ema_ok_short = True
        if use_ema:
            ema_val = df["ema"].values[i]
            ema_ok_long = close[i] > ema_val
            ema_ok_short = close[i] < ema_val

        vwap_ok_long = True
        vwap_ok_short = True
        if use_vwap:
            vwap_val = df["vwap"].values[i]
            vwap_ok_long = close[i] > vwap_val
            vwap_ok_short = close[i] < vwap_val

        vol_ok = True
        if use_volume:
            vol_sma_val = df["vol_sma"].values[i]
            vol_ok = vol[i] > vol_sma_val

        atr_exp_ok = True
        if use_atr_exp:
            atr_val_scalar = atr[i]
            atr_sma_val = df["atr_sma"].values[i]
            atr_exp_ok = atr_val_scalar > atr_sma_val

        # --- LONG SIGNAL ---
        rsi_falling_prev = (rsi[i - 1] < rsi[i - 2]) and (rsi[i - 1] < rsi_sma[i - 1])
        rsi_turn_up = rsi[i] > rsi[i - 1]
        rsi_bull_ok = not use_rsi_zone or (rsi_bull_min <= rsi[i] <= rsi_bull_max)
        htf_bull = not use_htf or htf_dir[i] == -1
        high_breakout = high[i] > high[i - 1]

        long_sig[i] = (
            st_direction[i] == -1
            and rsi_falling_prev
            and rsi_turn_up
            and rsi_bull_ok
            and adx_ok
            and htf_bull
            and high_breakout
            and ema_ok_long
            and vwap_ok_long
            and vol_ok
            and atr_exp_ok
        )

        # --- SHORT SIGNAL ---
        rsi_rising_prev = (rsi[i - 1] > rsi[i - 2]) and (rsi[i - 1] > rsi_sma[i - 1])
        rsi_turn_down = rsi[i] < rsi[i - 1]
        rsi_bear_ok = not use_rsi_zone or (rsi_bear_min <= rsi[i] <= rsi_bear_max)
        htf_bear = not use_htf or htf_dir[i] == 1
        low_breakout = low[i] < low[i - 1]

        short_sig[i] = (
            st_direction[i] == 1
            and rsi_rising_prev
            and rsi_turn_down
            and rsi_bear_ok
            and adx_ok
            and htf_bear
            and low_breakout
            and ema_ok_short
            and vwap_ok_short
            and vol_ok
            and atr_exp_ok
        )

    df["long_signal"] = long_sig
    df["short_signal"] = short_sig
    return df


# --- Simulator ---
def simulate_trades(
    df: pd.DataFrame, params: dict, initial_cash=100000.0, commission_pct=0.03, slippage_pct=0.02
) -> tuple:
    tp1_mult = params.get("tp1_mult", 1.0)
    tp2_mult = params.get("tp2_mult", 2.0)

    close_vals = df["close"].values
    open_vals = df["open"].values
    high_vals = df["high"].values
    low_vals = df["low"].values
    timestamps = df.index

    long_sig = df["long_signal"].values
    short_sig = df["short_signal"].values
    atr = df["atr"].values
    st_val = df["st_value"].values

    cost_factor = (commission_pct + slippage_pct) / 100.0

    position = 0  # 0, 1 (Long), -1 (Short)
    entry_price = 0.0
    entry_idx = 0
    entry_atr = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    tp1_hit = False

    trades = []
    capital = initial_cash
    equity = np.full(len(df), capital)

    for i in range(len(df)):
        t = timestamps[i]
        hour, minute = t.hour, t.minute
        is_eod = hour == 15 and minute == 15

        if position != 0:
            current_close = close_vals[i]
            if position == 1:
                paper_pnl = (current_close - entry_price) / entry_price
            else:
                paper_pnl = (entry_price - current_close) / entry_price
            equity[i] = capital * (1.0 + paper_pnl)
        else:
            equity[i] = capital

        if position != 0:
            # Active Position exits check
            sl_price = entry_price if tp1_hit else st_val[entry_idx]

            high_val = high_vals[i]
            low_val = low_vals[i]

            exit_triggered = False
            exit_price = close_vals[i]
            reason = "EOD"

            if position == 1:
                if not tp1_hit:
                    # check SL
                    if low_val <= sl_price:
                        exit_triggered = True
                        exit_price = sl_price
                        reason = "SL"
                    # check TP1
                    elif high_val >= tp1_price:
                        tp1_hit = True
                        pnl1 = (tp1_price - entry_price) / entry_price - cost_factor
                        capital += (initial_cash * 0.5) * pnl1

                        # Check remaining in the same candle
                        if low_val <= entry_price:
                            exit_triggered = True
                            exit_price = entry_price
                            reason = "SL_TRAILED"
                        elif high_val >= tp2_price:
                            exit_triggered = True
                            exit_price = tp2_price
                            reason = "TP2"
                else:
                    # TP1 already hit, check remaining half
                    if low_val <= entry_price:
                        exit_triggered = True
                        exit_price = entry_price
                        reason = "SL_TRAILED"
                    elif high_val >= tp2_price:
                        exit_triggered = True
                        exit_price = tp2_price
                        reason = "TP2"

            elif position == -1:
                if not tp1_hit:
                    # check SL
                    if high_val >= sl_price:
                        exit_triggered = True
                        exit_price = sl_price
                        reason = "SL"
                    # check TP1
                    elif low_val <= tp1_price:
                        tp1_hit = True
                        pnl1 = (entry_price - tp1_price) / entry_price - cost_factor
                        capital += (initial_cash * 0.5) * pnl1

                        # Check remaining in the same candle
                        if high_val >= entry_price:
                            exit_triggered = True
                            exit_price = entry_price
                            reason = "SL_TRAILED"
                        elif low_val <= tp2_price:
                            exit_triggered = True
                            exit_price = tp2_price
                            reason = "TP2"
                else:
                    # TP1 already hit, check remaining half
                    if high_val >= entry_price:
                        exit_triggered = True
                        exit_price = entry_price
                        reason = "SL_TRAILED"
                    elif low_val <= tp2_price:
                        exit_triggered = True
                        exit_price = tp2_price
                        reason = "TP2"

            if not exit_triggered and is_eod:
                exit_triggered = True
                exit_price = close_vals[i]
                reason = "EOD"

            if exit_triggered:
                # final exit computation
                if reason in ["EOD", "SL"]:
                    if not tp1_hit:
                        pnl = (
                            (exit_price - entry_price) / entry_price
                            if position == 1
                            else (entry_price - exit_price) / entry_price
                        ) - 2 * cost_factor
                        capital += initial_cash * pnl
                    else:
                        pnl2 = (
                            (exit_price - entry_price) / entry_price
                            if position == 1
                            else (entry_price - exit_price) / entry_price
                        ) - cost_factor
                        capital += (initial_cash * 0.5) * pnl2
                else:
                    pnl2 = (
                        (exit_price - entry_price) / entry_price
                        if position == 1
                        else (entry_price - exit_price) / entry_price
                    ) - cost_factor
                    capital += (initial_cash * 0.5) * pnl2

                trades.append(
                    {
                        "entry_time": timestamps[entry_idx],
                        "exit_time": t,
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "reason": reason,
                        "pnl_pct": (capital - initial_cash) / initial_cash * 100.0,
                    }
                )

                position = 0
                initial_cash = capital
                equity[i:] = capital

        else:
            # Entry Check
            if not is_eod and (hour < 15 or (hour == 15 and minute < 15)):
                if long_sig[i]:
                    position = 1
                    entry_idx = i
                    entry_price = open_vals[i + 1] if i + 1 < len(df) else close_vals[i]
                    entry_atr = atr[i]
                    tp1_price = entry_price + entry_atr * tp1_mult
                    tp2_price = entry_price + entry_atr * tp2_mult
                    tp1_hit = False
                elif short_sig[i]:
                    position = -1
                    entry_idx = i
                    entry_price = open_vals[i + 1] if i + 1 < len(df) else close_vals[i]
                    entry_atr = atr[i]
                    tp1_price = entry_price - entry_atr * tp1_mult
                    tp2_price = entry_price - entry_atr * tp2_mult
                    tp1_hit = False

    df_eq = pd.DataFrame(equity, index=df.index, columns=["equity"])
    df_eq["date"] = df_eq.index.date
    daily_eq = df_eq.groupby("date")["equity"].last()
    daily_returns = daily_eq.pct_change().dropna()

    return trades, daily_returns, daily_eq


# --- Evaluation Helper ---
def evaluate_symbol(df: pd.DataFrame, params: dict) -> dict:
    df_enriched = generate_signals(df, params)

    df_train = df_enriched.loc["2025-01-01":"2025-10-31"]
    df_val = df_enriched.loc["2025-11-01":"2026-02-28"]
    df_oos = df_enriched.loc["2026-03-01":"2026-06-25"]

    t_trades, t_ret, t_eq = simulate_trades(df_train, params)
    v_trades, v_ret, v_eq = simulate_trades(df_val, params)
    o_trades, o_ret, o_eq = simulate_trades(df_oos, params)

    def get_metrics(trades, ret, eq):
        if not trades:
            return {"adr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "trades": 0}
        adr = ret.mean() * 100.0
        std = ret.std()
        sharpe = (np.sqrt(252) * ret.mean() / std) if std > 0 else 0.0
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max
        max_dd = abs(dd.min()) * 100.0
        return {"adr": adr, "sharpe": sharpe, "max_dd": max_dd, "trades": len(trades)}

    return {
        "train": get_metrics(t_trades, t_ret, t_eq),
        "val": get_metrics(v_trades, v_ret, v_eq),
        "oos": get_metrics(o_trades, o_ret, o_eq),
    }


# --- Execution ---
if __name__ == "__main__":
    print("Loading NIFTY and BANKNIFTY data...")
    df_nifty = load_and_preprocess(NIFTY_PATH)
    df_bank = load_and_preprocess(BANKNIFTY_PATH)

    # Baseline Parameters
    baseline_params = {
        "st_atr_len": 10,
        "st_mult": 3.0,
        "rsi_len": 14,
        "rsi_sma_len": 14,
        "use_rsi_zone": True,
        "rsi_bull_min": 40.0,
        "rsi_bull_max": 50.0,
        "rsi_bear_min": 50.0,
        "rsi_bear_max": 60.0,
        "atr_len": 14,
        "tp1_mult": 1.0,
        "tp2_mult": 2.0,
        "use_adx": True,
        "adx_len": 14,
        "adx_threshold": 25.0,
        "use_htf": True,
        # New filters
        "use_ema_filter": False,
        "ema_len": 100,
        "use_vwap_filter": False,
        "use_volume_filter": False,
        "vol_sma_len": 20,
        "use_atr_expansion_filter": False,
        "atr_sma_len": 20,
    }

    current_params = baseline_params.copy()

    print("\nEvaluating Baseline...")
    n_metrics = evaluate_symbol(df_nifty, current_params)
    b_metrics = evaluate_symbol(df_bank, current_params)
    best_fitness = (n_metrics["train"]["adr"] + b_metrics["train"]["adr"]) / 2.0

    print("### BASELINE RESULTS")
    print(f"Parameters: {current_params}")
    print(
        f"NIFTY Train ADR: {n_metrics['train']['adr']:.4f}%, Sharpe: {n_metrics['train']['sharpe']:.2f}, MaxDD: {n_metrics['train']['max_dd']:.2f}%, Trades: {n_metrics['train']['trades']}"
    )
    print(
        f"NIFTY Val ADR: {n_metrics['val']['adr']:.4f}%, Sharpe: {n_metrics['val']['sharpe']:.2f}, MaxDD: {n_metrics['val']['max_dd']:.2f}%, Trades: {n_metrics['val']['trades']}"
    )
    print(
        f"BANKNIFTY Train ADR: {b_metrics['train']['adr']:.4f}%, Sharpe: {b_metrics['train']['sharpe']:.2f}, MaxDD: {b_metrics['train']['max_dd']:.2f}%, Trades: {b_metrics['train']['trades']}"
    )
    print(
        f"BANKNIFTY Val ADR: {b_metrics['val']['adr']:.4f}%, Sharpe: {b_metrics['val']['sharpe']:.2f}, MaxDD: {b_metrics['val']['max_dd']:.2f}%, Trades: {b_metrics['val']['trades']}"
    )
    print(f"Combined Fitness (Train ADR): {best_fitness:.4f}%")
    print("=" * 60)

    experiments = [
        {
            "name": "Iteration 1: Optimize ST Multiplier (lower to 1.5)",
            "changes": {"st_mult": 1.5},
            "reasoning": "Catch quicker intraday 15m trend reversals by narrowing the Supertrend multiplier.",
            "next_experiment": "Disable RSI Zone Filter",
        },
        {
            "name": "Iteration 2: Disable RSI Zone Filter",
            "changes": {"use_rsi_zone": False},
            "reasoning": "RSI zone restriction might block otherwise good pullbacks. Disabling it to trade any pullback.",
            "next_experiment": "Optimize RSI parameters (shorter period)",
        },
        {
            "name": "Iteration 3: Shorter RSI period (rsi_len=7, rsi_sma_len=7)",
            "changes": {"rsi_len": 7, "rsi_sma_len": 7},
            "reasoning": "A faster 7-period RSI can detect pullbacks and turns more quickly on the 15-minute chart.",
            "next_experiment": "Optimize Take Profit multipliers",
        },
        {
            "name": "Iteration 4: Increase Take Profit Multipliers (tp1_mult=1.5, tp2_mult=3.0)",
            "changes": {"tp1_mult": 1.5, "tp2_mult": 3.0},
            "reasoning": "Increase risk-reward ratio by targeting larger moves while holding trailing SL to breakeven.",
            "next_experiment": "Enable EMA filter",
        },
        {
            "name": "Iteration 5: Enable EMA Trend Filter (ema_len=50)",
            "changes": {"use_ema_filter": True, "ema_len": 50},
            "reasoning": "Ensure entry signals align with the short-to-medium term daily trend (50 EMA).",
            "next_experiment": "Enable VWAP filter",
        },
        {
            "name": "Iteration 6: Enable VWAP Filter",
            "changes": {"use_vwap_filter": True},
            "reasoning": "VWAP acts as a critical intraday pivot; longs above VWAP and shorts below VWAP verify buyers/sellers control.",
            "next_experiment": "Lower ADX threshold",
        },
        {
            "name": "Iteration 7: Lower ADX threshold (adx_threshold=20)",
            "changes": {"adx_threshold": 20.0},
            "reasoning": "Lowering ADX threshold from 25 to 20 to capture entries in moderate trending environments.",
            "next_experiment": "Enable Volume filter",
        },
        {
            "name": "Iteration 8: Enable Volume Filter (vol_sma_len=20)",
            "changes": {"use_volume_filter": True, "vol_sma_len": 20},
            "reasoning": "Filter entry setups to only occur when volume exceeds its 20-candle moving average, ensuring high liquidity.",
            "next_experiment": "Enable ATR expansion filter",
        },
        {
            "name": "Iteration 9: Enable ATR Expansion Filter (atr_sma_len=20)",
            "changes": {"use_atr_expansion_filter": True, "atr_sma_len": 20},
            "reasoning": "Ensure there is high relative volatility (ATR > SMA ATR) when entering trades to avoid range-bound chop.",
            "next_experiment": "Final verification",
        },
    ]

    for idx, exp in enumerate(experiments):
        print(f"\n### ITERATION {idx + 1}: {exp['name']}")

        # Propose changes
        test_params = current_params.copy()
        for k, v in exp["changes"].items():
            test_params[k] = v

        # Evaluate
        n_m = evaluate_symbol(df_nifty, test_params)
        b_m = evaluate_symbol(df_bank, test_params)

        test_fitness = (n_m["train"]["adr"] + b_m["train"]["adr"]) / 2.0

        # Check overfitting/robustness
        overfit = False
        overfit_reasons = []

        # Check decay only if train is positive
        if n_m["train"]["adr"] > 0:
            if n_m["val"]["adr"] <= 0:
                overfit = True
                overfit_reasons.append(
                    f"NIFTY Val ADR is non-positive ({n_m['val']['adr']:.4f}%) while Train ADR is positive."
                )
            elif n_m["val"]["adr"] < 0.4 * n_m["train"]["adr"]:
                overfit = True
                overfit_reasons.append(
                    f"NIFTY Val ADR ({n_m['val']['adr']:.4f}%) decayed by more than 60% compared to Train ADR ({n_m['train']['adr']:.4f}%)."
                )

        if b_m["train"]["adr"] > 0:
            if b_m["val"]["adr"] <= 0:
                overfit = True
                overfit_reasons.append(
                    f"BANKNIFTY Val ADR is non-positive ({b_m['val']['adr']:.4f}%) while Train ADR is positive."
                )
            elif b_m["val"]["adr"] < 0.4 * b_m["train"]["adr"]:
                overfit = True
                overfit_reasons.append(
                    f"BANKNIFTY Val ADR ({b_m['val']['adr']:.4f}%) decayed by more than 60% compared to Train ADR ({b_m['train']['adr']:.4f}%)."
                )

        # Drawdown limit check
        if n_m["train"]["max_dd"] > 15.0 or n_m["val"]["max_dd"] > 15.0:
            overfit = True
            overfit_reasons.append(
                f"NIFTY drawdown exceeds 15% (Train: {n_m['train']['max_dd']:.2f}%, Val: {n_m['val']['max_dd']:.2f}%)."
            )
        if b_m["train"]["max_dd"] > 15.0 or b_m["val"]["max_dd"] > 15.0:
            overfit = True
            overfit_reasons.append(
                f"BANKNIFTY drawdown exceeds 15% (Train: {b_m['train']['max_dd']:.2f}%, Val: {b_m['val']['max_dd']:.2f}%)."
            )

        # Minimum trade count check (at least 10 trades combined in train period)
        total_train_trades = n_m["train"]["trades"] + b_m["train"]["trades"]
        if total_train_trades < 10:
            overfit = True
            overfit_reasons.append(
                f"Insufficient trades: {total_train_trades} combined (minimum 10 required)."
            )

        improvement = test_fitness > best_fitness

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
            reasoning = (
                f"Combined fitness did not improve ({test_fitness:.4f}% <= {best_fitness:.4f}%)."
            )

        # Print output in exactly the requested structure
        print(f"1. Parameters tested: {exp['changes']}")
        print("2. Metrics for NIFTY:")
        print(
            f"   - Train Period: ADR = {n_m['train']['adr']:.4f}%, Sharpe = {n_m['train']['sharpe']:.2f}, MaxDD = {n_m['train']['max_dd']:.2f}%, Trades = {n_m['train']['trades']}"
        )
        print(
            f"   - Validation Period: ADR = {n_m['val']['adr']:.4f}%, Sharpe = {n_m['val']['sharpe']:.2f}, MaxDD = {n_m['val']['max_dd']:.2f}%, Trades = {n_m['val']['trades']}"
        )
        print(
            f"   - Out-of-Sample Period: ADR = {n_m['oos']['adr']:.4f}%, Sharpe = {n_m['oos']['sharpe']:.2f}, MaxDD = {n_m['oos']['max_dd']:.2f}%, Trades = {n_m['oos']['trades']}"
        )
        print("3. Metrics for BANKNIFTY:")
        print(
            f"   - Train Period: ADR = {b_m['train']['adr']:.4f}%, Sharpe = {b_m['train']['sharpe']:.2f}, MaxDD = {b_m['train']['max_dd']:.2f}%, Trades = {b_m['train']['trades']}"
        )
        print(
            f"   - Validation Period: ADR = {b_m['val']['adr']:.4f}%, Sharpe = {b_m['val']['sharpe']:.2f}, MaxDD = {b_m['val']['max_dd']:.2f}%, Trades = {b_m['val']['trades']}"
        )
        print(
            f"   - Out-of-Sample Period: ADR = {b_m['oos']['adr']:.4f}%, Sharpe = {b_m['oos']['sharpe']:.2f}, MaxDD = {b_m['oos']['max_dd']:.2f}%, Trades = {b_m['oos']['trades']}"
        )
        print(f"4. Combined fitness score: {test_fitness:.4f}%")
        print(f"5. Reasoning: {reasoning}")
        print(f"6. Decision: {decision}")
        print(f"7. Next experiment: {exp['next_experiment']}")
        print("=" * 60)

    print("\n### OPTIMIZATION SUMMARY")
    print(f"Final Optimized Parameters: {current_params}")
    print(f"Final Combined Train Fitness: {best_fitness:.4f}%")
