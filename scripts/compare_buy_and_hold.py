import numpy as np
from search_orb_fade import (
    BANKNIFTY_PATH,
    NIFTY_PATH,
    compute_ema,
    load_and_preprocess,
    simulate_orb_fade,
)


def compare_performance():
    df_nifty = load_and_preprocess(NIFTY_PATH)
    df_bank = load_and_preprocess(BANKNIFTY_PATH)

    # Precompute ema
    df_nifty["ema"] = compute_ema(df_nifty["close"], 50)
    df_bank["ema"] = compute_ema(df_bank["close"], 50)

    # Pre-group whole dataset by day
    n_days = [g.sort_index() for _, g in df_nifty.groupby(df_nifty.index.date)]
    b_days = [g.sort_index() for _, g in df_bank.groupby(df_bank.index.date)]

    params = {
        "or_bars": 1,
        "tp_mult": 1.0,
        "sl_mult": 0.5,
        "sl_type": "opposite",
        "buffer_pct": 0.05,
        "direction": "short",
        "min_range_pct": 0.0,
        "max_range_pct": 1.2,
        "use_trend_filter": True,
        "cutoff_time": "14:30",
    }

    # Run strategy
    _, n_returns, n_eq = simulate_orb_fade(n_days, params)
    _, b_returns, b_eq = simulate_orb_fade(b_days, params)

    # Buy & Hold calculation
    # We aggregate 15m to Daily closes
    n_daily = df_nifty.groupby(df_nifty.index.date)["close"].last()
    b_daily = df_bank.groupby(df_bank.index.date)["close"].last()

    n_bh_returns = n_daily.pct_change().dropna()
    b_bh_returns = b_daily.pct_change().dropna()

    n_bh_eq = 100000.0 * (1.0 + n_bh_returns).cumprod()
    b_bh_eq = 100000.0 * (1.0 + b_bh_returns).cumprod()

    # Performance helper
    def get_stats(ret_series, eq_series):
        total_ret = (eq_series.iloc[-1] - 100000.0) / 100000.0 * 100.0
        adr = ret_series.mean() * 100.0
        std = ret_series.std()
        sharpe = (np.sqrt(252) * ret_series.mean() / std) if std > 0 else 0.0
        cum_max = eq_series.cummax()
        dd = (eq_series - cum_max) / cum_max
        max_dd = abs(dd.min()) * 100.0
        return total_ret, adr, sharpe, max_dd

    n_strat_ret, n_strat_adr, n_strat_sharpe, n_strat_dd = get_stats(n_returns, n_eq)
    b_strat_ret, b_strat_adr, b_strat_sharpe, b_strat_dd = get_stats(b_returns, b_eq)

    n_bh_ret, n_bh_adr, n_bh_sharpe, n_bh_dd = get_stats(n_bh_returns, n_bh_eq)
    b_bh_ret, b_bh_adr, b_bh_sharpe, b_bh_dd = get_stats(b_bh_returns, b_bh_eq)

    print("\n### COMPARISON RESULTS (2025-01-01 to 2026-06-25)")
    print("\n--- NIFTY ---")
    print(
        f"Strategy: Total Return = {n_strat_ret:.2f}%, Daily Return = {n_strat_adr:.4f}%, Sharpe = {n_strat_sharpe:.2f}, Max Drawdown = {n_strat_dd:.2f}%"
    )
    print(
        f"Buy & Hold: Total Return = {n_bh_ret:.2f}%, Daily Return = {n_bh_adr:.4f}%, Sharpe = {n_bh_sharpe:.2f}, Max Drawdown = {n_bh_dd:.2f}%"
    )

    print("\n--- BANKNIFTY ---")
    print(
        f"Strategy: Total Return = {b_strat_ret:.2f}%, Daily Return = {b_strat_adr:.4f}%, Sharpe = {b_strat_sharpe:.2f}, Max Drawdown = {b_strat_dd:.2f}%"
    )
    print(
        f"Buy & Hold: Total Return = {b_bh_ret:.2f}%, Daily Return = {b_bh_adr:.4f}%, Sharpe = {b_bh_sharpe:.2f}, Max Drawdown = {b_bh_dd:.2f}%"
    )


if __name__ == "__main__":
    compare_performance()
