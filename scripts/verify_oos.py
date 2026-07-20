import numpy as np
from search_orb_fade import (
    BANKNIFTY_PATH,
    NIFTY_PATH,
    compute_ema,
    load_and_preprocess,
    simulate_orb_fade,
)


def evaluate_on_all_periods(df_nifty, df_bank, params):
    # Pre-group daily data
    # Train: 2025-01-01 to 2025-10-31
    # Val: 2025-11-01 to 2026-02-28
    # OOS: 2026-03-01 to 2026-06-25

    n_train = df_nifty.loc["2025-01-01":"2025-10-31"]
    n_val = df_nifty.loc["2025-11-01":"2026-02-28"]
    n_oos = df_nifty.loc["2026-03-01":"2026-06-25"]

    b_train = df_bank.loc["2025-01-01":"2025-10-31"]
    b_val = df_bank.loc["2025-11-01":"2026-02-28"]
    b_oos = df_bank.loc["2026-03-01":"2026-06-25"]

    n_tr_days = [g.sort_index() for _, g in n_train.groupby(n_train.index.date)]
    n_v_days = [g.sort_index() for _, g in n_val.groupby(n_val.index.date)]
    n_o_days = [g.sort_index() for _, g in n_oos.groupby(n_oos.index.date)]

    b_tr_days = [g.sort_index() for _, g in b_train.groupby(b_train.index.date)]
    b_v_days = [g.sort_index() for _, g in b_val.groupby(b_val.index.date)]
    b_o_days = [g.sort_index() for _, g in b_oos.groupby(b_oos.index.date)]

    # Simulate
    n_tr_t, n_tr_r, n_tr_e = simulate_orb_fade(n_tr_days, params)
    n_v_t, n_v_r, n_v_e = simulate_orb_fade(n_v_days, params)
    n_o_t, n_o_r, n_o_e = simulate_orb_fade(n_o_days, params)

    b_tr_t, b_tr_r, b_tr_e = simulate_orb_fade(b_tr_days, params)
    b_val_t, b_val_r, b_val_e = simulate_orb_fade(b_v_days, params)
    b_o_t, b_o_r, b_o_e = simulate_orb_fade(b_o_days, params)

    def get_stats(trades, ret, eq):
        if len(trades) == 0:
            return {"adr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "trades": 0}
        adr = ret.mean() * 100.0
        cum_max = np.maximum.accumulate(eq)
        dd = (eq - cum_max) / cum_max
        max_dd = abs(dd.min()) * 100.0
        std = ret.std()
        sharpe = (np.sqrt(252) * ret.mean() / std) if std > 0 else 0.0
        return {"adr": adr, "sharpe": sharpe, "max_dd": max_dd, "trades": len(trades)}

    return {
        "nifty": {
            "train": get_stats(n_tr_t, n_tr_r, n_tr_e),
            "val": get_stats(n_v_t, n_v_r, n_v_e),
            "oos": get_stats(n_o_t, n_o_r, n_o_e),
        },
        "banknifty": {
            "train": get_stats(b_tr_t, b_tr_r, b_tr_e),
            "val": get_stats(b_val_t, b_val_r, b_val_e),
            "oos": get_stats(b_o_t, b_o_r, b_o_e),
        },
    }


if __name__ == "__main__":
    df_nifty = load_and_preprocess(NIFTY_PATH)
    df_bank = load_and_preprocess(BANKNIFTY_PATH)
    df_nifty["ema"] = compute_ema(df_nifty["close"], 50)
    df_bank["ema"] = compute_ema(df_bank["close"], 50)

    # Rank 1: Short-only Fade
    rank1_params = {
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

    # Rank 4: Long-only Fade
    rank4_params = {
        "or_bars": 2,
        "tp_mult": 0.5,
        "sl_mult": 1.5,
        "sl_type": "midpoint",
        "buffer_pct": 0.02,
        "direction": "long",
        "min_range_pct": 0.0,
        "max_range_pct": 1.2,
        "use_trend_filter": True,
        "cutoff_time": "13:00",
    }

    print("Evaluating Rank 1 (Short-only Fade)...")
    r1_results = evaluate_on_all_periods(df_nifty, df_bank, rank1_params)
    print("NIFTY:", r1_results["nifty"])
    print("BANKNIFTY:", r1_results["banknifty"])

    print("\nEvaluating Rank 4 (Long-only Fade)...")
    r4_results = evaluate_on_all_periods(df_nifty, df_bank, rank4_params)
    print("NIFTY:", r4_results["nifty"])
    print("BANKNIFTY:", r4_results["banknifty"])
