import pandas as pd
from search_orb_fade import (
    BANKNIFTY_PATH,
    NIFTY_PATH,
    compute_ema,
    load_and_preprocess,
    simulate_orb_fade,
)


def generate_daily_report():
    print("Loading data...")
    df_nifty = load_and_preprocess(NIFTY_PATH)
    df_bank = load_and_preprocess(BANKNIFTY_PATH)

    # Precompute ema
    df_nifty["ema"] = compute_ema(df_nifty["close"], 50)
    df_bank["ema"] = compute_ema(df_bank["close"], 50)

    # Pre-group whole dataset by day
    n_days = [g.sort_index() for _, g in df_nifty.groupby(df_nifty.index.date)]
    b_days = [g.sort_index() for _, g in df_bank.groupby(df_bank.index.date)]

    # Rank 1 winning parameters
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

    # Run simulation
    n_trades, n_returns, n_equity = simulate_orb_fade(n_days, params)
    b_trades, b_returns, b_equity = simulate_orb_fade(b_days, params)

    # Align dates
    dates = sorted(df_nifty.groupby(df_nifty.index.date).groups.keys())

    # Build dataframe of daily returns
    daily_df = pd.DataFrame(
        {
            "NIFTY_Return_Pct": n_returns * 100.0,
            "BANKNIFTY_Return_Pct": b_returns * 100.0,
        },
        index=dates,
    )

    daily_df["Combined_Return_Pct"] = (
        daily_df["NIFTY_Return_Pct"] + daily_df["BANKNIFTY_Return_Pct"]
    ) / 2.0

    # Calculate stats
    stats = {}
    for col in daily_df.columns:
        series = daily_df[col]
        # Active days (days where a trade was taken, i.e. return != 0)
        active_series = series[series != 0]

        stats[col] = {
            "Total Days": len(series),
            "Trading Days": len(active_series),
            "Win Days (Positive)": int(sum(active_series > 0)),
            "Loss Days (Negative)": int(sum(active_series < 0)),
            "Daily Win Rate (on trading days)": f"{(sum(active_series > 0) / len(active_series) * 100.0):.1f}%"
            if len(active_series) > 0
            else "0.0%",
            "Max Daily Return": f"{series.max():.2f}%",
            "Min Daily Return": f"{series.min():.2f}%",
            "Avg Return on Trading Days": f"{active_series.mean():.4f}%"
            if len(active_series) > 0
            else "0.0000%",
        }

    print("\n### DAILY BACKTEST STATISTICS (Entire Period)")
    for col, stat in stats.items():
        print(f"\nInstrument: {col}")
        for k, v in stat.items():
            print(f"  - {k}: {v}")

    # Save CSV of daily returns
    output_path = "/root/.gemini/antigravity-ide/brain/92559aa9-befc-4394-bd56-597d5f9a972e/daily_returns_pnl.csv"
    daily_df.to_csv(output_path)
    print(f"\nDaily returns logs saved to {output_path}")


if __name__ == "__main__":
    generate_daily_report()
