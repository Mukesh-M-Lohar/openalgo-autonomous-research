import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from openalgo import api  # noqa: E402

from scripts.supertrend_scanner.indicators import bollinger_bands  # noqa: E402
from scripts.supertrend_scanner.ss_scanner import build_indicators, fetch_history  # noqa: E402

# --- Config ---
load_dotenv(find_dotenv(), override=False)
OPENALGO_API_KEY = os.getenv("OPENALGO_API_KEY")
OPENALGO_HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")

SYMBOLS = [
    {"symbol": "NIFTY", "exchange": "NSE_INDEX"},
]

INTERVAL = "5m"
HTF_INTERVAL = "15m"
START_DATE = "2024-01-01"
END_DATE = "2026-07-21"

BOUNCE_POINTS_BY_SYMBOL = {
    "NIFTY": 50,
}


def detect_bb_touches(df: pd.DataFrame) -> pd.DataFrame:
    """Detects when price touches the Bollinger Bands."""
    d = df.copy()

    # We only care about lower band for BUY touches for this mean reversion strategy
    # Or upper band for SELL touches. Let's do both to double the dataset.
    touched_lower = d["low"] <= d["bb_lower"]
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]  # noqa: E741
    touched_upper = d["high"] >= d["bb_upper"]

    d["signal_type"] = np.where(
        touched_lower, "BUY_TOUCH", np.where(touched_upper, "SELL_TOUCH", None)
    )

    return d[d["signal_type"].notna()].copy()


def evaluate_bb_touch_outcomes(
    full_df: pd.DataFrame, touches: pd.DataFrame, bounce_pts: float
) -> pd.DataFrame:
    """
    Evaluates whether the touch resulted in a successful mean-reversion bounce.
    A bounce is successful if it reaches +bounce_pts before reaching -bounce_pts
    and before touching the opposite band.
    """
    records = []
    # Convert index to integer positions for fast iteration
    time_to_idx = {t: i for i, t in enumerate(full_df.index)}

    for t_idx, touch_row in touches.iterrows():
        start_i = time_to_idx[t_idx]
        sig_type = touch_row["signal_type"]
        entry_price = touch_row["close"]

        bounced = False
        bounce_pct = 0.0
        peak_bounce = 0.0
        peak_time = None

        for i in range(start_i + 1, len(full_df)):
            curr_row = full_df.iloc[i]

            if sig_type == "BUY_TOUCH":
                points_gained = curr_row["high"] - entry_price
                points_lost = entry_price - curr_row["low"]
                opposite_band_touch = curr_row["high"] >= curr_row["bb_upper"]
            else:  # SELL_TOUCH
                points_gained = entry_price - curr_row["low"]
                points_lost = curr_row["high"] - entry_price
                opposite_band_touch = curr_row["low"] <= curr_row["bb_lower"]

            # Update peak bounce
            if points_gained > peak_bounce:
                peak_bounce = points_gained
                peak_time = full_df.index[i]

            # Check success condition
            if points_gained >= bounce_pts:
                bounced = True
                bounce_pct = bounce_pts / entry_price * 100
                break

            # Check failure condition (hit stop loss or touched opposite band)
            if points_lost >= bounce_pts or opposite_band_touch:
                break

        records.append(
            {
                "bounced": bounced,
                "bounce_pct": bounce_pct,
                "peak_bounce_points": peak_bounce,
                "peak_bounce_timestamp": peak_time,
                "bounce_threshold_points": bounce_pts,
            }
        )

    return pd.DataFrame(records, index=touches.index)


def main():
    client = api(api_key=OPENALGO_API_KEY, host=OPENALGO_HOST)

    out_dir = project_root / "scripts" / "bollinger_scanner" / "bollinger_touch_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_touches = []

    for sym_info in SYMBOLS:
        symbol = sym_info["symbol"]
        exchange = sym_info["exchange"]
        bounce_pts = BOUNCE_POINTS_BY_SYMBOL.get(symbol, 50)

        print(f"[{symbol}] Fetching {START_DATE} to {END_DATE}...")
        df_ltf = fetch_history(client, symbol, exchange, INTERVAL, START_DATE, END_DATE)
        df_htf = fetch_history(client, symbol, exchange, HTF_INTERVAL, START_DATE, END_DATE)

        print(f"[{symbol}] Computing indicators...")
        ind_ltf = build_indicators(df_ltf)
        ind_htf = build_indicators(df_htf)
        ind_htf.columns = [f"htf_{c}" for c in ind_htf.columns]

        # Add Bollinger Bands
        bb_mid, bb_upper, bb_lower, bb_pctb, bb_bw = bollinger_bands(
            ind_ltf["close"], period=20, num_std=2.0
        )
        ind_ltf["bb_lower"] = bb_lower
        ind_ltf["bb_upper"] = bb_upper

        print(f"[{symbol}] Merging HTF and LTF...")
        ind_htf_aligned = ind_htf.reindex(ind_ltf.index, method="ffill")
        df_merged = ind_ltf.join(ind_htf_aligned)

        print(f"[{symbol}] Detecting BB touches...")
        touches = detect_bb_touches(df_merged)
        print(f"[{symbol}] Found {len(touches)} touches. Evaluating outcomes...")

        outcomes = evaluate_bb_touch_outcomes(df_merged, touches, bounce_pts)

        # Combine
        combined = pd.concat([touches, outcomes], axis=1)
        combined.insert(0, "symbol", symbol)

        all_touches.append(combined)

    final_df = pd.concat(all_touches)
    out_path = out_dir / "bb_touches_ALL_5m.csv"
    final_df.to_csv(out_path)
    print(f"\nDone! Saved {len(final_df)} touches to {out_path}")


if __name__ == "__main__":
    main()
