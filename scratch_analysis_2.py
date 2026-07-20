import sys

import pandas as pd

csv_file = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/root/openalgo-autonomous-research/supertrend_touch_output/supertrend_touches_ALL.csv"
)
df = pd.read_csv(csv_file)
df["bounce_threshold_hit"] = df["bounce_threshold_hit"].astype(bool)

# Parse timestamp to extract hour and time of day
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["time_of_day"] = df["hour"] + df["minute"] / 60.0


def categorize_time(tod):
    if tod < 11:
        return "Morning (Before 11 AM)"
    elif tod < 13.5:
        return "Mid-day (11 AM to 1:30 PM)"
    else:
        return "Afternoon (After 1:30 PM)"


df["session"] = df["time_of_day"].apply(categorize_time)

buys = df[df["signal_type"] == "BUY_TOUCH"].copy()
sells = df[df["signal_type"] == "SELL_TOUCH"].copy()

print("=== TIME OF DAY ANALYSIS ===")
for session in [
    "Morning (Before 11 AM)",
    "Mid-day (11 AM to 1:30 PM)",
    "Afternoon (After 1:30 PM)",
]:
    b = buys[buys["session"] == session]
    s = sells[sells["session"] == session]
    print(f"{session}:")
    if len(b) > 0:
        print(f"  BUY Touches: {len(b)}, Hit Rate: {b['bounce_threshold_hit'].mean() * 100:.1f}%")
    if len(s) > 0:
        print(f"  SELL Touches: {len(s)}, Hit Rate: {s['bounce_threshold_hit'].mean() * 100:.1f}%")

print("\n=== BARS TO RESOLUTION ===")


def analyze_bars(df_subset, label):
    hits = df_subset[df_subset["bounce_threshold_hit"]]
    fails = df_subset[~df_subset["bounce_threshold_hit"]]

    print(f"{label}:")
    if not hits.empty:
        avg_bars_to_hit = hits["bars_to_threshold_hit"].mean()
        med_bars_to_hit = hits["bars_to_threshold_hit"].median()
        print(
            f"  Successful Bounces: took on avg {avg_bars_to_hit:.1f} bars (median {med_bars_to_hit:.1f} bars) to hit target"
        )

    if not fails.empty:
        avg_bars_to_rev = fails["bars_to_reversal"].mean()
        med_bars_to_rev = fails["bars_to_reversal"].median()
        print(
            f"  Failed Bounces (Reversals): took on avg {avg_bars_to_rev:.1f} bars (median {med_bars_to_rev:.1f} bars) to reverse"
        )


analyze_bars(buys, "BUY Touches")
analyze_bars(sells, "SELL Touches")

print("\n=== DISTANCE TO BAND (TOUCH CLOSENESS) ===")


# How close was the touch?
def analyze_closeness(df_subset, label):
    print(f"{label}:")
    very_close = df_subset[df_subset["band_distance_pct"] < 0.05]
    far_touch = df_subset[df_subset["band_distance_pct"] >= 0.05]
    if len(very_close) > 0:
        print(
            f"  Very Close (<0.05% away): {very_close['bounce_threshold_hit'].mean() * 100:.1f}% (N={len(very_close)})"
        )
    if len(far_touch) > 0:
        print(
            f"  Farther (0.05-0.15% away): {far_touch['bounce_threshold_hit'].mean() * 100:.1f}% (N={len(far_touch)})"
        )


analyze_closeness(buys, "BUY Touches")
analyze_closeness(sells, "SELL Touches")
