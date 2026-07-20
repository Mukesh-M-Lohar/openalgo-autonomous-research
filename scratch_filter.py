import pandas as pd

df = pd.read_csv(
    "/root/openalgo-autonomous-research/supertrend_touch_output/july_options_analysis.csv"
)
df["EOD Profit %"] = ((df["EOD Close"] / df["Entry Price"]) - 1) * 100

# Filter for Option ST Trend == "UP (1)"
filtered_df = df[df["Option ST Trend"] == "UP (1)"]

print("=== TRADES WHERE OPTION SUPERTREND IS BUY (UP) ===")
print(f"Total Trades: {len(filtered_df)}")

winners = filtered_df[filtered_df["EOD Profit %"] > 0]
print(
    f"Win Rate: {len(winners)} / {len(filtered_df)} ({len(winners) / len(filtered_df) * 100:.1f}%)"
)
print(f"Average Peak Profit: {filtered_df['Max Potential Profit %'].mean():.2f}%")
print(f"Average EOD Return: {filtered_df['EOD Profit %'].mean():.2f}%")

print("\nBreakdown by Underlying Signal:")
buys = filtered_df[filtered_df["Underlying Signal"] == "BUY_TOUCH"]
sells = filtered_df[filtered_df["Underlying Signal"] == "SELL_TOUCH"]

print("BUY_TOUCH (Buying CE while CE is in Uptrend):")
print(f"  Trades: {len(buys)}")
if len(buys) > 0:
    print(f"  Win Rate: {len(buys[buys['EOD Profit %'] > 0]) / len(buys) * 100:.1f}%")
    print(f"  Avg Peak Profit: {buys['Max Potential Profit %'].mean():.2f}%")
    print(f"  Avg EOD Return: {buys['EOD Profit %'].mean():.2f}%")

print("\nSELL_TOUCH (Buying PE while PE is in Uptrend):")
print(f"  Trades: {len(sells)}")
if len(sells) > 0:
    print(f"  Win Rate: {len(sells[sells['EOD Profit %'] > 0]) / len(sells) * 100:.1f}%")
    print(f"  Avg Peak Profit: {sells['Max Potential Profit %'].mean():.2f}%")
    print(f"  Avg EOD Return: {sells['EOD Profit %'].mean():.2f}%")
