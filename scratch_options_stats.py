import pandas as pd

df = pd.read_csv(
    "/root/openalgo-autonomous-research/supertrend_touch_output/july_options_analysis.csv"
)

print("=== OVERALL OPTIONS PERFORMANCE ===")
print(f"Total Trades: {len(df)}")
# Calculate potential drawdown
df["Max Potential Drawdown %"] = ((df["Max Adverse (Low)"] / df["Entry Price"]) - 1) * 100
df["EOD Profit %"] = ((df["EOD Close"] / df["Entry Price"]) - 1) * 100

print(f"Average Max Favorable Excursion: {df['Max Potential Profit %'].mean():.2f}%")
print(f"Average Max Adverse Excursion: {df['Max Potential Drawdown %'].mean():.2f}%")

winners_eod = df[df["EOD Profit %"] > 0]
print(f"EOD Win Rate: {len(winners_eod)} / {len(df)} ({len(winners_eod) / len(df) * 100:.1f}%)")
print(f"Average EOD Return: {df['EOD Profit %'].mean():.2f}%")

print("\n=== PERFORMANCE BY SIGNAL TYPE ===")
buys = df[df["Underlying Signal"] == "BUY_TOUCH"]
sells = df[df["Underlying Signal"] == "SELL_TOUCH"]
print(
    f"BUY Touches (CE Trades): Avg Max Profit {buys['Max Potential Profit %'].mean():.2f}%, Avg EOD Return {buys['EOD Profit %'].mean():.2f}%"
)
print(
    f"SELL Touches (PE Trades): Avg Max Profit {sells['Max Potential Profit %'].mean():.2f}%, Avg EOD Return {sells['EOD Profit %'].mean():.2f}%"
)

print("\n=== OPTION SUPERTREND ALIGNMENT ===")
aligned = df[df["Option ST Trend"] == "UP (1)"]
misaligned = df[df["Option ST Trend"] == "DOWN (-1)"]
print(
    f"Option in Uptrend: {len(aligned)} trades, Avg Max Profit {aligned['Max Potential Profit %'].mean():.2f}%, EOD Win Rate {len(aligned[aligned['EOD Profit %'] > 0]) / max(1, len(aligned)) * 100:.1f}%"
)
print(
    f"Option in Downtrend: {len(misaligned)} trades, Avg Max Profit {misaligned['Max Potential Profit %'].mean():.2f}%, EOD Win Rate {len(misaligned[misaligned['EOD Profit %'] > 0]) / max(1, len(misaligned)) * 100:.1f}%"
)
