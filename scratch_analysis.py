import pandas as pd

# Load data
df = pd.read_csv(
    "/root/openalgo-autonomous-research/supertrend_touch_output/supertrend_touches_ALL.csv"
)
df["bounce_threshold_hit"] = df["bounce_threshold_hit"].astype(bool)

# Separate BUY and SELL touches
buys = df[df["signal_type"] == "BUY_TOUCH"].copy()
sells = df[df["signal_type"] == "SELL_TOUCH"].copy()

print("=== OVERALL HIT RATES ===")
print(f"BUY Touches: {len(buys)}, Hit Rate: {buys['bounce_threshold_hit'].mean() * 100:.1f}%")
print(f"SELL Touches: {len(sells)}, Hit Rate: {sells['bounce_threshold_hit'].mean() * 100:.1f}%\n")


def analyze_condition(df_subset, condition_series, condition_name):
    total = len(df_subset)
    if total == 0:
        return
    hits = df_subset[condition_series]["bounce_threshold_hit"]
    if len(hits) == 0:
        return
    print(f"  {condition_name}: {hits.mean() * 100:.1f}% (N={len(hits)})")


print("=== BUY TOUCHES ANALYSIS (Trend: Green/Up) ===")
# Trend strength
analyze_condition(buys, buys["adx"] > 25, "Strong Trend (ADX > 25)")
analyze_condition(buys, buys["adx"] <= 25, "Weak Trend (ADX <= 25)")
# Momentum
analyze_condition(buys, buys["rsi"] > 50, "RSI > 50")
analyze_condition(buys, buys["rsi"] <= 50, "RSI <= 50")
# MACD
analyze_condition(buys, buys["macd_hist"] > 0, "MACD Histogram Positive")
analyze_condition(buys, buys["macd_hist"] <= 0, "MACD Histogram Negative")
# Distance from 200 EMA
analyze_condition(buys, buys["close"] > buys["ema_200"], "Price Above 200 EMA")
analyze_condition(buys, buys["close"] <= buys["ema_200"], "Price Below 200 EMA")

print("\n=== SELL TOUCHES ANALYSIS (Trend: Red/Down) ===")
analyze_condition(sells, sells["adx"] > 25, "Strong Trend (ADX > 25)")
analyze_condition(sells, sells["adx"] <= 25, "Weak Trend (ADX <= 25)")
analyze_condition(sells, sells["rsi"] < 50, "RSI < 50")
analyze_condition(sells, sells["rsi"] >= 50, "RSI >= 50")
analyze_condition(sells, sells["macd_hist"] < 0, "MACD Histogram Negative")
analyze_condition(sells, sells["macd_hist"] >= 0, "MACD Histogram Positive")
analyze_condition(sells, sells["close"] < sells["ema_200"], "Price Below 200 EMA")
analyze_condition(sells, sells["close"] >= sells["ema_200"], "Price Above 200 EMA")
