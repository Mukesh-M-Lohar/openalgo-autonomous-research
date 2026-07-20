import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openalgo import api

load_dotenv(".env")
client = api(api_key=os.getenv("OPENALGO_API_KEY"), host="http://127.0.0.1:5000")

# Load Trades
trades = pd.read_csv("supertrend_touch_output/backtest_trades.csv")
trades["Touch Time"] = pd.to_datetime(trades["Touch Time"])
trades = trades.sort_values("Touch Time")

# Simple Cumulative Return (%)
trades["Cumulative Return %"] = trades["Return %"].cumsum()

# Fetch Underlying Data
hist = client.history(
    symbol="BANKNIFTY",
    exchange="NSE_INDEX",
    interval="1d",
    start_date="2026-07-01",
    end_date=datetime.now().strftime("%Y-%m-%d"),
)
bnh = pd.DataFrame(hist)
bnh["datetime"] = pd.to_datetime(bnh["datetime"])
bnh = bnh.sort_values("datetime")
initial_price = bnh.iloc[0]["close"]
bnh["B&H Return %"] = ((bnh["close"] / initial_price) - 1) * 100

plt.figure(figsize=(12, 6))
plt.plot(
    trades["Touch Time"],
    trades["Cumulative Return %"],
    label="Strategy (Offset 500 | Trail 10%)",
    color="blue",
    linewidth=2,
)
plt.plot(
    bnh["datetime"],
    bnh["B&H Return %"],
    label="BANKNIFTY Buy & Hold",
    color="orange",
    linewidth=2,
    linestyle="--",
)

plt.title("Strategy Performance vs Buy & Hold (Non-Compounded)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("supertrend_touch_output/equity_curve.png")
print("Plot saved to supertrend_touch_output/equity_curve.png")
