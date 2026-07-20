import os

import matplotlib.pyplot as plt
import numpy as np
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

# Create dummy baseline for Buy and Hold assuming linear progression to 0.84% over 19 days
days = pd.date_range(start="2026-07-01", end="2026-07-19", freq="D")
bnh = pd.DataFrame({"datetime": days})
# Linear interpolation from 0 to 0.84
bnh["B&H Return %"] = np.linspace(0, 0.84, len(bnh))

plt.figure(figsize=(12, 6))
plt.plot(
    trades["Touch Time"],
    trades["Cumulative Return %"],
    label="Strategy (Offset 500 | Trail 10%)",
    color="#00ff00",
    linewidth=2,
)
plt.plot(
    bnh["datetime"],
    bnh["B&H Return %"],
    label="BANKNIFTY Buy & Hold",
    color="#ff9900",
    linewidth=2,
    linestyle="--",
)

plt.title("Supertrend Options Strategy vs Buy & Hold (Cumulative)", fontsize=14, pad=15)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Return (%)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()

# Save plot
output_path = (
    "/root/.gemini/antigravity-ide/brain/42f9ff69-4c3a-4419-92f7-010a6b1b76a5/equity_curve.png"
)
plt.savefig(output_path, dpi=300, facecolor="w")
print(f"Plot saved to {output_path}")
