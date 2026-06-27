import os

import httpx
import pandas as pd

host = "http://127.0.0.1:5000"
api_key = os.environ.get("OPENALGO_API_KEY", "test")
symbol = "COPPER31JUL26FUT"
exchange = "MCX"

payload = {
    "apikey": api_key,
    "symbol": symbol,
    "exchange": exchange,
    "interval": "5m",
    "start_date": "2026-06-20",
    "end_date": "2026-06-26",
    "source": "db",
}

url = f"{host}/api/v1/history"
response = httpx.post(url, json=payload)
if response.status_code == 200:
    data = response.json()
    if data.get("status") == "success":
        df = pd.DataFrame(data["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        # Group by date and get min/max times
        df["date"] = df["timestamp"].dt.date
        df["time"] = df["timestamp"].dt.time
        summary = df.groupby("date").agg(
            min_time=("time", "min"), max_time=("time", "max"), count=("time", "count")
        )
        print(summary)
