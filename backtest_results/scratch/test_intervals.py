import os

import httpx
import pandas as pd

host = "http://127.0.0.1:5000"
api_key = os.environ.get("OPENALGO_API_KEY", "test")
symbol = "COPPER31JUL26FUT"
exchange = "MCX"
intervals = ["5m", "15m", "1h"]

for interval in intervals:
    payload = {
        "apikey": api_key,
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "start_date": "2025-01-01",
        "end_date": "2026-06-27",
        "source": "db",
    }
    url = f"{host}/api/v1/history"
    try:
        response = httpx.post(url, json=payload, timeout=15.0)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                records = data.get("data", [])
                if records:
                    df = pd.DataFrame(records)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    print(
                        f"Interval {interval}: fetched {len(records)} records. Range: {df['timestamp'].min()} to {df['timestamp'].max()}"
                    )
                else:
                    print(f"Interval {interval}: fetched 0 records.")
            else:
                print(f"Interval {interval} failed: {data.get('message')}")
        else:
            print(f"Interval {interval} HTTP status: {response.status_code}")
    except Exception as e:
        print(f"Interval {interval} error: {e}")
