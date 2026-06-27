import os

import httpx
import pandas as pd

API_KEY = "b45feb0a6973ed00fe86d25ace49d4da8dfe8d0a78c334455d46254ded28a26d"
API_HOST = "http://127.0.0.1:5000"
DATA_DIR = "/root/openalgo-autonomous-research/autoresearch/data"

os.makedirs(DATA_DIR, exist_ok=True)

SYMBOLS = [
    ("PROTEAN", "NSE"),
    ("ZEEL", "NSE"),
    ("BAHETI-SM", "NSE"),
    ("CDSL", "NSE"),
    ("ANGELONE", "NSE"),
    ("SCI", "NSE"),
    ("ACUTAAS", "NSE"),
    ("SAMMAANCAP", "NSE"),
    ("CLEAN", "NSE"),
    ("COPPER31JUL26FUT", "MCX"),
    ("COPPER31AUG26FUT", "MCX"),
    ("MCX", "NSE"),
    ("SBIN", "NSE"),
    ("BSE", "NSE"),
    ("NIFTY", "NSE_INDEX"),
    ("BANKNIFTY", "NSE_INDEX"),
]


def download_data():
    print("Preparing 15m historical data shards...")
    for sym, exch in SYMBOLS:
        dest_path = os.path.join(DATA_DIR, f"{sym}_{exch}_15m.csv")
        payload = {
            "apikey": API_KEY,
            "symbol": sym,
            "exchange": exch,
            "interval": "15m",
            "start_date": "2025-01-01",
            "end_date": "2026-06-27",
            "source": "api",
        }
        try:
            response = httpx.post(f"{API_HOST}/api/v1/history", json=payload, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    df = pd.DataFrame(data["data"])
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                        df = df.set_index("timestamp").sort_index()
                        df.to_csv(dest_path)
                        print(f"  Saved {sym}:{exch} ({len(df)} rows)")
                        continue
            print(f"  Failed downloading {sym}:{exch}")
        except Exception as e:
            print(f"  Error downloading {sym}:{exch}: {e}")


if __name__ == "__main__":
    download_data()
