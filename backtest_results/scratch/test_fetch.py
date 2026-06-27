import os

import httpx

host = "http://127.0.0.1:5000"
api_key = os.environ.get("OPENALGO_API_KEY", "test")

payload = {
    "apikey": api_key,
    "symbol": "COPPER31JUL26FUT",
    "exchange": "MCX",
    "interval": "5m",
    "start_date": "2026-06-01",
    "end_date": "2026-06-27",
    "source": "db",
}

url = f"{host}/api/v1/history"
try:
    response = httpx.post(url, json=payload, timeout=10.0)
    print("Status code:", response.status_code)
    try:
        data = response.json()
        print("Response status:", data.get("status"))
        if data.get("status") == "success":
            records = data.get("data", [])
            print(f"Successfully fetched {len(records)} records.")
            if records:
                print("First record:", records[0])
                print("Last record:", records[-1])
        else:
            print("Response error:", data.get("message"))
    except Exception as e:
        print("JSON parse error:", e)
        print("Raw response:", response.text[:500])
except Exception as e:
    print("HTTP request error:", e)
