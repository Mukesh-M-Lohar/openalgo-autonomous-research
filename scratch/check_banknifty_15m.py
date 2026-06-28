# ruff: noqa: E402
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from quant_engine.config import OpenAlgoConfig
from quant_engine.data.client import OpenAlgoClient

cfg = OpenAlgoConfig(
    host=os.environ.get("OPENALGO_HOST", "http://127.0.0.1:5000"),
    api_key=os.environ.get("OPENALGO_API_KEY", ""),
    source=os.environ.get("OPENALGO_SOURCE", "db"),
)

with OpenAlgoClient(cfg, cache_dir=ROOT / "data" / "cache") as client:
    df = client.fetch_history("BANKNIFTY", "NSE_INDEX", "15m", "2024-01-01", "2026-06-25")

print(f"BANKNIFTY 15m dataframe length: {len(df)}")
if len(df) > 0:
    print(df.head())
    print(df.tail())
    print("Volume values info:")
    print(df["volume"].describe())
