# ruff: noqa: E402
import os
import sys
from pathlib import Path

import strategy as strat
from dotenv import load_dotenv
from prepare import _simulate_trades

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

# Let's import strategy
sys.path.insert(0, str(ROOT / ".autoresearch"))

sys.path.insert(0, str(ROOT / ".autoresearch"))

signals_df = strat.generate_signals(df.copy())
print("Signals df columns:", signals_df.columns)
print("Entry count:", signals_df["entry"].sum())
print("Exit count:", signals_df["exit"].sum())

trades, equity_curve = _simulate_trades(
    df,
    signals_df,
    atr_series=None,
    atr_mult_sl=0.0,
    stop_loss_pct=None,
    take_profit_pct=None,
    trailing_stop_pct=None,
    max_hold_bars=None,
)

print("Simulated trades length:", len(trades))
if len(trades) > 0:
    print("First trade:", trades[0])
