# ruff: noqa: E402
import os
import sys
from pathlib import Path

import pandas as pd
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

# Let's import strategy
sys.path.insert(0, str(ROOT / ".autoresearch"))
import strategy as strat

close = df["close"]
high = df["high"]
low = df["low"]
volume = df["volume"]

ema_fast = strat._ema(close, strat.EMA_FAST)
ema_slow = strat._ema(close, strat.EMA_SLOW)
ema_trend = strat._ema(close, strat.EMA_TREND)
rsi = strat._rsi(close, strat.RSI_PERIOD)
macd_line, macd_sig, macd_hist = strat._macd(
    close, strat.MACD_FAST, strat.MACD_SLOW, strat.MACD_SIGNAL
)
atr = strat._atr(df, strat.ATR_PERIOD)
adx = strat._adx(df, 14)

has_vol = "volume" in df.columns and not (df["volume"] == 0).all() and not df["volume"].isna().all()
vol_surge = pd.Series(True, index=df.index)
if has_vol:
    vol_avg = volume.rolling(20).mean()
    vol_surge = volume > (strat.VOL_MULT * vol_avg)

ema_bull = (ema_fast > ema_slow) & (close > ema_trend)
ema_cross = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
macd_turn = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
rsi_zone = (rsi > strat.RSI_ENTRY_LO) & (rsi < strat.RSI_ENTRY_HI)
adx_ok = adx > strat.ADX_MIN

print(f"Total bars: {len(df)}")
print(f"ema_bull count: {ema_bull.sum()}")
print(f"ema_cross count: {ema_cross.sum()}")
print(f"macd_turn count: {macd_turn.sum()}")
print(f"rsi_zone count: {rsi_zone.sum()}")
print(f"vol_surge count: {vol_surge.sum()}")
print(f"adx_ok count: {adx_ok.sum()}")

signal_trigger = ema_cross | macd_turn
print(f"signal_trigger count: {signal_trigger.sum()}")

entry = signal_trigger & ema_bull & rsi_zone & vol_surge & adx_ok
print(f"entry count: {entry.sum()}")
