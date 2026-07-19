"""
Standalone Bot: Hurst Exponent Adaptive Supertrend [QuantAlgo]

This bot connects to the OpenAlgo SDK, subscribes to real-time data, and executes
trades based on the Hurst Adaptive Supertrend indicator.
"""

import os
import time
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from openalgo import api

# --- Configuration ---
API_KEY = os.getenv("OPENALGO_API_KEY", "openalgo-apikey")
API_HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_URL = os.getenv("OPENALGO_WS_URL", "ws://127.0.0.1:8765")

SYMBOL = os.getenv("SYMBOL", "RELIANCE")
EXCHANGE = os.getenv("EXCHANGE", "NSE")
QUANTITY = int(os.getenv("QUANTITY", "1"))
PRODUCT = os.getenv("PRODUCT", "MIS")
TIMEFRAME = os.getenv("TIMEFRAME", "5m")

# Hurst Supertrend Parameters (Default Preset)
H_PERIOD = 60
H_LAG = 10
KF_GAIN = 0.20
ATR_LEN = 10
ATR_BASE = 1.5
ATR_HSCALE = 3.0


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def compute_hurst_supertrend(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns (supertrend, direction)"""
    close = df["close"]
    
    var1 = (close - close.shift(1)).rolling(H_PERIOD).var()
    varq = (close - close.shift(H_LAG)).rolling(H_PERIOD).var()
    
    H_raw = np.log(varq / np.maximum(var1, 1e-10)) / (2.0 * np.log(H_LAG))
    H = np.maximum(0.0, np.minimum(H_raw, 1.0))
    safeH = H.fillna(0.5)

    adaptive_gain = np.maximum(np.minimum(KF_GAIN * (0.5 + safeH), 0.99), 0.01)
    
    kf = np.zeros(len(close))
    if len(close) > 0:
        kf[0] = close.iloc[0] if not np.isnan(close.iloc[0]) else 0.0
    for i in range(1, len(close)):
        if np.isnan(kf[i-1]):
            kf[i] = close.iloc[i] if not np.isnan(close.iloc[i]) else 0.0
        else:
            val = close.iloc[i] if not np.isnan(close.iloc[i]) else kf[i-1]
            kf[i] = kf[i-1] + adaptive_gain.iloc[i] * (val - kf[i-1])
    kf_series = pd.Series(kf, index=close.index)

    atr = _compute_atr(df, ATR_LEN)
    h_mult = ATR_BASE + ATR_HSCALE * (1.0 - safeH)
    band = atr * h_mult

    upBand = kf_series - band
    dnBand = kf_series + band

    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        prev_dir = direction.iloc[i - 1]
        prev_up = upBand.iloc[i - 1] if not np.isnan(upBand.iloc[i - 1]) else close.iloc[i]
        prev_dn = dnBand.iloc[i - 1] if not np.isnan(dnBand.iloc[i - 1]) else close.iloc[i]

        curr_up = kf_series.iloc[i] - band.iloc[i]
        curr_dn = kf_series.iloc[i] + band.iloc[i]

        if prev_dir == 1:
            upBand.iloc[i] = max(curr_up, prev_up)
        else:
            upBand.iloc[i] = curr_up

        if prev_dir == -1:
            dnBand.iloc[i] = min(curr_dn, prev_dn)
        else:
            dnBand.iloc[i] = curr_dn

        if kf_series.iloc[i] > prev_dn:
            direction.iloc[i] = 1
        elif kf_series.iloc[i] < prev_up:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_dir

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = upBand.iloc[i]
        else:
            supertrend.iloc[i] = dnBand.iloc[i]

    return supertrend, direction


class HurstBot:
    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)
        self.ltp = 0.0
        self.position = 0 # 1 for Long, -1 for Short, 0 for None
        self.stop_event = threading.Event()

    def on_ltp(self, data):
        if data.get("type") == "market_data" and data.get("symbol") == SYMBOL:
            self.ltp = float(data["data"]["ltp"])

    def fetch_history(self) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10) # Enough for lookbacks
        data = self.client.history(
            symbol=SYMBOL,
            exchange=EXCHANGE,
            interval=TIMEFRAME,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and data.get("status") == "success":
            return pd.DataFrame(data.get("data", []))
        return pd.DataFrame()

    def place_order(self, action: str):
        print(f"[BOT] Placing {action} order for {SYMBOL}...")
        resp = self.client.placeorder(
            strategy="HurstBot",
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=action,
            quantity=QUANTITY,
            price_type="MARKET",
            product=PRODUCT,
        )
        if resp.get("status") == "success":
            print(f"[BOT] {action} Order successful. Order ID: {resp.get('orderid')}")
        else:
            print(f"[BOT] {action} Order failed: {resp}")

    def run(self):
        print(f"[BOT] Connecting to OpenAlgo for {SYMBOL}...")
        ws_thread = threading.Thread(target=self.start_ws, daemon=True)
        ws_thread.start()
        time.sleep(2)

        print("[BOT] Starting main execution loop...")
        try:
            while not self.stop_event.is_set():
                df = self.fetch_history()
                if not df.empty and len(df) > H_PERIOD + H_LAG:
                    st, direction = compute_hurst_supertrend(df)
                    
                    curr_dir = direction.iloc[-1]
                    prev_dir = direction.iloc[-2]
                    
                    # Check for trend flips
                    if curr_dir == 1 and prev_dir == -1:
                        print(f"[SIGNAL] BULLISH trend confirmed for {SYMBOL} at {self.ltp}")
                        if self.position == -1:
                            # Close short, then open long
                            self.place_order("BUY")  # close short
                            self.place_order("BUY")  # open long
                        elif self.position == 0:
                            self.place_order("BUY")
                        self.position = 1

                    elif curr_dir == -1 and prev_dir == 1:
                        print(f"[SIGNAL] BEARISH trend confirmed for {SYMBOL} at {self.ltp}")
                        if self.position == 1:
                            # Close long, then open short
                            self.place_order("SELL") # close long
                            self.place_order("SELL") # open short
                        elif self.position == 0:
                            self.place_order("SELL")
                        self.position = -1

                time.sleep(60) # check every minute
        except KeyboardInterrupt:
            print("[BOT] Shutting down...")
        finally:
            self.stop_event.set()

    def start_ws(self):
        try:
            self.client.connect()
            self.client.subscribe_ltp([{"symbol": SYMBOL, "exchange": EXCHANGE}], on_data_received=self.on_ltp)
            while not self.stop_event.is_set():
                time.sleep(1)
        except Exception as e:
            print(f"[WS ERROR] {e}")


if __name__ == "__main__":
    bot = HurstBot()
    bot.run()
