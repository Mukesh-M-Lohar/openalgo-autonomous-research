"""
OpenAlgo Strategy Bot — Production Template
============================================
Copy this file and implement the three sections marked with TODO:
  1. CONFIGURATION  — symbols, timeframes, parameters
  2. compute_signals — your entry/exit logic (indicators go here)
  3. (optional) extra indicators / helper functions

SDK verified against OpenAlgo source (2026-06-29):
  - client.history()  → returns pd.DataFrame (timestamp as index) on success,
                         or dict {"status":"error",...} on failure.
  - client.funds()    → data key is 'availablecash'  (NOT 'available_balance')
  - client.quotes()   → data key is 'ltp'            (NOT 'last_price')
  - client.whatsapp() → use to=[list] for multi-recipient (up to 5 numbers)
  - client.placeorder() → always returns dict, check .get("status")=="success"
"""

# ==============================================================================
# IMPORTS  (keep all imports at the top)
# ==============================================================================
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# ---------- Logging -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("StrategyBot")

# ---------- OpenAlgo SDK ------------------------------------------------------
try:
    from openalgo import api
except ImportError:
    logger.error("Install the SDK first:  pip install openalgo")
    sys.exit(1)


# ==============================================================================
# TODO 1 — CONFIGURATION
# Set every value here (or via environment variables for deployment).
# ==============================================================================

# --- API connection ---
API_KEY = os.getenv("OPENALGO_API_KEY", "your-api-key-here")
API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

# --- Instrument ---
SYMBOL = os.getenv("SYMBOL", "NIFTY")
EXCHANGE = os.getenv("EXCHANGE", "NSE_INDEX")
QUANTITY = int(os.getenv("QUANTITY", "1"))
PRODUCT = os.getenv("PRODUCT", "MIS")  # MIS = intraday, CNC = delivery, NRML = F&O

# --- Data ---
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "15m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "10"))
SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", "15"))  # seconds

# --- Strategy parameters ---
# TODO: add your own parameters here
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))  # SL = entry +/- ATR * mult

# --- WhatsApp alerts (optional) ---
# Leave empty to disable.  Format: E.164 digits without '+' e.g. "919876543210"
# Up to 5 numbers are supported natively by the API in a single broadcast call.
WHATSAPP_NUMBERS: list[str] = [
    n.strip()
    for n in os.getenv("WHATSAPP_NUMBERS", "919566029048,919790856795").split(",")
    if n.strip()
]
# Set True to also notify the paired device's own number (the operator)
WHATSAPP_NOTIFY_SELF = os.getenv("WHATSAPP_NOTIFY_SELF", "True").lower() == "true"

STRATEGY_NAME = os.getenv("STRATEGY_NAME", "MyStrategy_v1")
ALLOW_SHORT = os.getenv("ALLOW_SHORT", "True").lower() == "true"


# ==============================================================================
# INDICATOR HELPERS
# ==============================================================================


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP — resets each calendar day. Requires a 'timestamp' column."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["pv"] = df["close"] * df["volume"]
    return df.groupby("date")["pv"].cumsum() / df.groupby("date")["volume"].cumsum().replace(
        0, np.nan
    )


# ==============================================================================
# TODO 2 — SIGNAL LOGIC
# Replace the stub below with your own indicator-based entry / exit logic.
# Return values:
#   "BUY"   → open a long
#   "SHORT" → open a short  (only when ALLOW_SHORT=True)
#   "EXIT"  → close current position
#   None    → no action
# ==============================================================================


def compute_signals(df: pd.DataFrame, position: Optional[str]) -> Optional[str]:
    """
    Called on every cycle with the latest OHLCV DataFrame.

    Parameters
    ----------
    df       : DataFrame with columns [timestamp, open, high, low, close, volume]
               Live LTP is already injected as the last close.
    position : current position — "BUY", "SHORT", or None

    Returns
    -------
    "BUY" | "SHORT" | "EXIT" | None
    """
    # Guard — need enough bars for the slowest indicator
    min_bars = max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD) + 5
    if len(df) < min_bars:
        return None

    close = df["close"]
    ema_fast = compute_ema(close, EMA_FAST)
    ema_slow = compute_ema(close, EMA_SLOW)
    rsi = compute_rsi(close, RSI_PERIOD)

    f, s, r = ema_fast.iloc[-1], ema_slow.iloc[-1], rsi.iloc[-1]
    f_prev, s_prev = ema_fast.iloc[-2], ema_slow.iloc[-2]

    # --- TODO: replace with your actual entry / exit rules ---

    bull_cross = (f_prev <= s_prev) and (f > s)
    bear_cross = (f_prev >= s_prev) and (f < s)

    # Exits
    if position == "BUY" and (bear_cross or r > 80):
        return "EXIT"
    if position == "SHORT" and (bull_cross or r < 20):
        return "EXIT"

    # Entries
    if position is None:
        if bull_cross and 40 < r < 75:
            return "BUY"
        if ALLOW_SHORT and bear_cross and 25 < r < 60:
            return "SHORT"

    return None


# ==============================================================================
# BOT ENGINE  — no edits needed below
# ==============================================================================


class StrategyBot:
    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)
        self.position: Optional[str] = None
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.ltp: Optional[float] = None
        self.running = True
        self.stop_event = threading.Event()
        self.instrument = [{"exchange": EXCHANGE, "symbol": SYMBOL}]
        self.daily_trade_taken = False
        self.last_trade_date = None

        logger.info(
            f"[{STRATEGY_NAME}] started | {SYMBOL}:{EXCHANGE} | "
            f"qty={QUANTITY} product={PRODUCT} tf={CANDLE_TIMEFRAME}"
        )

    # ------------------------------------------------------------------ WebSocket LTP
    def _on_ltp(self, data):
        if data.get("type") == "market_data" and data.get("symbol") == SYMBOL:
            self.ltp = float(data["data"]["ltp"])

    def _ws_thread(self):
        try:
            self.client.connect()
            self.client.subscribe_ltp(self.instrument, on_data_received=self._on_ltp)
            while not self.stop_event.is_set():
                time.sleep(1)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            try:
                self.client.unsubscribe_ltp(self.instrument)
                self.client.disconnect()
            except Exception:
                pass

    # ------------------------------------------------------------------ Data fetch
    def get_historical_data(self) -> pd.DataFrame:
        try:
            end = datetime.now()
            start = end - timedelta(days=LOOKBACK_DAYS)
            result = self.client.history(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval=CANDLE_TIMEFRAME,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
            # SDK returns DataFrame with 'timestamp' as index on success.
            if isinstance(result, pd.DataFrame) and not result.empty:
                df = result.reset_index()  # promote timestamp to column
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            if isinstance(result, dict):
                logger.warning(f"History API: {result.get('message', result)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"get_historical_data error: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------ Price estimate
    def _get_price_estimate(self) -> float:
        """Best-effort price: LTP → quotes → last close."""
        price = self.ltp or 0.0
        if price <= 0:
            r = self.client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
            if isinstance(r, dict) and r.get("status") == "success":
                price = float(r.get("data", {}).get("ltp", 0.0))  # SDK key = 'ltp'
        if price <= 0:
            df = self.get_historical_data()
            if not df.empty:
                price = float(df["close"].iloc[-1])
        return price

    # ------------------------------------------------------------------ Funds check
    def check_funds(self) -> bool:
        try:
            r = self.client.funds()
            if isinstance(r, dict) and r.get("status") == "success":
                # Verified SDK key is 'availablecash' (not 'available_balance')
                available = float(r.get("data", {}).get("availablecash", 0.0))
                price = self._get_price_estimate()
                cost = price * QUANTITY
                logger.info(f"Funds | Available: {available:.2f}  Est. cost: {cost:.2f}")
                if available < cost:
                    logger.warning(f"Insufficient funds — need {cost:.2f}, have {available:.2f}")
                    return False
                return True
            msg = r.get("message", r) if isinstance(r, dict) else r
            logger.warning(f"Funds API: {msg}. Proceeding anyway.")
            return True
        except Exception as e:
            logger.error(f"check_funds error: {e}")
            return True

    # ------------------------------------------------------------------ Notifications
    def notify(self, action: str, status: str, price: float = 0.0, extra: str = ""):
        """
        Send a WhatsApp alert.

        Recipient strategy (in order of precedence):
          1. WHATSAPP_NUMBERS list  → broadcast to up to 5 numbers via to=[...]
          2. WHATSAPP_NOTIFY_SELF=True → send to operator's own paired number
          3. Both empty             → notifications silently disabled
        """
        msg = (
            f"[{STRATEGY_NAME}]\n"
            f"Action : {action}\n"
            f"Status : {status}\n"
            f"Symbol : {SYMBOL} ({EXCHANGE})\n"
            f"Price  : {price:.2f}\n"
            f"Qty    : {QUANTITY} | Product: {PRODUCT}\n"
            + (f"Note   : {extra}\n" if extra else "")
            + f"Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            if WHATSAPP_NUMBERS:
                # Single API call broadcasts to all numbers (server caps at 5)
                r = self.client.whatsapp(msg, to=WHATSAPP_NUMBERS[:5])
            elif WHATSAPP_NOTIFY_SELF:
                # Sends to the paired device's own number
                r = self.client.whatsapp(msg)
            else:
                return  # disabled

            if isinstance(r, dict) and r.get("status") != "success":
                logger.warning(f"WhatsApp: {r.get('message', r)}")
            else:
                logger.info("WhatsApp notification sent.")
        except Exception as e:
            logger.warning(f"WhatsApp notification failed: {e}")

    # ------------------------------------------------------------------ Orders
    def place_order(self, action: str) -> bool:
        if not self.check_funds():
            logger.warning("Order aborted — insufficient funds.")
            return False

        resp = self.client.placeorder(
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=action,
            quantity=QUANTITY,
            price_type="MARKET",
            product=PRODUCT,
        )
        if resp.get("status") == "success":
            self.position = action
            self.entry_price = self._get_price_estimate()

            df = self.get_historical_data()
            if not df.empty:
                atr = compute_atr(df, ATR_PERIOD).iloc[-1]
                self.sl_price = (
                    self.entry_price - atr * ATR_SL_MULT
                    if action == "BUY"
                    else self.entry_price + atr * ATR_SL_MULT
                )

            self.daily_trade_taken = True
            self.last_trade_date = datetime.now().date()
            logger.info(f"Entry {action} @ {self.entry_price:.2f} | SL: {self.sl_price:.2f}")
            self.notify(action, "success", self.entry_price)
            return True

        logger.error(f"Order failed: {resp}")
        self.notify(action, f"FAILED — {resp.get('message', resp)}")
        return False

    def close_position(self, reason: str = "signal"):
        if self.position is None:
            return
        exit_action = "SELL" if self.position == "BUY" else "BUY"
        resp = self.client.placeorder(
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=exit_action,
            quantity=QUANTITY,
            price_type="MARKET",
            product=PRODUCT,
        )
        if resp.get("status") == "success":
            exit_price = self._get_price_estimate()
            logger.info(f"Exit {exit_action} @ {exit_price:.2f} | Reason: {reason}")
            self.notify(exit_action, f"success — exit: {reason}", exit_price)
            self.position = None
            self.entry_price = 0.0
            self.sl_price = 0.0
        else:
            logger.error(f"Exit order failed: {resp}")
            self.notify(exit_action, f"EXIT FAILED — {resp.get('message', resp)}")

    # ------------------------------------------------------------------ Main loop
    def check_signals(self):
        now = datetime.now()
        current_date = now.date()

        if self.last_trade_date != current_date:
            self.daily_trade_taken = False

        is_exit_time = now.hour == 15 and now.minute >= 15

        # EOD square-off for MIS
        if is_exit_time and self.position is not None and PRODUCT == "MIS":
            logger.info("EOD square-off (15:15).")
            self.close_position("EOD")
            return

        # Stop-loss from live LTP
        if self.position and self.ltp and self.sl_price > 0:
            sl_hit = (self.position == "BUY" and self.ltp <= self.sl_price) or (
                self.position == "SHORT" and self.ltp >= self.sl_price
            )
            if sl_hit:
                logger.info(f"SL hit — LTP: {self.ltp:.2f}  SL: {self.sl_price:.2f}")
                self.close_position("SL")
                return

        if self.daily_trade_taken or is_exit_time:
            return

        df = self.get_historical_data()
        if df.empty:
            return
        if self.ltp is not None:
            df.loc[df.index[-1], "close"] = self.ltp

        signal = compute_signals(df, self.position)

        if signal == "EXIT":
            self.close_position("signal")
        elif signal == "BUY" and self.position is None:
            logger.info("Signal: BUY")
            self.place_order("BUY")
        elif signal == "SHORT" and self.position is None and ALLOW_SHORT:
            logger.info("Signal: SHORT")
            self.place_order("SHORT")

    def run(self):
        ws = threading.Thread(target=self._ws_thread, daemon=True)
        ws.start()
        time.sleep(2)

        logger.info("Bot running. Ctrl+C to stop.")
        try:
            while self.running:
                self.check_signals()
                time.sleep(SIGNAL_CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Bot stopped.")
        finally:
            self.stop_event.set()
            self.running = False


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    StrategyBot().run()
