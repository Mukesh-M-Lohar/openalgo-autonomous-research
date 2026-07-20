"""
OpenAlgo Strategy Bot — Generic Supertrend + RSI Strategy
=========================================================
This production bot replicates the Pine Script strategy:
- Supertrend calculation with dual-mode ATR support (Wilder vs SMA).
- RSI and RSI SMA indicators.
- Entry signals checked on the previous closed bar:
  - buyCondition = trend == 1 and rsi < rsiSma and rsiCooldown and rsi < rsiOverbought
  - sellCondition = trend == -1 and rsi > rsiSma and rsiCooldown and rsi > rsiOversold
- Exit conditions checked on the current bar:
  - longExit = trend == -1 or rsi >= rsiOverbought
  - shortExit = trend == 1 or rsi <= rsiOversold
- Fixed Stop Loss (fixedSL) offset logic.
- Configurable symbol, exchange, and timeframes.
- All notification timestamps are printed in Indian Standard Time (IST).
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Logging -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SupertrendRSIMCXBot")

# ---------- OpenAlgo SDK ------------------------------------------------------
try:
    from openalgo import api
except ImportError:
    logger.error("Install the SDK first:  pip install openalgo")
    sys.exit(1)

# ==============================================================================
# 1. CONFIGURATION (Read from environment variables with defaults)
# ==============================================================================

# --- API Connection ---
API_KEY = os.getenv("OPENALGO_API_KEY", "your-api-key-here")
API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

# --- Instrument & Execution ---
SYMBOL = os.getenv("SYMBOL", "SILVER")
EXCHANGE = os.getenv("EXCHANGE", "MCX")
QUANTITY = int(os.getenv("QUANTITY", "1"))
PRODUCT = os.getenv("PRODUCT", "MIS")  # MIS = Intraday, CNC/NRML = Carry Forward
STRATEGY_NAME = os.getenv("STRATEGY_NAME", "SupertrendRSI_MCX_v1")
ALLOW_SHORT = os.getenv("ALLOW_SHORT", "True").lower() == "true"

# --- Data Settings ---
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "15m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "15"))
SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", "15"))  # seconds

# --- Supertrend Parameters ---
ST_ATR_LEN = int(os.getenv("ST_ATR_LEN", "10"))
ST_MULT = float(os.getenv("ST_MULT", "3.5"))
CHANGE_ATR = os.getenv("CHANGE_ATR", "True").lower() == "true"

# --- RSI Parameters ---
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_SMA_LEN = int(os.getenv("RSI_SMA_LEN", "14"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70.0"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30.0"))

# --- Stop Loss Parameter ---
FIXED_SL = float(os.getenv("FIXED_SL", "800.0"))  # stop loss in price points/Rs.

# --- Session Control (Optional) ---
USE_SESSION = os.getenv("USE_SESSION", "False").lower() == "true"
SESSION_START = os.getenv("SESSION_START", "09:00")
SESSION_END = os.getenv("SESSION_END", "23:30")

# --- WhatsApp / Telegram alerts ---
WHATSAPP_PHONES: list[str] = [
    n.strip() for n in os.getenv("WHATSAPP_PHONES", "").split(",") if n.strip()
]
WHATSAPP_NOTIFY_SELF = os.getenv("WHATSAPP_NOTIFY_SELF", "True").lower() == "true"

# ==============================================================================
# TIMEZONE & TIME HELPERS (Asia/Kolkata timezone)
# ==============================================================================
IST = timezone(timedelta(hours=5, minutes=30))


def get_now_ist() -> datetime:
    """Returns the current date and time localized in Indian Standard Time (IST)."""
    return datetime.now(timezone.utc).astimezone(IST)


def is_in_session() -> bool:
    """Checks if the current Indian time is within standard trading session."""
    if not USE_SESSION:
        return True
    try:
        now_ist = get_now_ist()
        current_time = now_ist.time()
        start_h, start_m = map(int, SESSION_START.split(":"))
        end_h, end_m = map(int, SESSION_END.split(":"))
        return dt_time(start_h, start_m) <= current_time < dt_time(end_h, end_m)
    except Exception as e:
        logger.error(f"Error parsing session times: {e}")
        return True


# ==============================================================================
# TECHNICAL INDICATOR MATHEMATICS (Identical to Pine Script Formulas)
# ==============================================================================


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    g = d.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    losses = (-d.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + g / losses.replace(0, np.nan))


def compute_atr(df: pd.DataFrame, period: int = 14, change_atr: bool = True) -> pd.Series:
    hl = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    if change_atr:
        # Wilder's RMA/EMA smoothing
        return tr.ewm(com=period - 1, min_periods=period).mean()
    else:
        # Simple Moving Average smoothing
        return tr.rolling(period).mean()


def compute_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.5, change_atr: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Computes Supertrend.
    Returns (supertrend_value_series, direction_series)
    direction: 1 for bullish, -1 for bearish (matching Pine's trend == 1 / -1)
    """
    hl2 = (df["high"] + df["low"]) / 2
    atr = compute_atr(df, period, change_atr)

    up_raw = hl2 - multiplier * atr
    dn_raw = hl2 + multiplier * atr

    up = pd.Series(0.0, index=df.index)
    dn = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)

    for i in range(len(df)):
        if i == 0:
            up.iloc[i] = up_raw.iloc[i]
            dn.iloc[i] = dn_raw.iloc[i]
            trend.iloc[i] = 1
            continue

        close_prev = df["close"].iloc[i - 1]
        up_prev = up.iloc[i - 1]
        dn_prev = dn.iloc[i - 1]
        trend_prev = trend.iloc[i - 1]

        # up band update
        if close_prev > up_prev:
            up.iloc[i] = max(up_raw.iloc[i], up_prev)
        else:
            up.iloc[i] = up_raw.iloc[i]

        # dn band update
        if close_prev < dn_prev:
            dn.iloc[i] = min(dn_raw.iloc[i], dn_prev)
        else:
            dn.iloc[i] = dn_raw.iloc[i]

        # trend update
        if trend_prev == -1 and df["close"].iloc[i] > dn.iloc[i - 1]:
            trend.iloc[i] = 1
        elif trend_prev == 1 and df["close"].iloc[i] < up.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend_prev

    active_st = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        active_st.iloc[i] = up.iloc[i] if trend.iloc[i] == 1 else dn.iloc[i]

    return active_st, trend


# ==============================================================================
# SIGNAL CALCULATIONS
# ==============================================================================


def compute_signals(df: pd.DataFrame) -> dict:
    """Computes technical indicators and entry/exit signals from OHLCV data."""
    min_bars = max(ST_ATR_LEN, RSI_LEN, RSI_SMA_LEN) + 5
    if len(df) < min_bars:
        return {
            "buy": False,
            "sell": False,
            "long_exit": False,
            "short_exit": False,
            "indicators": {},
        }

    # Standard indicators
    supertrend, trend = compute_supertrend(df, ST_ATR_LEN, ST_MULT, CHANGE_ATR)
    rsi = compute_rsi(df["close"], RSI_LEN)
    rsi_sma = compute_sma(rsi, RSI_SMA_LEN)

    # previous closed bar calculations [-2] for entry signals
    trend_prev = trend.iloc[-2]
    rsi_prev = rsi.iloc[-2]
    rsi_sma_prev = rsi_sma.iloc[-2]
    rsi_prev2 = rsi.iloc[-3]

    rsi_cooldown_prev = rsi_prev < rsi_prev2

    buy_condition_prev = (
        trend_prev == 1
        and rsi_prev < rsi_sma_prev
        and rsi_cooldown_prev
        and rsi_prev < RSI_OVERBOUGHT
    )

    sell_condition_prev = (
        trend_prev == -1
        and rsi_prev > rsi_sma_prev
        and rsi_cooldown_prev
        and rsi_prev > RSI_OVERSOLD
    )

    # current bar calculations [-1] for exit signals
    trend_curr = trend.iloc[-1]
    rsi_curr = rsi.iloc[-1]

    long_exit = (trend_curr == -1) or (rsi_curr >= RSI_OVERBOUGHT)
    short_exit = (trend_curr == 1) or (rsi_curr <= RSI_OVERSOLD)

    indicators = {
        "rsi": rsi_curr,
        "rsi_sma": rsi_sma.iloc[-1],
        "st_value": supertrend.iloc[-1],
        "st_trend": trend_curr,
    }

    return {
        "buy": bool(buy_condition_prev),
        "sell": bool(sell_condition_prev),
        "long_exit": bool(long_exit),
        "short_exit": bool(short_exit),
        "indicators": indicators,
    }


# ==============================================================================
# STRATEGY BOT ENGINE
# ==============================================================================


class MCXSupertrendRSIBot:
    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)

        # Position states
        self.position: Optional[str] = None  # "BUY", "SELL", or None
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.ltp: Optional[float] = None
        self.running = True
        self.stop_event = threading.Event()
        self.instrument = [{"exchange": EXCHANGE, "symbol": SYMBOL}]

        self.daily_trade_taken = False
        self.last_trade_date = None

        # Historical Caching
        self._cache_df = None
        self._cache_last_fetched = 0.0

        logger.info(
            f"[{STRATEGY_NAME}] Initialized. Trading Symbol: {SYMBOL} on {EXCHANGE} "
            f"| Quantity: {QUANTITY} | Product: {PRODUCT}"
        )

    # ---------------------------------------------------------------- WebSocket Feed
    def _on_ltp_received(self, data):
        if data.get("type") == "market_data" and data.get("symbol") == SYMBOL:
            self.ltp = float(data["data"]["ltp"])

    def _websocket_worker(self):
        try:
            self.client.connect()
            self.client.subscribe_ltp(self.instrument, on_data_received=self._on_ltp_received)
            while not self.stop_event.is_set():
                time.sleep(1)
        except Exception as e:
            logger.error(f"WebSocket execution error: {e}")
        finally:
            try:
                self.client.unsubscribe_ltp(self.instrument)
                self.client.disconnect()
            except Exception:
                pass

    # ---------------------------------------------------------------- Data Fetching
    def get_historical_data(self) -> pd.DataFrame:
        """Fetches primary timeframe data."""
        try:
            cache_expiry = 30 if "1m" in CANDLE_TIMEFRAME else 300
            current_time = time.time()
            if self._cache_df is not None and (
                current_time - self._cache_last_fetched < cache_expiry
            ):
                return self._cache_df.copy()

            end = datetime.now()
            start = end - timedelta(days=LOOKBACK_DAYS)
            source = "db" if EXCHANGE.endswith("_INDEX") else "api"

            result = self.client.history(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval=CANDLE_TIMEFRAME,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                source=source,
            )

            if isinstance(result, pd.DataFrame) and not result.empty:
                df = result.reset_index()
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                self._cache_df = df.copy()
                self._cache_last_fetched = current_time
                return df

            if isinstance(result, dict):
                logger.warning(f"History API response warning: {result.get('message', result)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    # ---------------------------------------------------------------- Price Estimation & Risk
    def _get_price_estimate(self) -> float:
        price = self.ltp or 0.0
        if price <= 0:
            r = self.client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
            if isinstance(r, dict) and r.get("status") == "success":
                price = float(r.get("data", {}).get("ltp", 0.0))
        if price <= 0:
            df = self.get_historical_data()
            if not df.empty:
                price = float(df["close"].iloc[-1])
        return price

    def check_funds(self) -> bool:
        try:
            r = self.client.funds()
            if isinstance(r, dict) and r.get("status") == "success":
                available = float(r.get("data", {}).get("availablecash", 0.0))
                price = self._get_price_estimate()
                cost = price * QUANTITY
                logger.info(
                    f"Funds Check | Cash Available: {available:.2f} | Est. Order Cost: {cost:.2f}"
                )
                if available < cost:
                    logger.warning(
                        f"Insufficient funds: needs {cost:.2f}, only have {available:.2f}"
                    )
                    return False
                return True
            logger.warning("Funds API did not return success. Proceeding order execution.")
            return True
        except Exception as e:
            logger.error(f"Error checking funds: {e}")
            return True

    # ---------------------------------------------------------------- Notifications
    def notify(self, action: str, status: str, price: float = 0.0, extra: str = ""):
        """Sends WhatsApp notifications with timestamps in Indian Standard Time (IST)."""
        now_ist = get_now_ist()
        msg = (
            f"[{STRATEGY_NAME}]\n"
            f"Action : {action}\n"
            f"Status : {status}\n"
            f"Symbol : {SYMBOL} ({EXCHANGE})\n"
            f"Price  : {price:.2f}\n"
            f"Qty    : {QUANTITY} | Product: {PRODUCT}\n"
            + (f"Note   : {extra}\n" if extra else "")
            + f"Time   : {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}"
        )
        try:
            if WHATSAPP_PHONES:
                r = self.client.whatsapp(msg, to=WHATSAPP_PHONES[:5])
            elif WHATSAPP_NOTIFY_SELF:
                r = self.client.whatsapp(msg)
            else:
                return

            if isinstance(r, dict) and r.get("status") != "success":
                logger.warning(f"WhatsApp Notification API response warning: {r.get('message', r)}")
            else:
                logger.info(
                    f"WhatsApp notification sent successfully. Message Time: {now_ist.strftime('%H:%M:%S IST')}"
                )
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")

    # ---------------------------------------------------------------- Order Routing
    def place_entry_order(self, action: str):
        if not self.check_funds():
            logger.warning("Order aborted — insufficient funds.")
            return

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

            # Setup fixed stop loss price
            if action == "BUY":
                self.sl_price = self.entry_price - FIXED_SL
            else:
                self.sl_price = self.entry_price + FIXED_SL

            self.daily_trade_taken = True
            self.last_trade_date = get_now_ist().date()

            logger.info(
                f"Entry {action} Order Successful @ {self.entry_price:.2f} | Stop Loss: {self.sl_price:.2f}"
            )
            self.notify(action, "ENTRY SUCCESS", self.entry_price, f"SL: {self.sl_price:.2f}")
        else:
            logger.error(f"Entry order failed: {resp}")
            self.notify(action, f"ENTRY FAILED: {resp.get('message', resp)}", 0.0)

    def place_exit_order(self, reason: str):
        exit_action = "SELL" if self.position == "BUY" else "BUY"
        logger.info(f"Placing exit order [{reason}] via {exit_action} for {QUANTITY} shares...")

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
            logger.info(f"Exit Successful @ {exit_price:.2f}. Reason: {reason}")
            self.notify(exit_action, f"EXIT SUCCESS ({reason})", exit_price)

            # Reset values
            self.position = None
            self.entry_price = 0.0
            self.sl_price = 0.0
        else:
            logger.error(f"Exit order failed: {resp}")
            self.notify(exit_action, f"EXIT FAILED ({reason}): {resp.get('message', resp)}", 0.0)

    # ---------------------------------------------------------------- Signal Checking & Monitoring
    def check_signals(self):
        now_ist = get_now_ist()
        current_date = now_ist.date()

        if self.last_trade_date != current_date:
            self.daily_trade_taken = False

        in_sess = is_in_session()

        if self.position is not None:
            # Monitor active trades using current LTP
            current_price = self.ltp if self.ltp is not None else 0.0
            if current_price <= 0.0:
                return

            exit_triggered = False
            exit_reason = ""

            # 1. Stop Loss check
            if self.position == "BUY" and current_price <= self.sl_price:
                logger.info(f"Stop Loss hit ({current_price:.2f} <= {self.sl_price:.2f})")
                exit_triggered = True
                exit_reason = "SL"
            elif self.position == "SELL" and current_price >= self.sl_price:
                logger.info(f"Stop Loss hit ({current_price:.2f} >= {self.sl_price:.2f})")
                exit_triggered = True
                exit_reason = "SL"

            # 2. Exit condition check
            if not exit_triggered:
                df = self.get_historical_data()
                if not df.empty:
                    if self.ltp is not None:
                        df.loc[df.index[-1], "close"] = self.ltp

                    signal_res = compute_signals(df)
                    if self.position == "BUY" and signal_res["long_exit"]:
                        logger.info("Long Exit condition met.")
                        exit_triggered = True
                        exit_reason = "EXIT SIGNAL"
                    elif self.position == "SELL" and signal_res["short_exit"]:
                        logger.info("Short Exit condition met.")
                        exit_triggered = True
                        exit_reason = "EXIT SIGNAL"

            if exit_triggered:
                self.place_exit_order(exit_reason)

        else:
            # Check for entries only when in session and daily trade limit isn't reached
            if in_sess and not self.daily_trade_taken:
                df = self.get_historical_data()
                if df.empty:
                    return

                # Update last row with real-time LTP
                if self.ltp is not None:
                    df.loc[df.index[-1], "close"] = self.ltp
                    if self.ltp > df.loc[df.index[-1], "high"]:
                        df.loc[df.index[-1], "high"] = self.ltp
                    if self.ltp < df.loc[df.index[-1], "low"]:
                        df.loc[df.index[-1], "low"] = self.ltp

                signal_res = compute_signals(df)

                if signal_res["buy"]:
                    logger.info("Signal generated: BUY")
                    self.place_entry_order("BUY")
                elif signal_res["sell"] and ALLOW_SHORT:
                    logger.info("Signal generated: SELL")
                    self.place_entry_order("SELL")

    def run(self):
        # Start background WebSocket thread
        ws_thread = threading.Thread(target=self._websocket_worker, daemon=True)
        ws_thread.start()
        time.sleep(2)  # Allow WebSocket connection to establish

        logger.info("Bot is active and running. Waiting for signals...")
        try:
            while self.running:
                self.check_signals()
                time.sleep(SIGNAL_CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt detected. Stopping bot...")
        finally:
            self.stop_event.set()
            self.running = False
            if self.position is not None:
                logger.info("Exiting remaining positions on shutdown.")
                self.place_exit_order("Bot Shutdown")


if __name__ == "__main__":
    bot = MCXSupertrendRSIBot()
    bot.run()
