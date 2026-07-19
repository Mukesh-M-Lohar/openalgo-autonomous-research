"""
OpenAlgo Strategy Bot — Supertrend RSI Pullback Strategy [PRO FINAL]
===================================================================
This production bot replicates the Pine Script strategy:
- Primary indicator signals: Supertrend trend alignment, RSI pullback/zone filters, ADX filter.
- Higher Time Frame (HTF) trend filter validation.
- Intraday Session control (09:20 - 15:15 IST) and EOD square-off.
- Precise multi-target exits (TP1 & TP2) and trailing stop loss to breakeven after TP1.
- All notification timestamps are printed in Indian Standard Time (IST).
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Logging -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SupertrendRSIPullbackBot")

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
SYMBOL = os.getenv("SYMBOL", "NHPC")
EXCHANGE = os.getenv("EXCHANGE", "NSE")
QUANTITY = int(os.getenv("QUANTITY", "100"))
PRODUCT = os.getenv("PRODUCT", "MIS")  # MIS = Intraday, CNC = Delivery
STRATEGY_NAME = os.getenv("STRATEGY_NAME", "SupertrendRSIPullback_v1")
ALLOW_SHORT = os.getenv("ALLOW_SHORT", "True").lower() == "true"

# --- Data Settings ---
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "5m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "15"))
SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", "15"))  # seconds

# --- Supertrend Parameters ---
ST_ATR_LEN = int(os.getenv("ST_ATR_LEN", "10"))
ST_MULT = float(os.getenv("ST_MULT", "3.0"))

# --- RSI Parameters ---
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
RSI_SMA_LEN = int(os.getenv("RSI_SMA_LEN", "14"))

# --- RSI Zone Filter ---
USE_RSI_ZONE = os.getenv("USE_RSI_ZONE", "True").lower() == "true"
RSI_BULL_MIN = float(os.getenv("RSI_BULL_MIN", "40.0"))
RSI_BULL_MAX = float(os.getenv("RSI_BULL_MAX", "50.0"))
RSI_BEAR_MIN = float(os.getenv("RSI_BEAR_MIN", "50.0"))
RSI_BEAR_MAX = float(os.getenv("RSI_BEAR_MAX", "60.0"))

# --- Exit Parameters ---
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
TP1_MULT = float(os.getenv("TP1_MULT", "1.0"))
TP2_MULT = float(os.getenv("TP2_MULT", "2.0"))

# --- ADX Filter ---
USE_ADX = os.getenv("USE_ADX", "True").lower() == "true"
ADX_LEN = int(os.getenv("ADX_LEN", "14"))
ADX_THRESHOLD = float(os.getenv("ADX_THRESHOLD", "25.0"))

# --- Higher Timeframe Filter ---
USE_HTF = os.getenv("USE_HTF", "True").lower() == "true"
HTF_TIMEFRAME = os.getenv("HTF_TIMEFRAME", "15m")

# --- WhatsApp / Telegram alerts ---
WHATSAPP_PHONES: list[str] = [
    n.strip()
    for n in os.getenv("WHATSAPP_PHONES", "").split(",")
    if n.strip()
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
    """Checks if the current Indian time is within the trading session (09:20 to 15:15 IST)."""
    now_ist = get_now_ist()
    current_time = now_ist.time()
    session_start = dt_time(9, 20)
    session_end = dt_time(15, 15)
    return session_start <= current_time < session_end

def is_square_off_time() -> bool:
    """Checks if the current Indian time is past the intraday square-off limit (15:15 IST)."""
    now_ist = get_now_ist()
    current_time = now_ist.time()
    square_off_limit = dt_time(15, 15)
    return current_time >= square_off_limit

# ==============================================================================
# TECHNICAL INDICATOR MATHEMATICS (Identical to Pine Script Formulas)
# ==============================================================================

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    g = d.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    losses = (-d.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + g / losses.replace(0, np.nan))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Computes Supertrend.
    Returns (supertrend_value_series, direction_series)
    direction: 1 for bearish, -1 for bullish (matching Pine's trend == 1 / -1)
    """
    hl2 = (df["high"] + df["low"]) / 2
    atr = compute_atr(df, period)
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    final_upper_band = pd.Series(0.0, index=df.index)
    final_lower_band = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    
    for i in range(len(df)):
        if i == 0:
            final_upper_band.iloc[i] = upper_band.iloc[i]
            final_lower_band.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
            supertrend.iloc[i] = final_upper_band.iloc[i]
            continue
            
        # Upper band update
        if upper_band.iloc[i] < final_upper_band.iloc[i-1] or df["close"].iloc[i-1] > final_upper_band.iloc[i-1]:
            final_upper_band.iloc[i] = upper_band.iloc[i]
        else:
            final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
            
        # Lower band update
        if lower_band.iloc[i] > final_lower_band.iloc[i-1] or df["close"].iloc[i-1] < final_lower_band.iloc[i-1]:
            final_lower_band.iloc[i] = lower_band.iloc[i]
        else:
            final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
            
        # Direction changes
        if supertrend.iloc[i-1] == final_upper_band.iloc[i-1]:
            if df["close"].iloc[i] > final_upper_band.iloc[i]:
                direction.iloc[i] = -1  # switch to bullish
                supertrend.iloc[i] = final_lower_band.iloc[i]
            else:
                direction.iloc[i] = 1  # stay bearish
                supertrend.iloc[i] = final_upper_band.iloc[i]
        else:
            if df["close"].iloc[i] < final_lower_band.iloc[i]:
                direction.iloc[i] = 1  # switch to bearish
                supertrend.iloc[i] = final_upper_band.iloc[i]
            else:
                direction.iloc[i] = -1  # stay bullish
                supertrend.iloc[i] = final_lower_band.iloc[i]
                
    return supertrend, direction

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    
    tr_rma = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / tr_rma.replace(0, np.nan)
    plus_di = plus_di.fillna(0.0)
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / tr_rma.replace(0, np.nan)
    minus_di = minus_di.fillna(0.0)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    dx = dx.fillna(0.0)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx

# ==============================================================================
# SIGNAL CALCULATIONS
# ==============================================================================

def compute_signals(df: pd.DataFrame, df_htf: Optional[pd.DataFrame] = None) -> dict:
    """Computes technical indicators and entry signals from OHLCV data."""
    min_bars = max(ST_ATR_LEN, RSI_LEN, RSI_SMA_LEN, ATR_LEN, ADX_LEN) + 5
    if len(df) < min_bars:
        return {"long": False, "short": False, "indicators": {}}
        
    # Standard indicators
    supertrend, direction = compute_supertrend(df, ST_ATR_LEN, ST_MULT)
    rsi = compute_rsi(df["close"], RSI_LEN)
    rsi_sma = compute_sma(rsi, RSI_SMA_LEN)
    atr = compute_atr(df, ATR_LEN)
    adx = compute_adx(df, ADX_LEN)
    
    # Conditions at last closed candle [-2] and current candle [-1]
    rsi_falling_prev = (rsi.iloc[-2] < rsi.iloc[-3]) and (rsi.iloc[-2] < rsi_sma.iloc[-2])
    rsi_rising_prev = (rsi.iloc[-2] > rsi.iloc[-3]) and (rsi.iloc[-2] > rsi_sma.iloc[-2])
    
    rsi_turn_up = rsi.iloc[-1] > rsi.iloc[-2]
    rsi_turn_down = rsi.iloc[-1] < rsi.iloc[-2]
    
    rsi_bull_ok = not USE_RSI_ZONE or (RSI_BULL_MIN <= rsi.iloc[-1] <= RSI_BULL_MAX)
    rsi_bear_ok = not USE_RSI_ZONE or (RSI_BEAR_MIN <= rsi.iloc[-1] <= RSI_BEAR_MAX)
    
    adx_ok = not USE_ADX or (adx.iloc[-1] > ADX_THRESHOLD)
    
    # HTF trend calculation
    htf_trend = 0
    if USE_HTF and df_htf is not None and len(df_htf) >= min_bars:
        _, htf_direction = compute_supertrend(df_htf, ST_ATR_LEN, ST_MULT)
        htf_trend = htf_direction.iloc[-1]
        
    htf_bull = not USE_HTF or htf_trend == -1
    htf_bear = not USE_HTF or htf_trend == 1
    
    # Breakout conditions
    high_breakout = df["high"].iloc[-1] > df["high"].iloc[-2]
    low_breakout = df["low"].iloc[-1] < df["low"].iloc[-2]
    
    trend = direction.iloc[-1]
    
    long_signal = (
        trend == -1 and
        rsi_falling_prev and
        rsi_turn_up and
        rsi_bull_ok and
        adx_ok and
        htf_bull and
        high_breakout
    )
    
    short_signal = (
        trend == 1 and
        rsi_rising_prev and
        rsi_turn_down and
        rsi_bear_ok and
        adx_ok and
        htf_bear and
        low_breakout
    )
    
    indicators = {
        "rsi": rsi.iloc[-1],
        "rsi_sma": rsi_sma.iloc[-1],
        "adx": adx.iloc[-1],
        "atr": atr.iloc[-1],
        "st_value": supertrend.iloc[-1],
        "st_direction": trend,
        "htf_trend": htf_trend
    }
    
    return {
        "long": bool(long_signal),
        "short": bool(short_signal),
        "indicators": indicators
    }

# ==============================================================================
# STRATEGY BOT ENGINE
# ==============================================================================

class SupertrendRSIPullbackBot:
    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)
        
        # Position states
        self.position: Optional[str] = None  # "BUY", "SELL", or None
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.tp1_price = 0.0
        self.tp2_price = 0.0
        self.tp1_hit = False
        self.remaining_qty = 0
        self.qty_tp1 = 0
        self.qty_tp2 = 0
        
        self.ltp: Optional[float] = None
        self.running = True
        self.stop_event = threading.Event()
        self.instrument = [{"exchange": EXCHANGE, "symbol": SYMBOL}]
        
        self.daily_trade_taken = False
        self.last_trade_date = None
        
        # Historical Caching
        self._cache_df = None
        self._cache_last_fetched = 0.0
        self._cache_htf_df = None
        self._cache_htf_last_fetched = 0.0
        
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
            if self._cache_df is not None and (current_time - self._cache_last_fetched < cache_expiry):
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

    def get_htf_historical_data(self) -> pd.DataFrame:
        """Fetches Higher Time Frame (HTF) timeframe data."""
        try:
            cache_expiry = 300
            current_time = time.time()
            if self._cache_htf_df is not None and (current_time - self._cache_htf_last_fetched < cache_expiry):
                return self._cache_htf_df.copy()

            end = datetime.now()
            start = end - timedelta(days=LOOKBACK_DAYS)
            source = "db" if EXCHANGE.endswith("_INDEX") else "api"
            
            result = self.client.history(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval=HTF_TIMEFRAME,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                source=source,
            )
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                df = result.reset_index()
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                self._cache_htf_df = df.copy()
                self._cache_htf_last_fetched = current_time
                return df
                
            if isinstance(result, dict):
                logger.warning(f"HTF History API response warning: {result.get('message', result)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching HTF historical data: {e}")
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
                logger.info(f"Funds Check | Cash Available: {available:.2f} | Est. Order Cost: {cost:.2f}")
                if available < cost:
                    logger.warning(f"Insufficient funds: needs {cost:.2f}, only have {available:.2f}")
                    return False
                return True
            logger.warning(f"Funds API did not return success. Responding dictionary: {r}. Proceeding order execution.")
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
                logger.info(f"WhatsApp notification sent successfully. Message Time: {now_ist.strftime('%H:%M:%S IST')}")
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")

    # ---------------------------------------------------------------- Order Routing
    def place_entry_order(self, action: str, atr_val: float):
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
            self.entry_atr = atr_val
            
            # Setup exit targets
            multiplier = 1.0 if action == "BUY" else -1.0
            self.tp1_price = self.entry_price + self.entry_atr * TP1_MULT * multiplier
            self.tp2_price = self.entry_price + self.entry_atr * TP2_MULT * multiplier
            self.tp1_hit = False
            self.remaining_qty = QUANTITY
            
            # Split exit quantities for partial profit taking
            self.qty_tp1 = int(QUANTITY * 0.5)
            if QUANTITY > 0 and self.qty_tp1 == 0:
                self.qty_tp1 = 1
            self.qty_tp2 = QUANTITY - self.qty_tp1
            
            self.daily_trade_taken = True
            self.last_trade_date = get_now_ist().date()
            
            logger.info(
                f"Entry {action} Order Successful @ {self.entry_price:.2f} | "
                f"TP1: {self.tp1_price:.2f} (qty={self.qty_tp1}) | "
                f"TP2: {self.tp2_price:.2f} (qty={self.qty_tp2})"
            )
            self.notify(action, "ENTRY SUCCESS", self.entry_price, f"TP1: {self.tp1_price:.2f}, TP2: {self.tp2_price:.2f}")
        else:
            logger.error(f"Entry order failed: {resp}")
            self.notify(action, f"ENTRY FAILED: {resp.get('message', resp)}", 0.0)

    def place_partial_exit_order(self, qty: int, target_label: str):
        exit_action = "SELL" if self.position == "BUY" else "BUY"
        logger.info(f"Placing partial exit order [{target_label}] via {exit_action} for {qty} shares...")
        
        resp = self.client.placeorder(
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=exit_action,
            quantity=qty,
            price_type="MARKET",
            product=PRODUCT,
        )
        if resp.get("status") == "success":
            exit_price = self._get_price_estimate()
            logger.info(f"Partial Exit Successful @ {exit_price:.2f} for {qty} shares. Reason: {target_label}")
            self.notify(exit_action, f"PARTIAL EXIT SUCCESS ({target_label})", exit_price, f"Qty: {qty}")
        else:
            logger.error(f"Partial Exit order failed: {resp}")
            self.notify(exit_action, f"PARTIAL EXIT FAILED ({target_label}): {resp.get('message', resp)}", 0.0)

    def place_full_exit_order(self, reason: str):
        exit_action = "SELL" if self.position == "BUY" else "BUY"
        qty = self.remaining_qty
        logger.info(f"Placing full exit order [{reason}] via {exit_action} for {qty} shares...")
        
        resp = self.client.placeorder(
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=exit_action,
            quantity=qty,
            price_type="MARKET",
            product=PRODUCT,
        )
        if resp.get("status") == "success":
            exit_price = self._get_price_estimate()
            logger.info(f"Full Exit Successful @ {exit_price:.2f}. Reason: {reason}")
            self.notify(exit_action, f"FULL EXIT SUCCESS ({reason})", exit_price, f"Qty: {qty}")
            
            # Reset values
            self.position = None
            self.entry_price = 0.0
            self.entry_atr = 0.0
            self.tp1_price = 0.0
            self.tp2_price = 0.0
            self.tp1_hit = False
            self.remaining_qty = 0
        else:
            logger.error(f"Full Exit order failed: {resp}")
            self.notify(exit_action, f"FULL EXIT FAILED ({reason}): {resp.get('message', resp)}", 0.0)

    # ---------------------------------------------------------------- Signal Checking & Monitoring
    def check_signals(self):
        now_ist = get_now_ist()
        current_date = now_ist.date()
        
        if self.last_trade_date != current_date:
            self.daily_trade_taken = False
            
        in_sess = is_in_session()
        eod_exit = is_square_off_time()
        
        # EOD Close out (For Intraday MIS trades)
        if eod_exit and self.position is not None and PRODUCT == "MIS":
            logger.info("Forced daily EOD exit time reached (15:15 IST). Exiting active position.")
            self.place_full_exit_order("EOD")
            return
            
        if self.position is not None:
            # Monitor active trades using current LTP
            current_price = self.ltp if self.ltp is not None else 0.0
            if current_price <= 0.0:
                return
                
            # Dynamic Stop Loss
            sl_price = 0.0
            if not self.tp1_hit:
                df = self.get_historical_data()
                if not df.empty:
                    if self.ltp is not None:
                        df.loc[df.index[-1], "close"] = self.ltp
                    supertrend, _ = compute_supertrend(df, ST_ATR_LEN, ST_MULT)
                    sl_price = float(supertrend.iloc[-1])
            else:
                sl_price = self.entry_price
                
            exit_triggered = False
            exit_reason = ""
            
            if self.position == "BUY":
                # Check TP1 hit
                if not self.tp1_hit and current_price >= self.tp1_price:
                    logger.info(f"Long TP1 reached ({current_price:.2f} >= {self.tp1_price:.2f}). Exiting {self.qty_tp1} shares.")
                    self.tp1_hit = True
                    self.place_partial_exit_order(self.qty_tp1, "TP1")
                    self.remaining_qty -= self.qty_tp1
                    logger.info(f"Stop loss trailed to Breakeven @ {self.entry_price:.2f}")
                    return
                # Check TP2 hit (only valid if TP1 is already hit)
                elif self.tp1_hit and current_price >= self.tp2_price:
                    logger.info(f"Long TP2 reached ({current_price:.2f} >= {self.tp2_price:.2f}). Exiting remaining.")
                    exit_triggered = True
                    exit_reason = "TP2"
                # Check Stop Loss
                elif current_price <= sl_price:
                    logger.info(f"Long Stop Loss hit ({current_price:.2f} <= {sl_price:.2f}). Exiting remaining.")
                    exit_triggered = True
                    exit_reason = "SL"
                    
            elif self.position == "SELL":
                # Check TP1 hit
                if not self.tp1_hit and current_price <= self.tp1_price:
                    logger.info(f"Short TP1 reached ({current_price:.2f} <= {self.tp1_price:.2f}). Exiting {self.qty_tp1} shares.")
                    self.tp1_hit = True
                    self.place_partial_exit_order(self.qty_tp1, "TP1")
                    self.remaining_qty -= self.qty_tp1
                    logger.info(f"Stop loss trailed to Breakeven @ {self.entry_price:.2f}")
                    return
                # Check TP2 hit
                elif self.tp1_hit and current_price <= self.tp2_price:
                    logger.info(f"Short TP2 reached ({current_price:.2f} <= {self.tp2_price:.2f}). Exiting remaining.")
                    exit_triggered = True
                    exit_reason = "TP2"
                # Check Stop Loss
                elif current_price >= sl_price:
                    logger.info(f"Short Stop Loss hit ({current_price:.2f} >= {sl_price:.2f}). Exiting remaining.")
                    exit_triggered = True
                    exit_reason = "SL"
                    
            if exit_triggered:
                self.place_full_exit_order(exit_reason)
                
        else:
            # Check for entries only when in session and daily trade limit isn't reached
            if in_sess and not self.daily_trade_taken:
                df = self.get_historical_data()
                if df.empty:
                    return
                    
                df_htf = None
                if USE_HTF:
                    df_htf = self.get_htf_historical_data()
                    if df_htf.empty:
                        return
                        
                # Update last row with real-time LTP
                if self.ltp is not None:
                    # Main df update
                    df.loc[df.index[-1], "close"] = self.ltp
                    if self.ltp > df.loc[df.index[-1], "high"]:
                        df.loc[df.index[-1], "high"] = self.ltp
                    if self.ltp < df.loc[df.index[-1], "low"]:
                        df.loc[df.index[-1], "low"] = self.ltp
                    
                    # HTF df update
                    df_htf.loc[df_htf.index[-1], "close"] = self.ltp
                    if self.ltp > df_htf.loc[df_htf.index[-1], "high"]:
                        df_htf.loc[df_htf.index[-1], "high"] = self.ltp
                    if self.ltp < df_htf.loc[df_htf.index[-1], "low"]:
                        df_htf.loc[df_htf.index[-1], "low"] = self.ltp
                        
                signal_res = compute_signals(df, df_htf)
                
                if signal_res["long"]:
                    logger.info("Signal generated: BUY")
                    atr_val = signal_res["indicators"]["atr"]
                    self.place_entry_order("BUY", atr_val)
                elif signal_res["short"] and ALLOW_SHORT:
                    logger.info("Signal generated: SELL")
                    atr_val = signal_res["indicators"]["atr"]
                    self.place_entry_order("SELL", atr_val)

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
                self.place_full_exit_order("Bot Shutdown")


if __name__ == "__main__":
    bot = SupertrendRSIPullbackBot()
    bot.run()
