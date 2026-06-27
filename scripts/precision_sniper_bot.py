"""
Precision Sniper [WillyAlgoTrader] Live Trading Bot for OpenAlgo
--------------------------------------------------------------
Replicates the v1.4.0 Pine Script logic including presets, confluence
scoring engine, volatility filters, and structure-based trailing stop-loss.
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PrecSniperBot")

try:
    from openalgo import api
except ImportError:
    logger.error(
        "The 'openalgo' package is required to run this bot. Install it with: pip install openalgo"
    )
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEY = os.getenv("OPENALGO_API_KEY", "openalgo-apikey")
API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

# Instrument details
SYMBOL = os.getenv("SYMBOL", "MCX")
EXCHANGE = os.getenv("EXCHANGE", "NSE")
QUANTITY = int(os.getenv("QUANTITY", "1"))
PRODUCT = os.getenv("PRODUCT", "MIS")  # MIS (Intraday), CNC, etc.
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "15m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "10"))
SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", "15"))

# Main strategy settings
PRESET = os.getenv(
    "PRESET", "Default"
)  # Default, Conservative, Aggressive, Scalping, Swing, Crypto 24/7, Custom
HTF_TIMEFRAME = os.getenv(
    "HTF_TIMEFRAME", ""
)  # Empty to disable HTF filter, or e.g., '1h', '4h', '1d'
MIN_SCORE_INPUT = int(os.getenv("MIN_SCORE", "5"))
RSI_LEN_INPUT = int(os.getenv("RSI_LEN", "13"))
ATR_LEN_INPUT = int(os.getenv("ATR_LEN", "14"))

# Gating & Grade Filters
GRADE_FILTER = os.getenv("GRADE_FILTER", "All")  # All, A+ and A, A+ Only
HIDE_C_GRADE = os.getenv("HIDE_C_GRADE", "True").lower() == "true"
VOL_FILTER_MODE = os.getenv("VOL_FILTER_MODE", "Skip Signals")  # Off, Skip Signals, Widen SL
VOL_WIDEN_FACTOR = float(os.getenv("VOL_WIDEN_FACTOR", "1.5"))
HIGH_VOL_THRESH = float(os.getenv("HIGH_VOL_THRESH", "1.3"))

# Risk Settings
SL_MULT_INPUT = float(os.getenv("SL_MULT", "1.5"))
TP1_MULT = float(os.getenv("TP1_MULT", "1.0"))
TP2_MULT = float(os.getenv("TP2_MULT", "2.0"))
TP3_MULT = float(os.getenv("TP3_MULT", "3.0"))
USE_TRAIL = os.getenv("USE_TRAIL", "True").lower() == "true"
FULL_EXIT_TP3 = os.getenv("FULL_EXIT_TP3", "True").lower() == "true"
USE_STRUCTURE_SL = os.getenv("USE_STRUCTURE_SL", "True").lower() == "true"
SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "10"))

# Thresholds as ratios of actual max score
GRADE_APLUS_R = 0.80
GRADE_A_R = 0.65
GRADE_B_R = 0.50

# Presets mapping (overrides inputs)
PRESETS = {
    "Scalping": {
        "ema_fast": 5,
        "ema_slow": 13,
        "ema_trend": 34,
        "rsi_len": 8,
        "atr_len": 10,
        "min_score": 4,
        "sl_mult": 0.8,
    },
    "Aggressive": {
        "ema_fast": 8,
        "ema_slow": 18,
        "ema_trend": 50,
        "rsi_len": 11,
        "atr_len": 12,
        "min_score": 3,
        "sl_mult": 1.2,
    },
    "Default": {
        "ema_fast": 9,
        "ema_slow": 21,
        "ema_trend": 55,
        "rsi_len": 13,
        "atr_len": 14,
        "min_score": 5,
        "sl_mult": 1.5,
    },
    "Conservative": {
        "ema_fast": 12,
        "ema_slow": 26,
        "ema_trend": 89,
        "rsi_len": 14,
        "atr_len": 14,
        "min_score": 7,
        "sl_mult": 2.0,
    },
    "Swing": {
        "ema_fast": 13,
        "ema_slow": 34,
        "ema_trend": 89,
        "rsi_len": 21,
        "atr_len": 20,
        "min_score": 6,
        "sl_mult": 2.5,
    },
    "Crypto 24/7": {
        "ema_fast": 9,
        "ema_slow": 21,
        "ema_trend": 55,
        "rsi_len": 14,
        "atr_len": 20,
        "min_score": 5,
        "sl_mult": 2.0,
    },
}

# Resolve preset variables
resolved_preset = PRESET
if PRESET == "Auto":
    # Autodetect based on candle timeframe (mapped to minutes)
    tf_str = CANDLE_TIMEFRAME.lower()
    minutes = 5.0
    if tf_str.endswith("m"):
        minutes = float(tf_str[:-1])
    elif tf_str.endswith("h"):
        minutes = float(tf_str[:-1]) * 60.0
    elif tf_str.endswith("d"):
        minutes = 1440.0

    if minutes <= 5:
        resolved_preset = "Scalping"
    elif minutes <= 60:
        resolved_preset = "Default"
    elif minutes < 240:
        resolved_preset = "Conservative"
    else:
        resolved_preset = "Swing"

params = PRESETS.get(
    resolved_preset,
    {
        "ema_fast": int(os.getenv("EMA_FAST_LEN", "9")),
        "ema_slow": int(os.getenv("EMA_SLOW_LEN", "21")),
        "ema_trend": int(os.getenv("EMA_TREND_LEN", "55")),
        "rsi_len": RSI_LEN_INPUT,
        "atr_len": ATR_LEN_INPUT,
        "min_score": MIN_SCORE_INPUT,
        "sl_mult": SL_MULT_INPUT,
    },
)

EMA_FAST_LEN = params["ema_fast"]
EMA_SLOW_LEN = params["ema_slow"]
EMA_TREND_LEN = params["ema_trend"]
RSI_LEN = params["rsi_len"]
ATR_LEN = params["atr_len"]
MIN_SCORE = params["min_score"]
SL_MULT = params["sl_mult"]


# ==============================================================================
# INDICATORS
# ==============================================================================
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["pv"] = df["close"] * df["volume"]
    cum_pv = df.groupby("date")["pv"].cumsum()
    cum_vol = df.groupby("date")["volume"].cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def compute_dmi(df: pd.DataFrame, period: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = (
        100
        * pd.Series(plus_dm).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / atr
    )
    minus_di = (
        100
        * pd.Series(minus_dm).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / atr
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return plus_di, minus_di, adx


# ==============================================================================
# BOT IMPLEMENTATION
# ==============================================================================
class PrecisionSniperBot:
    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)
        self.strategy_name = os.getenv("STRATEGY_NAME", "PrecisionSniper_v140")

        # Position states
        self.position = None  # "BUY", "SHORT", or None
        self.entry_price = 0.0
        self.ltp = None
        self.running = True
        self.stop_event = threading.Event()
        self.instrument = [{"exchange": EXCHANGE, "symbol": SYMBOL}]
        self.daily_trade_taken = False
        self.last_trade_date = None

        # Strategy dynamic stop/targets tracking
        self.sl_price = 0.0
        self.trail_price = 0.0
        self.tp1_price = 0.0
        self.tp2_price = 0.0
        self.tp3_price = 0.0
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False

        # Confluence state
        self.last_direction = 0  # 1 for Long, -1 for Short, 0 after stopped out

        logger.info(
            f"Initialized Precision Sniper Bot (v1.4.0) for {SYMBOL}:{EXCHANGE}\n"
            f"Preset: {resolved_preset} | Qty: {QUANTITY} | Product: {PRODUCT}"
        )

    def on_ltp_update(self, data):
        if data.get("type") == "market_data" and data.get("symbol") == SYMBOL:
            self.ltp = float(data["data"]["ltp"])

    def websocket_thread(self):
        try:
            self.client.connect()
            self.client.subscribe_ltp(self.instrument, on_data_received=self.on_ltp_update)
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

    def get_historical_data(self, tf: str = CANDLE_TIMEFRAME) -> pd.DataFrame:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=LOOKBACK_DAYS)
            history_data = self.client.history(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval=tf,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
            if history_data and isinstance(history_data, list):
                df = pd.DataFrame(history_data)
                # Parse numeric columns
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch historical data for TF {tf}: {e}")
            return pd.DataFrame()

    def get_htf_bias(self) -> int:
        """Fetch HTF candles and return 1 if Fast > Slow EMA, -1 if Fast < Slow, 0 otherwise."""
        if not HTF_TIMEFRAME:
            return 0
        try:
            df_htf = self.get_historical_data(tf=HTF_TIMEFRAME)
            if df_htf.empty or len(df_htf) < max(EMA_FAST_LEN, EMA_SLOW_LEN) + 1:
                return 0

            fast_ema = compute_ema(df_htf["close"], EMA_FAST_LEN)
            slow_ema = compute_ema(df_htf["close"], EMA_SLOW_LEN)
            latest_fast = fast_ema.iloc[-1]
            latest_slow = slow_ema.iloc[-1]
            if latest_fast > latest_slow:
                return 1
            elif latest_fast < latest_slow:
                return -1
            return 0
        except Exception as e:
            logger.error(f"Error checking HTF bias: {e}")
            return 0

    def check_funds_before_order(self) -> bool:
        """Verify funds availability based on estimated entry price."""
        try:
            funds_resp = self.client.funds()
            if funds_resp and funds_resp.get("status") == "success":
                funds_data = funds_resp.get("data", {})
                available_balance = float(funds_data.get("available_balance", 0.0))

                price = self.ltp if self.ltp is not None else 0.0
                if price <= 0.0:
                    quotes_resp = self.client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
                    if quotes_resp and quotes_resp.get("status") == "success":
                        price = float(quotes_resp.get("data", {}).get("last_price", 0.0))

                if price <= 0.0:
                    df = self.get_historical_data()
                    if not df.empty:
                        price = float(df["close"].iloc[-1])

                estimated_cost = price * QUANTITY
                logger.info(
                    f"Checking funds: Available Balance = {available_balance:.2f}, Est. Cost = {estimated_cost:.2f}"
                )

                if available_balance < estimated_cost:
                    logger.warning(
                        f"Insufficient funds! Needed: {estimated_cost:.2f}, Available: {available_balance:.2f}"
                    )
                    return False
                return True
            else:
                logger.warning("Could not fetch funds details. Proceeding anyway.")
                return True
        except Exception as e:
            logger.error(f"Error checking funds: {e}")
            return True

    def send_whatsapp_notification(self, action: str, status: str, price: float = 0.0):
        url = f"{API_HOST}/api/v1/whatsapp/notify"
        api_key = os.getenv("WHATSAPP_API_KEY", API_KEY)

        msg = (
            f"[PRECISION SNIPER BOT]\n"
            f"Strategy: {self.strategy_name}\n"
            f"Action: {action}\n"
            f"Status: {status}\n"
            f"Symbol: {SYMBOL}\n"
            f"Price: {price:.2f}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        payload = {"apikey": api_key, "self": True, "message": msg}
        from urllib.request import Request, urlopen

        try:
            req = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=5.0) as response:
                response.read()
            logger.info("WhatsApp notification sent successfully.")
        except Exception as e:
            logger.warning(f"WhatsApp notification failed: {e}")

    def get_grade(self, score: float, max_s: float) -> str:
        ratio = score / max_s if max_s > 0 else 0.0
        if ratio >= GRADE_APLUS_R:
            return "A+"
        if ratio >= GRADE_A_R:
            return "A"
        if ratio >= GRADE_B_R:
            return "B"
        return "C"

    def passes_grade_filter(self, score: float, max_s: float) -> bool:
        ratio = score / max_s if max_s > 0 else 0.0
        grade_ok = True
        if GRADE_FILTER == "A+ Only":
            grade_ok = ratio >= GRADE_APLUS_R
        elif GRADE_FILTER == "A+ and A":
            grade_ok = ratio >= GRADE_A_R

        c_ok = True
        if HIDE_C_GRADE:
            c_ok = ratio >= GRADE_B_R
        return grade_ok and c_ok

    def place_entry_order(
        self, action: str, atr_val: float, high_vol_now: bool, swing_low: float, swing_high: float
    ):
        if not self.check_funds_before_order():
            logger.warning("Entry aborted due to insufficient funds.")
            return

        logger.info(f"Placing entry {action} order for {QUANTITY} shares of {SYMBOL}...")
        response = self.client.placeorder(
            strategy=self.strategy_name,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=action,
            quantity=QUANTITY,
            price_type="MARKET",
            product=PRODUCT,
        )
        if response.get("status") == "success":
            self.position = action
            self.entry_price = self.ltp if self.ltp is not None else 0.0
            if self.entry_price <= 0.0:
                quotes_resp = self.client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
                if quotes_resp and quotes_resp.get("status") == "success":
                    self.entry_price = float(quotes_resp.get("data", {}).get("last_price", 0.0))

            # Setup dynamic SL/TP targets
            sl_vol_mult = (
                VOL_WIDEN_FACTOR if (VOL_FILTER_MODE == "Widen SL" and high_vol_now) else 1.0
            )
            risk = atr_val * SL_MULT * sl_vol_mult

            # Stop Loss Calculation
            is_long = action == "BUY"
            atr_stop = self.entry_price - risk if is_long else self.entry_price + risk

            if USE_STRUCTURE_SL:
                struct_stop = (
                    (swing_low - atr_val * 0.2) if is_long else (swing_high + atr_val * 0.2)
                )
                final_stop = min(atr_stop, struct_stop) if is_long else max(atr_stop, struct_stop)

                # Cap structure stops
                max_dist = risk * 1.5
                dist = abs(self.entry_price - final_stop)
                if dist > max_dist:
                    final_stop = (
                        (self.entry_price - max_dist) if is_long else (self.entry_price + max_dist)
                    )

                # Floor structure stops
                min_dist = atr_val * 0.5
                dist = abs(self.entry_price - final_stop)
                if dist < min_dist:
                    final_stop = (
                        (self.entry_price - min_dist) if is_long else (self.entry_price + min_dist)
                    )

                self.sl_price = final_stop
            else:
                self.sl_price = atr_stop

            trade_risk = abs(self.entry_price - self.sl_price)
            multiplier = 1.0 if is_long else -1.0
            self.tp1_price = self.entry_price + trade_risk * TP1_MULT * multiplier
            self.tp2_price = self.entry_price + trade_risk * TP2_MULT * multiplier
            self.tp3_price = self.entry_price + trade_risk * TP3_MULT * multiplier
            self.trail_price = self.sl_price

            self.tp1_hit = False
            self.tp2_hit = False
            self.tp3_hit = False
            self.last_direction = 1 if is_long else -1
            self.daily_trade_taken = True
            self.last_trade_date = datetime.now().date()

            logger.info(
                f"Entry Order Successful. Price: {self.entry_price:.2f} | "
                f"SL: {self.sl_price:.2f} | TP1: {self.tp1_price:.2f} | "
                f"TP2: {self.tp2_price:.2f} | TP3: {self.tp3_price:.2f}"
            )
            self.send_whatsapp_notification(action, "success", self.entry_price)
        else:
            logger.error(f"Entry order failed: {response}")
            self.send_whatsapp_notification(action, "failed", 0.0)

    def place_exit_order(self, reason: str):
        exit_action = "SELL" if self.position == "BUY" else "BUY"
        logger.info(f"Placing exit order [{reason}] via {exit_action} for {QUANTITY} shares...")
        response = self.client.placeorder(
            strategy=self.strategy_name,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=exit_action,
            quantity=QUANTITY,
            price_type="MARKET",
            product=PRODUCT,
        )
        if response.get("status") == "success":
            exit_price = self.ltp if self.ltp is not None else 0.0
            if exit_price <= 0.0:
                quotes_resp = self.client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
                if quotes_resp and quotes_resp.get("status") == "success":
                    exit_price = float(quotes_resp.get("data", {}).get("last_price", 0.0))
            logger.info(f"Exit Order Successful at {exit_price:.2f}. Reason: {reason}")
            self.send_whatsapp_notification(exit_action, f"success (Exit: {reason})", exit_price)
            self.position = None
            self.entry_price = 0.0
            self.sl_price = 0.0
            self.trail_price = 0.0
            # Note: last_direction is reset to 0 upon stop-loss event or TP3 full exit
            if reason == "SL" or (reason == "TP3" and FULL_EXIT_TP3):
                self.last_direction = 0
        else:
            logger.error(f"Exit order failed: {response}")
            self.send_whatsapp_notification(exit_action, f"failed (Exit: {reason})", 0.0)

    def check_signals(self):
        now = datetime.now()
        current_date = now.date()

        # Reset daily trade state
        if self.last_trade_date != current_date:
            self.daily_trade_taken = False

        # Forced intraday close check (if intraday MIS)
        is_exit_time = now.hour == 15 and now.minute >= 15

        if self.position is not None:
            # We are active. Perform trailing and check SL/TP exits using live LTP.
            current_price = self.ltp if self.ltp is not None else 0.0
            if current_price <= 0.0:
                return

            exit_triggered = False
            exit_reason = ""

            if self.position == "BUY":
                # Trailing logic: After TP1 hit, trail to breakeven. After TP2 hit, trail to TP1 price.
                if current_price >= self.tp1_price and not self.tp1_hit:
                    self.tp1_hit = True
                    if USE_TRAIL:
                        self.trail_price = self.entry_price
                        logger.info(
                            f"TP1 hit. Trailing Stop moved to breakeven: {self.trail_price:.2f}"
                        )
                if current_price >= self.tp2_price and not self.tp2_hit:
                    self.tp2_hit = True
                    if USE_TRAIL:
                        self.trail_price = self.tp1_price
                        logger.info(f"TP2 hit. Trailing Stop moved to TP1: {self.trail_price:.2f}")
                if current_price >= self.tp3_price and not self.tp3_hit:
                    self.tp3_hit = True
                    if USE_TRAIL:
                        self.trail_price = self.tp2_price
                        logger.info(f"TP3 hit. Trailing Stop moved to TP2: {self.trail_price:.2f}")

                # Check hits
                if current_price <= self.trail_price:
                    exit_triggered = True
                    exit_reason = "SL"
                elif self.tp3_hit and FULL_EXIT_TP3:
                    exit_triggered = True
                    exit_reason = "TP3"

            elif self.position == "SHORT":
                if current_price <= self.tp1_price and not self.tp1_hit:
                    self.tp1_hit = True
                    if USE_TRAIL:
                        self.trail_price = self.entry_price
                        logger.info(
                            f"TP1 hit. Trailing Stop moved to breakeven: {self.trail_price:.2f}"
                        )
                if current_price <= self.tp2_price and not self.tp2_hit:
                    self.tp2_hit = True
                    if USE_TRAIL:
                        self.trail_price = self.tp1_price
                        logger.info(f"TP2 hit. Trailing Stop moved to TP1: {self.trail_price:.2f}")
                if current_price <= self.tp3_price and not self.tp3_hit:
                    self.tp3_hit = True
                    if USE_TRAIL:
                        self.trail_price = self.tp2_price
                        logger.info(f"TP3 hit. Trailing Stop moved to TP2: {self.trail_price:.2f}")

                # Check hits
                if current_price >= self.trail_price:
                    exit_triggered = True
                    exit_reason = "SL"
                elif self.tp3_hit and FULL_EXIT_TP3:
                    exit_triggered = True
                    exit_reason = "TP3"

            if is_exit_time and not exit_triggered and PRODUCT == "MIS":
                exit_triggered = True
                exit_reason = "EOD"

            if exit_triggered:
                self.place_exit_order(exit_reason)

        else:
            # Flat position. Evaluate entry condition.
            if not self.daily_trade_taken and not is_exit_time:
                df = self.get_historical_data()
                if df.empty or len(df) < max(EMA_TREND_LEN, 50):
                    return

                # Incorporate latest websocket LTP as close
                if self.ltp is not None:
                    df.loc[df.index[-1], "close"] = self.ltp

                # Compute Indicators
                close = df["close"]
                high = df["high"]
                low = df["low"]
                volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

                ema_fast = compute_ema(close, EMA_FAST_LEN)
                ema_slow = compute_ema(close, EMA_SLOW_LEN)
                ema_trend = compute_ema(close, EMA_TREND_LEN)
                atr_val = compute_atr(df, ATR_LEN)
                rsi_val = compute_rsi(close, RSI_LEN)
                macd_line, signal_line, macd_hist = compute_macd(close)

                # Vol check & Regimes
                atr_sma_42 = atr_val.rolling(42).mean()
                vol_ratio = atr_val / atr_sma_42.replace(0, np.nan)
                latest_vol_ratio = vol_ratio.iloc[-1]
                vol_regime = (
                    "High"
                    if latest_vol_ratio > HIGH_VOL_THRESH
                    else "Low"
                    if latest_vol_ratio < 0.7
                    else "Normal"
                )
                high_vol_now = vol_regime == "High"

                vol_filter_ok = True
                if VOL_FILTER_MODE == "Skip Signals" and high_vol_now:
                    vol_filter_ok = False

                # Volume SMA indicator
                sym_has_volume = bool((volume > 0).any())
                vol_sma_20 = volume.rolling(20).mean()
                vol_above_avg = sym_has_volume and (volume.iloc[-1] > vol_sma_20.iloc[-1] * 1.2)

                # DMI / ADX
                di_plus, di_minus, adx = compute_dmi(df, 14)
                latest_adx = adx.iloc[-1]
                latest_di_plus = di_plus.iloc[-1]
                latest_di_minus = di_minus.iloc[-1]
                strong_trend = latest_adx > 20

                # VWAP (intraday only + volume)
                vwap_val = compute_vwap(df)
                is_intraday = True  # We assume intraday calculations apply for bot
                vwap_point_valid = sym_has_volume and is_intraday
                latest_vwap = vwap_val.iloc[-1]

                # HTF Trend Bias
                htf_bias = self.get_htf_bias()

                # Adaptive Max Score
                max_score = (
                    8.0 + (1.0 if sym_has_volume else 0.0) + (1.0 if vwap_point_valid else 0.0)
                )
                effective_min_score = MIN_SCORE * max_score / 10.0

                # Latest values
                latest_close = close.iloc[-1]
                latest_fast = ema_fast.iloc[-1]
                latest_slow = ema_slow.iloc[-1]
                latest_trend = ema_trend.iloc[-1]
                latest_rsi = rsi_val.iloc[-1]
                latest_macd_hist = macd_hist.iloc[-1]
                latest_macd_hist_prev = (
                    macd_hist.iloc[-2] if len(macd_hist) > 1 else latest_macd_hist
                )
                latest_atr = atr_val.iloc[-1]

                # Confluence scoring
                bull_score = 0.0
                bull_score += 1.0 if latest_fast > latest_slow else 0.0
                bull_score += 1.0 if latest_close > latest_trend else 0.0
                bull_score += 1.0 if 50 < latest_rsi < 75 else 0.0
                bull_score += 1.0 if latest_macd_hist > 0 else 0.0
                bull_score += 1.0 if latest_macd_hist > latest_macd_hist_prev else 0.0
                bull_score += 1.0 if (vwap_point_valid and latest_close > latest_vwap) else 0.0
                bull_score += 1.0 if vol_above_avg else 0.0
                bull_score += 1.0 if (strong_trend and latest_di_plus > latest_di_minus) else 0.0
                bull_score += 1.5 if htf_bias == 1 else 0.0
                bull_score += 0.5 if latest_close > latest_fast else 0.0

                bear_score = 0.0
                bear_score += 1.0 if latest_fast < latest_slow else 0.0
                bear_score += 1.0 if latest_close < latest_trend else 0.0
                bear_score += 1.0 if 25 < latest_rsi < 50 else 0.0
                bear_score += 1.0 if latest_macd_hist < 0 else 0.0
                bear_score += 1.0 if latest_macd_hist < latest_macd_hist_prev else 0.0
                bear_score += 1.0 if (vwap_point_valid and latest_close < latest_vwap) else 0.0
                bear_score += 1.0 if vol_above_avg else 0.0
                bear_score += 1.0 if (strong_trend and latest_di_minus > latest_di_plus) else 0.0
                bear_score += 1.5 if htf_bias == -1 else 0.0
                bear_score += 0.5 if latest_close < latest_fast else 0.0

                # Cross-over triggers
                ema_fast_prev = ema_fast.iloc[-2] if len(ema_fast) > 1 else latest_fast
                ema_slow_prev = ema_slow.iloc[-2] if len(ema_slow) > 1 else latest_slow
                ema_bull_cross = (ema_fast_prev <= ema_slow_prev) and (latest_fast > latest_slow)
                ema_bear_cross = (ema_fast_prev >= ema_slow_prev) and (latest_fast < latest_slow)

                bull_momentum = (latest_close > latest_fast) and (latest_close > latest_slow)
                bear_momentum = (latest_close < latest_fast) and (latest_close < latest_slow)

                rsi_not_ob = latest_rsi < 75
                rsi_not_os = latest_rsi > 25

                # Raw entry checks
                raw_buy = (
                    ema_bull_cross
                    and bull_momentum
                    and rsi_not_ob
                    and vol_filter_ok
                    and bull_score >= effective_min_score
                    and self.passes_grade_filter(bull_score, max_score)
                )

                raw_sell = (
                    ema_bear_cross
                    and bear_momentum
                    and rsi_not_os
                    and vol_filter_ok
                    and bear_score >= effective_min_score
                    and self.passes_grade_filter(bear_score, max_score)
                )

                # Structure swings
                recent_swing_low = low.iloc[-(SWING_LOOKBACK + 1) :].min()
                recent_swing_high = high.iloc[-(SWING_LOOKBACK + 1) :].max()

                # Trigger BUY
                if raw_buy and self.last_direction != 1:
                    logger.info(
                        f"🟢 Long Entry Triggered: Bull Score = {bull_score:.1f}/{max_score:.1f} "
                        f"(Grade: {self.get_grade(bull_score, max_score)})"
                    )
                    self.place_entry_order(
                        "BUY", latest_atr, high_vol_now, recent_swing_low, recent_swing_high
                    )

                # Trigger SHORT
                elif raw_sell and self.last_direction != -1:
                    logger.info(
                        f"🔴 Short Entry Triggered: Bear Score = {bear_score:.1f}/{max_score:.1f} "
                        f"(Grade: {self.get_grade(bear_score, max_score)})"
                    )
                    self.place_entry_order(
                        "SHORT", latest_atr, high_vol_now, recent_swing_low, recent_swing_high
                    )

    def run(self):
        ws_thread = threading.Thread(target=self.websocket_thread, daemon=True)
        ws_thread.start()
        time.sleep(2)  # Wait for WebSocket client connection

        logger.info("Precision Sniper execution loop started.")
        try:
            while self.running:
                self.check_signals()
                time.sleep(SIGNAL_CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Bot execution manually stopped.")
        finally:
            self.stop_event.set()
            self.running = False


if __name__ == "__main__":
    bot = PrecisionSniperBot()
    bot.run()
