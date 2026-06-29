"""
BANKNIFTY Optimized Intraday/Daily Strategy Bot for OpenAlgo
------------------------------------------------------------
This script implements the optimized long-only strategy discovered for BANKNIFTY (exchange=NSE).
It reads API connections and trading parameters from environment variables.
"""

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
logger = logging.getLogger("BankNifty_Bot")

try:
    from openalgo import api
except ImportError:
    logger.error(
        "The 'openalgo' package is required to run this bot. Install it with: pip install openalgo"
    )
    sys.exit(1)

# ==============================================================================
# CONFIGURATION (Read from environment variables with defaults)
# ==============================================================================

# API Details
API_KEY = os.getenv(
    "OPENALGO_API_KEY", "b45feb0a6973ed00fe86d25ace49d4da8dfe8d0a78c334455d46254ded28a26d"
)
API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

# WhatsApp alerts — comma-separated E.164 digits (no '+'), up to 5 numbers.
# Leave empty to notify only the paired device (operator self-notification).
# Example: "919876543210,919900112233"
WHATSAPP_PHONES: list[str] = [
    n.strip() for n in os.getenv("WHATSAPP_PHONES", "").split(",") if n.strip()
]

# Trade Parameters
SYMBOL = os.getenv("SYMBOL", "BANKNIFTY")
EXCHANGE = os.getenv("EXCHANGE", "NSE_INDEX")  # Default exchange for trading
QUANTITY = int(os.getenv("QUANTITY", "1"))
PRODUCT = os.getenv("PRODUCT", "MIS")  # Intraday
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "D")  # Optimized timeframe is D
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "100"))  # Warming up EMAs/MACD
SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", "60"))

# Optimized Strategy Parameters
EMA_FAST = 3
EMA_SLOW = 12

RSI_PERIOD = 14
RSI_ENTRY_LO = 35
RSI_ENTRY_HI = 75
RSI_EXIT_HI = 90

MACD_FAST = 5
MACD_SLOW = 13
MACD_SIGNAL = 7

ATR_PERIOD = 14
ATR_MULT_SL = 3.0  # Volatility stop-loss multiplier

TAKE_PROFIT_PCT = 2.0  # Capped profit target
STOP_LOSS_PCT = 3.0  # Capped stop loss target

# ==============================================================================
# INDICATOR FUNCTIONS (Identical to strategy.py backtest logic)
# ==============================================================================


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    g = d.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    losses = (-d.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + g / losses.replace(0, np.nan))


def compute_macd(series: pd.Series, fast: int = 5, slow: int = 13, signal: int = 7):
    macd_line = compute_ema(series, fast) - compute_ema(series, slow)
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ==============================================================================
# STRATEGY EXECUTION ENGINE
# ==============================================================================


class OpenAlgoStrategyBot:
    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)
        self.strategy_name = os.getenv("STRATEGY_NAME", "BankNifty_Optimized_Daily")
        self.position = None  # "BUY" or None (Long-only)
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.ltp = None
        self.running = True
        self.stop_event = threading.Event()
        self.instrument = [{"exchange": EXCHANGE, "symbol": SYMBOL}]
        self.daily_trade_taken = False
        self.last_trade_date = None

        logger.info(
            f"Initialized BankNifty Bot for {SYMBOL}:{EXCHANGE} (Qty: {QUANTITY}, Product: {PRODUCT})"
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

    def get_historical_data(self) -> pd.DataFrame:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=LOOKBACK_DAYS)
            history_data = self.client.history(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval=CANDLE_TIMEFRAME,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
            if isinstance(history_data, pd.DataFrame) and not history_data.empty:
                df = history_data.reset_index()  # bring 'timestamp' back as a column
                # Parse numeric columns
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            # Error dict returned by SDK (e.g. no_data, api_error)
            if isinstance(history_data, dict):
                logger.warning(
                    f"History API returned no data: {history_data.get('message', history_data)}"
                )
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()

    def check_funds_before_order(self) -> bool:
        """Verify available balance against estimated order value."""
        try:
            funds_resp = self.client.funds()
            if isinstance(funds_resp, dict) and funds_resp.get("status") == "success":
                funds_data = funds_resp.get("data", {})
                # SDK returns 'availablecash' (not 'available_balance')
                available_balance = float(funds_data.get("availablecash", 0.0))

                price = self.ltp if self.ltp is not None else 0.0
                if price <= 0.0:
                    quotes_resp = self.client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
                    if isinstance(quotes_resp, dict) and quotes_resp.get("status") == "success":
                        # SDK returns 'ltp' (not 'last_price')
                        price = float(quotes_resp.get("data", {}).get("ltp", 0.0))

                if price <= 0.0:
                    df = self.get_historical_data()
                    if not df.empty:
                        price = float(df["close"].iloc[-1])

                estimated_cost = price * QUANTITY
                logger.info(
                    f"Available Balance: {available_balance:.2f}, Est. Cost: {estimated_cost:.2f}"
                )

                if available_balance < estimated_cost:
                    logger.warning(
                        f"Insufficient funds! Needed: {estimated_cost:.2f}, Available: {available_balance:.2f}"
                    )
                    return False
                return True
            else:
                logger.warning(
                    f"Could not fetch funds info: {funds_resp.get('message', funds_resp) if isinstance(funds_resp, dict) else funds_resp}. Proceeding anyway."
                )
                return True
        except Exception as e:
            logger.error(f"Error checking funds: {e}")
            return True

    def send_whatsapp_notification(self, action: str, status: str, price: float = 0.0):
        """Send WhatsApp alert via SDK. Broadcasts to all configured numbers in one call."""
        msg = (
            f"[BANKNIFTY BOT]\n"
            f"Strategy: {self.strategy_name}\n"
            f"Action: {action}\n"
            f"Status: {status}\n"
            f"Symbol: {SYMBOL}\n"
            f"Quantity: {QUANTITY}\n"
            f"Price: {price:.2f}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            if WHATSAPP_PHONES:
                # Single API call broadcasts to all numbers (server caps at 5)
                r = self.client.whatsapp(msg, to=WHATSAPP_PHONES[:5])
            else:
                # Fallback: notify the paired device (operator's own number)
                r = self.client.whatsapp(msg)

            if isinstance(r, dict) and r.get("status") != "success":
                logger.warning(f"WhatsApp notification issue: {r.get('message', r)}")
            else:
                logger.info("WhatsApp notification sent successfully.")
        except Exception as e:
            logger.warning(f"Failed to send WhatsApp notification: {e}")

    def place_entry_order(self):
        if not self.check_funds_before_order():
            logger.warning("Aborting entry order due to insufficient funds.")
            return

        logger.info(f"Placing entry BUY order for {QUANTITY} shares of {SYMBOL}...")
        response = self.client.placeorder(
            strategy=self.strategy_name,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action="BUY",
            quantity=QUANTITY,
            price_type="MARKET",
            product=PRODUCT,
        )
        if response.get("status") == "success":
            self.position = "BUY"
            self.entry_price = self.ltp if self.ltp is not None else 0.0
            if self.entry_price <= 0.0:
                quotes_resp = self.client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
                if isinstance(quotes_resp, dict) and quotes_resp.get("status") == "success":
                    self.entry_price = float(quotes_resp.get("data", {}).get("ltp", 0.0))
            logger.info(f"Entry order successful. Entry Price: {self.entry_price}")

            # Store the current ATR value for volatility SL
            df = self.get_historical_data()
            if not df.empty:
                df["atr"] = compute_atr(df, ATR_PERIOD)
                self.entry_atr = float(df["atr"].iloc[-1])
            else:
                self.entry_atr = 0.0

            self.daily_trade_taken = True
            self.last_trade_date = datetime.now().date()
            self.send_whatsapp_notification("BUY", "success", self.entry_price)
        else:
            logger.error(f"Entry order failed: {response}")
            self.send_whatsapp_notification("BUY", "failed", 0.0)

    def place_exit_order(self, reason: str = "signal"):
        logger.info(f"Placing exit SELL order for {QUANTITY} shares of {SYMBOL} due to {reason}...")
        response = self.client.placeorder(
            strategy=self.strategy_name,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action="SELL",
            quantity=QUANTITY,
            price_type="MARKET",
            product=PRODUCT,
        )
        if response.get("status") == "success":
            exit_price = self.ltp if self.ltp is not None else 0.0
            if exit_price <= 0.0:
                quotes_resp = self.client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
                if isinstance(quotes_resp, dict) and quotes_resp.get("status") == "success":
                    exit_price = float(quotes_resp.get("data", {}).get("ltp", 0.0))
            logger.info(f"Exit order successful at {exit_price}")
            self.send_whatsapp_notification("SELL", f"success ({reason})", exit_price)
            self.position = None
            self.entry_price = 0.0
            self.entry_atr = 0.0
        else:
            logger.error(f"Exit order failed: {response}")
            self.send_whatsapp_notification("SELL", "failed", 0.0)

    def check_signals(self):
        now = datetime.now()
        current_date = now.date()

        # Reset daily trade limit at midnight or new day
        if self.last_trade_date != current_date:
            self.daily_trade_taken = False

        # Intraday force exit condition at 15:15
        is_exit_time = now.hour == 15 and now.minute >= 15

        if self.position is not None:
            # We are currently in a position. Check risk management stops/targets.
            current_price = self.ltp if self.ltp is not None else 0.0
            if current_price <= 0.0:
                return

            exit_triggered = False
            exit_reason = ""

            # Check fixed Stop Loss & Take Profit
            tp_price = self.entry_price * (1 + TAKE_PROFIT_PCT / 100)
            sl_price = self.entry_price * (1 - STOP_LOSS_PCT / 100)

            # Check ATR Stop Loss (if enabled)
            atr_sl_price = 0.0
            if ATR_MULT_SL > 0 and self.entry_atr > 0:
                atr_sl_price = self.entry_price - (ATR_MULT_SL * self.entry_atr)

            if current_price >= tp_price:
                logger.info(f"Take Profit hit: {current_price} >= {tp_price}")
                exit_triggered = True
                exit_reason = "take_profit"
            elif current_price <= sl_price:
                logger.info(f"Stop Loss hit: {current_price} <= {sl_price}")
                exit_triggered = True
                exit_reason = "stop_loss"
            elif atr_sl_price > 0 and current_price <= atr_sl_price:
                logger.info(f"ATR Stop Loss hit: {current_price} <= {atr_sl_price}")
                exit_triggered = True
                exit_reason = "atr_stop_loss"

            # Check Technical Exit Signals from Indicators
            if not exit_triggered:
                df = self.get_historical_data()
                if not df.empty and len(df) >= max(EMA_SLOW, RSI_PERIOD, MACD_SLOW) + 1:
                    # Append live price
                    df.loc[df.index[-1], "close"] = current_price

                    df["ema_fast"] = compute_ema(df["close"], EMA_FAST)
                    df["ema_slow"] = compute_ema(df["close"], EMA_SLOW)
                    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
                    macd_line, macd_sig, macd_hist = compute_macd(
                        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
                    )

                    # Signal exit states
                    ema_bear = (df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]) and (
                        df["ema_fast"].iloc[-2] >= df["ema_slow"].iloc[-2]
                    )
                    rsi_ob = df["rsi"].iloc[-1] > RSI_EXIT_HI
                    macd_fail = (macd_hist.iloc[-1] < 0) and (macd_hist.iloc[-2] >= 0)
                    below_slow = df["close"].iloc[-1] < df["ema_slow"].iloc[-1]

                    if ema_bear or rsi_ob or macd_fail or below_slow:
                        exit_triggered = True
                        exit_reason = "technical_exit"
                        logger.info(
                            f"Technical exit signal triggered: DeathCross={ema_bear}, RSI={df['rsi'].iloc[-1]:.1f}, MACDFail={macd_fail}, BelowSlow={below_slow}"
                        )

            if is_exit_time and not exit_triggered:
                logger.info("Forced daily exit time reached (15:15).")
                exit_triggered = True
                exit_reason = "end_of_day"

            if exit_triggered:
                self.place_exit_order(reason=exit_reason)

        else:
            # Flat position. Check for entries.
            if not self.daily_trade_taken and not is_exit_time:
                df = self.get_historical_data()
                if df.empty or len(df) < max(EMA_SLOW, RSI_PERIOD, MACD_SLOW) + 2:
                    return

                # Append live LTP as the latest closing price
                if self.ltp is not None:
                    df.loc[df.index[-1], "close"] = self.ltp

                df["ema_fast"] = compute_ema(df["close"], EMA_FAST)
                df["ema_slow"] = compute_ema(df["close"], EMA_SLOW)
                df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
                macd_line, macd_sig, macd_hist = compute_macd(
                    df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
                )

                # Entry indicators
                ema_bull = df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]
                ema_cross = (df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]) and (
                    df["ema_fast"].iloc[-2] <= df["ema_slow"].iloc[-2]
                )
                macd_turn = (macd_hist.iloc[-1] > 0) and (macd_hist.iloc[-2] <= 0)
                rsi_zone = RSI_ENTRY_LO < df["rsi"].iloc[-1] < RSI_ENTRY_HI

                signal_trigger = ema_cross or macd_turn

                if signal_trigger and ema_bull and rsi_zone:
                    logger.info(
                        f"Long Entry Triggered: Cross={ema_cross}, MACDTurn={macd_turn}, RSI={df['rsi'].iloc[-1]:.1f}"
                    )
                    self.place_entry_order()

    def run(self):
        ws_thread = threading.Thread(target=self.websocket_thread, daemon=True)
        ws_thread.start()
        time.sleep(2)  # Wait for WebSocket handshake

        logger.info("Execution loop started.")
        try:
            while self.running:
                self.check_signals()
                time.sleep(SIGNAL_CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Bot execution stopped by user.")
        finally:
            self.stop_event.set()
            self.running = False


if __name__ == "__main__":
    bot = OpenAlgoStrategyBot()
    bot.run()
