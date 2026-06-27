"""
MCX VWAP + RSI Pullback Intraday Strategy Bot for OpenAlgo
---------------------------------------------------------
This script can be executed standalone or uploaded directly to the OpenAlgo Strategy Runner.
It reads API connections and trading parameters from environment variables.
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
logger = logging.getLogger("VWAP_RSI_Bot")

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

# Trade Parameters
SYMBOL = os.getenv("SYMBOL", "MCX")
EXCHANGE = os.getenv("EXCHANGE", "NSE")
QUANTITY = int(os.getenv("QUANTITY", "1"))
PRODUCT = os.getenv("PRODUCT", "MIS")  # Intraday
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "15m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "3"))
SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", "15"))

# Strategy Configuration
RSI_PERIOD = 14
RSI_LONG_THRESHOLD = 40.0
RSI_SHORT_THRESHOLD = 60.0
TAKE_PROFIT_PCT = 1.5
STOP_LOSS_PCT = 2.0

# ==============================================================================
# INDICATOR FUNCTIONS
# ==============================================================================


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["pv"] = df["close"] * df["volume"]

    cum_pv = df.groupby("date")["pv"].cumsum()
    cum_vol = df.groupby("date")["volume"].cumsum()

    return cum_pv / cum_vol.replace(0, np.nan)


# ==============================================================================
# STRATEGY EXECUTION ENGINE
# ==============================================================================


class OpenAlgoStrategyBot:
    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)
        self.strategy_name = os.getenv("STRATEGY_NAME", "MCX_VWAP_RSI_Intraday")
        self.position = None  # "BUY", "SHORT", or None
        self.entry_price = 0.0
        self.ltp = None
        self.running = True
        self.stop_event = threading.Event()
        self.instrument = [{"exchange": EXCHANGE, "symbol": SYMBOL}]
        self.daily_trade_taken = False
        self.last_trade_date = None

        logger.info(
            f"Initialized Strategy Bot for {SYMBOL}:{EXCHANGE} (Qty: {QUANTITY}, Product: {PRODUCT})"
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
            if history_data and isinstance(history_data, list):
                df = pd.DataFrame(history_data)
                # Parse numeric columns
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()

    def check_funds_before_order(self) -> bool:
        """Verify available balance against estimated order value."""
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
                    f"Available Balance: {available_balance:.2f}, Est. Cost: {estimated_cost:.2f}"
                )

                if available_balance < estimated_cost:
                    logger.warning(
                        f"Insufficient funds! Needed: {estimated_cost:.2f}, Available: {available_balance:.2f}"
                    )
                    return False
                return True
            else:
                logger.warning("Could not fetch funds info to verify. Proceeding anyway.")
                return True
        except Exception as e:
            logger.error(f"Error checking funds: {e}")
            return True

    def send_whatsapp_notification(self, action: str, status: str, price: float = 0.0):
        url = f"{API_HOST}/api/v1/whatsapp/notify"

        msg = (
            f"[OPENALGO BOT]\n"
            f"Strategy: {self.strategy_name}\n"
            f"Action: {action}\n"
            f"Status: {status}\n"
            f"Symbol: {SYMBOL}\n"
            f"Quantity: {QUANTITY}\n"
            f"Price: {price:.2f}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        payload = {"apikey": API_KEY, "self": True, "message": msg}

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
            logger.warning(f"Failed to send WhatsApp notification: {e}")

    def place_entry_order(self, action: str):
        if not self.check_funds_before_order():
            logger.warning("Aborting entry order due to insufficient funds.")
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
            logger.info(f"Entry order successful. Entry Price: {self.entry_price}")
            self.daily_trade_taken = True
            self.last_trade_date = datetime.now().date()
            self.send_whatsapp_notification(action, "success", self.entry_price)
        else:
            logger.error(f"Entry order failed: {response}")
            self.send_whatsapp_notification(action, "failed", 0.0)

    def place_exit_order(self):
        exit_action = "SELL" if self.position == "BUY" else "BUY"
        logger.info(f"Placing exit {exit_action} order for {QUANTITY} shares of {SYMBOL}...")
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
            logger.info(f"Exit order successful at {exit_price}")
            self.send_whatsapp_notification(exit_action, "success", exit_price)
            self.position = None
            self.entry_price = 0.0
        else:
            logger.error(f"Exit order failed: {response}")
            self.send_whatsapp_notification(exit_action, "failed", 0.0)

    def check_signals(self):
        now = datetime.now()
        current_date = now.date()

        # Reset daily trade limit at midnight or new day
        if self.last_trade_date != current_date:
            self.daily_trade_taken = False

        # Intraday force exit condition at 15:15
        is_exit_time = now.hour == 15 and now.minute >= 15

        if self.position is not None:
            # We are currently in a position. Check for exit targets.
            current_price = self.ltp if self.ltp is not None else 0.0
            if current_price <= 0.0:
                return

            exit_triggered = False

            if self.position == "BUY":
                tp_price = self.entry_price * (1 + TAKE_PROFIT_PCT / 100)
                sl_price = self.entry_price * (1 - STOP_LOSS_PCT / 100)
                if current_price >= tp_price:
                    logger.info(f"Take Profit hit: {current_price} >= {tp_price}")
                    exit_triggered = True
                elif current_price <= sl_price:
                    logger.info(f"Stop Loss hit: {current_price} <= {sl_price}")
                    exit_triggered = True
            elif self.position == "SHORT":
                tp_price = self.entry_price * (1 - TAKE_PROFIT_PCT / 100)
                sl_price = self.entry_price * (1 + STOP_LOSS_PCT / 100)
                if current_price <= tp_price:
                    logger.info(f"Take Profit hit: {current_price} <= {tp_price}")
                    exit_triggered = True
                elif current_price >= sl_price:
                    logger.info(f"Stop Loss hit: {current_price} >= {sl_price}")
                    exit_triggered = True

            if is_exit_time and not exit_triggered:
                logger.info("Forced daily exit time reached (15:15).")
                exit_triggered = True

            if exit_triggered:
                self.place_exit_order()

        else:
            # Flat position. Check for entries if daily limit is not exceeded and time is before 15:15
            if not self.daily_trade_taken and not is_exit_time:
                df = self.get_historical_data()
                if df.empty or len(df) < RSI_PERIOD + 1:
                    return

                # Append live LTP as the latest closing price
                if self.ltp is not None:
                    df.loc[df.index[-1], "close"] = self.ltp

                df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
                df["vwap"] = compute_vwap(df)

                latest_close = df["close"].iloc[-1]
                latest_rsi = df["rsi"].iloc[-1]
                latest_vwap = df["vwap"].iloc[-1]

                if pd.isna(latest_vwap) or pd.isna(latest_rsi):
                    return

                # Check Entry Rules
                if latest_close < latest_vwap and latest_rsi < RSI_LONG_THRESHOLD:
                    logger.info(
                        f"Long Entry Triggered: Close {latest_close:.2f} < VWAP {latest_vwap:.2f} & RSI {latest_rsi:.1f} < {RSI_LONG_THRESHOLD}"
                    )
                    self.place_entry_order("BUY")
                elif latest_close > latest_vwap and latest_rsi > RSI_SHORT_THRESHOLD:
                    logger.info(
                        f"Short Entry Triggered: Close {latest_close:.2f} > VWAP {latest_vwap:.2f} & RSI {latest_rsi:.1f} > {RSI_SHORT_THRESHOLD}"
                    )
                    self.place_entry_order("SHORT")

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
