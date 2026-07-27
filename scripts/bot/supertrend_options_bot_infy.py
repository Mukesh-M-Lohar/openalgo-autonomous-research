"""
Standalone Bot: Supertrend Options Touch Bot for Infosys (INFY)
===================================================================
Rank 6 Stock: Infosys (INFY)
Reason: Strong IT sector moves

This production bot executes live/paper options trading based on Supertrend band touch signals:
- Subscribes to OpenAlgo WebSocket streaming feed for real-time underlying price monitoring (NSE:INFY).
- Calculates Supertrend on completed underlying stock candles (5m) to avoid lookahead bias.
- Auto-detects stock lot size from OpenAlgo NFO instrument master.
- Selects nearest-expiry ITM CE/PE contracts dynamically based on underlying price & strike offset.
- Evaluates volume and Open Interest across candidate strikes to pick the most liquid option contract.
- Trailing stop-loss protection (10%), stock Supertrend direction flip exit, and intraday time filters.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Core Indicator Functions (No external dependencies)
# ----------------------------------------------------------------------


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used by ATR)."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range calculation."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return wilder_smooth(tr, period)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Returns df with columns: st_upperband, st_lowerband, st_trend (1=up, -1=down),
    and st_line (the plotted supertrend line itself).
    """
    high, low, close = df["high"], df["low"], df["close"]
    hl2 = (high + low) / 2
    tr_atr = atr(df, period)

    basic_upper = hl2 + multiplier * tr_atr
    basic_lower = hl2 - multiplier * tr_atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        prev_final_upper = final_upper.iloc[i - 1]
        prev_final_lower = final_lower.iloc[i - 1]

        if pd.isna(prev_final_upper):
            final_upper.iloc[i] = basic_upper.iloc[i]
        elif basic_upper.iloc[i] < prev_final_upper or close.iloc[i - 1] > prev_final_upper:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_final_upper

        if pd.isna(prev_final_lower):
            final_lower.iloc[i] = basic_lower.iloc[i]
        elif basic_lower.iloc[i] > prev_final_lower or close.iloc[i - 1] < prev_final_lower:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_final_lower

        if pd.isna(prev_final_upper) or pd.isna(prev_final_lower):
            trend.iloc[i] = trend.iloc[i - 1]
        elif close.iloc[i] > prev_final_upper:
            trend.iloc[i] = 1
        elif close.iloc[i] < prev_final_lower:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
            if trend.iloc[i] == 1 and final_lower.iloc[i] < prev_final_lower:
                final_lower.iloc[i] = prev_final_lower
            if trend.iloc[i] == -1 and final_upper.iloc[i] > prev_final_upper:
                final_upper.iloc[i] = prev_final_upper

    st_line = np.where(trend == 1, final_lower, final_upper)

    out = df.copy()
    out["st_upperband"] = final_upper
    out["st_lowerband"] = final_lower
    out["st_trend"] = trend
    out["st_line"] = st_line
    return out


# --- SDK Import with guard ---
try:
    from openalgo import api, ta
except ImportError:
    print("ERROR: Install the SDK first:  pip install openalgo")
    sys.exit(1)

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SupertrendOptionsBot_INFY")

# --- Configuration ---
API_KEY = os.getenv(
    "OPENALGO_API_KEY", "b45feb0a6973ed00fe86d25ace49d4da8dfe8d0a78c334455d46254ded28a26d"
)
API_HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_ENDPOINT = "ws://127.0.0.1:8765"

# Fixed Strategy Parameters
UNDERLYING_SYMBOL = "INFY"
EXCHANGE = "NSE"
TIMEFRAME = "5m"
ST_PERIOD = 10
ST_MULT = 3.0
DEFAULT_STRIKE_STEP = 20
STRIKE_STEPS_ITM = 1  # 1 step ITM
LOTS = 1  # Number of option lots
PRODUCT = "MIS"
AVOID_0DTE = False
STRATEGY_NAME = "SupertrendOptionsBot_INFY"

# Time of Day Filter (IST)
ALLOWED_START_HOUR = 9
ALLOWED_END_HOUR = 14

# Exits
TRAIL_SL_PCT = 10.0  # Trailing stop percentage (0 to disable)
TAKE_PROFIT_PCT = None  # None or float

# Touch Detection
TOUCH_PCT = 0.07  # How close to band to count as touch (%)

# ADX Filter
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0

# Moving Average Filter
MA_PERIOD = 150  # Default 150 for approx 2 days of 5m candles

# Option Supertrend (for analysis logging, matches backtester)
OPTION_TIMEFRAME = "3m"
OPTION_ST_PERIOD = 10
OPTION_ST_MULT = 3.0

# Cache Lookback Window
LOOKBACK_BARS = max(200, MA_PERIOD + 50)  # Number of historical bars to keep in memory

# WebSocket Data Feed Toggle
USE_WEBSOCKET = True

# Daily Trade Limit
MAX_TRADES_PER_DAY = 3

# Re-entry Cooldown (seconds) after an exit
REENTRY_COOLDOWN_SECONDS = 300

# EOD Square-off time (IST)
EOD_SQUAREOFF_HOUR = 15
EOD_SQUAREOFF_MINUTE = 15

# --- IST Timezone ---
IST = timezone(timedelta(hours=5, minutes=30))


def get_now_ist() -> datetime:
    """Returns the current date and time localized in Indian Standard Time (IST)."""
    return datetime.now(timezone.utc).astimezone(IST)


class SupertrendStockOptionsBot:
    def __init__(self):
        logger.info("Initializing Supertrend Options Bot for Infosys (INFY)...")
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_ENDPOINT)
        self.history_cache = pd.DataFrame()
        self.active_position = None
        self.active_direction = None
        self.max_favorable_price = 0.0
        self.current_sl_price = 0.0
        self.entry_price = 0.0
        self.use_websocket = USE_WEBSOCKET

        # Dynamic Lot size & calculated trade quantity
        self.lot_size = None
        self.trade_quantity = None

        # Dynamic strike step
        self.strike_step = DEFAULT_STRIKE_STEP

        # Candle-boundary throttle: track last processed bar timestamp
        self._last_processed_bar_ts = None

        # Daily trade limit tracking
        self._daily_trade_count = 0
        self._trade_date = get_now_ist().date()

        # Re-entry cooldown tracking
        self._last_exit_time = 0.0

        # Instrument cache (loaded once at startup)
        self._instruments_cache = None

        self._prime_cache()
        self._cache_instruments()
        self._init_websocket()

    def _cache_instruments(self):
        """Fetch and cache the NFO instrument master once at startup."""
        logger.info("Fetching NFO instrument master (one-time cache)...")
        try:
            instruments = self.client.instruments(exchange="NFO")
            if isinstance(instruments, pd.DataFrame) and not instruments.empty:
                self._instruments_cache = instruments
                logger.info(f"Instrument master cached: {len(self._instruments_cache)} records.")

                # Auto-detect Lot Size for UNDERLYING_SYMBOL
                df_opts = self._instruments_cache[
                    (self._instruments_cache["name"] == UNDERLYING_SYMBOL)
                    & (self._instruments_cache["instrumenttype"].isin(["CE", "PE"]))
                ]
                if not df_opts.empty:
                    if "lotsize" in df_opts.columns:
                        self.lot_size = int(df_opts["lotsize"].iloc[0])
                    elif "lot_size" in df_opts.columns:
                        self.lot_size = int(df_opts["lot_size"].iloc[0])

                    if self.lot_size:
                        self.trade_quantity = self.lot_size * LOTS
                        logger.info(
                            f"Detected Lot Size for {UNDERLYING_SYMBOL}: {self.lot_size}. Trade Qty ({LOTS} lot): {self.trade_quantity}"
                        )

                    # Auto-detect strike step if possible
                    if "strike" in df_opts.columns:
                        strikes = sorted(df_opts["strike"].dropna().unique())
                        if len(strikes) > 1:
                            diffs = np.diff(strikes)
                            pos_diffs = diffs[diffs > 0]
                            if len(pos_diffs) > 0:
                                detected_step = float(np.min(pos_diffs))
                                if detected_step > 0:
                                    self.strike_step = detected_step
                                    logger.info(
                                        f"Auto-detected strike step for {UNDERLYING_SYMBOL}: {self.strike_step}"
                                    )
            else:
                logger.warning("Instrument master returned empty. Will retry on first signal.")
        except Exception as e:
            logger.error(f"Failed to fetch instrument master: {e}")

    def _init_websocket(self):
        """Initialize WebSocket connection and subscribe to underlying stock."""
        if not self.use_websocket:
            logger.info("WebSocket streaming is disabled in configuration (using REST polling).")
            return

        try:
            logger.info("Connecting to OpenAlgo WebSocket feed...")
            self.client.connect()
            self.client.subscribe_ltp(
                [{"exchange": EXCHANGE, "symbol": UNDERLYING_SYMBOL}],
                on_data_received=self._on_ws_price_update,
            )
            logger.info(f"✅ WebSocket connected and subscribed to {EXCHANGE}:{UNDERLYING_SYMBOL}")
        except Exception as e:
            logger.warning(f"⚠️ WebSocket initialization failed, falling back to REST: {e}")
            self.use_websocket = False

    def _on_ws_price_update(self, data):
        """Callback for WebSocket LTP updates. Logs at debug level."""
        logging.debug(f"WS price update: {data}")

    def get_live_ltp(self, exchange, symbol):
        """Fetch live LTP from WebSocket cache if connected; fall back to REST quotes API."""
        if self.use_websocket:
            try:
                res = self.client.get_ltp(exchange, symbol)
                if isinstance(res, dict):
                    if "ltp" in res and exchange in res["ltp"] and symbol in res["ltp"][exchange]:
                        price = float(res["ltp"][exchange][symbol].get("ltp", 0))
                        if price > 0:
                            return price
            except Exception as e:
                logger.warning(f"Failed to fetch WS LTP for {exchange}:{symbol}: {e}")

        # Fallback to REST Quote API
        if not hasattr(self, "_rest_ltp_cache"):
            self._rest_ltp_cache = {}

        now = time.time()
        cache_key = f"{exchange}:{symbol}"
        if cache_key in self._rest_ltp_cache:
            last_time, last_price = self._rest_ltp_cache[cache_key]
            if (now - last_time) < 1.0:
                return last_price

        try:
            q_resp = self.client.quotes(symbol=symbol, exchange=exchange)
            if q_resp and q_resp.get("status") == "success":
                data = q_resp.get("data", {})
                price = float(data.get("ltp", 0))
                self._rest_ltp_cache[cache_key] = (now, price)
                return price
        except Exception as e:
            logger.warning(f"REST quote fallback failed for {exchange}:{symbol}: {e}")

        return 0.0

    def _parse_history_df(self, hist):
        """Parse the history() response into a clean DataFrame."""
        if not isinstance(hist, pd.DataFrame):
            raise ValueError(f"Unexpected history response type: {type(hist)}")

        if hist.empty:
            return hist

        df = hist.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")

        df = df.sort_index()
        return df

    def _prime_cache(self):
        """Fetch initial history to prime the Supertrend calculation."""
        logger.info(f"Fetching last {LOOKBACK_BARS} bars of {UNDERLYING_SYMBOL} to prime cache...")

        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            hist = self.client.history(
                symbol=UNDERLYING_SYMBOL,
                exchange=EXCHANGE,
                interval=TIMEFRAME,
                start_date=start_date,
                end_date=end_date,
            )
            df = self._parse_history_df(hist)
        except Exception as e:
            raise RuntimeError(f"Failed to prime cache: {e}") from e

        if df.empty:
            raise RuntimeError("History returned empty DataFrame — cannot prime cache.")

        self.history_cache = df.tail(LOOKBACK_BARS)
        logger.info(f"Cache primed with {len(self.history_cache)} bars.")

    def update_cache_and_calc(self):
        """Fetch the latest bars, update rolling cache, and calc Supertrend."""
        now = time.time()
        if hasattr(self, "_last_hist_fetch_time") and (now - self._last_hist_fetch_time) < 60:
            if hasattr(self, "_last_hist_df") and hasattr(self, "_last_hist_st"):
                return self._last_hist_df, self._last_hist_st

        start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            hist = self.client.history(
                symbol=UNDERLYING_SYMBOL,
                exchange=EXCHANGE,
                interval=TIMEFRAME,
                start_date=start_date,
                end_date=end_date,
            )
            df = self._parse_history_df(hist)
        except Exception as e:
            logger.warning(f"Failed to fetch latest data: {e}")
            return None, None

        if df.empty:
            logger.warning("Latest history fetch returned empty DataFrame.")
            return None, None

        combined = pd.concat([self.history_cache, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        self.history_cache = combined.tail(LOOKBACK_BARS)

        df_calc = self.history_cache.copy()
        st = supertrend(df_calc, period=ST_PERIOD, multiplier=ST_MULT)
        if st is None or st.empty:
            return None, None

        self._last_hist_fetch_time = time.time()
        self._last_hist_df = df_calc
        self._last_hist_st = st[["st_line", "st_trend"]]

        return self._last_hist_df, self._last_hist_st

    def get_option_contract(self, underlying_price, direction):
        """Find the correct CE/PE contract at the configured strike step offset."""
        logger.info(
            f"Looking up option contract for {UNDERLYING_SYMBOL} at {underlying_price} ({direction})..."
        )

        if self._instruments_cache is None or self._instruments_cache.empty:
            self._cache_instruments()
        if self._instruments_cache is None or self._instruments_cache.empty:
            logger.error("No instrument data available.")
            return None

        df_inst = self._instruments_cache
        df_opts = df_inst[
            (df_inst["name"] == UNDERLYING_SYMBOL) & (df_inst["instrumenttype"].isin(["CE", "PE"]))
        ].copy()
        df_opts["expiry_date"] = pd.to_datetime(df_opts["expiry"])

        if AVOID_0DTE:
            future_opts = df_opts[df_opts["expiry_date"] > pd.Timestamp.today().normalize()]
        else:
            future_opts = df_opts[df_opts["expiry_date"] >= pd.Timestamp.today().normalize()]

        if future_opts.empty:
            return None

        nearest_expiry = future_opts["expiry_date"].min()
        valid_opts = future_opts[future_opts["expiry_date"] == nearest_expiry]

        # Calculate Target Strike based on strike step
        step = self.strike_step
        base_strike = int(round(underlying_price / step)) * step
        strike_offset = STRIKE_STEPS_ITM * step

        if direction == "UP":
            target_strike = base_strike - strike_offset  # CE ITM
            opt_type = "CE"
        else:
            target_strike = base_strike + strike_offset  # PE ITM
            opt_type = "PE"

        strikes_to_check = [target_strike - step, target_strike, target_strike + step]

        candidates = valid_opts[
            (valid_opts["strike"].isin(strikes_to_check))
            & (valid_opts["instrumenttype"] == opt_type)
        ]

        if candidates.empty:
            return None

        best_symbol = None
        best_score = -1

        for _, row in candidates.iterrows():
            sym = row["symbol"]
            try:
                q = self.client.quotes(symbol=sym, exchange="NFO")
                if isinstance(q, dict) and q.get("status") == "success":
                    data = q.get("data", {})
                    if not isinstance(data, dict):
                        continue
                    vol = float(data.get("volume", 0))
                    oi = float(data.get("oi", 0))
                    score = vol + oi

                    logger.info(f"  -> Candidate {sym}: Volume={vol}, OI={oi}")

                    if score > best_score:
                        best_score = score
                        best_symbol = sym
            except Exception as e:
                logger.warning(f"  -> Failed to quote {sym}: {e}")

        if best_symbol:
            logger.info(f"✅ Selected Most Liquid Contract: {best_symbol} (Score: {best_score})")
            return best_symbol

        return None

    def _get_fill_price(self, order_id, fallback_exchange, fallback_symbol):
        """Attempt to get the actual fill price from orderstatus; fall back to LTP."""
        if order_id:
            try:
                status_resp = self.client.orderstatus(order_id=order_id)
                if isinstance(status_resp, dict) and status_resp.get("status") == "success":
                    avg_price = float(status_resp.get("data", {}).get("average_price", 0))
                    if avg_price > 0:
                        return avg_price
            except Exception as e:
                logger.warning(f"Failed to fetch fill price from orderstatus: {e}")

        return self.get_live_ltp(fallback_exchange, fallback_symbol)

    def _check_option_supertrend(self, symbol) -> bool:
        """Fetch recent option history, calculate Supertrend, and verify it is UP."""
        try:
            start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

            opt_hist = self.client.history(
                symbol=symbol,
                exchange="NFO",
                interval=OPTION_TIMEFRAME,
                start_date=start_date,
                end_date=end_date,
            )
            opt_df = self._parse_history_df(opt_hist)

            if opt_df.empty or len(opt_df) <= OPTION_ST_PERIOD:
                logger.info(f"Option ST Filter: Not enough bars for {symbol} ({len(opt_df)} bars)")
                return False

            opt_st = supertrend(opt_df, period=OPTION_ST_PERIOD, multiplier=OPTION_ST_MULT)
            if opt_st is None or opt_st.empty:
                return False

            latest_dir = opt_st["st_trend"].iloc[-1]

            if latest_dir == 1:
                logger.info(f"✅ Option ST Filter Passed for {symbol}: Trend is UP")
                return True
            else:
                logger.info(f"🚫 Option ST Filter Failed for {symbol}: Trend is DOWN/UNKNOWN")
                return False

        except Exception as e:
            logger.warning(f"Option ST calculation failed for {symbol}: {e}")
            return False

    def execute_trade(self, symbol, direction="UP"):
        """Place Market Order and initialize position tracking."""
        if self.active_position:
            logger.warning(
                f"Ignoring entry signal for {symbol} — already in active trade: "
                f"{self.active_position} ({self.active_direction})"
            )
            return

        logger.info(f"Checking Option Supertrend filter for {symbol}...")
        if not self._check_option_supertrend(symbol):
            logger.warning(
                f"Skipping trade for {symbol} — Option ST is not UP. Entering 60s cooldown."
            )
            self._failed_entry_cooldown_until = time.time() + 60
            return

        # Determine quantity
        order_qty = self.trade_quantity if self.trade_quantity else 100

        logger.info(f"EXECUTING ENTRY FOR {symbol} ({direction}) Qty: {order_qty}")
        try:
            resp = self.client.placeorder(
                symbol=symbol,
                action="BUY",
                exchange="NFO",
                price_type="MARKET",
                product=PRODUCT,
                quantity=order_qty,
                strategy=STRATEGY_NAME,
            )
            logger.info(f"Order Response: {resp}")

            if not isinstance(resp, dict) or resp.get("status") != "success":
                logger.error(f"Entry order REJECTED or FAILED: {resp}")
                return

            order_id = resp.get("orderid")

            if self.use_websocket:
                try:
                    self.client.subscribe_ltp([{"exchange": "NFO", "symbol": symbol}])
                    logger.info(f"Subscribed WebSocket LTP feed for active option {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to subscribe WS for {symbol}: {e}")

            time.sleep(0.5)
            self.entry_price = self._get_fill_price(order_id, "NFO", symbol)
            self.max_favorable_price = self.entry_price

            if TRAIL_SL_PCT > 0 and self.entry_price > 0:
                self.current_sl_price = self.entry_price * (1 - (TRAIL_SL_PCT / 100.0))

            self.active_position = symbol
            self.active_direction = direction

            self._daily_trade_count += 1

            logger.info(
                f"✅ Entered {symbol} ({direction}) at {self.entry_price:.2f}. "
                f"SL set to {self.current_sl_price:.2f}. "
                f"Trade #{self._daily_trade_count} today."
            )

        except Exception as e:
            logger.error(f"Execution failed: {e}")

    def exit_trade(self, reason):
        """Exit the current active position."""
        logger.info(f"EXECUTING EXIT: {reason}")
        symbol_to_exit = self.active_position
        order_qty = self.trade_quantity if self.trade_quantity else 100
        try:
            resp = self.client.placeorder(
                symbol=symbol_to_exit,
                action="SELL",
                exchange="NFO",
                price_type="MARKET",
                product=PRODUCT,
                quantity=order_qty,
                strategy=STRATEGY_NAME,
            )
            logger.info(f"Exit Order Response: {resp}")

            if self.use_websocket and symbol_to_exit:
                try:
                    self.client.unsubscribe_ltp([{"exchange": "NFO", "symbol": symbol_to_exit}])
                    logger.info(f"Unsubscribed WebSocket LTP feed for {symbol_to_exit}")
                except Exception as e:
                    logger.warning(f"Failed to unsubscribe WS for {symbol_to_exit}: {e}")

            self.active_position = None
            self.active_direction = None
            self._last_exit_time = time.time()
        except Exception as e:
            logger.error(f"Exit failed: {e}")

    def _is_eod_squareoff_time(self) -> bool:
        """Check if we've passed the EOD square-off deadline (default 15:15 IST)."""
        now_ist = get_now_ist()
        return now_ist.time() >= dt_time(EOD_SQUAREOFF_HOUR, EOD_SQUAREOFF_MINUTE)

    def _is_new_candle(self) -> bool:
        """Check if the latest bar in the cache is a new candle we haven't processed yet."""
        if self.history_cache.empty:
            return False
        latest_ts = self.history_cache.index[-1]
        if self._last_processed_bar_ts is None or latest_ts != self._last_processed_bar_ts:
            self._last_processed_bar_ts = latest_ts
            return True
        return False

    def manage_position(self):
        """Monitor live price via WebSocket/REST and manage Trailing Stop / Take Profit / Supertrend Flip."""
        if not self.active_position:
            return

        if PRODUCT == "MIS" and self._is_eod_squareoff_time():
            logger.info(
                f"EOD square-off time reached ({EOD_SQUAREOFF_HOUR}:{EOD_SQUAREOFF_MINUTE:02d} IST). "
                "Force-exiting active position."
            )
            self.exit_trade("EOD_SQUAREOFF")
            return

        try:
            current_price = self.get_live_ltp("NFO", self.active_position)
            if current_price <= 0:
                return

            # 1. Update Trailing Stop
            if current_price > self.max_favorable_price:
                self.max_favorable_price = current_price
                if TRAIL_SL_PCT > 0:
                    new_sl = self.max_favorable_price * (1 - (TRAIL_SL_PCT / 100.0))
                    if new_sl > self.current_sl_price:
                        self.current_sl_price = new_sl
                        logger.info(f"Trailing SL moved up to: {self.current_sl_price:.2f}")

            # 2. Check Stop Loss Hit
            if TRAIL_SL_PCT > 0 and current_price <= self.current_sl_price:
                logger.info(f"Price {current_price} crossed SL {self.current_sl_price:.2f}")
                self.exit_trade("TRAILING_STOP_HIT")
                return

            # 3. Check Take Profit Hit
            if TAKE_PROFIT_PCT is not None and TAKE_PROFIT_PCT > 0:
                tp_target = self.entry_price * (1 + (TAKE_PROFIT_PCT / 100.0))
                if current_price >= tp_target:
                    logger.info(f"Price {current_price} hit TP {tp_target:.2f}")
                    self.exit_trade("TAKE_PROFIT_HIT")
                    return

            # 4. Check Underlying Stock Supertrend Flip (only on new candle boundaries)
            if self.active_direction:
                df, st = self.update_cache_and_calc()
                if (
                    df is not None
                    and not df.empty
                    and st is not None
                    and not st.empty
                    and len(st) >= 2
                    and self._is_new_candle()
                ):
                    prev_st = st.iloc[-2]
                    st_dir = prev_st.iloc[1]
                    if self.active_direction == "UP" and st_dir == -1:
                        logger.info(
                            f"Underlying stock {UNDERLYING_SYMBOL} Supertrend flipped to Bearish (-1) while holding CE!"
                        )
                        self.exit_trade("STOCK_ST_FLIP")
                        return
                    elif self.active_direction == "DOWN" and st_dir == 1:
                        logger.info(
                            f"Underlying stock {UNDERLYING_SYMBOL} Supertrend flipped to Bullish (1) while holding PE!"
                        )
                        self.exit_trade("STOCK_ST_FLIP")
                        return

        except Exception as e:
            logger.error(f"Position management error: {e}")

    def _reset_daily_counters_if_needed(self):
        """Reset daily trade counter on a new trading day."""
        today = get_now_ist().date()
        if today != self._trade_date:
            self._trade_date = today
            self._daily_trade_count = 0
            logger.info(f"New trading day: {today}. Daily trade count reset.")

    def _shutdown(self):
        """Graceful shutdown: exit positions, unsubscribe, disconnect WebSocket."""
        logger.info("Initiating graceful shutdown...")

        if self.active_position:
            logger.warning("Active position detected during shutdown — exiting immediately.")
            self.exit_trade("BOT_SHUTDOWN")

        if self.use_websocket:
            try:
                self.client.unsubscribe_ltp([{"exchange": EXCHANGE, "symbol": UNDERLYING_SYMBOL}])
            except Exception:
                pass
            try:
                self.client.disconnect()
                logger.info("WebSocket disconnected.")
            except Exception:
                pass

        logger.info("Shutdown complete.")

    def run(self):
        logger.info(f"Bot for {UNDERLYING_SYMBOL} is now actively monitoring the market...")
        try:
            while True:
                try:
                    self._reset_daily_counters_if_needed()

                    if self.active_position:
                        self.manage_position()
                        time.sleep(0.5)
                        continue

                    if time.time() < getattr(self, "_failed_entry_cooldown_until", 0):
                        time.sleep(2)
                        continue

                    if self._is_eod_squareoff_time():
                        time.sleep(60)
                        continue

                    ist_now = get_now_ist()
                    current_ist_hour = ist_now.hour
                    if ALLOWED_START_HOUR is not None and ALLOWED_END_HOUR is not None:
                        if not (ALLOWED_START_HOUR <= current_ist_hour <= ALLOWED_END_HOUR):
                            time.sleep(60)
                            continue

                    if self._daily_trade_count >= MAX_TRADES_PER_DAY:
                        time.sleep(60)
                        continue

                    if (time.time() - self._last_exit_time) < REENTRY_COOLDOWN_SECONDS:
                        time.sleep(5)
                        continue

                    df, st = self.update_cache_and_calc()
                    if df is not None and not df.empty and len(df) >= 2:
                        prev_st = st.iloc[-2]
                        st_val = prev_st.iloc[0]
                        st_dir = prev_st.iloc[1]

                        live_ltp = self.get_live_ltp(EXCHANGE, UNDERLYING_SYMBOL)
                        if live_ltp == 0.0:
                            live_ltp = float(df.iloc[-1]["close"])

                        touch_detected = False
                        if live_ltp > 0:
                            active_band = st_val
                            dist_pct = abs(live_ltp - active_band) / live_ltp * 100
                            if dist_pct <= TOUCH_PCT:
                                if st_dir == 1 or st_dir == -1:
                                    touch_detected = True

                        if touch_detected:
                            direction = "UP" if st_dir == 1 else "DOWN"
                            logger.info(
                                f"[{ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}] Touch Detected for {UNDERLYING_SYMBOL}! "
                                f"Direction: {direction} | Live LTP: {live_ltp} | ST (Prev Bar): {st_val:.2f}"
                            )

                            filters_passed = True

                            # 1. ADX Filter
                            if len(df) > ADX_PERIOD:
                                _, _, adx_series = ta.adx(
                                    df["high"], df["low"], df["close"], period=ADX_PERIOD
                                )
                                current_adx = float(adx_series[-2])
                                if current_adx < ADX_THRESHOLD:
                                    logger.info(
                                        f"Skipping trade: ADX ({current_adx:.2f}) < Threshold ({ADX_THRESHOLD})"
                                    )
                                    filters_passed = False
                            else:
                                logger.warning(
                                    f"Not enough data for ADX calculation. Need > {ADX_PERIOD} bars, have {len(df)}"
                                )
                                filters_passed = False

                            # 2. Moving Average Filter
                            if len(df) > MA_PERIOD:
                                ma_series = ta.sma(df["close"], period=MA_PERIOD)
                                current_ma = float(ma_series[-2])
                                prev_close = float(df.iloc[-2]["close"])

                                if direction == "UP" and prev_close < current_ma:
                                    logger.info(
                                        f"Skipping trade: Direction is UP but Close ({prev_close}) < MA ({current_ma:.2f})"
                                    )
                                    filters_passed = False
                                elif direction == "DOWN" and prev_close > current_ma:
                                    logger.info(
                                        f"Skipping trade: Direction is DOWN but Close ({prev_close}) > MA ({current_ma:.2f})"
                                    )
                                    filters_passed = False
                            else:
                                logger.warning(
                                    f"Not enough data for MA calculation. Need > {MA_PERIOD} bars, have {len(df)}"
                                )
                                filters_passed = False

                            if not filters_passed:
                                self._failed_entry_cooldown_until = time.time() + 60
                                continue

                            contract = self.get_option_contract(live_ltp, direction)
                            if contract:
                                self.execute_trade(contract, direction=direction)
                            else:
                                logger.warning(
                                    "No valid option contract found. Entering 60s cooldown."
                                )
                                self._failed_entry_cooldown_until = time.time() + 60

                    time.sleep(2)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user (KeyboardInterrupt).")
        finally:
            self._shutdown()


if __name__ == "__main__":
    bot = SupertrendStockOptionsBot()
    bot.run()
