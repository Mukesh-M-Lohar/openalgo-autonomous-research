"""
Standalone Bot: Supertrend Options Touch Bot for NIFTY Index
===================================================================
Underlying Symbol: NIFTY
Exchange: NSE_INDEX
Reason: Benchmark index with maximum liquidity and tight option spreads.

This production bot executes live/paper options trading based on Supertrend band touch signals:
- Subscribes to OpenAlgo WebSocket streaming feed for real-time underlying price monitoring (NSE_INDEX:NIFTY).
- Calculates Supertrend on completed underlying index candles (5m) to avoid lookahead bias.
- Auto-detects NIFTY lot size from OpenAlgo NFO instrument master.
- Selects nearest-expiry ITM CE/PE contracts dynamically based on underlying price & strike offset.
- Evaluates volume and Open Interest across candidate strikes to pick the most liquid option contract.
- Trailing stop-loss protection (10%), index Supertrend direction flip exit, and intraday time filters.
"""

import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- OpenAlgo SDK Import ---
try:
    from openalgo import api, ta
except ImportError:
    print("ERROR: Install the SDK first:  pip install openalgo")
    sys.exit(1)

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SupertrendOptionsBot_NIFTY")

# ======================================================================
# CONFIGURATION & BOT CONSTANTS
# ======================================================================

# ----------------------------------------------------------------------
# 1. API Credentials & OpenAlgo Connection
# ----------------------------------------------------------------------
API_KEY = os.getenv(
    "OPENALGO_API_KEY", "b45feb0a6973ed00fe86d25ace49d4da8dfe8d0a78c334455d46254ded28a26d"
)
API_HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_ENDPOINT = os.getenv("OPENALGO_WS_URL", "ws://127.0.0.1:8765")

# ----------------------------------------------------------------------
# 2. Strategy & Underlying Instrument Parameters
# ----------------------------------------------------------------------
STRATEGY_NAME = "SupertrendOptionsBot_NIFTY"
UNDERLYING_SYMBOL = "NIFTY"
EXCHANGE = "NSE_INDEX"
TIMEFRAME = "5m"
ST_PERIOD = 10
ST_MULT = 3.0
DEFAULT_STRIKE_STEP = 50
STRIKE_STEPS_ITM = 2  # 2 steps ITM (100 points ITM for NIFTY)
LOTS = 1  # Number of option lots
PRODUCT = "MIS"  # Intraday order product type
AVOID_0DTE = False  # True = avoid same-day expiry contracts

# ----------------------------------------------------------------------
# 3. Intraday Time Filters (IST)
# ----------------------------------------------------------------------
USE_TIME_FILTER = True  # True = Enable time of day filter, False = Disable (trade anytime)
ALLOWED_START_HOUR = 9  # Start allowing entries at 09:00 IST
ALLOWED_END_HOUR = 14  # Stop taking new entries at 14:00 IST
EOD_SQUAREOFF_HOUR = 15  # Intraday square-off hour
EOD_SQUAREOFF_MINUTE = 15  # Intraday square-off minute (15:15 IST)
IST = timezone(timedelta(hours=5, minutes=30))

# ----------------------------------------------------------------------
# 4. Exit & Risk Management Parameters
# ----------------------------------------------------------------------
TRAIL_SL_PCT = 10.0  # Trailing stop percentage (0.0 to disable)
TAKE_PROFIT_PCT = None  # Take profit percentage (None or float)
MAX_TRADES_PER_DAY = 3  # Maximum allowed completed trades per day
REENTRY_COOLDOWN_SECONDS = 300  # Cooldown seconds after an exit (5 minutes)

# ----------------------------------------------------------------------
# 5. Signal & Band Touch Detection Parameters
# ----------------------------------------------------------------------
TOUCH_PCT = 0.07  # How close price must get to Supertrend band (%)

# ----------------------------------------------------------------------
# 6. Technical Indicator Filters (Toggle On / Off)
# ----------------------------------------------------------------------
USE_ADX_FILTER = False  # True = Enable ADX trend strength check, False = Disable
USE_MA_FILTER = False  # True = Enable Moving Average trend direction check, False = Disable
USE_OPTION_ST_FILTER = True  # True = Enable Option Contract Supertrend check, False = Disable
USE_VWAP_FILTER = False  # True = Enable Intraday VWAP trend check, False = Disable
USE_MVWAP_FILTER = False  # True = Enable Moving VWAP (Rolling) trend check, False = Disable

# ADX Settings
ADX_PERIOD = 14
ADX_THRESHOLD = 25.0

# Moving Average Settings
MA_PERIOD = 150  # 150 bars (~2 days of 5m candles)

# VWAP & MVWAP Settings
MVWAP_PERIOD = 14  # Rolling period for MVWAP

# Option Contract Supertrend Settings
OPTION_TIMEFRAME = "3m"
OPTION_ST_PERIOD = 10
OPTION_ST_MULT = 3.0

# ----------------------------------------------------------------------
# 7. WebSocket Data Feed Toggle
# ----------------------------------------------------------------------
USE_WEBSOCKET = True  # True = WebSocket streaming feed, False = REST polling

# ----------------------------------------------------------------------
# 8. Real-Time WebSocket Order Flow & Options Filters Configuration
# ----------------------------------------------------------------------
USE_WEBSOCKET_ORDERFLOW = True  # Enable WS 5-Level Market Depth & OBI filter
USE_LOCAL_GREEKS_FILTER = True  # Enable Local Black-Scholes IV & Delta filter
MIN_OPTION_DELTA = 0.40  # Minimum absolute Option Delta (e.g. >= 0.40)
MAX_OPTION_IV = 35.0  # Maximum Implied Volatility % (e.g. <= 35.0%)
MIN_ORDER_BOOK_IMBALANCE = 0.05  # Minimum Order Book Imbalance (+0.05 = buyer dominance)
MAX_BID_ASK_SPREAD = 2.0  # Maximum allowed Bid-Ask spread in Rupees
ALLOW_TRADE_ON_NO_FLOW = False  # Strict Mode: Skip trade if depth/flow data missing

# Open Interest (OI) & Liquidity Contract Selection Settings
MIN_OPTION_OI = 100000  # Minimum required Open Interest (1 Lakh contracts)
OI_WEIGHT = 0.70  # 70% weight to Open Interest (position stability & institutional backing)
VOLUME_WEIGHT = 0.30  # 30% weight to Volume (intraday execution speed)
DELTA_WEIGHT = 0.20  # 20% weight to Option Delta
OBI_WEIGHT = 0.15  # 15% weight to Order Book Imbalance

# ----------------------------------------------------------------------
# 9. System & Cache Settings
# ----------------------------------------------------------------------
LOOKBACK_BARS = max(200, MA_PERIOD + 50)  # Rolling bars cache window


# ======================================================================
# CORE INDICATOR FUNCTIONS & LOCAL GREEKS ENGINE
# ======================================================================


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


def vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday Volume Weighted Average Price (resets per daily trading session)."""
    high, low, close = df["high"], df["low"], df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume

    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.strftime("%Y-%m-%d")
        cum_tpv = tpv.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
    else:
        cum_tpv = tpv.cumsum()
        cum_vol = volume.cumsum()

    cum_vol = cum_vol.replace(0, np.nan)
    return cum_tpv / cum_vol


def mvwap(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Moving Volume Weighted Average Price (Rolling N-period SMA of VWAP)."""
    vwap_series = vwap(df)
    return vwap_series.rolling(window=period, min_periods=1).mean()


class LocalOptionGreeks:
    """Pure Python Standard Library Black-Scholes calculator for 0ms WebSocket latency."""

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Standard Normal Cumulative Distribution Function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        """Standard Normal Probability Density Function."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @classmethod
    def bs_price(
        cls, S: float, K: float, T: float, r: float, sigma: float, option_type: str = "CE"
    ) -> float:
        """Calculates Black-Scholes option price."""
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == "CE":
            return S * cls.norm_cdf(d1) - K * math.exp(-r * T) * cls.norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * cls.norm_cdf(-d2) - S * cls.norm_cdf(-d1)

    @classmethod
    def calculate_iv(
        cls, S: float, K: float, T: float, r: float, market_price: float, option_type: str = "CE"
    ) -> float:
        """Solves Implied Volatility (IV in %) using Newton-Raphson method."""
        if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
            return 0.0

        sigma = 0.20  # Initial guess 20%
        for _ in range(100):
            price = cls.bs_price(S, K, T, r, sigma, option_type)
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            vega = S * math.sqrt(T) * cls.norm_pdf(d1)

            diff = price - market_price
            if abs(diff) < 1e-4:
                return sigma * 100.0

            if vega < 1e-8:
                break

            sigma -= diff / vega

        return max(0.0, sigma * 100.0)

    @classmethod
    def calculate_delta(
        cls, S: float, K: float, T: float, r: float, iv_pct: float, option_type: str = "CE"
    ) -> float:
        """Calculates Option Delta."""
        sigma = iv_pct / 100.0
        if sigma <= 0 or T <= 0:
            return 0.5

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if option_type == "CE":
            return float(cls.norm_cdf(d1))
        else:
            return float(cls.norm_cdf(d1) - 1.0)


def get_now_ist() -> datetime:
    """Returns the current date and time localized in Indian Standard Time (IST)."""
    return datetime.now(timezone.utc).astimezone(IST)


class SupertrendIndexOptionsBot:
    def __init__(self):
        logger.info("Initializing Supertrend Options Bot for NIFTY Index...")
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

        # In-memory candle aggregator for 0-HTTP WebSocket streaming
        self._current_candle_time = None
        self._current_candle = None

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
        """Initialize WebSocket connection and subscribe to underlying index."""
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

    def _update_in_memory_candle(self, price: float):
        """Build 5m OHLCV candles directly in RAM from WebSocket ticks."""
        ist_now = get_now_ist()
        minute_floored = (ist_now.minute // 5) * 5
        bar_time = ist_now.replace(minute=minute_floored, second=0, microsecond=0)

        if self._current_candle_time is None or self._current_candle_time != bar_time:
            if self._current_candle is not None and self._current_candle_time is not None:
                new_row = pd.DataFrame([self._current_candle], index=[self._current_candle_time])
                self.history_cache = pd.concat([self.history_cache, new_row])
                self.history_cache = self.history_cache[
                    ~self.history_cache.index.duplicated(keep="last")
                ].tail(LOOKBACK_BARS)

            self._current_candle_time = bar_time
            self._current_candle = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
            }
        else:
            self._current_candle["high"] = max(self._current_candle["high"], price)
            self._current_candle["low"] = min(self._current_candle["low"], price)
            self._current_candle["close"] = price

    def _on_ws_price_update(self, data):
        """Callback for WebSocket LTP updates. Updates real-time 5m candle in RAM."""
        try:
            if isinstance(data, dict):
                price = None
                if "ltp" in data and isinstance(data["ltp"], (int, float)):
                    price = float(data["ltp"])
                elif "data" in data and isinstance(data["data"], dict) and "ltp" in data["data"]:
                    price = float(data["data"]["ltp"])

                if price and price > 0:
                    self._update_in_memory_candle(price)
        except Exception:
            pass

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
        """Recalculate Supertrend on rolling cache in RAM without periodic HTTP history calls."""
        if self.use_websocket and self.history_cache is not None and not self.history_cache.empty:
            df_calc = self.history_cache.copy()
            if self._current_candle is not None and self._current_candle_time is not None:
                live_row = pd.DataFrame([self._current_candle], index=[self._current_candle_time])
                df_calc = pd.concat([df_calc, live_row])
                df_calc = df_calc[~df_calc.index.duplicated(keep="last")].tail(LOOKBACK_BARS)

            st = supertrend(df_calc, period=ST_PERIOD, multiplier=ST_MULT)
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
        df_opts["expiry_date"] = pd.to_datetime(df_opts["expiry"], format="mixed", errors="coerce")

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

        # Check up to 10 strike steps outward on both sides to guarantee liquidity is found
        strikes_to_check = [base_strike + (i * step) for i in range(-10, 11)]

        candidates = valid_opts[
            (valid_opts["strike"].isin(strikes_to_check))
            & (valid_opts["instrumenttype"] == opt_type)
        ].copy()

        if candidates.empty:
            return None

        # Sort candidates by proximity to preferred target strike
        candidates["dist"] = (candidates["strike"] - target_strike).abs()
        candidates = candidates.sort_values("dist")

        def _evaluate_candidates(min_oi_threshold):
            best_sym = None
            best_sc = -1e9

            for _, row in candidates.iterrows():
                sym = row["symbol"]
                strike_price = float(row["strike"])
                try:
                    data = None
                    # 1. WebSocket-First: Check WebSocket RAM depth cache
                    if self.use_websocket:
                        try:
                            self.client.subscribe_depth([{"exchange": "NFO", "symbol": sym}])
                        except Exception:
                            pass
                        ws_depth = self.client.get_depth("NFO", sym)
                        if isinstance(ws_depth, dict) and (
                            "depth" in ws_depth or "buy" in ws_depth or "ltp" in ws_depth
                        ):
                            data = ws_depth.get("data", ws_depth)

                    # Fall back to REST Quotes API if WS RAM cache is unpopulated
                    if (
                        data is None
                        or not isinstance(data, dict)
                        or ("depth" not in data and "last_price" not in data and "ltp" not in data)
                    ):
                        q = self.client.quotes(symbol=sym, exchange="NFO")
                        if isinstance(q, dict) and q.get("status") == "success":
                            data = q.get("data", {})

                    if not isinstance(data, dict):
                        continue

                    vol = float(data.get("volume", 0))
                    oi = float(data.get("oi", 0))
                    ltp = float(data.get("last_price", data.get("ltp", 0)))

                    if min_oi_threshold > 0 and oi < min_oi_threshold:
                        logger.info(
                            f"  -> Skipping {sym}: OI ({oi:,.0f}) < Threshold ({min_oi_threshold:,.0f}). Searching next best strike..."
                        )
                        continue

                    # Real-time Market Depth & OBI
                    bids = data.get("depth", {}).get("buy", [])
                    asks = data.get("depth", {}).get("sell", [])

                    obi = 0.0
                    spread = 0.0
                    if bids and asks:
                        total_bid_qty = sum(float(b.get("quantity", 0)) for b in bids[:5])
                        total_ask_qty = sum(float(a.get("quantity", 0)) for a in asks[:5])
                        total_depth = total_bid_qty + total_ask_qty
                        if total_depth > 0:
                            obi = (total_bid_qty - total_ask_qty) / total_depth

                        best_bid = float(bids[0].get("price", 0)) if bids else 0
                        best_ask = float(asks[0].get("price", 0)) if asks else 0
                        if best_bid > 0 and best_ask > 0:
                            spread = best_ask - best_bid

                    # Skip if spread exceeds limit
                    if MAX_BID_ASK_SPREAD > 0 and spread > MAX_BID_ASK_SPREAD:
                        logger.info(
                            f"  -> Skipping {sym}: Bid-Ask Spread (₹{spread:.2f}) > Limit (₹{MAX_BID_ASK_SPREAD:.2f})"
                        )
                        continue

                    # Local Black-Scholes Delta & IV (0ms)
                    iv, delta = LocalOptionGreeks.get_greeks(
                        underlying_price=underlying_price,
                        strike_price=strike_price,
                        expiry_date_str=str(nearest_expiry.strftime("%Y-%m-%d")),
                        option_type=opt_type,
                        option_ltp=ltp if ltp > 0 else 100.0,
                    )

                    if MIN_OPTION_DELTA > 0 and abs(delta) < MIN_OPTION_DELTA:
                        logger.info(
                            f"  -> Skipping {sym}: Delta ({abs(delta):.2f}) < Min ({MIN_OPTION_DELTA:.2f})"
                        )
                        continue

                    if MAX_OPTION_IV > 0 and iv > MAX_OPTION_IV:
                        logger.info(
                            f"  -> Skipping {sym}: IV ({iv:.1f}%) > Max ({MAX_OPTION_IV:.1f}%)"
                        )
                        continue

                    # Option Supertrend Trend Check for candidate strike in option chain
                    if USE_OPTION_ST_FILTER and not self._check_option_supertrend(sym):
                        logger.info(
                            f"  -> Skipping {sym}: Option ST Trend is NOT UP. Searching next strike in option chain..."
                        )
                        continue

                    # Multi-Factor Quantitative Advantage Score
                    score = (
                        (oi * OI_WEIGHT)
                        + (vol * VOLUME_WEIGHT)
                        + (abs(delta) * 100_000 * DELTA_WEIGHT)
                        + (obi * 100_000 * OBI_WEIGHT)
                        - (spread * 50_000)
                    )

                    logger.info(
                        f"  -> Candidate {sym}: OI={oi:,.0f}, Vol={vol:,.0f}, Delta={abs(delta):.2f}, "
                        f"OBI={obi:+.2f}, Spread=₹{spread:.2f} | Multi-Factor Score: {score:,.1f}"
                    )

                    if score > best_sc:
                        best_sc = score
                        best_sym = sym

                except Exception as e:
                    logger.warning(f"  -> Failed to evaluate candidate {sym}: {e}")

            return best_sym, best_sc

        # Pass 1: Evaluate candidates with strict MIN_OPTION_OI threshold
        selected_symbol, selected_score = _evaluate_candidates(MIN_OPTION_OI)

        # Pass 2: Fallback — if no strike met strict threshold, evaluate all liquid strikes to select next best contract
        if not selected_symbol and MIN_OPTION_OI > 0:
            logger.warning(
                f"⚠️ No contract met strict MIN_OPTION_OI ({MIN_OPTION_OI:,.0f}). Searching next best liquid strike across all candidates..."
            )
            selected_symbol, selected_score = _evaluate_candidates(0)

        if selected_symbol:
            logger.info(
                f"✅ Selected Superior Contract via Multi-Factor Score: {selected_symbol} (Score: {selected_score:,.1f})"
            )
            return selected_symbol

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
        if not USE_OPTION_ST_FILTER:
            logger.info(
                f"Option ST Filter is disabled (USE_OPTION_ST_FILTER=False). Skipping check for {symbol}."
            )
            return True

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

    def check_websocket_filters(self, symbol: str, direction: str = "UP") -> dict:
        """
        Pure WebSocket In-Memory Pre-Trade Filter Engine:
        1. Reads live Market Depth & LTP from WS RAM Cache (with graceful REST fallback if WS depth missing).
        2. Solves Implied Volatility (IV) & Option Delta locally in Python via Black-Scholes.
        3. Calculates 5-Level Order Book Imbalance (OBI) & Bid-Ask Spread.
        """
        if not USE_WEBSOCKET_ORDERFLOW and not USE_LOCAL_GREEKS_FILTER:
            return {"pass": True, "reason": "WS Order Flow & Local Greeks Filters Disabled"}

        # 1. Subscribe WS Depth for option contract if connected
        if self.use_websocket:
            try:
                self.client.subscribe_depth([{"exchange": "NFO", "symbol": symbol}])
                time.sleep(0.15)  # Allow WS depth tick to enter RAM cache
            except Exception as e:
                logger.warning(f"Failed to subscribe WS Depth for {symbol}: {e}")

        # 2. Fetch Live Spot Price & Option Depth from RAM Cache
        underlying_ltp = self.get_live_ltp(EXCHANGE, UNDERLYING_SYMBOL)

        depth_data = None
        data_source = "WS_CACHE"
        if self.use_websocket:
            depth_data = self.client.get_depth("NFO", symbol)

        # Fallback to REST Depth API if WS cache is empty
        if (
            not isinstance(depth_data, dict)
            or "depth" not in depth_data
            or not isinstance(depth_data.get("depth"), dict)
            or not depth_data["depth"].get("buy")
        ):
            data_source = "REST_FALLBACK"
            try:
                resp = self.client.depth(symbol=symbol, exchange="NFO")
                if isinstance(resp, dict) and resp.get("status") == "success":
                    depth_data = resp.get("data", {})
            except Exception as e:
                logger.warning(f"REST Depth fallback failed for {symbol}: {e}")

        # Handle 'No Flow' case
        if (
            not isinstance(depth_data, dict)
            or "depth" not in depth_data
            or not isinstance(depth_data.get("depth"), dict)
        ):
            if ALLOW_TRADE_ON_NO_FLOW:
                logger.warning(
                    f"⚠️ NO FLOW available for {symbol}, but ALLOW_TRADE_ON_NO_FLOW=True. Bypassing filter."
                )
                return {"pass": True, "reason": "No Flow Bypassed"}
            else:
                logger.error(
                    f"⛔ REJECTED: NO FLOW / Depth available for {symbol}. Skipping trade for safety."
                )
                return {"pass": False, "reason": "No Market Depth Flow Available"}

        bids = depth_data["depth"].get("buy", [])
        asks = depth_data["depth"].get("sell", [])

        if not bids or not asks:
            return {"pass": False, "reason": "Illiquid Contract (Empty Book)"}

        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 0))
        opt_ltp = float(depth_data.get("ltp", (best_bid + best_ask) / 2.0))

        if best_bid <= 0 or best_ask <= 0 or opt_ltp <= 0:
            return {"pass": False, "reason": "Invalid Bid/Ask Prices"}

        # 3. Calculate Spread & Order Book Imbalance (OBI)
        spread = round(best_ask - best_bid, 2)
        total_bids = sum(int(b.get("quantity", 0)) for b in bids)
        total_asks = sum(int(a.get("quantity", 0)) for a in asks)
        total_qty = total_bids + total_asks

        obi = (total_bids - total_asks) / total_qty if total_qty > 0 else 0.0

        # 4. Calculate Local Black-Scholes IV & Delta
        local_iv = 0.0
        local_delta = 0.5
        if USE_LOCAL_GREEKS_FILTER and underlying_ltp > 0:
            strike_val = underlying_ltp
            if self._instruments_cache is not None and not self._instruments_cache.empty:
                match_row = self._instruments_cache[self._instruments_cache["symbol"] == symbol]
                if not match_row.empty and "strike" in match_row.columns:
                    strike_val = float(match_row["strike"].iloc[0])

            days_to_expiry = max(0.001, 3.0 / 365.0)
            option_type = "CE" if direction == "UP" else "PE"

            local_iv = LocalOptionGreeks.calculate_iv(
                S=underlying_ltp,
                K=strike_val,
                T=days_to_expiry,
                r=0.065,
                market_price=opt_ltp,
                option_type=option_type,
            )
            local_delta = abs(
                LocalOptionGreeks.calculate_delta(
                    S=underlying_ltp,
                    K=strike_val,
                    T=days_to_expiry,
                    r=0.065,
                    iv_pct=local_iv,
                    option_type=option_type,
                )
            )

        logger.info(
            f"⚡ [{data_source} Pre-Trade Check] {symbol} ({direction}) | "
            f"Spread: ₹{spread} | OBI: {obi:+.2f} | Local IV: {local_iv:.1f}% | Local Delta: {local_delta:.2f}"
        )

        # 5. Evaluate Filter Thresholds
        if USE_WEBSOCKET_ORDERFLOW:
            if spread > MAX_BID_ASK_SPREAD:
                return {
                    "pass": False,
                    "reason": f"Spread ₹{spread} > Max Allowed ₹{MAX_BID_ASK_SPREAD}",
                }

            if obi < MIN_ORDER_BOOK_IMBALANCE:
                return {
                    "pass": False,
                    "reason": f"OBI ({obi:+.2f}) < Min Threshold ({MIN_ORDER_BOOK_IMBALANCE})",
                }

        if USE_LOCAL_GREEKS_FILTER:
            if local_iv > MAX_OPTION_IV and local_iv > 0:
                return {
                    "pass": False,
                    "reason": f"Local IV ({local_iv:.1f}%) > Max ({MAX_OPTION_IV}%)",
                }

            if local_delta < MIN_OPTION_DELTA:
                return {
                    "pass": False,
                    "reason": f"Local Delta ({local_delta:.2f}) < Min ({MIN_OPTION_DELTA})",
                }

        return {
            "pass": True,
            "reason": "All WebSocket Filters Passed",
            "obi": obi,
            "spread": spread,
            "iv": local_iv,
            "delta": local_delta,
        }

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

        logger.info(f"Checking WebSocket Order Flow & Local Greeks filters for {symbol}...")
        ws_eval = self.check_websocket_filters(symbol, direction=direction)
        if not ws_eval["pass"]:
            logger.warning(
                f"Skipping trade for {symbol} — WS Filter Failed: {ws_eval['reason']}. Entering 60s cooldown."
            )
            self._failed_entry_cooldown_until = time.time() + 60
            return

        # Determine quantity
        order_qty = self.trade_quantity if self.trade_quantity else 25

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
        order_qty = self.trade_quantity if self.trade_quantity else 25
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

            # 4. Check Underlying Index Supertrend Flip (only on new candle boundaries)
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
                            f"Underlying index {UNDERLYING_SYMBOL} Supertrend flipped to Bearish (-1) while holding CE!"
                        )
                        self.exit_trade("INDEX_ST_FLIP")
                        return
                    elif self.active_direction == "DOWN" and st_dir == 1:
                        logger.info(
                            f"Underlying index {UNDERLYING_SYMBOL} Supertrend flipped to Bullish (1) while holding PE!"
                        )
                        self.exit_trade("INDEX_ST_FLIP")
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
                    ist_now = get_now_ist()
                    current_ist_hour = ist_now.hour
                    if (
                        USE_TIME_FILTER
                        and ALLOWED_START_HOUR is not None
                        and ALLOWED_END_HOUR is not None
                    ):
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
                            if USE_ADX_FILTER:
                                if len(df) > ADX_PERIOD:
                                    _, _, adx_series = ta.adx(
                                        df["high"], df["low"], df["close"], period=ADX_PERIOD
                                    )
                                    current_adx = float(
                                        adx_series.iloc[-2]
                                        if hasattr(adx_series, "iloc")
                                        else adx_series[-2]
                                    )
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
                            else:
                                logger.info(
                                    "ADX Filter is disabled (USE_ADX_FILTER=False). Skipping check."
                                )

                            # 2. Moving Average Filter
                            if USE_MA_FILTER:
                                if len(df) > MA_PERIOD:
                                    ma_series = ta.sma(df["close"], period=MA_PERIOD)
                                    current_ma = float(
                                        ma_series.iloc[-2]
                                        if hasattr(ma_series, "iloc")
                                        else ma_series[-2]
                                    )
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
                            else:
                                logger.info(
                                    "Moving Average Filter is disabled (USE_MA_FILTER=False). Skipping check."
                                )

                            # 3. Intraday VWAP Filter
                            if USE_VWAP_FILTER:
                                vwap_series = vwap(df)
                                if not vwap_series.empty and len(vwap_series) >= 2:
                                    current_vwap = float(vwap_series.iloc[-2])
                                    prev_close = float(df.iloc[-2]["close"])
                                    if direction == "UP" and prev_close < current_vwap:
                                        logger.info(
                                            f"Skipping trade: Direction is UP but Close ({prev_close}) < VWAP ({current_vwap:.2f})"
                                        )
                                        filters_passed = False
                                    elif direction == "DOWN" and prev_close > current_vwap:
                                        logger.info(
                                            f"Skipping trade: Direction is DOWN but Close ({prev_close}) > VWAP ({current_vwap:.2f})"
                                        )
                                        filters_passed = False
                                else:
                                    logger.warning("Not enough data for VWAP calculation.")
                                    filters_passed = False
                            else:
                                logger.info(
                                    "VWAP Filter is disabled (USE_VWAP_FILTER=False). Skipping check."
                                )

                            # 4. Moving VWAP (MVWAP) Filter
                            if USE_MVWAP_FILTER:
                                mvwap_series = mvwap(df, period=MVWAP_PERIOD)
                                if not mvwap_series.empty and len(mvwap_series) >= 2:
                                    current_mvwap = float(mvwap_series.iloc[-2])
                                    prev_close = float(df.iloc[-2]["close"])
                                    if direction == "UP" and prev_close < current_mvwap:
                                        logger.info(
                                            f"Skipping trade: Direction is UP but Close ({prev_close}) < MVWAP ({current_mvwap:.2f})"
                                        )
                                        filters_passed = False
                                    elif direction == "DOWN" and prev_close > current_mvwap:
                                        logger.info(
                                            f"Skipping trade: Direction is DOWN but Close ({prev_close}) > MVWAP ({current_mvwap:.2f})"
                                        )
                                        filters_passed = False
                                else:
                                    logger.warning("Not enough data for MVWAP calculation.")
                                    filters_passed = False
                            else:
                                logger.info(
                                    "MVWAP Filter is disabled (USE_MVWAP_FILTER=False). Skipping check."
                                )

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
    bot = SupertrendIndexOptionsBot()
    bot.run()
