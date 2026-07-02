"""
OpenAlgo Strategy Bot — NIFTY Weekly Options OI Surge Bot
========================================================
Detects Open Interest (OI) surges on NIFTY option chains to place intraday MIS
options orders with dynamic SL, trailing SL, and advanced exits.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NiftyOISurger")

try:
    from openalgo import api
except ImportError:
    logger.error("Install the SDK first: pip install openalgo")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_KEY = os.getenv(
    "OPENALGO_API_KEY", "b45feb0a6973ed00fe86d25ace49d4da8dfe8d0a78c334455d46254ded28a26d"
)
API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

# WhatsApp Notifications
WHATSAPP_PHONES = [n.strip() for n in os.getenv("WHATSAPP_PHONES", "").split(",") if n.strip()]

# Trading Settings
TRADING_MODE = os.getenv("TRADING_MODE", "analyze").lower()  # "analyze" (sim) or "live"
SYMBOL = "NIFTY"
UNDERLYING_EXCHANGE = "NSE_INDEX"
OPTIONS_EXCHANGE = "NFO"
QUANTITY = int(os.getenv("QUANTITY", "65"))
PRODUCT = os.getenv("PRODUCT", "MIS")
MIN_OPTION_PREMIUM = 40.0
REENTRY_COOLDOWN_MINUTES = 30

# Strategy Settings / Thresholds
HIGHER_OI_PCT = 0.09
HIGHER_OI_FLOOR = 6000000
OI_DELTA_PCT = 0.04
OI_DELTA_FLOOR = 4000000
LOWER_OI_PCT = 0.02
LOWER_OI_FLOOR = 3000000

DYNAMIC_REVERSAL_OI_PCT = 0.09
REVERSAL_OI_FLOOR = 6000000
REVERSAL_TIGHTEN_GAP = 5.0
LTP_MOM_LOOKBACK_MIN = 3  # ~12 cycles of 15 seconds
LTP_MOM_THRESHOLD = -3.0
EARLY_ADVERSE_MINUTES = 5
EARLY_ADVERSE_PTS = 13.0

INITIAL_SL_PCT = 0.13
INITIAL_SL_MIN = 12.0
INITIAL_SL_MAX = 22.0
INITIAL_SL_MIN_HIGH_VIX = 15.0
INITIAL_SL_MAX_HIGH_VIX = 25.0
VIX_HIGH_THRESHOLD = 18.0

TRAIL_GAP_NORMAL = 13.0
TRAIL_GAP_FAST = 10.0
TRAIL_BUFFER_FIRST = 20.0
TRAIL_BUFFER_AFTER_PROFIT = 10.0

MIN_OI_ROWS = 17

# CSV Logging disabled (replaced with in-memory caching and SDK-based state queries)

# ==============================================================================
# BOT ENGINE
# ==============================================================================


class NiftyOISurgerBot:
    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)
        self.strategy_name = "NIFTY_OI_Surger_v1"
        self.position: Optional[Dict[str, Any]] = None
        self.oi_history: list[Dict[str, Any]] = []
        self.ltp_history: list[float] = []
        self.is_reentry = False
        self.cooldown_until: Optional[datetime] = None
        self.last_chain_fetch_min = -1
        self.running = True

        # Position Exit/Trailing State
        self.current_sl_trigger = 0.0
        self.buffer_price = 0.0
        self.trail_count = 0
        self.best_premium = 0.0
        self.early_adverse_checked = False

        logger.info(
            f"Initialized Nifty OI Surger Bot | mode={TRADING_MODE} | qty={QUANTITY} | product={PRODUCT}"
        )

    # ------------------------------------------------------------------ Recovery & API Warmup
    def get_nearest_nifty_expiry(self) -> str:
        """Find the nearest NIFTY expiry date from the master instruments database."""
        try:
            df = self.client.instruments(exchange="NFO")
            if isinstance(df, pd.DataFrame) and not df.empty:
                nifty_df = df[df["name"] == SYMBOL].copy()
                if not nifty_df.empty:
                    expiries = nifty_df["expiry"].unique()
                    parsed_expiries = []
                    for exp in expiries:
                        try:
                            # Try parsing 'DD-MMM-YY' (e.g., '07-JUL-26')
                            dt = datetime.strptime(exp, "%d-%b-%y")
                            parsed_expiries.append((dt, exp))
                        except Exception:
                            try:
                                # Try parsing 'DD-MMM-YYYY' (e.g., '07-JUL-2026')
                                dt = datetime.strptime(exp, "%d-%b-%Y")
                                parsed_expiries.append((dt, exp))
                            except Exception:
                                pass

                    parsed_expiries.sort(key=lambda x: x[0])
                    today = datetime.now().date()
                    for dt, orig_exp in parsed_expiries:
                        if dt.date() >= today:
                            # Format to 'DDMMMYY' expected by optionschain API
                            clean_exp = orig_exp.replace("-", "").upper()
                            return clean_exp
        except Exception as e:
            logger.warning(f"Failed to auto-detect nearest NIFTY expiry: {e}")
        return ""

    def warmup_history_from_api(self):
        """Warm up the rolling OI history in-memory using history candles from the SDK."""
        logger.info("Starting in-memory history warmup using OpenAlgo SDK...")
        expiry_date = self.get_nearest_nifty_expiry()
        if not expiry_date:
            logger.warning(
                "Could not resolve nearest expiry date for NIFTY. History warmup skipped."
            )
            return

        # 1. Fetch current option chain to resolve the strike symbols
        chain_data = self.client.optionchain(
            underlying=SYMBOL,
            exchange=UNDERLYING_EXCHANGE,
            expiry_date=expiry_date,
            strike_count=15,
        )
        if not isinstance(chain_data, dict) or chain_data.get("status") != "success":
            logger.warning(f"Failed to fetch option chain for warmup: {chain_data}")
            return

        # 2. Extract CE and PE symbols
        ce_symbols = []
        pe_symbols = []
        for item in chain_data.get("chain", []):
            ce = item.get("ce") or {}
            pe = item.get("pe") or {}
            if ce.get("symbol"):
                ce_symbols.append(ce["symbol"])
            if pe.get("symbol"):
                pe_symbols.append(pe["symbol"])

        if not ce_symbols or not pe_symbols:
            logger.warning("No option chain symbols resolved. Warmup aborted.")
            return

        # 3. Fetch underlying index history to get base timestamps and spot prices
        today = datetime.now()
        start_date = today - timedelta(days=2)  # Check past 2 days to guarantee enough 1m candles

        spot_df = self.client.history(
            symbol=SYMBOL,
            exchange=UNDERLYING_EXCHANGE,
            interval="1m",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
            source="db",
        )
        if not isinstance(spot_df, pd.DataFrame) or spot_df.empty:
            logger.warning("Could not fetch underlying spot history. Warmup aborted.")
            return

        # Align to the last 60 minutes of spot data
        target_timestamps = spot_df.index[-60:]

        # Initialize historical mapping
        history_map = {}
        for ts in target_timestamps:
            close_price = spot_df.loc[ts, "close"]
            # Handle possible duplicate index entries
            if isinstance(close_price, pd.Series):
                close_price = close_price.iloc[0]

            history_map[ts] = {
                "spot": float(close_price),
                "ce_oi": 0.0,
                "pe_oi": 0.0,
                "volume_ce": 0.0,
                "volume_pe": 0.0,
            }

        # 4. Fetch history for all option symbols in the chain and aggregate
        logger.info(f"Fetching history for {len(ce_symbols) + len(pe_symbols)} option contracts...")

        for ce_sym in ce_symbols:
            try:
                df = self.client.history(
                    symbol=ce_sym,
                    exchange=OPTIONS_EXCHANGE,
                    interval="1m",
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=today.strftime("%Y-%m-%d"),
                    source="api",
                )
                if isinstance(df, pd.DataFrame) and not df.empty:
                    for ts in target_timestamps:
                        if ts in df.index:
                            row_data = df.loc[ts]
                            oi_val = row_data["oi"]
                            vol_val = row_data["volume"]
                            if isinstance(oi_val, pd.Series):
                                oi_val = oi_val.iloc[0]
                            if isinstance(vol_val, pd.Series):
                                vol_val = vol_val.iloc[0]

                            if pd.notna(oi_val):
                                history_map[ts]["ce_oi"] += float(oi_val)
                            if pd.notna(vol_val):
                                history_map[ts]["volume_ce"] += float(vol_val)
            except Exception as e:
                logger.debug(f"Could not fetch/parse history for CE {ce_sym}: {e}")

        for pe_sym in pe_symbols:
            try:
                df = self.client.history(
                    symbol=pe_sym,
                    exchange=OPTIONS_EXCHANGE,
                    interval="1m",
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=today.strftime("%Y-%m-%d"),
                    source="api",
                )
                if isinstance(df, pd.DataFrame) and not df.empty:
                    for ts in target_timestamps:
                        if ts in df.index:
                            row_data = df.loc[ts]
                            oi_val = row_data["oi"]
                            vol_val = row_data["volume"]
                            if isinstance(oi_val, pd.Series):
                                oi_val = oi_val.iloc[0]
                            if isinstance(vol_val, pd.Series):
                                vol_val = vol_val.iloc[0]

                            if pd.notna(oi_val):
                                history_map[ts]["pe_oi"] += float(oi_val)
                            if pd.notna(vol_val):
                                history_map[ts]["volume_pe"] += float(vol_val)
            except Exception as e:
                logger.debug(f"Could not fetch/parse history for PE {pe_sym}: {e}")

        # 5. Populate self.oi_history
        self.oi_history = []
        for ts in sorted(history_map.keys()):
            row = history_map[ts]
            spot_val = row["spot"]
            atm_strike = round(spot_val / 50) * 50
            ts_naive = ts.to_pydatetime().replace(tzinfo=None)

            self.oi_history.append(
                {
                    "timestamp": ts_naive,
                    "spot": spot_val,
                    "atm_strike": float(atm_strike),
                    "ce_oi": row["ce_oi"],
                    "pe_oi": row["pe_oi"],
                    "total_volume": row["volume_ce"] + row["volume_pe"],
                    "volume_ce": row["volume_ce"],
                    "volume_pe": row["volume_pe"],
                }
            )

        logger.info(
            f"Successfully warmed up in-memory history with {len(self.oi_history)} data points."
        )

    def check_reentry_and_recovery(self):
        """Query tradebook and positionbook from SDK to recover trade state and enforce cooldown/re-entry logic."""
        logger.info("Checking today's tradebook and active positions for recovery...")

        # 1. Check current positions to recover active trades
        try:
            p_resp = self.client.positionbook()
            if isinstance(p_resp, dict) and p_resp.get("status") == "success":
                positions = p_resp.get("data", [])
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    qty = int(pos.get("quantity", 0))
                    # Check if we have an active (non-zero) position in a NIFTY weekly option
                    if symbol.startswith(SYMBOL) and qty != 0:
                        logger.info(
                            f"Active position found in NIFTY options: {symbol} with quantity {qty}"
                        )

                        # Recover position details
                        action = "BUY" if qty > 0 else "SELL"
                        avg_price = float(pos.get("average_price", 0.0))

                        # Parse option type and strike from symbol
                        # NIFTY option symbols follow format: NIFTY[expiry][strike][type]
                        # E.g. NIFTY07JUL2623900CE -> CE, strike=23900
                        option_type = "CE" if symbol.endswith("CE") else "PE"

                        # Extract strike digits from symbol
                        strike_str = ""
                        for char in reversed(symbol[:-2]):
                            if char.isdigit():
                                strike_str = char + strike_str
                            else:
                                break
                        strike = float(strike_str) if strike_str else 0.0

                        vix = self.get_india_vix()
                        high_vix = vix > VIX_HIGH_THRESHOLD
                        sl_min = INITIAL_SL_MIN_HIGH_VIX if high_vix else INITIAL_SL_MIN
                        sl_max = INITIAL_SL_MAX_HIGH_VIX if high_vix else INITIAL_SL_MAX
                        initial_sl = max(sl_min, min(sl_max, round(avg_price * INITIAL_SL_PCT)))

                        self.current_sl_trigger = max(1.0, avg_price - initial_sl)
                        self.buffer_price = avg_price + initial_sl
                        self.trail_count = 0
                        self.best_premium = avg_price
                        self.early_adverse_checked = False

                        self.position = {
                            "trading_symbol": symbol,
                            "option_type": option_type,
                            "entry_price": avg_price,
                            "entry_time": datetime.now(),  # fallback to now
                            "vix": vix,
                            "expiry_date": "",  # will be resolved dynamically
                            "trend_list": "Recovered",
                            "strike": strike,
                            "initial_sl": initial_sl,
                        }
                        self.ltp_history = [avg_price]
                        logger.info(f"Successfully recovered position: {symbol} @ {avg_price}")
                        return
        except Exception as e:
            logger.error(f"Error checking position book: {e}")

        # 2. Check trades executed today for reentry and cooldown logic
        try:
            t_resp = self.client.tradebook()
            if not isinstance(t_resp, dict) or t_resp.get("status") != "success":
                logger.warning(f"Could not retrieve tradebook for recovery: {t_resp}")
                return

            trades = t_resp.get("data", [])
            # Filter trades matching NIFTY options and MIS product
            option_trades = []
            for t in trades:
                symbol = t.get("symbol", "")
                product = t.get("product", "")
                if symbol.startswith(SYMBOL) and product == PRODUCT:
                    option_trades.append(t)

            if not option_trades:
                logger.info("No NIFTY option trades found for today. Starting fresh.")
                return

            # Set reentry flag since trades occurred today
            self.is_reentry = True

            # Enforce no re-entry on Expiry day (Tuesdays)
            if datetime.now().weekday() == 1:
                logger.warning("Re-entry is blocked on Tuesday (expiry day). Exiting bot.")
                sys.exit(0)

            # Sort trades by timestamp descending to find last sell/exit trade
            # Timestamp format example: '30-Jun-2026 13:15:20'
            def get_timestamp(trade):
                ts_str = trade.get("timestamp", "")
                try:
                    return datetime.strptime(ts_str, "%d-%b-%Y %H:%M:%S")
                except Exception:
                    return datetime.min

            option_trades.sort(key=get_timestamp, reverse=True)

            # Find the most recent SELL trade to determine cooldown
            last_sell = None
            for t in option_trades:
                if t.get("action") == "SELL":
                    last_sell = t
                    break

            if not last_sell:
                logger.warning(
                    "BUY trade found today, but no matching SELL trade found. Safe restart assumed closed."
                )
                return

            # Apply reentry recovery cooldown
            exit_time = get_timestamp(last_sell)
            if exit_time != datetime.min:
                elapsed = (datetime.now() - exit_time).total_seconds() / 60.0
                if elapsed < REENTRY_COOLDOWN_MINUTES:
                    cooldown_rem = REENTRY_COOLDOWN_MINUTES - elapsed
                    self.cooldown_until = datetime.now() + timedelta(minutes=cooldown_rem)
                    logger.info(
                        f"Re-entry cooldown active. Cooldown remaining: {cooldown_rem:.1f} minutes."
                    )
                else:
                    logger.info("Re-entry cooldown has expired. Ready to search for entries.")
        except Exception as e:
            logger.error(f"Error parsing tradebook recovery data: {e}")

    # ------------------------------------------------------------------ Market Timing
    def is_market_open(self, now: datetime) -> bool:
        if now.weekday() >= 5:
            return False
        start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return start <= now <= end

    def can_place_new_trades(self, now: datetime) -> bool:
        if now.weekday() >= 5:
            return False
        # Expiry Day (Tuesday = 1): 9:25 AM to 11:00 AM
        if now.weekday() == 1:
            start = now.replace(hour=9, minute=25, second=0, microsecond=0)
            stop = now.replace(hour=11, minute=0, second=0, microsecond=0)
            return start <= now <= stop
        else:
            # Mon/Wed-Fri: 9:25 AM to 2:30 PM
            start = now.replace(hour=9, minute=25, second=0, microsecond=0)
            stop = now.replace(hour=14, minute=30, second=0, microsecond=0)
            return start <= now <= stop

    # ------------------------------------------------------------------ India VIX Check
    def get_india_vix(self) -> float:
        try:
            r = self.client.quotes(symbol="INDIA VIX", exchange="NSE_INDEX")
            if isinstance(r, dict) and r.get("status") == "success":
                return float(r.get("data", {}).get("ltp", 15.0))
            r = self.client.quotes(symbol="INDIA VIX", exchange="NSE")
            if isinstance(r, dict) and r.get("status") == "success":
                return float(r.get("data", {}).get("ltp", 15.0))
        except Exception as e:
            logger.warning(f"Failed to fetch India VIX: {e}. Defaulting to VIX=15.0")
        return 15.0

    # ------------------------------------------------------------------ Alerts
    def send_alert(self, title: str, details: str):
        msg = (
            f"[{title} - {'LIVE' if TRADING_MODE == 'live' else 'SIM'}]\n"
            f"Strategy: {self.strategy_name}\n"
            f"{details}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(f"ALERT: {msg}")
        try:
            if WHATSAPP_PHONES:
                self.client.whatsapp(msg, to=WHATSAPP_PHONES[:5])
            else:
                self.client.whatsapp(msg)
        except Exception as e:
            logger.warning(f"WhatsApp alert failed: {e}")

    # ------------------------------------------------------------------ Option helpers
    def _fetch_and_gate(self, option_symbol: str) -> str:
        """Fetch 15-minute history and calculate RSI(14) and SuperTrend(10,3) for logging."""
        try:
            end = datetime.now()
            start = end - timedelta(days=5)
            resp = self.client.history(
                symbol=option_symbol,
                exchange=OPTIONS_EXCHANGE,
                interval="15m",
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                source="api",
            )
            if isinstance(resp, pd.DataFrame) and not resp.empty:
                df = resp.reset_index()
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                if len(df) >= 15:
                    # Calculate RSI
                    close = df["close"]
                    delta = close.diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
                    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
                    rs = avg_gain / avg_loss.replace(0, np.nan)
                    rsi_series = 100 - (100 / (1 + rs))

                    # Calculate SuperTrend
                    hl2 = (df["high"] + df["low"]) / 2
                    tr1 = df["high"] - df["low"]
                    tr2 = (df["high"] - df["close"].shift(1)).abs()
                    tr3 = (df["low"] - df["close"].shift(1)).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.ewm(alpha=1 / 10, min_periods=10, adjust=False).mean()

                    up = hl2 + 3.0 * atr
                    dn = hl2 - 3.0 * atr

                    supertrend = pd.Series(np.nan, index=df.index)
                    direction = pd.Series(1, index=df.index)
                    for i in range(1, len(df)):
                        if df["close"].iloc[i] > up.iloc[i - 1]:
                            direction.iloc[i] = 1
                        elif df["close"].iloc[i] < dn.iloc[i - 1]:
                            direction.iloc[i] = -1
                        else:
                            direction.iloc[i] = direction.iloc[i - 1]

                        supertrend.iloc[i] = dn.iloc[i] if direction.iloc[i] == 1 else up.iloc[i]

                    rsi_val = rsi_series.iloc[-1]
                    st_dir = "bullish" if direction.iloc[-1] == 1 else "bearish"
                    st_val = supertrend.iloc[-1]
                    return f"RSI={rsi_val:.1f}, ST={st_dir}({st_val:.2f})"
        except Exception as e:
            logger.debug(f"Could not calculate option chain RSI/ST: {e}")
        return "RSI=50.0, ST=bullish(0)"

    def get_option_details_from_chain(
        self, chain_data: Dict[str, Any], option_type: str
    ) -> Tuple[Optional[str], Optional[float]]:
        atm_strike = chain_data.get("atm_strike", 0)
        # NIFTY Strike steps are 50 points.
        # CE buys ATM - 100. PE buys ATM + 100.
        target_strike = atm_strike - 100 if option_type == "CE" else atm_strike + 100

        for item in chain_data.get("chain", []):
            if int(item["strike"]) == int(target_strike):
                opt = item.get(option_type.lower()) or {}
                return opt.get("symbol"), opt.get("ltp")
        return None, None

    # ------------------------------------------------------------------ Signal Indicators
    def get_trend_short(self, type_side: str) -> str:
        """Trend helpers comparing sequential 3-minute block changes of (CE - PE) or (PE - CE)."""

        def calculate_block_change(r_end, r_start, side):
            if side == "CE":
                return (r_end["ce_oi"] - r_end["pe_oi"]) - (r_start["ce_oi"] - r_start["pe_oi"])
            else:
                return (r_end["pe_oi"] - r_end["ce_oi"]) - (r_start["pe_oi"] - r_start["ce_oi"])

        blocks = [
            calculate_block_change(self.oi_history[-1], self.oi_history[-4], type_side),
            calculate_block_change(self.oi_history[-4], self.oi_history[-7], type_side),
            calculate_block_change(self.oi_history[-7], self.oi_history[-10], type_side),
            calculate_block_change(self.oi_history[-10], self.oi_history[-13], type_side),
        ]

        if all(b >= 0 for b in blocks):
            return "Uptrend"
        elif all(b < 0 for b in blocks):
            return "Downtrend"
        else:
            return "Sideways"

    def get_past_five_3min_trend(self, type_side: str) -> str:
        """Trend helper checking 5 blocks of 3-min changes for Tiers 11/12."""

        def calculate_block_change(r_end, r_start, side):
            if side == "CE":
                return (r_end["ce_oi"] - r_end["pe_oi"]) - (r_start["ce_oi"] - r_start["pe_oi"])
            else:
                return (r_end["pe_oi"] - r_end["ce_oi"]) - (r_start["pe_oi"] - r_start["ce_oi"])

        blocks = [
            calculate_block_change(self.oi_history[-1], self.oi_history[-4], type_side),
            calculate_block_change(self.oi_history[-4], self.oi_history[-7], type_side),
            calculate_block_change(self.oi_history[-7], self.oi_history[-10], type_side),
            calculate_block_change(self.oi_history[-10], self.oi_history[-13], type_side),
            calculate_block_change(self.oi_history[-13], self.oi_history[-16], type_side),
        ]
        if all(b >= 0 for b in blocks):
            return "Uptrend"
        elif all(b < 0 for b in blocks):
            return "Downtrend"
        else:
            return "Sideways"

    def get_spot_trend(self) -> str:
        blocks = [
            self.oi_history[-1]["spot"] - self.oi_history[-4]["spot"],
            self.oi_history[-4]["spot"] - self.oi_history[-7]["spot"],
            self.oi_history[-7]["spot"] - self.oi_history[-10]["spot"],
            self.oi_history[-10]["spot"] - self.oi_history[-13]["spot"],
        ]
        if all(b >= 0 for b in blocks):
            return "Uptrend"
        elif all(b < 0 for b in blocks):
            return "Downtrend"
        else:
            return "Sideways"

    def calculate_indicators(self) -> Dict[str, Any]:
        latest = self.oi_history[-1]

        # Base row lookback (up to 30 rows)
        lookback_len = min(30, len(self.oi_history) - 1)
        base_row = self.oi_history[-1 - lookback_len]

        oi_change_ce = latest["ce_oi"] - base_row["ce_oi"]
        oi_change_pe = latest["pe_oi"] - base_row["pe_oi"]

        ratio_ce_pe = oi_change_ce / (oi_change_pe if oi_change_pe != 0 else 1)
        ratio_pe_ce = oi_change_pe / (oi_change_ce if oi_change_ce != 0 else 1)

        row_5min = self.oi_history[-6]
        ce_change_5min = latest["ce_oi"] - row_5min["ce_oi"]
        pe_change_5min = latest["pe_oi"] - row_5min["pe_oi"]

        # 3-min changes
        ce_change_1st_3min = latest["ce_oi"] - self.oi_history[-4]["ce_oi"]
        pe_change_1st_3min = latest["pe_oi"] - self.oi_history[-4]["pe_oi"]
        ce_change_2nd_3min = self.oi_history[-4]["ce_oi"] - self.oi_history[-7]["ce_oi"]
        pe_change_2nd_3min = self.oi_history[-4]["pe_oi"] - self.oi_history[-7]["pe_oi"]

        trend_ce_short = self.get_trend_short("CE")
        trend_pe_short = self.get_trend_short("PE")

        spot_trend = self.get_spot_trend()
        spot_range_3min = max(r["spot"] for r in self.oi_history[-4:]) - min(
            r["spot"] for r in self.oi_history[-4:]
        )

        total_volume = latest["total_volume"]
        avg_vol = np.mean([r["total_volume"] for r in self.oi_history[-10:]])
        volume_bias_pe = latest["volume_pe"] > latest["volume_ce"]
        volume_bias_ce = latest["volume_ce"] > latest["volume_pe"]
        volume_spike_recent = any(r["total_volume"] > 1.4 * avg_vol for r in self.oi_history[-3:])

        avg_oi = np.mean([r["ce_oi"] + r["pe_oi"] for r in self.oi_history[-10:]])
        higher_oi_threshold = max(avg_oi * HIGHER_OI_PCT, HIGHER_OI_FLOOR)
        oi_delta_threshold = max(avg_oi * OI_DELTA_PCT, OI_DELTA_FLOOR)
        lower_oi_threshold = max(avg_oi * LOWER_OI_PCT, LOWER_OI_FLOOR)

        return {
            "ratio_ce_pe": ratio_ce_pe,
            "ratio_pe_ce": ratio_pe_ce,
            "trend_ce_short": trend_ce_short,
            "trend_pe_short": trend_pe_short,
            "ce_change_5min": ce_change_5min,
            "pe_change_5min": pe_change_5min,
            "pe_change_1st_3min": pe_change_1st_3min,
            "pe_change_2nd_3min": pe_change_2nd_3min,
            "ce_change_1st_3min": ce_change_1st_3min,
            "ce_change_2nd_3min": ce_change_2nd_3min,
            "spot_range_3min": spot_range_3min,
            "total_volume": total_volume,
            "avg_vol": avg_vol,
            "spot_trend": spot_trend,
            "higher_oi_threshold": higher_oi_threshold,
            "oi_delta_threshold": oi_delta_threshold,
            "lower_oi_threshold": lower_oi_threshold,
            "volume_bias_ce": volume_bias_ce,
            "volume_bias_pe": volume_bias_pe,
            "volume_spike_recent": volume_spike_recent,
        }

    # ------------------------------------------------------------------ Strategy Tiers
    def evaluate_signals(self, ind: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        now = datetime.now()
        is_simulation = TRADING_MODE == "analyze"

        # Time restrictions checks
        # Mon/Wed-Fri Active window: 9:25 AM to 2:30 PM (14:30)
        # Tuesday Expiry Active window: 9:25 AM to 11:00 AM
        current_time = now.time()
        time_1230 = datetime.strptime("12:30:00", "%H:%M:%S").time()

        # In live mode:
        # Before 12:30 PM: Tiers 1-10 active.
        # After 12:30 PM: Tiers 1-6 active.
        # In simulation mode: Tiers 1-14 active all day.

        # TIER 1 - Extreme PE surge -> BUY CE (ATM - 100)
        if (
            ind["ratio_pe_ce"] > 10
            and ind["trend_pe_short"] != "Downtrend"
            and (ind["pe_change_5min"] - ind["ce_change_5min"]) > 2000000
        ):
            return "Tier 1", "CE"

        # TIER 2 - Extreme CE surge -> BUY PE (ATM + 100)
        if (
            ind["ratio_ce_pe"] > 10
            and ind["trend_ce_short"] != "Downtrend"
            and (ind["ce_change_5min"] - ind["pe_change_5min"]) > 2000000
        ):
            return "Tier 2", "PE"

        # TIER 3 - Higher PE surge -> BUY CE (ATM - 100)
        if (
            ind["ratio_pe_ce"] > 1.35
            and ind["trend_pe_short"] != "Downtrend"
            and (ind["pe_change_5min"] - ind["ce_change_5min"]) > ind["higher_oi_threshold"]
        ):
            return "Tier 3", "CE"

        # TIER 4 - Higher CE surge -> BUY PE (ATM + 100)
        if (
            ind["ratio_ce_pe"] > 1.35
            and ind["trend_ce_short"] != "Downtrend"
            and (ind["ce_change_5min"] - ind["pe_change_5min"]) > ind["higher_oi_threshold"]
        ):
            return "Tier 4", "PE"

        # TIER 5 - PE absorption -> BUY CE (ATM - 100)
        if (
            ind["pe_change_1st_3min"] > 2000000
            and ind["pe_change_2nd_3min"] > 2000000
            and ind["pe_change_1st_3min"] > ind["ce_change_1st_3min"]
            and ind["spot_range_3min"] < 15
            and ind["total_volume"] > ind["avg_vol"]
            and ind["ratio_pe_ce"] > 1.35
            and ind["spot_trend"] != "Downtrend"
        ):
            return "Tier 5", "CE"

        # TIER 6 - CE absorption -> BUY PE (ATM + 100)
        if (
            ind["ce_change_1st_3min"] > 2000000
            and ind["ce_change_2nd_3min"] > 2000000
            and ind["ce_change_1st_3min"] > ind["pe_change_1st_3min"]
            and ind["spot_range_3min"] < 15
            and ind["total_volume"] > ind["avg_vol"]
            and ind["ratio_ce_pe"] > 1.35
            and ind["spot_trend"] != "Uptrend"
        ):
            return "Tier 6", "PE"

        # Tiers 7-10 only active before 12:30 PM (or active all day in simulation)
        if is_simulation or current_time < time_1230:
            # TIER 7 - Strong CE surge -> BUY PE
            if (
                ind["ratio_ce_pe"] > 2
                and ind["trend_ce_short"] == "Uptrend"
                and ind["ce_change_5min"] > 0
                and (ind["ce_change_5min"] - ind["pe_change_5min"]) > ind["oi_delta_threshold"]
            ):
                return "Tier 7", "PE"

            # TIER 8 - Strong PE surge -> BUY CE
            if (
                ind["ratio_pe_ce"] > 2
                and ind["trend_pe_short"] == "Uptrend"
                and ind["pe_change_5min"] > 0
                and (ind["pe_change_5min"] - ind["ce_change_5min"]) > ind["oi_delta_threshold"]
            ):
                return "Tier 8", "CE"

            # TIER 9 - Sniper PE entry -> BUY PE
            if (
                ind["ratio_ce_pe"] > 1.25
                and ind["pe_change_1st_3min"] < 0
                and ind["ce_change_1st_3min"] > 3500000
                and ind["spot_trend"] == "Downtrend"
            ):
                return "Tier 9", "PE"

            # TIER 10 - Sniper CE entry -> BUY CE
            if (
                ind["ratio_pe_ce"] > 1.25
                and ind["ce_change_1st_3min"] < 0
                and ind["pe_change_1st_3min"] > 3500000
                and ind["spot_trend"] == "Uptrend"
            ):
                return "Tier 10", "CE"

        # Tiers 11-14 are simulation-only at all times
        if is_simulation:
            # TIER 11 - Lower CE surge + volume -> BUY PE
            if (
                ind["ratio_ce_pe"] > 1.3
                and self.get_past_five_3min_trend("CE") == "Uptrend"
                and (ind["ce_change_5min"] - ind["pe_change_5min"]) > ind["lower_oi_threshold"]
                and ind["volume_bias_ce"]
                and ind["volume_spike_recent"]
            ):
                return "Tier 11", "PE"

            # TIER 12 - Lower PE surge + volume -> BUY CE
            if (
                ind["ratio_pe_ce"] > 1.3
                and self.get_past_five_3min_trend("PE") == "Uptrend"
                and (ind["pe_change_5min"] - ind["ce_change_5min"]) > ind["lower_oi_threshold"]
                and ind["volume_bias_pe"]
                and ind["volume_spike_recent"]
            ):
                return "Tier 12", "CE"

            # TIER 13 - High volume spike + PE bias -> BUY CE
            if (
                ind["volume_spike_recent"]
                and ind["volume_bias_pe"]
                and ind["pe_change_5min"] > ind["lower_oi_threshold"]
            ):
                return "Tier 13", "CE"

            # TIER 14 - High volume spike + CE bias -> BUY PE
            if (
                ind["volume_spike_recent"]
                and ind["volume_bias_ce"]
                and ind["ce_change_5min"] > ind["lower_oi_threshold"]
            ):
                return "Tier 14", "PE"

        return None

    # ------------------------------------------------------------------ Executions
    def check_signals(self):
        """Fetch chain, evaluate indicators, and enter trade if signal triggers."""
        now = datetime.now()

        # Enforce trading windows
        if not self.can_place_new_trades(now):
            return

        # Fetch option chain
        logger.info("Polling option chain for signals...")
        expiry_date = self.get_nearest_nifty_expiry()
        if not expiry_date:
            logger.warning("Could not resolve nearest expiry date for NIFTY. Signal check aborted.")
            return

        chain_data = self.client.optionchain(
            underlying=SYMBOL,
            exchange=UNDERLYING_EXCHANGE,
            expiry_date=expiry_date,
            strike_count=15,
        )
        if not isinstance(chain_data, dict) or chain_data.get("status") != "success":
            logger.warning(f"Failed to fetch option chain: {chain_data}")
            return

        spot = chain_data.get("underlying_ltp", 0.0)
        atm_strike = chain_data.get("atm_strike", 0.0)
        expiry = chain_data.get("expiry_date", "")

        # Compute total CE and PE OI and volumes across the chain
        ce_oi, pe_oi = 0.0, 0.0
        volume_ce, volume_pe = 0.0, 0.0
        for item in chain_data.get("chain", []):
            ce = item.get("ce") or {}
            pe = item.get("pe") or {}
            ce_oi += ce.get("oi", 0.0)
            pe_oi += pe.get("oi", 0.0)
            volume_ce += ce.get("volume", 0.0)
            volume_pe += pe.get("volume", 0.0)

        # Update in-memory history
        new_row = {
            "timestamp": now,
            "spot": spot,
            "atm_strike": atm_strike,
            "ce_oi": ce_oi,
            "pe_oi": pe_oi,
            "total_volume": volume_ce + volume_pe,
            "volume_ce": volume_ce,
            "volume_pe": volume_pe,
        }
        self.oi_history.append(new_row)
        if len(self.oi_history) > 60:
            self.oi_history.pop(0)

        # Save to CSV disabled (now utilizing in-memory history rollup)

        if len(self.oi_history) < MIN_OI_ROWS:
            logger.info(f"Warming up indicators ({len(self.oi_history)}/{MIN_OI_ROWS} rows).")
            return

        # Calculate indicators
        ind = self.calculate_indicators()

        # Check signals
        signal_result = self.evaluate_signals(ind)
        if not signal_result:
            return

        signal_name, option_type = signal_result
        logger.info(f"💥 SIGNAL FIRED: {signal_name} ({option_type})")

        # Resolve strike symbol & LTP
        trading_symbol, opt_ltp = self.get_option_details_from_chain(chain_data, option_type)
        if not trading_symbol or opt_ltp is None:
            logger.warning(f"Could not resolve option details for strike ATM {option_type}")
            return

        if opt_ltp < MIN_OPTION_PREMIUM:
            logger.info(
                f"Signal skipped: Option premium ({opt_ltp} pts) is below minimum of {MIN_OPTION_PREMIUM} pts."
            )
            return

        # Fetch RSI & Supertrend details for log
        trend_list = self._fetch_and_gate(trading_symbol)

        # Execute Order
        logger.info(f"Placing LIMIT order for {trading_symbol} @ {opt_ltp}")
        resp = self.client.optionsorder(
            strategy=self.strategy_name,
            underlying=SYMBOL,
            exchange=UNDERLYING_EXCHANGE,
            offset="ITM2",
            option_type=option_type,
            action="BUY",
            quantity=QUANTITY,
            product=PRODUCT,
            expiry_date=expiry,
            price_type="LIMIT",
            price=str(opt_ltp),
        )

        if isinstance(resp, dict) and resp.get("status") == "success":
            resolved_symbol = resp.get("symbol", trading_symbol)
            order_id = resp.get("orderid", "MOCK-ORDER")
            fill_price = opt_ltp  # fallback to quote LTP if execution details are not returned

            logger.info(
                f"✅ Position opened: {resolved_symbol} @ {fill_price} | Order ID: {order_id}"
            )

            # Fetch India VIX to adjust SL
            vix = self.get_india_vix()
            high_vix = vix > VIX_HIGH_THRESHOLD
            sl_min = INITIAL_SL_MIN_HIGH_VIX if high_vix else INITIAL_SL_MIN
            sl_max = INITIAL_SL_MAX_HIGH_VIX if high_vix else INITIAL_SL_MAX

            # Initial SL & Trailing state
            initial_sl = max(sl_min, min(sl_max, round(fill_price * INITIAL_SL_PCT)))
            self.current_sl_trigger = max(1.0, fill_price - initial_sl)
            self.buffer_price = fill_price + initial_sl
            self.trail_count = 0
            self.best_premium = fill_price
            self.early_adverse_checked = False

            self.position = {
                "trading_symbol": resolved_symbol,
                "option_type": option_type,
                "entry_price": fill_price,
                "entry_time": now,
                "vix": vix,
                "expiry_date": expiry,
                "trend_list": trend_list,
                "strike": atm_strike - 100 if option_type == "CE" else atm_strike + 100,
                "initial_sl": initial_sl,
            }
            self.ltp_history = [fill_price]

            # Placed order logging to CSV disabled (state maintained in-memory and recovered via API)

            # Send Notification
            self.send_alert(
                "ENTRY",
                f"Opened Long {option_type} ({resolved_symbol})\n"
                f"Entry Price: {fill_price:.2f}\n"
                f"Initial SL: {initial_sl:.2f} (Trigger: {self.current_sl_trigger:.2f})\n"
                f"Signal: {signal_name}\n"
                f"VIX: {vix:.2f}",
            )
        else:
            logger.error(f"❌ Entry order failed: {resp}")

    # ------------------------------------------------------------------ Exit Manager
    def close_position(self, reason: str):
        if not self.position:
            return

        symbol = self.position["trading_symbol"]
        option_type = self.position["option_type"]
        entry_price = self.position["entry_price"]

        # Fetch final quote price
        exit_price = entry_price
        try:
            q = self.client.quotes(symbol=symbol, exchange=OPTIONS_EXCHANGE)
            if isinstance(q, dict) and q.get("status") == "success":
                exit_price = float(q.get("data", {}).get("ltp", entry_price))
        except Exception as e:
            logger.warning(f"Could not fetch exit quote for {symbol}: {e}")

        logger.info(f"Exiting position for {symbol} | Reason: {reason} | Est price: {exit_price}")

        # Place Exit Order
        resp = self.client.optionsorder(
            strategy=self.strategy_name,
            underlying=SYMBOL,
            exchange=UNDERLYING_EXCHANGE,
            offset="ITM2",
            option_type=option_type,
            action="SELL",
            quantity=QUANTITY,
            product=PRODUCT,
            expiry_date=self.position["expiry_date"],
            price_type="MARKET",
        )

        if isinstance(resp, dict) and resp.get("status") == "success":
            pnl = exit_price - entry_price
            logger.info(f"✅ Closed position {symbol} @ {exit_price} | P&L: {pnl:.2f} pts")

            # Exit trade logging to CSV disabled (state maintained in-memory and recovered via API)

            # Send Notification
            self.send_alert(
                "EXIT",
                f"Closed Long {option_type} ({symbol})\n"
                f"Exit Price: {exit_price:.2f}\n"
                f"Entry Price: {entry_price:.2f}\n"
                f"PnL: {pnl:.2f} pts\n"
                f"Reason: {reason}",
            )

            # Reset states and apply cooldown
            self.position = None
            self.cooldown_until = datetime.now() + timedelta(minutes=REENTRY_COOLDOWN_MINUTES)
            logger.info(
                f"Re-entry cooldown activated. Blocked until {self.cooldown_until.strftime('%H:%M:%S')}"
            )
        else:
            logger.error(f"❌ Exit order failed: {resp}")

    def check_oi_reversal(self) -> bool:
        """Calculate dynamic OI reversal. Returns True if reversal is confirmed."""
        if len(self.oi_history) < 11:
            return False

        # Calculate avg_oi
        avg_oi = np.mean([r["ce_oi"] + r["pe_oi"] for r in self.oi_history[-10:]])
        delta_threshold = max(avg_oi * DYNAMIC_REVERSAL_OI_PCT, REVERSAL_OI_FLOOR)

        # 10-row change (difference between latest and 10 rows ago)
        ce_change_10min = self.oi_history[-1]["ce_oi"] - self.oi_history[-11]["ce_oi"]
        pe_change_10min = self.oi_history[-1]["pe_oi"] - self.oi_history[-11]["pe_oi"]

        ce_surge = False
        pe_surge = False

        if (ce_change_10min - pe_change_10min) > delta_threshold:
            ce_surge = True
        elif (pe_change_10min - ce_change_10min) > delta_threshold:
            pe_surge = True
        else:
            # Fallback: check 4-block trend direction
            trend_ce = self.get_trend_short("CE")
            trend_pe = self.get_trend_short("PE")
            if trend_ce == "Uptrend":
                ce_surge = True
            elif trend_pe == "Uptrend":
                pe_surge = True

        held_side = self.position["option_type"]
        # Reversal triggers if market writes on the held side (headwind)
        if held_side == "CE" and ce_surge:
            return True
        if held_side == "PE" and pe_surge:
            return True

        return False

    def monitor_position(self):
        """Checked every 15 seconds. Manages trailing stop loss, adverse cuts, and reversal exit checks."""
        if not self.position:
            return

        now = datetime.now()
        symbol = self.position["trading_symbol"]
        entry_price = self.position["entry_price"]
        entry_time = self.position["entry_time"]

        # Forced EOD Exit (3:15 PM)
        if now.time() >= datetime.strptime("15:15:00", "%H:%M:%S").time():
            self.close_position("EOD")
            return

        # Tuesday Expiry Exit (1:30 PM on Expiry Day)
        if now.weekday() == 1 and now.time() >= datetime.strptime("13:30:00", "%H:%M:%S").time():
            self.close_position("TUESDAY_EXPIRY_EXIT")
            return

        # Fetch option LTP quote
        try:
            q = self.client.quotes(symbol=symbol, exchange=OPTIONS_EXCHANGE)
            if isinstance(q, dict) and q.get("status") == "success":
                ltp = float(q.get("data", {}).get("ltp", 0.0))
            else:
                logger.warning(f"Could not get LTP quote: {q}")
                return
        except Exception as e:
            logger.error(f"Error fetching quote for exit check: {e}")
            return

        if ltp <= 0.0:
            return

        # Append to LTP history
        self.ltp_history.append(ltp)
        if len(self.ltp_history) > 30:
            self.ltp_history.pop(0)

        # 1. Stop-Loss Trigger Check
        if ltp <= self.current_sl_trigger:
            reason = (
                "SL_HIT_REVERSAL"
                if self.trail_count > 0
                or self.current_sl_trigger > (entry_price - self.position["initial_sl"])
                else "SL_HIT"
            )
            self.close_position(reason)
            return

        # 2. Early Adverse cut (checked exactly once at 5 minutes after entry)
        elapsed_seconds = (now - entry_time).total_seconds()
        if elapsed_seconds >= (EARLY_ADVERSE_MINUTES * 60) and not self.early_adverse_checked:
            self.early_adverse_checked = True
            if ltp <= entry_price - EARLY_ADVERSE_PTS:
                logger.info(
                    f"Early adverse hit: option is underwater {entry_price - ltp:.2f} pts at 5 min mark."
                )
                self.close_position("EARLY_ADVERSE")
                return

        # 3. Trailing SL Adjustment
        if ltp > self.best_premium:
            self.best_premium = ltp

        if ltp > self.buffer_price:
            profit = ltp - entry_price
            if profit <= 30.0:
                gap = TRAIL_GAP_NORMAL
                inc = TRAIL_BUFFER_FIRST
            else:
                gap = TRAIL_GAP_FAST
                inc = TRAIL_BUFFER_AFTER_PROFIT

            candidate_sl = ltp - gap
            if candidate_sl > self.current_sl_trigger:
                self.current_sl_trigger = candidate_sl
                self.buffer_price = ltp + inc
                self.trail_count += 1
                logger.info(
                    f"📈 Trailing SL moved up to {self.current_sl_trigger:.2f} (buffer: {self.buffer_price:.2f})"
                )

        # 4. OI Reversal exit checks
        if self.check_oi_reversal():
            # Check price confirmations: Compare latest to price 3 min ago (12 ticks ago)
            if len(self.ltp_history) >= 13:
                price_3min_ago = self.ltp_history[-13]
                price_change = ltp - price_3min_ago

                # LTP-momentum veto check: if price also rolled over (fallen >= 3 pts)
                if price_change <= LTP_MOM_THRESHOLD:
                    logger.info(
                        f"OI reversal confirmed by price momentum rollover ({price_change:.2f} pts in last 3 min)."
                    )
                    self.close_position("OI_REVERSAL_MOM")
                    return

            # Otherwise, tighten SL trigger to LTP - 5 (only move tighter)
            candidate_sl = ltp - REVERSAL_TIGHTEN_GAP
            if candidate_sl > self.current_sl_trigger:
                self.current_sl_trigger = candidate_sl
                logger.info(
                    f"⚠️ OI reversal detected without momentum rollover. Tightening SL to {self.current_sl_trigger:.2f}"
                )

    # ------------------------------------------------------------------ Main loop
    def run(self):
        logger.info("Bot execution started. Polling every 15 seconds.")
        self.check_reentry_and_recovery()
        self.warmup_history_from_api()

        while self.running:
            try:
                now = datetime.now()

                if not self.is_market_open(now):
                    logger.debug("Market is closed. Sleeping 15 seconds...")
                    time.sleep(15)
                    continue

                if self.position:
                    # Monitor exits if we hold an active trade
                    self.monitor_position()
                else:
                    # If we don't hold an active trade, pull chain and scan on the turn of the minute
                    current_min = now.minute
                    if current_min != self.last_chain_fetch_min:
                        # Clear cooldown if elapsed
                        if self.cooldown_until and now >= self.cooldown_until:
                            logger.info("Re-entry cooldown has expired.")
                            self.cooldown_until = None

                        if not self.cooldown_until:
                            self.last_chain_fetch_min = current_min
                            self.check_signals()
                        else:
                            # Still in cooldown, skip signal check but log once a minute
                            if current_min % 5 == 0:  # log every 5 mins
                                logger.info(
                                    f"Cooldown active until {self.cooldown_until.strftime('%H:%M:%S')}"
                                )

                time.sleep(15)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Stopping bot...")
                self.running = False
            except Exception as e:
                logger.error(f"Error in main loop cycle: {e}", exc_info=True)
                time.sleep(15)


if __name__ == "__main__":
    NiftyOISurgerBot().run()
