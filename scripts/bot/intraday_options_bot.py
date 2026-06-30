"""
Intraday Options Trading Bot for Indian Markets (NSE/NFO)
---------------------------------------------------------
A comprehensive options trading bot implementing best practices for intraday
options trading in India. Supports directional scalps, long straddles,
bull/bear spreads, and iron condors with full risk management.

Features:
  • Full auto-scan of F&O universe to pick best underlyings by OI, volume, IV
  • 4 strategy modes: directional, straddle, spread, iron_condor
  • Option chain analysis with Greeks, PCR, OI buildup detection
  • Per-trade risk caps, daily drawdown limits, auto square-off
  • Market hours management (pre-market → active → wind-down → close)
  • Trade journaling (CSV) and WhatsApp notifications
  • Analyze mode (paper trading) by default for safety

Usage:
  TRADING_MODE=analyze STRATEGY_MODE=directional python scripts/intraday_options_bot.py
"""

import csv
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("OptionsBot")

try:
    from openalgo import api
except ImportError:
    logger.error("The 'openalgo' package is required. Install with: pip install openalgo")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- API Connection ---
API_KEY = os.getenv(
    "OPENALGO_API_KEY",
    "b45feb0a6973ed00fe86d25ace49d4da8dfe8d0a78c334455d46254ded28a26d",
)
API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

# --- WhatsApp Notifications ---
WHATSAPP_PHONES: list[str] = [
    n.strip()
    for n in os.getenv("WHATSAPP_PHONES", "919566029048,919790856795").split(",")
    if n.strip()
]

# --- Trading Mode ---
# "analyze" = paper trading (safe default), "live" = real orders
TRADING_MODE = os.getenv("TRADING_MODE", "analyze").lower()

# --- Underlying & Exchange ---
# Primary underlying (also used for WebSocket LTP feed)
UNDERLYING = os.getenv("UNDERLYING", "NIFTY")
UNDERLYING_EXCHANGE = os.getenv("UNDERLYING_EXCHANGE", "NSE_INDEX")
OPTIONS_EXCHANGE = os.getenv("OPTIONS_EXCHANGE", "NFO")

# --- F&O Auto-Scan Watchlist ---
# Top liquid F&O underlyings for auto-scanning.
# The bot fetches option chains for each and ranks by OI + volume + IV score.
DEFAULT_WATCHLIST = [
    # Indices
    {"symbol": "NIFTY", "exchange": "NSE_INDEX", "opt_exchange": "NFO", "lot": 75},
    {"symbol": "BANKNIFTY", "exchange": "NSE_INDEX", "opt_exchange": "NFO", "lot": 30},
    {"symbol": "FINNIFTY", "exchange": "NSE_INDEX", "opt_exchange": "NFO", "lot": 40},
    {"symbol": "MIDCPNIFTY", "exchange": "NSE_INDEX", "opt_exchange": "NFO", "lot": 75},
    # Large-cap stocks (high F&O liquidity)
    {"symbol": "RELIANCE", "exchange": "NSE", "opt_exchange": "NFO", "lot": 250},
    {"symbol": "TCS", "exchange": "NSE", "opt_exchange": "NFO", "lot": 150},
    {"symbol": "HDFCBANK", "exchange": "NSE", "opt_exchange": "NFO", "lot": 550},
    {"symbol": "INFY", "exchange": "NSE", "opt_exchange": "NFO", "lot": 400},
    {"symbol": "ICICIBANK", "exchange": "NSE", "opt_exchange": "NFO", "lot": 700},
    {"symbol": "SBIN", "exchange": "NSE", "opt_exchange": "NFO", "lot": 750},
    {"symbol": "AXISBANK", "exchange": "NSE", "opt_exchange": "NFO", "lot": 600},
    {"symbol": "BAJFINANCE", "exchange": "NSE", "opt_exchange": "NFO", "lot": 125},
    {"symbol": "TATAMOTORS", "exchange": "NSE", "opt_exchange": "NFO", "lot": 575},
    {"symbol": "LT", "exchange": "NSE", "opt_exchange": "NFO", "lot": 150},
    {"symbol": "MARUTI", "exchange": "NSE", "opt_exchange": "NFO", "lot": 50},
    {"symbol": "ITC", "exchange": "NSE", "opt_exchange": "NFO", "lot": 1600},
    {"symbol": "KOTAKBANK", "exchange": "NSE", "opt_exchange": "NFO", "lot": 400},
    {"symbol": "TATASTEEL", "exchange": "NSE", "opt_exchange": "NFO", "lot": 3350},
    {"symbol": "WIPRO", "exchange": "NSE", "opt_exchange": "NFO", "lot": 1500},
    {"symbol": "BHARTIARTL", "exchange": "NSE", "opt_exchange": "NFO", "lot": 475},
    {"symbol": "HCLTECH", "exchange": "NSE", "opt_exchange": "NFO", "lot": 350},
    {"symbol": "SUNPHARMA", "exchange": "NSE", "opt_exchange": "NFO", "lot": 350},
    {"symbol": "ADANIENT", "exchange": "NSE", "opt_exchange": "NFO", "lot": 250},
    {"symbol": "ONGC", "exchange": "NSE", "opt_exchange": "NFO", "lot": 3850},
    {"symbol": "POWERGRID", "exchange": "NSE", "opt_exchange": "NFO", "lot": 2700},
    {"symbol": "HINDUNILVR", "exchange": "NSE", "opt_exchange": "NFO", "lot": 300},
    {"symbol": "COALINDIA", "exchange": "NSE", "opt_exchange": "NFO", "lot": 2100},
    {"symbol": "DRREDDY", "exchange": "NSE", "opt_exchange": "NFO", "lot": 125},
    {"symbol": "M&M", "exchange": "NSE", "opt_exchange": "NFO", "lot": 350},
    {"symbol": "JSWSTEEL", "exchange": "NSE", "opt_exchange": "NFO", "lot": 675},
]

# User can override with a comma-separated list, e.g. "NIFTY,BANKNIFTY,RELIANCE"
_custom_wl = os.getenv("WATCHLIST", "")
if _custom_wl.strip():
    SCAN_WATCHLIST = []
    for sym in _custom_wl.split(","):
        sym = sym.strip().upper()
        # Find in default list or create a basic entry
        found = next((w for w in DEFAULT_WATCHLIST if w["symbol"] == sym), None)
        if found:
            SCAN_WATCHLIST.append(found)
        else:
            SCAN_WATCHLIST.append(
                {"symbol": sym, "exchange": "NSE", "opt_exchange": "NFO", "lot": 100}
            )
else:
    SCAN_WATCHLIST = DEFAULT_WATCHLIST

MAX_SCAN_PICKS = int(os.getenv("MAX_SCAN_PICKS", "3"))  # Top N picks from scanner

# --- Strategy Settings ---
STRATEGY_MODE = os.getenv("STRATEGY_MODE", "directional").lower()
# directional, straddle, spread, iron_condor

# Strike offsets for each strategy
DIRECTIONAL_OFFSET = os.getenv("DIRECTIONAL_OFFSET", "ATM")  # ATM, ITM1, OTM1
SPREAD_WIDTH = int(os.getenv("SPREAD_WIDTH", "3"))  # OTM distance for short leg
CONDOR_SHORT_OFFSET = int(os.getenv("CONDOR_SHORT_OFFSET", "5"))  # OTM5
CONDOR_LONG_OFFSET = int(os.getenv("CONDOR_LONG_OFFSET", "10"))  # OTM10

# --- Risk Management ---
MAX_CAPITAL = float(os.getenv("MAX_CAPITAL", "200000"))  # Total capital for options
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))  # % risk per trade
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))  # Daily drawdown cap
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "2"))  # Max concurrent trades
PREMIUM_SL_PCT = float(os.getenv("PREMIUM_SL_PCT", "50.0"))  # Stop if premium drops 50%
PREMIUM_TARGET_PCT = float(os.getenv("PREMIUM_TARGET_PCT", "100.0"))  # Target 100% gain
PRODUCT = os.getenv("PRODUCT", "MIS")

# --- Candle & Signal ---
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "5m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "5"))
SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", "30"))  # seconds

# --- Indicator Parameters ---
EMA_FAST_LEN = int(os.getenv("EMA_FAST_LEN", "9"))
EMA_SLOW_LEN = int(os.getenv("EMA_SLOW_LEN", "21"))
EMA_TREND_LEN = int(os.getenv("EMA_TREND_LEN", "55"))
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
MIN_CONFLUENCE_SCORE = float(os.getenv("MIN_CONFLUENCE_SCORE", "5.0"))

# --- Market Hours ---
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 15
NO_NEW_TRADES_HOUR = int(os.getenv("NO_NEW_TRADES_HOUR", "14"))
NO_NEW_TRADES_MIN = int(os.getenv("NO_NEW_TRADES_MIN", "30"))
SQUAREOFF_HOUR = int(os.getenv("SQUAREOFF_HOUR", "15"))
SQUAREOFF_MIN = int(os.getenv("SQUAREOFF_MIN", "20"))
FIRST_CANDLE_WAIT_MIN = int(os.getenv("FIRST_CANDLE_WAIT_MIN", "15"))

# --- Trade Journal ---
JOURNAL_DIR = os.getenv(
    "JOURNAL_DIR",
    str(Path(__file__).parent / "trade_journals"),
)

# --- IV Thresholds ---
IV_MAX_THRESHOLD = float(os.getenv("IV_MAX_THRESHOLD", "40.0"))  # Skip if IV > 40%
IV_MIN_THRESHOLD = float(os.getenv("IV_MIN_THRESHOLD", "8.0"))  # Skip if IV < 8%


# ==============================================================================
# INDICATORS
# ==============================================================================
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (intraday reset)."""
    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    df2["date"] = df2["timestamp"].dt.date
    df2["pv"] = df2["close"] * df2["volume"]
    cum_pv = df2.groupby("date")["pv"].cumsum()
    cum_vol = df2.groupby("date")["volume"].cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def compute_dmi(df: pd.DataFrame, period: int = 14):
    """Directional Movement Index — +DI, -DI, ADX."""
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
# OPTION CHAIN ANALYZER
# ==============================================================================
class OptionChainAnalyzer:
    """Analyzes option chain data for strike selection, OI, IV, and PCR."""

    def __init__(self, client):
        self.client = client

    def fetch_chain(
        self, underlying: str, exchange: str, expiry_date: str = None, strike_count: int = 15
    ) -> dict | None:
        """Fetch option chain via OpenAlgo SDK."""
        try:
            kwargs = {
                "underlying": underlying,
                "exchange": exchange,
                "strike_count": strike_count,
            }
            if expiry_date:
                kwargs["expiry_date"] = expiry_date
            resp = self.client.optionchain(**kwargs)
            if isinstance(resp, dict) and resp.get("status") == "success":
                return resp
            logger.warning(f"Option chain fetch failed for {underlying}: {resp}")
            return None
        except Exception as e:
            logger.error(f"Error fetching option chain for {underlying}: {e}")
            return None

    def compute_pcr(self, chain_data: dict) -> float:
        """Compute Put-Call Ratio from OI data. PCR > 1 = bullish, < 1 = bearish."""
        total_call_oi = 0
        total_put_oi = 0
        for strike_item in chain_data.get("chain", []):
            ce = strike_item.get("ce") or {}
            pe = strike_item.get("pe") or {}
            total_call_oi += ce.get("oi", 0)
            total_put_oi += pe.get("oi", 0)
        if total_call_oi == 0:
            return 0.0
        return total_put_oi / total_call_oi

    def max_oi_strikes(self, chain_data: dict) -> dict:
        """Find strikes with maximum Call OI and Put OI (support/resistance levels)."""
        max_call_oi, max_call_strike = 0, 0
        max_put_oi, max_put_strike = 0, 0
        for strike_item in chain_data.get("chain", []):
            ce = strike_item.get("ce") or {}
            pe = strike_item.get("pe") or {}
            if ce.get("oi", 0) > max_call_oi:
                max_call_oi = ce["oi"]
                max_call_strike = strike_item["strike"]
            if pe.get("oi", 0) > max_put_oi:
                max_put_oi = pe["oi"]
                max_put_strike = strike_item["strike"]
        return {
            "max_call_oi_strike": max_call_strike,
            "max_call_oi": max_call_oi,
            "max_put_oi_strike": max_put_strike,
            "max_put_oi": max_put_oi,
        }

    def total_oi_and_volume(self, chain_data: dict) -> dict:
        """Compute total OI and volume across the chain."""
        total_oi = 0
        total_volume = 0
        for strike_item in chain_data.get("chain", []):
            for side in ("ce", "pe"):
                opt = strike_item.get(side) or {}
                total_oi += opt.get("oi", 0)
                total_volume += opt.get("volume", 0)
        return {"total_oi": total_oi, "total_volume": total_volume}

    def atm_iv_estimate(self, chain_data: dict) -> float:
        """Estimate ATM implied volatility from the chain's ATM strike."""
        atm_strike = chain_data.get("atm_strike", 0)
        for strike_item in chain_data.get("chain", []):
            if strike_item["strike"] == atm_strike:
                ce = strike_item.get("ce") or {}
                pe = strike_item.get("pe") or {}
                # Use LTP-based rough IV proxy: (CE_LTP + PE_LTP) / underlying_LTP * annualization
                ce_ltp = ce.get("ltp", 0)
                pe_ltp = pe.get("ltp", 0)
                underlying_ltp = chain_data.get("underlying_ltp", 1)
                if underlying_ltp > 0:
                    # Rough straddle premium ratio as IV proxy
                    # Annualize assuming 7 days to expiry (weekly)
                    straddle_pct = (ce_ltp + pe_ltp) / underlying_ltp * 100
                    return straddle_pct * np.sqrt(365 / 7)  # rough annualized
        return 0.0

    def bid_ask_quality(self, chain_data: dict) -> float:
        """Score bid-ask spread quality at ATM (0-1, higher = tighter = better)."""
        atm_strike = chain_data.get("atm_strike", 0)
        scores = []
        for strike_item in chain_data.get("chain", []):
            if abs(strike_item["strike"] - atm_strike) <= atm_strike * 0.01:
                for side in ("ce", "pe"):
                    opt = strike_item.get(side) or {}
                    bid = opt.get("bid", 0)
                    ask = opt.get("ask", 0)
                    if ask > 0 and bid > 0:
                        spread_pct = (ask - bid) / ask
                        scores.append(max(0, 1 - spread_pct * 10))  # Penalty for wide spread
        return np.mean(scores) if scores else 0.5

    def get_greeks(self, symbol: str, exchange: str, interest_rate: float = 6.5) -> dict | None:
        """Fetch option Greeks via SDK."""
        try:
            resp = self.client.optiongreeks(
                symbol=symbol, exchange=exchange, interest_rate=interest_rate
            )
            if isinstance(resp, dict) and resp.get("status") == "success":
                return resp
            return None
        except Exception as e:
            logger.warning(f"Greeks fetch failed for {symbol}: {e}")
            return None


# ==============================================================================
# AUTO-SCANNER / STOCK SCREENER
# ==============================================================================
class FnOScanner:
    """Scans the F&O watchlist and ranks underlyings for options trading."""

    def __init__(self, client, chain_analyzer: OptionChainAnalyzer):
        self.client = client
        self.analyzer = chain_analyzer

    def scan_and_rank(self, watchlist: list[dict], max_picks: int = 3) -> list[dict]:
        """
        Scan each underlying in the watchlist, fetch option chain,
        and rank by a composite score of:
          - Total OI (liquidity)
          - Total Volume (activity)
          - PCR alignment (sentiment)
          - Bid-ask quality (execution quality)
          - IV in range (not too expensive / too cheap)

        Returns top N picks sorted by score descending.
        """
        candidates = []
        logger.info(f"🔍 Scanning {len(watchlist)} F&O underlyings...")

        for item in watchlist:
            sym = item["symbol"]
            exch = item["exchange"]
            try:
                chain = self.analyzer.fetch_chain(sym, exch, strike_count=10)
                if chain is None:
                    continue

                oi_vol = self.analyzer.total_oi_and_volume(chain)
                pcr = self.analyzer.compute_pcr(chain)
                baq = self.analyzer.bid_ask_quality(chain)
                iv_est = self.analyzer.atm_iv_estimate(chain)
                max_oi = self.analyzer.max_oi_strikes(chain)
                underlying_ltp = chain.get("underlying_ltp", 0)
                atm_strike = chain.get("atm_strike", 0)
                expiry = chain.get("expiry_date", "")

                # Composite scoring (0-100 scale)
                score = 0.0

                # OI score (normalize: more OI = better, up to 30 points)
                oi_score = min(oi_vol["total_oi"] / 100000, 30)
                score += oi_score

                # Volume score (up to 20 points)
                vol_score = min(oi_vol["total_volume"] / 50000, 20)
                score += vol_score

                # Bid-ask quality (up to 15 points)
                score += baq * 15

                # IV in range (up to 20 points — best if IV between 15-30)
                if IV_MIN_THRESHOLD < iv_est < IV_MAX_THRESHOLD:
                    iv_score = 20 - abs(iv_est - 22) * 0.5  # Sweet spot ~22%
                    score += max(0, iv_score)

                # PCR bonus: meaningful PCR (not extreme) → 15 points
                if 0.5 < pcr < 1.5:
                    pcr_score = 15 - abs(pcr - 1.0) * 10
                    score += max(0, pcr_score)

                candidates.append(
                    {
                        "symbol": sym,
                        "exchange": exch,
                        "opt_exchange": item["opt_exchange"],
                        "lot_size": item["lot"],
                        "score": round(score, 1),
                        "underlying_ltp": underlying_ltp,
                        "atm_strike": atm_strike,
                        "expiry_date": expiry,
                        "total_oi": oi_vol["total_oi"],
                        "total_volume": oi_vol["total_volume"],
                        "pcr": round(pcr, 2),
                        "iv_estimate": round(iv_est, 1),
                        "bid_ask_quality": round(baq, 2),
                        "max_call_oi_strike": max_oi["max_call_oi_strike"],
                        "max_put_oi_strike": max_oi["max_put_oi_strike"],
                    }
                )
                logger.info(
                    f"  {sym}: Score={score:.1f} | OI={oi_vol['total_oi']:,} | "
                    f"Vol={oi_vol['total_volume']:,} | PCR={pcr:.2f} | IV≈{iv_est:.1f}%"
                )

            except Exception as e:
                logger.warning(f"  {sym}: Scan error — {e}")
                continue

            # Small delay to avoid API throttling
            time.sleep(0.3)

        # Sort by score descending and pick top N
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_picks = candidates[:max_picks]

        if top_picks:
            logger.info("📊 Top picks for today:")
            for i, pick in enumerate(top_picks, 1):
                logger.info(
                    f"  #{i} {pick['symbol']} (Score: {pick['score']}) — "
                    f"LTP: {pick['underlying_ltp']}, ATM: {pick['atm_strike']}, "
                    f"PCR: {pick['pcr']}, IV: {pick['iv_estimate']}%"
                )
        else:
            logger.warning("⚠️ No suitable underlyings found in scan!")

        return top_picks


# ==============================================================================
# TRADE JOURNAL
# ==============================================================================
class TradeJournal:
    """CSV-based trade journal for post-market analysis."""

    def __init__(self, journal_dir: str):
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._today_file = None
        self._writer = None
        self._fh = None

    def _get_today_file(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        filepath = self.journal_dir / f"trades_{today_str}.csv"
        if self._today_file != filepath:
            if self._fh:
                self._fh.close()
            is_new = not filepath.exists()
            self._fh = open(filepath, "a", newline="")
            self._writer = csv.writer(self._fh)
            if is_new:
                self._writer.writerow(
                    [
                        "timestamp",
                        "underlying",
                        "option_symbol",
                        "strategy",
                        "action",
                        "direction",
                        "entry_price",
                        "exit_price",
                        "quantity",
                        "pnl",
                        "reason",
                        "duration_min",
                    ]
                )
            self._today_file = filepath
        return self._writer

    def log_trade(
        self,
        underlying: str,
        option_symbol: str,
        strategy: str,
        action: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        pnl: float,
        reason: str,
        duration_min: float,
    ):
        writer = self._get_today_file()
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                underlying,
                option_symbol,
                strategy,
                action,
                direction,
                f"{entry_price:.2f}",
                f"{exit_price:.2f}",
                quantity,
                f"{pnl:.2f}",
                reason,
                f"{duration_min:.1f}",
            ]
        )
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


# ==============================================================================
# INTRADAY OPTIONS BOT
# ==============================================================================
class IntradayOptionsBot:
    """
    Main bot class orchestrating the full intraday options trading workflow:
    auto-scan → signal generation → strategy execution → risk management → exit.
    """

    def __init__(self):
        self.client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)
        self.strategy_name = os.getenv("STRATEGY_NAME", "IntradayOptions_v1")

        # Sub-components
        self.chain_analyzer = OptionChainAnalyzer(self.client)
        self.scanner = FnOScanner(self.client, self.chain_analyzer)
        self.journal = TradeJournal(JOURNAL_DIR)

        # Runtime state
        self.running = True
        self.stop_event = threading.Event()
        self.ltp = None  # Primary underlying LTP
        self.ltp_map: dict[str, float] = {}  # Multi-underlying LTP cache
        self.instrument = [{"exchange": UNDERLYING_EXCHANGE, "symbol": UNDERLYING}]

        # Scan results (filled during pre-market)
        self.todays_picks: list[dict] = []

        # Position tracking
        self.positions: list[dict] = []
        # Each position dict: {
        #   "id": int, "underlying": str, "exchange": str, "opt_exchange": str,
        #   "strategy": str, "direction": str (bull/bear/neutral),
        #   "legs": [{"symbol": str, "action": str, "quantity": int, "entry_price": float}],
        #   "entry_time": datetime, "lot_size": int,
        #   "total_premium_paid": float, "sl_price": float, "target_price": float,
        #   "status": "open"/"closed", "exit_reason": str, "pnl": float
        # }
        self._next_position_id = 1

        # Daily P&L tracking
        self.daily_realized_pnl = 0.0
        self.daily_trade_count = 0
        self.scan_done_today = False
        self.first_candle_waited = False
        self.last_scan_date = None

        logger.info(
            f"🚀 Intraday Options Bot initialized\n"
            f"   Mode: {'📝 ANALYZE (Paper)' if TRADING_MODE == 'analyze' else '🔴 LIVE TRADING'}\n"
            f"   Strategy: {STRATEGY_MODE.upper()}\n"
            f"   Risk: {RISK_PER_TRADE_PCT}% per trade, {MAX_DAILY_LOSS_PCT}% daily cap\n"
            f"   Capital: ₹{MAX_CAPITAL:,.0f}\n"
            f"   Square-off: {SQUAREOFF_HOUR}:{SQUAREOFF_MIN:02d}\n"
            f"   Watchlist: {len(SCAN_WATCHLIST)} underlyings"
        )

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------
    def on_ltp_update(self, data):
        if data.get("type") == "market_data":
            sym = data.get("symbol", "")
            ltp = float(data["data"]["ltp"])
            self.ltp_map[sym] = ltp
            if sym == UNDERLYING:
                self.ltp = ltp

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

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------
    def get_historical_data(
        self, symbol: str = UNDERLYING, exchange: str = UNDERLYING_EXCHANGE
    ) -> pd.DataFrame:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=LOOKBACK_DAYS)
            source = "db" if exchange.endswith("_INDEX") else "api"
            history = self.client.history(
                symbol=symbol,
                exchange=exchange,
                interval=CANDLE_TIMEFRAME,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                source=source,
            )
            if isinstance(history, pd.DataFrame) and not history.empty:
                df = history.reset_index()
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            if isinstance(history, dict):
                logger.warning(
                    f"History returned no data for {symbol}: {history.get('message', history)}"
                )
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch history for {symbol}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Market Hours
    # ------------------------------------------------------------------
    def get_market_phase(self) -> str:
        """
        Determine current market phase:
        - 'pre_market': 7:00 - 9:15
        - 'first_candle': 9:15 - 9:30 (wait period)
        - 'active': 9:30 - 14:30 (trading allowed)
        - 'wind_down': 14:30 - 15:20 (exit only)
        - 'squareoff': 15:20+ (force close)
        - 'closed': outside hours
        """
        now = datetime.now()
        h, m = now.hour, now.minute
        t = h * 60 + m

        if t < 7 * 60:
            return "closed"
        if t < MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MIN:
            return "pre_market"
        if t < (MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MIN + FIRST_CANDLE_WAIT_MIN):
            return "first_candle"
        if t < NO_NEW_TRADES_HOUR * 60 + NO_NEW_TRADES_MIN:
            return "active"
        if t < SQUAREOFF_HOUR * 60 + SQUAREOFF_MIN:
            return "wind_down"
        if t < 16 * 60:
            return "squareoff"
        return "closed"

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------
    def send_notification(self, action: str, status: str, details: str = ""):
        msg = (
            f"[OPTIONS BOT {'📝' if TRADING_MODE == 'analyze' else '🔴'}]\n"
            f"Strategy: {self.strategy_name}\n"
            f"Mode: {STRATEGY_MODE.upper()}\n"
            f"Action: {action}\n"
            f"Status: {status}\n"
            f"{details}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            if WHATSAPP_PHONES:
                r = self.client.whatsapp(msg, to=WHATSAPP_PHONES[:5])
            else:
                r = self.client.whatsapp(msg)
            if isinstance(r, dict) and r.get("status") != "success":
                logger.warning(f"WhatsApp issue: {r.get('message', r)}")
            else:
                logger.info("WhatsApp notification sent.")
        except Exception as e:
            logger.warning(f"WhatsApp failed: {e}")

    # ------------------------------------------------------------------
    # Funds check
    # ------------------------------------------------------------------
    def check_funds(self, required: float = 0) -> bool:
        try:
            resp = self.client.funds()
            if isinstance(resp, dict) and resp.get("status") == "success":
                avail = float(resp.get("data", {}).get("availablecash", 0))
                logger.info(f"Available funds: ₹{avail:,.2f} | Required: ₹{required:,.2f}")
                return avail >= required
            logger.warning(f"Funds check returned: {resp}")
            return True  # Proceed anyway if can't check
        except Exception as e:
            logger.error(f"Funds check error: {e}")
            return True

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------
    def can_open_new_position(self) -> bool:
        """Check if risk limits allow a new position."""
        open_count = sum(1 for p in self.positions if p["status"] == "open")
        if open_count >= MAX_OPEN_POSITIONS:
            logger.info(f"Max open positions reached ({open_count}/{MAX_OPEN_POSITIONS})")
            return False

        max_daily_loss = MAX_CAPITAL * MAX_DAILY_LOSS_PCT / 100
        if self.daily_realized_pnl <= -max_daily_loss:
            logger.warning(
                f"⛔ Daily loss limit hit! P&L: ₹{self.daily_realized_pnl:,.2f} "
                f"(Limit: -₹{max_daily_loss:,.2f})"
            )
            return False

        return True

    def get_position_size(self, lot_size: int) -> int:
        """Calculate lots based on risk-per-trade."""
        max_risk = MAX_CAPITAL * RISK_PER_TRADE_PCT / 100
        # For now, trade 1 lot. Future: estimate premium and size accordingly.
        return lot_size

    # ------------------------------------------------------------------
    # Signal Engine — Confluence Scoring
    # ------------------------------------------------------------------
    def compute_signals(self, symbol: str, exchange: str) -> dict:
        """
        Compute confluence signals for the underlying.
        Returns: {"direction": "bull"/"bear"/"neutral", "score": float, "max_score": float, "details": {}}
        """
        df = self.get_historical_data(symbol, exchange)
        if df.empty or len(df) < max(EMA_TREND_LEN, 50):
            return {"direction": "neutral", "score": 0, "max_score": 0, "details": {}}

        # Update LTP if available
        ltp = self.ltp_map.get(symbol)
        if ltp is not None:
            df.loc[df.index[-1], "close"] = ltp

        close = df["close"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

        # Compute indicators
        ema_fast = compute_ema(close, EMA_FAST_LEN)
        ema_slow = compute_ema(close, EMA_SLOW_LEN)
        ema_trend = compute_ema(close, EMA_TREND_LEN)
        rsi_val = compute_rsi(close, RSI_LEN)
        macd_line, signal_line, macd_hist = compute_macd(close)
        di_plus, di_minus, adx = compute_dmi(df, 14)

        sym_has_volume = bool((volume > 0).any())
        vol_sma_20 = volume.rolling(20).mean()

        # VWAP
        try:
            vwap_val = compute_vwap(df)
            latest_vwap = vwap_val.iloc[-1]
            vwap_valid = sym_has_volume
        except Exception:
            latest_vwap = close.iloc[-1]
            vwap_valid = False

        # Latest values
        lc = close.iloc[-1]
        lf = ema_fast.iloc[-1]
        ls = ema_slow.iloc[-1]
        lt = ema_trend.iloc[-1]
        lr = rsi_val.iloc[-1]
        lmh = macd_hist.iloc[-1]
        lmh_prev = macd_hist.iloc[-2] if len(macd_hist) > 1 else lmh
        ladx = adx.iloc[-1]
        ldp = di_plus.iloc[-1]
        ldm = di_minus.iloc[-1]
        vol_above = sym_has_volume and (volume.iloc[-1] > vol_sma_20.iloc[-1] * 1.2)

        # EMA cross detection
        fp = ema_fast.iloc[-2] if len(ema_fast) > 1 else lf
        sp = ema_slow.iloc[-2] if len(ema_slow) > 1 else ls
        bull_cross = (fp <= sp) and (lf > ls)
        bear_cross = (fp >= sp) and (lf < ls)

        # Adaptive max score
        max_score = 9.0 + (1.0 if vwap_valid else 0.0)

        # ---- Bull scoring ----
        bull = 0.0
        bull += 1.0 if lf > ls else 0.0  # EMA fast > slow
        bull += 1.0 if lc > lt else 0.0  # Price > trend EMA
        bull += 1.0 if 50 < lr < 75 else 0.0  # RSI bullish zone
        bull += 1.0 if lmh > 0 else 0.0  # MACD histogram positive
        bull += 1.0 if lmh > lmh_prev else 0.0  # MACD hist rising
        bull += 1.0 if (vwap_valid and lc > latest_vwap) else 0.0  # Above VWAP
        bull += 1.0 if vol_above else 0.0  # Volume confirmation
        bull += 1.0 if (ladx > 20 and ldp > ldm) else 0.0  # ADX trending bullish
        bull += 1.0 if bull_cross else 0.0  # EMA crossover event
        bull += 0.5 if lc > lf else 0.0  # Price above fast EMA

        # ---- Bear scoring ----
        bear = 0.0
        bear += 1.0 if lf < ls else 0.0
        bear += 1.0 if lc < lt else 0.0
        bear += 1.0 if 25 < lr < 50 else 0.0
        bear += 1.0 if lmh < 0 else 0.0
        bear += 1.0 if lmh < lmh_prev else 0.0
        bear += 1.0 if (vwap_valid and lc < latest_vwap) else 0.0
        bear += 1.0 if vol_above else 0.0
        bear += 1.0 if (ladx > 20 and ldm > ldp) else 0.0
        bear += 1.0 if bear_cross else 0.0
        bear += 0.5 if lc < lf else 0.0

        # Determine direction
        effective_min = MIN_CONFLUENCE_SCORE * max_score / 10.0
        if bull >= effective_min and bull > bear:
            direction = "bull"
            score = bull
        elif bear >= effective_min and bear > bull:
            direction = "bear"
            score = bear
        else:
            direction = "neutral"
            score = max(bull, bear)

        details = {
            "bull_score": bull,
            "bear_score": bear,
            "rsi": round(lr, 1),
            "adx": round(ladx, 1),
            "ema_cross": "bull" if bull_cross else ("bear" if bear_cross else "none"),
            "vwap_bias": "above" if lc > latest_vwap else "below",
            "volume_ok": vol_above,
        }

        return {
            "direction": direction,
            "score": score,
            "max_score": max_score,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Strategy Executors
    # ------------------------------------------------------------------
    def execute_directional(self, pick: dict, direction: str) -> dict | None:
        """Buy ATM CE (bull) or ATM PE (bear) for directional scalp."""
        opt_type = "CE" if direction == "bull" else "PE"
        qty = self.get_position_size(pick["lot_size"])

        logger.info(
            f"📈 Directional {direction.upper()} on {pick['symbol']} — "
            f"Buying {DIRECTIONAL_OFFSET} {opt_type}, qty={qty}"
        )

        resp = self.client.optionsorder(
            strategy=self.strategy_name,
            underlying=pick["symbol"],
            exchange=pick["exchange"],
            offset=DIRECTIONAL_OFFSET,
            option_type=opt_type,
            action="BUY",
            quantity=qty,
            product=PRODUCT,
            expiry_date=pick.get("expiry_date"),
        )

        if isinstance(resp, dict) and resp.get("status") == "success":
            option_symbol = resp.get("symbol", "UNKNOWN")
            logger.info(f"✅ Order placed: {option_symbol} | OrderID: {resp.get('orderid')}")
            return self._create_position(
                pick,
                "directional",
                direction,
                legs=[
                    {
                        "symbol": option_symbol,
                        "offset": DIRECTIONAL_OFFSET,
                        "option_type": opt_type,
                        "action": "BUY",
                        "quantity": qty,
                        "entry_price": 0.0,  # Will be filled from quotes
                    }
                ],
            )
        else:
            logger.error(f"❌ Directional order failed: {resp}")
            self.send_notification(
                f"DIRECTIONAL {opt_type}",
                "FAILED",
                f"Underlying: {pick['symbol']}\nError: {resp}",
            )
            return None

    def execute_straddle(self, pick: dict) -> dict | None:
        """Buy ATM Call + ATM Put (Long Straddle) for volatility play."""
        qty = self.get_position_size(pick["lot_size"])

        logger.info(f"📊 Long Straddle on {pick['symbol']} — ATM CE + PE, qty={qty}")

        resp = self.client.optionsmultiorder(
            strategy=self.strategy_name,
            underlying=pick["symbol"],
            exchange=pick["exchange"],
            expiry_date=pick.get("expiry_date"),
            legs=[
                {"offset": "ATM", "option_type": "CE", "action": "BUY", "quantity": qty},
                {"offset": "ATM", "option_type": "PE", "action": "BUY", "quantity": qty},
            ],
        )

        if isinstance(resp, dict) and resp.get("status") == "success":
            results = resp.get("results", [])
            legs = []
            for r in results:
                legs.append(
                    {
                        "symbol": r.get("symbol", "UNKNOWN"),
                        "offset": r.get("offset", "ATM"),
                        "option_type": r.get("option_type", ""),
                        "action": "BUY",
                        "quantity": qty,
                        "entry_price": 0.0,
                    }
                )
            logger.info(f"✅ Straddle placed: {[leg['symbol'] for leg in legs]}")
            return self._create_position(pick, "straddle", "neutral", legs=legs)
        else:
            logger.error(f"❌ Straddle order failed: {resp}")
            self.send_notification("STRADDLE", "FAILED", f"Error: {resp}")
            return None

    def execute_spread(self, pick: dict, direction: str) -> dict | None:
        """Bull Call Spread or Bear Put Spread."""
        qty = self.get_position_size(pick["lot_size"])
        short_offset = f"OTM{SPREAD_WIDTH}"

        if direction == "bull":
            # Bull Call Spread: Buy ATM CE, Sell OTM CE
            legs_config = [
                {"offset": "ATM", "option_type": "CE", "action": "BUY", "quantity": qty},
                {"offset": short_offset, "option_type": "CE", "action": "SELL", "quantity": qty},
            ]
            label = "Bull Call Spread"
        else:
            # Bear Put Spread: Buy ATM PE, Sell OTM PE
            legs_config = [
                {"offset": "ATM", "option_type": "PE", "action": "BUY", "quantity": qty},
                {"offset": short_offset, "option_type": "PE", "action": "SELL", "quantity": qty},
            ]
            label = "Bear Put Spread"

        logger.info(f"📐 {label} on {pick['symbol']} — qty={qty}")

        resp = self.client.optionsmultiorder(
            strategy=self.strategy_name,
            underlying=pick["symbol"],
            exchange=pick["exchange"],
            expiry_date=pick.get("expiry_date"),
            legs=legs_config,
        )

        if isinstance(resp, dict) and resp.get("status") == "success":
            results = resp.get("results", [])
            legs = []
            for i, r in enumerate(results):
                legs.append(
                    {
                        "symbol": r.get("symbol", "UNKNOWN"),
                        "offset": legs_config[i]["offset"],
                        "option_type": legs_config[i]["option_type"],
                        "action": legs_config[i]["action"],
                        "quantity": qty,
                        "entry_price": 0.0,
                    }
                )
            logger.info(f"✅ {label} placed: {[leg['symbol'] for leg in legs]}")
            return self._create_position(pick, "spread", direction, legs=legs)
        else:
            logger.error(f"❌ {label} failed: {resp}")
            self.send_notification(label, "FAILED", f"Error: {resp}")
            return None

    def execute_iron_condor(self, pick: dict) -> dict | None:
        """Iron Condor: Sell OTM CE/PE near, Buy OTM CE/PE far."""
        qty = self.get_position_size(pick["lot_size"])
        short_off = f"OTM{CONDOR_SHORT_OFFSET}"
        long_off = f"OTM{CONDOR_LONG_OFFSET}"

        logger.info(
            f"🦅 Iron Condor on {pick['symbol']} — Sell {short_off}, Buy {long_off}, qty={qty}"
        )

        legs_config = [
            {"offset": long_off, "option_type": "CE", "action": "BUY", "quantity": qty},
            {"offset": short_off, "option_type": "CE", "action": "SELL", "quantity": qty},
            {"offset": short_off, "option_type": "PE", "action": "SELL", "quantity": qty},
            {"offset": long_off, "option_type": "PE", "action": "BUY", "quantity": qty},
        ]

        resp = self.client.optionsmultiorder(
            strategy=self.strategy_name,
            underlying=pick["symbol"],
            exchange=pick["exchange"],
            expiry_date=pick.get("expiry_date"),
            legs=legs_config,
        )

        if isinstance(resp, dict) and resp.get("status") == "success":
            results = resp.get("results", [])
            legs = []
            for i, r in enumerate(results):
                legs.append(
                    {
                        "symbol": r.get("symbol", "UNKNOWN"),
                        "offset": legs_config[i]["offset"],
                        "option_type": legs_config[i]["option_type"],
                        "action": legs_config[i]["action"],
                        "quantity": qty,
                        "entry_price": 0.0,
                    }
                )
            logger.info(f"✅ Iron Condor placed: {[leg['symbol'] for leg in legs]}")
            return self._create_position(pick, "iron_condor", "neutral", legs=legs)
        else:
            logger.error(f"❌ Iron Condor failed: {resp}")
            self.send_notification("IRON CONDOR", "FAILED", f"Error: {resp}")
            return None

    def _create_position(self, pick: dict, strategy: str, direction: str, legs: list) -> dict:
        """Create and register a new position."""
        pos = {
            "id": self._next_position_id,
            "underlying": pick["symbol"],
            "exchange": pick["exchange"],
            "opt_exchange": pick["opt_exchange"],
            "strategy": strategy,
            "direction": direction,
            "legs": legs,
            "lot_size": pick["lot_size"],
            "entry_time": datetime.now(),
            "total_premium_paid": 0.0,
            "sl_pct": PREMIUM_SL_PCT,
            "target_pct": PREMIUM_TARGET_PCT,
            "status": "open",
            "exit_reason": "",
            "pnl": 0.0,
            "trailing_active": False,
            "best_premium": 0.0,
        }
        self._next_position_id += 1
        self.positions.append(pos)
        self.daily_trade_count += 1

        self.send_notification(
            f"ENTRY: {strategy.upper()} {direction.upper()}",
            "SUCCESS",
            f"Underlying: {pick['symbol']}\n"
            f"Legs: {len(legs)}\n"
            f"Symbols: {', '.join(leg['symbol'] for leg in legs)}",
        )

        return pos

    # ------------------------------------------------------------------
    # Position Monitor & Exit Engine
    # ------------------------------------------------------------------
    def monitor_positions(self):
        """Check all open positions for SL, target, or time-based exits."""
        for pos in self.positions:
            if pos["status"] != "open":
                continue

            # Estimate current premium value for each leg
            total_current = 0.0
            total_entry = 0.0
            legs_priced = True

            for leg in pos["legs"]:
                try:
                    q = self.client.quotes(symbol=leg["symbol"], exchange=pos["opt_exchange"])
                    if isinstance(q, dict) and q.get("status") == "success":
                        current_ltp = float(q.get("data", {}).get("ltp", 0))
                        # Set entry price on first check
                        if leg["entry_price"] <= 0:
                            leg["entry_price"] = current_ltp
                            logger.info(
                                f"  Leg {leg['symbol']} entry price set: ₹{current_ltp:.2f}"
                            )

                        if leg["action"] == "BUY":
                            total_entry += leg["entry_price"] * leg["quantity"]
                            total_current += current_ltp * leg["quantity"]
                        else:  # SELL (credit leg)
                            total_entry -= leg["entry_price"] * leg["quantity"]
                            total_current -= current_ltp * leg["quantity"]
                    else:
                        legs_priced = False
                except Exception as e:
                    logger.warning(f"  Quote fetch failed for {leg['symbol']}: {e}")
                    legs_priced = False

            if not legs_priced or total_entry == 0:
                continue

            # Set initial premium
            if pos["total_premium_paid"] <= 0:
                pos["total_premium_paid"] = abs(total_entry)

            premium_paid = pos["total_premium_paid"]
            current_value = total_current
            unrealized_pnl = current_value - total_entry

            # For buy strategies: PnL = current_value - entry_cost
            # For sell strategies (credit): PnL = credit_received - current_cost
            pnl_pct = (unrealized_pnl / premium_paid * 100) if premium_paid > 0 else 0

            # Track best premium for trailing
            if current_value > pos["best_premium"]:
                pos["best_premium"] = current_value

            # ---- Exit Checks ----
            exit_reason = ""

            # 1. Stop-loss: premium dropped by SL%
            if pnl_pct <= -pos["sl_pct"]:
                exit_reason = f"SL (P&L: {pnl_pct:.1f}%)"

            # 2. Target: premium gained by target%
            elif pnl_pct >= pos["target_pct"]:
                exit_reason = f"TARGET (P&L: {pnl_pct:.1f}%)"

            # 3. Trailing stop after 50% of target reached
            elif pnl_pct >= pos["target_pct"] * 0.5:
                if not pos["trailing_active"]:
                    pos["trailing_active"] = True
                    logger.info(
                        f"  📈 Trailing activated for position #{pos['id']} at {pnl_pct:.1f}%"
                    )
                # Trail: if dropped 30% from best
                best_pnl_pct = (
                    (pos["best_premium"] - total_entry) / premium_paid * 100
                    if premium_paid > 0
                    else 0
                )
                trail_from_best = best_pnl_pct - pnl_pct
                if trail_from_best > 30:
                    exit_reason = f"TRAIL (dropped {trail_from_best:.1f}% from peak)"

            # 4. Time-based: position held too long (> 3 hours intraday)
            elapsed_min = (datetime.now() - pos["entry_time"]).total_seconds() / 60
            if elapsed_min > 180 and not exit_reason:
                # Only exit on time if not in profit
                if pnl_pct < 10:
                    exit_reason = f"TIME (held {elapsed_min:.0f} min, P&L: {pnl_pct:.1f}%)"

            if exit_reason:
                self._exit_position(pos, exit_reason, unrealized_pnl)

    def _exit_position(self, pos: dict, reason: str, pnl: float):
        """Exit all legs of a position."""
        logger.info(
            f"🔚 Exiting position #{pos['id']} ({pos['strategy']} on {pos['underlying']}) "
            f"— Reason: {reason}"
        )

        for leg in pos["legs"]:
            reverse_action = "SELL" if leg["action"] == "BUY" else "BUY"
            try:
                resp = self.client.optionsorder(
                    strategy=self.strategy_name,
                    underlying=pos["underlying"],
                    exchange=pos["exchange"],
                    offset=leg["offset"],
                    option_type=leg["option_type"],
                    action=reverse_action,
                    quantity=leg["quantity"],
                    product=PRODUCT,
                )
                if isinstance(resp, dict) and resp.get("status") == "success":
                    logger.info(f"  ✅ Closed leg {leg['symbol']} via {reverse_action}")
                else:
                    logger.error(f"  ❌ Failed to close {leg['symbol']}: {resp}")
            except Exception as e:
                logger.error(f"  ❌ Exit error for {leg['symbol']}: {e}")

        # Record
        pos["status"] = "closed"
        pos["exit_reason"] = reason
        pos["pnl"] = pnl
        self.daily_realized_pnl += pnl
        duration = (datetime.now() - pos["entry_time"]).total_seconds() / 60

        # Journal
        self.journal.log_trade(
            underlying=pos["underlying"],
            option_symbol=", ".join(leg["symbol"] for leg in pos["legs"]),
            strategy=pos["strategy"],
            action="EXIT",
            direction=pos["direction"],
            entry_price=pos["total_premium_paid"],
            exit_price=pos["total_premium_paid"] + pnl,
            quantity=pos["lot_size"],
            pnl=pnl,
            reason=reason,
            duration_min=duration,
        )

        self.send_notification(
            f"EXIT: {pos['strategy'].upper()}",
            reason,
            f"Underlying: {pos['underlying']}\n"
            f"P&L: ₹{pnl:,.2f}\n"
            f"Duration: {duration:.0f} min\n"
            f"Daily P&L: ₹{self.daily_realized_pnl:,.2f}",
        )

    def force_squareoff_all(self):
        """Force close all open positions (end-of-day)."""
        open_positions = [p for p in self.positions if p["status"] == "open"]
        if not open_positions:
            return
        logger.info(f"⏰ SQUAREOFF: Closing {len(open_positions)} open position(s)...")
        for pos in open_positions:
            self._exit_position(pos, "EOD_SQUAREOFF", pos.get("pnl", 0))

    # ------------------------------------------------------------------
    # Main Trading Logic
    # ------------------------------------------------------------------
    def run_pre_market_scan(self):
        """Run the F&O scanner during pre-market."""
        today = datetime.now().date()
        if self.last_scan_date == today:
            return

        logger.info("=" * 60)
        logger.info("🌅 PRE-MARKET SCAN STARTING")
        logger.info("=" * 60)

        self.todays_picks = self.scanner.scan_and_rank(SCAN_WATCHLIST, MAX_SCAN_PICKS)
        self.last_scan_date = today
        self.scan_done_today = True

        if self.todays_picks:
            pick_summary = "\n".join(
                f"  {p['symbol']}: Score={p['score']}, PCR={p['pcr']}, IV={p['iv_estimate']}%"
                for p in self.todays_picks
            )
            self.send_notification(
                "PRE-MARKET SCAN",
                "COMPLETE",
                f"Top {len(self.todays_picks)} picks:\n{pick_summary}",
            )

    def check_and_trade(self):
        """Main signal check and trade execution loop iteration."""
        phase = self.get_market_phase()

        # --- Pre-market: scan ---
        if phase == "pre_market":
            self.run_pre_market_scan()
            return

        # --- First candle wait ---
        if phase == "first_candle":
            if not self.scan_done_today:
                self.run_pre_market_scan()
            return

        # --- Squareoff ---
        if phase == "squareoff":
            self.force_squareoff_all()
            return

        # --- Closed ---
        if phase == "closed":
            return

        # --- Monitor existing positions (all active phases) ---
        self.monitor_positions()

        # --- Wind-down: only monitor, no new trades ---
        if phase == "wind_down":
            return

        # --- Active: evaluate entries ---
        if phase == "active":
            if not self.can_open_new_position():
                return

            if not self.todays_picks:
                if not self.scan_done_today:
                    self.run_pre_market_scan()
                if not self.todays_picks:
                    return

            # Evaluate each pick for entry signals
            for pick in self.todays_picks:
                # Skip if already have a position on this underlying
                active_syms = [p["underlying"] for p in self.positions if p["status"] == "open"]
                if pick["symbol"] in active_syms:
                    continue

                if not self.can_open_new_position():
                    break

                # Compute signals
                signals = self.compute_signals(pick["symbol"], pick["exchange"])

                if signals["direction"] == "neutral":
                    continue

                logger.info(
                    f"🎯 Signal on {pick['symbol']}: {signals['direction'].upper()} "
                    f"(Score: {signals['score']:.1f}/{signals['max_score']:.1f})\n"
                    f"   Details: {signals['details']}"
                )

                # Execute based on strategy mode
                try:
                    if STRATEGY_MODE == "directional":
                        self.execute_directional(pick, signals["direction"])

                    elif STRATEGY_MODE == "straddle":
                        # Straddle doesn't need direction, just volatility
                        self.execute_straddle(pick)

                    elif STRATEGY_MODE == "spread":
                        self.execute_spread(pick, signals["direction"])

                    elif STRATEGY_MODE == "iron_condor":
                        # Iron condor is direction-neutral; enter on any strong signal
                        self.execute_iron_condor(pick)

                    else:
                        logger.warning(f"Unknown strategy mode: {STRATEGY_MODE}")

                except Exception as e:
                    logger.error(f"Strategy execution error on {pick['symbol']}: {e}")

    def print_daily_summary(self):
        """Print and notify end-of-day summary."""
        closed_today = [
            p
            for p in self.positions
            if p["status"] == "closed" and p["entry_time"].date() == datetime.now().date()
        ]
        total_pnl = sum(p["pnl"] for p in closed_today)
        winners = sum(1 for p in closed_today if p["pnl"] > 0)
        losers = sum(1 for p in closed_today if p["pnl"] <= 0)
        win_rate = (winners / len(closed_today) * 100) if closed_today else 0

        summary = (
            f"\n{'=' * 60}\n"
            f"📊 DAILY SUMMARY — {datetime.now().strftime('%Y-%m-%d')}\n"
            f"{'=' * 60}\n"
            f"  Total Trades: {len(closed_today)}\n"
            f"  Winners: {winners} | Losers: {losers}\n"
            f"  Win Rate: {win_rate:.1f}%\n"
            f"  Total P&L: ₹{total_pnl:,.2f}\n"
            f"  Strategy: {STRATEGY_MODE.upper()}\n"
            f"{'=' * 60}"
        )
        logger.info(summary)

        self.send_notification(
            "DAILY SUMMARY",
            "END OF DAY",
            f"Trades: {len(closed_today)} | "
            f"W/L: {winners}/{losers} ({win_rate:.1f}%)\n"
            f"P&L: ₹{total_pnl:,.2f}",
        )

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    def run(self):
        """Main execution loop."""
        # Start WebSocket for primary underlying LTP
        ws_thread = threading.Thread(target=self.websocket_thread, daemon=True)
        ws_thread.start()
        time.sleep(2)

        logger.info("🏁 Options Bot execution loop started.")
        summary_printed = False

        try:
            while self.running:
                now = datetime.now()
                phase = self.get_market_phase()

                # Reset daily state at midnight / market open
                if now.hour < 7 and self.daily_trade_count > 0:
                    self.daily_realized_pnl = 0.0
                    self.daily_trade_count = 0
                    self.scan_done_today = False
                    self.todays_picks = []
                    summary_printed = False
                    logger.info("🔄 Daily state reset.")

                self.check_and_trade()

                # Print daily summary once after square-off
                if phase in ("squareoff", "closed") and not summary_printed:
                    if self.daily_trade_count > 0:
                        self.print_daily_summary()
                    summary_printed = True

                time.sleep(SIGNAL_CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("⛔ Bot manually stopped.")
        finally:
            self.force_squareoff_all()
            if self.daily_trade_count > 0 and not summary_printed:
                self.print_daily_summary()
            self.journal.close()
            self.stop_event.set()
            self.running = False
            logger.info("👋 Options Bot shut down complete.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    bot = IntradayOptionsBot()
    bot.run()
