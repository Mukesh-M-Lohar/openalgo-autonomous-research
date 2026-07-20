"""
Standalone Bot: Supertrend Options Touch Bot with OpenAlgo WebSocket Streaming
"""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from openalgo import api

load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("OPENALGO_API_KEY", "openalgo-apikey")
API_HOST = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")

# Strategy Parameters (Best from Backtest)
UNDERLYING_SYMBOL = "BANKNIFTY"
EXCHANGE = "NSE_INDEX"
TIMEFRAME = "5m"
ST_PERIOD = 10
ST_MULT = 3.0
STRIKE_OFFSET = 500  # 500 points ITM
QUANTITY = int(os.getenv("QUANTITY", "30"))
PRODUCT = "MIS"
AVOID_0DTE = True  # If True, avoids 0DTE contracts by rolling to the next expiry

# Time of Day Filter (IST) - e.g., 11 AM to 3 PM IST
ALLOWED_START_HOUR = 11
ALLOWED_END_HOUR = 14

# Exits
TRAIL_SL_PCT = 10.0  # Trailing stop percentage (None to disable)
TAKE_PROFIT_PCT = None  # None for letting it ride

# Cache Lookback Window
LOOKBACK_BARS = 100  # Number of historical bars to keep in memory

# WebSocket Data Feed Toggle
USE_WEBSOCKET = os.getenv("USE_WEBSOCKET", "true").lower() == "true"


class SupertrendOptionsBot:
    def __init__(self):
        print(f"Initializing Supertrend Options Bot for {UNDERLYING_SYMBOL}...")
        self.client = api(api_key=API_KEY, host=API_HOST)
        self.history_cache = pd.DataFrame()
        self.active_position = None
        self.max_favorable_price = 0.0
        self.current_sl_price = 0.0
        self.entry_price = 0.0
        self.use_websocket = USE_WEBSOCKET

        self._prime_cache()
        self._init_websocket()

    def _init_websocket(self):
        """Initialize WebSocket connection and subscribe to underlying index."""
        if not self.use_websocket:
            print("WebSocket streaming is disabled in configuration (using REST polling).")
            return

        try:
            print("Connecting to OpenAlgo WebSocket feed...")
            self.client.connect()
            self.client.subscribe_ltp([{"exchange": EXCHANGE, "symbol": UNDERLYING_SYMBOL}])
            print(f"✅ WebSocket connected and subscribed to {EXCHANGE}:{UNDERLYING_SYMBOL}")
        except Exception as e:
            print(f"⚠️ WebSocket initialization failed, falling back to REST: {e}")
            self.use_websocket = False

    def get_live_ltp(self, exchange, symbol):
        """Fetch live LTP from WebSocket feed if connected; fall back to REST quotes API."""
        if self.use_websocket:
            try:
                res = self.client.get_ltp(exchange, symbol)
                ltp_data = res.get("ltp", {}).get(exchange, {}).get(symbol, {})
                price = float(ltp_data.get("ltp", 0))
                if price > 0:
                    return price
            except Exception as e:
                print(f"Warning: Failed to fetch WS LTP for {exchange}:{symbol}: {e}")

        # Fallback to REST Quote API
        try:
            q_resp = self.client.quotes(symbol=symbol, exchange=exchange)
            if q_resp and q_resp.get("status") == "success":
                data = q_resp.get("data", {})
                return float(data.get("ltp", 0))
        except Exception as e:
            print(f"Warning: REST quote fallback failed for {exchange}:{symbol}: {e}")

        return 0.0

    def _prime_cache(self):
        """Fetch initial history to prime the Supertrend calculation."""
        print(f"Fetching last {LOOKBACK_BARS} bars of {UNDERLYING_SYMBOL} to prime cache...")

        # Calculate start date based on timeframe (generous buffer)
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        hist = self.client.history(
            symbol=UNDERLYING_SYMBOL,
            exchange=EXCHANGE,
            interval=TIMEFRAME,
            start_date=start_date,
            end_date=end_date,
        )

        if isinstance(hist, dict) and hist.get("status") == "error":
            raise Exception(f"Failed to prime cache: {hist}")

        df = hist if isinstance(hist, pd.DataFrame) else pd.DataFrame(hist)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        df = df.sort_index()
        if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex) and df.index.min().hour < 7:
            df.index = df.index + pd.Timedelta(hours=5, minutes=30)

        # Keep only the latest LOOKBACK_BARS
        self.history_cache = df.tail(LOOKBACK_BARS)
        print(f"Cache primed with {len(self.history_cache)} bars.")

    def update_cache_and_calc(self):
        """Fetch the single latest bar, update rolling cache, and calc Supertrend."""
        start_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        hist = self.client.history(
            symbol=UNDERLYING_SYMBOL,
            exchange=EXCHANGE,
            interval=TIMEFRAME,
            start_date=start_date,
            end_date=end_date,
        )

        if isinstance(hist, dict) and hist.get("status") == "error":
            print(f"Warning: Failed to fetch latest data: {hist}")
            return None, None

        df = hist if isinstance(hist, pd.DataFrame) else pd.DataFrame(hist)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        df = df.sort_index()
        if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex) and df.index.min().hour < 7:
            df.index = df.index + pd.Timedelta(hours=5, minutes=30)

        # Update rolling cache
        self.history_cache = pd.concat([self.history_cache, df]).drop_duplicates()
        self.history_cache = self.history_cache.tail(LOOKBACK_BARS)

        # Calculate Supertrend
        df_calc = self.history_cache.copy()
        st = ta.supertrend(
            df_calc["high"], df_calc["low"], df_calc["close"], length=ST_PERIOD, multiplier=ST_MULT
        )
        if st is None or st.empty:
            return None, None

        st_col = [
            c
            for c in st.columns
            if c.startswith("SUPERT_")
            and not c.startswith("SUPERTd_")
            and not c.startswith("SUPERTl_")
            and not c.startswith("SUPERTs_")
        ][0]
        dir_col = [c for c in st.columns if c.startswith("SUPERTd_")][0]

        return df_calc, st[[st_col, dir_col]]

    def get_option_contract(self, underlying_price, direction):
        """Find the correct CE/PE contract at the configured strike offset."""
        print(
            f"Looking up option contract for {UNDERLYING_SYMBOL} at {underlying_price} ({direction})..."
        )
        instruments = self.client.instruments(exchange="NFO")
        if isinstance(instruments, dict) and instruments.get("status") == "error":
            print("Failed to fetch instruments.")
            return None

        df_inst = (
            instruments if isinstance(instruments, pd.DataFrame) else pd.DataFrame(instruments)
        )
        df_opts = df_inst[
            (df_inst["name"] == UNDERLYING_SYMBOL) & (df_inst["instrumenttype"].isin(["CE", "PE"]))
        ].copy()
        df_opts["expiry_date"] = pd.to_datetime(df_opts["expiry"])

        # Get nearest expiry
        if AVOID_0DTE:
            future_opts = df_opts[df_opts["expiry_date"] > pd.Timestamp.today().normalize()]
        else:
            future_opts = df_opts[df_opts["expiry_date"] >= pd.Timestamp.today().normalize()]

        if future_opts.empty:
            return None

        nearest_expiry = future_opts["expiry_date"].min()
        valid_opts = future_opts[future_opts["expiry_date"] == nearest_expiry]

        # Calculate Target Strike
        base_strike = round(underlying_price / 100) * 100
        if direction == "UP":
            target_strike = base_strike - STRIKE_OFFSET  # CE ITM
            opt_type = "CE"
        else:
            target_strike = base_strike + STRIKE_OFFSET  # PE ITM
            opt_type = "PE"

        # Liquidity Window: Target Strike and +/- 100 points
        step = 100
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
                    vol = float(data.get("volume", 0))
                    oi = float(data.get("oi", 0))
                    score = vol + oi

                    print(f"  -> Candidate {sym}: Volume={vol}, OI={oi}")

                    if score > best_score:
                        best_score = score
                        best_symbol = sym
            except Exception as e:
                print(f"  -> Failed to quote {sym}: {e}")

        if best_symbol:
            print(f"✅ Selected Most Liquid Contract: {best_symbol} (Score: {best_score})")
            return best_symbol

        return None

    def execute_trade(self, symbol, direction="UP"):
        """Place Market Order and initialize position tracking."""
        print(f"\n🚀 EXECUTING ENTRY FOR {symbol} ({direction})")
        try:
            resp = self.client.placeorder(
                symbol=symbol,
                action="BUY",
                exchange="NFO",
                price_type="MARKET",
                product=PRODUCT,
                quantity=QUANTITY,
                strategy="SupertrendOptionsBot",
            )
            print(f"Order Response: {resp}")

            # Subscribe to option WebSocket stream if active
            if self.use_websocket:
                try:
                    self.client.subscribe_ltp([{"exchange": "NFO", "symbol": symbol}])
                    print(f"Subscribed WebSocket LTP feed for active option {symbol}")
                except Exception as e:
                    print(f"Failed to subscribe WS for {symbol}: {e}")

            # Fetch execution price
            self.entry_price = self.get_live_ltp("NFO", symbol)
            self.max_favorable_price = self.entry_price

            if TRAIL_SL_PCT and self.entry_price > 0:
                self.current_sl_price = self.entry_price * (1 - (TRAIL_SL_PCT / 100.0))

            self.active_position = symbol
            self.active_direction = direction
            print(
                f"✅ Entered {symbol} ({direction}) at {self.entry_price}. SL set to {self.current_sl_price:.2f}"
            )
        except Exception as e:
            print(f"Execution failed: {e}")

    def exit_trade(self, reason):
        """Exit the current active position."""
        print(f"\n🛑 EXECUTING EXIT: {reason}")
        symbol_to_exit = self.active_position
        try:
            resp = self.client.placeorder(
                symbol=symbol_to_exit,
                action="SELL",
                exchange="NFO",
                price_type="MARKET",
                product=PRODUCT,
                quantity=QUANTITY,
                strategy="SupertrendOptionsBot",
            )
            print(f"Exit Order Response: {resp}")

            # Unsubscribe option WebSocket feed
            if self.use_websocket and symbol_to_exit:
                try:
                    self.client.unsubscribe_ltp([{"exchange": "NFO", "symbol": symbol_to_exit}])
                    print(f"Unsubscribed WebSocket LTP feed for {symbol_to_exit}")
                except Exception as e:
                    print(f"Failed to unsubscribe WS for {symbol_to_exit}: {e}")

            self.active_position = None
            self.active_direction = None
        except Exception as e:
            print(f"Exit failed: {e}")

    def manage_position(self):
        """Monitor live price via WebSocket/REST and manage Trailing Stop / Take Profit / Index ST Flip."""
        if not self.active_position:
            return

        try:
            current_price = self.get_live_ltp("NFO", self.active_position)
            if current_price <= 0:
                return

            # 1. Update Trailing Stop
            if current_price > self.max_favorable_price:
                self.max_favorable_price = current_price
                if TRAIL_SL_PCT:
                    new_sl = self.max_favorable_price * (1 - (TRAIL_SL_PCT / 100.0))
                    if new_sl > self.current_sl_price:
                        self.current_sl_price = new_sl
                        print(f"Trailing SL moved up to: {self.current_sl_price:.2f}")

            # 2. Check Stop Loss Hit
            if TRAIL_SL_PCT and current_price <= self.current_sl_price:
                print(f"Price {current_price} crossed SL {self.current_sl_price:.2f}")
                self.exit_trade("TRAILING_STOP_HIT")
                return

            # 3. Check Take Profit Hit
            if TAKE_PROFIT_PCT:
                tp_target = self.entry_price * (1 + (TAKE_PROFIT_PCT / 100.0))
                if current_price >= tp_target:
                    print(f"Price {current_price} hit TP {tp_target:.2f}")
                    self.exit_trade("TAKE_PROFIT_HIT")
                    return

            # 4. Check Underlying Index Supertrend Flip
            if hasattr(self, "active_direction") and self.active_direction:
                df, st = self.update_cache_and_calc()
                if (
                    df is not None
                    and not df.empty
                    and st is not None
                    and not st.empty
                    and len(st) >= 2
                ):
                    prev_st = st.iloc[-2]
                    st_dir = prev_st.iloc[1]
                    if self.active_direction == "UP" and st_dir == -1:
                        print(
                            "Underlying Index Supertrend flipped to Bearish (-1) while holding CE!"
                        )
                        self.exit_trade("INDEX_ST_FLIP")
                        return
                    elif self.active_direction == "DOWN" and st_dir == 1:
                        print(
                            "Underlying Index Supertrend flipped to Bullish (1) while holding PE!"
                        )
                        self.exit_trade("INDEX_ST_FLIP")
                        return

        except Exception as e:
            print(f"Position management error: {e}")

    def run(self):
        print("\nBot is now actively monitoring the market...")
        while True:
            try:
                # 1. Manage Active Position (Fast execution loop)
                if self.active_position:
                    self.manage_position()
                    time.sleep(0.5)  # Fast sub-second monitoring for exits
                    continue

                # 2. Monitor for Entry

                # Time of Day Filter (IST Timezone)
                ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
                current_ist_hour = ist_now.hour
                if ALLOWED_START_HOUR is not None and ALLOWED_END_HOUR is not None:
                    if not (ALLOWED_START_HOUR <= current_ist_hour <= ALLOWED_END_HOUR):
                        time.sleep(60)  # Sleep outside trading hours
                        continue

                df, st = self.update_cache_and_calc()
                if df is not None and not df.empty and len(df) >= 2:
                    # Eliminate Lookahead Bias: Use Supertrend level from previous COMPLETED candle (iloc[-2]).
                    prev_st = st.iloc[-2]
                    st_val = prev_st.iloc[0]
                    st_dir = prev_st.iloc[1]

                    # Fetch live LTP of underlying symbol from WebSocket feed / REST fallback
                    live_ltp = self.get_live_ltp(EXCHANGE, UNDERLYING_SYMBOL)
                    if live_ltp == 0.0:
                        live_ltp = float(df.iloc[-1]["close"])

                    latest_bar = df.iloc[-1]

                    # Touch logic against static ST line (prev bar):
                    touch_detected = False
                    if live_ltp > 0:
                        if st_dir == 1 and (live_ltp <= st_val or latest_bar["low"] <= st_val):
                            touch_detected = True
                        elif st_dir == -1 and (live_ltp >= st_val or latest_bar["high"] >= st_val):
                            touch_detected = True

                    if touch_detected:
                        direction = "UP" if st_dir == 1 else "DOWN"
                        print(
                            f"[{ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}] Touch Detected! Direction: {direction} | Live LTP: {live_ltp} | ST (Prev Bar): {st_val:.2f}"
                        )

                        contract = self.get_option_contract(live_ltp, direction)
                        if contract:
                            self.execute_trade(contract)

                time.sleep(2)

            except KeyboardInterrupt:
                print("\nBot stopped by user.")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(5)


if __name__ == "__main__":
    bot = SupertrendOptionsBot()
    bot.run()
