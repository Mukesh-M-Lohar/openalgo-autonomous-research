# Python Strategy Guide

Get your first strategy running in 5 minutes with the OpenAlgo Python SDK.

## Quick Start

### 1. Install OpenAlgo SDK
```bash
pip install openalgo
```

### 2. Get your API Key
Go to the **API Key** page and copy your OpenAlgo API key.

### 3. Write your strategy
Create a Python file (`.py`) with your trading logic. Read configuration from environment variables so it works both standalone and under the `/python` runner without edits. See the sample strategy below.

### 4. Upload and configure
On the **Python Strategies** page, click **Add Strategy**. Pick a name, select the exchange (NSE / MCX / CRYPTO / etc.), and add any custom parameters as environment variables.

### 5. Start or schedule
Click **Start** to run immediately, or configure a schedule (e.g. `09:15–15:30 Mon–Fri` for NSE). The host auto-starts and auto-stops your strategy at the scheduled times, respecting the exchange's holiday calendar.

---

## How It Works

### Process Isolation
Each strategy runs as a separate `subprocess.Popen` process with its own PID, memory, and file descriptors. A crash in one strategy cannot affect another or the host.

### Environment Injection
The host injects `OPENALGO_API_KEY`, `STRATEGY_ID`, `STRATEGY_NAME`, and `OPENALGO_STRATEGY_EXCHANGE` into each strategy's environment. Your `.env` variables (like `HOST_SERVER`, `WEBSOCKET_URL`) are also inherited. Custom parameters from the upload form become additional env vars.

### Exchange-Aware Calendar
Each strategy is tagged with an exchange. The host uses that exchange's holiday calendar to decide whether to start/stop the strategy on any given day. An MCX strategy keeps running on an NSE holiday during the MCX evening session, and a CRYPTO strategy ignores all holidays.

### Logging
All `print()` output is captured in timestamped log files under `log/strategies/`. View them from the dashboard or via the **Logs** button on each strategy card.

---

## Environment Variables

These variables are available inside your strategy script.

### Injected by the Platform
These are set directly on each strategy subprocess by the `/python` runner:

| Variable | Description |
|---|---|
| `OPENALGO_API_KEY` | Decrypted API key for this user |
| `STRATEGY_ID` | Unique identifier for this strategy |
| `STRATEGY_NAME` | Name of the strategy (as entered at upload) |
| `OPENALGO_STRATEGY_EXCHANGE` | Exchange picked at upload/edit (NSE / BSE / NFO / BFO / MCX / BCD / CDS / CRYPTO). Read this so your trading calls match the calendar the host gates against |
| `OPENALGO_HOST` | Convenience fallback (http://127.0.0.1:5000). Prefer `HOST_SERVER` instead |

### Inherited from `.env`
Strategies inherit every variable from OpenAlgo's `.env` via `os.environ.copy()`. The key ones for connecting back to OpenAlgo:

| Variable | Description |
|---|---|
| `HOST_SERVER` | REST host, e.g. `http://127.0.0.1:5000` — canonical name in `.env`, prefer this in scripts |
| `WEBSOCKET_URL` | Full WebSocket URL, e.g. `ws://127.0.0.1:8765` |
| `WEBSOCKET_HOST` | WebSocket host component, e.g. `127.0.0.1` |
| `WEBSOCKET_PORT` | WebSocket port, e.g. `8765` |
| `FLASK_HOST_IP` / `FLASK_PORT` | Flask binding address (available if you need raw components) |

### Recommended Pattern in Scripts

```python
import os

# API Configuration — auto-injected by the /python runner
API_KEY = os.getenv("OPENALGO_API_KEY", "openalgo-apikey")
API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

# Exchange — reads OPENALGO_STRATEGY_EXCHANGE so your script
# trades on the same exchange the host gates its calendar against.
EXCHANGE = os.getenv(
    "OPENALGO_STRATEGY_EXCHANGE",
    os.getenv("EXCHANGE", "NSE"),
)

# Strategy identity (for tagging orders/logs)
STRATEGY_NAME = os.getenv("STRATEGY_NAME", "MyStrategy")
STRATEGY_ID = os.getenv("STRATEGY_ID", "")

# All custom parameters from the upload form are also available:
MY_PARAM = os.getenv("MY_PARAM", "default_value")
```

> [!IMPORTANT]
> **Reading `OPENALGO_STRATEGY_EXCHANGE` is strongly recommended**
> If your script hardcodes `exchange = "NSE"`, the host will still gate it correctly per its config (e.g. the host runs your script during the MCX evening session because `exchange=MCX`), but your `client.placeorder(exchange="NSE", ...)` calls will still send NSE orders — and the broker will reject them. Wiring the env var keeps host calendar and script orders aligned.

---

## Exchange-Aware Scheduling

Each strategy's exchange drives which holiday calendar the host uses.

When you upload or edit a strategy, you pick an exchange. The host uses that exchange's calendar to decide whether to start/stop the strategy on any given day. This means:
- An MCX strategy keeps running on NSE/BSE holidays if MCX has a session.
- A CRYPTO strategy ignores all holidays and weekends (24/7).
- `SPECIAL_SESSION` rows (Muhurat, DR-drill) override weekend rejects per-exchange.

### Supported Exchanges

- **NSE** — Equity (09:15–15:30)
- **BSE** — Equity (09:15–15:30)
- **NFO** — NSE F&O (09:15–15:30)
- **BFO** — BSE F&O (09:15–15:30)
- **CDS** — NSE Currency (09:00–17:00)
- **BCD** — BSE Currency (09:00–17:00)
- **MCX** — Commodity (09:00–23:55)
- **CRYPTO** — 24/7 (no holidays)

*Timings shown are defaults. Per-date overrides (partial holidays, special sessions) come from the market calendar DB.*

### Schedule Intersection Rule

The effective trading window is the intersection of your start..stop time and the exchange's session for that specific date.

**Example: MCX strategy scheduled 09:15–23:55**

On 14-Apr-2026 (Ambedkar Jayanti), MCX has a partial holiday with an evening session 17:00–23:55. The effective window becomes 17:00–23:55 (the intersection). You don't need to change the schedule for partial holidays.

### Worked Examples

| Scenario | Exchange | Strategy Behavior |
|---|---|---|
| **14-Apr-2026**<br>Ambedkar Jayanti | NSE / BSE / NFO | Closed all day. Strategies paused at 00:01 IST |
| | MCX | Open 17:00–23:55. MCX strategies auto-start at 17:00 |
| | CRYPTO | Unaffected (24/7) |
| **8-Nov-2026**<br>Sunday Diwali Muhurat | NSE / BSE / NFO | SPECIAL_SESSION 18:00–19:15. Runs only inside that window, despite being Sunday |
| | MCX | SPECIAL_SESSION 18:00–00:15 next day |
| | CRYPTO | Unaffected (24/7) |

### How the Host Gates Strategies

1. **Cron job** — Fires `start_<sid>` at your `start_time` on each day in `schedule_days`.
2. **Daily check (00:01 IST)** — For each scheduled strategy, looks up `get_market_status(config["exchange"])`. If the exchange has no session today, the strategy is stopped and marked `paused_reason=holiday|weekend`.
3. **Per-minute enforcer** — Same per-strategy check. When the exchange reopens (or a special session starts), previously-paused strategies are auto-resumed (unless `manually_stopped`).

### Smart Defaults When Uploading
Picking an exchange pre-fills sensible defaults: CRYPTO auto-selects all 7 days and 00:00–23:59, MCX defaults to 09:00–23:55 weekdays, and equity exchanges default to 09:15–15:30 Mon–Fri.

---

## Sample Strategy: EMA Crossover

WebSocket-driven EMA crossover with real-time SL/target monitoring. All config is overridable via env vars — works standalone and under the `/python` runner without edits. Full version at `examples/python/emacrossover_strategy_python.py`.

```python
"""
===============================================================================
                EMA CROSSOVER WITH FIXED DATETIME HANDLING
                            OpenAlgo Trading Bot
===============================================================================

Run standalone:
    export OPENALGO_API_KEY="your-api-key"
    python emacrossover_strategy_python.py

Run via OpenAlgo's /python strategy runner:
    OPENALGO_API_KEY            : injected per-strategy (PR #1247).
    OPENALGO_STRATEGY_EXCHANGE  : set from the strategy's `exchange` config
                                  (NSE / BSE / NFO / BFO / MCX / BCD / CDS / CRYPTO).
                                  Drives both this script's trading exchange and
                                  the host's calendar/holiday gating, so the two
                                  always agree.
    STRATEGY_ID / STRATEGY_NAME : injected for log/order tagging.
    HOST_SERVER / WEBSOCKET_URL : inherited from OpenAlgo's .env.
    No code changes required.
"""

import os
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
from openalgo import api

# ===============================================================================
# TRADING CONFIGURATION
# ===============================================================================

# API Configuration — read from environment with sensible fallbacks.
# When launched via OpenAlgo's /python runner, these come from the platform:
#   OPENALGO_API_KEY : injected per-strategy (decrypted from DB)
#   HOST_SERVER      : inherited from OpenAlgo's .env
#   WEBSOCKET_URL    : inherited from OpenAlgo's .env
API_KEY = os.getenv("OPENALGO_API_KEY", "openalgo-apikey")
API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

# Trade Settings
# EXCHANGE prefers OPENALGO_STRATEGY_EXCHANGE (set by /python runner from the
# strategy's config) so the script trades on whichever exchange the host is
# gating its calendar against. Falls back to EXCHANGE env var, then NSE.
SYMBOL = os.getenv("SYMBOL", "NHPC")              # Stock to trade
EXCHANGE = os.getenv(
    "OPENALGO_STRATEGY_EXCHANGE",
    os.getenv("EXCHANGE", "NSE"),
)                                                 # NSE, BSE, NFO, MCX, etc.
QUANTITY = int(os.getenv("QUANTITY", "1"))        # Number of shares
PRODUCT = os.getenv("PRODUCT", "MIS")             # MIS (Intraday) or CNC (Delivery)

# Strategy Parameters
FAST_EMA_PERIOD = int(os.getenv("FAST_EMA_PERIOD", "2"))
SLOW_EMA_PERIOD = int(os.getenv("SLOW_EMA_PERIOD", "4"))
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "5m")

# Historical Data Lookback (1-30 days)
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "3"))

# Risk Management (Rupees)
STOPLOSS = float(os.getenv("STOPLOSS", "0.1"))
TARGET = float(os.getenv("TARGET", "0.2"))

# Direction Control: LONG, SHORT, BOTH
TRADE_DIRECTION = os.getenv("TRADE_DIRECTION", "BOTH")

# Signal Check Interval (seconds)
SIGNAL_CHECK_INTERVAL = int(os.getenv("SIGNAL_CHECK_INTERVAL", "5"))

# ===============================================================================
# TRADING BOT
# ===============================================================================

class ConfigurableEMABot:
    def __init__(self):
        # Initialize API client
        self.client = api(
            api_key=API_KEY,
            host=API_HOST,
            ws_url=WS_URL,
        )

        # Position tracking
        self.position = None
        self.entry_price = 0
        self.stoploss_price = 0
        self.target_price = 0

        # Real-time price tracking
        self.ltp = None
        self.exit_in_progress = False

        # Thread control
        self.running = True
        self.stop_event = threading.Event()

        # Instrument for WebSocket
        self.instrument = [{"exchange": EXCHANGE, "symbol": SYMBOL}]

        # Strategy name from the platform
        self.strategy_name = os.getenv("STRATEGY_NAME", f"EMA_{TRADE_DIRECTION}")

        print(f"[BOT] Symbol: {SYMBOL} on {EXCHANGE}")
        print(f"[BOT] Strategy: {FAST_EMA_PERIOD} EMA x {SLOW_EMA_PERIOD} EMA")

    # =========================================================================
    # WEBSOCKET HANDLER — real-time SL/Target monitoring
    # =========================================================================

    def on_ltp_update(self, data):
        if data.get("type") == "market_data" and data.get("symbol") == SYMBOL:
            self.ltp = float(data["data"]["ltp"])

            if self.position and not self.exit_in_progress:
                # Check stoploss / target
                exit_reason = None
                if self.position == "BUY":
                    if self.ltp <= self.stoploss_price:
                        exit_reason = "STOPLOSS HIT"
                    elif self.ltp >= self.target_price:
                        exit_reason = "TARGET HIT"
                elif self.position == "SELL":
                    if self.ltp >= self.stoploss_price:
                        exit_reason = "STOPLOSS HIT"
                    elif self.ltp <= self.target_price:
                        exit_reason = "TARGET HIT"

                if exit_reason and not self.exit_in_progress:
                    self.exit_in_progress = True
                    threading.Thread(
                        target=self.place_exit_order, args=(exit_reason,)
                    ).start()

    def websocket_thread(self):
        try:
            self.client.connect()
            self.client.subscribe_ltp(
                self.instrument, on_data_received=self.on_ltp_update
            )
            while not self.stop_event.is_set():
                time.sleep(1)
        except Exception as e:
            print(f"[ERROR] WebSocket error: {e}")
        finally:
            try:
                self.client.unsubscribe_ltp(self.instrument)
                self.client.disconnect()
            except Exception:
                pass

    # =========================================================================
    # TRADING FUNCTIONS
    # =========================================================================

    def get_historical_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS)
        return self.client.history(
            symbol=SYMBOL,
            exchange=EXCHANGE,
            interval=CANDLE_TIMEFRAME,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

    def check_for_signal(self, data):
        if data is None or len(data) < SLOW_EMA_PERIOD + 2:
            return None

        data["fast_ema"] = data["close"].ewm(
            span=FAST_EMA_PERIOD, adjust=False
        ).mean()
        data["slow_ema"] = data["close"].ewm(
            span=SLOW_EMA_PERIOD, adjust=False
        ).mean()

        prev = data.iloc[-3]
        last = data.iloc[-2]

        # BUY: fast crosses above slow
        if prev["fast_ema"] <= prev["slow_ema"] and \
           last["fast_ema"] > last["slow_ema"]:
            if TRADE_DIRECTION in ["LONG", "BOTH"]:
                return "BUY"
        # SELL: fast crosses below slow
        if prev["fast_ema"] >= prev["slow_ema"] and \
           last["fast_ema"] < last["slow_ema"]:
            if TRADE_DIRECTION in ["SELL", "BOTH"]:
                return "SELL"
        return None

    def place_entry_order(self, signal):
        response = self.client.placeorder(
            strategy=self.strategy_name,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=signal,
            quantity=QUANTITY,
            price_type="MARKET",
            product=PRODUCT,
        )
        if response.get("status") == "success":
            # ... track position, set SL/target levels
            self.position = signal

    def place_exit_order(self, reason="Manual"):
        exit_action = "SELL" if self.position == "BUY" else "BUY"
        self.client.placeorder(
            strategy=self.strategy_name,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=exit_action,
            quantity=QUANTITY,
            price_type="MARKET",
            product=PRODUCT,
        )
        self.position = None
        self.exit_in_progress = False

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        # Start WebSocket thread for real-time SL/Target
        ws_thread = threading.Thread(target=self.websocket_thread, daemon=True)
        ws_thread.start()
        time.sleep(2)

        try:
            while self.running:
                if not self.position and not self.exit_in_progress:
                    data = self.get_historical_data()
                    signal = self.check_for_signal(data)
                    if signal:
                        self.place_entry_order(signal)
                time.sleep(SIGNAL_CHECK_INTERVAL)
        except KeyboardInterrupt:
            self.running = False
            self.stop_event.set()
            if self.position:
                self.place_exit_order("Bot Shutdown")

if __name__ == "__main__":
    bot = ConfigurableEMABot()
    bot.run()
```

---

## OpenAlgo SDK Quick Reference

### Initialize Client
```python
client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)
```

### Place Order
```python
client.placeorder(strategy, symbol, exchange, action, quantity, price_type, product)
```

### Historical Data
```python
client.history(symbol, exchange, interval, start_date, end_date)
```

### WebSocket (LTP)
```python
client.connect()
client.subscribe_ltp(instruments, on_data_received=callback)
```

### Get Quotes
```python
client.quotes(symbol="RELIANCE", exchange="NSE")
```

### Order Status
```python
client.orderstatus(order_id=order_id, strategy=strategy_name)
```

### Positions / Holdings
```python
client.positionbook()
client.holdings()
```

For complete SDK documentation, visit [docs.openalgo.in](https://docs.openalgo.in).

---

## Directory Structure
```
strategies/
  scripts/          # Uploaded strategy files
  examples/         # Example strategies
  configs.json      # Strategy configurations (atomic write)
  README.md         # Detailed documentation
  RESOURCE_LIMITS.md

log/
  strategies/       # Strategy log files (per-strategy rotation)

examples/
  python/           # Standalone example scripts
    emacrossover_strategy_python.py  # Full EMA crossover sample
```
