# OpenAlgo Python SDK Complete Reference Guide

This guide provides a comprehensive, highly precise developer reference for the OpenAlgo Python SDK. It outlines the exact parameters, `**kwargs` usage, and return dictionary schemas for every method in the SDK.

---

## 1. Client Instantiation & Authentication

The core client class is `api`, imported from the `openalgo` package. It manages HTTP connection pooling and handles WebSocket subscriptions.

### Instantiation Reference

```python
from openalgo import api

client = api(
    api_key="your-key",             # REST/WS authentication key.
    host="http://127.0.0.1:5000",   # REST Server base URL.
    version="v1",                   # API version segment.
    timeout=120.0,                  # Request timeout in seconds.
    ws_port=8765,                   # WebSocket server port.
    ws_url=None,                    # Custom WebSocket URL (e.g. ws://addr:port).
    verbose=False,                  # Logging level (0=errors, 1=info, 2=full debug).
    auto_reconnect=True             # Transparently reconnect WebSocket drops.
)
```

---

## 2. Account Management

Inherited from the `AccountAPI` class, these methods manage balances, order queues, open positions, and margin checks.

### `funds()`
Gets funds and cash balances of the connected trading account.
* **`**kwargs`**: None.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": {
          "availablecash": "100000.00",    // Available trading cash
          "collateral": "0.00",            // Value of pledged shares
          "m2mrealized": "1500.00",        // Booked profit/loss
          "m2munrealized": "-200.00",      // Floating profit/loss
          "utiliseddebits": "5000.00"      // Margin blocked for open positions
      }
  }
  ```

### `orderbook()`
Gets the history of all orders sent today with status statistics.
* **`**kwargs`**: None.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": {
          "orders": [
              {
                  "orderid": "2606300001",
                  "symbol": "SBIN",
                  "exchange": "NSE",
                  "action": "BUY",
                  "pricetype": "LIMIT",
                  "price": 750.50,
                  "trigger_price": 0.0,
                  "quantity": 10,
                  "product": "MIS",
                  "order_status": "complete",   // complete, pending, rejected, cancelled
                  "timestamp": "30-Jun-2026 13:15:20"
              }
          ],
          "statistics": {
              "total_buy_orders": 1,
              "total_sell_orders": 0,
              "total_completed_orders": 1,
              "total_open_orders": 0,
              "total_rejected_orders": 0
          }
      }
  }
  ```

### `tradebook()`
Gets the list of executed trades for the day.
* **`**kwargs`**: None.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": [
          {
              "orderid": "2606300001",
              "symbol": "SBIN",
              "exchange": "NSE",
              "action": "BUY",
              "average_price": 750.50,
              "quantity": 10,
              "product": "MIS",
              "trade_value": 7505.00,
              "timestamp": "30-Jun-2026 13:15:20"
          }
      ]
  }
  ```

### `positionbook()`
Gets the current active net positions.
* **`**kwargs`**: None.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": [
          {
              "symbol": "SBIN",
              "exchange": "NSE",
              "product": "MIS",
              "quantity": 10,             // Positive for long, negative for short
              "average_price": 750.50
          }
      ]
  }
  ```

### `holdings()`
Gets the long-term equity stock holdings.
* **`**kwargs`**: None.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": {
          "holdings": [
              {
                  "symbol": "RELIANCE",
                  "exchange": "NSE",
                  "product": "CNC",
                  "quantity": 50,
                  "pnl": 2500.00,
                  "pnlpercent": 2.15
              }
          ],
          "statistics": {
              "totalholdingvalue": 120000.00,
              "totalinvvalue": 117500.00,
              "totalprofitandloss": 2500.00,
              "totalpnlpercentage": 2.13
          }
      }
  }
  ```

### `analyzerstatus()` & `analyzertoggle(mode)`
Queries or changes paper trading (simulation) vs live execution modes.
* **Parameters**:
  * `mode` (bool): `True` for analyze mode, `False` for live.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": {
          "mode": "analyze",
          "analyze_mode": true,
          "total_logs": 5,
          "message": "Analyzer mode switched to analyze"
      }
  }
  ```

### `margin(positions)`
Computes margin requirements for a basket of trades, factoring in hedge/spread benefits.
* **Parameters**:
  * `positions` (list[dict]): Basket of positions (max 50). Each dictionary contains:
    * `symbol` (str), `exchange` (str), `action` (str: `"BUY"` or `"SELL"`), `product` (str: `"CNC"`, `"MIS"`, `"NRML"`), `pricetype` (str: `"MARKET"`, `"LIMIT"`, `"SL"`, `"SL-M"`), `quantity` (str/int).
    * `price` (str/float, optional): Defaults to `"0"`.
    * `trigger_price` (str/float, optional): Defaults to `"0"`.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": {
          "total_margin_required": 145000.00,
          "span_margin": 105000.00,
          "exposure_margin": 40000.00
      }
  }
  ```

---

## 3. Orders Management

Inherited from the `OrderAPI` class, these methods route trading actions to the broker.

### `placeorder()`
Sends a new order to the broker.
* **Parameters**:
  * `strategy` (str): Label for the strategy. Defaults to `"Python"`.
  * `symbol` (str): Trading ticker symbol. Required.
  * `action` (str): `"BUY"` or `"SELL"`. Required.
  * `exchange` (str): Exchange code (e.g. `"NSE"`, `"NFO"`, `"MCX"`). Required.
  * `price_type` (str): `"MARKET"`, `"LIMIT"`, `"SL"`, `"SL-M"`. Defaults to `"MARKET"`.
  * `product` (str): `"MIS"` (intraday), `"CNC"` (delivery), `"NRML"` (derivatives). Defaults to `"MIS"`.
  * `quantity` (int/str): Order quantity. Defaults to `1`.
  * **`**kwargs`**:
    * `price` (str/float): Required for `LIMIT` and `SL` orders.
    * `trigger_price` (str/float): Required for `SL` and `SL-M` orders.
    * `disclosed_quantity` (str/int): Visible exchange quantity.
    * `target` (str/float): Target price (Bracket orders).
    * `stoploss` (str/float): Stop loss price (Bracket orders).
    * `trailing_sl` (str/float): Trailing stop loss points.
* **Returns**:
  ```json
  {
      "status": "success",
      "orderid": "2606300002",
      "symbol": "SBIN",
      "exchange": "NSE",
      "message": "Order placed successfully"
  }
  ```

### `placesmartorder()`
Places an order that compares the target `position_size` with the current position size to decide whether to place a buy or sell order.
* **Parameters**:
  * Same as `placeorder()`, plus `position_size` (int/str) representing the target quantity.
* **Returns**: Same as `placeorder()`.

### `basketorder(orders)`
Executes multiple orders at the same time.
* **Parameters**:
  * `orders` (list[dict]): List of order payloads.
* **Returns**:
  ```json
  {
      "status": "success",
      "results": [
          {
              "symbol": "SBIN",
              "status": "success",
              "orderid": "2606300003"
          }
      ]
  }
  ```

### `splitorder(quantity, splitsize)`
Splits a large order into multiple iceberg chunks.
* **Parameters**: Same as `placeorder()`, plus:
  * `quantity` (int/str): Total iceberg size.
  * `splitsize` (int/str): Quantity per slice.
* **Returns**:
  ```json
  {
      "status": "success",
      "total_quantity": 1000,
      "split_size": 200,
      "results": [
          {"order_num": 1, "orderid": "2606300004", "quantity": 200, "status": "success"},
          ...
      ]
  }
  ```

### `orderstatus(order_id)`
Check the status of a specific order.
* **Parameters**:
  * `order_id` (str): Unique order ID.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": {
          "orderid": "2606300001",
          "symbol": "SBIN",
          "exchange": "NSE",
          "action": "BUY",
          "pricetype": "LIMIT",
          "price": 750.50,
          "trigger_price": 0.0,
          "quantity": 10,
          "product": "MIS",
          "order_status": "complete",
          "timestamp": "30-Jun-2026 13:15:20"
      }
  }
  ```

### `modifyorder()`, `cancelorder()`, `closeposition()`, `cancelallorder()`
Operations to manage open order/position lifecycles.
* **`modifyorder` Parameters**:
  * `order_id` (str), `symbol` (str), `action` (str), `exchange` (str), `product` (str), `quantity` (int/str), `price` (str/float), `disclosed_quantity` (str/float), `trigger_price` (str/float).
* **`cancelorder` Parameters**: `order_id` (str).
* **`closeposition` & `cancelallorder` Parameters**: `strategy` (str).

---

## 4. Options-Specific Orders & Greeks

Inherited from the `OptionsAPI` class, these methods facilitate options trading and option chain calculations.

### `optionsorder()`
Automates symbol resolution using strike offset notation.
* **Parameters**:
  * `underlying` (str): Underlying symbol (e.g. `"NIFTY"`, `"BANKNIFTY"`). Required.
  * `exchange` (str): Underlying exchange code (e.g. `"NSE_INDEX"`). Required.
  * `offset` (str): Strike offset relative to ATM (e.g. `"ATM"`, `"ITM1"`, `"ITM2"`, `"OTM1"`, `"OTM2"`). Required.
  * `option_type` (str): `"CE"` or `"PE"`. Required.
  * `action` (str): `"BUY"` or `"SELL"`. Required.
  * `quantity` (int/str): Quantity. Required.
  * `expiry_date` (str, optional): Date in `DDMMMYY` format.
  * `price_type` (str, optional): Defaults to `"MARKET"`.
  * `product` (str, optional): Defaults to `"MIS"`.
  * **`**kwargs`**:
    * `price` (str/float): Required for `LIMIT` orders.
    * `trigger_price` (str/float): Required for `SL` and `SL-M` orders.
    * `disclosed_quantity` (str/int): Visible quantity.
* **Returns**:
  ```json
  {
      "status": "success",
      "orderid": "2606300005",
      "symbol": "NIFTY26FEB2623100CE",
      "exchange": "NFO",
      "underlying": "NIFTY",
      "underlying_ltp": 23210.50,
      "offset": "ITM2",
      "option_type": "CE"
  }
  ```

### `optionsmultiorder(legs)`
Places multi-leg option strategies in a single call.
* **Parameters**:
  * `legs` (list[dict]): Leg options: each dict contains `offset` (str), `option_type` (str), `action` (str), `quantity` (int/str), and optional `expiry_date` (str), `pricetype` (str), `price` (str/float), `product` (str).
* **Returns**:
  ```json
  {
      "status": "success",
      "results": [
          {
              "symbol": "NIFTY26FEB2623200CE",
              "offset": "ATM",
              "option_type": "CE",
              "action": "BUY",
              "quantity": 50,
              "orderid": "2606300006"
          }
      ]
  }
  ```

### `optionchain()`
Returns the complete option chain.
* **Parameters**:
  * `underlying` (str): Symbol. Required.
  * `exchange` (str): Exchange. Required.
  * `expiry_date` (str, optional): `DDMMMYY` format.
  * `strike_count` (int, optional): Number of strikes above and below ATM.
* **Returns**:
  ```json
  {
      "status": "success",
      "underlying": "NIFTY",
      "underlying_ltp": 23210.50,
      "expiry_date": "26FEB26",
      "atm_strike": 23200,
      "chain": [
          {
              "strike": 23200,
              "ce": {
                  "symbol": "NIFTY26FEB2623200CE",
                  "label": "ATM",
                  "ltp": 120.50,
                  "bid": 120.10,
                  "ask": 120.90,
                  "open": 100.00,
                  "high": 150.00,
                  "low": 90.00,
                  "prev_close": 110.00,
                  "volume": 50000,
                  "oi": 1500000,
                  "lotsize": 50,
                  "tick_size": 0.05
              },
              "pe": { ... pe schema is identical to ce ... }
          }
      ]
  }
  ```

### `optiongreeks()`
Calculates option pricing greeks and implied volatility using the Black-76 model.
* **Parameters**:
  * `symbol` (str), `exchange` (str). Required.
  * `interest_rate` (float, optional): Annual risk-free interest rate (e.g. `6.5`).
  * `forward_price` (float, optional): Synthetic futures/forward price.
  * `underlying_symbol` (str, optional): Custom underlying symbol.
  * `underlying_exchange` (str, optional): Custom underlying exchange.
  * `expiry_time` (str, optional): Custom expiry time in `"HH:MM"` format.
* **Returns**:
  ```json
  {
      "status": "success",
      "symbol": "NIFTY26FEB2623200CE",
      "exchange": "NFO",
      "underlying": "NIFTY",
      "strike": 23200,
      "option_type": "CE",
      "expiry_date": "2026-02-26",
      "days_to_expiry": 28,
      "spot_price": 23210.50,
      "option_price": 120.50,
      "interest_rate": 6.5,
      "implied_volatility": 15.42,
      "greeks": {
          "delta": 0.524,
          "gamma": 0.0012,
          "theta": -4.25,
          "vega": 12.30,
          "rho": 8.45
      }
  }
  ```

---

## 5. Data Management

Inherited from the `DataAPI` class, these methods retrieve symbol specs and prices.

### `quotes()` & `multiquotes()`
Retrieves real-time quotes.
* **`quotes` Parameters**: `symbol` (str), `exchange` (str). Required.
* **`multiquotes` Parameters**: `symbols` (list[dict]) list of `{"symbol": "...", "exchange": "..."}` dicts. Required.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": {
          "symbol": "SBIN",
          "exchange": "NSE",
          "ltp": 750.50,
          "open": 740.00,
          "high": 755.00,
          "low": 738.00,
          "close": 748.00,
          "volume": 1200000,
          "oi": 0,
          "bid": 750.20,
          "ask": 750.80,
          "bid_qty": 500,
          "ask_qty": 400,
          "timestamp": "30-Jun-2026 13:15:20"
      }
  }
  ```

### `depth()`
Retrieves the order book (depth queue) for a symbol.
* **Parameters**: `symbol` (str), `exchange` (str). Required.
* **Returns**:
  ```json
  {
      "status": "success",
      "data": {
          "symbol": "SBIN",
          "exchange": "NSE",
          "ltp": 750.50,
          "depth": {
              "buy": [{"price": 750.20, "quantity": 100, "orders": 2}, ...],
              "sell": [{"price": 750.80, "quantity": 150, "orders": 3}, ...]
          }
      }
  }
  ```

### `history()`
Returns historical data directly in a **Pandas DataFrame** format.
* **Parameters**:
  * `symbol` (str), `exchange` (str), `interval` (str: `"1m"`, `"5m"`, `"15m"`, `"1h"`, `"D"`), `start_date` (str: `"YYYY-MM-DD"`), `end_date` (str: `"YYYY-MM-DD"`). Required.
  * `source` (str, optional): `"api"` (calls broker) or `"db"` (OpenAlgo DuckDB). Defaults to `"api"`.
* **Returns**: `pandas.DataFrame` where index is localized to IST for intraday intervals.
  * **Columns**: `open` (float), `high` (float), `low` (float), `close` (float), `volume` (int/float).

### `instruments()`
Downloads the broker instrument master database.
* **Parameters**:
  * `exchange` (str, optional): NSE, BSE, NFO, BFO, MCX, CDS, etc. If omitted, downloads all exchanges.
* **Returns**: `pandas.DataFrame` containing the columns:
  * `symbol` (str), `brsymbol` (str), `name` (str), `exchange` (str), `token` (str), `expiry` (str), `strike` (float), `lotsize` (int), `instrumenttype` (str), `tick_size` (float).

### `syntheticfuture()`
Calculates the synthetic futures price (`ATM Strike + Call Premium - Put Premium`).
* **Parameters**:
  * `underlying` (str), `exchange` (str), `expiry_date` (str: `DDMMMYY`). Required.
* **Returns**:
  ```json
  {
      "status": "success",
      "underlying": "NIFTY",
      "underlying_ltp": 23210.50,
      "expiry": "26FEB26",
      "atm_strike": 23200,
      "synthetic_future_price": 23205.20
  }
  ```

---

## 6. WebSocket Feed

WebSocket client for real-time market streams running in a background thread.

### Client Subscriptions

```python
def on_price_update(data):
    # data: {'type': 'market_data', 'symbol': 'NIFTY', 'exchange': 'NSE_INDEX', 'data': {'ltp': 23210.50, ...}}
    print(data)

# Connect to WebSocket
client.connect()

# Subscribe
client.subscribe_ltp([{"exchange": "NSE_INDEX", "symbol": "NIFTY"}], on_data_received=on_price_update)
```

### WebSocket Streaming Methods

| Method | Subscription Payload Received |
|---|---|
| `subscribe_ltp(instruments)` | `{ "symbol": "...", "data": { "ltp": 23210.50 } }` |
| `subscribe_quote(instruments)` | `{ "symbol": "...", "data": { "open": 23100.0, "high": 23250.0, "low": 23090.0, "close": 23110.0, "ltp": 23210.50, "volume": 12000, "timestamp": 1785984700 } }` |
| `subscribe_depth(instruments)` | `{ "symbol": "...", "data": { "ltp": 23210.50, "depth": { "buy": [...], "sell": [...] } } }` |

* **Unsubscribing**: Pass the same instruments list to `unsubscribe_ltp(instruments)`, `unsubscribe_quote(instruments)`, or `unsubscribe_depth(instruments)`.
* **Caching**: The current state is saved in `client.ltp_data`, `client.quotes_data`, and `client.depth_data`. Query using `client.get_ltp(exchange, symbol)`, `client.get_quotes(exchange, symbol)`, and `client.get_depth(exchange, symbol)`.

---

## 7. Utilities

Utility APIs for holidays, timings, and messaging.

### `holidays()`
Retrieves list of market holidays for the specified year.
* **Parameters**:
  * `year` (int, optional): Defaults to current year.
* **Returns**:
  ```json
  {
      "status": "success",
      "year": 2026,
      "data": [
          {
              "date": "2026-01-26",
              "description": "Republic Day",
              "holiday_type": "TRADING_HOLIDAY",
              "closed_exchanges": ["NSE", "BSE", "NFO", "BFO", "CDS", "BCD"],
              "open_exchanges": []
          }
      ]
  }
  ```

### `telegram()`
Sends an alert message to Telegram.
* **Parameters**:
  * `username` (str): OpenAlgo user login name (case sensitive). Required.
  * `message` (str): Markdown formatted message. Required.
  * `priority` (int, optional): Message priority `1-10`. Defaults to 5.
* **Returns**:
  ```json
  {
      "status": "success",
      "message": "Notification sent successfully"
  }
  ```

### `whatsapp()`
Sends a WhatsApp message with optional image/document attachments.
* **Parameters**:
  * `message` (str, optional): Plain text body.
  * `to` (str or list[str], optional): E.164 phone number(s) (max 5 numbers in list).
  * `username` (str, optional): OpenAlgo username.
  * `image` (str, optional): Server-local file path to an image file.
  * `document` (str, optional): Server-local file path to a document.
  * `caption` (str, optional): Caption text.
  * `filename` (str, optional): Attachment display filename.
  * `wait_for_delivery` (bool, optional): Blocks until status returns if `True`.
* **Returns**:
  ```json
  {
      "status": "success",
      "message": "Delivered to 1, failed 0",
      "data": {
          "sent": ["919876543210"],
          "failed": [],
          "skipped": 0
      }
  }
  ```
