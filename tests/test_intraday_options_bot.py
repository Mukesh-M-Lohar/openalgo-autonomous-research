import os
import sys

import pytest

# Ensure project root and scripts are in system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts/bot")))

from openalgo import api

from scripts.bot.intraday_options_bot import FnOScanner, OptionChainAnalyzer


@pytest.fixture
def client():
    # Instantiate actual client connecting to the running local server
    API_KEY = os.getenv(
        "OPENALGO_API_KEY", "b45feb0a6973ed00fe86d25ace49d4da8dfe8d0a78c334455d46254ded28a26d"
    )
    API_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")
    WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")
    return api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)


def test_actual_optionchain_analyzer(client):
    analyzer = OptionChainAnalyzer(client)

    # 1. Test get_nearest_expiry on actual server
    expiry = analyzer.get_nearest_expiry("NIFTY")
    assert expiry != ""
    assert len(expiry) == 7
    assert expiry[-2:].isdigit()

    # 2. Test fetch_chain without expiry_date (dynamically resolves NIFTY)
    chain = analyzer.fetch_chain("NIFTY", "NSE_INDEX")
    assert chain is not None
    assert chain.get("status") == "success"
    assert "chain" in chain
    assert len(chain["chain"]) > 0


def test_actual_fno_scanner(client):
    analyzer = OptionChainAnalyzer(client)
    scanner = FnOScanner(client, analyzer)

    # Use a small subset watchlist to check API scanning
    watchlist = [
        {"symbol": "NIFTY", "exchange": "NSE_INDEX", "opt_exchange": "NFO", "lot": 75},
        {"symbol": "BANKNIFTY", "exchange": "NSE_INDEX", "opt_exchange": "NFO", "lot": 30},
    ]

    picks = scanner.scan_and_rank(watchlist)
    # The scan should run without exceptions and return scored picks
    assert isinstance(picks, list)
    if picks:
        assert "symbol" in picks[0]
        assert "score" in picks[0]


def test_actual_funds_check(client):
    # Verify that client.funds works cleanly on the running server
    resp = client.funds()
    assert isinstance(resp, dict)
    assert resp.get("status") == "success"


def test_actual_orders_and_notifications(client):
    # 1. Force analyze (paper) mode for safety
    toggle_resp = client.analyzertoggle(True)
    assert toggle_resp.get("status") == "success"

    status_resp = client.analyzerstatus()
    assert status_resp.get("status") == "success"
    assert status_resp.get("data", {}).get("analyze_mode") is True

    # 2. Resolve nearest expiry
    analyzer = OptionChainAnalyzer(client)
    expiry = analyzer.get_nearest_expiry("NIFTY")
    assert expiry != ""

    # Resolve lot size dynamically
    lot_size = 75
    if analyzer._instruments_df is not None:
        sym_df = analyzer._instruments_df[analyzer._instruments_df["name"] == "NIFTY"]
        if not sym_df.empty:
            lot_size = int(sym_df["lotsize"].iloc[0])

    # 3. Test Options Order Placement
    order_resp = client.optionsorder(
        strategy="PyTest_Verification",
        underlying="NIFTY",
        exchange="NSE_INDEX",
        offset="ATM",
        option_type="CE",
        action="BUY",
        quantity=lot_size,
        product="NRML",
        expiry_date=expiry,
    )
    assert isinstance(order_resp, dict)
    assert order_resp.get("status") == "success"
    assert "orderid" in order_resp

    # 4. Test WhatsApp Notification API
    wa_resp = client.whatsapp("Hello from options bot unit tests!", to=["919876543210"])
    assert isinstance(wa_resp, dict)
    assert wa_resp.get("status") in ("success", "error")

    # 5. Test Telegram Notification API
    tg_resp = client.telegram(
        username="test_options_bot_user", message="Hello from options bot unit tests!"
    )
    assert isinstance(tg_resp, dict)
    assert tg_resp.get("status") in ("success", "error")
