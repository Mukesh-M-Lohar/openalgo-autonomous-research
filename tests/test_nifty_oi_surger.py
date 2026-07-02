import os
import sys
from unittest.mock import MagicMock

import pytest

# Ensure project root and scripts are in system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts/bot")))

from scripts.bot.nifty_oi_surger_bot import NiftyOISurgerBot


@pytest.fixture
def actual_bot():
    bot = NiftyOISurgerBot()
    # Mock only optionsorder and whatsapp endpoints to avoid placing orders and sending alerts
    bot.client.optionsorder = MagicMock(
        return_value={"status": "success", "symbol": "NIFTY07JUL2623900CE", "orderid": "MOCK-ORDER"}
    )
    bot.client.whatsapp = MagicMock(return_value={"status": "success"})
    yield bot


def test_actual_nearest_expiry(actual_bot):
    expiry = actual_bot.get_nearest_nifty_expiry()
    assert expiry != ""
    # Should follow format DDMMMYY (e.g. 07JUL26, 7 chars, ending in year digits)
    assert len(expiry) == 7
    assert expiry[-2:].isdigit()


def test_actual_warmup_from_api(actual_bot):
    # Warm up history using actual API calls to instruments, optionchain, and history
    actual_bot.warmup_history_from_api()

    # Verify history is populated
    assert len(actual_bot.oi_history) > 0
    # Spot price should be a positive float
    assert actual_bot.oi_history[-1]["spot"] > 0.0
    # Total volume and CE/PE OI should be valid numbers
    assert isinstance(actual_bot.oi_history[-1]["total_volume"], (int, float))


def test_actual_check_reentry_and_recovery(actual_bot):
    # Execute recovery query on actual tradebook and positionbook API endpoints
    actual_bot.check_reentry_and_recovery()

    # Should not throw any errors and self.position should be None or Dict depending on tradebook
    assert actual_bot.position is None or isinstance(actual_bot.position, dict)


def test_actual_check_signals(actual_bot):
    # Set history with at least MIN_OI_ROWS rows to compute indicators
    actual_bot.warmup_history_from_api()

    if len(actual_bot.oi_history) >= 17:
        # Check signals using actual option chain data
        # We mocked optionsorder inside actual_bot fixture to prevent any order execution
        actual_bot.check_signals()
        assert actual_bot.position is None or isinstance(actual_bot.position, dict)
