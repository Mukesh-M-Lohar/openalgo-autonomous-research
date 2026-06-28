"""Unit tests for LabelGenerator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.ml.labels.generator import LabelGenerator


@pytest.fixture
def dummy_ohlcv() -> pd.DataFrame:
    dates = pd.date_range(start="2026-01-01", periods=100)
    close = 100.0 * (1.001 ** np.arange(100))  # constantly upward trending
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
        index=dates,
    )


def test_label_generator_binary(dummy_ohlcv: pd.DataFrame) -> None:
    config = {"type": "binary", "future_horizon": 5, "threshold": 0.002}
    generator = LabelGenerator(config)
    labels = generator.generate_labels(dummy_ohlcv)

    assert len(labels) == len(dummy_ohlcv)
    # Check timezone/horizon NaNs
    assert pd.isna(labels.iloc[-5:]).all()
    # Trend is up, returns are positive: should be 1
    assert labels.iloc[0] == 1


def test_label_generator_three_class(dummy_ohlcv: pd.DataFrame) -> None:
    config = {"type": "three_class", "future_horizon": 5, "threshold": 0.002}
    generator = LabelGenerator(config)
    labels = generator.generate_labels(dummy_ohlcv)

    assert len(labels) == len(dummy_ohlcv)
    assert pd.isna(labels.iloc[-5:]).all()
    # Trend is positive: should be Buy (2)
    assert labels.iloc[0] == 2


def test_label_generator_regression(dummy_ohlcv: pd.DataFrame) -> None:
    config = {"type": "regression", "future_horizon": 5}
    generator = LabelGenerator(config)
    labels = generator.generate_labels(dummy_ohlcv)

    assert len(labels) == len(dummy_ohlcv)
    assert pd.isna(labels.iloc[-5:]).all()
    # Expected return over 5 days: log(1.001^5) = 5 * log(1.001) ~ 0.005
    assert pytest.approx(labels.iloc[0], abs=1e-4) == 5 * np.log(1.001)


def test_label_generator_meta_labeling(dummy_ohlcv: pd.DataFrame) -> None:
    config = {"type": "meta_labeling", "threshold": 0.0}
    generator = LabelGenerator(config)

    # 3 dummy trades
    trades = [
        {"entry_time": "2026-01-05 00:00:00", "pnl_pct": 1.5},
        {"entry_time": "2026-01-10 00:00:00", "pnl_pct": -0.5},
        {"entry_time": "2026-01-15 00:00:00", "pnl_pct": 2.2},
    ]

    labels = generator.generate_labels(dummy_ohlcv, trades=trades)

    assert len(labels) == len(dummy_ohlcv)
    # Check that labels are not NaN only at trade timestamps
    trade_dates = [pd.to_datetime(t["entry_time"]) for t in trades]
    for date in dummy_ohlcv.index:
        if date in trade_dates:
            assert not pd.isna(labels.loc[date])
        else:
            assert pd.isna(labels.loc[date])

    # Check correct labeling: PnL > 0 => 1, PnL <= 0 => 0
    assert labels.loc[pd.to_datetime("2026-01-05")] == 1
    assert labels.loc[pd.to_datetime("2026-01-10")] == 0
    assert labels.loc[pd.to_datetime("2026-01-15")] == 1
