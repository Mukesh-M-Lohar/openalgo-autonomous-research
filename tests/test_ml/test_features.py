"""Unit tests for FeatureExtractor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_engine.ml.features.extractor import FeatureExtractor


@pytest.fixture
def dummy_ohlcv() -> pd.DataFrame:
    """Create a dummy OHLCV DataFrame with a DatetimeIndex."""
    dates = pd.date_range(start="2026-01-01 09:15:00", periods=200, freq="15min")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.normal(0, 1, 200))
    open_ = close + np.random.normal(0, 0.5, 200)
    high = np.maximum(open_, close) + np.random.uniform(0, 1, 200)
    low = np.minimum(open_, close) - np.random.uniform(0, 1, 200)
    volume = np.random.randint(1000, 10000, 200).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_feature_extractor_all_enabled(dummy_ohlcv: pd.DataFrame) -> None:
    config = {
        "technical": True,
        "candle": True,
        "volatility": True,
        "volume": True,
        "time": True,
        "session": True,
        "gap": True,
        "rolling": True,
        "rolling_windows": [5, 10],
    }
    extractor = FeatureExtractor(config)
    features = extractor.extract_features(dummy_ohlcv)

    # Check that it returns a DataFrame with the same index length
    assert len(features) == len(dummy_ohlcv)

    # Check some expected columns are present
    expected_cols = [
        "feat_candle_body",
        "feat_candle_range",
        "feat_gap_overnight",
        "feat_tech_sma_10_ratio",
        "feat_tech_rsi_14",
        "feat_tech_macd_ratio",
        "feat_vol_natr",
        "feat_volume_log_return",
        "feat_time_hour",
        "feat_session_morning",
        "feat_roll_close_mean_5",
    ]
    for col in expected_cols:
        assert col in features.columns

    # Make sure we don't have infs in the output DataFrame
    assert not np.isinf(features.to_numpy()).any()


def test_feature_extractor_disabled(dummy_ohlcv: pd.DataFrame) -> None:
    config = {
        "technical": False,
        "candle": False,
        "volatility": False,
        "volume": False,
        "time": False,
        "session": False,
        "gap": False,
        "rolling": False,
    }
    extractor = FeatureExtractor(config)
    features = extractor.extract_features(dummy_ohlcv)

    # Columns should be empty
    assert len(features.columns) == 0
