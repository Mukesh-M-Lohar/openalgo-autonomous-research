"""Tests for market regime analyzer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from quant_engine.data.analyzer import MarketRegime, RegimeAnalyzer


@pytest.fixture
def trending_df():
    """Create a synthetic trending series (strong autocorrelation)."""
    np.random.seed(42)
    n = 1000
    # Create an autocorrelated series
    returns = np.random.randn(n) * 0.1
    # Smooth the returns to create strong trend (momentum)
    returns = pd.Series(returns).rolling(20, min_periods=1).mean().values + 0.1
    prices = np.cumsum(returns) + 100
    return pd.DataFrame({"close": prices})


@pytest.fixture
def mean_reverting_df():
    """Create a synthetic mean-reverting series (Ornstein-Uhlenbeck)."""
    np.random.seed(42)
    n = 1000
    prices = np.zeros(n)
    prices[0] = 100.0

    # dy = theta * (mu - y) * dt + sigma * dW
    theta = 0.1  # mean reversion speed
    mu = 100.0  # long term mean
    sigma = 1.0  # volatility

    for i in range(1, n):
        prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + sigma * np.random.randn()

    return pd.DataFrame({"close": prices})


@pytest.fixture
def random_walk_df():
    """Create a pure random walk."""
    np.random.seed(42)
    n = 1000
    returns = np.random.randn(n)
    prices = np.cumsum(returns) + 100
    return pd.DataFrame({"close": prices})


class TestRegimeAnalyzer:
    def test_trending_regime(self, trending_df):
        analyzer = RegimeAnalyzer(hurst_trending_threshold=0.55)
        analysis = analyzer.analyze(trending_df)

        # Hurst > 0.55 for trending
        assert analysis.hurst_exponent > 0.55
        assert analysis.regime == MarketRegime.TRENDING
        # Half life should be infinite for trending
        assert analysis.half_life_bars == float("inf")

    def test_mean_reverting_regime(self, mean_reverting_df):
        analyzer = RegimeAnalyzer(hurst_mr_threshold=0.45)
        analysis = analyzer.analyze(mean_reverting_df)

        # Hurst < 0.45 for mean reverting
        assert analysis.hurst_exponent < 0.45
        assert analysis.regime == MarketRegime.MEAN_REVERTING
        # Half life should be positive and finite
        assert 0 < analysis.half_life_bars < 50

    def test_random_walk_regime(self, random_walk_df):
        # A pure random walk should have Hurst ~ 0.5
        analyzer = RegimeAnalyzer(hurst_trending_threshold=0.55, hurst_mr_threshold=0.45)
        analysis = analyzer.analyze(random_walk_df)

        assert 0.45 <= analysis.hurst_exponent <= 0.55
        assert analysis.regime == MarketRegime.RANDOM_WALK

    def test_missing_column(self, trending_df):
        analyzer = RegimeAnalyzer()
        df = trending_df.rename(columns={"close": "price"})
        with pytest.raises(ValueError, match="Column 'close' not found"):
            analyzer.analyze(df)

    def test_custom_column(self, trending_df):
        analyzer = RegimeAnalyzer()
        df = trending_df.rename(columns={"close": "price"})
        analysis = analyzer.analyze(df, price_col="price")
        assert analysis.regime == MarketRegime.TRENDING
