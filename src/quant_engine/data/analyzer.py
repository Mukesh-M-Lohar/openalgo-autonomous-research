"""Market regime analysis and stationarity tests.

Based on Ernie Chan's 'Algorithmic Trading' (2013, Wiley):
- Hurst Exponent for trend vs mean-reversion classification (Ch.2 pp.57-64)
- Half-life of mean reversion (Ch.2 pp.64-66)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RANDOM_WALK = "random_walk"


@dataclass
class RegimeAnalysis:
    regime: MarketRegime
    hurst_exponent: float
    half_life_bars: float


class RegimeAnalyzer:
    """Analyzes price data to determine the market regime."""

    def __init__(self, hurst_trending_threshold: float = 0.55, hurst_mr_threshold: float = 0.45):
        self._hurst_trending_threshold = hurst_trending_threshold
        self._hurst_mr_threshold = hurst_mr_threshold

    def analyze(self, df: pd.DataFrame, price_col: str = "close") -> RegimeAnalysis:
        """Analyze the given dataframe and return market regime parameters."""
        if price_col not in df.columns:
            raise ValueError(f"Column '{price_col}' not found in dataframe.")

        prices = df[price_col].values

        # Calculate Hurst Exponent
        hurst = self._compute_hurst(prices)

        # Calculate Half-Life
        half_life = self._compute_half_life(prices)

        # Determine Regime
        if hurst > self._hurst_trending_threshold:
            regime = MarketRegime.TRENDING
        elif hurst < self._hurst_mr_threshold:
            regime = MarketRegime.MEAN_REVERTING
        else:
            regime = MarketRegime.RANDOM_WALK

        logger.info(
            f"Regime Analysis: Hurst={hurst:.4f}, Half-Life={half_life:.1f} bars "
            f"-> {regime.value.upper()}"
        )

        return RegimeAnalysis(
            regime=regime,
            hurst_exponent=hurst,
            half_life_bars=half_life,
        )

    def _compute_hurst(self, prices: np.ndarray, max_lag: int = 100) -> float:
        """Compute the Hurst Exponent using variance of differences method.

        H < 0.5: Mean reverting
        H = 0.5: Geometric random walk
        H > 0.5: Trending
        """
        if len(prices) < max_lag + 10:
            max_lag = max(2, len(prices) // 4)

        lags = np.arange(2, max_lag)
        # Compute standard deviation of price differences for various lags
        # std(p(t) - p(t-lag)) ~ lag^H
        tau = [np.std(prices[lag:] - prices[:-lag]) for lag in lags]

        # Avoid log(0) if data is constant
        if np.any(np.array(tau) == 0):
            return 0.5

        # Fit a line to log-log plot to extract H
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(poly[0])

    def _compute_half_life(self, prices: np.ndarray) -> float:
        """Compute the half-life of mean reversion.

        Using Ornstein-Uhlenbeck differential equation approximation:
        dy(t) = lambda * y(t-1) + mu
        Half-life = -ln(2) / lambda
        """
        if len(prices) < 3:
            return float("inf")

        y_prev = prices[:-1]
        dy = prices[1:] - y_prev

        # Regression dy = lambda * y_prev + mu
        poly = np.polyfit(y_prev, dy, 1)
        lam = poly[0]

        # If lambda >= 0, it's not mean-reverting
        if lam >= -1e-8:
            return float("inf")

        return float(-np.log(2) / lam)
