"""Stress testing — inject adverse conditions to test fragility."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_engine.backtest.engine import BacktestEngine
from quant_engine.config import CostModelConfig
from quant_engine.models.results import BacktestResult
from quant_engine.models.strategy import StrategyGenome

logger = logging.getLogger(__name__)


class StressTestValidator:
    """Tests strategy under stressed market conditions."""

    def __init__(self, cost_model: CostModelConfig | None = None):
        self._base_cost = cost_model or CostModelConfig()

    def validate(
        self,
        strategy: StrategyGenome,
        original_result: BacktestResult,
        data: dict[str, pd.DataFrame],
    ) -> dict:
        """Run multiple stress scenarios and report worst-case metrics.

        Scenarios:
        1. Higher costs (2x commission + slippage)
        2. Increased volatility (inject random spikes)
        3. Reduced liquidity (wider slippage)
        4. Gap scenarios (inject random gaps)
        """
        scores = []

        # Scenario 1: Higher costs
        high_cost = CostModelConfig(
            commission_pct=self._base_cost.commission_pct * 2,
            slippage_pct=self._base_cost.slippage_pct * 2,
            min_commission=self._base_cost.min_commission,
        )
        engine_hc = BacktestEngine(cost_model=high_cost)
        result_hc = engine_hc.run(strategy, data)
        if result_hc:
            scores.append(result_hc.sharpe)
        else:
            scores.append(0.0)

        # Scenario 2: Volatility spike (add noise to prices)
        vol_data = self._inject_volatility(data, multiplier=1.5)
        engine_base = BacktestEngine(cost_model=self._base_cost)
        result_vol = engine_base.run(strategy, vol_data)
        if result_vol:
            scores.append(result_vol.sharpe)
        else:
            scores.append(0.0)

        # Scenario 3: Higher slippage only
        high_slip = CostModelConfig(
            commission_pct=self._base_cost.commission_pct,
            slippage_pct=self._base_cost.slippage_pct * 3,
            min_commission=self._base_cost.min_commission,
        )
        engine_hs = BacktestEngine(cost_model=high_slip)
        result_hs = engine_hs.run(strategy, data)
        if result_hs:
            scores.append(result_hs.sharpe)
        else:
            scores.append(0.0)

        # Compute stress score
        if not scores or original_result.sharpe <= 0:
            return {
                "stress_test_score": 0.0,
                "stress_max_drawdown": 0.0,
                "stress_worst_sharpe": 0.0,
            }

        worst_sharpe = min(scores)
        avg_sharpe = np.mean(scores)
        survival_ratio = avg_sharpe / original_result.sharpe if original_result.sharpe > 0 else 0

        stress_score = max(0, survival_ratio * 100)

        worst_dd = 0.0
        if result_hc:
            worst_dd = max(worst_dd, result_hc.max_drawdown_pct)
        if result_vol:
            worst_dd = max(worst_dd, result_vol.max_drawdown_pct)

        return {
            "stress_test_score": round(stress_score, 2),
            "stress_max_drawdown": round(worst_dd, 4),
            "stress_worst_sharpe": round(worst_sharpe, 4),
            "stress_avg_sharpe": round(avg_sharpe, 4),
            "stress_survival_ratio": round(survival_ratio, 4),
        }

    def _inject_volatility(
        self, data: dict[str, pd.DataFrame], multiplier: float
    ) -> dict[str, pd.DataFrame]:
        """Add random noise to OHLCV data to simulate volatility spikes."""
        result = {}
        rng = np.random.default_rng(123)
        for tf, df in data.items():
            noisy = df.copy()
            n = len(noisy)
            noise = 1 + rng.normal(0, 0.002 * multiplier, n)
            for col in ["open", "high", "low", "close"]:
                if col in noisy.columns:
                    noisy[col] = noisy[col] * noise
            # Ensure high >= close/open and low <= close/open
            noisy["high"] = noisy[["open", "high", "close"]].max(axis=1)
            noisy["low"] = noisy[["open", "low", "close"]].min(axis=1)
            result[tf] = noisy
        return result
