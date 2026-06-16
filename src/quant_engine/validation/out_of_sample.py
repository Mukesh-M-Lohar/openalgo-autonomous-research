"""Out-of-sample validation — test on unseen data."""

from __future__ import annotations

import logging

import pandas as pd

from quant_engine.backtest.engine import BacktestEngine
from quant_engine.config import CostModelConfig
from quant_engine.models.results import BacktestResult
from quant_engine.models.strategy import StrategyGenome

logger = logging.getLogger(__name__)


class OOSValidator:
    """Validates strategies on out-of-sample data not seen during backtest."""

    def __init__(self, cost_model: CostModelConfig | None = None):
        self._engine = BacktestEngine(cost_model=cost_model)

    def validate(
        self,
        strategy: StrategyGenome,
        in_sample_result: BacktestResult,
        oos_data: dict[str, pd.DataFrame],
    ) -> dict:
        """Run strategy on OOS data and measure performance decay.

        Returns dict with oos_sharpe, oos_sharpe_decay, oos_profit_factor.
        """
        oos_result = self._engine.run(strategy, oos_data)

        if oos_result is None:
            return {
                "oos_sharpe": 0.0,
                "oos_sharpe_decay": 1.0,
                "oos_profit_factor": 0.0,
                "oos_total_trades": 0,
            }

        # Measure decay from in-sample performance
        is_sharpe = in_sample_result.sharpe if in_sample_result.sharpe > 0 else 0.001
        sharpe_decay = 1.0 - (oos_result.sharpe / is_sharpe) if is_sharpe > 0 else 1.0
        sharpe_decay = max(0.0, min(sharpe_decay, 1.0))

        return {
            "oos_sharpe": round(oos_result.sharpe, 4),
            "oos_sharpe_decay": round(sharpe_decay, 4),
            "oos_profit_factor": round(oos_result.profit_factor, 4),
            "oos_total_trades": oos_result.total_trades,
            "oos_win_rate": round(oos_result.win_rate, 4),
            "oos_max_drawdown_pct": round(oos_result.max_drawdown_pct, 4),
        }
