"""Walk-forward validation — rolling window train/test analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_engine.backtest.engine import BacktestEngine
from quant_engine.config import CostModelConfig
from quant_engine.data.preprocessor import DataPreprocessor
from quant_engine.models.strategy import StrategyGenome

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Evaluates strategy robustness via walk-forward analysis."""

    def __init__(
        self,
        train_bars: int = 500,
        test_bars: int = 125,
        step_bars: int | None = None,
        min_windows: int = 4,
        cost_model: CostModelConfig | None = None,
    ):
        self._train_bars = train_bars
        self._test_bars = test_bars
        self._step_bars = step_bars or test_bars
        self._min_windows = min_windows
        self._engine = BacktestEngine(cost_model=cost_model)
        self._preprocessor = DataPreprocessor()

    def validate(self, strategy: StrategyGenome, data: dict[str, pd.DataFrame]) -> dict:
        """Run walk-forward analysis.

        Returns dict with walk_forward_score, walk_forward_consistency, window details.
        """
        primary_tf = strategy.timeframes_used[0].value
        df = data.get(primary_tf)
        if df is None or len(df) < self._train_bars + self._test_bars:
            return {"walk_forward_score": 0.0, "walk_forward_consistency": 0.0}

        windows = self._preprocessor.rolling_windows(
            df, self._train_bars, self._test_bars, self._step_bars
        )

        if len(windows) < self._min_windows:
            return {"walk_forward_score": 0.0, "walk_forward_consistency": 0.0}

        window_results = []
        for train_df, test_df in windows:
            # Backtest on test window
            test_data = {primary_tf: test_df}
            result = self._engine.run(strategy, test_data)
            if result is not None:
                window_results.append(
                    {
                        "sharpe": result.sharpe,
                        "profit_factor": result.profit_factor,
                        "trades": result.total_trades,
                        "profitable": result.net_profit_pct > 0,
                    }
                )
            else:
                window_results.append(
                    {
                        "sharpe": 0.0,
                        "profit_factor": 0.0,
                        "trades": 0,
                        "profitable": False,
                    }
                )

        profitable_windows = sum(1 for w in window_results if w["profitable"])
        consistency = profitable_windows / len(window_results)

        sharpes = [w["sharpe"] for w in window_results]
        avg_sharpe = np.mean(sharpes) if sharpes else 0.0
        score = consistency * 50 + min(avg_sharpe, 2.0) * 25

        return {
            "walk_forward_score": round(score, 2),
            "walk_forward_consistency": round(consistency, 4),
            "windows_tested": len(window_results),
            "profitable_windows": profitable_windows,
            "avg_window_sharpe": round(avg_sharpe, 4),
        }
