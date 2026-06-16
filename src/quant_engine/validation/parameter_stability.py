"""Parameter stability testing — perturb parameters, check if performance holds."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_engine.backtest.engine import BacktestEngine
from quant_engine.config import CostModelConfig
from quant_engine.models.results import BacktestResult
from quant_engine.models.strategy import (
    CompositeCondition,
    ConditionNode,
    ConditionTree,
    IndicatorNode,
    StrategyGenome,
)

logger = logging.getLogger(__name__)


class ParameterStabilityValidator:
    """Tests if small parameter changes cause large performance swings."""

    def __init__(
        self,
        n_perturbations: int = 10,
        perturbation_pct: float = 0.15,
        cost_model: CostModelConfig | None = None,
    ):
        self._n_perturbations = n_perturbations
        self._perturbation_pct = perturbation_pct
        self._engine = BacktestEngine(cost_model=cost_model)

    def validate(
        self,
        strategy: StrategyGenome,
        original_result: BacktestResult,
        data: dict[str, pd.DataFrame],
    ) -> dict:
        """Perturb strategy parameters and measure performance stability.

        Returns dict with param_stability_score and param_stability_decay.
        """
        if original_result.sharpe <= 0:
            return {"param_stability_score": 0.0, "param_stability_decay": 1.0}

        perturbed_sharpes = []
        for _ in range(self._n_perturbations):
            perturbed = self._perturb_strategy(strategy)
            result = self._engine.run(perturbed, data)
            if result is not None:
                perturbed_sharpes.append(result.sharpe)
            else:
                perturbed_sharpes.append(0.0)

        if not perturbed_sharpes:
            return {"param_stability_score": 0.0, "param_stability_decay": 1.0}

        avg_perturbed_sharpe = np.mean(perturbed_sharpes)
        std_perturbed_sharpe = np.std(perturbed_sharpes)

        # Decay: how much Sharpe drops on average
        decay = 1.0 - (avg_perturbed_sharpe / original_result.sharpe)
        decay = max(0.0, min(decay, 1.0))

        # Stability score: low variance + low decay = high stability
        cv = std_perturbed_sharpe / abs(avg_perturbed_sharpe) if avg_perturbed_sharpe != 0 else 1.0
        stability_score = max(0, (1.0 - decay) * 50 + (1.0 - min(cv, 1.0)) * 50)

        return {
            "param_stability_score": round(stability_score, 2),
            "param_stability_decay": round(decay, 4),
            "param_stability_avg_sharpe": round(avg_perturbed_sharpe, 4),
            "param_stability_std_sharpe": round(std_perturbed_sharpe, 4),
            "param_stability_min_sharpe": round(min(perturbed_sharpes), 4),
        }

    def _perturb_strategy(self, strategy: StrategyGenome) -> StrategyGenome:
        """Create a copy with slightly perturbed parameters."""
        perturbed_entry = self._perturb_tree(strategy.entry_long)
        return StrategyGenome(
            trading_style=strategy.trading_style,
            entry_long=perturbed_entry,
            exit_long=strategy.exit_long,
            entry_short=strategy.entry_short,
            exit_short=strategy.exit_short,
            timeframes_used=strategy.timeframes_used,
            product_type=strategy.product_type,
            forced_exit_time=strategy.forced_exit_time,
            generation=strategy.generation,
        )

    def _perturb_tree(self, tree: ConditionTree) -> ConditionTree:
        if isinstance(tree, ConditionNode):
            return self._perturb_condition(tree)
        elif isinstance(tree, CompositeCondition):
            children = tuple(self._perturb_tree(c) for c in tree.children)
            return CompositeCondition(logic=tree.logic, children=children)
        return tree

    def _perturb_condition(self, cond: ConditionNode) -> ConditionNode:
        left = self._perturb_node(cond.left)
        right = self._perturb_node(cond.right)
        return ConditionNode(left=left, op=cond.op, right=right)

    def _perturb_node(self, node):
        if isinstance(node, (int, float)):
            delta = node * self._perturbation_pct * np.random.uniform(-1, 1)
            return round(node + delta, 2)
        elif isinstance(node, IndicatorNode):
            new_params = {}
            for key, val in node.params_dict.items():
                delta = val * self._perturbation_pct * np.random.uniform(-1, 1)
                new_val = max(1, round(val + delta))
                new_params[key] = new_val
            return node.with_params(**new_params)
        return node
