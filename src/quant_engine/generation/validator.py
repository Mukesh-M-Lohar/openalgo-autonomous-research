"""Fast-reject validator — kills bad strategies before expensive backtesting."""

from __future__ import annotations

import logging

from quant_engine.config import FastRejectFilters
from quant_engine.models.results import RejectionRecord
from quant_engine.models.strategy import (
    CompareOp,
    CompositeCondition,
    ConditionNode,
    ConditionTree,
    IndicatorNode,
    IndicatorType,
    StrategyGenome,
)

logger = logging.getLogger(__name__)


class FastRejectValidator:
    """Rejects structurally invalid or obviously bad strategies."""

    def __init__(self, filters: FastRejectFilters):
        self._filters = filters

    def validate(self, strategy: StrategyGenome) -> RejectionRecord | None:
        """Returns a RejectionRecord if strategy should be rejected, None if valid."""
        checks = [
            self._check_missing_exit,
            self._check_complexity,
            self._check_impossible_conditions,
            self._check_redundant_conditions,
            self._check_contradictory_conditions,
            self._check_min_indicators,
        ]
        for check in checks:
            rejection = check(strategy)
            if rejection is not None:
                return rejection
        return None

    def validate_batch(
        self, strategies: list[StrategyGenome]
    ) -> tuple[list[StrategyGenome], list[RejectionRecord]]:
        """Validate a batch, returning (passed, rejections)."""
        passed = []
        rejections = []
        for s in strategies:
            result = self.validate(s)
            if result is None:
                passed.append(s)
            else:
                rejections.append(result)
        return passed, rejections

    def _check_missing_exit(self, s: StrategyGenome) -> RejectionRecord | None:
        if s.exit_long.stop_loss_pct is None and s.exit_long.exit_signal is None:
            if s.exit_long.max_hold_bars is None:
                return RejectionRecord(
                    strategy_id=s.id,
                    stage="fast_reject",
                    rejection_reason="missing_exit_rule",
                    threshold="at_least_one_exit_mechanism",
                    actual_value="none",
                )
        return None

    def _check_complexity(self, s: StrategyGenome) -> RejectionRecord | None:
        count = _count_conditions(s.entry_long)
        if count > self._filters.max_complexity:
            return RejectionRecord(
                strategy_id=s.id,
                stage="fast_reject",
                rejection_reason="excessive_complexity",
                threshold=str(self._filters.max_complexity),
                actual_value=str(count),
            )
        return None

    def _check_impossible_conditions(self, s: StrategyGenome) -> RejectionRecord | None:
        conditions = _extract_conditions(s.entry_long)
        for cond in conditions:
            if isinstance(cond.right, (int, float)):
                if isinstance(cond.left, IndicatorNode):
                    if cond.left.indicator_type == IndicatorType.RSI:
                        if cond.right > 100 or cond.right < 0:
                            return RejectionRecord(
                                strategy_id=s.id,
                                stage="fast_reject",
                                rejection_reason="impossible_condition",
                                threshold="RSI [0,100]",
                                actual_value=str(cond.right),
                            )
                    if cond.left.indicator_type in (IndicatorType.STOCH_K, IndicatorType.STOCH_D):
                        if cond.right > 100 or cond.right < 0:
                            return RejectionRecord(
                                strategy_id=s.id,
                                stage="fast_reject",
                                rejection_reason="impossible_condition",
                                threshold="Stochastic [0,100]",
                                actual_value=str(cond.right),
                            )
        return None

    def _check_redundant_conditions(self, s: StrategyGenome) -> RejectionRecord | None:
        conditions = _extract_conditions(s.entry_long)
        seen = set()
        for cond in conditions:
            key = (
                _node_key(cond.left),
                cond.op.value,
                _node_key(cond.right),
            )
            if key in seen:
                return RejectionRecord(
                    strategy_id=s.id,
                    stage="fast_reject",
                    rejection_reason="redundant_conditions",
                    threshold="unique",
                    actual_value="duplicate_found",
                )
            seen.add(key)
        return None

    def _check_contradictory_conditions(self, s: StrategyGenome) -> RejectionRecord | None:
        if not isinstance(s.entry_long, CompositeCondition):
            return None
        if s.entry_long.logic != "and":
            return None

        conditions = _extract_conditions(s.entry_long)
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i + 1 :]:
                if _are_contradictory(c1, c2):
                    return RejectionRecord(
                        strategy_id=s.id,
                        stage="fast_reject",
                        rejection_reason="contradictory_conditions",
                        threshold="no_contradictions",
                        actual_value="found",
                    )
        return None

    def _check_min_indicators(self, s: StrategyGenome) -> RejectionRecord | None:
        indicators = _extract_indicators(s.entry_long)
        if len(indicators) < self._filters.min_indicators:
            return RejectionRecord(
                strategy_id=s.id,
                stage="fast_reject",
                rejection_reason="too_few_indicators",
                threshold=str(self._filters.min_indicators),
                actual_value=str(len(indicators)),
            )
        return None


def _count_conditions(tree: ConditionTree) -> int:
    if isinstance(tree, ConditionNode):
        return 1
    elif isinstance(tree, CompositeCondition):
        return sum(_count_conditions(c) for c in tree.children)
    return 0


def _extract_conditions(tree: ConditionTree) -> list[ConditionNode]:
    if isinstance(tree, ConditionNode):
        return [tree]
    elif isinstance(tree, CompositeCondition):
        result = []
        for child in tree.children:
            result.extend(_extract_conditions(child))
        return result
    return []


def _extract_indicators(tree: ConditionTree) -> set[IndicatorType]:
    indicators = set()
    for cond in _extract_conditions(tree):
        if isinstance(cond.left, IndicatorNode):
            indicators.add(cond.left.indicator_type)
        if isinstance(cond.right, IndicatorNode):
            indicators.add(cond.right.indicator_type)
    return indicators


def _node_key(node) -> str:
    if isinstance(node, IndicatorNode):
        return f"{node.indicator_type.value}_{node.params}_{node.timeframe.value}"
    return str(node)


def _are_contradictory(c1: ConditionNode, c2: ConditionNode) -> bool:
    """Check if two conditions on the same indicator are contradictory."""
    if not (isinstance(c1.left, IndicatorNode) and isinstance(c2.left, IndicatorNode)):
        return False
    if c1.left != c2.left:
        return False
    # Same indicator, check for GT x AND LT y where y < x
    if isinstance(c1.right, (int, float)) and isinstance(c2.right, (int, float)):
        if c1.op == CompareOp.GT and c2.op == CompareOp.LT and c2.right <= c1.right:
            return True
        if c1.op == CompareOp.LT and c2.op == CompareOp.GT and c1.right <= c2.right:
            return True
    return False
