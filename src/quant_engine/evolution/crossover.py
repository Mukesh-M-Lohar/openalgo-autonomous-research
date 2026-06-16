"""Strategy crossover operators — breed two parent strategies."""

from __future__ import annotations

import random

from quant_engine.models.strategy import (
    CompositeCondition,
    ConditionNode,
    ConditionTree,
    ExitRule,
    LogicOp,
    StrategyGenome,
)


class Crossover:
    """Combines traits from two parent strategies."""

    def __init__(self, crossover_rate: float = 0.5):
        self._rate = crossover_rate

    def cross(self, parent_a: StrategyGenome, parent_b: StrategyGenome) -> StrategyGenome:
        """Create offspring by combining entry/exit logic from two parents."""
        if random.random() > self._rate:
            return parent_a

        entry = self._cross_entries(parent_a.entry_long, parent_b.entry_long)
        exit_rule = self._cross_exits(parent_a.exit_long, parent_b.exit_long)

        return StrategyGenome(
            trading_style=parent_a.trading_style,
            entry_long=entry,
            exit_long=exit_rule,
            entry_short=parent_a.entry_short,
            exit_short=parent_a.exit_short,
            timeframes_used=parent_a.timeframes_used,
            product_type=parent_a.product_type,
            forced_exit_time=parent_a.forced_exit_time,
            generation=max(parent_a.generation, parent_b.generation) + 1,
            parent_ids=(parent_a.id, parent_b.id),
        )

    def _cross_entries(self, entry_a: ConditionTree, entry_b: ConditionTree) -> ConditionTree:
        """Combine conditions from both parents."""
        conditions_a = _flatten_conditions(entry_a)
        conditions_b = _flatten_conditions(entry_b)

        # Take some conditions from each parent
        n_from_a = max(1, len(conditions_a) // 2)
        n_from_b = max(1, len(conditions_b) // 2)

        selected_a = random.sample(conditions_a, min(n_from_a, len(conditions_a)))
        selected_b = random.sample(conditions_b, min(n_from_b, len(conditions_b)))

        all_conditions = selected_a + selected_b
        # Limit total conditions
        all_conditions = all_conditions[:4]

        if len(all_conditions) == 1:
            return all_conditions[0]

        logic = random.choice([LogicOp.AND, LogicOp.AND, LogicOp.OR])
        return CompositeCondition(logic=logic, children=tuple(all_conditions))

    def _cross_exits(self, exit_a: ExitRule, exit_b: ExitRule) -> ExitRule:
        """Blend exit parameters from both parents."""
        sl = _blend_optional(exit_a.stop_loss_pct, exit_b.stop_loss_pct)
        tp = _blend_optional(exit_a.take_profit_pct, exit_b.take_profit_pct)
        trail = _blend_optional(exit_a.trailing_stop_pct, exit_b.trailing_stop_pct)
        max_hold = exit_a.max_hold_bars if random.random() < 0.5 else exit_b.max_hold_bars

        return ExitRule(
            stop_loss_pct=round(sl, 2) if sl else None,
            take_profit_pct=round(tp, 2) if tp else None,
            trailing_stop_pct=round(trail, 2) if trail else None,
            max_hold_bars=max_hold,
        )


def _flatten_conditions(tree: ConditionTree) -> list[ConditionNode]:
    """Extract all leaf conditions from a tree."""
    if isinstance(tree, ConditionNode):
        return [tree]
    elif isinstance(tree, CompositeCondition):
        result = []
        for child in tree.children:
            result.extend(_flatten_conditions(child))
        return result
    return []


def _blend_optional(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    # Random blend between the two values
    alpha = random.uniform(0.3, 0.7)
    return a * alpha + b * (1 - alpha)
