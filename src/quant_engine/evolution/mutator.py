"""Strategy mutation operators for evolutionary optimization."""

from __future__ import annotations

import random

from quant_engine.generation.grammar import (
    GrammarConfig,
    generate_condition,
    sample_indicator_params,
)
from quant_engine.generation.indicators import INDICATOR_CATEGORIES
from quant_engine.models.strategy import (
    CompareOp,
    CompositeCondition,
    ConditionNode,
    ConditionTree,
    ExitRule,
    IndicatorNode,
    LogicOp,
    StrategyGenome,
)


class Mutator:
    """Applies random mutations to strategy genomes."""

    def __init__(self, mutation_rate: float = 0.3, grammar_config: GrammarConfig | None = None):
        self._rate = mutation_rate
        self._grammar = grammar_config

    def mutate(self, strategy: StrategyGenome) -> StrategyGenome:
        """Apply one or more random mutations to a strategy."""
        mutations = [
            self._mutate_params,
            self._mutate_operator,
            self._mutate_indicator_swap,
            self._mutate_condition_add_remove,
            self._mutate_exit,
            self._mutate_threshold,
        ]

        entry = strategy.entry_long
        exit_rule = strategy.exit_long

        for mutation_fn in mutations:
            if random.random() < self._rate:
                entry, exit_rule = mutation_fn(entry, exit_rule)

        return StrategyGenome(
            trading_style=strategy.trading_style,
            entry_long=entry,
            exit_long=exit_rule,
            entry_short=strategy.entry_short,
            exit_short=strategy.exit_short,
            timeframes_used=strategy.timeframes_used,
            product_type=strategy.product_type,
            forced_exit_time=strategy.forced_exit_time,
            generation=strategy.generation + 1,
            parent_ids=(strategy.id,),
        )

    def _mutate_params(
        self, entry: ConditionTree, exit_rule: ExitRule
    ) -> tuple[ConditionTree, ExitRule]:
        """Perturb numeric parameters by 10-30%."""
        return _perturb_tree(entry, 0.15), exit_rule

    def _mutate_operator(
        self, entry: ConditionTree, exit_rule: ExitRule
    ) -> tuple[ConditionTree, ExitRule]:
        """Swap comparison operator."""
        return _swap_operator(entry), exit_rule

    def _mutate_indicator_swap(
        self, entry: ConditionTree, exit_rule: ExitRule
    ) -> tuple[ConditionTree, ExitRule]:
        """Replace one indicator with another from the same category."""
        return _swap_indicator(entry), exit_rule

    def _mutate_condition_add_remove(
        self, entry: ConditionTree, exit_rule: ExitRule
    ) -> tuple[ConditionTree, ExitRule]:
        """Add or remove a condition."""
        if isinstance(entry, CompositeCondition) and len(entry.children) > 2:
            # Remove a random condition
            idx = random.randint(0, len(entry.children) - 1)
            children = list(entry.children)
            children.pop(idx)
            if len(children) == 1:
                return children[0], exit_rule
            return CompositeCondition(logic=entry.logic, children=tuple(children)), exit_rule
        elif self._grammar:
            # Add a condition
            new_cond = generate_condition(self._grammar)
            if isinstance(entry, CompositeCondition):
                children = entry.children + (new_cond,)
                return CompositeCondition(logic=entry.logic, children=children), exit_rule
            else:
                return CompositeCondition(logic=LogicOp.AND, children=(entry, new_cond)), exit_rule
        return entry, exit_rule

    def _mutate_exit(
        self, entry: ConditionTree, exit_rule: ExitRule
    ) -> tuple[ConditionTree, ExitRule]:
        """Modify stop-loss, take-profit, or trailing stop."""
        sl = exit_rule.stop_loss_pct
        tp = exit_rule.take_profit_pct
        trail = exit_rule.trailing_stop_pct

        choice = random.choice(["sl", "tp", "trail"])
        if choice == "sl" and sl is not None:
            sl = max(0.3, sl * random.uniform(0.8, 1.2))
        elif choice == "tp" and tp is not None:
            tp = max(0.5, tp * random.uniform(0.8, 1.2))
        elif choice == "trail":
            if trail is not None:
                trail = max(0.3, trail * random.uniform(0.8, 1.2))
            else:
                trail = random.uniform(1.0, 3.0)

        return entry, ExitRule(
            stop_loss_pct=round(sl, 2) if sl else None,
            take_profit_pct=round(tp, 2) if tp else None,
            trailing_stop_pct=round(trail, 2) if trail else None,
            exit_signal=exit_rule.exit_signal,
            max_hold_bars=exit_rule.max_hold_bars,
        )

    def _mutate_threshold(
        self, entry: ConditionTree, exit_rule: ExitRule
    ) -> tuple[ConditionTree, ExitRule]:
        """Adjust constant thresholds."""
        return _perturb_constants(entry, 0.1), exit_rule


def _perturb_tree(tree: ConditionTree, pct: float) -> ConditionTree:
    if isinstance(tree, ConditionNode):
        left = _perturb_node(tree.left, pct)
        right = _perturb_node(tree.right, pct)
        return ConditionNode(left=left, op=tree.op, right=right)
    elif isinstance(tree, CompositeCondition):
        children = tuple(_perturb_tree(c, pct) for c in tree.children)
        return CompositeCondition(logic=tree.logic, children=children)
    return tree


def _perturb_node(node, pct: float):
    if isinstance(node, (int, float)):
        delta = node * pct * random.uniform(-1, 1)
        return round(node + delta, 2)
    elif isinstance(node, IndicatorNode):
        new_params = {}
        for key, val in node.params_dict.items():
            delta = val * pct * random.uniform(-1, 1)
            new_params[key] = max(1, round(val + delta))
        return node.with_params(**new_params)
    return node


def _perturb_constants(tree: ConditionTree, pct: float) -> ConditionTree:
    if isinstance(tree, ConditionNode):
        right = tree.right
        if isinstance(right, (int, float)):
            delta = right * pct * random.uniform(-1, 1)
            right = round(right + delta, 2)
        return ConditionNode(left=tree.left, op=tree.op, right=right)
    elif isinstance(tree, CompositeCondition):
        children = tuple(_perturb_constants(c, pct) for c in tree.children)
        return CompositeCondition(logic=tree.logic, children=children)
    return tree


def _swap_operator(tree: ConditionTree) -> ConditionTree:
    if isinstance(tree, ConditionNode):
        ops = [CompareOp.GT, CompareOp.LT, CompareOp.CROSS_ABOVE, CompareOp.CROSS_BELOW]
        new_op = random.choice([o for o in ops if o != tree.op])
        return ConditionNode(left=tree.left, op=new_op, right=tree.right)
    elif isinstance(tree, CompositeCondition):
        idx = random.randint(0, len(tree.children) - 1)
        children = list(tree.children)
        children[idx] = _swap_operator(children[idx])
        return CompositeCondition(logic=tree.logic, children=tuple(children))
    return tree


def _swap_indicator(tree: ConditionTree) -> ConditionTree:
    if isinstance(tree, ConditionNode):
        if isinstance(tree.left, IndicatorNode):
            new_left = _find_replacement(tree.left)
            return ConditionNode(left=new_left, op=tree.op, right=tree.right)
        return tree
    elif isinstance(tree, CompositeCondition):
        idx = random.randint(0, len(tree.children) - 1)
        children = list(tree.children)
        children[idx] = _swap_indicator(children[idx])
        return CompositeCondition(logic=tree.logic, children=tuple(children))
    return tree


def _find_replacement(node: IndicatorNode) -> IndicatorNode:
    """Find a same-category indicator replacement."""
    for cat, members in INDICATOR_CATEGORIES.items():
        if node.indicator_type in members:
            alternatives = [m for m in members if m != node.indicator_type]
            if alternatives:
                new_type = random.choice(alternatives)
                new_params = sample_indicator_params(new_type)
                return IndicatorNode(
                    indicator_type=new_type,
                    params=new_params,
                    timeframe=node.timeframe,
                    source=node.source,
                )
    return node
