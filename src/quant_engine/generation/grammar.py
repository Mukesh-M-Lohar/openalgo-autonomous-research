"""Strategy grammar — production rules that define the search space.

Grammar structure:
  STRATEGY -> ENTRY_LONG + EXIT_LONG [+ ENTRY_SHORT + EXIT_SHORT]
  ENTRY -> CONDITION | COMPOSITE(AND/OR, CONDITION, CONDITION, ...)
  CONDITION -> INDICATOR CompareOp INDICATOR | INDICATOR CompareOp CONSTANT
  EXIT -> {stop_loss, take_profit, trailing_stop, exit_signal, max_hold_bars}
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import numpy as np

from quant_engine.generation.indicators import (
    INDICATOR_CATEGORIES,
    INDICATOR_PARAM_RANGES,
)
from quant_engine.models.strategy import (
    CompareOp,
    CompositeCondition,
    ConditionNode,
    ConditionTree,
    ExitRule,
    IndicatorNode,
    IndicatorType,
    LogicOp,
    PriceSource,
    StrategyGenome,
    TimeframeType,
    TradingStyle,
)


@dataclass
class GrammarConfig:
    """Controls what the grammar is allowed to produce."""

    allowed_indicators: list[IndicatorType]
    allowed_timeframes: list[TimeframeType]
    trading_style: TradingStyle
    max_conditions: int = 4
    allow_short: bool = False
    product_type: str = "MIS"
    forced_exit_time: str | None = None
    max_hold_bars: int | None = None
    min_hold_bars: int | None = None


# Operator pairings based on indicator type
TREND_OPS = [CompareOp.GT, CompareOp.LT, CompareOp.CROSS_ABOVE, CompareOp.CROSS_BELOW]
MOMENTUM_OPS = [CompareOp.GT, CompareOp.LT, CompareOp.CROSS_ABOVE, CompareOp.CROSS_BELOW]
CROSSOVER_OPS = [CompareOp.CROSS_ABOVE, CompareOp.CROSS_BELOW]

# Threshold ranges for oscillators
RSI_THRESHOLDS = [20, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80]
STOCH_THRESHOLDS = [20, 30, 40, 50, 60, 70, 80]
CCI_THRESHOLDS = [-200, -150, -100, -50, 0, 50, 100, 150, 200]
ADX_THRESHOLDS = [15, 20, 25, 30, 35, 40]

# Stop/profit ranges
STOP_LOSS_PCTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
TAKE_PROFIT_PCTS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
TRAILING_STOP_PCTS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def sample_indicator_params(indicator_type: IndicatorType) -> tuple[tuple[str, int | float], ...]:
    """Sample random valid parameters for an indicator."""
    ranges = INDICATOR_PARAM_RANGES.get(indicator_type, {})
    params = {}
    for param_name, (low, high, step) in ranges.items():
        values = np.arange(low, high + step, step)
        params[param_name] = float(random.choice(values))
    return tuple(sorted(params.items()))


def make_indicator_node(
    indicator_type: IndicatorType,
    timeframe: TimeframeType,
    source: PriceSource = PriceSource.CLOSE,
    params: tuple[tuple[str, int | float], ...] | None = None,
) -> IndicatorNode:
    """Create an indicator node with sampled or provided parameters."""
    if params is None:
        params = sample_indicator_params(indicator_type)
    return IndicatorNode(
        indicator_type=indicator_type,
        params=params,
        timeframe=timeframe,
        source=source,
    )


def generate_condition(config: GrammarConfig) -> ConditionNode:
    """Generate a random valid condition from the grammar."""
    tf = random.choice(config.allowed_timeframes)
    ind_type = random.choice(config.allowed_indicators)
    left = make_indicator_node(ind_type, tf)

    # Decide: compare to another indicator or a constant
    if _is_oscillator(ind_type):
        threshold = _sample_threshold(ind_type)
        op = random.choice([CompareOp.GT, CompareOp.LT, CompareOp.CROSS_ABOVE, CompareOp.CROSS_BELOW])
        return ConditionNode(left=left, op=op, right=threshold)
    else:
        # Compare two indicators (e.g., EMA cross, price vs band)
        if random.random() < 0.7:
            right_type = _compatible_indicator(ind_type, config.allowed_indicators)
            right = make_indicator_node(right_type, tf)
            op = random.choice(CROSSOVER_OPS if _is_same_category(ind_type, right_type) else TREND_OPS)
        else:
            right = make_indicator_node(ind_type, tf)
            op = random.choice(CROSSOVER_OPS)
        return ConditionNode(left=left, op=op, right=right)


def generate_entry(config: GrammarConfig) -> ConditionTree:
    """Generate entry logic — 1 to max_conditions combined with AND/OR."""
    n_conditions = random.randint(1, min(config.max_conditions, 4))

    if n_conditions == 1:
        return generate_condition(config)

    logic = random.choice([LogicOp.AND, LogicOp.AND, LogicOp.AND, LogicOp.OR])
    children = tuple(generate_condition(config) for _ in range(n_conditions))
    return CompositeCondition(logic=logic, children=children)


def generate_exit(config: GrammarConfig) -> ExitRule:
    """Generate exit rules appropriate for the trading style."""
    sl = random.choice(STOP_LOSS_PCTS)
    tp = random.choice(TAKE_PROFIT_PCTS)

    # Ensure risk:reward >= 1:1
    while tp < sl:
        tp = random.choice(TAKE_PROFIT_PCTS)

    trailing = random.choice(TRAILING_STOP_PCTS) if random.random() < 0.4 else None
    max_hold = config.max_hold_bars

    return ExitRule(
        stop_loss_pct=sl,
        take_profit_pct=tp,
        trailing_stop_pct=trailing,
        max_hold_bars=max_hold,
    )


def generate_strategy(config: GrammarConfig) -> StrategyGenome:
    """Generate a complete random strategy from the grammar."""
    entry_long = generate_entry(config)
    exit_long = generate_exit(config)

    entry_short = None
    exit_short = None
    if config.allow_short:
        entry_short = generate_entry(config)
        exit_short = generate_exit(config)

    timeframes = tuple(sorted(set(_extract_timeframes(entry_long)), key=lambda t: t.value))
    if not timeframes:
        timeframes = (config.allowed_timeframes[0],)

    return StrategyGenome(
        trading_style=config.trading_style,
        entry_long=entry_long,
        exit_long=exit_long,
        entry_short=entry_short,
        exit_short=exit_short,
        timeframes_used=timeframes,
        product_type=config.product_type,
        forced_exit_time=config.forced_exit_time,
    )


# --- Helpers ---

def _is_oscillator(ind_type: IndicatorType) -> bool:
    return ind_type in {
        IndicatorType.RSI, IndicatorType.STOCH_K, IndicatorType.STOCH_D,
        IndicatorType.CCI, IndicatorType.ADX, IndicatorType.ROC,
        IndicatorType.MOMENTUM,
    }


def _sample_threshold(ind_type: IndicatorType) -> float:
    if ind_type == IndicatorType.RSI:
        return float(random.choice(RSI_THRESHOLDS))
    elif ind_type in (IndicatorType.STOCH_K, IndicatorType.STOCH_D):
        return float(random.choice(STOCH_THRESHOLDS))
    elif ind_type == IndicatorType.CCI:
        return float(random.choice(CCI_THRESHOLDS))
    elif ind_type == IndicatorType.ADX:
        return float(random.choice(ADX_THRESHOLDS))
    else:
        return 0.0


def _compatible_indicator(
    ind_type: IndicatorType, allowed: list[IndicatorType]
) -> IndicatorType:
    """Pick a compatible indicator for comparison."""
    category = None
    for cat, members in INDICATOR_CATEGORIES.items():
        if ind_type in members:
            category = cat
            break

    if category:
        same_cat = [i for i in INDICATOR_CATEGORIES[category] if i in allowed and i != ind_type]
        if same_cat:
            return random.choice(same_cat)

    # Fallback: price or same indicator with different params
    if IndicatorType.PRICE in allowed:
        return IndicatorType.PRICE
    return ind_type


def _is_same_category(a: IndicatorType, b: IndicatorType) -> bool:
    for members in INDICATOR_CATEGORIES.values():
        if a in members and b in members:
            return True
    return False


def _extract_timeframes(tree: ConditionTree) -> list[TimeframeType]:
    """Extract all timeframes used in a condition tree."""
    tfs = []
    if isinstance(tree, ConditionNode):
        if isinstance(tree.left, IndicatorNode):
            tfs.append(tree.left.timeframe)
        if isinstance(tree.right, IndicatorNode):
            tfs.append(tree.right.timeframe)
    elif isinstance(tree, CompositeCondition):
        for child in tree.children:
            tfs.extend(_extract_timeframes(child))
    return tfs
