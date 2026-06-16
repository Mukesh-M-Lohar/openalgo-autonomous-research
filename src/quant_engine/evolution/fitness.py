"""Multi-objective fitness functions for strategy evaluation."""

from __future__ import annotations

from quant_engine.config import ObjectiveConfig
from quant_engine.models.results import BacktestResult, ValidationResult


def weighted_fitness(
    backtest: BacktestResult,
    validation: ValidationResult | None,
    objectives: list[ObjectiveConfig],
) -> float:
    """Compute weighted composite fitness score."""
    total_weight = sum(o.weight for o in objectives)
    if total_weight == 0:
        return 0.0

    score = 0.0
    for obj in objectives:
        raw_value = _get_metric(backtest, validation, obj.metric)
        normalized = _normalize_metric(raw_value, obj.metric, obj.direction)
        score += normalized * (obj.weight / total_weight)

    return round(score, 6)


def _get_metric(
    backtest: BacktestResult, validation: ValidationResult | None, metric: str
) -> float:
    """Extract a metric value from backtest or validation results."""
    if hasattr(backtest, metric):
        return getattr(backtest, metric)
    if validation and hasattr(validation, metric):
        return getattr(validation, metric)
    return 0.0


def _normalize_metric(value: float, metric: str, direction: str) -> float:
    """Normalize a metric to [0, 1] range based on typical bounds."""
    bounds = METRIC_BOUNDS.get(metric, (-10, 10))
    low, high = bounds

    if direction == "minimize":
        value = -value
        low, high = -high, -low

    if high == low:
        return 0.5

    normalized = (value - low) / (high - low)
    return max(0.0, min(1.0, normalized))


METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "sharpe": (-1.0, 4.0),
    "sortino": (-1.0, 6.0),
    "calmar": (0.0, 5.0),
    "cagr": (-20.0, 100.0),
    "profit_factor": (0.0, 5.0),
    "max_drawdown_pct": (0.0, 50.0),
    "win_rate": (0.0, 1.0),
    "recovery_factor": (0.0, 10.0),
    "expectancy": (-5.0, 5.0),
    "total_trades": (0, 500),
    "walk_forward_score": (0.0, 100.0),
    "monte_carlo_score": (0.0, 100.0),
    "param_stability_score": (0.0, 100.0),
    "stress_test_score": (0.0, 100.0),
    "robustness_score": (0.0, 100.0),
}
