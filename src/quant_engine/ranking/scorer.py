"""Ranking engine — scores, ranks, and categorizes surviving strategies."""

from __future__ import annotations

import logging

from quant_engine.config import RankingConfig
from quant_engine.evolution.fitness import weighted_fitness
from quant_engine.models.results import BacktestResult, RankedStrategy, ValidationResult
from quant_engine.ranking.pareto import compute_pareto_fronts

logger = logging.getLogger(__name__)


class RankingEngine:
    """Ranks strategies using multi-objective scoring."""

    def __init__(self, config: RankingConfig):
        self._config = config

    def rank(
        self,
        strategies_data: list[dict],
    ) -> list[RankedStrategy]:
        """Rank strategies and assign categories.

        Args:
            strategies_data: List of dicts with keys:
                - strategy_id
                - backtest: BacktestResult
                - validation: ValidationResult
        """
        if not strategies_data:
            return []

        ranked = []
        for item in strategies_data:
            bt = item["backtest"]
            val = item.get("validation")

            if self._config.mode == "weighted":
                score = weighted_fitness(bt, val, self._config.objectives)
            elif self._config.mode == "robustness_first":
                score = self._robustness_first_score(bt, val)
            else:
                score = weighted_fitness(bt, val, self._config.objectives)

            ranked.append(RankedStrategy(
                strategy_id=item["strategy_id"],
                backtest=bt,
                validation=val or ValidationResult(strategy_id=item["strategy_id"]),
                composite_score=score,
            ))

        # Sort by composite score
        ranked.sort(key=lambda r: r.composite_score, reverse=True)

        # Assign ranks
        for i, r in enumerate(ranked):
            r.rank = i + 1

        # Pareto front assignment
        if self._config.mode == "pareto":
            self._assign_pareto_fronts(ranked)

        # Assign categories
        self._assign_categories(ranked)

        return ranked

    def get_winners(self, ranked: list[RankedStrategy]) -> dict[str, list[RankedStrategy]]:
        """Extract winners per category."""
        winners = {
            "best_overall": ranked[: self._config.export_top_n],
            "best_sharpe": sorted(ranked, key=lambda r: r.backtest.sharpe, reverse=True)[
                : self._config.export_per_category
            ],
            "best_cagr": sorted(ranked, key=lambda r: r.backtest.cagr, reverse=True)[
                : self._config.export_per_category
            ],
            "best_profit_factor": sorted(
                ranked, key=lambda r: r.backtest.profit_factor, reverse=True
            )[: self._config.export_per_category],
            "lowest_drawdown": sorted(ranked, key=lambda r: r.backtest.max_drawdown_pct)[
                : self._config.export_per_category
            ],
            "best_robust": sorted(
                ranked, key=lambda r: r.validation.robustness_score, reverse=True
            )[: self._config.export_per_category],
        }
        return winners

    def compose_portfolios(
        self, ranked: list[RankedStrategy]
    ) -> dict[str, list[RankedStrategy]]:
        """Compose conservative, balanced, and aggressive portfolios."""
        if len(ranked) < 3:
            return {"balanced": ranked}

        # Conservative: low drawdown, high consistency
        conservative = sorted(
            ranked,
            key=lambda r: (r.backtest.max_drawdown_pct, -r.backtest.win_rate),
        )[:5]

        # Balanced: best composite score
        balanced = ranked[:5]

        # Aggressive: highest returns
        aggressive = sorted(ranked, key=lambda r: r.backtest.cagr, reverse=True)[:5]

        return {
            "conservative": conservative,
            "balanced": balanced,
            "aggressive": aggressive,
        }

    def _robustness_first_score(
        self, bt: BacktestResult, val: ValidationResult | None
    ) -> float:
        """Score that prioritizes robustness over raw returns."""
        if val is None:
            return weighted_fitness(bt, val, self._config.objectives) * 0.5

        robustness = (
            val.walk_forward_score * 0.3
            + val.monte_carlo_score * 0.25
            + val.param_stability_score * 0.25
            + val.stress_test_score * 0.2
        ) / 100

        performance = weighted_fitness(bt, val, self._config.objectives)
        return robustness * 0.6 + performance * 0.4

    def _assign_pareto_fronts(self, ranked: list[RankedStrategy]) -> None:
        """Assign Pareto front indices to strategies."""
        objectives_data = []
        for r in ranked:
            point = []
            for obj in self._config.objectives:
                val = getattr(r.backtest, obj.metric, 0.0)
                if obj.direction == "minimize":
                    val = -val
                point.append(val)
            objectives_data.append(point)

        fronts = compute_pareto_fronts(objectives_data)
        for i, front_idx in enumerate(fronts):
            for idx in front_idx:
                if idx < len(ranked):
                    ranked[idx].pareto_front = i + 1

    def _assign_categories(self, ranked: list[RankedStrategy]) -> None:
        """Assign descriptive categories based on strategy characteristics."""
        for r in ranked:
            if r.backtest.avg_hold_bars < 10:
                r.category = "scalper"
            elif r.backtest.avg_hold_bars < 50:
                r.category = "swing"
            elif r.backtest.max_drawdown_pct < 10:
                r.category = "conservative"
            elif r.backtest.cagr > 50:
                r.category = "aggressive"
            else:
                r.category = "balanced"
