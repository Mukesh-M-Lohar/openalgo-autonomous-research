"""Tests for validation engines — Monte Carlo, Pareto, ranking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


from quant_engine.config import ObjectiveConfig, RankingConfig
from quant_engine.evolution.fitness import weighted_fitness
from quant_engine.models.results import BacktestResult, ValidationResult
from quant_engine.ranking.pareto import compute_pareto_fronts
from quant_engine.ranking.scorer import RankingEngine
from quant_engine.validation.monte_carlo import MonteCarloValidator


class TestMonteCarlo:
    def test_profitable_trades(self):
        mc = MonteCarloValidator(n_simulations=200)
        bt = BacktestResult(strategy_id="test", sharpe=2.0, total_trades=50)
        trades = [{"pnl_pct": 1.5} for _ in range(50)]
        result = mc.validate(bt, trades)
        assert result["monte_carlo_score"] > 50
        assert result["monte_carlo_profitable_pct"] > 90

    def test_losing_trades(self):
        mc = MonteCarloValidator(n_simulations=200)
        bt = BacktestResult(strategy_id="test", sharpe=-1.0, total_trades=50)
        trades = [{"pnl_pct": -1.0} for _ in range(50)]
        result = mc.validate(bt, trades)
        assert result["monte_carlo_profitable_pct"] < 10

    def test_few_trades_returns_zero(self):
        mc = MonteCarloValidator()
        bt = BacktestResult(strategy_id="test")
        trades = [{"pnl_pct": 1.0}] * 5  # too few
        result = mc.validate(bt, trades)
        assert result["monte_carlo_score"] == 0.0

    def test_empty_trades(self):
        mc = MonteCarloValidator()
        bt = BacktestResult(strategy_id="test")
        result = mc.validate(bt, [])
        assert result["monte_carlo_score"] == 0.0


class TestPareto:
    def test_single_point(self):
        fronts = compute_pareto_fronts([[1.0, 2.0]])
        assert fronts == [[0]]

    def test_dominated_point(self):
        points = [
            [3.0, 3.0],  # dominates point 1
            [2.0, 2.0],  # dominated
        ]
        fronts = compute_pareto_fronts(points)
        assert 0 in fronts[0]
        assert 1 in fronts[1]

    def test_non_dominated_set(self):
        points = [
            [3.0, 1.0],  # Pareto optimal
            [1.0, 3.0],  # Pareto optimal
            [2.0, 2.0],  # dominated by neither (non-dominated)
        ]
        fronts = compute_pareto_fronts(points)
        # All three are non-dominated (no one dominates all others)
        assert len(fronts[0]) == 3

    def test_clear_dominance(self):
        points = [
            [5.0, 5.0, 5.0],  # dominates all
            [3.0, 3.0, 3.0],  # dominated by 0
            [1.0, 1.0, 1.0],  # dominated by 0 and 1
        ]
        fronts = compute_pareto_fronts(points)
        assert fronts[0] == [0]
        assert fronts[1] == [1]
        assert fronts[2] == [2]

    def test_empty_input(self):
        fronts = compute_pareto_fronts([])
        assert fronts == []


class TestRankingEngine:
    def test_ranking_produces_ordered_results(self):
        config = RankingConfig(mode="weighted")
        engine = RankingEngine(config)

        data = [
            {
                "strategy_id": "a",
                "backtest": BacktestResult(strategy_id="a", sharpe=2.0, cagr=30),
            },
            {
                "strategy_id": "b",
                "backtest": BacktestResult(strategy_id="b", sharpe=1.0, cagr=15),
            },
            {
                "strategy_id": "c",
                "backtest": BacktestResult(strategy_id="c", sharpe=3.0, cagr=45),
            },
        ]
        ranked = engine.rank(data)
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[2].rank == 3
        # Best sharpe should rank highest with default weights
        assert ranked[0].backtest.sharpe >= ranked[1].backtest.sharpe

    def test_get_winners_categories(self):
        config = RankingConfig(export_per_category=2)
        engine = RankingEngine(config)

        data = [
            {
                "strategy_id": f"s{i}",
                "backtest": BacktestResult(
                    strategy_id=f"s{i}",
                    sharpe=float(i),
                    cagr=float(i * 10),
                    profit_factor=float(i * 0.5),
                    max_drawdown_pct=float(20 - i),
                ),
                "validation": ValidationResult(strategy_id=f"s{i}", robustness_score=float(i * 10)),
            }
            for i in range(1, 11)
        ]
        ranked = engine.rank(data)
        winners = engine.get_winners(ranked)

        assert "best_overall" in winners
        assert "best_sharpe" in winners
        assert "best_cagr" in winners
        assert "lowest_drawdown" in winners
        assert len(winners["best_sharpe"]) <= 2

    def test_compose_portfolios(self):
        config = RankingConfig()
        engine = RankingEngine(config)

        data = [
            {
                "strategy_id": f"s{i}",
                "backtest": BacktestResult(
                    strategy_id=f"s{i}",
                    sharpe=float(i),
                    cagr=float(i * 10),
                    max_drawdown_pct=float(50 - i * 5),
                    win_rate=0.5 + i * 0.03,
                ),
            }
            for i in range(1, 11)
        ]
        ranked = engine.rank(data)
        portfolios = engine.compose_portfolios(ranked)

        assert "conservative" in portfolios
        assert "balanced" in portfolios
        assert "aggressive" in portfolios

    def test_empty_input(self):
        config = RankingConfig()
        engine = RankingEngine(config)
        ranked = engine.rank([])
        assert ranked == []


class TestFitness:
    def test_weighted_fitness_basic(self):
        bt = BacktestResult(strategy_id="test", sharpe=2.0, cagr=30.0, profit_factor=2.5)
        objectives = [
            ObjectiveConfig(metric="sharpe", weight=0.5),
            ObjectiveConfig(metric="cagr", weight=0.5),
        ]
        score = weighted_fitness(bt, None, objectives)
        assert 0 <= score <= 1

    def test_higher_metrics_higher_fitness(self):
        bt_good = BacktestResult(strategy_id="good", sharpe=3.0, cagr=50.0)
        bt_bad = BacktestResult(strategy_id="bad", sharpe=0.5, cagr=5.0)
        objectives = [
            ObjectiveConfig(metric="sharpe", weight=0.5),
            ObjectiveConfig(metric="cagr", weight=0.5),
        ]
        score_good = weighted_fitness(bt_good, None, objectives)
        score_bad = weighted_fitness(bt_bad, None, objectives)
        assert score_good > score_bad

    def test_minimize_direction(self):
        bt_low_dd = BacktestResult(strategy_id="low", max_drawdown_pct=5.0, sharpe=1.5)
        bt_high_dd = BacktestResult(strategy_id="high", max_drawdown_pct=40.0, sharpe=1.5)
        objectives = [
            ObjectiveConfig(metric="max_drawdown_pct", weight=1.0, direction="minimize"),
        ]
        score_low = weighted_fitness(bt_low_dd, None, objectives)
        score_high = weighted_fitness(bt_high_dd, None, objectives)
        assert score_low > score_high
