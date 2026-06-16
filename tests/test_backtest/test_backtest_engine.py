"""Tests for the backtest engine and metrics computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from quant_engine.backtest.engine import BacktestEngine
from quant_engine.backtest.metrics import compute_metrics
from quant_engine.config import CostModelConfig
from quant_engine.generation.grammar import GrammarConfig, generate_strategy
from quant_engine.generation.indicators import INDICATOR_CATEGORIES
from quant_engine.models.results import BacktestResult
from quant_engine.models.strategy import (
    CompareOp,
    ConditionNode,
    ExitRule,
    IndicatorNode,
    IndicatorType,
    PriceSource,
    StrategyGenome,
    TimeframeType,
    TradingStyle,
)


@pytest.fixture
def trending_data():
    """Strongly trending data — strategies should find trades."""
    np.random.seed(123)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="15min")
    # Create a clear uptrend
    trend = np.linspace(100, 150, n) + np.random.randn(n) * 2
    return pd.DataFrame(
        {
            "open": trend + np.random.randn(n) * 0.5,
            "high": trend + abs(np.random.randn(n) * 1.0),
            "low": trend - abs(np.random.randn(n) * 1.0),
            "close": trend,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def mean_reverting_data():
    """Mean-reverting data."""
    np.random.seed(456)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="15min")
    close = 100 + np.random.randn(n) * 3  # oscillates around 100
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.2,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def simple_strategy():
    """A simple EMA crossover strategy for testing."""
    return StrategyGenome(
        trading_style=TradingStyle.SWING,
        entry_long=ConditionNode(
            left=IndicatorNode(
                IndicatorType.EMA, (("period", 10),), TimeframeType.M15, PriceSource.CLOSE
            ),
            op=CompareOp.CROSS_ABOVE,
            right=IndicatorNode(
                IndicatorType.EMA, (("period", 30),), TimeframeType.M15, PriceSource.CLOSE
            ),
        ),
        exit_long=ExitRule(stop_loss_pct=3.0, take_profit_pct=5.0, max_hold_bars=50),
        timeframes_used=(TimeframeType.M15,),
        product_type="CNC",
    )


class TestBacktestEngine:
    def test_engine_runs(self, simple_strategy, trending_data):
        engine = BacktestEngine()
        result = engine.run(simple_strategy, {"15m": trending_data})
        # May or may not produce trades depending on data
        # Just verify it doesn't crash
        assert result is None or isinstance(result, BacktestResult)

    def test_engine_with_cost_model(self, simple_strategy, trending_data):
        cost = CostModelConfig(commission_pct=0.05, slippage_pct=0.03)
        engine = BacktestEngine(cost_model=cost)
        result = engine.run(simple_strategy, {"15m": trending_data})
        assert result is None or isinstance(result, BacktestResult)

    def test_empty_data_returns_none(self, simple_strategy):
        engine = BacktestEngine()
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = engine.run(simple_strategy, {"15m": empty_df})
        assert result is None

    def test_insufficient_data_returns_none(self, simple_strategy):
        engine = BacktestEngine()
        short_df = pd.DataFrame(
            {"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]},
            index=pd.date_range("2023-01-01", periods=1, freq="15min"),
        )
        result = engine.run(simple_strategy, {"15m": short_df})
        assert result is None

    def test_missing_timeframe_returns_none(self, simple_strategy, trending_data):
        engine = BacktestEngine()
        result = engine.run(simple_strategy, {"1h": trending_data})  # strategy uses 15m
        assert result is None

    def test_batch_run(self, trending_data):
        allowed = INDICATOR_CATEGORIES["trend"] + INDICATOR_CATEGORIES["momentum"]
        cfg = GrammarConfig(
            allowed_indicators=allowed,
            allowed_timeframes=[TimeframeType.M15],
            trading_style=TradingStyle.INTRADAY,
            max_conditions=2,
            product_type="MIS",
            max_hold_bars=50,
        )
        strategies = [generate_strategy(cfg) for _ in range(20)]
        engine = BacktestEngine()
        results = engine.run_batch(strategies, {"15m": trending_data})
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, BacktestResult)


class TestMetrics:
    def test_empty_trades(self):
        result = compute_metrics("test", [], pd.DataFrame({"equity": [100000]}), 100000, 100)
        assert result.total_trades == 0
        assert result.sharpe == 0.0

    def test_all_winning_trades(self):
        trades = [{"pnl_pct": 2.0, "bars_held": 5} for _ in range(10)]
        equity = pd.DataFrame({"equity": np.linspace(100000, 120000, 100)})
        result = compute_metrics("test", trades, equity, 100000, 100)
        assert result.win_rate == 1.0
        assert result.profit_factor > 0
        assert result.sharpe > 0

    def test_all_losing_trades(self):
        trades = [{"pnl_pct": -1.5, "bars_held": 3} for _ in range(10)]
        equity = pd.DataFrame({"equity": np.linspace(100000, 85000, 100)})
        result = compute_metrics("test", trades, equity, 100000, 100)
        assert result.win_rate == 0.0
        assert result.sharpe < 0

    def test_mixed_trades(self):
        trades = [
            {"pnl_pct": 3.0, "bars_held": 5},
            {"pnl_pct": -1.0, "bars_held": 3},
            {"pnl_pct": 2.5, "bars_held": 4},
            {"pnl_pct": -0.5, "bars_held": 2},
            {"pnl_pct": 4.0, "bars_held": 6},
        ]
        equity = pd.DataFrame({"equity": np.linspace(100000, 108000, 100)})
        result = compute_metrics("test", trades, equity, 100000, 100)
        assert result.total_trades == 5
        assert result.winning_trades == 3
        assert result.losing_trades == 2
        assert 0 < result.win_rate < 1
        assert result.profit_factor > 1
        assert result.max_consecutive_wins >= 1
        assert result.max_consecutive_losses >= 1

    def test_consecutive_wins(self):
        trades = [
            {"pnl_pct": 1.0, "bars_held": 2},
            {"pnl_pct": 1.0, "bars_held": 2},
            {"pnl_pct": 1.0, "bars_held": 2},
            {"pnl_pct": -0.5, "bars_held": 2},
            {"pnl_pct": 1.0, "bars_held": 2},
        ]
        equity = pd.DataFrame({"equity": [100000] * 100})
        result = compute_metrics("test", trades, equity, 100000, 100)
        assert result.max_consecutive_wins == 3

    def test_drawdown_calculation(self):
        # Equity goes up then drops
        equity_values = [100000] * 50 + [90000] * 50
        equity = pd.DataFrame({"equity": equity_values})
        trades = [{"pnl_pct": -10.0, "bars_held": 50}]
        result = compute_metrics("test", trades, equity, 100000, 100)
        assert result.max_drawdown_pct > 0
