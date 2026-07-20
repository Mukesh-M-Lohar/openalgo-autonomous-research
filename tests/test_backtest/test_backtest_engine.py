"""Tests for the backtest engine and metrics computation.

Tests cover the Ernie Chan improvements:
- Look-ahead bias fix (entry at next-bar open)
- Bar-by-bar equity curve (mark-to-market)
- Sharpe from equity returns (not per-trade PnL)
- Drawdown duration tracking
- Kelly criterion
- Deflated Sharpe ratio
- Information ratio
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from quant_engine.backtest.engine import BacktestEngine
from quant_engine.backtest.metrics import (
    _compute_deflated_sharpe,
    _compute_drawdown_durations,
    _compute_information_ratio,
    _compute_kelly,
    compute_metrics,
)
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

    def test_engine_returns_trades(self, simple_strategy, trending_data):
        engine = BacktestEngine()
        result = engine.run(simple_strategy, {"15m": trending_data})
        if result is not None:
            assert isinstance(result.trades, list)
            assert len(result.trades) > 0
            assert "entry_time" in result.trades[0]
            assert "pnl_pct" in result.trades[0]

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


class TestLookAheadBiasFix:
    """Verify entry happens at next-bar open, not same-bar close (Chan Ch.1 p.22)."""

    def test_entry_at_next_bar_open(self, simple_strategy, trending_data):
        """Entry price should be an open price, not a close price."""
        engine = BacktestEngine(cost_model=CostModelConfig(commission_pct=0, slippage_pct=0))
        result = engine.run(simple_strategy, {"15m": trending_data})
        if result is not None and result.trades:
            # Each entry price should match an open value from the data
            open_values = set(trending_data["open"].values)
            for trade in result.trades:
                assert trade["entry_price"] in open_values, (
                    f"Entry price {trade['entry_price']} not found in open prices — "
                    "possible look-ahead bias"
                )

    def test_signal_bar_different_from_entry_bar(self):
        """Construct data where close[i] != open[i+1] and verify correct entry."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        # EMA(10) crosses above EMA(30): need a trend that starts low then rises
        prices = np.concatenate(
            [
                np.linspace(100, 90, 40),  # downtrend
                np.linspace(90, 120, 60),  # strong uptrend to force crossover
            ]
        )
        # Set open different from previous close to test look-ahead
        opens = prices + 0.5  # open always 0.5 above close
        df = pd.DataFrame(
            {
                "open": opens,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        strategy = StrategyGenome(
            trading_style=TradingStyle.SWING,
            entry_long=ConditionNode(
                left=IndicatorNode(
                    IndicatorType.EMA, (("period", 10),), TimeframeType.D1, PriceSource.CLOSE
                ),
                op=CompareOp.CROSS_ABOVE,
                right=IndicatorNode(
                    IndicatorType.EMA, (("period", 30),), TimeframeType.D1, PriceSource.CLOSE
                ),
            ),
            exit_long=ExitRule(max_hold_bars=20),
            timeframes_used=(TimeframeType.D1,),
            product_type="CNC",
        )

        engine = BacktestEngine(cost_model=CostModelConfig(commission_pct=0, slippage_pct=0))
        result = engine.run(strategy, {"1d": df})
        if result is not None and result.trades:
            for trade in result.trades:
                # Entry price should be an open price (not close)
                assert trade["entry_price"] in opens, (
                    f"Entry {trade['entry_price']} should be an open price"
                )


class TestBarByBarEquityCurve:
    """Verify equity curve reflects intra-trade mark-to-market (Chan Ch.1 pp.34-40)."""

    def test_equity_not_flat_during_trades(self, simple_strategy, trending_data):
        """In trending data with trades, equity should vary bar-by-bar."""
        engine = BacktestEngine()
        # Access the internals to check the equity array
        df = trending_data
        entries_raw = engine._evaluate_entry(simple_strategy.entry_long, df, {"15m": df})
        exits_raw = engine._evaluate_exit(simple_strategy.exit_long, df, {"15m": df})

        if entries_raw is not None and exits_raw is not None:
            entries = entries_raw.fillna(False).astype(bool)
            exits = exits_raw.fillna(False).astype(bool)

            if entries.sum() > 0:
                trades, bar_equity = engine._simulate_trades(
                    df, entries, exits, simple_strategy.exit_long, None
                )
                if trades:
                    # Equity should NOT be all the same value (flat)
                    unique_values = len(set(bar_equity))
                    assert unique_values > 2, (
                        f"Equity curve has only {unique_values} unique values — "
                        "should reflect intra-trade mark-to-market"
                    )

    def test_equity_length_matches_data(self, simple_strategy, trending_data):
        """Bar-by-bar equity array should have same length as data."""
        engine = BacktestEngine()
        df = trending_data
        entries_raw = engine._evaluate_entry(simple_strategy.entry_long, df, {"15m": df})
        exits_raw = engine._evaluate_exit(simple_strategy.exit_long, df, {"15m": df})

        if entries_raw is not None and exits_raw is not None:
            entries = entries_raw.fillna(False).astype(bool)
            exits = exits_raw.fillna(False).astype(bool)

            trades, bar_equity = engine._simulate_trades(
                df, entries, exits, simple_strategy.exit_long, None
            )
            assert len(bar_equity) == len(df)


class TestSharpeFromEquityCurve:
    """Verify Sharpe is computed from equity returns, not per-trade PnL."""

    def test_sharpe_positive_for_upward_equity(self):
        """Monotonically rising equity should give positive Sharpe."""
        equity = np.linspace(100000, 120000, 252)
        equity_df = pd.DataFrame({"equity": equity})
        trades = [{"pnl_pct": 2.0, "bars_held": 5} for _ in range(10)]

        result = compute_metrics("test", trades, equity_df, 100000, 252)
        assert result.sharpe > 0, f"Expected positive Sharpe, got {result.sharpe}"

    def test_sharpe_negative_for_downward_equity(self):
        """Monotonically falling equity should give negative Sharpe."""
        equity = np.linspace(100000, 80000, 252)
        equity_df = pd.DataFrame({"equity": equity})
        trades = [{"pnl_pct": -2.0, "bars_held": 5} for _ in range(10)]

        result = compute_metrics("test", trades, equity_df, 100000, 252)
        assert result.sharpe < 0, f"Expected negative Sharpe, got {result.sharpe}"

    def test_sharpe_zero_for_flat_equity(self):
        """Flat equity should give zero Sharpe."""
        equity = np.full(252, 100000.0)
        equity_df = pd.DataFrame({"equity": equity})
        trades = [{"pnl_pct": 0.0, "bars_held": 5}]

        result = compute_metrics("test", trades, equity_df, 100000, 252)
        assert result.sharpe == 0.0

    def test_sortino_greater_than_sharpe_for_positive_returns(self):
        """With positive skew, Sortino should be >= Sharpe."""
        np.random.seed(42)
        # Generate returns with positive skew
        equity = 100000 * np.cumprod(1 + np.abs(np.random.randn(252)) * 0.01)
        equity_df = pd.DataFrame({"equity": equity})
        trades = [{"pnl_pct": 1.0, "bars_held": 5} for _ in range(20)]

        result = compute_metrics("test", trades, equity_df, 100000, 252)
        assert result.sortino >= result.sharpe, (
            f"Sortino {result.sortino} should be >= Sharpe {result.sharpe} "
            "for positively skewed returns"
        )


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


class TestDrawdownDuration:
    """Verify drawdown duration tracking (Chan Ch.1 p.40)."""

    def test_no_drawdown(self):
        """Monotonically rising equity has no drawdown."""
        equity = np.linspace(100, 200, 100)
        durations = _compute_drawdown_durations(equity)
        assert len(durations) == 0

    def test_single_drawdown(self):
        """One dip and recovery."""
        equity = np.array([100, 100, 90, 85, 90, 100, 110])
        durations = _compute_drawdown_durations(equity)
        assert len(durations) == 1
        assert durations[0] == 3  # bars 2,3,4 are in drawdown; bar 5 recovers

    def test_multiple_drawdowns(self):
        """Multiple separate drawdown periods."""
        equity = np.array([100, 110, 100, 110, 120, 110, 120, 130])
        durations = _compute_drawdown_durations(equity)
        assert len(durations) == 2  # two drawdown periods

    def test_never_recovered_drawdown(self):
        """Drawdown that never recovers by end of data."""
        equity = np.array([100, 110, 105, 100, 95])
        durations = _compute_drawdown_durations(equity)
        assert len(durations) == 1
        assert durations[0] == 3  # bars 2,3,4 in drawdown (never recovered)

    def test_drawdown_duration_in_metrics(self):
        """Verify drawdown duration appears in BacktestResult."""
        # Equity: up, then sustained dip for 30 bars
        equity = np.concatenate(
            [
                np.linspace(100000, 110000, 50),
                np.linspace(110000, 95000, 30),
                np.linspace(95000, 105000, 20),
            ]
        )
        equity_df = pd.DataFrame({"equity": equity})
        trades = [{"pnl_pct": 1.0, "bars_held": 5}]
        result = compute_metrics("test", trades, equity_df, 100000, 100)
        assert result.max_drawdown_duration_bars > 0
        assert result.avg_drawdown_duration_bars > 0


class TestKellyCriterion:
    """Verify Kelly fraction computation (Chan Ch.8 pp.190-195)."""

    def test_positive_kelly_for_profitable_strategy(self):
        """Positive mean returns -> positive Kelly."""
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)  # slightly positive
        kelly, half_kelly, leverage = _compute_kelly(returns)
        assert kelly > 0
        assert half_kelly == kelly / 2.0
        assert leverage > 0

    def test_negative_kelly_for_losing_strategy(self):
        """Negative mean returns -> negative Kelly (don't trade)."""
        returns = pd.Series(np.random.randn(252) * 0.01 - 0.002)  # negative drift
        kelly, half_kelly, leverage = _compute_kelly(returns)
        assert kelly < 0
        assert leverage == 0.0  # optimal leverage clamped to 0

    def test_kelly_in_metrics_result(self):
        """Kelly shows up in BacktestResult."""
        equity = np.linspace(100000, 120000, 252)
        equity_df = pd.DataFrame({"equity": equity})
        trades = [{"pnl_pct": 2.0, "bars_held": 5} for _ in range(10)]
        result = compute_metrics("test", trades, equity_df, 100000, 252)
        # Rising equity -> positive Kelly
        assert result.kelly_fraction > 0
        assert result.half_kelly_fraction == round(result.kelly_fraction / 2, 6)


class TestDeflatedSharpe:
    """Verify deflated Sharpe ratio for multiple testing (Chan Ch.1 pp.22-25)."""

    def test_single_strategy_no_deflation(self):
        """With n_strategies=1, deflated Sharpe = observed Sharpe."""
        returns = pd.Series(np.random.randn(252) * 0.01)
        dsr = _compute_deflated_sharpe(1.5, 252, 1, returns)
        assert dsr == 1.5

    def test_more_strategies_lower_deflated_sharpe(self):
        """More strategies tested -> lower deflated Sharpe."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)
        dsr_10 = _compute_deflated_sharpe(2.0, 252, 10, returns)
        dsr_1000 = _compute_deflated_sharpe(2.0, 252, 1000, returns)
        dsr_100000 = _compute_deflated_sharpe(2.0, 252, 100000, returns)
        # More strategies = higher benchmark = lower deflated Sharpe
        assert dsr_10 > dsr_1000 > dsr_100000

    def test_deflated_sharpe_in_metrics(self):
        """Deflated Sharpe shows up in BacktestResult."""
        equity = np.linspace(100000, 120000, 252)
        equity_df = pd.DataFrame({"equity": equity})
        trades = [{"pnl_pct": 2.0, "bars_held": 5} for _ in range(10)]
        result = compute_metrics("test", trades, equity_df, 100000, 252, n_strategies_tested=100)
        assert isinstance(result.deflated_sharpe, float)


class TestInformationRatio:
    """Verify information ratio vs buy-and-hold (Chan Ch.1 p.41)."""

    def test_outperforming_strategy(self):
        """Strategy beating benchmark -> positive IR."""
        strategy_returns = pd.Series(np.ones(252) * 0.002)
        benchmark_returns = pd.Series(np.ones(252) * 0.001)
        ir = _compute_information_ratio(strategy_returns, benchmark_returns)
        assert ir > 0

    def test_underperforming_strategy(self):
        """Strategy lagging benchmark -> negative IR."""
        np.random.seed(42)
        noise = np.random.randn(252) * 0.001
        strategy_returns = pd.Series(noise + 0.0005)
        benchmark_returns = pd.Series(noise + 0.002)
        ir = _compute_information_ratio(strategy_returns, benchmark_returns)
        assert ir < 0

    def test_no_benchmark_returns_zero(self):
        """No benchmark provided -> IR = 0."""
        strategy_returns = pd.Series(np.ones(252) * 0.002)
        ir = _compute_information_ratio(strategy_returns, None)
        assert ir == 0.0

    def test_information_ratio_in_metrics(self):
        """IR shows up in BacktestResult when benchmark provided."""
        equity = np.linspace(100000, 120000, 252)
        equity_df = pd.DataFrame({"equity": equity})
        trades = [{"pnl_pct": 2.0, "bars_held": 5} for _ in range(10)]
        benchmark = pd.Series(np.ones(252) * 0.001)
        result = compute_metrics(
            "test", trades, equity_df, 100000, 252, benchmark_returns=benchmark
        )
        assert isinstance(result.information_ratio, float)


class TestNewResultFields:
    """Verify all new BacktestResult fields from the Chan audit."""

    def test_new_fields_exist(self):
        """All new fields should have proper defaults."""
        result = BacktestResult(strategy_id="test")
        assert hasattr(result, "max_drawdown_duration_bars")
        assert hasattr(result, "avg_drawdown_duration_bars")
        assert hasattr(result, "kelly_fraction")
        assert hasattr(result, "half_kelly_fraction")
        assert hasattr(result, "optimal_leverage")
        assert hasattr(result, "deflated_sharpe")
        assert hasattr(result, "information_ratio")

    def test_new_fields_in_to_dict(self):
        """All new fields should appear in to_dict() output."""
        result = BacktestResult(strategy_id="test")
        d = result.to_dict()
        assert "max_drawdown_duration_bars" in d
        assert "avg_drawdown_duration_bars" in d
        assert "kelly_fraction" in d
        assert "half_kelly_fraction" in d
        assert "optimal_leverage" in d
        assert "deflated_sharpe" in d
        assert "information_ratio" in d
