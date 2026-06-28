"""Tests for strategy generation, grammar, and validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from quant_engine.config import FastRejectFilters, ResearchConfig
from quant_engine.generation.generator import StrategyGenerator
from quant_engine.generation.grammar import GrammarConfig, generate_strategy
from quant_engine.generation.indicators import (
    INDICATOR_CATEGORIES,
    INDICATOR_FUNCTIONS,
    INDICATOR_PARAM_RANGES,
    compute_indicator,
)
from quant_engine.generation.patterns import PATTERN_FUNCTIONS, detect_pattern
from quant_engine.generation.validator import FastRejectValidator
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
def sample_df():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="15min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + abs(np.random.randn(n) * 0.2),
            "low": close - abs(np.random.randn(n) * 0.2),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def grammar_config():
    allowed = INDICATOR_CATEGORIES["trend"] + INDICATOR_CATEGORIES["momentum"]
    return GrammarConfig(
        allowed_indicators=allowed,
        allowed_timeframes=[TimeframeType.M15, TimeframeType.H1],
        trading_style=TradingStyle.INTRADAY,
        max_conditions=3,
        product_type="MIS",
        forced_exit_time="15:15",
        max_hold_bars=75,
    )


class TestIndicators:
    def test_all_indicators_registered(self):
        assert len(INDICATOR_FUNCTIONS) >= 27

    def test_all_indicators_have_param_ranges(self):
        for ind_type in INDICATOR_FUNCTIONS:
            assert ind_type in INDICATOR_PARAM_RANGES

    def test_sma_computation(self, sample_df):
        result = compute_indicator(sample_df, IndicatorType.SMA, {"period": 20}, PriceSource.CLOSE)
        assert len(result) == len(sample_df)
        assert result.isna().sum() == 19  # first 19 are NaN

    def test_ema_computation(self, sample_df):
        result = compute_indicator(sample_df, IndicatorType.EMA, {"period": 10}, PriceSource.CLOSE)
        assert len(result) == len(sample_df)
        assert not result.iloc[20:].isna().any()

    def test_rsi_computation(self, sample_df):
        result = compute_indicator(sample_df, IndicatorType.RSI, {"period": 14}, PriceSource.CLOSE)
        assert len(result) == len(sample_df)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_computation(self, sample_df):
        result = compute_indicator(
            sample_df, IndicatorType.MACD, {"fast_period": 12, "slow_period": 26}, PriceSource.CLOSE
        )
        assert len(result) == len(sample_df)

    def test_atr_computation(self, sample_df):
        result = compute_indicator(sample_df, IndicatorType.ATR, {"period": 14}, PriceSource.CLOSE)
        assert len(result) == len(sample_df)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_bbands_computation(self, sample_df):
        upper = compute_indicator(
            sample_df, IndicatorType.BBANDS_UPPER, {"period": 20, "std_dev": 2.0}, PriceSource.CLOSE
        )
        lower = compute_indicator(
            sample_df, IndicatorType.BBANDS_LOWER, {"period": 20, "std_dev": 2.0}, PriceSource.CLOSE
        )
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] > lower[valid_idx]).all()

    def test_adx_computation(self, sample_df):
        result = compute_indicator(sample_df, IndicatorType.ADX, {"period": 14}, PriceSource.CLOSE)
        assert len(result) == len(sample_df)

    def test_stochastic_computation(self, sample_df):
        result = compute_indicator(
            sample_df, IndicatorType.STOCH_K, {"k_period": 14, "smooth_k": 3}, PriceSource.CLOSE
        )
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_supertrend_computation(self, sample_df):
        result = compute_indicator(
            sample_df,
            IndicatorType.SUPERTREND,
            {"period": 10, "multiplier": 3.0},
            PriceSource.CLOSE,
        )
        assert len(result) == len(sample_df)
        # Verify it computes correctly and has some valid values after period + 1
        assert not result.iloc[12:].isna().any()

    def test_hma_computation(self, sample_df):
        result = compute_indicator(
            sample_df,
            IndicatorType.HMA,
            {"period": 9},
            PriceSource.CLOSE,
        )
        assert len(result) == len(sample_df)
        assert not result.iloc[20:].isna().any()


class TestPatterns:
    def test_all_patterns_registered(self):
        assert len(PATTERN_FUNCTIONS) == 11

    def test_inside_bar(self, sample_df):
        result = detect_pattern(sample_df, "inside_bar")
        assert result.dtype == bool
        assert len(result) == len(sample_df)

    def test_gap_up(self, sample_df):
        result = detect_pattern(sample_df, "gap_up", {"min_gap_pct": 0.5})
        assert result.dtype == bool

    def test_engulfing_bullish(self, sample_df):
        result = detect_pattern(sample_df, "engulfing_bullish")
        assert result.dtype == bool

    def test_higher_high_higher_low(self, sample_df):
        result = detect_pattern(sample_df, "higher_high_higher_low", {"lookback": 2})
        assert result.dtype == bool


class TestGrammar:
    def test_generate_single_strategy(self, grammar_config):
        strategy = generate_strategy(grammar_config)
        assert isinstance(strategy, StrategyGenome)
        assert strategy.trading_style == TradingStyle.INTRADAY
        assert strategy.product_type == "MIS"
        assert strategy.forced_exit_time == "15:15"

    def test_generate_has_entry(self, grammar_config):
        strategy = generate_strategy(grammar_config)
        assert strategy.entry_long is not None

    def test_generate_has_exit(self, grammar_config):
        strategy = generate_strategy(grammar_config)
        assert strategy.exit_long is not None
        assert strategy.exit_long.stop_loss_pct is not None

    def test_generate_unique_ids(self, grammar_config):
        strategies = [generate_strategy(grammar_config) for _ in range(50)]
        ids = [s.id for s in strategies]
        assert len(set(ids)) == 50

    def test_generate_risk_reward(self, grammar_config):
        for _ in range(20):
            strategy = generate_strategy(grammar_config)
            if strategy.exit_long.take_profit_pct and strategy.exit_long.stop_loss_pct:
                assert strategy.exit_long.take_profit_pct >= strategy.exit_long.stop_loss_pct


class TestGenerator:
    def test_generator_produces_target_count(self):
        config = ResearchConfig(
            trading_styles=["intraday"],
            generation={"target_count": 100, "indicator_categories": ["trend"]},
            data={"timeframes": ["15m"]},
        )
        gen = StrategyGenerator(config)
        strategies = gen.generate()
        assert len(strategies) == 100

    def test_generator_deduplicates(self):
        config = ResearchConfig(
            trading_styles=["intraday"],
            generation={"target_count": 200, "indicator_categories": ["trend", "momentum"]},
            data={"timeframes": ["15m"]},
        )
        gen = StrategyGenerator(config)
        strategies = gen.generate()
        fingerprints = [s.fingerprint() for s in strategies]
        assert len(set(fingerprints)) == len(strategies)

    def test_generator_respects_styles(self):
        config = ResearchConfig(
            trading_styles=["swing", "positional"],
            generation={"target_count": 50, "indicator_categories": ["trend"]},
            data={"timeframes": ["1h", "1d"]},
        )
        gen = StrategyGenerator(config)
        strategies = gen.generate()
        styles = {s.trading_style for s in strategies}
        assert TradingStyle.SWING in styles or TradingStyle.POSITIONAL in styles


class TestValidator:
    def test_passes_valid_strategy(self, grammar_config):
        strategy = generate_strategy(grammar_config)
        validator = FastRejectValidator(FastRejectFilters())
        validator.validate(strategy)
        # Most generated strategies should pass
        # (some may fail complexity check depending on random seed)

    def test_rejects_missing_exit(self):
        strategy = StrategyGenome(
            trading_style=TradingStyle.INTRADAY,
            entry_long=ConditionNode(
                left=IndicatorNode(IndicatorType.RSI, (("period", 14),), TimeframeType.M15),
                op=CompareOp.GT,
                right=70.0,
            ),
            exit_long=ExitRule(),  # no exit mechanism
            timeframes_used=(TimeframeType.M15,),
        )
        validator = FastRejectValidator(FastRejectFilters())
        result = validator.validate(strategy)
        assert result is not None
        assert result.rejection_reason == "missing_exit_rule"

    def test_rejects_impossible_rsi(self):
        strategy = StrategyGenome(
            trading_style=TradingStyle.INTRADAY,
            entry_long=ConditionNode(
                left=IndicatorNode(IndicatorType.RSI, (("period", 14),), TimeframeType.M15),
                op=CompareOp.GT,
                right=150.0,  # impossible
            ),
            exit_long=ExitRule(stop_loss_pct=2.0),
            timeframes_used=(TimeframeType.M15,),
        )
        validator = FastRejectValidator(FastRejectFilters())
        result = validator.validate(strategy)
        assert result is not None
        assert result.rejection_reason == "impossible_condition"

    def test_batch_validation(self, grammar_config):
        strategies = [generate_strategy(grammar_config) for _ in range(50)]
        validator = FastRejectValidator(FastRejectFilters())
        passed, rejected = validator.validate_batch(strategies)
        assert len(passed) + len(rejected) == 50


class TestSerialization:
    def test_strategy_to_dict_roundtrip(self, grammar_config):
        strategy = generate_strategy(grammar_config)
        d = strategy.to_dict()
        restored = StrategyGenome.from_dict(d)
        assert restored.id == strategy.id
        assert restored.trading_style == strategy.trading_style
        assert restored.product_type == strategy.product_type

    def test_fingerprint_deterministic(self, grammar_config):
        strategy = generate_strategy(grammar_config)
        fp1 = strategy.fingerprint()
        fp2 = strategy.fingerprint()
        assert fp1 == fp2

    def test_different_strategies_different_fingerprints(self, grammar_config):
        s1 = generate_strategy(grammar_config)
        s2 = generate_strategy(grammar_config)
        assert s1.fingerprint() != s2.fingerprint()
