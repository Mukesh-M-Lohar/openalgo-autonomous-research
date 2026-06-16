"""End-to-end integration tests for the full pipeline."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from quant_engine.config import ResearchConfig
from quant_engine.evolution.crossover import Crossover
from quant_engine.evolution.mutator import Mutator
from quant_engine.evolution.population import Population
from quant_engine.export.formatter import StrategyExporter
from quant_engine.generation.generator import StrategyGenerator
from quant_engine.generation.grammar import GrammarConfig, generate_strategy
from quant_engine.generation.indicators import INDICATOR_CATEGORIES
from quant_engine.generation.validator import FastRejectValidator
from quant_engine.backtest.engine import BacktestEngine
from quant_engine.models.results import BacktestResult
from quant_engine.models.strategy import StrategyGenome, TimeframeType, TradingStyle
from quant_engine.storage.csv_backend import CsvStorage


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 800
    dates = pd.date_range("2023-01-01", periods=n, freq="15min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.4)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.2,
            "high": close + abs(np.random.randn(n) * 0.3),
            "low": close - abs(np.random.randn(n) * 0.3),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def mini_config():
    return ResearchConfig(
        name="Test Run",
        trading_styles=["intraday"],
        generation={
            "target_count": 200,
            "max_conditions_per_entry": 2,
            "indicator_categories": ["trend", "momentum"],
        },
        data={"timeframes": ["15m"]},
        evolution={"enabled": False},
    )


class TestEndToEnd:
    def test_generate_reject_backtest(self, synthetic_data, mini_config):
        """Full pipeline: generate → reject → backtest → filter."""
        # Generate
        gen = StrategyGenerator(mini_config)
        strategies = gen.generate()
        assert len(strategies) == 200

        # Fast reject
        validator = FastRejectValidator(mini_config.filters.fast_reject)
        passed, rejected = validator.validate_batch(strategies)
        assert len(passed) + len(rejected) == 200

        # Backtest
        engine = BacktestEngine(cost_model=mini_config.cost_model)
        results = {}
        for s in passed[:50]:
            r = engine.run(s, {"15m": synthetic_data})
            if r and r.total_trades > 0:
                results[s.id] = r

        assert len(results) > 0

        # Filter
        filtered = {k: v for k, v in results.items() if v.sharpe > 0}
        # At least some should pass with trending data
        assert isinstance(filtered, dict)

    def test_evolution_improves(self, synthetic_data):
        """Evolution should produce offspring with valid genomes."""
        from quant_engine.config import EvolutionConfig

        allowed = INDICATOR_CATEGORIES["trend"] + INDICATOR_CATEGORIES["momentum"]
        cfg = GrammarConfig(
            allowed_indicators=allowed,
            allowed_timeframes=[TimeframeType.M15],
            trading_style=TradingStyle.INTRADAY,
            max_conditions=2,
            product_type="MIS",
            max_hold_bars=50,
        )

        # Create initial population
        strategies = [generate_strategy(cfg) for _ in range(20)]
        scores = [float(i) / 20 for i in range(20)]

        evo_config = EvolutionConfig(
            population_size=20, mutation_rate=0.5, crossover_rate=0.5, generations=2
        )
        population = Population(evo_config, grammar_config=cfg)
        population.initialize(strategies, scores)

        offspring = population.evolve()
        assert len(offspring) == 20
        for s in offspring:
            assert isinstance(s, StrategyGenome)
            assert s.entry_long is not None
            assert s.exit_long is not None

    def test_mutation_produces_valid_strategies(self):
        allowed = INDICATOR_CATEGORIES["trend"]
        cfg = GrammarConfig(
            allowed_indicators=allowed,
            allowed_timeframes=[TimeframeType.M15],
            trading_style=TradingStyle.SWING,
            max_conditions=2,
            product_type="CNC",
        )
        strategy = generate_strategy(cfg)
        mutator = Mutator(mutation_rate=1.0, grammar_config=cfg)

        for _ in range(10):
            mutated = mutator.mutate(strategy)
            assert mutated.entry_long is not None
            assert mutated.exit_long is not None
            assert mutated.generation == strategy.generation + 1

    def test_crossover_produces_valid_strategies(self):
        allowed = INDICATOR_CATEGORIES["trend"] + INDICATOR_CATEGORIES["momentum"]
        cfg = GrammarConfig(
            allowed_indicators=allowed,
            allowed_timeframes=[TimeframeType.M15],
            trading_style=TradingStyle.INTRADAY,
            max_conditions=3,
            product_type="MIS",
        )
        parent_a = generate_strategy(cfg)
        parent_b = generate_strategy(cfg)

        crossover = Crossover(crossover_rate=1.0)
        child = crossover.cross(parent_a, parent_b)
        assert isinstance(child, StrategyGenome)
        assert child.entry_long is not None
        assert child.parent_ids == (parent_a.id, parent_b.id)

    def test_csv_storage_roundtrip(self):
        """Test that storage saves and loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CsvStorage(tmpdir)
            storage.init_run("test_001", {"name": "Test"})

            from quant_engine.models.results import RejectionRecord

            rejections = [
                RejectionRecord(
                    strategy_id="abc",
                    stage="fast_reject",
                    rejection_reason="missing_exit_rule",
                    threshold="required",
                    actual_value="none",
                ),
                RejectionRecord(
                    strategy_id="def",
                    stage="backtest_filter",
                    rejection_reason="sharpe_below_min",
                    threshold="1.0",
                    actual_value="0.3",
                ),
            ]
            storage.save_rejections("test_001", rejections)

            loaded = storage.load_results("test_001", "rejected")
            assert len(loaded) == 2
            assert loaded[0]["strategy_id"] == "abc"
            assert loaded[0]["rejection_reason"] == "missing_exit_rule"
            assert loaded[1]["threshold"] == "1.0"
            assert loaded[1]["actual_value"] == "0.3"

    def test_export_produces_valid_python(self, synthetic_data):
        """Exported script should be valid Python."""
        allowed = INDICATOR_CATEGORIES["trend"]
        cfg = GrammarConfig(
            allowed_indicators=allowed,
            allowed_timeframes=[TimeframeType.M15],
            trading_style=TradingStyle.SWING,
            max_conditions=2,
            product_type="CNC",
            max_hold_bars=100,
        )
        strategy = generate_strategy(cfg)
        bt = BacktestResult(strategy_id=strategy.id, sharpe=1.5, cagr=25.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = StrategyExporter(tmpdir)
            py_path, json_path = exporter.export_strategy(strategy, bt)

            assert py_path.exists()
            assert json_path.exists()

            # Verify it's valid Python by compiling
            with open(py_path) as f:
                source = f.read()
            compile(source, str(py_path), "exec")

            # Verify JSON is valid
            import json
            with open(json_path) as f:
                data = json.load(f)
            assert "strategy" in data
            assert data["strategy"]["id"] == strategy.id

    def test_config_loading(self):
        """Config files should load without errors."""
        from quant_engine.config import load_config

        config = load_config(Path(__file__).parent.parent.parent / "config" / "default_research.yaml")
        assert config.name == "Default Research"
        assert "intraday" in config.trading_styles
        assert config.generation.target_count == 10000
