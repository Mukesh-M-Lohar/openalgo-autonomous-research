"""Pipeline orchestrator — sequences all stages from generation to export."""

from __future__ import annotations

import json
import logging
import time
import uuid

import pandas as pd

from quant_engine.backtest.engine import BacktestEngine
from quant_engine.config import ResearchConfig
from quant_engine.data.client import OpenAlgoClient
from quant_engine.data.preprocessor import DataPreprocessor
from quant_engine.evolution.fitness import weighted_fitness
from quant_engine.evolution.population import Population
from quant_engine.generation.generator import StrategyGenerator
from quant_engine.generation.validator import FastRejectValidator
from quant_engine.models.results import BacktestResult, RejectionRecord, ValidationResult
from quant_engine.models.strategy import StrategyGenome
from quant_engine.parallel.pool import WorkerPool
from quant_engine.parallel.progress import RunProgress
from quant_engine.ranking.scorer import RankingEngine
from quant_engine.storage.csv_backend import CsvStorage
from quant_engine.validation.monte_carlo import MonteCarloValidator
from quant_engine.validation.out_of_sample import OOSValidator
from quant_engine.validation.parameter_stability import ParameterStabilityValidator
from quant_engine.validation.stress_test import StressTestValidator
from quant_engine.validation.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full research pipeline from config to ranked results."""

    def __init__(self, config: ResearchConfig):
        self._config = config
        self._storage = CsvStorage(config.output.base_dir)
        self._pool = WorkerPool(max_workers=config.execution.max_workers)
        self._progress = RunProgress(run_id="")
        self._stop_requested = False

    def run(self, run_id: str | None = None) -> str:
        """Execute the full research pipeline.

        Returns the run_id.
        """
        run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self._progress = RunProgress(run_id=run_id)
        self._stop_requested = False

        logger.info(f"Starting research run: {run_id}")
        self._storage.init_run(run_id, self._config.model_dump())

        try:
            # Stage 1: Fetch data
            data = self._fetch_data()
            if not data:
                logger.error("No data fetched. Aborting.")
                self._progress.status = "failed"
                return run_id

            # Split data
            preprocessor = DataPreprocessor(
                train_pct=self._config.data.train_pct,
                validation_pct=self._config.data.validation_pct,
                test_pct=self._config.data.test_pct,
            )

            # Use first symbol's data for backtesting
            first_key = list(data.keys())[0]
            symbol_data = data[first_key]
            primary_tf = list(symbol_data.keys())[0]
            full_df = symbol_data[primary_tf]
            train_df, val_df, test_df = preprocessor.split(full_df)
            train_data = {primary_tf: train_df}
            val_data = {primary_tf: val_df}
            test_data = {primary_tf: test_df}

            # Stage 2: Generate strategies
            strategies = self._generate(run_id)
            if self._stop_requested:
                return run_id

            # Stage 3: Fast reject
            strategies, rejections = self._fast_reject(run_id, strategies)
            if self._stop_requested:
                return run_id

            # Stage 4: Backtest
            backtest_results = self._backtest(run_id, strategies, train_data)
            if self._stop_requested:
                return run_id

            # Stage 5: Filter by backtest metrics
            survivors, bt_rejections = self._filter_backtest(run_id, strategies, backtest_results)
            if self._stop_requested:
                return run_id

            # Stage 6: Walk-forward validation
            wf_results = self._walk_forward(run_id, survivors, symbol_data)
            if self._stop_requested:
                return run_id

            # Stage 7: Out-of-sample testing
            oos_results = self._out_of_sample(run_id, survivors, backtest_results, test_data)
            if self._stop_requested:
                return run_id

            # Stage 8: Robustness testing
            robustness_results = self._robustness(run_id, survivors, backtest_results, train_data)
            if self._stop_requested:
                return run_id

            # Stage 9: Evolution (if enabled)
            if self._config.evolution.enabled and survivors:
                evolved = self._evolve(run_id, survivors, backtest_results, train_data)
                survivors.extend(evolved)

            # Stage 10: Final ranking
            self._rank_and_export(run_id, survivors, backtest_results, robustness_results)

            self._progress.status = "completed"
            logger.info(f"Research run {run_id} completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self._progress.status = "failed"

        finally:
            self._pool.stop()

        return run_id

    def stop(self) -> None:
        self._stop_requested = True
        self._progress.status = "stopping"

    @property
    def progress(self) -> RunProgress:
        return self._progress

    def _fetch_data(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Stage 1: Fetch OHLCV data from OpenAlgo."""
        logger.info("Stage 1: Fetching data from OpenAlgo...")
        client = OpenAlgoClient(self._config.data.openalgo)
        try:
            return client.fetch_all(
                symbols=self._config.data.symbols,
                timeframes=self._config.data.timeframes,
                start_date=self._config.data.start_date,
                end_date=self._config.data.end_date,
            )
        finally:
            client.close()

    def _generate(self, run_id: str) -> list[StrategyGenome]:
        """Stage 2: Generate candidate strategies."""
        logger.info("Stage 2: Generating strategies...")
        progress = self._progress.start_stage("generation", self._config.generation.target_count)

        generator = StrategyGenerator(self._config)
        strategies = generator.generate()

        progress.completed = len(strategies)
        progress.passed = len(strategies)

        if self._config.output.save_all_candidates:
            summaries = [{"id": s.id, "style": s.trading_style.value, "fingerprint": s.fingerprint()} for s in strategies]
            self._storage.save_generated(run_id, summaries)

        logger.info(f"Generated {len(strategies)} unique strategies")
        return strategies

    def _fast_reject(
        self, run_id: str, strategies: list[StrategyGenome]
    ) -> tuple[list[StrategyGenome], list[RejectionRecord]]:
        """Stage 3: Fast-reject invalid strategies."""
        logger.info("Stage 3: Fast reject...")
        progress = self._progress.start_stage("fast_reject", len(strategies))

        validator = FastRejectValidator(self._config.filters.fast_reject)
        passed, rejections = validator.validate_batch(strategies)

        progress.completed = len(strategies)
        progress.passed = len(passed)
        progress.rejected = len(rejections)

        self._storage.save_rejections(run_id, rejections)
        self._save_rejection_details(run_id, strategies, rejections)

        logger.info(f"Fast reject: {len(passed)} passed, {len(rejections)} rejected")
        return passed, rejections

    def _backtest(
        self, run_id: str, strategies: list[StrategyGenome], data: dict[str, pd.DataFrame]
    ) -> dict[str, BacktestResult]:
        """Stage 4: Backtest all strategies."""
        logger.info(f"Stage 4: Backtesting {len(strategies)} strategies...")
        progress = self._progress.start_stage("backtest", len(strategies))

        engine = BacktestEngine(cost_model=self._config.cost_model)
        results: dict[str, BacktestResult] = {}

        for i, strategy in enumerate(strategies):
            result = engine.run(strategy, data)
            if result is not None:
                results[strategy.id] = result
            progress.completed = i + 1

        progress.passed = len(results)
        progress.rejected = len(strategies) - len(results)

        bt_list = [r.to_dict() for r in results.values()]
        self._storage.save_backtest_results(run_id, bt_list)

        logger.info(f"Backtest complete: {len(results)} produced results")
        return results

    def _filter_backtest(
        self,
        run_id: str,
        strategies: list[StrategyGenome],
        results: dict[str, BacktestResult],
    ) -> tuple[list[StrategyGenome], list[RejectionRecord]]:
        """Stage 5: Filter strategies by backtest performance thresholds."""
        logger.info("Stage 5: Filtering by backtest metrics...")
        filters = self._config.filters.backtest
        passed = []
        rejections = []

        for strategy in strategies:
            bt = results.get(strategy.id)
            if bt is None:
                rejections.append(RejectionRecord(
                    strategy_id=strategy.id,
                    stage="backtest_filter",
                    rejection_reason="no_backtest_result",
                    threshold="required",
                    actual_value="none",
                ))
                continue

            rejection = self._check_bt_filters(strategy.id, bt, filters)
            if rejection:
                rejections.append(rejection)
            else:
                passed.append(strategy)

        self._storage.save_rejections(run_id, rejections)
        logger.info(f"Backtest filter: {len(passed)} passed, {len(rejections)} rejected")
        return passed, rejections

    def _check_bt_filters(self, sid: str, bt: BacktestResult, filters) -> RejectionRecord | None:
        checks = [
            ("min_trades", bt.total_trades, filters.min_trades, "total_trades_below_min"),
            ("min_sharpe", bt.sharpe, filters.min_sharpe, "sharpe_below_min"),
            ("min_profit_factor", bt.profit_factor, filters.min_profit_factor, "profit_factor_below_min"),
        ]
        for threshold_name, actual, threshold, reason in checks:
            if actual < threshold:
                return RejectionRecord(
                    strategy_id=sid,
                    stage="backtest_filter",
                    rejection_reason=reason,
                    threshold=str(threshold),
                    actual_value=str(round(actual, 4)),
                )
        if bt.max_drawdown_pct > filters.max_drawdown * 100:
            return RejectionRecord(
                strategy_id=sid,
                stage="backtest_filter",
                rejection_reason="max_drawdown_exceeded",
                threshold=str(filters.max_drawdown),
                actual_value=str(round(bt.max_drawdown_pct / 100, 4)),
            )
        return None

    def _walk_forward(
        self, run_id: str, strategies: list[StrategyGenome], data: dict[str, pd.DataFrame]
    ) -> dict[str, dict]:
        """Stage 6: Walk-forward validation."""
        logger.info(f"Stage 6: Walk-forward validation for {len(strategies)} strategies...")
        wf = WalkForwardValidator(cost_model=self._config.cost_model)
        results = {}
        for s in strategies:
            results[s.id] = wf.validate(s, data)
        self._storage.save_validation_results(run_id, "walkforward", list(results.values()))
        return results

    def _out_of_sample(
        self,
        run_id: str,
        strategies: list[StrategyGenome],
        backtest_results: dict[str, BacktestResult],
        oos_data: dict[str, pd.DataFrame],
    ) -> dict[str, dict]:
        """Stage 7: Out-of-sample testing."""
        logger.info(f"Stage 7: OOS testing for {len(strategies)} strategies...")
        oos = OOSValidator(cost_model=self._config.cost_model)
        results = {}
        for s in strategies:
            bt = backtest_results.get(s.id)
            if bt:
                results[s.id] = oos.validate(s, bt, oos_data)
        self._storage.save_validation_results(run_id, "oos_results", list(results.values()))
        return results

    def _robustness(
        self,
        run_id: str,
        strategies: list[StrategyGenome],
        backtest_results: dict[str, BacktestResult],
        data: dict[str, pd.DataFrame],
    ) -> dict[str, dict]:
        """Stage 8: Robustness testing (Monte Carlo, param stability, stress)."""
        logger.info(f"Stage 8: Robustness testing for {len(strategies)} strategies...")
        mc = MonteCarloValidator()
        ps = ParameterStabilityValidator(cost_model=self._config.cost_model)
        st = StressTestValidator(cost_model=self._config.cost_model)

        results = {}
        for s in strategies:
            bt = backtest_results.get(s.id)
            if not bt:
                continue

            # Monte Carlo needs trades — re-run quick backtest to get them
            engine = BacktestEngine(cost_model=self._config.cost_model)
            bt_run = engine.run(s, data)
            trades = []  # simplified: use backtest result metrics

            mc_result = mc.validate(bt, trades)
            ps_result = ps.validate(s, bt, data)
            st_result = st.validate(s, bt, data)

            combined = {
                "strategy_id": s.id,
                **mc_result,
                **ps_result,
                **st_result,
                "robustness_score": round(
                    (mc_result.get("monte_carlo_score", 0)
                     + ps_result.get("param_stability_score", 0)
                     + st_result.get("stress_test_score", 0)) / 3, 2
                ),
            }
            results[s.id] = combined

        self._storage.save_validation_results(run_id, "robustness", list(results.values()))
        return results

    def _evolve(
        self,
        run_id: str,
        survivors: list[StrategyGenome],
        backtest_results: dict[str, BacktestResult],
        data: dict[str, pd.DataFrame],
    ) -> list[StrategyGenome]:
        """Stage 9: Evolution — mutate and breed top strategies."""
        logger.info("Stage 9: Evolution...")
        evo_config = self._config.evolution
        population = Population(evo_config)

        fitness_scores = []
        for s in survivors:
            bt = backtest_results.get(s.id)
            if bt:
                score = weighted_fitness(bt, None, self._config.ranking.objectives)
                fitness_scores.append(score)
            else:
                fitness_scores.append(0.0)

        population.initialize(survivors, fitness_scores)

        all_evolved = []
        engine = BacktestEngine(cost_model=self._config.cost_model)

        for gen in range(evo_config.generations):
            offspring = population.evolve()

            new_scores = []
            for s in offspring:
                result = engine.run(s, data)
                if result:
                    backtest_results[s.id] = result
                    score = weighted_fitness(result, None, self._config.ranking.objectives)
                    new_scores.append(score)
                else:
                    new_scores.append(0.0)

            population.update_fitness(offspring, new_scores)
            logger.info(f"Evolution gen {gen + 1}: top fitness = {max(new_scores):.4f}")

        all_evolved = population.get_top(evo_config.population_size // 2)
        logger.info(f"Evolution produced {len(all_evolved)} improved strategies")
        return all_evolved

    def _rank_and_export(
        self,
        run_id: str,
        survivors: list[StrategyGenome],
        backtest_results: dict[str, BacktestResult],
        robustness_results: dict[str, dict],
    ) -> None:
        """Stage 10: Rank and save results."""
        logger.info("Stage 10: Ranking and exporting...")
        ranking_engine = RankingEngine(self._config.ranking)

        strategies_data = []
        for s in survivors:
            bt = backtest_results.get(s.id)
            if not bt:
                continue
            rob = robustness_results.get(s.id, {})
            val = ValidationResult(
                strategy_id=s.id,
                walk_forward_score=rob.get("walk_forward_score", 0),
                monte_carlo_score=rob.get("monte_carlo_score", 0),
                param_stability_score=rob.get("param_stability_score", 0),
                stress_test_score=rob.get("stress_test_score", 0),
                robustness_score=rob.get("robustness_score", 0),
            )
            strategies_data.append({
                "strategy_id": s.id,
                "backtest": bt,
                "validation": val,
            })

        ranked = ranking_engine.rank(strategies_data)

        # Save survivors
        survivor_dicts = [r.to_dict() for r in ranked]
        self._storage.save_survivors(run_id, survivor_dicts)

        # Save winners
        winners = ranking_engine.get_winners(ranked)
        winner_dicts = []
        for category, strats in winners.items():
            for s in strats:
                d = s.to_dict()
                d["winner_category"] = category
                winner_dicts.append(d)
        self._storage.save_winners(run_id, winner_dicts)

        logger.info(f"Ranked {len(ranked)} strategies, saved winners")

    def _save_rejection_details(
        self, run_id: str, strategies: list[StrategyGenome], rejections: list[RejectionRecord]
    ) -> None:
        """Save detailed info about rejected strategies for analysis."""
        rejected_ids = {r.strategy_id for r in rejections}
        details = []
        for s in strategies:
            if s.id in rejected_ids:
                details.append({
                    "id": s.id,
                    "trading_style": s.trading_style.value,
                    "entry_summary": str(s.entry_long.to_dict()),
                    "exit_summary": str(s.exit_long.to_dict()),
                    "timeframes": ",".join(tf.value for tf in s.timeframes_used),
                })
        self._storage.save_rejection_details(run_id, details)
