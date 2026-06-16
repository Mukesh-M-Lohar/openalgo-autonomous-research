"""Storage backend protocol — swap CSV for DuckDB without changing pipeline logic."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from quant_engine.models.results import RejectionRecord


class StorageBackend(Protocol):
    """Abstract storage interface for research runs."""

    def init_run(self, run_id: str, config: dict) -> None:
        """Initialize storage for a new research run."""
        ...

    def save_generated(self, run_id: str, strategies: list[dict]) -> None:
        """Save all generated strategy definitions."""
        ...

    def save_rejections(self, run_id: str, rejections: list[RejectionRecord]) -> None:
        """Save rejected strategies with reasons."""
        ...

    def save_rejection_details(self, run_id: str, details: list[dict]) -> None:
        """Save full params of rejected strategies for analysis."""
        ...

    def save_backtest_results(self, run_id: str, results: list[dict]) -> None:
        """Save strategies that passed backtesting with metrics."""
        ...

    def save_validation_results(self, run_id: str, stage: str, results: list[dict]) -> None:
        """Save validation stage results (walk-forward, OOS, robustness)."""
        ...

    def save_survivors(self, run_id: str, survivors: list[dict]) -> None:
        """Save final surviving strategies."""
        ...

    def save_winners(self, run_id: str, winners: list[dict]) -> None:
        """Save top ranked winners."""
        ...

    def save_trade_log(self, run_id: str, strategy_id: str, trades: pd.DataFrame) -> None:
        """Save trade log for a strategy."""
        ...

    def save_equity_curve(self, run_id: str, strategy_id: str, equity: pd.DataFrame) -> None:
        """Save equity curve for a strategy."""
        ...

    def load_run_config(self, run_id: str) -> dict:
        """Load the config used for a run."""
        ...

    def load_results(self, run_id: str, stage: str) -> list[dict]:
        """Load results from a specific stage."""
        ...

    def load_winners(self, run_id: str) -> list[dict]:
        """Load winning strategies."""
        ...

    def list_runs(self) -> list[str]:
        """List all run IDs."""
        ...
