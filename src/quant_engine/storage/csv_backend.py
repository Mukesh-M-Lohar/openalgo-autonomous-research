"""CSV storage backend — V1 implementation, human-readable output."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from quant_engine.models.results import RejectionRecord

logger = logging.getLogger(__name__)


class CsvStorage:
    """Stores all research run data as CSV files with clear directory structure."""

    def __init__(self, base_dir: str | Path = "./data/runs"):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        d = self._base / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def init_run(self, run_id: str, config: dict) -> None:
        run_dir = self._run_dir(run_id)
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        (run_dir / "trade_logs").mkdir(exist_ok=True)
        (run_dir / "equity_curves").mkdir(exist_ok=True)
        (run_dir / "exports").mkdir(exist_ok=True)
        logger.info(f"Initialized run storage: {run_dir}")

    def save_generated(self, run_id: str, strategies: list[dict]) -> None:
        if not strategies:
            return
        df = pd.DataFrame(strategies)
        path = self._run_dir(run_id) / "generated.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(strategies)} generated strategies to {path}")

    def save_rejections(self, run_id: str, rejections: list[RejectionRecord]) -> None:
        if not rejections:
            return
        rows = [r.to_dict() for r in rejections]
        df = pd.DataFrame(rows)
        path = self._run_dir(run_id) / "rejected.csv"
        if path.exists():
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(rejections)} rejections to {path}")

    def save_rejection_details(self, run_id: str, details: list[dict]) -> None:
        if not details:
            return
        rows = []
        for d in details:
            rows.append({
                "strategy_id": d.get("id", ""),
                "trading_style": d.get("trading_style", ""),
                "entry_indicators": d.get("entry_summary", ""),
                "exit_type": d.get("exit_summary", ""),
                "timeframes": d.get("timeframes", ""),
                "params_json": json.dumps(d, default=str),
            })
        df = pd.DataFrame(rows)
        path = self._run_dir(run_id) / "rejected_details.csv"
        if path.exists():
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(path, index=False)

    def save_backtest_results(self, run_id: str, results: list[dict]) -> None:
        if not results:
            return
        df = pd.DataFrame(results)
        path = self._run_dir(run_id) / "backtested.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(results)} backtest results to {path}")

    def save_validation_results(self, run_id: str, stage: str, results: list[dict]) -> None:
        if not results:
            return
        df = pd.DataFrame(results)
        path = self._run_dir(run_id) / f"{stage}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(results)} {stage} results to {path}")

    def save_survivors(self, run_id: str, survivors: list[dict]) -> None:
        if not survivors:
            return
        df = pd.DataFrame(survivors)
        path = self._run_dir(run_id) / "survivors.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(survivors)} survivors to {path}")

    def save_winners(self, run_id: str, winners: list[dict]) -> None:
        if not winners:
            return
        df = pd.DataFrame(winners)
        path = self._run_dir(run_id) / "winners.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(winners)} winners to {path}")

    def save_trade_log(self, run_id: str, strategy_id: str, trades: pd.DataFrame) -> None:
        if trades.empty:
            return
        path = self._run_dir(run_id) / "trade_logs" / f"{strategy_id}_trades.csv"
        trades.to_csv(path, index=True)

    def save_equity_curve(self, run_id: str, strategy_id: str, equity: pd.DataFrame) -> None:
        if equity.empty:
            return
        path = self._run_dir(run_id) / "equity_curves" / f"{strategy_id}_equity.csv"
        equity.to_csv(path, index=True)

    def load_run_config(self, run_id: str) -> dict:
        path = self._run_dir(run_id) / "config.yaml"
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f)

    def load_results(self, run_id: str, stage: str) -> list[dict]:
        path = self._run_dir(run_id) / f"{stage}.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    def load_winners(self, run_id: str) -> list[dict]:
        return self.load_results(run_id, "winners")

    def list_runs(self) -> list[str]:
        runs = []
        for d in self._base.iterdir():
            if d.is_dir() and (d / "config.yaml").exists():
                runs.append(d.name)
        return sorted(runs)
