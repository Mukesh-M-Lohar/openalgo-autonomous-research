"""Configuration loader — YAML + Pydantic validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file so that ${VAR} interpolation in YAML configs works
load_dotenv()


class OpenAlgoConfig(BaseModel):
    host: str = "http://127.0.0.1:5000"
    api_key: str = ""
    source: Literal["api", "db"] = "db"


class SymbolConfig(BaseModel):
    symbol: str
    exchange: str = "NSE"


class DataConfig(BaseModel):
    openalgo: OpenAlgoConfig = OpenAlgoConfig()
    symbols: list[SymbolConfig] = []
    timeframes: list[str] = ["5m", "15m", "1h", "1d"]
    start_date: str = "2020-01-01"
    end_date: str = "2025-01-01"
    train_pct: float = 0.7
    validation_pct: float = 0.15
    test_pct: float = 0.15


class StyleOverride(BaseModel):
    max_hold_bars: int | None = None
    min_hold_bars: int | None = None
    forced_exit_time: str | None = None
    product_type: str = "MIS"
    min_trades: int = 30


class FastRejectFilters(BaseModel):
    max_complexity: int = 5
    min_indicators: int = 1


class BacktestFilters(BaseModel):
    min_trades: int = 30
    min_sharpe: float = 1.0
    max_drawdown: float = 0.30
    min_profit_factor: float = 1.3
    min_win_rate: float = 0.0
    min_cagr: float = 0.0


class ValidationFilters(BaseModel):
    max_oos_sharpe_decay: float = 0.40
    min_walk_forward_consistency: float = 0.6
    monte_carlo_confidence: float = 0.95
    param_stability_tolerance: float = 0.25


class FiltersConfig(BaseModel):
    fast_reject: FastRejectFilters = FastRejectFilters()
    backtest: BacktestFilters = BacktestFilters()
    validation: ValidationFilters = ValidationFilters()


class GenerationConfig(BaseModel):
    mode: Literal["random", "exhaustive", "guided"] = "random"
    target_count: int = 100000
    max_conditions_per_entry: int = 4
    allow_short: bool = False
    indicator_categories: list[str] = Field(default=["trend", "momentum", "volatility", "volume"])
    multi_timeframe: bool = True


class EvolutionConfig(BaseModel):
    enabled: bool = True
    generations: int = 5
    population_size: int = 500
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    elitism_pct: float = 0.1
    tournament_size: int = 5


class ObjectiveConfig(BaseModel):
    metric: str
    weight: float = 1.0
    direction: Literal["maximize", "minimize"] = "maximize"


class RankingConfig(BaseModel):
    mode: Literal["weighted", "pareto", "constraint", "robustness_first"] = "robustness_first"
    objectives: list[ObjectiveConfig] = Field(
        default_factory=lambda: [
            ObjectiveConfig(metric="sharpe", weight=0.3),
            ObjectiveConfig(metric="sortino", weight=0.2),
            ObjectiveConfig(metric="max_drawdown", weight=0.2, direction="minimize"),
            ObjectiveConfig(metric="profit_factor", weight=0.15),
            ObjectiveConfig(metric="cagr", weight=0.15),
        ]
    )
    export_top_n: int = 20
    export_per_category: int = 5


class ExecutionConfig(BaseModel):
    max_workers: int | None = None
    chunk_size: int = 200
    memory_limit_gb: float = 8.0


class CostModelConfig(BaseModel):
    commission_pct: float = 0.03
    slippage_pct: float = 0.02
    min_commission: float = 20.0


class OutputConfig(BaseModel):
    storage: Literal["csv", "duckdb"] = "csv"
    base_dir: str = "./data/runs"
    export_dir: str = "./data/exports"
    save_all_candidates: bool = True


class MLDatasetConfig(BaseModel):
    source: Literal["ohlcv", "generated_trades"] = "ohlcv"
    features: dict[str, bool] = Field(
        default_factory=lambda: {
            "technical": True,
            "candle": True,
            "volatility": True,
            "volume": True,
            "time": True,
            "session": True,
            "gap": True,
            "rolling": True,
        }
    )
    rolling_windows: list[int] = Field(default_factory=lambda: [5, 10, 20, 50])


class MLLabelsConfig(BaseModel):
    type: Literal["binary", "three_class", "regression", "meta_labeling"] = "binary"
    future_horizon: int = 10
    threshold: float = 0.005


class MLTuningConfig(BaseModel):
    enabled: bool = False
    trials: int = 100
    cv_folds: int = 3
    objectives: list[str] = Field(default_factory=lambda: ["accuracy"])


class MLExplainabilityConfig(BaseModel):
    shap: bool = True
    permutation_importance: bool = True
    feature_importance: bool = True


class MLDeploymentConfig(BaseModel):
    confidence_threshold: float = 0.80


class MachineLearningConfig(BaseModel):
    enabled: bool = False
    dataset: MLDatasetConfig = MLDatasetConfig()
    labels: MLLabelsConfig = MLLabelsConfig()
    models: list[str] = Field(default_factory=lambda: ["lightgbm", "xgboost"])
    tuning: MLTuningConfig = MLTuningConfig()
    explainability: MLExplainabilityConfig = MLExplainabilityConfig()
    deployment: MLDeploymentConfig = MLDeploymentConfig()


class ResearchConfig(BaseModel):
    """Top-level research configuration."""

    name: str = "Unnamed Research"
    description: str = ""
    trading_styles: list[str] = Field(default=["intraday", "swing"])
    style_overrides: dict[str, StyleOverride] = Field(
        default_factory=lambda: {
            "intraday": StyleOverride(
                max_hold_bars=75,
                forced_exit_time="15:15",
                product_type="MIS",
                min_trades=200,
            ),
            "btst": StyleOverride(
                min_hold_bars=50,
                max_hold_bars=150,
                product_type="CNC",
            ),
            "swing": StyleOverride(
                min_hold_bars=10,
                max_hold_bars=360,
                product_type="CNC",
            ),
            "positional": StyleOverride(
                min_hold_bars=5,
                product_type="CNC",
                min_trades=20,
            ),
        }
    )
    data: DataConfig = DataConfig()
    generation: GenerationConfig = GenerationConfig()
    filters: FiltersConfig = FiltersConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    ranking: RankingConfig = RankingConfig()
    execution: ExecutionConfig = ExecutionConfig()
    cost_model: CostModelConfig = CostModelConfig()
    output: OutputConfig = OutputConfig()
    machine_learning: MachineLearningConfig = Field(default_factory=MachineLearningConfig)


def load_config(path: str | Path) -> ResearchConfig:
    """Load and validate a YAML research config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _interpolate_env_vars(raw)
    return ResearchConfig(**raw)


def _interpolate_env_vars(obj):
    """Replace ${VAR_NAME} patterns with environment variable values."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, "")
        return obj
    elif isinstance(obj, dict):
        return {k: _interpolate_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_interpolate_env_vars(v) for v in obj]
    return obj
