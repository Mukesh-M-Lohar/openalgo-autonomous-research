"""Result models for backtest, validation, and rejection tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BacktestResult:
    """Complete backtest metrics for a strategy."""

    strategy_id: str
    net_profit: float = 0.0
    net_profit_pct: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    ulcer_index: float = 0.0
    avg_hold_bars: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    trades: list[dict] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "net_profit": self.net_profit,
            "net_profit_pct": self.net_profit_pct,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_trade_pct": self.avg_trade_pct,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "recovery_factor": self.recovery_factor,
            "expectancy": self.expectancy,
            "ulcer_index": self.ulcer_index,
            "avg_hold_bars": self.avg_hold_bars,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
        }


@dataclass
class ValidationResult:
    """Results from walk-forward, OOS, or robustness testing."""

    strategy_id: str
    walk_forward_score: float = 0.0
    walk_forward_consistency: float = 0.0
    oos_sharpe: float = 0.0
    oos_sharpe_decay: float = 0.0
    oos_profit_factor: float = 0.0
    monte_carlo_score: float = 0.0
    monte_carlo_p5_sharpe: float = 0.0
    param_stability_score: float = 0.0
    param_stability_decay: float = 0.0
    stress_test_score: float = 0.0
    stress_max_drawdown: float = 0.0
    robustness_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "walk_forward_score": self.walk_forward_score,
            "walk_forward_consistency": self.walk_forward_consistency,
            "oos_sharpe": self.oos_sharpe,
            "oos_sharpe_decay": self.oos_sharpe_decay,
            "oos_profit_factor": self.oos_profit_factor,
            "monte_carlo_score": self.monte_carlo_score,
            "monte_carlo_p5_sharpe": self.monte_carlo_p5_sharpe,
            "param_stability_score": self.param_stability_score,
            "param_stability_decay": self.param_stability_decay,
            "stress_test_score": self.stress_test_score,
            "stress_max_drawdown": self.stress_max_drawdown,
            "robustness_score": self.robustness_score,
        }


@dataclass
class RejectionRecord:
    """Tracks why a strategy was rejected and at which stage."""

    strategy_id: str
    stage: str
    rejection_reason: str
    threshold: str
    actual_value: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "stage": self.stage,
            "rejection_reason": self.rejection_reason,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "timestamp": self.timestamp,
        }


@dataclass
class RankedStrategy:
    """A strategy that survived all validation with final scores."""

    strategy_id: str
    backtest: BacktestResult
    validation: ValidationResult
    composite_score: float = 0.0
    rank: int = 0
    category: str = ""
    pareto_front: int = 0
    genome: dict | None = None
    signal_logic: dict | None = None

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "backtest": self.backtest.to_dict(),
            "validation": self.validation.to_dict(),
            "composite_score": self.composite_score,
            "rank": self.rank,
            "category": self.category,
            "pareto_front": self.pareto_front,
            "genome": self.genome,
            "signal_logic": self.signal_logic,
        }
