"""Progress tracking for long-running pipeline stages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageProgress:
    """Tracks progress of a pipeline stage."""

    stage_name: str
    total: int = 0
    completed: int = 0
    passed: int = 0
    rejected: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def pct(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        if self.elapsed == 0:
            return 0.0
        return self.completed / self.elapsed

    @property
    def eta_seconds(self) -> float:
        if self.rate == 0:
            return 0.0
        remaining = self.total - self.completed
        return remaining / self.rate

    def to_dict(self) -> dict:
        return {
            "stage": self.stage_name,
            "total": self.total,
            "completed": self.completed,
            "passed": self.passed,
            "rejected": self.rejected,
            "pct": round(self.pct, 1),
            "elapsed_sec": round(self.elapsed, 1),
            "rate_per_sec": round(self.rate, 1),
            "eta_sec": round(self.eta_seconds, 1),
        }


@dataclass
class RunProgress:
    """Tracks overall research run progress."""

    run_id: str
    stages: dict[str, StageProgress] = field(default_factory=dict)
    current_stage: str = ""
    status: str = "running"

    def start_stage(self, name: str, total: int) -> StageProgress:
        progress = StageProgress(stage_name=name, total=total)
        self.stages[name] = progress
        self.current_stage = name
        return progress

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }
