"""Cost model — commission, slippage, and market impact calculations."""

from __future__ import annotations

from quant_engine.config import CostModelConfig


class CostModel:
    """Applies trading costs to backtest results."""

    def __init__(self, config: CostModelConfig):
        self.commission_pct = config.commission_pct
        self.slippage_pct = config.slippage_pct
        self.min_commission = config.min_commission

    def round_trip_cost_pct(self) -> float:
        """Total cost for entry + exit."""
        return (self.commission_pct + self.slippage_pct) * 2

    def apply_to_trade(self, pnl_pct: float) -> float:
        """Apply costs to a trade's raw PnL percentage."""
        return pnl_pct - self.round_trip_cost_pct()

    def stress_cost(self, multiplier: float = 2.0) -> "CostModel":
        """Return a stressed cost model with higher costs for robustness testing."""
        stressed_config = CostModelConfig(
            commission_pct=self.commission_pct * multiplier,
            slippage_pct=self.slippage_pct * multiplier,
            min_commission=self.min_commission,
        )
        return CostModel(stressed_config)
