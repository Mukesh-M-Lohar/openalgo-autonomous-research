"""Monte Carlo simulation — randomize trade sequences to estimate confidence intervals."""

from __future__ import annotations

import logging

import numpy as np

from quant_engine.models.results import BacktestResult

logger = logging.getLogger(__name__)


class MonteCarloValidator:
    """Tests strategy robustness via Monte Carlo simulations."""

    def __init__(self, n_simulations: int = 1000, confidence_level: float = 0.95):
        self._n_sims = n_simulations
        self._confidence = confidence_level

    def validate(self, backtest_result: BacktestResult, trades: list[dict]) -> dict:
        """Run Monte Carlo simulations on the trade sequence.

        Shuffles trade order to test if results are sequence-dependent.
        Adds random slippage/variation to test fragility.

        Returns dict with monte_carlo_score, confidence intervals.
        """
        if not trades or len(trades) < 10:
            return {
                "monte_carlo_score": 0.0,
                "monte_carlo_p5_sharpe": 0.0,
                "monte_carlo_median_pnl": 0.0,
                "monte_carlo_p5_drawdown": 0.0,
            }

        pnls = np.array([t["pnl_pct"] for t in trades])
        n_trades = len(pnls)

        # Simulation results
        sim_total_pnls = np.zeros(self._n_sims)
        sim_max_drawdowns = np.zeros(self._n_sims)
        sim_sharpes = np.zeros(self._n_sims)

        rng = np.random.default_rng(42)

        for i in range(self._n_sims):
            # Shuffle trade order
            shuffled = rng.permutation(pnls)

            # Add small random noise (simulating execution variance)
            noise = rng.normal(0, 0.02, n_trades)
            noisy_pnls = shuffled + noise

            # Compute equity curve
            equity = np.cumprod(1 + noisy_pnls / 100)
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / np.where(running_max > 0, running_max, 1)

            sim_total_pnls[i] = noisy_pnls.sum()
            sim_max_drawdowns[i] = abs(drawdowns.min()) * 100
            mean_ret = noisy_pnls.mean()
            std_ret = noisy_pnls.std()
            sim_sharpes[i] = (mean_ret / std_ret * np.sqrt(min(n_trades, 252))) if std_ret > 0 else 0

        # Confidence intervals
        p5_idx = int(self._n_sims * 0.05)
        sorted_pnls = np.sort(sim_total_pnls)
        sorted_sharpes = np.sort(sim_sharpes)
        sorted_dd = np.sort(sim_max_drawdowns)

        p5_pnl = sorted_pnls[p5_idx]
        p5_sharpe = sorted_sharpes[p5_idx]
        p95_drawdown = sorted_dd[int(self._n_sims * 0.95)]
        median_pnl = np.median(sim_total_pnls)

        # Score: % of simulations that are profitable
        profitable_sims = (sim_total_pnls > 0).sum() / self._n_sims
        positive_sharpe_sims = (sim_sharpes > 0).sum() / self._n_sims

        score = (profitable_sims * 50 + positive_sharpe_sims * 50)

        return {
            "monte_carlo_score": round(score, 2),
            "monte_carlo_p5_sharpe": round(p5_sharpe, 4),
            "monte_carlo_median_pnl": round(median_pnl, 4),
            "monte_carlo_p5_drawdown": round(p95_drawdown, 4),
            "monte_carlo_profitable_pct": round(profitable_sims * 100, 2),
            "monte_carlo_p5_pnl": round(p5_pnl, 4),
        }
