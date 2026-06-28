"""Performance metrics computation from trade lists."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.models.results import BacktestResult


def compute_metrics(
    strategy_id: str,
    trades: list[dict],
    equity_curve: pd.DataFrame,
    initial_capital: float,
    total_bars: int,
) -> BacktestResult:
    """Compute all performance metrics from a list of trades."""
    if not trades:
        return BacktestResult(strategy_id=strategy_id)

    pnls = np.array([t["pnl_pct"] for t in trades])
    bars_held = np.array([t["bars_held"] for t in trades])
    total_trades = len(trades)

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    winning_trades = len(wins)
    losing_trades = len(losses)

    net_profit_pct = pnls.sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    avg_trade_pct = pnls.mean()
    avg_win_pct = wins.mean() if len(wins) > 0 else 0.0
    avg_loss_pct = losses.mean() if len(losses) > 0 else 0.0
    avg_hold_bars = bars_held.mean()

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Equity curve based metrics
    equity = equity_curve["equity"].values
    final_equity = equity[-1]
    net_profit = final_equity - initial_capital

    # CAGR
    years = total_bars / 252 if total_bars > 252 else max(total_bars / 252, 0.1)
    if final_equity > 0 and initial_capital > 0:
        cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    # Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.where(running_max > 0, running_max, 1)
    max_drawdown_pct = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0.0
    max_drawdown = abs((equity - running_max).min())

    # Returns series for Sharpe/Sortino
    returns = pd.Series(pnls / 100)
    mean_ret = returns.mean()
    std_ret = returns.std()

    # Sharpe (annualized, assume 252 trading days)
    ann_factor = np.sqrt(min(total_trades, 252))
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0

    # Sortino (only downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else std_ret
    sortino = (mean_ret / downside_std * ann_factor) if downside_std > 0 else 0.0

    # Calmar
    calmar = cagr / max_drawdown_pct if max_drawdown_pct > 0 else 0.0

    # Recovery factor
    recovery_factor = net_profit / max_drawdown if max_drawdown > 0 else 0.0

    # Expectancy
    expectancy = (win_rate * avg_win_pct) + ((1 - win_rate) * avg_loss_pct)

    # Ulcer index
    if len(drawdowns) > 0:
        ulcer_index = np.sqrt(np.mean(drawdowns**2)) * 100
    else:
        ulcer_index = 0.0

    # Consecutive wins/losses
    max_consec_wins = _max_consecutive(pnls > 0)
    max_consec_losses = _max_consecutive(pnls < 0)

    return BacktestResult(
        strategy_id=strategy_id,
        net_profit=net_profit,
        net_profit_pct=net_profit_pct,
        cagr=round(cagr, 4),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        calmar=round(calmar, 4),
        profit_factor=round(profit_factor, 4),
        max_drawdown=round(max_drawdown, 2),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        win_rate=round(win_rate, 4),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        avg_trade_pct=round(avg_trade_pct, 4),
        avg_win_pct=round(avg_win_pct, 4),
        avg_loss_pct=round(avg_loss_pct, 4),
        recovery_factor=round(recovery_factor, 4),
        expectancy=round(expectancy, 4),
        ulcer_index=round(ulcer_index, 4),
        avg_hold_bars=round(avg_hold_bars, 2),
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
    )


def _max_consecutive(mask: np.ndarray) -> int:
    """Find max consecutive True values in a boolean array."""
    if len(mask) == 0:
        return 0
    max_count = 0
    count = 0
    for val in mask:
        if val:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count
