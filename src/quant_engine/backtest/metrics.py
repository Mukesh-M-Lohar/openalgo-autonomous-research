"""Performance metrics computation from trade lists.

Metrics follow Ernie Chan's 'Algorithmic Trading' (2013, Wiley):
- Sharpe/Sortino from bar-by-bar equity returns (Ch.1 pp.34-40)
- Drawdown duration (Ch.1 p.40)
- Kelly criterion (Ch.8 pp.190-195)
- Deflated Sharpe ratio for multiple testing (Ch.1 pp.22-25)
- Information ratio vs buy-and-hold (Ch.1 p.41)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import special  # for erfinv in deflated Sharpe

from quant_engine.models.results import BacktestResult


def compute_metrics(
    strategy_id: str,
    trades: list[dict],
    equity_curve: pd.DataFrame,
    initial_capital: float,
    total_bars: int,
    benchmark_returns: pd.Series | None = None,
    n_strategies_tested: int = 1,
) -> BacktestResult:
    """Compute all performance metrics from a list of trades.

    Args:
        strategy_id: Unique strategy identifier.
        trades: List of trade dicts with 'pnl_pct', 'bars_held' keys.
        equity_curve: DataFrame with 'equity' column, bar-by-bar mark-to-market.
        initial_capital: Starting capital.
        total_bars: Total number of bars in the data.
        benchmark_returns: Buy-and-hold per-bar returns for information ratio.
        n_strategies_tested: Number of strategies tested (for deflated Sharpe).
    """
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

    # --- Drawdown analysis (Chan Ch.1 p.40) ---
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.where(running_max > 0, running_max, 1)
    max_drawdown_pct = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0.0
    max_drawdown = abs((equity - running_max).min())

    # Drawdown duration (Chan: "Do not bother to backtest strategies with
    # a maximum drawdown duration longer than what you can endure." p.40)
    dd_durations = _compute_drawdown_durations(equity)
    max_drawdown_duration_bars = max(dd_durations) if dd_durations else 0
    avg_drawdown_duration_bars = float(np.mean(dd_durations)) if dd_durations else 0.0

    # --- Sharpe from equity curve returns (Chan Ch.1 pp.34-40) ---
    # Per Chan: Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
    # Using bar-by-bar equity returns instead of per-trade PnL
    equity_returns = pd.Series(equity).pct_change().dropna()
    # Replace inf/nan from zero-equity bars
    equity_returns = equity_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    mean_ret = equity_returns.mean()
    std_ret = equity_returns.std()

    # Annualization: sqrt(bars_per_year). For daily = 252, 15min ~= 252*26
    # We use total_bars/years to estimate bars_per_year from the data
    bars_per_year = total_bars / years if years > 0 else 252
    ann_factor = np.sqrt(bars_per_year)

    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0.0

    # --- Sortino from equity curve (downside deviation only) ---
    downside_returns = equity_returns[equity_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else std_ret
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

    # --- Kelly criterion (Chan Ch.8 pp.190-195) ---
    # f* = μ / σ² where μ = mean return, σ² = variance of returns
    # Chan recommends half-Kelly in practice
    kelly_fraction, half_kelly, optimal_leverage = _compute_kelly(equity_returns)

    # --- Deflated Sharpe ratio (Chan Ch.1 pp.22-25, Bailey & Lopez de Prado 2014) ---
    deflated_sharpe = _compute_deflated_sharpe(
        sharpe, total_bars, n_strategies_tested, equity_returns
    )

    # --- Information ratio vs buy-and-hold (Chan Ch.1 p.41) ---
    information_ratio = _compute_information_ratio(equity_returns, benchmark_returns)

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
        max_drawdown_duration_bars=max_drawdown_duration_bars,
        avg_drawdown_duration_bars=round(avg_drawdown_duration_bars, 2),
        kelly_fraction=round(kelly_fraction, 6),
        half_kelly_fraction=round(half_kelly, 6),
        optimal_leverage=round(optimal_leverage, 4),
        deflated_sharpe=round(deflated_sharpe, 4),
        information_ratio=round(information_ratio, 4),
        trades=trades,
    )


def _compute_drawdown_durations(equity: np.ndarray) -> list[int]:
    """Compute the duration (in bars) of each drawdown period.

    A drawdown period starts when equity drops below its running max
    and ends when equity recovers to a new high.
    (Chan Ch.1 p.40: drawdown duration is a key disqualifier.)
    """
    running_max = np.maximum.accumulate(equity)
    in_drawdown = equity < running_max

    durations = []
    current_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0
    # If still in drawdown at end
    if current_duration > 0:
        durations.append(current_duration)

    return durations


def _compute_kelly(equity_returns: pd.Series) -> tuple[float, float, float]:
    """Compute Kelly fraction for optimal position sizing.

    Chan Ch.8 pp.190-195:
    f* = μ / σ²  (for continuously compounded returns)
    Half-Kelly is the practical recommendation.

    Returns:
        Tuple of (kelly_fraction, half_kelly_fraction, optimal_leverage).
    """
    mean_ret = equity_returns.mean()
    var_ret = equity_returns.var()

    if var_ret > 0 and mean_ret > 0:
        kelly = mean_ret / var_ret
    elif var_ret > 0 and mean_ret <= 0:
        kelly = mean_ret / var_ret  # negative Kelly = don't trade
    else:
        kelly = 0.0

    half_kelly = kelly / 2.0
    # Optimal leverage is Kelly fraction (for a single strategy)
    optimal_leverage = max(0.0, kelly)

    return kelly, half_kelly, optimal_leverage


def _compute_deflated_sharpe(
    observed_sharpe: float,
    n_bars: int,
    n_strategies: int,
    equity_returns: pd.Series,
) -> float:
    """Compute the deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

    Chan Ch.1 pp.22-25: Data-snooping bias increases with the number of
    strategies tested. The deflated Sharpe adjusts for multiple testing.

    DSR = (SR_observed - SR_benchmark) / SE(SR)

    where SR_benchmark = sqrt(2) * erfinv(1 - 1/N) estimates the expected
    maximum Sharpe from N random strategies.
    """
    if n_strategies <= 1:
        return observed_sharpe

    n = len(equity_returns)
    if n < 2:
        return 0.0

    # Expected maximum Sharpe from N random strategies (Euler-Mascheroni approx)
    # SR_benchmark ≈ sqrt(2 * ln(N)) for large N
    try:
        p = 1.0 - 1.0 / n_strategies
        p = min(p, 1.0 - 1e-15)  # clamp for erfinv stability
        sr_benchmark = math.sqrt(2) * special.erfinv(p)
    except (ValueError, OverflowError):
        sr_benchmark = math.sqrt(2 * math.log(max(n_strategies, 2)))

    # Standard error of the Sharpe ratio
    # SE(SR) ≈ sqrt((1 + 0.5*SR²) / n), adjusted for skew and kurtosis
    skew = float(equity_returns.skew()) if len(equity_returns) > 2 else 0.0
    kurt = float(equity_returns.kurtosis()) if len(equity_returns) > 3 else 0.0
    sr = observed_sharpe

    se_sr_sq = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / n
    se_sr = math.sqrt(max(se_sr_sq, 1e-10))

    deflated = (sr - sr_benchmark) / se_sr if se_sr > 0 else 0.0
    return deflated


def _compute_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> float:
    """Compute information ratio = mean(excess) / std(excess).

    Chan Ch.1 p.41: "The appropriate benchmark of a long-only strategy
    is the return of a buy-and-hold position — the information ratio
    rather than the Sharpe ratio."
    """
    if benchmark_returns is None or len(benchmark_returns) < 2:
        return 0.0

    # Align lengths
    min_len = min(len(strategy_returns), len(benchmark_returns))
    strat = strategy_returns.iloc[:min_len].values
    bench = benchmark_returns.iloc[:min_len].values

    excess = strat - bench
    std_excess = np.std(excess)

    if std_excess > 0:
        return float(np.mean(excess) / std_excess * np.sqrt(252))
    return 0.0


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
