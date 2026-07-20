"""Backtest engine — translates StrategyGenome into signals and runs vectorized simulation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_engine.backtest.metrics import compute_metrics
from quant_engine.config import CostModelConfig
from quant_engine.generation.indicators import compute_indicator
from quant_engine.models.results import BacktestResult
from quant_engine.models.strategy import (
    CompareOp,
    CompositeCondition,
    ConditionNode,
    ConditionTree,
    ExitRule,
    IndicatorNode,
    LogicOp,
    StrategyGenome,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Vectorized backtesting engine using pure NumPy/Pandas."""

    def __init__(
        self,
        cost_model: CostModelConfig | None = None,
        initial_capital: float = 100000.0,
    ):
        self._cost = cost_model or CostModelConfig()
        self._initial_capital = initial_capital

    def run(
        self,
        strategy: StrategyGenome,
        data: dict[str, pd.DataFrame],
    ) -> BacktestResult | None:
        """Run backtest for a single strategy.

        Args:
            strategy: The strategy genome to test.
            data: Dict of {timeframe: OHLCV DataFrame}
        """
        try:
            primary_tf = strategy.timeframes_used[0].value
            if primary_tf not in data:
                return None

            df = data[primary_tf]
            if df.empty or len(df) < 50:
                return None

            entries = self._evaluate_entry(strategy.entry_long, df, data)
            exits = self._evaluate_exit(strategy.exit_long, df, data)

            if entries is None or exits is None:
                return None

            entries = entries.fillna(False).astype(bool)
            exits = exits.fillna(False).astype(bool)

            if entries.sum() == 0:
                return None

            trades, bar_equity = self._simulate_trades(
                df, entries, exits, strategy.exit_long, strategy.forced_exit_time
            )

            if not trades:
                return None

            equity_curve = pd.DataFrame({"equity": bar_equity}, index=df.index)

            # Buy-and-hold benchmark for information ratio (Chan Ch.1 p.41)
            bh_returns = df["close"].pct_change().fillna(0.0)

            return compute_metrics(
                strategy_id=strategy.id,
                trades=trades,
                equity_curve=equity_curve,
                initial_capital=self._initial_capital,
                total_bars=len(df),
                benchmark_returns=bh_returns,
            )
        except Exception as e:
            logger.debug(f"Backtest failed for {strategy.id}: {e}")
            return None

    def run_batch(
        self,
        strategies: list[StrategyGenome],
        data: dict[str, pd.DataFrame],
    ) -> list[BacktestResult]:
        """Run backtests for a batch of strategies."""
        results = []
        for s in strategies:
            result = self.run(s, data)
            if result is not None:
                results.append(result)
        return results

    def _evaluate_entry(
        self, tree: ConditionTree, df: pd.DataFrame, all_data: dict[str, pd.DataFrame]
    ) -> pd.Series | None:
        """Evaluate entry condition tree to a boolean Series."""
        try:
            return self._eval_condition_tree(tree, df, all_data)
        except Exception:
            return None

    def _evaluate_exit(
        self, exit_rule: ExitRule, df: pd.DataFrame, all_data: dict[str, pd.DataFrame]
    ) -> pd.Series | None:
        """Evaluate exit signal if present."""
        if exit_rule.exit_signal is not None:
            try:
                return self._eval_condition_tree(exit_rule.exit_signal, df, all_data)
            except Exception:
                pass
        return pd.Series(False, index=df.index)

    def _eval_condition_tree(
        self, tree: ConditionTree, df: pd.DataFrame, all_data: dict[str, pd.DataFrame]
    ) -> pd.Series:
        if isinstance(tree, ConditionNode):
            return self._eval_condition(tree, df, all_data)
        elif isinstance(tree, CompositeCondition):
            results = [self._eval_condition_tree(c, df, all_data) for c in tree.children]
            if tree.logic == LogicOp.AND:
                combined = results[0]
                for r in results[1:]:
                    combined = combined & r
                return combined
            else:
                combined = results[0]
                for r in results[1:]:
                    combined = combined | r
                return combined
        return pd.Series(False, index=df.index)

    def _eval_condition(
        self, cond: ConditionNode, df: pd.DataFrame, all_data: dict[str, pd.DataFrame]
    ) -> pd.Series:
        left_series = self._resolve_value(cond.left, df, all_data)
        right_series = self._resolve_value(cond.right, df, all_data)

        if cond.op == CompareOp.GT:
            return left_series > right_series
        elif cond.op == CompareOp.LT:
            return left_series < right_series
        elif cond.op == CompareOp.GTE:
            return left_series >= right_series
        elif cond.op == CompareOp.LTE:
            return left_series <= right_series
        elif cond.op == CompareOp.CROSS_ABOVE:
            return (left_series > right_series) & (left_series.shift(1) <= right_series.shift(1))
        elif cond.op == CompareOp.CROSS_BELOW:
            return (left_series < right_series) & (left_series.shift(1) >= right_series.shift(1))
        elif cond.op == CompareOp.SLOPE_POS:
            return left_series.diff() > 0
        elif cond.op == CompareOp.SLOPE_NEG:
            return left_series.diff() < 0
        return pd.Series(False, index=df.index)

    def _resolve_value(
        self, node: IndicatorNode | float, df: pd.DataFrame, all_data: dict[str, pd.DataFrame]
    ) -> pd.Series:
        if isinstance(node, (int, float)):
            return pd.Series(node, index=df.index)

        target_df = all_data.get(node.timeframe.value, df)

        series = compute_indicator(target_df, node.indicator_type, node.params_dict, node.source)

        if len(series) != len(df) and node.timeframe.value != list(all_data.keys())[0]:
            series = series.reindex(df.index, method="ffill")

        return series

    def _simulate_trades(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        exit_rule: ExitRule,
        forced_exit_time: str | None,
    ) -> tuple[list[dict], np.ndarray]:
        """Simulate trades with stop-loss, take-profit, trailing stop, and time exit.

        Returns:
            Tuple of (trades list, bar-by-bar equity array).

        Look-ahead bias prevention (Chan Ch.1 p.22):
            Signal at bar i -> entry at open[i+1].
        Bar-by-bar equity (Chan Ch.1 pp.34-40):
            Mark-to-market at every bar for accurate Sharpe/drawdown.
        """

        n = len(df)
        trades = []
        position_open = False
        entry_price = 0.0
        entry_bar = 0
        max_price_since_entry = 0.0
        bars_held = 0

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_ = df["open"].values
        index = df.index

        # Bar-by-bar equity tracking (Chan Ch.1 pp.34-40)
        capital = self._initial_capital
        bar_equity = np.full(n, self._initial_capital, dtype=np.float64)
        position_size = 0.0  # number of "units" held (capital / entry_price)

        # Pre-compute round-trip cost percentages
        commission_rt = self._cost.commission_pct * 2
        slippage_rt = self._cost.slippage_pct * 2

        for i in range(1, n):
            if position_open:
                bars_held += 1
                current_high = high[i]
                current_low = low[i]
                current_close = close[i]
                max_price_since_entry = max(max_price_since_entry, current_high)

                exit_price = None
                exit_reason = ""

                # Check stop loss
                if exit_rule.stop_loss_pct is not None:
                    sl_price = entry_price * (1 - exit_rule.stop_loss_pct / 100)
                    if current_low <= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"

                # Check take profit
                if exit_price is None and exit_rule.take_profit_pct is not None:
                    tp_price = entry_price * (1 + exit_rule.take_profit_pct / 100)
                    if current_high >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"

                # Check trailing stop
                if exit_price is None and exit_rule.trailing_stop_pct is not None:
                    trail_price = max_price_since_entry * (1 - exit_rule.trailing_stop_pct / 100)
                    if current_low <= trail_price:
                        exit_price = trail_price
                        exit_reason = "trailing_stop"

                # Check max hold bars
                if exit_price is None and exit_rule.max_hold_bars is not None:
                    if bars_held >= exit_rule.max_hold_bars:
                        exit_price = current_close
                        exit_reason = "max_hold"

                # Check forced exit time (intraday)
                if exit_price is None and forced_exit_time is not None:
                    if hasattr(index[i], "hour"):
                        parts = forced_exit_time.split(":")
                        exit_hour, exit_min = int(parts[0]), int(parts[1])
                        if index[i].hour > exit_hour or (
                            index[i].hour == exit_hour and index[i].minute >= exit_min
                        ):
                            exit_price = current_close
                            exit_reason = "forced_time_exit"

                # Check exit signal
                if exit_price is None and exits.iloc[i]:
                    exit_price = current_close
                    exit_reason = "signal"

                if exit_price is not None:
                    # Close trade
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    net_pnl_pct = pnl_pct - commission_rt - slippage_rt

                    trades.append(
                        {
                            "entry_time": index[entry_bar],
                            "exit_time": index[i],
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl_pct": net_pnl_pct,
                            "bars_held": bars_held,
                            "exit_reason": exit_reason,
                        }
                    )
                    # Update capital with realized PnL
                    capital = capital * (1 + net_pnl_pct / 100)
                    position_open = False
                    position_size = 0.0
                    bar_equity[i] = capital
                else:
                    # Still in trade: mark-to-market using current close
                    unrealized_pnl_pct = (current_close - entry_price) / entry_price
                    bar_equity[i] = capital * (1 + unrealized_pnl_pct)

            else:
                # No position — check for entry signal
                # Look-ahead bias fix (Chan Ch.1 p.22):
                # Signal at bar i -> entry at open[i+1]
                # We check signal at bar i-1 and enter at open[i]
                if i >= 2 and entries.iloc[i - 1] and not position_open:
                    entry_price = open_[i]
                    entry_bar = i
                    max_price_since_entry = high[i]
                    bars_held = 0
                    position_open = True
                    position_size = capital / entry_price
                    # Mark-to-market at this bar's close
                    unrealized_pnl_pct = (close[i] - entry_price) / entry_price
                    bar_equity[i] = capital * (1 + unrealized_pnl_pct)
                else:
                    bar_equity[i] = capital

        # Close any open position at end
        if position_open:
            exit_price = close[-1]
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            net_pnl_pct = pnl_pct - commission_rt - slippage_rt
            trades.append(
                {
                    "entry_time": index[entry_bar],
                    "exit_time": index[-1],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": net_pnl_pct,
                    "bars_held": bars_held,
                    "exit_reason": "end_of_data",
                }
            )
            capital = capital * (1 + net_pnl_pct / 100)
            bar_equity[-1] = capital

        return trades, bar_equity
