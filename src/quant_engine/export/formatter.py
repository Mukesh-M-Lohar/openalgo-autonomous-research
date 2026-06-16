"""Strategy exporter — converts StrategyGenome to standalone signal scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from quant_engine.models.results import BacktestResult, RankedStrategy
from quant_engine.models.strategy import (
    CompareOp,
    CompositeCondition,
    ConditionNode,
    ConditionTree,
    IndicatorNode,
    LogicOp,
    StrategyGenome,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


class StrategyExporter:
    """Exports strategies as Python signal scripts and JSON definitions."""

    def __init__(self, output_dir: str | Path = "./data/exports"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def export_strategy(
        self,
        strategy: StrategyGenome,
        backtest: BacktestResult | None = None,
    ) -> tuple[Path, Path]:
        """Export a strategy as both Python script and JSON.

        Returns (python_path, json_path).
        """
        # Export JSON definition
        json_path = self._output_dir / f"{strategy.id}_strategy.json"
        json_data = {
            "strategy": strategy.to_dict(),
            "backtest_metrics": backtest.to_dict() if backtest else None,
            "signal_logic": self._describe_logic(strategy),
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        # Export Python signal script
        py_path = self._output_dir / f"{strategy.id}_strategy.py"
        script = self._generate_script(strategy, backtest)
        with open(py_path, "w") as f:
            f.write(script)

        logger.info(f"Exported strategy {strategy.id} to {py_path}")
        return py_path, json_path

    def export_batch(
        self,
        ranked: list[RankedStrategy],
        strategies: dict[str, StrategyGenome],
    ) -> list[tuple[Path, Path]]:
        """Export multiple strategies."""
        paths = []
        for r in ranked:
            strategy = strategies.get(r.strategy_id)
            if strategy:
                p = self.export_strategy(strategy, r.backtest)
                paths.append(p)
        return paths

    def _generate_script(self, strategy: StrategyGenome, backtest: BacktestResult | None) -> str:
        """Generate the Python signal script from a template."""
        try:
            template = self._env.get_template("signal_strategy.py.j2")
        except Exception:
            return self._generate_script_inline(strategy, backtest)

        return template.render(
            strategy=strategy,
            backtest=backtest,
            indicators=self._extract_indicator_code(strategy.entry_long),
            entry_conditions=self._condition_to_python(strategy.entry_long),
            exit_logic=self._exit_to_python(strategy.exit_long),
            timeframes=strategy.timeframes_used,
        )

    def _generate_script_inline(self, strategy: StrategyGenome, backtest: BacktestResult | None) -> str:
        """Fallback: generate script without template."""
        indicators = self._extract_indicator_code(strategy.entry_long)
        entry_code = self._condition_to_python(strategy.entry_long)
        exit_code = self._exit_to_python(strategy.exit_long)

        metrics_comment = ""
        if backtest:
            metrics_comment = f"""# Backtest Metrics:
#   Sharpe: {backtest.sharpe}
#   CAGR: {backtest.cagr}%
#   Profit Factor: {backtest.profit_factor}
#   Max Drawdown: {backtest.max_drawdown_pct}%
#   Win Rate: {backtest.win_rate * 100:.1f}%
#   Total Trades: {backtest.total_trades}
"""

        return f'''"""
Auto-generated signal strategy: {strategy.id}
Trading Style: {strategy.trading_style.value.upper()}
Timeframes: {", ".join(tf.value for tf in strategy.timeframes_used)}
Product Type: {strategy.product_type}

{metrics_comment}
NOTE: This is a SIGNAL GENERATOR only. It does NOT place orders.
Wire into OpenAlgo or your execution system separately.
"""

import pandas as pd
import numpy as np


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators needed for this strategy."""
{indicators}
    return df


def generate_entry_signal(df: pd.DataFrame) -> pd.Series:
    """Generate entry (buy) signal. Returns boolean Series."""
    df = compute_indicators(df)
    return {entry_code}


def generate_exit_signal(df: pd.DataFrame, entry_price: float) -> pd.Series:
    """Generate exit signal. Returns boolean Series."""
{exit_code}


def get_strategy_params() -> dict:
    """Return strategy parameters for reference."""
    return {{
        "id": "{strategy.id}",
        "trading_style": "{strategy.trading_style.value}",
        "product_type": "{strategy.product_type}",
        "forced_exit_time": {repr(strategy.forced_exit_time)},
        "stop_loss_pct": {strategy.exit_long.stop_loss_pct},
        "take_profit_pct": {strategy.exit_long.take_profit_pct},
        "trailing_stop_pct": {strategy.exit_long.trailing_stop_pct},
    }}


if __name__ == "__main__":
    print(f"Strategy: {strategy.id}")
    print(f"Style: {strategy.trading_style.value}")
    print("This is a signal generator. Feed OHLCV data to generate_entry_signal().")
'''

    def _extract_indicator_code(self, tree: ConditionTree) -> str:
        """Generate Python code to compute all indicators in the tree."""
        indicators = set()
        self._collect_indicators(tree, indicators)

        lines = []
        for ind in sorted(indicators, key=str):
            code = self._indicator_to_python(ind)
            lines.append(f"    {code}")
        return "\n".join(lines) if lines else "    pass"

    def _collect_indicators(self, tree: ConditionTree, indicators: set) -> None:
        if isinstance(tree, ConditionNode):
            if isinstance(tree.left, IndicatorNode):
                indicators.add(tree.left)
            if isinstance(tree.right, IndicatorNode):
                indicators.add(tree.right)
        elif isinstance(tree, CompositeCondition):
            for child in tree.children:
                self._collect_indicators(child, indicators)

    def _indicator_to_python(self, node: IndicatorNode) -> str:
        """Convert an IndicatorNode to Python computation code."""
        params = node.params_dict
        name = f"ind_{node.indicator_type.value}_{hash(node) % 10000:04d}"

        if node.indicator_type.value in ("sma", "ema", "wma"):
            period = int(params.get("period", 20))
            if node.indicator_type.value == "sma":
                return f'df["{name}"] = df["close"].rolling({period}).mean()'
            elif node.indicator_type.value == "ema":
                return f'df["{name}"] = df["close"].ewm(span={period}, adjust=False).mean()'
            else:
                return f'df["{name}"] = df["close"].rolling({period}).apply(lambda x: np.dot(x, np.arange(1,{period}+1)) / np.arange(1,{period}+1).sum(), raw=True)'
        elif node.indicator_type.value == "rsi":
            period = int(params.get("period", 14))
            return f'df["{name}"] = _compute_rsi(df["close"], {period})'
        elif node.indicator_type.value == "atr":
            period = int(params.get("period", 14))
            return f'df["{name}"] = _compute_atr(df, {period})'
        else:
            return f'df["{name}"] = df["close"]  # {node.indicator_type.value}({params})'

    def _condition_to_python(self, tree: ConditionTree) -> str:
        """Convert condition tree to a Python boolean expression."""
        if isinstance(tree, ConditionNode):
            left = self._value_to_python(tree.left)
            right = self._value_to_python(tree.right)
            if tree.op == CompareOp.GT:
                return f"({left} > {right})"
            elif tree.op == CompareOp.LT:
                return f"({left} < {right})"
            elif tree.op == CompareOp.CROSS_ABOVE:
                return f"(({left} > {right}) & ({left}.shift(1) <= {right}.shift(1) if hasattr({right}, 'shift') else ({left}.shift(1) <= {right})))"
            elif tree.op == CompareOp.CROSS_BELOW:
                return f"(({left} < {right}) & ({left}.shift(1) >= {right}.shift(1) if hasattr({right}, 'shift') else ({left}.shift(1) >= {right})))"
            else:
                return f"({left} > {right})"
        elif isinstance(tree, CompositeCondition):
            parts = [self._condition_to_python(c) for c in tree.children]
            joiner = " & " if tree.logic == LogicOp.AND else " | "
            return f"({joiner.join(parts)})"
        return "pd.Series(False, index=df.index)"

    def _value_to_python(self, node) -> str:
        if isinstance(node, (int, float)):
            return str(node)
        elif isinstance(node, IndicatorNode):
            name = f"ind_{node.indicator_type.value}_{hash(node) % 10000:04d}"
            return f'df["{name}"]'
        return "0"

    def _exit_to_python(self, exit_rule) -> str:
        lines = []
        lines.append("    signals = pd.Series(False, index=df.index)")
        if exit_rule.stop_loss_pct:
            lines.append(f"    # Stop loss: {exit_rule.stop_loss_pct}%")
            lines.append(f'    signals = signals | (df["close"] <= entry_price * (1 - {exit_rule.stop_loss_pct}/100))')
        if exit_rule.take_profit_pct:
            lines.append(f"    # Take profit: {exit_rule.take_profit_pct}%")
            lines.append(f'    signals = signals | (df["close"] >= entry_price * (1 + {exit_rule.take_profit_pct}/100))')
        lines.append("    return signals")
        return "\n".join(lines)

    def _describe_logic(self, strategy: StrategyGenome) -> dict:
        """Human-readable description of strategy logic."""
        return {
            "entry_long": self._describe_tree(strategy.entry_long),
            "exit_long": {
                "stop_loss_pct": strategy.exit_long.stop_loss_pct,
                "take_profit_pct": strategy.exit_long.take_profit_pct,
                "trailing_stop_pct": strategy.exit_long.trailing_stop_pct,
                "max_hold_bars": strategy.exit_long.max_hold_bars,
            },
            "trading_style": strategy.trading_style.value,
            "timeframes": [tf.value for tf in strategy.timeframes_used],
        }

    def _describe_tree(self, tree: ConditionTree) -> str:
        if isinstance(tree, ConditionNode):
            left = self._describe_value(tree.left)
            right = self._describe_value(tree.right)
            return f"{left} {tree.op.value} {right}"
        elif isinstance(tree, CompositeCondition):
            parts = [self._describe_tree(c) for c in tree.children]
            joiner = f" {tree.logic.value.upper()} "
            return f"({joiner.join(parts)})"
        return "unknown"

    def _describe_value(self, node) -> str:
        if isinstance(node, (int, float)):
            return str(node)
        elif isinstance(node, IndicatorNode):
            params_str = ", ".join(f"{k}={v}" for k, v in node.params_dict.items())
            return f"{node.indicator_type.value.upper()}({params_str})@{node.timeframe.value}"
        return "?"
