"""Core strategy representation — the 'DNA' of every generated strategy."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class IndicatorType(str, Enum):
    SMA = "sma"
    EMA = "ema"
    WMA = "wma"
    VWMA = "vwma"
    RSI = "rsi"
    MACD = "macd"
    MACD_SIGNAL = "macd_signal"
    MACD_HIST = "macd_hist"
    ADX = "adx"
    ATR = "atr"
    BBANDS_UPPER = "bbands_upper"
    BBANDS_MIDDLE = "bbands_middle"
    BBANDS_LOWER = "bbands_lower"
    KELTNER_UPPER = "keltner_upper"
    KELTNER_LOWER = "keltner_lower"
    DONCHIAN_UPPER = "donchian_upper"
    DONCHIAN_LOWER = "donchian_lower"
    SUPERTREND = "supertrend"
    STOCH_K = "stoch_k"
    STOCH_D = "stoch_d"
    CCI = "cci"
    ROC = "roc"
    MOMENTUM = "momentum"
    VWAP = "vwap"
    OBV = "obv"
    VOLUME_SMA = "volume_sma"
    PRICE = "price"


class CompareOp(str, Enum):
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    BETWEEN = "between"
    SLOPE_POS = "slope_pos"
    SLOPE_NEG = "slope_neg"


class LogicOp(str, Enum):
    AND = "and"
    OR = "or"


class TimeframeType(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class TradingStyle(str, Enum):
    INTRADAY = "intraday"
    BTST = "btst"
    SWING = "swing"
    POSITIONAL = "positional"


class PriceSource(str, Enum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    HL2 = "hl2"
    HLC3 = "hlc3"
    OHLC4 = "ohlc4"


@dataclass(frozen=True)
class IndicatorNode:
    """A computed indicator with specific parameters."""

    indicator_type: IndicatorType
    params: tuple[tuple[str, int | float], ...]
    timeframe: TimeframeType
    source: PriceSource = PriceSource.CLOSE

    @property
    def params_dict(self) -> dict[str, int | float]:
        return dict(self.params)

    def with_params(self, **kwargs: int | float) -> IndicatorNode:
        current = dict(self.params)
        current.update(kwargs)
        return IndicatorNode(
            indicator_type=self.indicator_type,
            params=tuple(sorted(current.items())),
            timeframe=self.timeframe,
            source=self.source,
        )

    def to_dict(self) -> dict:
        return {
            "type": "indicator",
            "indicator_type": self.indicator_type.value,
            "params": self.params_dict,
            "timeframe": self.timeframe.value,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> IndicatorNode:
        return cls(
            indicator_type=IndicatorType(d["indicator_type"]),
            params=tuple(sorted(d["params"].items())),
            timeframe=TimeframeType(d["timeframe"]),
            source=PriceSource(d.get("source", "close")),
        )


@dataclass(frozen=True)
class ConditionNode:
    """A comparison between two values."""

    left: Union[IndicatorNode, float]
    op: CompareOp
    right: Union[IndicatorNode, float]

    def to_dict(self) -> dict:
        return {
            "type": "condition",
            "left": self.left.to_dict() if isinstance(self.left, IndicatorNode) else self.left,
            "op": self.op.value,
            "right": (
                self.right.to_dict() if isinstance(self.right, IndicatorNode) else self.right
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConditionNode:
        left = IndicatorNode.from_dict(d["left"]) if isinstance(d["left"], dict) else d["left"]
        right = IndicatorNode.from_dict(d["right"]) if isinstance(d["right"], dict) else d["right"]
        return cls(left=left, op=CompareOp(d["op"]), right=right)


@dataclass(frozen=True)
class CompositeCondition:
    """Logical combination of conditions."""

    logic: LogicOp
    children: tuple[Union[ConditionNode, "CompositeCondition"], ...]

    def to_dict(self) -> dict:
        return {
            "type": "composite",
            "logic": self.logic.value,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CompositeCondition:
        children = []
        for c in d["children"]:
            if c["type"] == "condition":
                children.append(ConditionNode.from_dict(c))
            else:
                children.append(CompositeCondition.from_dict(c))
        return cls(logic=LogicOp(d["logic"]), children=tuple(children))


ConditionTree = Union[ConditionNode, CompositeCondition]


@dataclass(frozen=True)
class ExitRule:
    """Exit logic definition."""

    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    exit_signal: ConditionTree | None = None
    max_hold_bars: int | None = None

    def to_dict(self) -> dict:
        d: dict = {}
        if self.stop_loss_pct is not None:
            d["stop_loss_pct"] = self.stop_loss_pct
        if self.take_profit_pct is not None:
            d["take_profit_pct"] = self.take_profit_pct
        if self.trailing_stop_pct is not None:
            d["trailing_stop_pct"] = self.trailing_stop_pct
        if self.exit_signal is not None:
            d["exit_signal"] = self.exit_signal.to_dict()
        if self.max_hold_bars is not None:
            d["max_hold_bars"] = self.max_hold_bars
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExitRule:
        exit_signal = None
        if "exit_signal" in d:
            sig = d["exit_signal"]
            if sig["type"] == "condition":
                exit_signal = ConditionNode.from_dict(sig)
            else:
                exit_signal = CompositeCondition.from_dict(sig)
        return cls(
            stop_loss_pct=d.get("stop_loss_pct"),
            take_profit_pct=d.get("take_profit_pct"),
            trailing_stop_pct=d.get("trailing_stop_pct"),
            exit_signal=exit_signal,
            max_hold_bars=d.get("max_hold_bars"),
        )


@dataclass
class StrategyGenome:
    """Complete strategy representation — generated, mutated, backtested, exported."""

    trading_style: TradingStyle
    entry_long: ConditionTree
    exit_long: ExitRule
    entry_short: ConditionTree | None = None
    exit_short: ExitRule | None = None
    timeframes_used: tuple[TimeframeType, ...] = (TimeframeType.D1,)
    product_type: str = "MIS"
    forced_exit_time: str | None = None
    position_size_pct: float = 100.0
    max_positions: int = 1
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    generation: int = 0
    parent_ids: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "trading_style": self.trading_style.value,
            "entry_long": self.entry_long.to_dict(),
            "exit_long": self.exit_long.to_dict(),
            "timeframes_used": [tf.value for tf in self.timeframes_used],
            "product_type": self.product_type,
            "position_size_pct": self.position_size_pct,
            "max_positions": self.max_positions,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
        }
        if self.entry_short is not None:
            d["entry_short"] = self.entry_short.to_dict()
        if self.exit_short is not None:
            d["exit_short"] = self.exit_short.to_dict()
        if self.forced_exit_time is not None:
            d["forced_exit_time"] = self.forced_exit_time
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StrategyGenome:
        def parse_condition(c: dict) -> ConditionTree:
            if c["type"] == "condition":
                return ConditionNode.from_dict(c)
            return CompositeCondition.from_dict(c)

        entry_short = None
        if "entry_short" in d:
            entry_short = parse_condition(d["entry_short"])

        exit_short = None
        if "exit_short" in d:
            exit_short = ExitRule.from_dict(d["exit_short"])

        return cls(
            id=d["id"],
            name=d.get("name", ""),
            trading_style=TradingStyle(d["trading_style"]),
            entry_long=parse_condition(d["entry_long"]),
            exit_long=ExitRule.from_dict(d["exit_long"]),
            entry_short=entry_short,
            exit_short=exit_short,
            timeframes_used=tuple(TimeframeType(tf) for tf in d["timeframes_used"]),
            product_type=d.get("product_type", "MIS"),
            forced_exit_time=d.get("forced_exit_time"),
            position_size_pct=d.get("position_size_pct", 100.0),
            max_positions=d.get("max_positions", 1),
            generation=d.get("generation", 0),
            parent_ids=tuple(d.get("parent_ids", ())),
        )

    def describe(self) -> str:
        """Human-readable summary of the strategy."""
        parts = [f"[{self.trading_style.value.upper()}]"]
        parts.append(f"TF: {','.join(tf.value for tf in self.timeframes_used)}")
        if self.forced_exit_time:
            parts.append(f"Exit@{self.forced_exit_time}")
        return " | ".join(parts)
