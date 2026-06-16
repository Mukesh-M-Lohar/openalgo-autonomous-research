"""Combinatorial strategy generator — produces 50K-500K+ candidates."""

from __future__ import annotations

import logging
import random
from typing import Iterator

from quant_engine.config import ResearchConfig, StyleOverride
from quant_engine.generation.grammar import GrammarConfig, generate_strategy
from quant_engine.generation.indicators import INDICATOR_CATEGORIES
from quant_engine.models.strategy import (
    IndicatorType,
    StrategyGenome,
    TimeframeType,
    TradingStyle,
)

logger = logging.getLogger(__name__)

# Timeframes allowed per trading style
STYLE_TIMEFRAMES: dict[TradingStyle, list[TimeframeType]] = {
    TradingStyle.INTRADAY: [TimeframeType.M1, TimeframeType.M5, TimeframeType.M15],
    TradingStyle.BTST: [TimeframeType.M15, TimeframeType.M30, TimeframeType.H1],
    TradingStyle.SWING: [TimeframeType.H1, TimeframeType.H4, TimeframeType.D1],
    TradingStyle.POSITIONAL: [TimeframeType.D1, TimeframeType.W1],
}


class StrategyGenerator:
    """Generates candidate strategies based on research config."""

    def __init__(self, config: ResearchConfig):
        self._config = config
        self._seen_fingerprints: set[str] = set()

    def generate(self) -> list[StrategyGenome]:
        """Generate all candidates, deduplicating by fingerprint."""
        strategies = []
        target = self._config.generation.target_count
        styles = [TradingStyle(s) for s in self._config.trading_styles]
        per_style = target // len(styles)

        for style in styles:
            grammar_config = self._build_grammar_config(style)
            count = 0
            attempts = 0
            max_attempts = per_style * 3

            while count < per_style and attempts < max_attempts:
                strategy = generate_strategy(grammar_config)
                fp = strategy.fingerprint()
                if fp not in self._seen_fingerprints:
                    self._seen_fingerprints.add(fp)
                    strategies.append(strategy)
                    count += 1
                attempts += 1

            logger.info(
                f"Generated {count} strategies for {style.value} "
                f"({attempts} attempts, {attempts - count} duplicates)"
            )

        random.shuffle(strategies)
        logger.info(f"Total generated: {len(strategies)} unique strategies")
        return strategies

    def generate_batch(self, batch_size: int = 1000) -> Iterator[list[StrategyGenome]]:
        """Generate strategies in batches for memory efficiency."""
        target = self._config.generation.target_count
        styles = [TradingStyle(s) for s in self._config.trading_styles]
        generated = 0

        while generated < target:
            batch = []
            for _ in range(min(batch_size, target - generated)):
                style = random.choice(styles)
                grammar_config = self._build_grammar_config(style)
                strategy = generate_strategy(grammar_config)
                fp = strategy.fingerprint()
                if fp not in self._seen_fingerprints:
                    self._seen_fingerprints.add(fp)
                    batch.append(strategy)
            generated += len(batch)
            if batch:
                yield batch

    def _build_grammar_config(self, style: TradingStyle) -> GrammarConfig:
        gen_cfg = self._config.generation
        style_override = self._config.style_overrides.get(style.value, StyleOverride())

        # Resolve allowed indicators from categories
        allowed_indicators = []
        for cat in gen_cfg.indicator_categories:
            allowed_indicators.extend(INDICATOR_CATEGORIES.get(cat, []))
        if not allowed_indicators:
            allowed_indicators = list(IndicatorType)

        # Resolve allowed timeframes
        config_tfs = [TimeframeType(tf) for tf in self._config.data.timeframes]
        style_tfs = STYLE_TIMEFRAMES.get(style, config_tfs)
        allowed_tfs = [tf for tf in style_tfs if tf in config_tfs] or config_tfs[:2]

        return GrammarConfig(
            allowed_indicators=allowed_indicators,
            allowed_timeframes=allowed_tfs,
            trading_style=style,
            max_conditions=gen_cfg.max_conditions_per_entry,
            allow_short=gen_cfg.allow_short,
            product_type=style_override.product_type,
            forced_exit_time=style_override.forced_exit_time,
            max_hold_bars=style_override.max_hold_bars,
            min_hold_bars=style_override.min_hold_bars,
        )

    @property
    def unique_count(self) -> int:
        return len(self._seen_fingerprints)
