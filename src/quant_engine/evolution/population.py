"""Population management for evolutionary strategy optimization."""

from __future__ import annotations

import logging
import random

from quant_engine.config import EvolutionConfig
from quant_engine.evolution.crossover import Crossover
from quant_engine.evolution.fitness import weighted_fitness
from quant_engine.evolution.mutator import Mutator
from quant_engine.generation.grammar import GrammarConfig
from quant_engine.models.results import BacktestResult, ValidationResult
from quant_engine.models.strategy import StrategyGenome

logger = logging.getLogger(__name__)


class Population:
    """Manages a population of strategies through evolutionary generations."""

    def __init__(self, config: EvolutionConfig, grammar_config: GrammarConfig | None = None):
        self._config = config
        self._mutator = Mutator(mutation_rate=config.mutation_rate, grammar_config=grammar_config)
        self._crossover = Crossover(crossover_rate=config.crossover_rate)
        self._members: list[tuple[StrategyGenome, float]] = []

    @property
    def size(self) -> int:
        return len(self._members)

    def initialize(
        self,
        strategies: list[StrategyGenome],
        fitness_scores: list[float],
    ) -> None:
        """Initialize population with strategies and their fitness scores."""
        self._members = list(zip(strategies, fitness_scores))
        self._members.sort(key=lambda x: x[1], reverse=True)
        # Keep top N
        self._members = self._members[: self._config.population_size]
        logger.info(f"Population initialized with {len(self._members)} members")

    def evolve(self) -> list[StrategyGenome]:
        """Generate next generation via selection, crossover, and mutation."""
        if not self._members:
            return []

        pop_size = self._config.population_size
        elite_count = max(1, int(pop_size * self._config.elitism_pct))

        # Elitism: keep top performers unchanged
        offspring = [s for s, _ in self._members[:elite_count]]

        # Fill rest via tournament selection + crossover + mutation
        while len(offspring) < pop_size:
            parent_a = self._tournament_select()
            parent_b = self._tournament_select()

            child = self._crossover.cross(parent_a, parent_b)
            child = self._mutator.mutate(child)
            offspring.append(child)

        logger.info(
            f"Evolved generation: {elite_count} elite + "
            f"{pop_size - elite_count} offspring = {len(offspring)} total"
        )
        return offspring

    def update_fitness(
        self,
        strategies: list[StrategyGenome],
        fitness_scores: list[float],
    ) -> None:
        """Update population with new fitness evaluations."""
        self._members = list(zip(strategies, fitness_scores))
        self._members.sort(key=lambda x: x[1], reverse=True)
        self._members = self._members[: self._config.population_size]

    def get_top(self, n: int = 10) -> list[StrategyGenome]:
        """Get top N strategies by fitness."""
        return [s for s, _ in self._members[:n]]

    def _tournament_select(self) -> StrategyGenome:
        """Select a parent via tournament selection."""
        tournament_size = min(self._config.tournament_size, len(self._members))
        contestants = random.sample(self._members, tournament_size)
        winner = max(contestants, key=lambda x: x[1])
        return winner[0]
