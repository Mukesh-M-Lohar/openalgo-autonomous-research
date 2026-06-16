"""Pareto frontier computation — NSGA-II style non-dominated sorting."""

from __future__ import annotations


def compute_pareto_fronts(points: list[list[float]]) -> list[list[int]]:
    """Compute Pareto fronts via non-dominated sorting.

    All objectives are maximized (caller negates minimize objectives).

    Returns list of fronts, each front is a list of indices.
    Front 0 = Pareto-optimal, Front 1 = next best, etc.
    """
    n = len(points)
    if n == 0:
        return []

    domination_count = [0] * n
    dominated_set: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = []

    # For each pair, determine domination
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(points[i], points[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _dominates(points[j], points[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

    # First front: non-dominated solutions
    current_front = [i for i in range(n) if domination_count[i] == 0]
    fronts.append(current_front)

    # Subsequent fronts
    while current_front:
        next_front = []
        for i in current_front:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current_front = next_front

    return fronts


def _dominates(a: list[float], b: list[float]) -> bool:
    """Check if solution a dominates solution b (all objectives maximized)."""
    at_least_one_better = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            at_least_one_better = True
    return at_least_one_better
