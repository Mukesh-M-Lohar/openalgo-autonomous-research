# OpenAlgo Quant Research Engine

A standalone, fully offline quantitative research microservice that autonomously discovers, evaluates, validates, ranks, and exports deployable trading strategies.

## What It Does

1. **Generates** 50K–500K candidate strategies from indicator/pattern combinations
2. **Rejects** structurally invalid strategies before expensive backtesting
3. **Backtests** survivors with vectorized simulation (multi-core)
4. **Validates** via walk-forward, out-of-sample, Monte Carlo, and stress tests
5. **Evolves** top performers through mutation and crossover
6. **Ranks** using Pareto-optimal multi-objective scoring
7. **Exports** as standalone Python signal scripts

## Key Properties

- **Fully Offline** — No LLM, AI, cloud, or internet dependency
- **Signal Only** — No order placement. Outputs signal logic + backtest results
- **Transparent** — Every rejection tracked with reason, threshold, and actual value
- **Multi-Style** — Intraday, BTST, Swing, and Positional strategies
- **Robust-First** — Prioritizes consistency over raw profit

## Quick Links

- [Installation & Setup](getting-started.md)
- [Developer Guide](developer-guide.md)
- [API Reference](api/api-reference.md)
- [Low-Level Design](low-level-design.md)
