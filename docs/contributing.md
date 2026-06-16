# Contributing

## Getting Started

1. Fork the repository
2. Run the setup script: `make setup` or `./scripts/setup-dev.sh`
3. Create a feature branch: `git checkout -b feat/my-feature`
4. Make your changes
5. Run tests: `make test`
6. Run linter: `make lint`
7. Commit and push
8. Open a Pull Request

## Pull Request Requirements

- All tests pass (`make test`)
- No lint errors (`make lint`)
- New features include tests
- Breaking changes documented in PR description

## Areas to Contribute

### Beginner-Friendly
- Add new technical indicators (RSI variants, Williams %R, etc.)
- Add new candlestick patterns (Doji, Hammer, Morning Star, etc.)
- Improve error messages and logging
- Add more example configs for different markets

### Intermediate
- Implement DuckDB storage backend
- Add more mutation operators to evolution engine
- Improve export templates (multi-timeframe, advanced exits)
- Add performance benchmarks

### Advanced
- Implement genetic programming (typed GP with tree depth control)
- Add portfolio-level optimization (correlation-based, risk parity)
- Distributed execution backend (Ray/Dask integration)
- Walk-forward optimization (not just validation)
- Regime detection for adaptive strategies

## Architecture Decisions

If your change affects architecture:
1. Update `docs/low-level-design.md`
2. Ensure backward compatibility or document the migration
3. Add an entry to the changelog
