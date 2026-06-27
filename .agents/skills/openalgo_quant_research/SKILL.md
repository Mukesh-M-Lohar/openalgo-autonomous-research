---
name: OpenAlgo Quant Research
description: Guides and commands to retrieve historical data, execute tests/backtests, and run evolutionary strategy generation/optimization in the OpenAlgo Autonomous Quant Research engine.
---

# OpenAlgo Quant Research Guide

This workspace skill provides instructions and references on how to work with the **OpenAlgo Autonomous Quant Research Engine**, covering data fetching, vectorized backtesting, and automated strategy generation/optimization.

---

## 1. Reading and Retrieving Data

The engine fetches historical OHLCV data using the OpenAlgo REST API via [OpenAlgoClient](file:///root/openalgo-autonomous-research/src/quant_engine/data/client.py).

### Historical Data Client
* **API Details**: Configured in YAML via the `data.openalgo` block. Reads API keys, host, and data source (`api` or `db`).
* **Caching**: Fetched data is cached locally under `./data/cache/` to minimize rate-limiting blocks and speed up repeat runs.
* **Format**: All fetched data is loaded into `pandas.DataFrame` where:
  - The index is a `DatetimeIndex` named `timestamp`.
  - Columns are: `open`, `high`, `low`, `close`, `volume` (and optionally `oi` for derivatives).
  - All columns are parsed to numeric datatypes.

### Commands to Manually Fetch/Prepare Data
To run autonomous loops or prepare offline data:
* Use the data prep script:
  ```bash
  python3 autoresearch/prepare.py
  ```

---

## 2. Testing and Backtesting

Backtesting evaluates strategy rules against historical data using a vectorized simulation engine.

### Core Backtest Engine
* **Engine File**: [BacktestEngine](file:///root/openalgo-autonomous-research/src/quant_engine/backtest/engine.py)
* **Execution**: Translates entry/exit condition trees into boolean series and simulates trades.
* **Trade Logic**:
  - Checks entries (`StrategyGenome.entry_long`) on candle close, executing at the next bar's open.
  - Simulates exits based on dynamic stop loss (`ExitRule.stop_loss_pct`), take profit (`ExitRule.take_profit_pct`), or custom exit signals (`ExitRule.exit_signal`).
  - Includes a transaction cost model (brokerage commission, slippage) defined in the configuration.
* **Metrics**: Compares portfolio performance using metrics like Sharpe Ratio, Sortino Ratio, CAGR, Max Drawdown, and Profit Factor.

### Standalone Testing Scripts
You can run standalone backtesting scripts located in the `scripts/` folder:
* **Precision Sniper Bot**: [precision_sniper_bot.py](file:///root/openalgo-autonomous-research/scripts/precision_sniper_bot.py) (run using `python3 scripts/precision_sniper_bot.py`).
* **RSI Divergence Backtest**: [rsi_divergence_backtest.py](file:///root/openalgo-autonomous-research/scripts/rsi_divergence_backtest.py).
* **VWAP/RSI Pullback Bot**: [mcx_vwap_rsi_strategy.py](file:///root/openalgo-autonomous-research/scripts/mcx_vwap_rsi_strategy.py).

---

## 3. Strategy Generation and Optimization

The engine automatically generates, evolves, and ranks candidate strategies based on combinatorial indicators.

### How Generation Works
1. **Grammar Production**: [GrammarConfig](file:///root/openalgo-autonomous-research/src/quant_engine/generation/grammar.py) parses a YAML configuration file and builds random combinations of allowed indicators (`trend`, `momentum`, `volatility`, `volume`) and mathematical comparison operators (e.g., crossovers).
2. **Evolutionary Loop**: Evolved over multiple generations. In each generation:
   - Evaluates population fitness.
   - Applies crossover (combining conditions) and mutation (tweaking parameters or indicators).
   - Promotes elitism to retain top performing structures.
3. **Multi-Objective Ranking**: Filters candidates through Sharpe, CAGR, and Drawdown constraints, sorting them based on OOS (Out-of-Sample) and Walk-Forward stability (NSGA-II).

### CLI Workflow

#### Run Research
Initiate a search pipeline using a YAML config file:
```bash
python -m quant_engine run config/default_research.yaml
# Or run with the Precision Sniper parameters config
python -m quant_engine run config/precision_sniper_research.yaml --debug
```

#### List Research Runs
View active and past search tasks:
```bash
python -m quant_engine list
```

#### Check Run Status & Winners
Display the top 5 accepting strategies from a specific run:
```bash
python -m quant_engine status run_<id>
```
```

#### Export Deployable Bots
Export the best strategies as standalone Python scripts (which can be run directly as live trading bots):
```bash
python -m quant_engine export run_<id> --top-n 10
```
Exported Python strategies will be saved under the run's exports folder (e.g., `./data/runs/run_<id>/exports/`).


### GIT
always take a pull from main
Take care of the merge conflicts if any
PR descrition should have purpose
no mention Of the deleted files in the PR
no force push to main
commit messages should be clear and descriptive no AI mention in the commit message

---

## 4. Documentation & Reports

All general project documentation and generated quantitative reports must reside under the [docs/](file:///root/openalgo-autonomous-research/docs/) directory.
* **Strategy Reports**: E.g. [strategy-report.md](file:///root/openalgo-autonomous-research/docs/strategy-report.md) documenting portfolio walkthroughs and WFA outcomes.
* **Architecture Docs**: Development and Low-Level Design specifications.
* **Standard Dashboards**: HTML reports such as `docs/dashboard.html` for offline visualization.
