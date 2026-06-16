# Low-Level Design (LLD)

## 1. System Context

```
┌─────────────────┐       HTTP/REST        ┌──────────────────────────┐
│                 │ ──────────────────────► │                          │
│   OpenAlgo      │   POST /api/v1/history  │   Quant Research Engine  │
│   (Data Source) │ ◄────────────────────── │                          │
│                 │       OHLCV JSON        │                          │
└─────────────────┘                         └──────────────────────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────────┐
                                            │   CSV / DuckDB Storage   │
                                            │   + Signal Scripts (.py) │
                                            └──────────────────────────┘
```

## 2. Module Dependency Graph

```
__main__.py (CLI)
    │
    ├── config.py ◄─── YAML files
    │
    ├── api/ (FastAPI server)
    │   └── routes.py → pipeline/orchestrator.py
    │
    └── pipeline/orchestrator.py
            │
            ├── data/client.py → OpenAlgo API
            │   └── data/cache.py → Local Parquet cache
            │
            ├── generation/generator.py
            │   ├── generation/grammar.py
            │   ├── generation/indicators.py
            │   └── generation/patterns.py
            │
            ├── generation/validator.py (Fast Reject)
            │
            ├── backtest/engine.py
            │   ├── backtest/metrics.py
            │   └── backtest/cost_model.py
            │
            ├── validation/
            │   ├── walk_forward.py
            │   ├── out_of_sample.py
            │   ├── monte_carlo.py
            │   ├── parameter_stability.py
            │   └── stress_test.py
            │
            ├── evolution/
            │   ├── mutator.py
            │   ├── crossover.py
            │   ├── fitness.py
            │   └── population.py
            │
            ├── ranking/
            │   ├── scorer.py
            │   └── pareto.py
            │
            ├── export/formatter.py → Jinja2 templates
            │
            ├── storage/csv_backend.py
            │
            └── parallel/pool.py
```

## 3. Data Models

### 3.1 StrategyGenome (AST Representation)

The core data structure is an Abstract Syntax Tree (AST) that represents strategies as data rather than code.

```
StrategyGenome
├── id: str (UUID-12)
├── trading_style: TradingStyle (INTRADAY|BTST|SWING|POSITIONAL)
├── entry_long: ConditionTree
│   ├── ConditionNode (leaf)
│   │   ├── left: IndicatorNode | float
│   │   ├── op: CompareOp (GT|LT|CROSS_ABOVE|CROSS_BELOW|...)
│   │   └── right: IndicatorNode | float
│   └── CompositeCondition (branch)
│       ├── logic: LogicOp (AND|OR)
│       └── children: tuple[ConditionTree, ...]
├── exit_long: ExitRule
│   ├── stop_loss_pct: float?
│   ├── take_profit_pct: float?
│   ├── trailing_stop_pct: float?
│   ├── exit_signal: ConditionTree?
│   └── max_hold_bars: int?
├── timeframes_used: tuple[TimeframeType, ...]
├── product_type: str (MIS|CNC)
├── forced_exit_time: str? ("15:15" for intraday)
├── generation: int (evolution generation)
└── parent_ids: tuple[str, ...] (lineage tracking)

IndicatorNode (frozen dataclass, hashable)
├── indicator_type: IndicatorType (27 types)
├── params: tuple[tuple[str, number], ...] (sorted, immutable)
├── timeframe: TimeframeType
└── source: PriceSource (close|high|low|open|volume|hl2|hlc3|ohlc4)
```

**Design Rationale:**
- `frozen=True` on nodes enables hashing, deduplication, and caching
- `tuple` instead of `dict` for params ensures immutability
- AST enables: grammar-based generation, subtree mutation, serialization, code generation

### 3.2 BacktestResult

```
BacktestResult
├── strategy_id: str
├── net_profit / net_profit_pct
├── cagr: float
├── sharpe / sortino / calmar: float
├── profit_factor: float
├── max_drawdown / max_drawdown_pct: float
├── win_rate: float
├── total_trades / winning_trades / losing_trades: int
├── avg_trade_pct / avg_win_pct / avg_loss_pct: float
├── recovery_factor / expectancy: float
├── ulcer_index: float
├── avg_hold_bars: float
└── max_consecutive_wins / max_consecutive_losses: int
```

### 3.3 RejectionRecord

```
RejectionRecord
├── strategy_id: str
├── stage: str (fast_reject|backtest_filter|walk_forward|oos_test|monte_carlo|...)
├── rejection_reason: str (human-readable: "sharpe_below_min")
├── threshold: str (configured value: "1.0")
├── actual_value: str (strategy's value: "0.42")
└── timestamp: str (ISO 8601)
```

## 4. Pipeline Stages (Detailed)

### Stage 1: Data Fetch

```python
class OpenAlgoClient:
    def fetch_history(symbol, exchange, interval, start_date, end_date) -> DataFrame
    def fetch_all(symbols, timeframes, start_date, end_date) -> dict[str, dict[str, DataFrame]]
```

- Rate-limited: Token bucket at 9 req/sec (below OpenAlgo's 10/sec limit)
- Cached: Parquet files keyed by MD5(symbol+exchange+interval+dates)
- Retry: httpx with 60s timeout
- Output: `dict[symbol_key, dict[timeframe, DataFrame]]`

### Stage 2: Strategy Generation

```
GrammarConfig → StrategyGenerator → list[StrategyGenome]
```

**Grammar Production Rules:**
```
STRATEGY → ENTRY_LONG + EXIT_LONG [+ ENTRY_SHORT + EXIT_SHORT]
ENTRY → CONDITION | COMPOSITE(AND|OR, CONDITION, ...)
CONDITION → INDICATOR CompareOp INDICATOR | INDICATOR CompareOp CONSTANT
INDICATOR → (type, params, timeframe, source)
EXIT → {stop_loss, take_profit, trailing_stop, signal, max_hold}
```

**Generation Algorithm:**
1. Select trading style → determines allowed timeframes
2. Select 1-4 indicators from allowed categories
3. For each indicator: sample parameters from defined ranges
4. Compose conditions with random operators
5. Generate exit rule with risk:reward >= 1:1
6. Dedup by content hash fingerprint

### Stage 3: Fast Rejection

```python
class FastRejectValidator:
    checks = [
        _check_missing_exit,      # No exit mechanism defined
        _check_complexity,         # >5 conditions (overfit risk)
        _check_impossible,         # RSI > 100, etc.
        _check_redundant,          # Duplicate conditions
        _check_contradictory,      # A AND NOT-A
        _check_min_indicators,     # Too few indicators
    ]
```

**Expected rejection rate:** 60-80% of generated strategies.

### Stage 4: Backtesting

```python
class BacktestEngine:
    def run(strategy, data) -> BacktestResult | None
```

**Signal Evaluation Pipeline:**
```
StrategyGenome.entry_long (AST)
    → _eval_condition_tree(tree, df, all_data)
    → Boolean Series (True = entry signal)

For each IndicatorNode in tree:
    → compute_indicator(df, type, params, source) → pd.Series

For each ConditionNode:
    left_series OP right_series → Boolean Series

For CompositeCondition:
    children[0] AND/OR children[1] AND/OR ... → Boolean Series
```

**Trade Simulation (bar-by-bar):**
```
for each bar:
    if position_open:
        check stop_loss (low <= sl_price)
        check take_profit (high >= tp_price)
        check trailing_stop (low <= trail_price)
        check max_hold_bars
        check forced_exit_time (intraday)
        check exit_signal
    elif entry_signal:
        open position at close price
```

**Cost Application:** Round-trip cost = (commission_pct + slippage_pct) * 2

### Stage 5: Backtest Filtering

Configurable thresholds applied to BacktestResult:
- `min_trades` (default: 30) — reject insufficient sample
- `min_sharpe` (default: 1.0) — reject poor risk-adjusted returns
- `max_drawdown` (default: 0.30) — reject excessive risk
- `min_profit_factor` (default: 1.3) — reject low edge

Each rejection stored with reason, threshold, and actual value.

### Stage 6: Walk-Forward Validation

```python
class WalkForwardValidator:
    def validate(strategy, data) -> dict
```

**Algorithm:**
1. Split data into rolling windows: `[train_N | test_N]`
2. For each window: backtest on test portion
3. Compute consistency = profitable_windows / total_windows
4. Score = consistency * 50 + avg_sharpe_bounded * 25

**Default:** train=500 bars, test=125 bars, step=125 bars, min 4 windows.

### Stage 7: Out-of-Sample Testing

```python
class OOSValidator:
    def validate(strategy, in_sample_result, oos_data) -> dict
```

**Key metric:** Sharpe decay = 1 - (OOS_sharpe / IS_sharpe)
- Decay < 0.40 → strategy generalizes
- Decay > 0.40 → likely overfit, rejected

### Stage 8: Robustness Testing

Three independent validators:

**Monte Carlo (trade sequence independence):**
1. Shuffle trade order 1000 times
2. Add execution noise (±0.02% per trade)
3. Compute confidence intervals
4. Score = profitable_sims_pct * 50 + positive_sharpe_pct * 50

**Parameter Stability (neighborhood robustness):**
1. Perturb all indicator params by ±15%
2. Re-backtest 10 perturbations
3. Measure Sharpe variance and decay
4. Score = (1 - decay) * 50 + (1 - CV) * 50

**Stress Testing (adverse conditions):**
1. 2x commission + slippage
2. 1.5x volatility injection (random noise on prices)
3. 3x slippage only
4. Score = survival_ratio * 100 (avg_stressed_sharpe / original_sharpe)

### Stage 9: Evolution

```python
class Population:
    def evolve() -> list[StrategyGenome]
```

**Evolutionary Algorithm:**
```
1. Select top N% as elite (direct copy to next gen)
2. For remaining slots:
   a. Tournament selection (pick 2-5 random, take fittest)
   b. Crossover: combine entry conditions from 2 parents
   c. Mutation: apply 1+ random operators
3. Re-backtest all offspring
4. Update population fitness, keep top population_size
5. Repeat for configured generations
```

**Mutation Operators (6):**
| Operator | Action | Example |
|----------|--------|---------|
| param_perturb | Adjust numeric params ±15% | EMA(20) → EMA(23) |
| operator_swap | Change comparison op | GT → CROSS_ABOVE |
| indicator_swap | Replace with same-category | EMA → SMA |
| condition_add_remove | Grow or prune logic tree | Add RSI filter |
| exit_modify | Adjust SL/TP/trailing | SL 2% → SL 2.4% |
| threshold_adjust | Change constant values | RSI > 70 → RSI > 68 |

### Stage 10: Ranking

**Four ranking modes:**

1. **Weighted:** Composite score = Σ(weight_i × normalize(metric_i))
2. **Pareto:** NSGA-II non-dominated sorting, front assignment
3. **Constraint + Objective:** Hard filters first, then optimize single metric
4. **Robustness-First (default):** 60% robustness + 40% performance

**Output categories:**
- Best Overall, Best Sharpe, Best CAGR, Best Profit Factor
- Lowest Drawdown, Best Robust
- Conservative/Balanced/Aggressive portfolios

## 5. Parallelism Design

```
┌─────────────────────────────────────────┐
│         PipelineOrchestrator            │
│  (single process, coordination)          │
├─────────────────────────────────────────┤
│                                          │
│  ┌────────────────────────────────────┐  │
│  │      WorkerPool (ProcessPool)      │  │
│  │   max_workers = cpu_count() - 1    │  │
│  │   context = "spawn" (Windows-safe) │  │
│  │                                    │  │
│  │   ┌──────┐ ┌──────┐ ┌──────┐     │  │
│  │   │Worker│ │Worker│ │Worker│ ... │  │
│  │   │  1   │ │  2   │ │  3   │     │  │
│  │   └──────┘ └──────┘ └──────┘     │  │
│  └────────────────────────────────────┘  │
│                                          │
│  Work distribution:                      │
│  map_chunks(fn, items, chunk_size=200)   │
│    → Split items into chunks              │
│    → Submit each chunk to worker          │
│    → Collect and flatten results          │
└─────────────────────────────────────────┘
```

**Per-stage parallelism:**
| Stage | Parallel? | Reason |
|-------|-----------|--------|
| Data Fetch | No | Rate-limited, I/O bound |
| Generation | No | Fast enough single-threaded |
| Fast Reject | Yes (chunks) | Embarrassingly parallel |
| Backtest | Yes (chunks) | CPU-intensive, independent |
| Walk-Forward | Yes (per strategy) | Independent per strategy |
| OOS/Robustness | Yes (per strategy) | Independent |
| Evolution | No | Requires population-level selection |
| Ranking | No | Requires global comparison |
| Export | No | I/O bound |

## 6. Storage Layer

### 6.1 CSV Backend (V1)

```python
class CsvStorage:
    def init_run(run_id, config)        # Creates directory + config.yaml
    def save_generated(run_id, list)     # generated.csv
    def save_rejections(run_id, list)    # rejected.csv (appends)
    def save_rejection_details(run_id, list)  # rejected_details.csv
    def save_backtest_results(run_id, list)   # backtested.csv
    def save_validation_results(run_id, stage, list)  # {stage}.csv
    def save_survivors(run_id, list)     # survivors.csv
    def save_winners(run_id, list)       # winners.csv
    def save_trade_log(run_id, sid, df)  # trade_logs/{sid}_trades.csv
    def save_equity_curve(run_id, sid, df)   # equity_curves/{sid}_equity.csv
```

### 6.2 Migration Path to DuckDB (V2)

The `StorageBackend` protocol enables seamless migration:
```python
class StorageBackend(Protocol):
    def save_rejections(run_id, rejections) -> None: ...
    def load_results(run_id, stage) -> list[dict]: ...
    # ... same interface, different implementation
```

DuckDB advantages: columnar compression, analytical queries, embedded (no server).

## 7. API Layer Design

```python
FastAPI App
├── GET  /health                          → {"status": "ok"}
├── POST /research/start                  → Starts BackgroundTask
│       Body: {config: {...}, run_id?: str}
│       Returns: {run_id, status: "started"}
├── GET  /research/status/{run_id}        → RunProgress.to_dict()
├── POST /research/stop/{run_id}          → Sets stop flag
├── GET  /research/results/{run_id}       → Loads survivors.csv
├── GET  /research/winners/{run_id}       → Loads winners.csv
├── GET  /research/rejections/{run_id}    → Loads rejected.csv
├── GET  /research/reports/{run_id}       → Summary statistics
├── POST /research/export/{run_id}/{sid}  → Generates .py + .json
└── GET  /research/runs                   → Lists run directories
```

**Background Task Lifecycle:**
```
POST /start → BackgroundTask(_run) → orchestrator.run()
                                          │
                                          ├── progress.status = "running"
                                          ├── stage changes update progress
                                          ├── progress.status = "completed"
                                          └── or "failed" on exception
```

## 8. Export Format

Generated signal scripts follow this structure:

```python
# Auto-generated by Quant Research Engine
# Strategy ID, style, metrics documented in header

def compute_indicators(df: DataFrame) -> DataFrame:
    # All indicator computations

def generate_entry_signal(df: DataFrame) -> Series:
    # Boolean series: True = buy signal

def generate_exit_signal(df: DataFrame, entry_price: float) -> Series:
    # Boolean series: True = exit signal

def get_strategy_params() -> dict:
    # Strategy metadata for reference
```

**Key design decision:** Signal scripts are pure functions, not trading bots. Users wire them into their execution system (OpenAlgo, custom, manual) as they see fit.

## 9. Configuration Schema

```
ResearchConfig (Pydantic BaseModel)
├── name: str
├── trading_styles: list[str]
├── style_overrides: dict[str, StyleOverride]
│   └── StyleOverride
│       ├── max_hold_bars / min_hold_bars
│       ├── forced_exit_time
│       ├── product_type (MIS|CNC)
│       └── min_trades
├── data: DataConfig
│   ├── openalgo: OpenAlgoConfig (host, api_key, source)
│   ├── symbols: list[SymbolConfig]
│   ├── timeframes: list[str]
│   ├── start_date / end_date
│   └── train_pct / validation_pct / test_pct
├── generation: GenerationConfig
│   ├── mode (random|exhaustive|guided)
│   ├── target_count
│   ├── max_conditions_per_entry
│   ├── allow_short
│   └── indicator_categories
├── filters: FiltersConfig
│   ├── fast_reject: FastRejectFilters
│   ├── backtest: BacktestFilters
│   └── validation: ValidationFilters
├── evolution: EvolutionConfig
├── ranking: RankingConfig
│   ├── mode (weighted|pareto|constraint|robustness_first)
│   └── objectives: list[ObjectiveConfig]
├── execution: ExecutionConfig (max_workers, chunk_size)
├── cost_model: CostModelConfig
└── output: OutputConfig (storage type, base_dir)
```

## 10. Error Handling & Resilience

| Scenario | Handling |
|----------|----------|
| OpenAlgo API down | Cache hit returns data; miss raises with clear error |
| Strategy crashes backtest | Returns None, skipped silently |
| Worker process dies | Future.result() catches exception, logs, continues |
| Run interrupted | Progress saved at each stage; future: checkpoint/resume |
| Disk full | Storage operations catch IOError, log warning |
| Invalid config | Pydantic validation raises before pipeline starts |

## 11. Performance Characteristics

| Stage | Complexity | Typical Time (100K strategies) |
|-------|-----------|-------------------------------|
| Generation | O(N) | ~2 sec |
| Fast Reject | O(N × conditions) | ~1 sec |
| Backtest | O(N × bars × indicators) | ~5-30 min (parallel) |
| Walk-Forward | O(survivors × windows × bars) | ~2-10 min |
| Robustness | O(survivors × simulations) | ~3-15 min |
| Evolution | O(pop × generations × bars) | ~2-5 min |
| Ranking | O(survivors²) for Pareto | < 1 sec |

**Bottleneck:** Backtesting (Stage 4). Parallelism provides near-linear speedup across cores.

## 12. Future Extension Points

| Extension | Where to add | Impact |
|-----------|-------------|--------|
| New indicator | `generation/indicators.py` + INDICATOR_FUNCTIONS registry | Zero impact on rest |
| New pattern | `generation/patterns.py` + PATTERN_FUNCTIONS registry | Zero impact |
| Genetic programming | `evolution/gp.py` (new file) | Deeper tree generation |
| Portfolio optimization | `ranking/portfolio.py` (extend) | Correlation-based selection |
| DuckDB storage | `storage/db_backend.py` (implement protocol) | Swap via config |
| Real-time signals | `export/templates/realtime.py.j2` | New export template |
| Distributed execution | Replace ProcessPool with Ray/Dask | Same interface |
| Custom fitness | Implement `FitnessFunction` protocol | Plug into evolution |
