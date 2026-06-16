# OpenAlgo Autonomous Quant Research Engine

A standalone, fully offline quantitative research microservice that autonomously discovers, evaluates, validates, ranks, and exports deployable trading strategies. Integrates with [OpenAlgo](https://github.com/marketcalls/openalgo) through REST APIs for historical data retrieval.

## Key Features

- **Fully Offline** — No LLM, AI, cloud, or internet dependency. Pure systematic quantitative research.
- **Grammar-Based Generation** — Produces 50K–500K+ candidate strategies from combinatorial indicator/pattern rules.
- **Multi-Style Support** — Intraday (MIS), BTST, Swing, and Positional (CNC) strategies with style-specific constraints.
- **Fast Rejection Pipeline** — Kills structurally invalid strategies before expensive backtesting.
- **Vectorized Backtesting** — NumPy/Pandas-based engine with full trade simulation, cost model, and 20+ metrics.
- **Robustness Testing** — Walk-forward, out-of-sample, Monte Carlo, parameter stability, and stress testing.
- **Evolutionary Optimization** — Mutation, crossover, tournament selection, and elitism across generations.
- **Multi-Objective Ranking** — Weighted scoring, NSGA-II Pareto fronts, and robustness-first modes.
- **Rejection Tracking** — Every rejected strategy logged with stage, reason, threshold, and actual value.
- **Signal-Only Export** — Generates standalone Python signal scripts (no order placement).
- **REST API** — FastAPI endpoints for starting runs, polling status, and retrieving results.
- **Multi-Core** — Automatic CPU detection and parallel backtesting via ProcessPoolExecutor.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH PIPELINE                                 │
│                                                                          │
│  Config → Data Fetch → Generate (50K-500K) → Fast Reject (~70% killed)  │
│    → Backtest (multi-core) → Filter (Sharpe/DD/trades)                   │
│    → Walk-Forward → OOS Test → Robustness (MC, stability, stress)        │
│    → Evolution Loop (mutate survivors, re-enter at backtest)             │
│    → Rank (Pareto/weighted/robustness-first) → Export (signal scripts)   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAlgo running locally (for data retrieval)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd openalgo-quant-engine

# Install with pip
pip install -e .

# Or install dependencies directly
pip install numpy pandas pydantic pydantic-settings pyyaml fastapi uvicorn typer jinja2 httpx
```

### Run Research (CLI)

```bash
# Run with example config
python -m quant_engine run config/example_research.yaml --debug

# Run with default config
python -m quant_engine run config/default_research.yaml

# Check results
python -m quant_engine status run_xxxxx

# List all runs
python -m quant_engine list

# Export top strategies
python -m quant_engine export run_xxxxx --top-n 10
```

### Run API Server

```bash
# Start the REST API
python -m quant_engine serve --port 8000

# Health check
curl http://localhost:8000/health

# Start a research run
curl -X POST http://localhost:8000/research/start \
  -H "Content-Type: application/json" \
  -d '{"config": {"name": "My Research", "trading_styles": ["intraday", "swing"], ...}}'

# Check status
curl http://localhost:8000/research/status/{run_id}

# Get winners
curl http://localhost:8000/research/winners/{run_id}

# Get rejections with reasons
curl http://localhost:8000/research/rejections/{run_id}
```

## Configuration

Research is controlled via YAML config files. See `config/example_research.yaml` for a complete example.

### Trading Styles

| Style      | Holding Period | Timeframes  | Product | Exit Logic             |
| ---------- | -------------- | ----------- | ------- | ---------------------- |
| Intraday   | Minutes–hours  | 1m, 5m, 15m | MIS     | Forced exit at 3:15 PM |
| BTST       | Overnight      | 15m, 1h     | CNC     | Next-day exit          |
| Swing      | 2–15 days      | 1h, 4h, 1d  | CNC     | Trailing stops         |
| Positional | Weeks–months   | 1d, 1w      | CNC     | Trend-following exits  |

### Key Config Sections

```yaml
trading_styles: [intraday, swing, btst, positional]

data:
  openalgo:
    host: "http://127.0.0.1:5000"
    api_key: "${OPENALGO_API_KEY}" # env var interpolation
    source: "db"
  symbols:
    - { symbol: NIFTY, exchange: NSE }
  timeframes: [5m, 15m, 1h, 1d]
  start_date: "2020-01-01"
  end_date: "2025-01-01"

generation:
  target_count: 100000
  max_conditions_per_entry: 3

filters:
  backtest:
    min_trades: 30
    min_sharpe: 1.0
    max_drawdown: 0.25
    min_profit_factor: 1.3

ranking:
  mode: "robustness_first" # weighted | pareto | constraint | robustness_first
```

## Output Structure

```
data/runs/run_001/
├── config.yaml                  # Research config used
├── generated.csv                # All generated strategies
├── rejected.csv                 # Rejections with stage + reason + threshold vs actual
├── rejected_details.csv         # Full params of rejected strategies
├── backtested.csv               # Strategies that passed backtest with metrics
├── walkforward.csv              # Walk-forward results
├── oos_results.csv              # Out-of-sample results
├── robustness.csv               # Monte Carlo + stability + stress scores
├── survivors.csv                # Final surviving strategies
├── winners.csv                  # Top ranked per category
├── trade_logs/                  # Per-strategy trade logs
├── equity_curves/               # Per-strategy equity curves
└── exports/                     # Generated signal scripts
    ├── {id}_strategy.py         # Standalone signal generator
    └── {id}_strategy.json       # Strategy definition + metrics
```

### Rejection CSV Format

Every rejected strategy includes:

- `strategy_id` — Unique ID
- `stage` — Which stage rejected it (fast_reject, backtest_filter, walk_forward, etc.)
- `rejection_reason` — Human-readable reason (sharpe_below_min, contradictory_conditions, etc.)
- `threshold` — The configured threshold that was violated
- `actual_value` — The strategy's actual value for that metric

## Supported Indicators (27)

**Trend:** SMA, EMA, WMA, VWMA, MACD, MACD Signal, MACD Histogram, ADX, Supertrend

**Momentum:** RSI, Stochastic %K/%D, CCI, ROC, Momentum

**Volatility:** ATR, Bollinger Bands (Upper/Middle/Lower), Keltner Channels, Donchian Channels

**Volume:** VWAP, OBV, Volume SMA

## Supported Patterns (11)

Opening Range Breakout, Gap Up/Down, Inside Bar, Outside Bar, Bullish/Bearish Engulfing, Bullish/Bearish Pin Bar, Higher High Higher Low, Lower High Lower Low

## API Endpoints

| Method | Path                                      | Description                     |
| ------ | ----------------------------------------- | ------------------------------- |
| GET    | `/health`                                 | Health check                    |
| POST   | `/research/start`                         | Start a new research run        |
| GET    | `/research/status/{run_id}`               | Poll run progress               |
| POST   | `/research/stop/{run_id}`                 | Stop a running research         |
| GET    | `/research/results/{run_id}`              | Get ranked results              |
| GET    | `/research/winners/{run_id}`              | Get top strategies per category |
| GET    | `/research/rejections/{run_id}`           | Get rejections with reasons     |
| GET    | `/research/reports/{run_id}`              | Get summary report              |
| POST   | `/research/export/{run_id}/{strategy_id}` | Export strategy                 |
| GET    | `/research/runs`                          | List all runs                   |

## Testing

```bash
# Run all tests
PYTHONPATH=src python -m pytest tests/ -v

# Run specific test module
PYTHONPATH=src python -m pytest tests/test_generation/ -v
PYTHONPATH=src python -m pytest tests/test_backtest/ -v
PYTHONPATH=src python -m pytest tests/test_validation/ -v
PYTHONPATH=src python -m pytest tests/test_pipeline/ -v
```

## Technology Stack

| Component       | Technology            | Rationale                              |
| --------------- | --------------------- | -------------------------------------- |
| Web Framework   | FastAPI               | Async, auto-docs, Pydantic integration |
| Backtesting     | Custom (NumPy/Pandas) | Vectorized, no external dependency     |
| Data Processing | Pandas + NumPy        | Industry standard for OHLCV data       |
| Config          | Pydantic + PyYAML     | Type-safe validation with YAML support |
| Parallelism     | multiprocessing       | Zero deps, Windows-compatible          |
| Storage V1      | CSV                   | Human-readable, zero setup             |
| Storage V2      | DuckDB                | Embedded columnar (migration path)     |
| Templates       | Jinja2                | Strategy code generation               |
| CLI             | Typer                 | Clean CLI with auto-help               |
| HTTP Client     | httpx                 | Modern async-capable HTTP              |

## OpenAlgo Integration

The engine communicates with OpenAlgo only for **data retrieval**:

```
POST /api/v1/history
{
  "apikey": "...",
  "symbol": "NIFTY",
  "exchange": "NSE",
  "interval": "5m",
  "start_date": "2020-01-01",
  "end_date": "2025-01-01",
  "source": "db"
}
```

**No orders are placed.** The engine outputs signal scripts and backtest results only.

## License

MIT
