# Quick Start

## Run Your First Research

### 1. Start with the default config (small, fast)

```bash
PYTHONPATH=src python -m quant_engine run config/default_research.yaml --debug
```

This generates 10,000 strategies for NIFTY (intraday + swing) and runs the full pipeline. Takes about 2-5 minutes depending on CPU.

### 2. Check results

```bash
PYTHONPATH=src python -m quant_engine list
PYTHONPATH=src python -m quant_engine status <run_id>
```

### 3. Examine output files

```
data/runs/<run_id>/
├── config.yaml         ← Config used
├── rejected.csv        ← WHY strategies were rejected
├── winners.csv         ← Top strategies with metrics
└── exports/            ← Signal scripts
```

## Run via API

### 1. Start the server

```bash
PYTHONPATH=src python -m quant_engine serve --port 8000
```

### 2. Start research

```bash
curl -X POST http://localhost:8000/research/start \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "name": "Quick Test",
      "trading_styles": ["intraday"],
      "data": {
        "openalgo": {"host": "http://127.0.0.1:5000", "api_key": "YOUR_KEY", "source": "db"},
        "symbols": [{"symbol": "NIFTY", "exchange": "NSE"}],
        "timeframes": ["15m"],
        "start_date": "2023-01-01",
        "end_date": "2024-12-31"
      },
      "generation": {"target_count": 5000}
    }
  }'
```

### 3. Poll status

```bash
curl http://localhost:8000/research/status/<run_id>
```

### 4. Get winners

```bash
curl http://localhost:8000/research/winners/<run_id>
```

## Understanding the Output

### winners.csv

Each winning strategy includes:
- **Sharpe/Sortino/Calmar** — Risk-adjusted returns
- **CAGR** — Compound annual growth rate
- **Profit Factor** — Gross profit / gross loss
- **Max Drawdown** — Worst peak-to-trough decline
- **Win Rate** — Percentage of profitable trades
- **Robustness Score** — Combined walk-forward + MC + stability + stress

### rejected.csv

Every rejected strategy tells you:
- **Which stage** rejected it (fast_reject, backtest_filter, walk_forward, etc.)
- **Why** (human-readable reason like "sharpe_below_min")
- **What threshold** it violated (e.g., "1.0")
- **What its actual value was** (e.g., "0.42")

Use this to tune your filter thresholds or understand which indicator combinations are consistently weak.

## Next: Customize Your Config

See [Configuration Guide](guides/configuration.md) for full YAML reference.
