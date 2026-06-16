# API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required (runs locally). For production, add your own auth middleware.

---

## Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "service": "quant-engine"
}
```

---

### Start Research Run

```http
POST /research/start
Content-Type: application/json
```

**Request Body:**
```json
{
  "config": {
    "name": "My Research Run",
    "trading_styles": ["intraday", "swing"],
    "data": {
      "openalgo": {
        "host": "http://127.0.0.1:5000",
        "api_key": "your-key-here",
        "source": "db"
      },
      "symbols": [
        {"symbol": "NIFTY", "exchange": "NSE"}
      ],
      "timeframes": ["15m", "1h"],
      "start_date": "2022-01-01",
      "end_date": "2024-12-31"
    },
    "generation": {
      "target_count": 10000,
      "max_conditions_per_entry": 3
    },
    "filters": {
      "backtest": {
        "min_trades": 30,
        "min_sharpe": 1.0,
        "max_drawdown": 0.25
      }
    },
    "ranking": {
      "mode": "robustness_first"
    }
  },
  "run_id": "my_custom_id"
}
```

**Response:**
```json
{
  "run_id": "my_custom_id",
  "status": "started",
  "progress": {}
}
```

---

### Get Run Status

```http
GET /research/status/{run_id}
```

**Response (running):**
```json
{
  "run_id": "run_abc123",
  "status": "running",
  "progress": {
    "run_id": "run_abc123",
    "status": "running",
    "current_stage": "backtest",
    "stages": {
      "generation": {
        "stage": "generation",
        "total": 10000,
        "completed": 10000,
        "passed": 10000,
        "rejected": 0,
        "pct": 100.0,
        "elapsed_sec": 2.3,
        "rate_per_sec": 4347.8,
        "eta_sec": 0.0
      },
      "fast_reject": {
        "stage": "fast_reject",
        "total": 10000,
        "completed": 10000,
        "passed": 3200,
        "rejected": 6800,
        "pct": 100.0
      },
      "backtest": {
        "stage": "backtest",
        "total": 3200,
        "completed": 1500,
        "passed": 420,
        "rejected": 1080,
        "pct": 46.9,
        "elapsed_sec": 45.2,
        "rate_per_sec": 33.2,
        "eta_sec": 51.2
      }
    }
  }
}
```

**Response (completed):**
```json
{
  "run_id": "run_abc123",
  "status": "completed",
  "progress": {}
}
```

---

### Stop Research Run

```http
POST /research/stop/{run_id}
```

**Response:**
```json
{
  "status": "stopping",
  "run_id": "run_abc123"
}
```

---

### Get Results

```http
GET /research/results/{run_id}
```

**Response:**
```json
{
  "run_id": "run_abc123",
  "total": 45,
  "strategies": [
    {
      "strategy_id": "abc12345",
      "composite_score": 0.823,
      "rank": 1,
      "category": "balanced",
      "backtest": {
        "sharpe": 2.15,
        "cagr": 34.2,
        "profit_factor": 2.8,
        "max_drawdown_pct": 8.5,
        "win_rate": 0.62,
        "total_trades": 145
      },
      "validation": {
        "walk_forward_score": 78.5,
        "monte_carlo_score": 82.3,
        "param_stability_score": 71.0,
        "stress_test_score": 65.8,
        "robustness_score": 74.4
      }
    }
  ]
}
```

---

### Get Winners

```http
GET /research/winners/{run_id}
```

**Response:**
```json
{
  "run_id": "run_abc123",
  "winners": [
    {
      "strategy_id": "abc12345",
      "winner_category": "best_sharpe",
      "composite_score": 0.823,
      "backtest": { "sharpe": 2.15, "cagr": 34.2, ... }
    }
  ]
}
```

---

### Get Rejections

```http
GET /research/rejections/{run_id}
```

**Response:**
```json
{
  "run_id": "run_abc123",
  "total": 8500,
  "rejections": [
    {
      "strategy_id": "def67890",
      "stage": "fast_reject",
      "rejection_reason": "missing_exit_rule",
      "threshold": "at_least_one_exit_mechanism",
      "actual_value": "none",
      "timestamp": "2025-06-16T10:00:00"
    },
    {
      "strategy_id": "ghi11111",
      "stage": "backtest_filter",
      "rejection_reason": "sharpe_below_min",
      "threshold": "1.0",
      "actual_value": "0.42",
      "timestamp": "2025-06-16T10:01:30"
    }
  ]
}
```

---

### Get Report

```http
GET /research/reports/{run_id}
```

**Response:**
```json
{
  "run_id": "run_abc123",
  "config_summary": {
    "name": "My Research",
    "trading_styles": ["intraday", "swing"],
    "target_count": 10000
  },
  "statistics": {
    "total_generated": 10000,
    "total_rejected": 9550,
    "total_backtested": 3200,
    "total_winners": 20
  },
  "top_strategies": [...]
}
```

---

### Export Strategy

```http
POST /research/export/{run_id}/{strategy_id}
```

**Response:**
```json
{
  "strategy_id": "abc12345",
  "message": "Strategy exported",
  "export_dir": "./data/runs/run_abc123/exports"
}
```

---

### List All Runs

```http
GET /research/runs
```

**Response:**
```json
{
  "runs": ["run_abc123", "run_def456", "run_ghi789"]
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Run run_xyz not found"
}
```

**Status Codes:**
| Code | Meaning |
|------|---------|
| 200 | Success |
| 404 | Run or strategy not found |
| 422 | Validation error (invalid config) |
| 500 | Internal server error |
