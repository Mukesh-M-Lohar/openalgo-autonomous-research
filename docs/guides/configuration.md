# Configuration Reference

All research is controlled via YAML config files. This document covers every option.

## Full Config Structure

```yaml
# Required
name: "My Research"
description: "Optional description"

# Trading styles to generate strategies for
trading_styles:
  - intraday    # MIS, forced exit at market close
  - btst        # Buy today sell tomorrow
  - swing       # 2-15 day holds
  - positional  # Weeks to months

# Override defaults per style
style_overrides:
  intraday:
    max_hold_bars: 75
    forced_exit_time: "15:15"
    product_type: "MIS"
    min_trades: 200
  btst:
    min_hold_bars: 50
    max_hold_bars: 150
    product_type: "CNC"
  swing:
    min_hold_bars: 10
    max_hold_bars: 360
    product_type: "CNC"
  positional:
    min_hold_bars: 5
    product_type: "CNC"
    min_trades: 20

# Data source
data:
  openalgo:
    host: "http://127.0.0.1:5000"
    api_key: "${OPENALGO_API_KEY}"  # Environment variable
    source: "db"                    # "db" (local) or "api" (broker)
  symbols:
    - symbol: "NIFTY"
      exchange: "NSE"
    - symbol: "BANKNIFTY"
      exchange: "NSE"
  timeframes: ["5m", "15m", "1h", "1d"]
  start_date: "2020-01-01"
  end_date: "2025-01-01"
  train_pct: 0.70      # 70% for backtesting
  validation_pct: 0.15  # 15% for walk-forward
  test_pct: 0.15        # 15% for out-of-sample

# Strategy generation
generation:
  mode: "random"          # "random" | "exhaustive" | "guided"
  target_count: 100000    # Number of strategies to generate
  max_conditions_per_entry: 4  # Max conditions in entry logic
  allow_short: false      # Generate short strategies?
  indicator_categories:   # Which indicators to use
    - trend               # SMA, EMA, MACD, ADX, Supertrend
    - momentum            # RSI, Stoch, CCI, ROC
    - volatility          # ATR, BBands, Keltner, Donchian
    - volume              # VWAP, OBV, Volume SMA
  multi_timeframe: true   # Allow multi-TF strategies?

# Rejection filters (strategies that fail any threshold are rejected)
filters:
  fast_reject:
    max_complexity: 5     # Max conditions before it's "overfit"
    min_indicators: 1     # At least 1 indicator required
  backtest:
    min_trades: 30        # Minimum trade count
    min_sharpe: 1.0       # Minimum Sharpe ratio
    max_drawdown: 0.30    # Maximum drawdown (30%)
    min_profit_factor: 1.3
    min_win_rate: 0.0     # No minimum by default
    min_cagr: 0.0         # No minimum by default
  validation:
    max_oos_sharpe_decay: 0.40    # Max Sharpe decay in OOS
    min_walk_forward_consistency: 0.6  # 60% windows profitable
    monte_carlo_confidence: 0.95
    param_stability_tolerance: 0.25    # Max 25% Sharpe decay

# Evolution (optional)
evolution:
  enabled: true
  generations: 5          # How many generations to evolve
  population_size: 500    # Population per generation
  mutation_rate: 0.3      # Probability of mutation per gene
  crossover_rate: 0.5     # Probability of crossover
  elitism_pct: 0.1        # Top 10% survive unchanged
  tournament_size: 5      # Tournament selection size

# Ranking mode
ranking:
  mode: "robustness_first"  # "weighted"|"pareto"|"constraint"|"robustness_first"
  objectives:
    - metric: "sharpe"
      weight: 0.30
      direction: "maximize"
    - metric: "sortino"
      weight: 0.20
      direction: "maximize"
    - metric: "max_drawdown_pct"
      weight: 0.20
      direction: "minimize"
    - metric: "profit_factor"
      weight: 0.15
      direction: "maximize"
    - metric: "cagr"
      weight: 0.15
      direction: "maximize"
  export_top_n: 20        # Total strategies to export
  export_per_category: 5  # Per category (best_sharpe, etc.)

# Execution
execution:
  max_workers: null       # null = auto (cpu_count - 1)
  chunk_size: 200         # Strategies per parallel chunk
  memory_limit_gb: 8.0

# Cost model
cost_model:
  commission_pct: 0.03    # 0.03% per trade
  slippage_pct: 0.02      # 0.02% slippage per trade
  min_commission: 20.0    # Rs 20 minimum

# Output
output:
  storage: "csv"          # "csv" or "duckdb"
  base_dir: "./data/runs"
  export_dir: "./data/exports"
  save_all_candidates: true  # Save generated.csv?
```

## Environment Variable Interpolation

Any value `"${VAR_NAME}"` is replaced with the environment variable at load time:

```yaml
api_key: "${OPENALGO_API_KEY}"  # reads os.environ["OPENALGO_API_KEY"]
```

## Ranking Modes Explained

### `weighted` — Simple weighted sum
Composite score = Σ(weight × normalized_metric). Best for simple "maximize returns" research.

### `pareto` — Multi-objective Pareto frontier
Uses NSGA-II non-dominated sorting. Returns strategies on the Pareto frontier — no single strategy dominates all others. Best when you want diverse options.

### `constraint` — Hard filters + single objective
Apply hard constraints (min_sharpe, max_drawdown), then maximize a single metric. Best for targeted searches.

### `robustness_first` (default) — Prioritize stability
Score = 60% robustness (walk-forward + MC + stability + stress) + 40% performance. Rejects high-return but fragile strategies. **Recommended for production use.**

## Tuning Tips

- **Start small:** `target_count: 5000` to verify your setup works
- **Loosen filters first:** Set `min_sharpe: 0.5` until you confirm strategies generate
- **Add complexity gradually:** Start with 1-2 indicator categories, add more later
- **Check rejection CSV:** If 99% reject at fast_reject, your constraints may be too tight
