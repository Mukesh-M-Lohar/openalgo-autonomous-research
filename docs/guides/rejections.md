# Understanding Rejections

The engine tracks every rejection with full context. This guide explains how to read and use rejection data.

## Where Rejections Are Stored

```
data/runs/<run_id>/
├── rejected.csv            ← All rejections (ID, stage, reason, threshold, actual)
└── rejected_details.csv    ← Full strategy params for analysis
```

## Rejection Stages

Strategies can be rejected at 6 different pipeline stages:

| Stage | When | What It Checks |
|-------|------|---------------|
| `fast_reject` | Before backtest | Structural validity |
| `backtest_filter` | After backtest | Performance thresholds |
| `walk_forward` | After WF analysis | Consistency across windows |
| `oos_test` | After OOS test | Sharpe decay |
| `monte_carlo` | After MC simulation | Confidence intervals |
| `param_stability` | After perturbation | Parameter sensitivity |

## Common Rejection Reasons

### Fast Reject Stage

| Reason | Meaning | Fix |
|--------|---------|-----|
| `missing_exit_rule` | No stop-loss, signal exit, or max-hold defined | Ensure grammar always generates at least one exit mechanism |
| `excessive_complexity` | Entry has > max_complexity conditions | Lower `max_conditions_per_entry` |
| `impossible_condition` | RSI > 100, Stoch < -10, etc. | Grammar bug — check threshold sampling |
| `redundant_conditions` | Duplicate conditions in AND/OR | Dedup logic in generator |
| `contradictory_conditions` | RSI > 70 AND RSI < 30 (impossible) | Better condition composition |
| `too_few_indicators` | Entry uses < min_indicators | Increase `min_indicators` |

### Backtest Filter Stage

| Reason | Meaning | Fix |
|--------|---------|-----|
| `no_backtest_result` | Strategy produced zero signals | Check data coverage, indicator warmup |
| `total_trades_below_min` | Too few trades (insufficient sample) | Lower `min_trades` or use more data |
| `sharpe_below_min` | Poor risk-adjusted returns | Lower `min_sharpe` threshold |
| `profit_factor_below_min` | Low edge (gross_profit / gross_loss) | Lower `min_profit_factor` |
| `max_drawdown_exceeded` | Too much risk | Increase `max_drawdown` |

### Validation Stages

| Reason | Meaning | Fix |
|--------|---------|-----|
| `consistency_below_min` | < 60% of walk-forward windows profitable | Strategy is regime-dependent |
| `oos_sharpe_decay_too_high` | Sharpe drops > 40% on unseen data | Strategy is overfit |
| `confidence_below_threshold` | Monte Carlo shows < 95% confidence | Results are sequence-dependent |
| `param_decay_above_tolerance` | Small param changes cause > 25% Sharpe drop | Strategy is fragile |

## Reading rejected.csv

```csv
strategy_id,stage,rejection_reason,threshold,actual_value,timestamp
abc-123,fast_reject,missing_exit_rule,at_least_one_exit_mechanism,none,2025-06-16T10:00:00
def-456,backtest_filter,sharpe_below_min,1.0,0.42,2025-06-16T10:01:00
ghi-789,backtest_filter,max_drawdown_exceeded,0.30,0.47,2025-06-16T10:01:00
```

**Columns:**
- `threshold` — What the config required (e.g., "1.0" for min_sharpe)
- `actual_value` — What the strategy actually scored (e.g., "0.42")

## Analyzing Rejection Patterns

### Find most common rejection reasons

```python
import pandas as pd
df = pd.read_csv("data/runs/run_001/rejected.csv")
print(df["rejection_reason"].value_counts())
```

### Find which indicator combos consistently fail

```python
details = pd.read_csv("data/runs/run_001/rejected_details.csv")
rejected_at_backtest = df[df["stage"] == "backtest_filter"]["strategy_id"]
failed_details = details[details["strategy_id"].isin(rejected_at_backtest)]
print(failed_details["entry_indicators"].value_counts().head(20))
```

### Check if your filters are too aggressive

```python
# If >99% reject at backtest_filter, loosen thresholds
bt_rejects = len(df[df["stage"] == "backtest_filter"])
total = len(df) + len(pd.read_csv("data/runs/run_001/winners.csv"))
print(f"Backtest rejection rate: {bt_rejects/total:.1%}")
```

## Tuning Based on Rejections

| Observation | Action |
|-------------|--------|
| 90%+ reject at fast_reject | Check grammar — may be generating invalid structures |
| 95%+ reject at backtest_filter | Lower min_sharpe/min_pf or add more data |
| Most reject for "no trades" | Conditions too restrictive; reduce max_conditions |
| High sharpe_decay in OOS | Too much in-sample optimization; reduce evolution generations |
| Low MC confidence | Strategy depends on specific trade order; not robust |
