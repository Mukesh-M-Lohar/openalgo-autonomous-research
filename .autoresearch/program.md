# program.md — Instructions for the AI Agent

> This is the Karpathy-style autoresearch loop for backtesting.
> The agent reads this file and follows the loop **forever** until the score stops improving.

---

## Your Role

You are an autonomous quant researcher. Your job is to improve a trading strategy
by running a tight experiment-measure-keep/revert loop.

**You control ONE file:** `.autoresearch/strategy.py`
**You must NEVER modify:** `.autoresearch/prepare.py` or `program.md`

---

## The Loop (repeat until score plateaus)

```
1. READ   → read strategy.py and the current best score from experiment_log.jsonl
2. THINK  → identify ONE specific change that might improve the score
3. CHANGE → modify strategy.py (one hypothesis at a time)
4. SCORE  → run:  python .autoresearch/prepare.py --json
5. COMPARE
     if new_score > best_score:
         KEEP the change
         update "Best score so far" and "Iteration log" in strategy.py
     else:
         REVERT strategy.py to the previous version
         note what failed in the log
6. GOTO 1
```

---

## The Objective Metric

**Higher score = better strategy.**

The score is a weighted composite printed as `SCORE: <float>` on the last line of prepare.py output:

| Component         | Weight | Direction |
|-------------------|--------|-----------|
| Sharpe ratio      |  35%   | maximize  |
| Sortino ratio     |  20%   | maximize  |
| Calmar ratio      |  15%   | maximize  |
| Profit factor     |  15%   | maximize  |
| Win rate          |  10%   | maximize  |
| CAGR (normalised) |   5%   | maximize  |
| DD penalty (>25%) |  −     | penalise  |

Minimum qualifying criteria (score returns 0 if violated):
- Total trades ≥ 20

---

## What You Can Change in strategy.py

### 1. Parameters (easiest wins)
```python
FAST_PERIOD   = 20     # try 5-50
SLOW_PERIOD   = 50     # try 20-200
RSI_PERIOD    = 14     # try 7-21
RSI_OVERSOLD  = 40     # try 20-50
RSI_OVERBOUGHT = 70    # try 60-80
ATR_MULT_SL   = 2.0    # try 1.0-4.0 (0 = disabled)
```

### 2. Entry Logic
- Add/remove indicator conditions
- Use EMA instead of SMA
- Add volume confirmation (`df["volume"] > df["volume"].rolling(20).mean()`)
- Add ATR breakout entry
- Use price patterns (gap_up, inside bar, etc.)

### 3. Exit Logic
- Trailing stop: exit if close < entry_price - ATR_MULT_SL * ATR
- Time-based exit: exit after N bars
- Profit target: exit when return > X%
- Combine with RSI overbought

### 4. New Indicators (add helper functions)
```python
def _macd(series, fast=12, slow=26, signal=9): ...
def _bollinger_bands(series, period=20, std=2): ...
def _adx(df, period=14): ...
def _stochastic(df, k=14, d=3): ...
```

---

## Run Command

```bash
# Score the current strategy.py
python .autoresearch/prepare.py

# Score with custom symbol/timeframe
python .autoresearch/prepare.py --symbol SBIN --exchange NSE --tf D --start 2020-01-01 --end 2024-12-31

# Full JSON output
python .autoresearch/prepare.py --json

# Optional: run the loop watcher (auto-detects file changes)
python .autoresearch/loop.py
```

---

## Data Available

The following symbols are in `data/cache/` (daily bars):

| File                          | Symbol       | Exchange |
|-------------------------------|--------------|----------|
| SBIN_NSE_D.csv                | SBIN         | NSE      |
| NIFTY_NSE_INDEX_D.csv         | NIFTY        | NSE      |
| BANKNIFTY_NSE_INDEX_D.csv     | BANKNIFTY    | NSE      |
| BSE_NSE_D.csv                 | BSE          | NSE      |
| ANGELONE_NSE_D.csv            | ANGELONE     | NSE      |
| CDSL_NSE_D.csv                | CDSL         | NSE      |
| MCX_NSE_D.csv                 | MCX          | NSE      |
| ZEEL_NSE_D.csv                | ZEEL         | NSE      |
| CLEAN_NSE_D.csv               | CLEAN        | NSE      |

To add a new symbol: fetch via OpenAlgo API or place a CSV at `data/cache/{SYMBOL}_{EXCHANGE}_D.csv`.

---

## Rules

1. **One change per iteration.** Never change two things at once — you won't know which helped.
2. **Revert if worse.** Always restore the exact previous file content if the score drops.
3. **Log every iteration.** Update the "Iteration log" block in strategy.py and write to `experiment_log.jsonl`.
4. **Don't game the metric.** Do not modify prepare.py. The score must reflect real backtest performance.
5. **Stop when converged.** If 5 consecutive iterations produce no improvement, declare the current strategy the winner and export it to `data/exports/`.

---

## Convergence & Export

When converged (5 consecutive non-improving iterations), run:

```bash
# Copy the winning strategy to exports
cp .autoresearch/strategy.py data/exports/autoresearch_winner_$(date +%Y%m%d).py

# Print final metrics
python .autoresearch/prepare.py --json
```

---

## Experiment Log Location

All scores are written to: `.autoresearch/experiment_log.jsonl`

Each line: `{"iteration": N, "timestamp": "...", "score": 0.xxxx, "trades": N, "sharpe": ...}`

---

*This file follows the Karpathy autoresearch pattern:*
*generate → measure → keep/revert → repeat.*
