# Backtest Agent Documentation

## Overview

`BacktestAgent` is a **strategy‑agnostic** utility that encapsulates the full back‑testing workflow for any strategy defined in `backtest.py`. It handles data acquisition, persistent on‑disk caching via the repository‑wide `DataCache`, optional resampling, execution of the strategy logic, and returns a rich result bundle including a performance summary.

## Installation

The back‑testing environment depends on the following Python packages (already listed in `requirements.txt`):
- **pandas**
- **numpy**
- **vectorbt**
- **openalgo**

> **Always use the project's virtual environment** to guarantee correct dependencies.
>
> ```bash
> source .venv/bin/activate   # or your preferred venv activation method
> pip install -r requirements.txt
> ```

## Caching Mechanism

`BacktestAgent` uses the **repository‑wide `DataCache`** (`quant_engine/data/cache.py`) which stores OHLCV `DataFrame`s as Parquet files on disk. This provides:
- **Persistent caching** across multiple runs or sessions.
- **Fast in‑process cache** for repeated calls within the same execution.
- Automatic fallback to the OpenAlgo API when a cache miss occurs, after which the data is written back to the cache for future use.

The default cache directory is `<project_root>/data/cache`. You can override it by passing a custom `Path` to the agent constructor.

## Usage Example

```python
# Import the agent from the dedicated module
from backtesting.ma_ribbon_stochastic.agent import BacktestAgent

# Instantiate the agent (custom cache directory or fee settings are optional)
agent = BacktestAgent()  # uses the default cache at <project_root>/data/cache

# Run the back‑test for a specific symbol / exchange / timeframe
result = agent.run(
    symbol="SBIN",
    exchange="NSE",
    timeframe="D",          # can also be "5m" or "15m"
    init_cash=1_000_000,
    stoch_variant="sequence",  # strategy‑specific argument; ignored for generic strategies
)

# The result dictionary contains the portfolio objects and a ready‑to‑print summary DataFrame
print(result["summary"])  # pretty‑print the performance table
```

## API Reference

### `BacktestAgent(cache_dir: Path | None = None, fees: float = 0.00111, fixed_fees: float = 20.0)`
- **`cache_dir`** – Directory used for persistent `DataCache`. Defaults to `<project_root>/data/cache`.
- **`fees`** – Fractional transaction fee applied to each trade (default `0.00111`).
- **`fixed_fees`** – Fixed fee per trade in currency units (default `20.0`).

### `BacktestAgent.run(
    symbol: str = "SBIN",
    exchange: str = "NSE",
    timeframe: str = "D",
    init_cash: float = 1_000_000,
    stoch_variant: str = "sequence",
    **kwargs
) -> dict`
Runs the back‑test and returns a dictionary with the following keys:
- **`df`** – Loaded price `DataFrame`.
- **`symbol`**, **`exchange`**, **`timeframe`**, **`variant`** – Metadata about the run.
- **`pf_long`**, **`pf_short`**, **`pf_bi`** – `vectorbt.Portfolio` objects for long‑only, short‑only, and bi‑directional results.
- **`summary`** – A `pandas.DataFrame` summarising total return, Sharpe ratio, Sortino ratio, max drawdown, win rate, trade count, profit factor, and expectancy.
- Additional **`**kwargs`** are accepted for future extensibility but are currently ignored.

## Extending the Agent

If you need custom fee structures, alternative preprocessing, or a different summary format, you can subclass `BacktestAgent`:

```python
class CustomAgent(BacktestAgent):
    def __init__(self, *args, custom_param=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_param = custom_param

    # Override or extend methods as needed
```

## Frequently Asked Questions

- **Q:** *Do I need to handle resampling manually?*\
  **A:** The agent does not perform resampling automatically; you can resample the `DataFrame` returned by `load_data` before passing it to your own strategy logic if required.

- **Q:** *Can I change the cache expiration policy?*\
  **A:** `DataCache` stores data indefinitely until you manually clear it via `DataCache.clear()`. Implement your own expiration logic if needed.

- **Q:** *Is the agent compatible with other strategies?*\
  **A:** Yes. The agent is generic; it simply loads data and calls the `run_backtest` function defined in `backtest.py`. Replace or modify `run_backtest` to implement a different strategy, and the same `BacktestAgent` can be used.

---

*This documentation lives in `backtesting/ma_ribbon_stochastic/agent.md` and is intended for developers and analysts integrating the back‑testing agent into notebooks, pipelines, or other automation scripts.*
