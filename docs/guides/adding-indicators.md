# Adding New Indicators

## Step-by-Step

### 1. Define the enum

In `src/quant_engine/models/strategy.py`:

```python
class IndicatorType(str, Enum):
    ...
    WILLIAMS_R = "williams_r"
```

### 2. Implement the computation

In `src/quant_engine/generation/indicators.py`:

```python
def _williams_r(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    period = int(params.get("period", 14))
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)
```

**Rules:**
- Function signature: `(df: DataFrame, src: Series, params: dict) -> Series`
- `df` has OHLCV columns
- `src` is the selected price source (close/high/low/etc.)
- `params` comes from the IndicatorNode's params dict
- Return a pandas Series with same index as input

### 3. Register in the function map

```python
INDICATOR_FUNCTIONS[IndicatorType.WILLIAMS_R] = _williams_r
```

### 4. Define parameter ranges

```python
INDICATOR_PARAM_RANGES[IndicatorType.WILLIAMS_R] = {
    "period": (7, 28, 7),  # (min, max, step)
}
```

The generator will sample values from this range: 7, 14, 21, 28.

### 5. Add to a category

```python
INDICATOR_CATEGORIES["momentum"].append(IndicatorType.WILLIAMS_R)
```

This determines which research configs include this indicator (via `indicator_categories`).

### 6. Write a test

In `tests/test_generation/test_strategy_generation.py`:

```python
def test_williams_r(self, sample_df):
    result = compute_indicator(
        sample_df, IndicatorType.WILLIAMS_R, {"period": 14}, PriceSource.CLOSE
    )
    assert len(result) == len(sample_df)
    valid = result.dropna()
    assert (valid >= -100).all()
    assert (valid <= 0).all()
```

### 7. Run tests

```bash
make test-generation
```

### 8. Add Export Support

To ensure the indicator is correctly translated into Python when exporting a strategy to a standalone signal script:

1. Update the translation map in [formatter.py](file:///root/openalgo-autonomous-research/src/quant_engine/export/formatter.py) inside the `_indicator_to_python` method:

```python
        elif node.indicator_type.value == "williams_r":
            period = int(params.get("period", 14))
            return f'df["{name}"] = _compute_williams_r(df, {period})'
```

2. Add the corresponding helper function definition (e.g., `_compute_williams_r`) inside the Jinja template at [signal_strategy.py.j2](file:///root/openalgo-autonomous-research/src/quant_engine/export/templates/signal_strategy.py.j2) so it is packaged with the exported code:

```python
def _compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)
```

## Multi-Output Indicators

For indicators with multiple outputs (e.g., Bollinger Bands), create separate enum values:

```python
BBANDS_UPPER = "bbands_upper"
BBANDS_MIDDLE = "bbands_middle"
BBANDS_LOWER = "bbands_lower"
```

Each gets its own computation function that shares logic but returns a different band.

## Indicators That Need Multiple Columns

Some indicators need high/low/close simultaneously (like ATR, ADX). Use `df` directly:

```python
def _atr(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
    # Use df["high"], df["low"], df["close"] directly
    # src is ignored or used as fallback
    ...
```

## Performance Tips

- Use `.rolling()` and `.ewm()` over manual loops
- Avoid `apply(lambda ...)` on large DataFrames — use vectorized operations
- For complex indicators (Supertrend), a Python loop is acceptable since it only runs once per strategy backtest
