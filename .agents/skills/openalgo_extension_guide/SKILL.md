---
name: OpenAlgo Extension Guide
description: Developer guide on extending the quant engine (adding indicators, updating grammar rules, modifying the Jinja2 bot exporter templates).
---

# Developer Extension Guide: Customizing & Extending the Quant Engine

Use this guide when you need to add new indicator math, support additional comparison conditions, or update the code generator templates.

---

## 1. How to Add a New Technical Indicator

To introduce a new indicator (e.g., Average Directional Index variant, customized oscillator, or channel boundary) to the automated generator and exporter, follow these four steps:

### Step 1: Register the Enum
Add the indicator key to the [IndicatorType](file:///root/openalgo-autonomous-research/src/quant_engine/models/strategy.py) enum inside `src/quant_engine/models/strategy.py`:
```python
class IndicatorType(str, Enum):
    # ... existing indicators
    MY_NEW_INDICATOR = "my_new_indicator"
```

### Step 2: Implement Calculation Math
In [src/quant_engine/generation/indicators.py](file:///root/openalgo-autonomous-research/src/quant_engine/generation/indicators.py):
1. Define a private calculation function:
   ```python
   def _my_new_indicator(df: pd.DataFrame, src: pd.Series, params: dict) -> pd.Series:
       period = int(params.get("period", 14))
       # Perform pandas/numpy operations
       return df["close"].rolling(window=period).mean() # Example placeholder
   ```
2. Register the function in `INDICATOR_FUNCTIONS`:
   ```python
   INDICATOR_FUNCTIONS = {
       # ...
       IndicatorType.MY_NEW_INDICATOR: _my_new_indicator,
   }
   ```
3. Register parameter ranges for generation in `INDICATOR_PARAM_RANGES`:
   ```python
   INDICATOR_PARAM_RANGES = {
       # ...
       IndicatorType.MY_NEW_INDICATOR: {"period": (5, 50, 5)},
   }
   ```
4. Categorize it (e.g., under `"trend"`, `"momentum"`, `"volatility"`, or `"volume"`) in `INDICATOR_CATEGORIES`.

### Step 3: Implement Code Generator Mapping
To allow exported strategy scripts to compute the new indicator, add its code translation to `_indicator_to_python` in [src/quant_engine/export/formatter.py](file:///root/openalgo-autonomous-research/src/quant_engine/export/formatter.py):
```python
elif node.indicator_type.value == "my_new_indicator":
    period = int(params.get("period", 14))
    return f'df["{name}"] = _compute_my_new_indicator(df["close"], {period})'
```

### Step 4: Define Helper Functions in Templates
Add any custom python calculations to the helper section of the Jinja2 template [signal_strategy.py.j2](file:///root/openalgo-autonomous-research/src/quant_engine/export/templates/signal_strategy.py.j2) and the inline generator fallback `_generate_script_inline` in [formatter.py](file:///root/openalgo-autonomous-research/src/quant_engine/export/formatter.py):
```python
def _compute_my_new_indicator(series: pd.Series, period: int = 14) -> pd.Series:
    # Calculations
    return series.rolling(period).mean()
```

---

## 2. Extending Grammar Rules

Grammar rules define the search space combinations for entry/exit conditions:
* **File**: [src/quant_engine/generation/grammar.py](file:///root/openalgo-autonomous-research/src/quant_engine/generation/grammar.py)
* **Modifying Triggers**: Edit `generate_condition()` to change how indicators are paired (e.g., comparing oscillators vs. constants or comparing trend lines against prices).
* **Exit Logic**: Edit `generate_exit()` to tweak stop-loss and take-profit ratios or trailing stop rules.

---

## 3. Running Engine Tests
After making changes to the models, indicator library, grammar, or export templates, always verify syntax and correctness by running:
```bash
pytest
```
