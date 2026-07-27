"""Agent module for MA Ribbon Stochastic backtesting.

This module provides a convenient entry point to run the backtest
via the :class:`BacktestAgent` defined in ``backtest.py``.
It can be imported by external scripts or notebooks:

```python
from backtesting.ma_ribbon_stochastic.agent import BacktestAgent

agent = BacktestAgent()
result = agent.run(symbol="SBIN", exchange="NSE", timeframe="D")
print(result["summary"])  # pandas DataFrame
```
"""

from .backtest import BacktestAgent

__all__ = ["BacktestAgent"]
