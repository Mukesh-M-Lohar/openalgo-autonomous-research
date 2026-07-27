# agents/backtest_agent.py
"""Generic BacktestAgent for easy discovery.

This module provides a thin wrapper that re‑exports the concrete ``BacktestAgent``
implementation from the back‑testing package. It is placed in the top‑level
``agents`` directory so that LLM‑driven tooling can locate the back‑testing
agent without having to know the internal back‑testing sub‑package path.

Usage example::

    from agents.backtest_agent import BacktestAgent

    agent = BacktestAgent()
    result = agent.run(symbol="SBIN", exchange="NSE", timeframe="D")
    print(result["summary"])  # DataFrame with performance metrics

The class itself is defined in ``backtesting.ma_ribbon_stochastic.backtest`` and
remains fully functional – all caching, data loading and strategy‑agnostic
behaviour are preserved.
"""

# Re‑export the concrete implementation
from backtesting.ma_ribbon_stochastic.backtest import BacktestAgent
