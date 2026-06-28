"""Unit and integration tests for HybridPipeline."""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pandas as pd

from quant_engine.ml.pipelines.hybrid_pipeline import HybridPipeline


def test_hybrid_pipeline() -> None:
    # 1. Create fake OHLCV data
    dates = pd.date_range(start="2026-01-01", periods=300)
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, 300))
    df = pd.DataFrame(
        {
            "open": close - np.random.normal(0, 0.2, 300),
            "high": close + np.random.uniform(0.1, 0.5, 300),
            "low": close - np.random.uniform(0.1, 0.5, 300),
            "close": close,
            "volume": 5000.0,
        },
        index=dates,
    )

    # 2. Create fake trades (e.g. 30 trades)
    np.random.seed(42)
    trades = []
    for idx in range(120, 280, 5):
        entry_time = dates[idx].isoformat()
        # profitable trades will have positive pnl, losing will have negative pnl
        pnl = 1.0 + float(np.random.normal(0, 1.5))
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": dates[idx + 2].isoformat(),
                "entry_price": float(close[idx]),
                "exit_price": float(close[idx + 2]),
                "pnl_pct": pnl,
            }
        )

    # 3. Instantiate pipeline
    temp_dir = tempfile.mkdtemp()
    try:
        config = {
            "output": {"base_dir": temp_dir},
            "ranking": {"export_top_n": 1},
            "machine_learning": {
                "enabled": True,
                "dataset": {"source": "generated_trades"},
                "labels": {"type": "meta_labeling", "threshold": 0.0},
                "models": ["random_forest"],
                "tuning": {"enabled": False},
                "deployment": {"confidence_threshold": 0.50},
            },
        }

        pipeline = HybridPipeline(config)
        comparison = pipeline.run_hybrid_filtering(
            strategy_id="strat_test",
            df=df,
            trades=trades,
            model_name="random_forest",
            run_id="run_test",
        )

        assert isinstance(comparison, dict)
        assert "original" in comparison
        assert "hybrid" in comparison
        assert "comparison" in comparison
        assert comparison["strategy_id"] == "strat_test"
    finally:
        shutil.rmtree(temp_dir)
