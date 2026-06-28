"""Unit tests for OptunaTuner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_engine.ml.tuning.tuner import OptunaTuner


def test_optuna_tuner() -> None:
    np.random.seed(42)
    # Generate some dummy data where f0 is directly predictive of y
    X = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=[f"f{i}" for i in range(3)])
    y = (X["f0"] > 0).astype(int)

    config = {
        "trials": 3,
        "cv_folds": 2,
        "objectives": ["accuracy"],
    }
    tuner = OptunaTuner(config)
    best_params = tuner.tune("random_forest", X, y, is_classifier=True)

    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
