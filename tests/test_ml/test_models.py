"""Unit tests for ML model adapters."""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from quant_engine.ml.models.adapters import get_model_adapter


@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    X = pd.DataFrame(np.random.normal(0, 1, (100, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y


@pytest.mark.parametrize(
    "model_name",
    ["random_forest", "extra_trees", "lightgbm", "xgboost", "catboost"],
)
def test_model_adapters_classification(
    model_name: str,
    sample_classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = sample_classification_data

    # Instantiate adapter
    config = {
        "is_classifier": True,
        # Keep n_estimators small for speed
        "params": {"n_estimators": 5, "max_depth": 3}
        if model_name != "catboost"
        else {"iterations": 5, "depth": 3},
    }
    adapter = get_model_adapter(model_name, config)

    # Train
    adapter.train(X, y)
    assert len(adapter.feature_names) == 5

    # Predict
    preds = adapter.predict(X)
    assert len(preds) == 100
    assert set(np.unique(preds)).issubset({0, 1})

    # Predict Proba
    probs = adapter.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all((probs >= 0) & (probs <= 1))

    # Serialization test
    temp_dir = tempfile.mkdtemp()
    try:
        model_path = os.path.join(temp_dir, "model.bin")
        adapter.save(model_path)

        # Load into new adapter
        loaded_adapter = get_model_adapter(model_name, {"is_classifier": True})
        loaded_adapter.load(model_path)

        assert loaded_adapter.is_classifier is True
        assert len(loaded_adapter.feature_names) == 5

        # Check predictions match
        loaded_preds = loaded_adapter.predict(X)
        np.testing.assert_array_equal(preds, loaded_preds)
    finally:
        shutil.rmtree(temp_dir)
