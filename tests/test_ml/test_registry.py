"""Unit tests for ModelRegistry."""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pandas as pd

from quant_engine.ml.models.adapters import get_model_adapter
from quant_engine.ml.persistence.registry import ModelRegistry


def test_model_registry_lifecycle() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        registry = ModelRegistry(temp_dir)

        # Create and train a dummy adapter
        X = pd.DataFrame(np.random.normal(0, 1, (10, 3)), columns=["a", "b", "c"])
        y = pd.Series([0, 1] * 5)
        adapter = get_model_adapter(
            "random_forest", {"is_classifier": True, "params": {"n_estimators": 2}}
        )
        adapter.train(X, y)

        # Save
        version = "v1.0.test"
        config = {"some_setting": True}
        metrics = {"accuracy": 0.9}
        registry.save_model("rf_test", adapter, version, config, metrics)

        # List models
        models = registry.list_models()
        assert "rf_test" in models
        assert version in models["rf_test"]

        # Load model back
        loaded = registry.load_model("rf_test", version)
        assert loaded.is_classifier is True
        assert loaded.feature_names == ["a", "b", "c"]

        # Make predictions to verify functional state
        preds = loaded.predict(X)
        assert len(preds) == 10
    finally:
        shutil.rmtree(temp_dir)
