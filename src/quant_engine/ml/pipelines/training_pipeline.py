"""Training Pipeline — runs the full training, tuning, evaluation, and explainability loop."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from quant_engine.ml.datasets.builder import DatasetBuilder
from quant_engine.ml.explainability.explainer import ExplainabilityEngine
from quant_engine.ml.metrics.evaluation import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)
from quant_engine.ml.models.adapters import get_model_adapter
from quant_engine.ml.persistence.registry import ModelRegistry
from quant_engine.ml.tuning.tuner import OptunaTuner

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Sequences data preparation, cross-validation tuning, final model fitting, evaluation, and explainability."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the TrainingPipeline.

        Args:
            config: Full research configuration dict (including machine_learning section).
        """
        self.config = config
        self.ml_config = config.get("machine_learning", {})

        self.dataset_builder = DatasetBuilder(
            feature_config=self.ml_config.get("dataset", {}),
            label_config=self.ml_config.get("labels", {}),
        )

        self.tuner = OptunaTuner(self.ml_config.get("tuning", {}))
        self.explainer = ExplainabilityEngine(self.ml_config.get("explainability", {}))
        self.registry = ModelRegistry(
            os.path.join(self.config.get("output", {}).get("base_dir", "./data/runs"), "models")
        )

    def run(
        self,
        df: pd.DataFrame,
        model_name: str,
        run_id: str,
        version: str = "v1.0",
    ) -> dict[str, Any]:
        """Execute the standard training pipeline.

        Args:
            df: Raw OHLCV DataFrame.
            model_name: Name of the model adapter to train (xgboost, lightgbm, etc.).
            run_id: Active research run ID.
            version: Model version identifier.

        Returns:
            Dictionary containing final validation metrics.
        """
        logger.info(
            f"Running ML Training Pipeline for model '{model_name}' under run '{run_id}'..."
        )

        # 1. Build dataset (extracts features & labels, drops NaNs)
        X, y = self.dataset_builder.build_dataset(df)
        if len(X) < 50:
            logger.error("Dataset is too small to train ML models. Need at least 50 clean samples.")
            return {}

        is_classifier = self.ml_config.get("labels", {}).get("type", "binary") != "regression"

        # 2. Chronological train/validation/test split
        train_pct = self.config.get("data", {}).get("train_pct", 0.7)
        val_pct = self.config.get("data", {}).get("validation_pct", 0.15)

        n_samples = len(X)
        train_end = int(n_samples * train_pct)
        val_end = int(n_samples * (train_pct + val_pct))

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        # Merge train and val for hyperparameter cross-validation search if needed
        X_tune = pd.concat([X_train, X_val])
        y_tune = pd.concat([y_train, y_val])

        # 3. Hyperparameter tuning (if enabled)
        params = {}
        if self.ml_config.get("tuning", {}).get("enabled", False):
            try:
                params = self.tuner.tune(model_name, X_tune, y_tune, is_classifier=is_classifier)
            except Exception as e:
                logger.error(
                    f"Hyperparameter tuning failed, falling back to default parameters: {e}"
                )

        # 4. Train final model on entire training + validation set
        logger.info(f"Training final {model_name} model with parameters: {params}")
        model_config = {
            "is_classifier": is_classifier,
            "params": params,
        }
        adapter = get_model_adapter(model_name, model_config)
        adapter.train(X_tune, y_tune)

        # 5. Predict and evaluate on out-of-sample Test set
        test_preds = adapter.predict(X_test)
        if is_classifier:
            metrics = calculate_classification_metrics(y_test, test_preds)
        else:
            metrics = calculate_regression_metrics(y_test, test_preds)

        logger.info(f"Test performance metrics: {metrics}")

        # 6. Generate explainability plots and reports
        run_output_dir = os.path.join(
            self.config.get("output", {}).get("base_dir", "./data/runs"), run_id, "ml"
        )
        explain_dir = os.path.join(run_output_dir, "explainability", model_name)
        logger.info(f"Generating explainability reports in {explain_dir}...")
        self.explainer.generate_explainability_report(adapter, X_test, y_test, explain_dir)

        # 7. Register model in the registry
        model_registry_dir = os.path.join(run_output_dir, "models")
        self.registry.base_dir = model_registry_dir
        self.registry.save_model(
            model_name=model_name,
            adapter=adapter,
            version=version,
            config=self.config,
            validation_metrics=metrics,
        )

        return metrics
