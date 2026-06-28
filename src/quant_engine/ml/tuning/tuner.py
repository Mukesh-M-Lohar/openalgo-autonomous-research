"""Optuna Tuner — hyperparameter optimization using time-series cross validation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quant_engine.ml.models.adapters import get_model_adapter

logger = logging.getLogger(__name__)


class OptunaTuner:
    """Orchestrates Bayesian hyperparameter optimization for ML models using Optuna."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the OptunaTuner.

        Args:
            config: Tuning configuration.
        """
        self.config = config or {}
        self.trials = int(self.config.get("trials", 50))
        self.cv_folds = int(self.config.get("cv_folds", 3))
        self.objectives = self.config.get("objectives", ["accuracy"])

    def tune(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        is_classifier: bool = True,
    ) -> dict[str, Any]:
        """Run Optuna search to find optimal hyperparameters.

        Args:
            model_name: Name of the model adapter (lightgbm, xgboost, catboost, etc.).
            X: Training features.
            y: Training targets.
            is_classifier: Whether to tune classification or regression models.

        Returns:
            Dictionary of best hyperparameters found.
        """
        try:
            import optuna

            # Disable optuna logs unless debugging
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as e:
            raise ImportError(
                "Optuna is not installed. Install it using: pip install optuna"
            ) from e

        # 1. Define objectives
        if len(self.objectives) > 1:
            directions = [
                "maximize" if obj in ["accuracy", "f1", "precision", "recall"] else "minimize"
                for obj in self.objectives
            ]
            study = optuna.create_study(directions=directions)
        else:
            direction = (
                "maximize"
                if self.objectives[0] in ["accuracy", "f1", "precision", "recall"]
                else "minimize"
            )
            study = optuna.create_study(direction=direction)

        # 2. Objective function for trials
        def objective_func(trial: optuna.Trial) -> float | tuple[float, ...]:
            params = self._suggest_params(trial, model_name, is_classifier)

            # CV fold splits (chronological TimeSeriesSplit style to prevent leakage)
            fold_size = len(X) // (self.cv_folds + 1)
            scores = []

            for i in range(self.cv_folds):
                train_end = fold_size * (i + 1)
                val_end = fold_size * (i + 2)

                X_train = X.iloc[:train_end]
                y_train = y.iloc[:train_end]
                X_val = X.iloc[train_end:val_end]
                y_val = y.iloc[train_end:val_end]

                if len(X_train) < 10 or len(X_val) < 5:
                    continue

                # Add early stopping if supported and data splits are correct
                model_config = {
                    "is_classifier": is_classifier,
                    "params": params,
                }
                adapter = get_model_adapter(model_name, model_config)

                try:
                    adapter.train(X_train, y_train)
                    preds = adapter.predict(X_val)

                    # Evaluate metrics
                    if is_classifier:
                        from sklearn.metrics import (
                            accuracy_score,
                            f1_score,
                            precision_score,
                            recall_score,
                        )

                        # If labels are multiclass, handle average
                        avg = "binary" if len(np.unique(y_train)) <= 2 else "macro"

                        acc = accuracy_score(y_val, preds)
                        f1 = f1_score(y_val, preds, average=avg, zero_division=0)
                        prec = precision_score(y_val, preds, average=avg, zero_division=0)
                        rec = recall_score(y_val, preds, average=avg, zero_division=0)

                        metrics_map = {
                            "accuracy": acc,
                            "f1": f1,
                            "precision": prec,
                            "recall": rec,
                        }
                    else:
                        from sklearn.metrics import mean_squared_error

                        mse = mean_squared_error(y_val, preds)
                        rmse = np.sqrt(mse)
                        metrics_map = {
                            "mse": mse,
                            "rmse": rmse,
                        }

                    scores.append([metrics_map.get(obj, 0.0) for obj in self.objectives])
                except Exception as ex:
                    logger.debug(f"Trial fail: {ex}")
                    # Penalize failed trial
                    scores.append(
                        [0.0 if "maximize" in self.objectives else 999.0 for _ in self.objectives]
                    )

            if not scores:
                return 0.0 if len(self.objectives) == 1 else tuple(0.0 for _ in self.objectives)

            # Average score across CV folds
            mean_scores = np.mean(scores, axis=0)
            if len(self.objectives) == 1:
                return float(mean_scores[0])
            return tuple(mean_scores)

        # 3. Run study
        logger.info(f"Tuning {model_name} for {self.trials} trials...")
        study.optimize(objective_func, n_trials=self.trials)

        # 4. Return results
        if len(self.objectives) > 1:
            # Multi-objective returns pareto front trials. We select the one with best first objective
            best_trial = min(study.best_trials, key=lambda t: t.values[0])
            best_params = best_trial.params
            logger.info(f"Multi-objective tuning completed. Best trial values: {best_trial.values}")
        else:
            best_params = study.best_value
            best_params = study.best_trial.params
            logger.info(f"Tuning completed. Best CV score: {study.best_value:.4f}")

        return best_params

    def _suggest_params(self, trial: Any, model_name: str, is_classifier: bool) -> dict[str, Any]:
        """Define parameter grids for each supported model."""
        clean_name = model_name.lower().replace("-", "_").replace(" ", "_")

        if clean_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            }
        elif clean_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }
        elif clean_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 50, 300, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }
        elif clean_name in ["random_forest", "extra_trees"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
        else:
            params = {}

        return params
