"""Model adapters for LightGBM, XGBoost, CatBoost, and scikit-learn models."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd

from quant_engine.ml.models.base import BaseModel


class XGBoostAdapter(BaseModel):
    """Adapter for XGBoost classifier and regressor."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.is_classifier = self.config.get("is_classifier", True)
        self.params = self.config.get("params", {})

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "XGBoost is not installed. Install it using: pip install xgboost"
            ) from e

        self.feature_names = list(X.columns)
        if self.is_classifier:
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        if not self.is_classifier:
            raise AttributeError("predict_proba is not available for regression models.")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        # Save both model and feature names/is_classifier metadata
        data = {
            "is_classifier": self.is_classifier,
            "feature_names": self.feature_names,
            "params": self.params,
        }
        with open(path + ".meta", "wb") as f:
            pickle.dump(data, f)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "XGBoost is not installed. Install it using: pip install xgboost"
            ) from e

        with open(path + ".meta", "rb") as f:
            data = pickle.load(f)
        self.is_classifier = data["is_classifier"]
        self.feature_names = data["feature_names"]
        self.params = data["params"]

        if self.is_classifier:
            self.model = xgb.XGBClassifier()
        else:
            self.model = xgb.XGBRegressor()
        self.model.load_model(path)


class LightGBMAdapter(BaseModel):
    """Adapter for LightGBM classifier and regressor."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.is_classifier = self.config.get("is_classifier", True)
        self.params = self.config.get("params", {})
        # Suppress lightgbm warnings during tuning unless requested
        if "verbose" not in self.params:
            self.params["verbose"] = -1

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "LightGBM is not installed. Install it using: pip install lightgbm"
            ) from e

        self.feature_names = list(X.columns)
        if self.is_classifier:
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        if not self.is_classifier:
            raise AttributeError("predict_proba is not available for regression models.")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        # LightGBM booster is highly compatible, but we can also pickle the whole wrapper
        with open(path, "wb") as f:
            pickle.dump((self.is_classifier, self.feature_names, self.params, self.model), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.is_classifier, self.feature_names, self.params, self.model = pickle.load(f)


class CatBoostAdapter(BaseModel):
    """Adapter for CatBoost classifier and regressor."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.is_classifier = self.config.get("is_classifier", True)
        self.params = self.config.get("params", {})
        if "verbose" not in self.params:
            self.params["verbose"] = 0

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            import catboost as cb
        except ImportError as e:
            raise ImportError(
                "CatBoost is not installed. Install it using: pip install catboost"
            ) from e

        self.feature_names = list(X.columns)
        if self.is_classifier:
            self.model = cb.CatBoostClassifier(**self.params)
        else:
            self.model = cb.CatBoostRegressor(**self.params)

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        if not self.is_classifier:
            raise AttributeError("predict_proba is not available for regression models.")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        data = {
            "is_classifier": self.is_classifier,
            "feature_names": self.feature_names,
            "params": self.params,
        }
        with open(path + ".meta", "wb") as f:
            pickle.dump(data, f)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        try:
            import catboost as cb
        except ImportError as e:
            raise ImportError(
                "CatBoost is not installed. Install it using: pip install catboost"
            ) from e

        with open(path + ".meta", "rb") as f:
            data = pickle.load(f)
        self.is_classifier = data["is_classifier"]
        self.feature_names = data["feature_names"]
        self.params = data["params"]

        if self.is_classifier:
            self.model = cb.CatBoostClassifier()
        else:
            self.model = cb.CatBoostRegressor()
        self.model.load_model(path)


class RandomForestAdapter(BaseModel):
    """Adapter for scikit-learn RandomForest classifier and regressor."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.is_classifier = self.config.get("is_classifier", True)
        self.params = self.config.get("params", {})

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError as e:
            raise ImportError(
                "scikit-learn is not installed. Install it using: pip install scikit-learn"
            ) from e

        self.feature_names = list(X.columns)
        if self.is_classifier:
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        if not self.is_classifier:
            raise AttributeError("predict_proba is not available for regression models.")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        with open(path, "wb") as f:
            pickle.dump((self.is_classifier, self.feature_names, self.params, self.model), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.is_classifier, self.feature_names, self.params, self.model = pickle.load(f)


class ExtraTreesAdapter(BaseModel):
    """Adapter for scikit-learn ExtraTrees classifier and regressor."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.is_classifier = self.config.get("is_classifier", True)
        self.params = self.config.get("params", {})

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        except ImportError as e:
            raise ImportError(
                "scikit-learn is not installed. Install it using: pip install scikit-learn"
            ) from e

        self.feature_names = list(X.columns)
        if self.is_classifier:
            self.model = ExtraTreesClassifier(**self.params)
        else:
            self.model = ExtraTreesRegressor(**self.params)

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        if not self.is_classifier:
            raise AttributeError("predict_proba is not available for regression models.")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        with open(path, "wb") as f:
            pickle.dump((self.is_classifier, self.feature_names, self.params, self.model), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.is_classifier, self.feature_names, self.params, self.model = pickle.load(f)


# Registry helper to instantiate adapters by string name
def get_model_adapter(name: str, config: dict[str, Any] | None = None) -> BaseModel:
    """Instantiate a model adapter by name.

    Args:
        name: Name of the model (xgboost, lightgbm, catboost, random_forest, extra_trees).
        config: Configurations dictionary.
    """
    clean_name = name.lower().replace("-", "_").replace(" ", "_")
    adapters = {
        "xgboost": XGBoostAdapter,
        "lightgbm": LightGBMAdapter,
        "catboost": CatBoostAdapter,
        "random_forest": RandomForestAdapter,
        "extra_trees": ExtraTreesAdapter,
    }
    adapter_cls = adapters.get(clean_name)
    if adapter_cls is None:
        raise ValueError(f"Unknown model adapter name: {name}")
    return adapter_cls(config)
