"""Abstract base model interface for ML adapters in the quant-engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all machine learning model adapters."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the model adapter.

        Args:
            config: Model hyperparameters and adapter configurations.
        """
        self.config = config or {}
        self.model: Any = None
        self.feature_names: list[str] = []

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the machine learning model on feature matrix X and label series y.

        Args:
            X: Pandas DataFrame of shape (n_samples, n_features) containing features.
            y: Pandas Series of shape (n_samples,) containing training labels.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the input feature matrix X.

        Args:
            X: Pandas DataFrame containing feature columns.

        Returns:
            Numpy array of shape (n_samples,) containing predicted classes or regression values.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities for the input feature matrix X.

        Only applicable for classification models.

        Args:
            X: Pandas DataFrame containing feature columns.

        Returns:
            Numpy array of shape (n_samples, n_classes) containing class probabilities.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the serialized model to the specified file path.

        Args:
            path: Absolute path to the destination file.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the serialized model from the specified file path.

        Args:
            path: Absolute path to the source model file.
        """
        pass
