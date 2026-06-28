"""Model Registry — tracks, versions, serializes, and deserializes ML models."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from typing import Any

from quant_engine.ml.models.adapters import get_model_adapter

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages the lifecycle, serialization, and retrieval of trained model adapters."""

    def __init__(self, base_dir: str = "./data/models") -> None:
        """Initialize the ModelRegistry with a storage path.

        Args:
            base_dir: Central path where versioned models will be stored.
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_model(
        self,
        model_name: str,
        adapter: Any,
        version: str,
        config: dict[str, Any],
        validation_metrics: dict[str, Any],
    ) -> str:
        """Serialize a model adapter and write its metadata to the registry.

        Args:
            model_name: Base identifier for the model.
            adapter: The trained model adapter instance.
            version: Version string (e.g. "v1.0", "run_c3a8").
            config: Configuration dictionary.
            validation_metrics: Evaluation metrics dictionary.

        Returns:
            The directory path where the model was saved.
        """
        model_dir = os.path.join(self.base_dir, model_name, version)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.bin")
        logger.info(f"Saving model to {model_path}...")
        adapter.save(model_path)

        # Build metadata
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        metadata = {
            "model_name": model_name,
            "version": version,
            "training_date": datetime.datetime.now().isoformat(),
            "feature_names": adapter.feature_names,
            "is_classifier": getattr(adapter, "is_classifier", True),
            "config_hash": config_hash,
            "validation_metrics": validation_metrics,
            "adapter_class": adapter.__class__.__name__,
        }

        meta_path = os.path.join(model_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Model and metadata saved successfully under {model_dir}")
        return model_dir

    def load_model(self, model_name: str, version: str) -> Any:
        """Retrieve and deserialize a model adapter from the registry.

        Args:
            model_name: Base identifier of the model.
            version: Version string.

        Returns:
            An instantiated and loaded BaseModel adapter.
        """
        model_dir = os.path.join(self.base_dir, model_name, version)
        meta_path = os.path.join(model_dir, "metadata.json")
        model_path = os.path.join(model_dir, "model.bin")

        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model version {version} for {model_name} not found in registry."
            )

        with open(meta_path) as f:
            metadata = json.load(f)

        adapter_class_name = metadata.get("adapter_class", "XGBoostAdapter")
        is_classifier = metadata.get("is_classifier", True)

        # Map class names back to string flags for adapter instantiation
        cls_map = {
            "XGBoostAdapter": "xgboost",
            "LightGBMAdapter": "lightgbm",
            "CatBoostAdapter": "catboost",
            "RandomForestAdapter": "random_forest",
            "ExtraTreesAdapter": "extra_trees",
        }
        adapter_name = cls_map.get(adapter_class_name, "lightgbm")

        model_config = {
            "is_classifier": is_classifier,
            "params": {},
        }
        adapter = get_model_adapter(adapter_name, model_config)
        adapter.load(model_path)
        adapter.feature_names = metadata.get("feature_names", [])

        logger.info(f"Model loaded successfully from {model_dir}")
        return adapter

    def list_models(self) -> dict[str, list[str]]:
        """List all models and their versions currently present in the registry.

        Returns:
            Dict mapping model names to lists of version strings.
        """
        models = {}
        if not os.path.exists(self.base_dir):
            return models

        for model_name in os.listdir(self.base_dir):
            model_path = os.path.join(self.base_dir, model_name)
            if os.path.isdir(model_path):
                versions = []
                for version in os.listdir(model_path):
                    if os.path.isdir(os.path.join(model_path, version)):
                        versions.append(version)
                models[model_name] = sorted(versions)
        return models
