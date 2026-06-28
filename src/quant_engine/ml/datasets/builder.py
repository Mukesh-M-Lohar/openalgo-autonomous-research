"""Dataset Builder — transforms raw OHLCV and trade logs into ML-ready datasets."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from quant_engine.ml.features.extractor import FeatureExtractor
from quant_engine.ml.labels.generator import LabelGenerator

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Combines feature extraction and label generation to build ML datasets."""

    def __init__(
        self,
        feature_config: dict[str, Any] | None = None,
        label_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize DatasetBuilder with feature and label configurations.

        Args:
            feature_config: Configuration dict for FeatureExtractor.
            label_config: Configuration dict for LabelGenerator.
        """
        self.extractor = FeatureExtractor(feature_config)
        self.label_gen = LabelGenerator(label_config)
        self.source = (feature_config or {}).get("source", "ohlcv")

    def build_dataset(
        self, df: pd.DataFrame, trades: list[dict[str, Any]] | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Process OHLCV DataFrame to build aligned feature and label datasets.

        Handles timezone stripping, sorting, alignment, dropping NaNs,
        and filtering for meta-labeling.

        Args:
            df: Raw OHLCV DataFrame.
            trades: Optional trade candidate list (required for meta-labeling).

        Returns:
            A tuple of (X, y) where X is the feature DataFrame and y is the label Series.
        """
        # Normalize index
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # 1. Compute features for all timestamps
        logger.debug("Extracting features from OHLCV data...")
        X = self.extractor.extract_features(df)

        # 2. Compute labels
        logger.debug("Generating target labels...")
        y = self.label_gen.generate_labels(df, trades=trades)

        # 3. Align and filter
        if self.label_gen.label_type == "meta_labeling":
            # For meta-labeling, we only train on trade candidate timestamps.
            # Filter rows where labels are defined (not NaN)
            valid_indices = y.dropna().index
            if len(valid_indices) == 0:
                logger.warning(
                    "No trade candidate timestamps found matching the feature index. Returning empty datasets."
                )
                return pd.DataFrame(columns=X.columns), pd.Series(name="target", dtype=float)

            X_filtered = X.loc[valid_indices]
            y_filtered = y.loc[valid_indices]

            # Drop features columns that are fully NaN/constant
            # and drop rows with remaining NaNs (due to early indicator warmup)
            non_nan_mask = X_filtered.notna().all(axis=1)
            X_clean = X_filtered.loc[non_nan_mask]
            y_clean = y_filtered.loc[non_nan_mask]

            logger.info(
                f"Meta-labeling dataset: {len(y_clean)} trade samples out of {len(trades or [])} trades."
            )
            return X_clean, y_clean
        else:
            # For standard labels (binary, three-class, regression), align and drop NaNs
            # (which occur at the beginning due to indicator warmup and at the end due to forward looking horizon).
            aligned = pd.concat([X, y], axis=1)
            aligned_clean = aligned.dropna()

            X_clean = aligned_clean[X.columns]
            y_clean = aligned_clean["target"]

            logger.info(f"Standard ML dataset: {len(y_clean)} rows.")
            return X_clean, y_clean
