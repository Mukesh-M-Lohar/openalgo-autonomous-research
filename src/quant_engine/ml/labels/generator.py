"""Label Generator — calculates target labels for ML datasets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generates target labels from OHLCV close series or backtest trade lists."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize LabelGenerator with a configuration.

        Args:
            config: Label configuration parameters.
        """
        self.config = config or {}
        self.label_type = self.config.get("type", "binary")
        self.future_horizon = int(self.config.get("future_horizon", 10))
        self.threshold = float(self.config.get("threshold", 0.005))

    def generate_labels(
        self, df: pd.DataFrame, trades: list[dict[str, Any]] | None = None
    ) -> pd.Series:
        """Compute the label series aligned with the index of the feature DataFrame.

        Args:
            df: The feature matrix or raw OHLCV DataFrame.
            trades: Optional trade candidate logs (required for meta-labeling).

        Returns:
            A pandas Series containing target labels.
        """
        close = df["close"]

        if self.label_type == "binary":
            # Future return = close(t + horizon) / close(t) - 1
            future_return = close.shift(-self.future_horizon) / close - 1.0
            labels = (future_return > self.threshold).astype(int)
            # Mark the last future_horizon elements as NaN because they look into the future beyond dataset boundaries
            labels.iloc[-self.future_horizon :] = np.nan
            return pd.Series(labels, index=df.index, name="target")

        elif self.label_type == "three_class":
            future_return = close.shift(-self.future_horizon) / close - 1.0
            labels = pd.Series(1, index=df.index)  # Default class 1: Hold
            labels[future_return > self.threshold] = 2  # Buy
            labels[future_return < -self.threshold] = 0  # Sell
            labels.iloc[-self.future_horizon :] = np.nan
            return pd.Series(labels, index=df.index, name="target")

        elif self.label_type == "regression":
            # Future log returns
            future_return = np.log(close.shift(-self.future_horizon) / close)
            future_return.iloc[-self.future_horizon :] = np.nan
            return pd.Series(future_return, index=df.index, name="target")

        elif self.label_type == "meta_labeling":
            if not trades:
                logger.warning(
                    "Meta-labeling requested but no trades list provided. Returning empty Series."
                )
                return pd.Series(np.nan, index=df.index, name="target")

            # Initialize series with NaN (representing no trade candidate present)
            labels = pd.Series(np.nan, index=df.index, name="target")

            # Find matching timestamp index for each trade and label it
            for trade in trades:
                entry_time = trade.get("entry_time")
                if entry_time is None:
                    continue

                # Ensure entry_time is a timestamp matching the df index type
                entry_time_ts = pd.to_datetime(entry_time)

                # Strip timezone if necessary to match index
                if df.index.tz is not None and entry_time_ts.tz is None:
                    entry_time_ts = entry_time_ts.tz_localize("UTC")
                elif df.index.tz is None and entry_time_ts.tz is not None:
                    entry_time_ts = entry_time_ts.tz_convert(None)

                # If entry time is not exactly in index, find the nearest preceding index
                if entry_time_ts in df.index:
                    idx_match = entry_time_ts
                else:
                    # Find nearest index
                    idx_match_pos = df.index.get_indexer([entry_time_ts], method="pad")[0]
                    if idx_match_pos == -1:
                        continue
                    idx_match = df.index[idx_match_pos]

                pnl = float(trade.get("pnl_pct", 0.0))
                # Label is 1 (success) if pnl is positive, 0 (failure) otherwise
                labels.loc[idx_match] = 1 if pnl > self.threshold * 100 else 0

            return labels

        else:
            raise ValueError(f"Unknown label strategy: {self.label_type}")
