"""Data preprocessing — splits, alignment, multi-timeframe aggregation."""

from __future__ import annotations

import pandas as pd


class DataPreprocessor:
    """Handles data splitting and multi-timeframe alignment."""

    def __init__(
        self, train_pct: float = 0.7, validation_pct: float = 0.15, test_pct: float = 0.15
    ):
        self._train_pct = train_pct
        self._validation_pct = validation_pct
        self._test_pct = test_pct

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets chronologically."""
        n = len(df)
        train_end = int(n * self._train_pct)
        val_end = train_end + int(n * self._validation_pct)

        train = df.iloc[:train_end]
        validation = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]
        return train, validation, test

    def rolling_windows(
        self, df: pd.DataFrame, train_bars: int, test_bars: int, step_bars: int | None = None
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate rolling train/test windows for walk-forward analysis."""
        if step_bars is None:
            step_bars = test_bars

        windows = []
        start = 0
        while start + train_bars + test_bars <= len(df):
            train = df.iloc[start : start + train_bars]
            test = df.iloc[start + train_bars : start + train_bars + test_bars]
            windows.append((train, test))
            start += step_bars

        return windows

    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample OHLCV data to a higher timeframe."""
        tf_map = {
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
            "1w": "1W",
        }
        rule = tf_map.get(target_tf, target_tf)
        resampled = (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        return resampled

    @staticmethod
    def align_timeframes(base_df: pd.DataFrame, higher_tf_df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill higher timeframe data to align with base timeframe index."""
        return higher_tf_df.reindex(base_df.index, method="ffill")
