"""Feature Extractor — transforms OHLCV data into dynamic ML feature matrices."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quant_engine.generation.indicators import compute_indicator
from quant_engine.models.strategy import IndicatorType, PriceSource

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Computes technical, candle, volatility, volume, time, session, gap, and rolling features."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize FeatureExtractor with a configuration.

        Args:
            config: Feature configuration flags.
        """
        self.config = config or {}
        # Set default values if not explicitly provided
        self.technical_enabled = self.config.get("technical", True)
        self.candle_enabled = self.config.get("candle", True)
        self.volatility_enabled = self.config.get("volatility", True)
        self.volume_enabled = self.config.get("volume", True)
        self.time_enabled = self.config.get("time", True)
        self.session_enabled = self.config.get("session", True)
        self.gap_enabled = self.config.get("gap", True)
        self.rolling_enabled = self.config.get("rolling", True)
        self.rolling_windows = self.config.get("rolling_windows", [5, 10, 20, 50])

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and append all enabled features from the input OHLCV DataFrame.

        Args:
            df: Pandas DataFrame with datetime index and columns (open, high, low, close, volume).

        Returns:
            A new DataFrame containing the feature matrix.
        """
        features_df = pd.DataFrame(index=df.index)

        # Standard OHLC data reference
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        if self.candle_enabled:
            features_df["feat_candle_body"] = (close - open_) / open_.replace(0, np.nan)
            features_df["feat_candle_range"] = (high - low) / open_.replace(0, np.nan)
            features_df["feat_candle_upper_shadow"] = (
                high - np.maximum(open_, close)
            ) / open_.replace(0, np.nan)
            features_df["feat_candle_lower_shadow"] = (
                np.minimum(open_, close) - low
            ) / open_.replace(0, np.nan)

        if self.gap_enabled:
            # Overnight / candle gaps
            features_df["feat_gap_overnight"] = np.log(open_ / close.shift(1).replace(0, np.nan))

        if self.technical_enabled:
            # EMAs and SMAs
            for p in [10, 20, 50, 100]:
                features_df[f"feat_tech_sma_{p}_ratio"] = (
                    close
                    / compute_indicator(
                        df, IndicatorType.SMA, {"period": p}, PriceSource.CLOSE
                    ).replace(0, np.nan)
                    - 1.0
                )
                features_df[f"feat_tech_ema_{p}_ratio"] = (
                    close
                    / compute_indicator(
                        df, IndicatorType.EMA, {"period": p}, PriceSource.CLOSE
                    ).replace(0, np.nan)
                    - 1.0
                )

            # RSI
            for p in [7, 14, 21]:
                features_df[f"feat_tech_rsi_{p}"] = compute_indicator(
                    df, IndicatorType.RSI, {"period": p}, PriceSource.CLOSE
                )

            # MACD
            macd_val = compute_indicator(
                df,
                IndicatorType.MACD,
                {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                PriceSource.CLOSE,
            )
            macd_signal = compute_indicator(
                df,
                IndicatorType.MACD_SIGNAL,
                {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                PriceSource.CLOSE,
            )
            macd_hist = compute_indicator(
                df,
                IndicatorType.MACD_HIST,
                {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                PriceSource.CLOSE,
            )
            features_df["feat_tech_macd_ratio"] = macd_val / close.replace(0, np.nan)
            features_df["feat_tech_macd_signal_ratio"] = macd_signal / close.replace(0, np.nan)
            features_df["feat_tech_macd_hist_ratio"] = macd_hist / close.replace(0, np.nan)

            # ADX
            features_df["feat_tech_adx_14"] = compute_indicator(
                df, IndicatorType.ADX, {"period": 14}, PriceSource.CLOSE
            )

            # ATR
            features_df["feat_tech_atr_14"] = compute_indicator(
                df, IndicatorType.ATR, {"period": 14}, PriceSource.CLOSE
            )

            # SuperTrend
            st = compute_indicator(
                df, IndicatorType.SUPERTREND, {"period": 10, "multiplier": 3.0}, PriceSource.CLOSE
            )
            features_df["feat_tech_supertrend_ratio"] = close / st.replace(0, np.nan) - 1.0

            # CCI and ROC
            features_df["feat_tech_cci_20"] = compute_indicator(
                df, IndicatorType.CCI, {"period": 20}, PriceSource.CLOSE
            )
            features_df["feat_tech_roc_12"] = compute_indicator(
                df, IndicatorType.ROC, {"period": 12}, PriceSource.CLOSE
            )
            features_df["feat_tech_momentum_10"] = compute_indicator(
                df, IndicatorType.MOMENTUM, {"period": 10}, PriceSource.CLOSE
            ) / close.replace(0, np.nan)

            # VWAP
            vw = compute_indicator(df, IndicatorType.VWAP, {}, PriceSource.CLOSE)
            features_df["feat_tech_vwap_ratio"] = close / vw.replace(0, np.nan) - 1.0

        if self.volatility_enabled:
            # Normalized ATR
            atr_14 = compute_indicator(df, IndicatorType.ATR, {"period": 14}, PriceSource.CLOSE)
            features_df["feat_vol_natr"] = atr_14 / close.replace(0, np.nan)

            # Rolling returns standard deviation
            returns = close.pct_change()
            for w in [10, 20, 30]:
                features_df[f"feat_vol_std_{w}"] = returns.rolling(window=w).std()

            # Bollinger Bands width and percent B
            bb_upper = compute_indicator(
                df, IndicatorType.BBANDS_UPPER, {"period": 20, "std_dev": 2.0}, PriceSource.CLOSE
            )
            bb_mid = compute_indicator(
                df, IndicatorType.BBANDS_MIDDLE, {"period": 20}, PriceSource.CLOSE
            )
            bb_lower = compute_indicator(
                df, IndicatorType.BBANDS_LOWER, {"period": 20, "std_dev": 2.0}, PriceSource.CLOSE
            )
            features_df["feat_vol_bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
            features_df["feat_vol_bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower).replace(
                0, np.nan
            )

            # Donchian width
            don_upper = compute_indicator(
                df, IndicatorType.DONCHIAN_UPPER, {"period": 20}, PriceSource.CLOSE
            )
            don_lower = compute_indicator(
                df, IndicatorType.DONCHIAN_LOWER, {"period": 20}, PriceSource.CLOSE
            )
            features_df["feat_vol_donchian_pct"] = (close - don_lower) / (
                don_upper - don_lower
            ).replace(0, np.nan)

        if self.volume_enabled:
            features_df["feat_volume_log_return"] = np.log(
                volume / volume.shift(1).replace(0, np.nan)
            )
            # Volume to its rolling SMAs
            for w in [10, 20, 50]:
                vol_sma = volume.rolling(window=w).mean()
                features_df[f"feat_volume_sma_{w}_ratio"] = volume / vol_sma.replace(0, np.nan)
            # OBV
            obv = compute_indicator(df, IndicatorType.OBV, {}, PriceSource.CLOSE)
            features_df["feat_volume_obv_ratio"] = (
                obv / obv.rolling(50).mean().replace(0, np.nan) - 1.0
            )

        if self.time_enabled:
            if isinstance(df.index, pd.DatetimeIndex):
                features_df["feat_time_hour"] = df.index.hour
                features_df["feat_time_dayofweek"] = df.index.dayofweek
                features_df["feat_time_month"] = df.index.month
                features_df["feat_time_is_weekend"] = (df.index.dayofweek >= 5).astype(int)
            else:
                logger.warning("Time features enabled but DataFrame index is not DatetimeIndex.")

        if self.session_enabled:
            if isinstance(df.index, pd.DatetimeIndex):
                # Standard sessions: morning (9:15 - 11:30), midday (11:30 - 13:30), afternoon (13:30 - 15:30)
                hours = df.index.hour + df.index.minute / 60.0
                features_df["feat_session_morning"] = ((hours >= 9.25) & (hours < 11.5)).astype(int)
                features_df["feat_session_midday"] = ((hours >= 11.5) & (hours < 13.5)).astype(int)
                features_df["feat_session_afternoon"] = (hours >= 13.5).astype(int)
            else:
                logger.warning("Session features enabled but DataFrame index is not DatetimeIndex.")

        if self.rolling_enabled:
            for w in self.rolling_windows:
                # Rolling stats of close price
                features_df[f"feat_roll_close_mean_{w}"] = (
                    close / close.rolling(w).mean().replace(0, np.nan) - 1.0
                )
                features_df[f"feat_roll_close_std_{w}"] = close.rolling(w).std() / close.replace(
                    0, np.nan
                )
                features_df[f"feat_roll_close_min_max_{w}"] = (close - close.rolling(w).min()) / (
                    close.rolling(w).max() - close.rolling(w).min()
                ).replace(0, np.nan)

                # Rolling stats of volume
                features_df[f"feat_roll_volume_mean_{w}"] = volume / volume.rolling(
                    w
                ).mean().replace(0, np.nan)

        # Clean infs/nans
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        return features_df
