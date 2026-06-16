"""OpenAlgo API client — rate-limited, cached OHLCV data fetching."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
import pandas as pd

from quant_engine.config import OpenAlgoConfig, SymbolConfig
from quant_engine.data.cache import DataCache

logger = logging.getLogger(__name__)

MIN_REQUEST_INTERVAL = 0.11  # ~9 requests/sec (stay under 10/sec limit)


class OpenAlgoClient:
    """Fetches historical OHLCV data from OpenAlgo REST API."""

    def __init__(self, config: OpenAlgoConfig, cache_dir: Path | None = None):
        self._config = config
        self._cache = DataCache(cache_dir or Path("./data/cache"))
        self._last_request_time = 0.0
        self._client = httpx.Client(timeout=60.0)

    def fetch_history(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV data, using cache if available."""
        cached = self._cache.get(symbol, exchange, interval, start_date, end_date)
        if cached is not None:
            logger.info(f"Cache hit: {symbol}/{exchange}/{interval}")
            return cached

        logger.info(f"Fetching from OpenAlgo: {symbol}/{exchange}/{interval} {start_date}->{end_date}")
        self._rate_limit()

        payload = {
            "apikey": self._config.api_key,
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "source": self._config.source,
        }

        url = f"{self._config.host.rstrip('/')}/api/v1/history"
        response = self._client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "success":
            raise ValueError(f"OpenAlgo API error: {data.get('message', 'Unknown error')}")

        df = pd.DataFrame(data["data"])
        if df.empty:
            logger.warning(f"No data returned for {symbol}/{exchange}/{interval}")
            return df

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "oi" in df.columns:
            df["oi"] = pd.to_numeric(df["oi"], errors="coerce").fillna(0)

        self._cache.put(symbol, exchange, interval, start_date, end_date, df)
        return df

    def fetch_all(
        self,
        symbols: list[SymbolConfig],
        timeframes: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Fetch data for all symbol/timeframe combinations.

        Returns: {symbol: {timeframe: DataFrame}}
        """
        result: dict[str, dict[str, pd.DataFrame]] = {}
        total = len(symbols) * len(timeframes)
        fetched = 0

        for sym_cfg in symbols:
            key = f"{sym_cfg.symbol}_{sym_cfg.exchange}"
            result[key] = {}
            for tf in timeframes:
                df = self.fetch_history(
                    symbol=sym_cfg.symbol,
                    exchange=sym_cfg.exchange,
                    interval=tf,
                    start_date=start_date,
                    end_date=end_date,
                )
                result[key][tf] = df
                fetched += 1
                logger.info(f"Data fetch progress: {fetched}/{total}")

        return result

    def _rate_limit(self):
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
