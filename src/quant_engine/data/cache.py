"""Local file cache for OHLCV data to avoid repeated API calls."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Caches OHLCV DataFrames as Parquet files on disk."""

    def __init__(self, cache_dir: Path):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, symbol: str, exchange: str, interval: str, start: str, end: str) -> str:
        raw = f"{symbol}_{exchange}_{interval}_{start}_{end}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.parquet"

    def get(
        self, symbol: str, exchange: str, interval: str, start: str, end: str
    ) -> pd.DataFrame | None:
        key = self._key(symbol, exchange, interval, start, end)
        path = self._path(key)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logger.warning(f"Cache read failed for {path}: {e}")
                path.unlink(missing_ok=True)
        return None

    def put(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: str,
        end: str,
        df: pd.DataFrame,
    ) -> None:
        if df.empty:
            return
        key = self._key(symbol, exchange, interval, start, end)
        path = self._path(key)
        try:
            df.to_parquet(path)
        except Exception as e:
            logger.warning(f"Cache write failed for {path}: {e}")

    def clear(self) -> int:
        count = 0
        for f in self._dir.glob("*.parquet"):
            f.unlink()
            count += 1
        return count
