"""Process pool for parallel backtesting and validation."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class WorkerPool:
    """Manages a process pool for CPU-intensive computation."""

    def __init__(self, max_workers: int | None = None):
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = max(1, cpu_count - 1)
        self._max_workers = max_workers
        self._executor: ProcessPoolExecutor | None = None
        logger.info(f"WorkerPool configured with {self._max_workers} workers")

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def start(self) -> None:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=mp.get_context("spawn"),
            )

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def map_chunks(
        self,
        fn: Callable[[list[T]], list[R]],
        items: list[T],
        chunk_size: int = 200,
    ) -> list[R]:
        """Split items into chunks, process in parallel, collect results."""
        if not items:
            return []

        self.start()
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
        logger.info(
            f"Processing {len(items)} items in {len(chunks)} chunks across {self._max_workers} workers"
        )

        results: list[R] = []
        futures = {self._executor.submit(fn, chunk): idx for idx, chunk in enumerate(chunks)}

        completed = 0
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Completed {completed}/{len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Chunk failed: {e}")
                completed += 1

        return results

    def map_single(
        self,
        fn: Callable[[T], R | None],
        items: list[T],
    ) -> list[R]:
        """Process items one at a time in parallel, filtering None results."""
        if not items:
            return []

        self.start()
        results: list[R] = []
        futures = {self._executor.submit(fn, item): idx for idx, item in enumerate(items)}

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Item processing failed: {e}")

        return results

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
