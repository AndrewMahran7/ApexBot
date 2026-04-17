"""
Unified bar data API.
=======================

Single entry-point for all bar access patterns:

  * ``get_last_n_bars(n)``   — most recent *n* bars from the store
  * ``stream_live(on_bar)``  — live Databento feed → store + callback
  * ``replay(start, end)``   — deterministic playback from the store

All three return the same :class:`Bar` objects, so downstream code
(strategy, engine, analytics) works identically regardless of source.

Usage:
    from data.bar_api import BarAPI

    api = BarAPI("data/bars.db")

    # Recent data
    last_20 = api.get_last_n_bars(20)

    # Replay
    bars = api.replay("2024-01-02T09:30:00", "2024-01-02T16:00:00",
                       on_bar=my_callback)

    # Live (blocking)
    api.stream_live(on_bar=my_callback)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from data.data_pipeline import Bar, LivePipeline
from data.storage import BarStore
from data.replay import ReplayEngine

logger = logging.getLogger(__name__)


class BarAPI:
    """
    Unified interface for bar storage, retrieval, replay, and live streaming.

    Parameters
    ----------
    db_path : str or Path
        SQLite database path.  Use ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str | Path = "data/bars.db") -> None:
        self._store = BarStore(db_path)
        self._replay_engine = ReplayEngine(self._store)
        self._live_pipeline: Optional[LivePipeline] = None

    # -- expose the store for advanced use ----------------------------------

    @property
    def store(self) -> BarStore:
        return self._store

    # -- 1. Query -----------------------------------------------------------

    def get_last_n_bars(self, n: int) -> list[Bar]:
        """
        Return the most recent *n* bars from the database.

        Bars are ordered chronologically (oldest first).
        Returns fewer than *n* if the store has fewer bars.
        """
        return self._store.get_last_n(n)

    # -- 2. Live streaming --------------------------------------------------

    def stream_live(
        self,
        on_bar: Callable[[Bar], None],
        *,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Stream live bars from Databento, persisting each bar to the
        store and forwarding it to *on_bar*.

        Blocks until *timeout* seconds elapse or ``.stop_live()`` is
        called from another thread.

        Parameters
        ----------
        on_bar : callable
            Invoked with each completed 5-min bar.
        api_key : str, optional
            Databento API key.  Falls back to ``DATABENTO_API_KEY`` env var.
        timeout : float, optional
            Auto-stop after this many seconds.  ``None`` = run forever.
        """
        def _persist_and_forward(bar: Bar) -> None:
            self._store.insert_bar(bar)
            on_bar(bar)

        self._live_pipeline = LivePipeline(
            on_bar=_persist_and_forward,
            api_key=api_key,
        )
        logger.info("BarAPI: starting live stream")
        self._live_pipeline.run(timeout=timeout)

    def stop_live(self) -> None:
        """Signal the live pipeline to stop."""
        if self._live_pipeline is not None:
            self._live_pipeline.stop()
        else:
            logger.warning("stop_live called but no live pipeline is running")

    # -- 3. Replay ----------------------------------------------------------

    def replay(
        self,
        start: str,
        end: str,
        on_bar: Callable[[Bar], None],
        speed_multiplier: float = 0.0,
    ) -> list[Bar]:
        """
        Replay stored bars in ``[start, end]`` through *on_bar*.

        Parameters
        ----------
        start, end : str
            ISO-8601 timestamps (inclusive).
        on_bar : callable
            Invoked with each bar in chronological order.
        speed_multiplier : float
            ``0`` = instant, ``1.0`` = real-time, ``>1`` = fast-forward.

        Returns
        -------
        list[Bar]  — the bars that were replayed.
        """
        return self._replay_engine.replay(
            start=start,
            end=end,
            on_bar=on_bar,
            speed_multiplier=speed_multiplier,
        )
