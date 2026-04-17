"""
Deterministic bar replay engine.
==================================

Reads bars from a :class:`BarStore` and feeds them one-by-one to
a callback, optionally simulating real-time pacing.

Key properties:
  * **Deterministic** — same store + same parameters → identical
    sequence every run (no threading, no jitter).
  * **Configurable speed** — ``speed_multiplier=0`` for instant replay,
    ``1.0`` for wall-clock pacing, ``>1`` for fast-forward.
  * **Produces the same Bar objects** that a live pipeline would emit,
    so downstream consumers (strategy, engine) cannot distinguish
    replay from live.

Usage:
    from data.storage import BarStore
    from data.replay import ReplayEngine

    store = BarStore("data/bars.db")
    engine = ReplayEngine(store)
    engine.replay("2024-01-02T09:30:00", "2024-01-02T16:00:00",
                  on_bar=my_callback, speed_multiplier=0)
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Callable, Optional

from data.data_pipeline import Bar
from data.storage import BarStore

logger = logging.getLogger(__name__)


class ReplayEngine:
    """
    Replays stored bars through a callback in chronological order.

    Parameters
    ----------
    store : BarStore
        The persistence backend to read bars from.
    """

    def __init__(self, store: BarStore) -> None:
        self._store = store

    def replay(
        self,
        start: str,
        end: str,
        on_bar: Callable[[Bar], None],
        speed_multiplier: float = 0.0,
    ) -> list[Bar]:
        """
        Replay bars in the ``[start, end]`` range.

        Parameters
        ----------
        start, end : str
            ISO-8601 timestamps defining the replay window (inclusive).
        on_bar : callable
            Called with each :class:`Bar` in sequence.
        speed_multiplier : float
            Controls pacing between bars:
            - ``0``   → instant (no delay) — best for testing / backtests
            - ``1.0`` → real-time (sleeps the actual inter-bar duration)
            - ``>1``  → fast-forward (e.g. 10 = 10× speed)
            Negative values are treated as 0.

        Returns
        -------
        list[Bar]  — all bars that were replayed, in order.
        """
        bars = self._store.get_bars(start=start, end=end)
        if not bars:
            logger.warning("ReplayEngine: no bars found in [%s, %s]", start, end)
            return []

        speed = max(speed_multiplier, 0.0)
        logger.info(
            "ReplayEngine: replaying %d bars [%s → %s], speed=%.1f×",
            len(bars),
            bars[0].timestamp.isoformat(),
            bars[-1].timestamp.isoformat(),
            speed,
        )

        prev_ts: Optional[datetime.datetime] = None
        for bar in bars:
            # Pacing
            if speed > 0.0 and prev_ts is not None:
                delta = (bar.timestamp - prev_ts).total_seconds()
                if delta > 0:
                    time.sleep(delta / speed)
            prev_ts = bar.timestamp

            on_bar(bar)

        logger.info("ReplayEngine: replay complete (%d bars emitted)", len(bars))
        return bars
