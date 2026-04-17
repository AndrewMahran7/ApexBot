"""
SQLite persistence layer for OHLCV bar data.
==============================================

Stores bars in a single ``bars`` table with schema:

    timestamp TEXT PRIMARY KEY   -- ISO-8601 with UTC offset
    open      REAL NOT NULL
    high      REAL NOT NULL
    low       REAL NOT NULL
    close     REAL NOT NULL
    volume    REAL NOT NULL

Thread-safe: each public method acquires/releases its own connection
via a short-lived context manager.  Suitable for concurrent reads from
a replay engine while a live pipeline is writing.

Usage:
    from data.storage import BarStore

    store = BarStore("data/bars.db")
    store.insert_bar(bar)
    bars = store.get_bars(start="2024-01-02T09:30:00", end="2024-01-02T16:00:00")
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from data.data_pipeline import Bar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS bars (
    timestamp TEXT    PRIMARY KEY,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL
) WITHOUT ROWID;
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars (timestamp);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class BarStore:
    """
    SQLite-backed storage for completed OHLCV bars.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.  Created if it does not exist.
        Use ``":memory:"`` for an ephemeral in-memory database.
    """

    def __init__(self, db_path: str | Path = "data/bars.db") -> None:
        self._db_path = str(db_path)
        # For :memory: databases we keep a single persistent connection,
        # because each new sqlite3.connect(":memory:") creates a separate DB.
        self._persistent_conn: sqlite3.Connection | None = None
        if self._db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
            self._persistent_conn.execute("PRAGMA journal_mode=WAL;")
            self._persistent_conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        logger.info("BarStore opened: %s", self._db_path)

    # -- connection helper --------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._persistent_conn is not None:
            # In-memory: reuse the same connection, no close
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)

    # -- write --------------------------------------------------------------

    def insert_bar(self, bar: Bar) -> None:
        """Insert a single bar.  Duplicates (same timestamp) are ignored."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO bars (timestamp, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (bar.timestamp.isoformat(), bar.open, bar.high, bar.low, bar.close, bar.volume),
            )

    def insert_bars(self, bars: list[Bar]) -> int:
        """
        Bulk-insert bars.  Duplicates are silently skipped.

        Returns the number of rows actually inserted.
        """
        with self._connect() as conn:
            before = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
            conn.executemany(
                "INSERT OR IGNORE INTO bars (timestamp, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume)
                    for b in bars
                ],
            )
            after = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
        inserted = after - before
        logger.info("Bulk insert: %d bars submitted, %d new", len(bars), inserted)
        return inserted

    # -- read ---------------------------------------------------------------

    def get_bars(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[Bar]:
        """
        Retrieve bars in ``[start, end]`` (inclusive).

        Parameters
        ----------
        start, end : str, optional
            ISO-8601 timestamp strings.  If omitted, no bound on that side.

        Returns
        -------
        list[Bar] ordered by timestamp ascending.
        """
        clauses: list[str] = []
        params: list[str] = []
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT timestamp, open, high, low, close, volume FROM bars{where} ORDER BY timestamp"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_bar(r) for r in rows]

    def get_last_n(self, n: int) -> list[Bar]:
        """Return the last *n* bars by timestamp (ascending order)."""
        if n <= 0:
            return []
        sql = (
            "SELECT timestamp, open, high, low, close, volume FROM bars "
            "ORDER BY timestamp DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (n,)).fetchall()
        return [self._row_to_bar(r) for r in reversed(rows)]

    def count(self) -> int:
        """Total number of stored bars."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]

    # -- maintenance --------------------------------------------------------

    def clear(self) -> None:
        """Delete all bars."""
        with self._connect() as conn:
            conn.execute("DELETE FROM bars")
        logger.info("BarStore cleared")

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_bar(row: tuple) -> Bar:
        ts_str, o, h, l, c, v = row
        ts = datetime.datetime.fromisoformat(ts_str)
        return Bar(timestamp=ts, open=o, high=h, low=l, close=c, volume=v)
