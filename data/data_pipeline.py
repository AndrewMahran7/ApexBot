"""
Real-time MES futures data pipeline.

Streams 1-minute OHLCV bars from Databento Live, aggregates into
5-minute bars, and emits completed bars via callbacks.

Supports:
  - Live streaming from Databento
  - Mock replay from historical CSV for testing
  - Automatic reconnection with duplicate filtering
  - CME session boundary handling
  - Partial bar flushing at session close

Usage:
    from data.data_pipeline import LivePipeline, ReplayPipeline

    # Live
    pipeline = LivePipeline(on_bar=my_callback)
    pipeline.run()

    # Replay (testing)
    pipeline = ReplayPipeline("data/mes_5m.csv", on_bar=my_callback)
    pipeline.run()
"""

from __future__ import annotations

import csv
import datetime
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pytz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET = "GLBX.MDP3"
SYMBOL = "MES.c.0"
STYPE_IN = "continuous"
SCHEMA = "ohlcv-1m"                       # 1-minute bars from Databento

BAR_INTERVAL_MINUTES = 5
ET = pytz.timezone("America/New_York")

# CME Globex MES session boundaries (Eastern Time).
# Sunday 6:00 PM  – Friday 5:00 PM, daily halt 4:15 PM – 4:30 PM.
SESSION_OPEN_ET = datetime.time(18, 0)     # 6:00 PM ET (Sunday open)
SESSION_CLOSE_ET = datetime.time(17, 0)    # 5:00 PM ET (next day close)
HALT_START_ET = datetime.time(16, 15)      # daily halt start
HALT_END_ET = datetime.time(16, 30)        # daily halt end

# Maximum bars kept in memory (5-min bars ≈ 288/day, keep ~5 days)
DEFAULT_BUFFER_SIZE = 1500

# Reconnect back-off
_RECONNECT_BASE_DELAY = 1.0               # seconds
_RECONNECT_MAX_DELAY = 60.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bar:
    """A completed OHLCV bar."""
    timestamp: datetime.datetime    # bar open time, tz-aware (ET)
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class _PartialBar:
    """Accumulates 1-minute bars into a multi-minute bar."""
    bucket: datetime.datetime       # bar open time (floored)
    open: float = 0.0
    high: float = float("-inf")
    low: float = float("inf")
    close: float = 0.0
    volume: float = 0.0
    tick_count: int = 0             # number of 1-min bars aggregated

    def update(self, o: float, h: float, l: float, c: float, v: float) -> None:
        if self.tick_count == 0:
            self.open = o
        self.high = max(self.high, h)
        self.low = min(self.low, l)
        self.close = c
        self.volume += v
        self.tick_count += 1

    def to_bar(self) -> Bar:
        return Bar(
            timestamp=self.bucket,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


# ---------------------------------------------------------------------------
# Bar aggregator (pure logic — no I/O)
# ---------------------------------------------------------------------------

class BarAggregator:
    """
    Aggregates 1-minute OHLCV ticks into 5-minute bars.

    Stateless with respect to I/O.  All side effects go through
    the ``on_bar`` callback passed at construction.

    Handles:
      - Time-bucket alignment (floor to BAR_INTERVAL_MINUTES)
      - Session boundaries (flush partial at session close / halt)
      - Duplicate filtering (monotonic ts_event guard)
      - Missing-tick detection (logs gaps)
    """

    def __init__(
        self,
        on_bar: Callable[[Bar], None],
        interval_minutes: int = BAR_INTERVAL_MINUTES,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        timezone: datetime.tzinfo = ET,
    ):
        self._on_bar = on_bar
        self._interval = interval_minutes
        self._tz = timezone
        self._partial: Optional[_PartialBar] = None
        self._last_ts: Optional[datetime.datetime] = None
        self._bars: deque[Bar] = deque(maxlen=buffer_size)
        self._expected_next_minute: Optional[datetime.datetime] = None

    # -- public interface ---------------------------------------------------

    @property
    def bars(self) -> list[Bar]:
        """Return buffered completed bars (oldest first)."""
        return list(self._bars)

    def process_tick(
        self,
        ts: datetime.datetime,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
    ) -> None:
        """
        Ingest a single 1-minute OHLCV bar.

        Parameters
        ----------
        ts : datetime.datetime
            Bar timestamp (tz-aware, Eastern).
        o, h, l, c, v : float
            OHLCV values.
        """
        if ts.tzinfo is None:
            raise ValueError(f"Timestamp must be tz-aware, got naive: {ts}")

        ts_et = ts.astimezone(self._tz)

        # --- Duplicate filter ---
        if self._last_ts is not None and ts_et <= self._last_ts:
            logger.debug("Skipping duplicate/out-of-order tick: %s (<= %s)", ts_et, self._last_ts)
            return

        # --- Missing-tick detection ---
        if self._expected_next_minute is not None:
            gap = ts_et - self._expected_next_minute
            gap_minutes = gap.total_seconds() / 60
            if gap_minutes > 1.5 and not self._is_across_halt(self._expected_next_minute, ts_et):
                logger.warning(
                    "Gap detected: expected ~%s, got %s (%.1f min gap)",
                    self._expected_next_minute.strftime("%H:%M"),
                    ts_et.strftime("%H:%M"),
                    gap_minutes,
                )

        self._last_ts = ts_et
        self._expected_next_minute = ts_et + datetime.timedelta(minutes=1)

        # --- Session boundary: flush partial before halt / close ---
        if self._partial is not None and self._crosses_boundary(self._partial.bucket, ts_et):
            self._emit_partial()

        # --- Bucket assignment ---
        bucket = self._floor_ts(ts_et)

        if self._partial is None or bucket != self._partial.bucket:
            # New bucket — emit old partial if any
            if self._partial is not None and self._partial.tick_count > 0:
                self._emit_partial()
            self._partial = _PartialBar(bucket=bucket)

        self._partial.update(o, h, l, c, v)

        # --- Emit when bucket is full ---
        if self._partial.tick_count >= self._interval:
            self._emit_partial()

    def flush(self) -> None:
        """Force-emit the current partial bar (e.g. on shutdown)."""
        if self._partial is not None and self._partial.tick_count > 0:
            logger.info("Flushing partial bar: %s (%d ticks)", self._partial.bucket, self._partial.tick_count)
            self._emit_partial()

    def reset(self) -> None:
        """Clear all state."""
        self._partial = None
        self._last_ts = None
        self._bars.clear()
        self._expected_next_minute = None

    # -- private ------------------------------------------------------------

    def _floor_ts(self, ts: datetime.datetime) -> datetime.datetime:
        """Floor timestamp to the nearest bar interval boundary."""
        minutes = ts.minute - (ts.minute % self._interval)
        return ts.replace(minute=minutes, second=0, microsecond=0)

    def _emit_partial(self) -> None:
        bar = self._partial.to_bar()
        self._bars.append(bar)
        self._partial = None
        try:
            self._on_bar(bar)
        except Exception:
            logger.exception("Error in on_bar callback for bar %s", bar.timestamp)

    def _crosses_boundary(self, bucket: datetime.datetime, ts: datetime.datetime) -> bool:
        """Check if ts crosses a session halt or close relative to bucket."""
        b_time = bucket.time()
        t_time = ts.time()
        # Crossed daily halt
        if b_time < HALT_START_ET and t_time >= HALT_END_ET:
            return True
        # Crossed session close (5:00 PM ET)
        if b_time < SESSION_CLOSE_ET and t_time >= SESSION_CLOSE_ET:
            return True
        return False

    @staticmethod
    def _is_across_halt(expected: datetime.datetime, actual: datetime.datetime) -> bool:
        """Return True if the gap spans the daily halt window."""
        e_time = expected.time()
        a_time = actual.time()
        return e_time <= HALT_START_ET and a_time >= HALT_END_ET


# ---------------------------------------------------------------------------
# Live pipeline (Databento)
# ---------------------------------------------------------------------------

class LivePipeline:
    """
    Connect to Databento Live, subscribe to MES 1-min bars,
    aggregate into 5-min bars, and emit via callback.

    Environment:
        DATABENTO_API_KEY must be set.
    """

    def __init__(
        self,
        on_bar: Callable[[Bar], None],
        symbol: str = SYMBOL,
        interval_minutes: int = BAR_INTERVAL_MINUTES,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        api_key: Optional[str] = None,
    ):
        self._symbol = symbol
        self._api_key = api_key or os.environ.get("DATABENTO_API_KEY")
        if not self._api_key:
            raise EnvironmentError(
                "DATABENTO_API_KEY not set. Export it or pass api_key=."
            )

        self._aggregator = BarAggregator(
            on_bar=on_bar,
            interval_minutes=interval_minutes,
            buffer_size=buffer_size,
        )
        self._running = False
        self._reconnect_count = 0
        self._last_ts_event: Optional[int] = None  # nanosecond UNIX for replay dedup

    # -- public -------------------------------------------------------------

    @property
    def bars(self) -> list[Bar]:
        return self._aggregator.bars

    def run(self, timeout: Optional[float] = None) -> None:
        """
        Start streaming. Blocks until timeout or manual stop.

        Parameters
        ----------
        timeout : float, optional
            Seconds to run before stopping.  None = run forever.
        """
        try:
            import databento as db
        except ImportError:
            raise ImportError(
                "The 'databento' package is required. Install: pip install databento"
            )

        self._running = True
        delay = _RECONNECT_BASE_DELAY

        while self._running:
            try:
                client = db.Live(
                    key=self._api_key,
                    reconnect_policy="reconnect",
                )

                client.subscribe(
                    dataset=DATASET,
                    schema=SCHEMA,
                    symbols=self._symbol,
                    stype_in=STYPE_IN,
                )

                client.add_callback(
                    record_callback=self._on_record,
                    exception_callback=self._on_error,
                )

                client.add_reconnect_callback(
                    reconnect_callback=self._on_reconnect,
                )

                logger.info(
                    "LivePipeline started: symbol=%s, schema=%s, dataset=%s",
                    self._symbol, SCHEMA, DATASET,
                )

                client.start()
                client.block_for_close(timeout=timeout)

                # If timeout was reached, we're done
                if timeout is not None:
                    self._running = False

                delay = _RECONNECT_BASE_DELAY  # reset on clean close

            except KeyboardInterrupt:
                logger.info("LivePipeline interrupted by user")
                self._running = False
            except Exception:
                if not self._running:
                    break
                self._reconnect_count += 1
                logger.exception(
                    "LivePipeline connection lost (reconnect #%d), retrying in %.1fs",
                    self._reconnect_count, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)

        self._aggregator.flush()
        logger.info(
            "LivePipeline stopped. Total bars: %d, reconnects: %d",
            len(self._aggregator.bars), self._reconnect_count,
        )

    def stop(self) -> None:
        """Signal the pipeline to stop after current iteration."""
        self._running = False

    # -- callbacks ----------------------------------------------------------

    def _on_record(self, record) -> None:
        """Databento record callback — dispatches OHLCV bars."""
        import databento as db

        # Skip non-OHLCV records (heartbeats, errors, system messages)
        if isinstance(record, db.ErrorMsg):
            logger.error("Databento error: %s (code=%s)", record.err, record.code)
            return
        if isinstance(record, db.SystemMsg):
            if record.is_heartbeat:
                logger.debug("Heartbeat received")
            else:
                logger.info("System message: %s (code=%s)", record.msg, record.code)
            return

        if not isinstance(record, db.OhlcvMsg):
            return

        # Duplicate guard: skip if ts_event <= last seen
        if self._last_ts_event is not None and record.ts_event <= self._last_ts_event:
            logger.debug("Skipping duplicate record ts_event=%s", record.ts_event)
            return
        self._last_ts_event = record.ts_event

        # Convert nanosecond UNIX → tz-aware datetime
        ts = datetime.datetime.fromtimestamp(
            record.ts_event / 1_000_000_000,
            tz=datetime.timezone.utc,
        ).astimezone(ET)

        self._aggregator.process_tick(
            ts=ts,
            o=record.open / 1e9,      # Databento fixed-point → float
            h=record.high / 1e9,
            l=record.low / 1e9,
            c=record.close / 1e9,
            v=float(record.volume),
        )

    def _on_error(self, exc: Exception) -> None:
        logger.error("Callback exception: %s", exc, exc_info=True)

    def _on_reconnect(self, last_ts, new_start_ts) -> None:
        self._reconnect_count += 1
        logger.warning(
            "Reconnected: gap from %s to %s (reconnect #%d)",
            last_ts, new_start_ts, self._reconnect_count,
        )


# ---------------------------------------------------------------------------
# Replay pipeline (mock — for testing / offline development)
# ---------------------------------------------------------------------------

class ReplayPipeline:
    """
    Replay historical 5-minute (or 1-minute) CSV data through the
    bar aggregator.  Useful for testing callbacks without a live feed.

    The CSV must have columns: timestamp, open, high, low, close, volume.
    Timestamps should be parseable by datetime and ideally tz-aware.
    """

    def __init__(
        self,
        csv_path: str,
        on_bar: Callable[[Bar], None],
        interval_minutes: int = BAR_INTERVAL_MINUTES,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        speed: float = 0.0,
    ):
        self._path = Path(csv_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Replay file not found: {csv_path}")

        self._speed = speed  # seconds between ticks (0 = instant)
        self._aggregator = BarAggregator(
            on_bar=on_bar,
            interval_minutes=interval_minutes,
            buffer_size=buffer_size,
        )

    @property
    def bars(self) -> list[Bar]:
        return self._aggregator.bars

    def run(self) -> None:
        """Replay the entire CSV through the aggregator."""
        logger.info("ReplayPipeline: loading %s", self._path)
        count = 0

        with open(self._path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV missing required columns. Need {required}, "
                    f"got {reader.fieldnames}"
                )

            for row in reader:
                ts = _parse_csv_timestamp(row["timestamp"])
                self._aggregator.process_tick(
                    ts=ts,
                    o=float(row["open"]),
                    h=float(row["high"]),
                    l=float(row["low"]),
                    c=float(row["close"]),
                    v=float(row["volume"]),
                )
                count += 1
                if self._speed > 0:
                    time.sleep(self._speed)

        self._aggregator.flush()
        logger.info("ReplayPipeline: processed %d ticks → %d bars", count, len(self._aggregator.bars))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv_timestamp(raw: str) -> datetime.datetime:
    """
    Parse a CSV timestamp string into a tz-aware datetime (ET).

    Handles:
      - ISO 8601 with offset  (2024-01-02T09:30:00-05:00)
      - ISO 8601 with Z       (2024-01-02T14:30:00Z)
      - Naive datetime        (2024-01-02 09:30:00) → assumed ET
    """
    raw = raw.strip()
    # Try ISO parse (Python 3.11+)
    try:
        dt = datetime.datetime.fromisoformat(raw)
    except ValueError:
        # Fall back to strptime for common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse timestamp: {raw!r}")

    if dt.tzinfo is None:
        dt = ET.localize(dt)
    return dt


def log_bar(bar: Bar) -> None:
    """Default callback that logs a completed bar to the console."""
    logger.info(
        "BAR %s  O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
        bar.timestamp.strftime("%Y-%m-%d %H:%M"),
        bar.open, bar.high, bar.low, bar.close, bar.volume,
    )
