"""Tests for data.storage, data.replay, and data.bar_api."""

import datetime
import tempfile
from pathlib import Path

import pytest
import pytz

from data.data_pipeline import Bar
from data.storage import BarStore
from data.replay import ReplayEngine
from data.bar_api import BarAPI

ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(hour: int, minute: int, price: float = 5000.0, day: int = 2) -> Bar:
    """Create a Bar with a given ET time and price."""
    ts = ET.localize(datetime.datetime(2025, 1, day, hour, minute, 0))
    return Bar(
        timestamp=ts,
        open=price,
        high=price + 2.0,
        low=price - 2.0,
        close=price + 1.0,
        volume=100.0,
    )


def _make_bars(n: int, start_hour: int = 9, start_minute: int = 30, day: int = 2) -> list[Bar]:
    """Generate n sequential 5-minute bars."""
    bars = []
    for i in range(n):
        m = start_minute + i * 5
        h = start_hour + m // 60
        m = m % 60
        bars.append(_bar(h, m, price=5000.0 + i, day=day))
    return bars


# ===========================================================================
# BarStore tests
# ===========================================================================

class TestBarStore:

    def test_insert_and_retrieve(self):
        store = BarStore(":memory:")
        bar = _bar(9, 30)
        store.insert_bar(bar)
        result = store.get_bars()
        assert len(result) == 1
        assert result[0].open == bar.open
        assert result[0].close == bar.close
        assert result[0].timestamp == bar.timestamp

    def test_duplicate_insert_ignored(self):
        store = BarStore(":memory:")
        bar = _bar(9, 30)
        store.insert_bar(bar)
        store.insert_bar(bar)  # same timestamp
        assert store.count() == 1

    def test_bulk_insert(self):
        store = BarStore(":memory:")
        bars = _make_bars(10)
        inserted = store.insert_bars(bars)
        assert inserted == 10
        assert store.count() == 10

    def test_bulk_insert_dedup(self):
        store = BarStore(":memory:")
        bars = _make_bars(5)
        store.insert_bars(bars)
        inserted = store.insert_bars(bars)  # all duplicates
        assert inserted == 0
        assert store.count() == 5

    def test_get_bars_with_range(self):
        store = BarStore(":memory:")
        bars = _make_bars(10)
        store.insert_bars(bars)

        start_ts = bars[3].timestamp.isoformat()
        end_ts = bars[6].timestamp.isoformat()
        result = store.get_bars(start=start_ts, end=end_ts)

        assert len(result) == 4
        assert result[0].timestamp == bars[3].timestamp
        assert result[-1].timestamp == bars[6].timestamp

    def test_get_bars_ascending_order(self):
        store = BarStore(":memory:")
        bars = _make_bars(5)
        # Insert in reverse order
        for b in reversed(bars):
            store.insert_bar(b)
        result = store.get_bars()
        timestamps = [b.timestamp for b in result]
        assert timestamps == sorted(timestamps)

    def test_get_last_n(self):
        store = BarStore(":memory:")
        bars = _make_bars(10)
        store.insert_bars(bars)

        last_3 = store.get_last_n(3)
        assert len(last_3) == 3
        # Should be the 3 most recent, in ascending order
        assert last_3[0].timestamp == bars[7].timestamp
        assert last_3[-1].timestamp == bars[9].timestamp

    def test_get_last_n_exceeds_total(self):
        store = BarStore(":memory:")
        bars = _make_bars(3)
        store.insert_bars(bars)
        result = store.get_last_n(100)
        assert len(result) == 3

    def test_get_last_n_zero(self):
        store = BarStore(":memory:")
        store.insert_bars(_make_bars(5))
        assert store.get_last_n(0) == []

    def test_clear(self):
        store = BarStore(":memory:")
        store.insert_bars(_make_bars(5))
        store.clear()
        assert store.count() == 0

    def test_file_persistence(self, tmp_path):
        db_path = tmp_path / "test.db"
        bars = _make_bars(5)

        # Write
        store1 = BarStore(db_path)
        store1.insert_bars(bars)

        # Read from new instance
        store2 = BarStore(db_path)
        assert store2.count() == 5
        result = store2.get_bars()
        for orig, loaded in zip(bars, result):
            assert orig.open == loaded.open
            assert orig.close == loaded.close
            assert orig.high == loaded.high
            assert orig.low == loaded.low
            assert orig.volume == loaded.volume

    def test_bar_field_precision(self):
        store = BarStore(":memory:")
        bar = Bar(
            timestamp=ET.localize(datetime.datetime(2025, 1, 2, 10, 0, 0)),
            open=5123.25,
            high=5125.50,
            low=5120.75,
            close=5124.00,
            volume=1234.0,
        )
        store.insert_bar(bar)
        result = store.get_bars()[0]
        assert result.open == pytest.approx(5123.25)
        assert result.high == pytest.approx(5125.50)
        assert result.low == pytest.approx(5120.75)
        assert result.close == pytest.approx(5124.00)
        assert result.volume == pytest.approx(1234.0)


# ===========================================================================
# ReplayEngine tests
# ===========================================================================

class TestReplayEngine:

    def _setup_store(self, bars: list[Bar]) -> BarStore:
        store = BarStore(":memory:")
        store.insert_bars(bars)
        return store

    def test_replay_emits_all_bars(self):
        bars = _make_bars(10)
        store = self._setup_store(bars)
        engine = ReplayEngine(store)

        received: list[Bar] = []
        result = engine.replay(
            start=bars[0].timestamp.isoformat(),
            end=bars[-1].timestamp.isoformat(),
            on_bar=received.append,
            speed_multiplier=0,
        )
        assert len(received) == 10
        assert len(result) == 10

    def test_replay_preserves_order(self):
        bars = _make_bars(10)
        store = self._setup_store(bars)
        engine = ReplayEngine(store)

        received: list[Bar] = []
        engine.replay(
            start=bars[0].timestamp.isoformat(),
            end=bars[-1].timestamp.isoformat(),
            on_bar=received.append,
            speed_multiplier=0,
        )
        timestamps = [b.timestamp for b in received]
        assert timestamps == sorted(timestamps)

    def test_replay_matches_stored_data(self):
        """Bars from replay must be identical to bars in the store."""
        bars = _make_bars(5)
        store = self._setup_store(bars)
        engine = ReplayEngine(store)

        received: list[Bar] = []
        engine.replay(
            start=bars[0].timestamp.isoformat(),
            end=bars[-1].timestamp.isoformat(),
            on_bar=received.append,
            speed_multiplier=0,
        )
        stored = store.get_bars()
        for replayed, original in zip(received, stored):
            assert replayed.timestamp == original.timestamp
            assert replayed.open == original.open
            assert replayed.high == original.high
            assert replayed.low == original.low
            assert replayed.close == original.close
            assert replayed.volume == original.volume

    def test_replay_subset_range(self):
        bars = _make_bars(10)
        store = self._setup_store(bars)
        engine = ReplayEngine(store)

        received: list[Bar] = []
        engine.replay(
            start=bars[2].timestamp.isoformat(),
            end=bars[5].timestamp.isoformat(),
            on_bar=received.append,
            speed_multiplier=0,
        )
        assert len(received) == 4
        assert received[0].timestamp == bars[2].timestamp
        assert received[-1].timestamp == bars[5].timestamp

    def test_replay_empty_range(self):
        bars = _make_bars(5)
        store = self._setup_store(bars)
        engine = ReplayEngine(store)

        received: list[Bar] = []
        result = engine.replay(
            start="2099-01-01T00:00:00-05:00",
            end="2099-01-02T00:00:00-05:00",
            on_bar=received.append,
            speed_multiplier=0,
        )
        assert received == []
        assert result == []

    def test_replay_deterministic(self):
        """Two replays of the same range produce identical sequences."""
        bars = _make_bars(20)
        store = self._setup_store(bars)
        engine = ReplayEngine(store)

        run1: list[Bar] = []
        run2: list[Bar] = []

        start = bars[0].timestamp.isoformat()
        end = bars[-1].timestamp.isoformat()

        engine.replay(start=start, end=end, on_bar=run1.append, speed_multiplier=0)
        engine.replay(start=start, end=end, on_bar=run2.append, speed_multiplier=0)

        assert len(run1) == len(run2)
        for b1, b2 in zip(run1, run2):
            assert b1.timestamp == b2.timestamp
            assert b1.open == b2.open
            assert b1.high == b2.high
            assert b1.low == b2.low
            assert b1.close == b2.close
            assert b1.volume == b2.volume

    def test_replay_with_speed(self):
        """Non-zero speed should introduce pacing (sanity check — not exact)."""
        bars = _make_bars(3)
        store = self._setup_store(bars)
        engine = ReplayEngine(store)

        received: list[Bar] = []
        import time
        t0 = time.monotonic()
        engine.replay(
            start=bars[0].timestamp.isoformat(),
            end=bars[-1].timestamp.isoformat(),
            on_bar=received.append,
            speed_multiplier=10000,  # very fast but non-zero
        )
        elapsed = time.monotonic() - t0
        # At 10000x speed, 10 min of bars → ~0.06s.  Just check it ran.
        assert len(received) == 3
        assert elapsed >= 0  # sanity: no crash


# ===========================================================================
# BarAPI integration tests
# ===========================================================================

class TestBarAPI:

    def test_get_last_n_bars(self):
        api = BarAPI(":memory:")
        bars = _make_bars(10)
        api.store.insert_bars(bars)

        result = api.get_last_n_bars(3)
        assert len(result) == 3
        assert result[-1].timestamp == bars[-1].timestamp

    def test_replay_through_api(self):
        api = BarAPI(":memory:")
        bars = _make_bars(8)
        api.store.insert_bars(bars)

        received: list[Bar] = []
        result = api.replay(
            start=bars[0].timestamp.isoformat(),
            end=bars[-1].timestamp.isoformat(),
            on_bar=received.append,
        )
        assert len(received) == 8
        assert len(result) == 8

    def test_replay_matches_get_bars(self):
        """Replay output must exactly match direct store query."""
        api = BarAPI(":memory:")
        bars = _make_bars(5)
        api.store.insert_bars(bars)

        start = bars[0].timestamp.isoformat()
        end = bars[-1].timestamp.isoformat()

        queried = api.store.get_bars(start=start, end=end)
        replayed: list[Bar] = []
        api.replay(start=start, end=end, on_bar=replayed.append)

        assert len(queried) == len(replayed)
        for q, r in zip(queried, replayed):
            assert q.timestamp == r.timestamp
            assert q.open == r.open
            assert q.high == r.high
            assert q.low == r.low
            assert q.close == r.close
            assert q.volume == r.volume

    def test_full_roundtrip(self):
        """Insert → store → replay → compare: end-to-end determinism."""
        api = BarAPI(":memory:")
        original_bars = _make_bars(15)
        api.store.insert_bars(original_bars)

        replayed: list[Bar] = []
        api.replay(
            start=original_bars[0].timestamp.isoformat(),
            end=original_bars[-1].timestamp.isoformat(),
            on_bar=replayed.append,
        )

        assert len(replayed) == len(original_bars)
        for orig, rep in zip(original_bars, replayed):
            assert orig.timestamp == rep.timestamp
            assert orig.open == rep.open
            assert orig.high == rep.high
            assert orig.low == rep.low
            assert orig.close == rep.close
            assert orig.volume == rep.volume

    def test_replay_deterministic_across_api_instances(self, tmp_path):
        """Two separate BarAPI instances on the same DB produce identical replays."""
        db_path = tmp_path / "det.db"
        bars = _make_bars(10)

        api1 = BarAPI(db_path)
        api1.store.insert_bars(bars)

        start = bars[0].timestamp.isoformat()
        end = bars[-1].timestamp.isoformat()

        run1: list[Bar] = []
        api1.replay(start=start, end=end, on_bar=run1.append)

        # New instance, same DB
        api2 = BarAPI(db_path)
        run2: list[Bar] = []
        api2.replay(start=start, end=end, on_bar=run2.append)

        assert len(run1) == len(run2)
        for b1, b2 in zip(run1, run2):
            assert b1.timestamp == b2.timestamp
            assert b1.open == b2.open
            assert b1.close == b2.close

    def test_multi_day_replay(self):
        """Bars spanning multiple days replay correctly."""
        api = BarAPI(":memory:")
        day1 = _make_bars(5, day=2)
        day2 = _make_bars(5, day=3)
        all_bars = day1 + day2
        api.store.insert_bars(all_bars)

        received: list[Bar] = []
        api.replay(
            start=all_bars[0].timestamp.isoformat(),
            end=all_bars[-1].timestamp.isoformat(),
            on_bar=received.append,
        )
        assert len(received) == 10
        # Verify day boundary ordering
        assert received[4].timestamp.day == 2
        assert received[5].timestamp.day == 3

    def test_empty_store(self):
        api = BarAPI(":memory:")
        assert api.get_last_n_bars(10) == []
        result = api.replay(
            start="2024-01-01T00:00:00-05:00",
            end="2024-12-31T23:59:59-05:00",
            on_bar=lambda b: None,
        )
        assert result == []
