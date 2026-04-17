"""Tests for data.data_pipeline — bar aggregation and replay."""

import datetime
import csv
import os
import tempfile

import pytest
import pytz

from data.data_pipeline import (
    Bar,
    BarAggregator,
    ReplayPipeline,
    _parse_csv_timestamp,
    BAR_INTERVAL_MINUTES,
    HALT_START_ET,
    HALT_END_ET,
)

ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(hour: int, minute: int, day: int = 2, month: int = 1, year: int = 2025) -> datetime.datetime:
    """Create a tz-aware ET datetime."""
    return ET.localize(datetime.datetime(year, month, day, hour, minute, 0))


def _write_csv(path: str, rows: list[dict]) -> None:
    """Write rows to a CSV with standard OHLCV columns."""
    fieldnames = ["timestamp", "open", "high", "low", "close", "volume"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_tick_rows(
    start_hour: int,
    start_minute: int,
    count: int,
    base_price: float = 5000.0,
) -> list[dict]:
    """Generate sequential 1-minute tick rows."""
    rows = []
    for i in range(count):
        ts = _ts(start_hour, start_minute + i)
        price = base_price + i
        rows.append({
            "timestamp": ts.isoformat(),
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price + 0.5,
            "volume": 100 + i,
        })
    return rows


# ===========================================================================
# BarAggregator unit tests
# ===========================================================================

class TestBarAggregator:

    def test_emits_bar_after_5_ticks(self):
        """5 one-minute ticks → 1 completed 5-min bar."""
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append)

        for m in range(5):
            agg.process_tick(_ts(9, 30 + m), 100 + m, 110 + m, 90 + m, 105 + m, 1000)

        assert len(emitted) == 1
        bar = emitted[0]
        assert bar.timestamp == _ts(9, 30)
        assert bar.open == 100          # first tick's open
        assert bar.close == 109         # last tick's close
        assert bar.high == 114          # max high
        assert bar.low == 90            # min low
        assert bar.volume == 5000       # sum of volumes

    def test_no_emit_before_5_ticks(self):
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append)

        for m in range(4):
            agg.process_tick(_ts(9, 30 + m), 100, 110, 90, 105, 100)

        assert len(emitted) == 0

    def test_flush_emits_partial(self):
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append)

        for m in range(3):
            agg.process_tick(_ts(9, 30 + m), 100, 110, 90, 105, 100)

        agg.flush()
        assert len(emitted) == 1
        assert emitted[0].volume == 300

    def test_two_consecutive_bars(self):
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append)

        # Bar 1: 09:30–09:34
        for m in range(5):
            agg.process_tick(_ts(9, 30 + m), 100, 110, 90, 105, 100)
        # Bar 2: 09:35–09:39
        for m in range(5):
            agg.process_tick(_ts(9, 35 + m), 200, 210, 190, 205, 200)

        assert len(emitted) == 2
        assert emitted[0].timestamp == _ts(9, 30)
        assert emitted[1].timestamp == _ts(9, 35)

    def test_bucket_alignment(self):
        """A tick at 09:37 should land in the 09:35 bucket."""
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append)

        for m in range(5):
            agg.process_tick(_ts(9, 37 + m), 100, 110, 90, 105, 100)

        # 09:37 → bucket 09:35, 09:42 → bucket 09:40
        # So ticks 37,38,39 go to 09:35 bucket (3 ticks), then
        # ticks 40,41 go to 09:40 bucket (2 ticks)
        # The 09:35 bucket emits when 09:40 bucket starts (bucket change)
        assert len(emitted) == 1
        assert emitted[0].timestamp == _ts(9, 35)
        assert emitted[0].volume == 300  # 3 ticks

    def test_duplicate_filtered(self):
        """Duplicate timestamps are silently dropped."""
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append)

        agg.process_tick(_ts(9, 30), 100, 110, 90, 105, 100)
        agg.process_tick(_ts(9, 30), 999, 999, 999, 999, 999)  # duplicate
        agg.process_tick(_ts(9, 31), 101, 111, 91, 106, 101)

        agg.flush()
        assert len(emitted) == 1
        # The duplicate should not have polluted the bar
        assert emitted[0].open == 100
        assert emitted[0].volume == 201  # 100 + 101, not 100 + 999 + 101

    def test_out_of_order_filtered(self):
        """Out-of-order (earlier) timestamps are dropped."""
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append)

        agg.process_tick(_ts(9, 32), 100, 110, 90, 105, 100)
        agg.process_tick(_ts(9, 31), 999, 999, 999, 999, 999)  # earlier → skip

        agg.flush()
        assert emitted[0].volume == 100

    def test_bars_stored_in_buffer(self):
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append, buffer_size=10)

        for m in range(10):
            agg.process_tick(_ts(9, 30 + m), 100, 110, 90, 105, 100)

        assert len(agg.bars) == 2
        assert agg.bars[0].timestamp == _ts(9, 30)
        assert agg.bars[1].timestamp == _ts(9, 35)

    def test_reset_clears_state(self):
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append)

        agg.process_tick(_ts(9, 30), 100, 110, 90, 105, 100)
        agg.reset()

        assert len(agg.bars) == 0
        # After reset, can process again without issues
        agg.process_tick(_ts(9, 30), 200, 210, 190, 205, 200)
        agg.flush()
        assert len(emitted) == 1
        assert emitted[0].open == 200

    def test_naive_timestamp_rejected(self):
        agg = BarAggregator(on_bar=lambda b: None)
        with pytest.raises(ValueError, match="tz-aware"):
            agg.process_tick(datetime.datetime(2025, 1, 2, 9, 30), 100, 110, 90, 105, 100)

    def test_callback_exception_does_not_crash(self):
        """A failing callback is logged but doesn't stop processing."""
        def bad_callback(bar):
            raise RuntimeError("boom")

        agg = BarAggregator(on_bar=bad_callback)
        # Should not raise
        for m in range(5):
            agg.process_tick(_ts(9, 30 + m), 100, 110, 90, 105, 100)

    def test_session_halt_flushes_partial(self):
        """Partial bar is emitted when crossing the daily halt boundary."""
        emitted: list[Bar] = []
        agg = BarAggregator(on_bar=emitted.append)

        # Ticks before halt (16:13, 16:14)
        agg.process_tick(_ts(16, 13), 100, 110, 90, 105, 100)
        agg.process_tick(_ts(16, 14), 101, 111, 91, 106, 100)

        # Tick after halt resumes (16:30)
        agg.process_tick(_ts(16, 30), 102, 112, 92, 107, 100)

        # The pre-halt partial should have been flushed
        assert len(emitted) >= 1
        assert emitted[0].timestamp == _ts(16, 10)
        assert emitted[0].volume == 200


# ===========================================================================
# Session boundary tests
# ===========================================================================

class TestSessionBoundaries:

    def test_halt_detection(self):
        agg = BarAggregator(on_bar=lambda b: None)
        # 16:10 bucket → 16:30 tick crosses halt
        assert agg._crosses_boundary(_ts(16, 10), _ts(16, 30)) is True
        # 09:30 bucket → 09:35 tick does NOT cross halt
        assert agg._crosses_boundary(_ts(9, 30), _ts(9, 35)) is False

    def test_halt_gap_not_flagged_as_missing(self):
        """Gap across the halt window should not trigger missing-tick warning."""
        assert BarAggregator._is_across_halt(_ts(16, 15), _ts(16, 30)) is True
        assert BarAggregator._is_across_halt(_ts(9, 30), _ts(9, 35)) is False


# ===========================================================================
# Bar data structure
# ===========================================================================

class TestBar:

    def test_as_dict(self):
        bar = Bar(timestamp=_ts(9, 30), open=100, high=110, low=90, close=105, volume=1000)
        d = bar.as_dict()
        assert d["open"] == 100
        assert d["volume"] == 1000
        assert "timestamp" in d

    def test_frozen(self):
        bar = Bar(timestamp=_ts(9, 30), open=100, high=110, low=90, close=105, volume=1000)
        with pytest.raises(AttributeError):
            bar.open = 999


# ===========================================================================
# Timestamp parsing
# ===========================================================================

class TestTimestampParsing:

    def test_iso_with_offset(self):
        ts = _parse_csv_timestamp("2025-01-02T09:30:00-05:00")
        assert ts.tzinfo is not None
        assert ts.hour == 9
        assert ts.minute == 30

    def test_naive_assumed_et(self):
        ts = _parse_csv_timestamp("2025-01-02 09:30:00")
        assert ts.tzinfo is not None
        assert str(ts.tzinfo) in ("US/Eastern", "America/New_York", "EST", "EDT")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_csv_timestamp("not-a-date")


# ===========================================================================
# ReplayPipeline integration test
# ===========================================================================

class TestReplayPipeline:

    def test_replay_from_csv(self, tmp_path):
        """Replay 10 one-minute ticks → 2 five-minute bars."""
        csv_path = str(tmp_path / "test_data.csv")
        rows = _make_tick_rows(9, 30, 10)
        _write_csv(csv_path, rows)

        emitted: list[Bar] = []
        pipeline = ReplayPipeline(
            csv_path=csv_path,
            on_bar=emitted.append,
            interval_minutes=5,
        )
        pipeline.run()

        assert len(emitted) == 2
        assert emitted[0].timestamp == _ts(9, 30)
        assert emitted[1].timestamp == _ts(9, 35)

    def test_replay_stores_bars(self, tmp_path):
        csv_path = str(tmp_path / "test_data.csv")
        rows = _make_tick_rows(9, 30, 5)
        _write_csv(csv_path, rows)

        pipeline = ReplayPipeline(csv_path=csv_path, on_bar=lambda b: None)
        pipeline.run()

        assert len(pipeline.bars) == 1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            ReplayPipeline(csv_path="/nonexistent/data.csv", on_bar=lambda b: None)

    def test_missing_columns_raises(self, tmp_path):
        csv_path = str(tmp_path / "bad.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "price"])
            writer.writerow(["2025-01-02 09:30:00", "5000"])

        pipeline = ReplayPipeline(csv_path=csv_path, on_bar=lambda b: None)
        with pytest.raises(ValueError, match="missing required columns"):
            pipeline.run()

    def test_replay_partial_flush(self, tmp_path):
        """3 ticks (not a full bar) are flushed at end of replay."""
        csv_path = str(tmp_path / "partial.csv")
        rows = _make_tick_rows(9, 30, 3)
        _write_csv(csv_path, rows)

        emitted: list[Bar] = []
        pipeline = ReplayPipeline(csv_path=csv_path, on_bar=emitted.append)
        pipeline.run()

        # Should get 1 partial bar (flushed)
        assert len(emitted) == 1
        assert emitted[0].volume == sum(100 + i for i in range(3))


# ===========================================================================
# LivePipeline construction tests (no network)
# ===========================================================================

class TestLivePipelineInit:

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="DATABENTO_API_KEY"):
            from data.data_pipeline import LivePipeline
            LivePipeline(on_bar=lambda b: None)

    def test_with_explicit_key(self, monkeypatch):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        from data.data_pipeline import LivePipeline
        pipeline = LivePipeline(on_bar=lambda b: None, api_key="test_key_1234567890abcdef")
        assert pipeline._api_key == "test_key_1234567890abcdef"
