"""Tests for execution.reconciliation module."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from execution.fail_safes import FailSafeConfig, FailSafeState, RunMode
from execution.reconciliation import (
    MismatchType,
    ReconciliationLoop,
    ReconciliationResult,
    reconcile_positions,
)
from execution.validators import PositionMonitor, UIState


# ── Helper factories ─────────────────────────────────────────────────────────


def _make_ui(
    has_position: bool = False,
    side: str | None = None,
    size: int = 0,
    symbol: str = "MNQ",
    window_found: bool = True,
    read_errors: list[str] | None = None,
) -> UIState:
    state = UIState()
    state.window_found = window_found
    state.active_symbol = symbol
    state.has_open_position = has_position
    state.open_position_side = side
    state.open_position_size = size
    if read_errors:
        state.read_errors = read_errors
    return state


def _make_fail_safe(**overrides) -> FailSafeState:
    cfg = FailSafeConfig(run_mode=RunMode.LIVE, **overrides)
    return FailSafeState(cfg)


# ═════════════════════════════════════════════════════════════════════════════
# reconcile_positions — pure function tests
# ═════════════════════════════════════════════════════════════════════════════


class TestReconcilePositions:
    """Tests for the stateless reconcile_positions() function."""

    def test_both_flat_match(self):
        result = reconcile_positions({}, _make_ui())
        assert result.matched
        assert result.mismatches == []

    def test_expected_long_observed_long_match(self):
        expected = {"MNQ": {"side": "long", "size": 1}}
        ui = _make_ui(has_position=True, side="long", size=1, symbol="MNQ")
        result = reconcile_positions(expected, ui)
        assert result.matched
        assert result.mismatches == []

    def test_expected_short_observed_short_match(self):
        expected = {"MES": {"side": "short", "size": 2}}
        ui = _make_ui(has_position=True, side="short", size=2, symbol="MES")
        result = reconcile_positions(expected, ui)
        assert result.matched

    def test_missing_position(self):
        expected = {"MNQ": {"side": "long", "size": 1}}
        ui = _make_ui()  # flat
        result = reconcile_positions(expected, ui)
        assert not result.matched
        assert any(MismatchType.MISSING_POSITION in m for m in result.mismatches)

    def test_unexpected_position(self):
        expected = {}  # expect flat
        ui = _make_ui(has_position=True, side="long", size=1, symbol="MNQ")
        result = reconcile_positions(expected, ui)
        assert not result.matched
        assert any(MismatchType.UNEXPECTED_POSITION in m for m in result.mismatches)

    def test_wrong_size(self):
        expected = {"MNQ": {"side": "long", "size": 2}}
        ui = _make_ui(has_position=True, side="long", size=1, symbol="MNQ")
        result = reconcile_positions(expected, ui)
        assert not result.matched
        assert any(MismatchType.WRONG_SIZE in m for m in result.mismatches)

    def test_wrong_side(self):
        expected = {"MNQ": {"side": "long", "size": 1}}
        ui = _make_ui(has_position=True, side="short", size=1, symbol="MNQ")
        result = reconcile_positions(expected, ui)
        assert not result.matched
        assert any(MismatchType.WRONG_SIDE in m for m in result.mismatches)

    def test_wrong_side_and_size(self):
        expected = {"MNQ": {"side": "long", "size": 2}}
        ui = _make_ui(has_position=True, side="short", size=1, symbol="MNQ")
        result = reconcile_positions(expected, ui)
        assert not result.matched
        assert len(result.mismatches) == 2

    def test_expected_and_observed_different_symbols(self):
        expected = {"MES": {"side": "long", "size": 1}}
        ui = _make_ui(has_position=True, side="long", size=1, symbol="MNQ")
        result = reconcile_positions(expected, ui)
        assert not result.matched
        # Missing MES and unexpected MNQ
        assert any(MismatchType.MISSING_POSITION in m for m in result.mismatches)
        assert any(MismatchType.UNEXPECTED_POSITION in m for m in result.mismatches)

    def test_result_summary_match(self):
        result = reconcile_positions({}, _make_ui())
        assert "no mismatches" in result.summary()

    def test_result_summary_mismatch(self):
        expected = {"MNQ": {"side": "long", "size": 1}}
        result = reconcile_positions(expected, _make_ui())
        assert "MISMATCH" in result.summary()

    def test_result_has_timestamp(self):
        result = reconcile_positions({}, _make_ui())
        assert result.timestamp  # non-empty

    def test_unknown_symbol_when_none(self):
        """When UI shows a position but active_symbol is None."""
        expected = {}
        ui = _make_ui(has_position=True, side="long", size=1)
        ui.active_symbol = None
        result = reconcile_positions(expected, ui)
        assert not result.matched
        assert "UNKNOWN" in result.mismatches[0]


# ═════════════════════════════════════════════════════════════════════════════
# ReconciliationLoop — integration tests
# ═════════════════════════════════════════════════════════════════════════════


class TestReconciliationLoop:
    """Tests for the background ReconciliationLoop class."""

    def _make_loop(
        self,
        ui_state: UIState | None = None,
        expected_positions: dict | None = None,
        interval: float = 0.05,
        max_failures: int = 3,
    ) -> tuple[ReconciliationLoop, MagicMock, FailSafeState, PositionMonitor, MagicMock]:
        driver = MagicMock()
        driver.read_ui_state.return_value = ui_state or _make_ui()

        fail_safe = _make_fail_safe()
        position_monitor = PositionMonitor(expected_max_positions=1)

        if expected_positions:
            for sym, info in expected_positions.items():
                position_monitor.update(sym, info["side"], info["size"])

        on_alert = MagicMock()

        loop = ReconciliationLoop(
            driver=driver,
            fail_safe=fail_safe,
            position_monitor=position_monitor,
            on_alert=on_alert,
            interval_seconds=interval,
            max_consecutive_failures=max_failures,
        )
        return loop, driver, fail_safe, position_monitor, on_alert

    def test_start_and_stop(self):
        loop, *_ = self._make_loop()
        loop.start()
        assert loop.running
        loop.stop()
        assert not loop.running

    def test_double_start_is_safe(self):
        loop, *_ = self._make_loop()
        loop.start()
        loop.start()  # should not raise
        assert loop.running
        loop.stop()

    def test_stop_without_start_is_safe(self):
        loop, *_ = self._make_loop()
        loop.stop()  # should not raise

    def test_match_no_kill_switch(self):
        """When positions match, kill switch should NOT fire."""
        ui = _make_ui(has_position=True, side="long", size=1, symbol="MNQ")
        expected = {"MNQ": {"side": "long", "size": 1}}
        loop, driver, fail_safe, _, on_alert = self._make_loop(
            ui_state=ui, expected_positions=expected
        )

        loop.start()
        time.sleep(0.2)
        loop.stop()

        assert not fail_safe.kill_switch_active
        on_alert.assert_not_called()
        snap = loop.snapshot()
        assert snap["check_count"] > 0
        assert snap["mismatch_count"] == 0

    def test_mismatch_triggers_kill_switch(self):
        """Missing position -> kill switch fires."""
        ui = _make_ui()  # flat
        expected = {"MNQ": {"side": "long", "size": 1}}
        loop, driver, fail_safe, _, on_alert = self._make_loop(
            ui_state=ui, expected_positions=expected
        )

        loop.start()
        time.sleep(0.2)
        loop.stop()

        assert fail_safe.kill_switch_active
        assert "reconciliation" in fail_safe._kill_reason.lower()
        on_alert.assert_called()
        level, msg = on_alert.call_args[0]
        assert level == "CRITICAL"
        assert "MISMATCH" in msg

    def test_unexpected_position_triggers_kill_switch(self):
        """UI shows position but none expected -> kill switch."""
        ui = _make_ui(has_position=True, side="short", size=2, symbol="MES")
        loop, driver, fail_safe, _, on_alert = self._make_loop(ui_state=ui)

        loop.start()
        time.sleep(0.2)
        loop.stop()

        assert fail_safe.kill_switch_active
        on_alert.assert_called()

    def test_ui_read_failure_triggers_after_max(self):
        """Persistent UI read failures activate kill switch."""
        ui = _make_ui(window_found=False)
        loop, driver, fail_safe, _, on_alert = self._make_loop(
            ui_state=ui, max_failures=2, interval=0.05
        )

        loop.start()
        time.sleep(0.4)
        loop.stop()

        assert fail_safe.kill_switch_active
        assert "failure" in fail_safe._kill_reason.lower()
        on_alert.assert_called()

    def test_ui_read_errors_count_as_failure(self):
        """read_errors in UIState count as a failure."""
        ui = _make_ui(read_errors=["position: timeout"])
        loop, driver, fail_safe, _, on_alert = self._make_loop(
            ui_state=ui, max_failures=2, interval=0.05
        )

        loop.start()
        time.sleep(0.4)
        loop.stop()

        assert fail_safe.kill_switch_active

    def test_consecutive_failures_reset_on_success(self):
        """A successful read resets the consecutive failure counter."""
        good_ui = _make_ui()
        bad_ui = _make_ui(window_found=False)

        driver = MagicMock()
        # First: bad, then good, then bad (should NOT hit max_failures=3)
        driver.read_ui_state.side_effect = [bad_ui, good_ui, bad_ui, good_ui]

        fail_safe = _make_fail_safe()
        pm = PositionMonitor()
        on_alert = MagicMock()

        loop = ReconciliationLoop(
            driver=driver,
            fail_safe=fail_safe,
            position_monitor=pm,
            on_alert=on_alert,
            interval_seconds=0.05,
            max_consecutive_failures=3,
        )

        loop.start()
        time.sleep(0.3)
        loop.stop()

        # Should NOT have hit kill switch (failures never consecutive 3x)
        assert not fail_safe.kill_switch_active

    def test_snapshot_structure(self):
        loop, *_ = self._make_loop()
        snap = loop.snapshot()
        assert "running" in snap
        assert "check_count" in snap
        assert "mismatch_count" in snap
        assert "consecutive_failures" in snap
        assert "interval_seconds" in snap
        assert "last_result" in snap
        assert snap["last_result"] is None  # nothing checked yet

    def test_snapshot_after_check(self):
        ui = _make_ui(has_position=True, side="long", size=1, symbol="MNQ")
        expected = {"MNQ": {"side": "long", "size": 1}}
        loop, *_ = self._make_loop(ui_state=ui, expected_positions=expected)

        loop.start()
        time.sleep(0.2)
        loop.stop()

        snap = loop.snapshot()
        assert snap["check_count"] > 0
        assert snap["last_result"] is not None
        assert snap["last_result"]["matched"] is True

    def test_no_alert_callback_is_safe(self):
        """Loop works fine when on_alert is None."""
        ui = _make_ui()  # flat
        expected = {"MNQ": {"side": "long", "size": 1}}

        driver = MagicMock()
        driver.read_ui_state.return_value = ui
        fail_safe = _make_fail_safe()
        pm = PositionMonitor()
        pm.update("MNQ", "long", 1)

        loop = ReconciliationLoop(
            driver=driver,
            fail_safe=fail_safe,
            position_monitor=pm,
            on_alert=None,
            interval_seconds=0.05,
        )

        loop.start()
        time.sleep(0.2)
        loop.stop()

        # Kill switch fires but no crash from missing callback
        assert fail_safe.kill_switch_active

    def test_daemon_thread(self):
        """Thread is a daemon so it won't block process exit."""
        loop, *_ = self._make_loop()
        loop.start()
        assert loop._thread.daemon
        loop.stop()

    def test_both_flat_match_no_kill(self):
        """No expected positions and UI flat -> no mismatch."""
        loop, driver, fail_safe, _, on_alert = self._make_loop()

        loop.start()
        time.sleep(0.2)
        loop.stop()

        assert not fail_safe.kill_switch_active
        on_alert.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# ReconciliationLoop._tick — direct unit tests (no threading)
# ═════════════════════════════════════════════════════════════════════════════


class TestTickDirect:
    """Test _tick() directly without background thread for determinism."""

    def test_tick_match(self):
        driver = MagicMock()
        driver.read_ui_state.return_value = _make_ui()
        fail_safe = _make_fail_safe()
        pm = PositionMonitor()

        loop = ReconciliationLoop(
            driver=driver,
            fail_safe=fail_safe,
            position_monitor=pm,
            interval_seconds=1,
        )

        loop._tick()
        assert not fail_safe.kill_switch_active
        assert loop._check_count == 1
        assert loop._mismatch_count == 0

    def test_tick_mismatch(self):
        driver = MagicMock()
        driver.read_ui_state.return_value = _make_ui()  # flat
        fail_safe = _make_fail_safe()
        pm = PositionMonitor()
        pm.update("MNQ", "long", 1)
        on_alert = MagicMock()

        loop = ReconciliationLoop(
            driver=driver,
            fail_safe=fail_safe,
            position_monitor=pm,
            on_alert=on_alert,
            interval_seconds=1,
        )

        loop._tick()
        assert fail_safe.kill_switch_active
        assert loop._mismatch_count == 1
        on_alert.assert_called_once()

    def test_tick_ui_failure_increments_counter(self):
        driver = MagicMock()
        driver.read_ui_state.return_value = _make_ui(window_found=False)
        fail_safe = _make_fail_safe()
        pm = PositionMonitor()

        loop = ReconciliationLoop(
            driver=driver,
            fail_safe=fail_safe,
            position_monitor=pm,
            interval_seconds=1,
            max_consecutive_failures=5,
        )

        loop._tick()
        assert loop._consecutive_failures == 1
        assert not fail_safe.kill_switch_active  # not enough failures yet

    def test_tick_ui_failure_triggers_at_max(self):
        driver = MagicMock()
        driver.read_ui_state.return_value = _make_ui(window_found=False)
        fail_safe = _make_fail_safe()
        pm = PositionMonitor()
        on_alert = MagicMock()

        loop = ReconciliationLoop(
            driver=driver,
            fail_safe=fail_safe,
            position_monitor=pm,
            on_alert=on_alert,
            interval_seconds=1,
            max_consecutive_failures=2,
        )

        loop._tick()
        assert not fail_safe.kill_switch_active
        loop._tick()
        assert fail_safe.kill_switch_active
        on_alert.assert_called()
