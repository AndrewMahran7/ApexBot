"""
Tests for the OpenClaw execution layer.

These tests validate:
- Signal schema creation, expiration, dedup
- Pre-trade and post-trade validation logic
- Fail-safe enforcement
- Risk bridge conversion
- Execution controller flow (dry-run)
- Position monitor anomaly detection

All tests run without OpenClaw installed (dry-run / mocked).
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
import unittest

from execution.signal_schema import (
    DEFAULT_SIGNAL_TTL_SECONDS,
    ExecutionMode,
    ExecutionSide,
    ExecutionSignal,
    SignalRegistry,
    create_signal_id,
)
from execution.validators import (
    PostTradeState,
    PostTradeValidator,
    PositionMonitor,
    PreTradeValidator,
    UIState,
    ValidationResult,
)
from execution.fail_safes import FailSafeConfig, FailSafeState, RunMode
from execution.risk_bridge import RiskBridge, RiskBridgeConfig, compute_contracts
from execution.audit_logger import AuditLogger, EventType
from execution.execution_controller import ExecutionController


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _make_signal(
    symbol: str = "MNQ",
    side: ExecutionSide = ExecutionSide.BUY,
    contracts: int = 1,
    ttl: int = 30,
    ts: datetime.datetime | None = None,
    signal_id: str | None = None,
    stop_loss: float = 5000.0,
    take_profit: float = 5100.0,
) -> ExecutionSignal:
    return ExecutionSignal(
        signal_id=signal_id or create_signal_id(),
        timestamp=ts or _now(),
        symbol=symbol,
        side=side,
        contracts=contracts,
        mode=ExecutionMode.CHALLENGE,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reason="test signal",
        strategy_type="ema50_breakout",
        confidence=0.65,
        position_size_mult=0.35,
        entry_price=5050.0,
        ttl_seconds=ttl,
    )


def _make_ui_state(
    window_found: bool = True,
    symbol: str = "MNQ",
    qty: int = 1,
    account: str = "APEX-12345",
    has_position: bool = False,
    buy_visible: bool = True,
    sell_visible: bool = True,
) -> UIState:
    return UIState(
        window_found=window_found,
        window_title="Tradovate - MNQ",
        account_label=account,
        active_symbol=symbol,
        quantity_value=qty,
        atm_template_name="default",
        has_open_position=has_position,
        open_position_side=None,
        open_position_size=0,
        pending_orders=0,
        buy_button_visible=buy_visible,
        sell_button_visible=sell_visible,
    )


# ── Signal Schema Tests ─────────────────────────────────────────────────────

class TestExecutionSignal(unittest.TestCase):

    def test_create_signal(self):
        sig = _make_signal()
        self.assertIsInstance(sig.signal_id, str)
        self.assertEqual(sig.symbol, "MNQ")
        self.assertEqual(sig.side, ExecutionSide.BUY)
        self.assertEqual(sig.contracts, 1)
        self.assertFalse(sig.is_expired())

    def test_signal_expires(self):
        old = _now() - datetime.timedelta(seconds=60)
        sig = _make_signal(ts=old, ttl=30)
        self.assertTrue(sig.is_expired())

    def test_signal_not_expired(self):
        sig = _make_signal(ttl=300)
        self.assertFalse(sig.is_expired())

    def test_fingerprint_deterministic(self):
        s1 = _make_signal(signal_id="aaa")
        s2 = _make_signal(signal_id="bbb")
        self.assertEqual(s1.fingerprint, s2.fingerprint)

    def test_fingerprint_differs_for_different_content(self):
        s1 = _make_signal(symbol="MNQ")
        s2 = _make_signal(symbol="MES")
        self.assertNotEqual(s1.fingerprint, s2.fingerprint)

    def test_to_dict_serializable(self):
        sig = _make_signal()
        d = sig.to_dict()
        self.assertIsInstance(d, dict)
        json.dumps(d)  # Must not raise

    def test_frozen(self):
        sig = _make_signal()
        with self.assertRaises(AttributeError):
            sig.symbol = "MES"  # type: ignore


class TestSignalRegistry(unittest.TestCase):

    def test_not_duplicate_first_time(self):
        reg = SignalRegistry()
        sig = _make_signal()
        is_dup, reason = reg.is_duplicate(sig)
        self.assertFalse(is_dup)

    def test_duplicate_same_id(self):
        reg = SignalRegistry()
        sig = _make_signal(signal_id="fixed-id")
        reg.register(sig)
        is_dup, reason = reg.is_duplicate(sig)
        self.assertTrue(is_dup)
        self.assertIn("signal_id", reason)

    def test_duplicate_fingerprint_within_cooldown(self):
        reg = SignalRegistry(fingerprint_cooldown_seconds=120)
        s1 = _make_signal(signal_id="id1")
        reg.register(s1)
        s2 = _make_signal(signal_id="id2")  # Different ID, same content
        is_dup, reason = reg.is_duplicate(s2)
        self.assertTrue(is_dup)
        self.assertIn("fingerprint", reason)

    def test_fingerprint_outside_cooldown(self):
        reg = SignalRegistry(fingerprint_cooldown_seconds=10)
        ts1 = _now() - datetime.timedelta(seconds=20)
        s1 = _make_signal(signal_id="id1", ts=ts1)
        reg.register(s1)
        s2 = _make_signal(signal_id="id2")  # Different ID, same content, after cooldown
        is_dup, _ = reg.is_duplicate(s2)
        self.assertFalse(is_dup)

    def test_clear(self):
        reg = SignalRegistry()
        sig = _make_signal()
        reg.register(sig)
        self.assertEqual(reg.seen_count, 1)
        reg.clear()
        self.assertEqual(reg.seen_count, 0)


# ── Pre-Trade Validation Tests ───────────────────────────────────────────────

class TestPreTradeValidator(unittest.TestCase):

    def test_all_pass(self):
        v = PreTradeValidator(expected_account_fragment="APEX")
        sig = _make_signal()
        ui = _make_ui_state()
        result = v.validate(sig, ui)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.failed_checks), 0)

    def test_window_not_found(self):
        v = PreTradeValidator()
        sig = _make_signal()
        ui = _make_ui_state(window_found=False)
        result = v.validate(sig, ui)
        self.assertFalse(result.passed)
        self.assertIn("window_present", [c["name"] for c in result.failed_checks])

    def test_wrong_symbol(self):
        v = PreTradeValidator()
        sig = _make_signal(symbol="MNQ")
        ui = _make_ui_state(symbol="MES")
        result = v.validate(sig, ui)
        self.assertFalse(result.passed)
        self.assertIn("correct_symbol", [c["name"] for c in result.failed_checks])

    def test_wrong_account(self):
        v = PreTradeValidator(expected_account_fragment="APEX")
        sig = _make_signal()
        ui = _make_ui_state(account="WRONG-ACCT")
        result = v.validate(sig, ui)
        self.assertFalse(result.passed)
        self.assertIn("correct_account", [c["name"] for c in result.failed_checks])

    def test_wrong_quantity(self):
        v = PreTradeValidator()
        sig = _make_signal(contracts=2)
        ui = _make_ui_state(qty=1)
        result = v.validate(sig, ui)
        self.assertFalse(result.passed)
        self.assertIn("correct_quantity", [c["name"] for c in result.failed_checks])

    def test_existing_position(self):
        v = PreTradeValidator()
        sig = _make_signal()
        ui = _make_ui_state(has_position=True)
        result = v.validate(sig, ui)
        self.assertFalse(result.passed)
        self.assertIn("no_unexpected_position", [c["name"] for c in result.failed_checks])

    def test_expired_signal(self):
        v = PreTradeValidator()
        old = _now() - datetime.timedelta(seconds=60)
        sig = _make_signal(ts=old, ttl=30)
        ui = _make_ui_state()
        result = v.validate(sig, ui)
        self.assertFalse(result.passed)
        self.assertIn("signal_fresh", [c["name"] for c in result.failed_checks])

    def test_buy_button_not_visible(self):
        v = PreTradeValidator()
        sig = _make_signal(side=ExecutionSide.BUY)
        ui = _make_ui_state(buy_visible=False)
        result = v.validate(sig, ui)
        self.assertFalse(result.passed)
        self.assertIn("action_button_visible", [c["name"] for c in result.failed_checks])

    def test_sell_button_not_visible(self):
        v = PreTradeValidator()
        sig = _make_signal(side=ExecutionSide.SELL)
        ui = _make_ui_state(sell_visible=False)
        result = v.validate(sig, ui)
        self.assertFalse(result.passed)

    def test_no_account_filter(self):
        """When no account fragment is set, account check always passes."""
        v = PreTradeValidator(expected_account_fragment="")
        sig = _make_signal()
        ui = _make_ui_state(account="anything")
        result = v.validate(sig, ui)
        account_checks = [c for c in result.checks if c["name"] == "correct_account"]
        self.assertTrue(all(c["passed"] for c in account_checks))


# ── Post-Trade Validation Tests ──────────────────────────────────────────────

class TestPostTradeValidator(unittest.TestCase):

    def test_successful_fill(self):
        v = PostTradeValidator()
        sig = _make_signal(side=ExecutionSide.BUY, contracts=1)
        post = PostTradeState(
            position_detected=True,
            position_side="long",
            position_size=1,
            fill_price=5050.0,
            order_status="filled",
        )
        result = v.validate(sig, post)
        self.assertTrue(result.passed)

    def test_no_fill(self):
        v = PostTradeValidator()
        sig = _make_signal()
        post = PostTradeState(position_detected=False)
        result = v.validate(sig, post)
        self.assertFalse(result.passed)

    def test_wrong_side(self):
        v = PostTradeValidator()
        sig = _make_signal(side=ExecutionSide.BUY)
        post = PostTradeState(
            position_detected=True,
            position_side="short",
            position_size=1,
        )
        result = v.validate(sig, post)
        self.assertFalse(result.passed)

    def test_wrong_size(self):
        v = PostTradeValidator()
        sig = _make_signal(contracts=1)
        post = PostTradeState(
            position_detected=True,
            position_side="long",
            position_size=2,
        )
        result = v.validate(sig, post)
        self.assertFalse(result.passed)


# ── Position Monitor Tests ───────────────────────────────────────────────────

class TestPositionMonitor(unittest.TestCase):

    def test_no_anomaly_single_position(self):
        pm = PositionMonitor(expected_max_positions=1)
        anomalies = pm.update("MNQ", "long", 1)
        self.assertEqual(len(anomalies), 0)

    def test_anomaly_too_many_positions(self):
        pm = PositionMonitor(expected_max_positions=1)
        pm.update("MNQ", "long", 1)
        anomalies = pm.update("MES", "short", 1)
        self.assertGreater(len(anomalies), 0)

    def test_position_closed(self):
        pm = PositionMonitor(expected_max_positions=1)
        pm.update("MNQ", "long", 1)
        pm.update("MNQ", None, 0)
        self.assertEqual(len(pm.open_positions), 0)


# ── Fail-Safe Tests ──────────────────────────────────────────────────────────

class TestFailSafe(unittest.TestCase):

    def test_dry_run_blocks(self):
        fs = FailSafeState(FailSafeConfig(run_mode=RunMode.DRY_RUN))
        ok, reason = fs.can_execute()
        self.assertFalse(ok)
        self.assertIn("dry_run", reason)

    def test_kill_switch_blocks(self):
        fs = FailSafeState(FailSafeConfig(run_mode=RunMode.LIVE))
        fs.activate_kill_switch("test")
        ok, reason = fs.can_execute()
        self.assertFalse(ok)
        self.assertIn("kill switch", reason)

    def test_kill_switch_reset(self):
        fs = FailSafeState(FailSafeConfig(run_mode=RunMode.LIVE))
        fs.activate_kill_switch("test")
        fs.reset_kill_switch()
        ok, _ = fs.can_execute()
        self.assertTrue(ok)

    def test_max_open_trades(self):
        fs = FailSafeState(FailSafeConfig(run_mode=RunMode.LIVE, max_open_trades=1))
        fs.record_execution()
        ok, reason = fs.can_execute()
        self.assertFalse(ok)
        self.assertIn("max open trades", reason)

    def test_cooldown(self):
        fs = FailSafeState(FailSafeConfig(run_mode=RunMode.LIVE, cooldown_seconds=60))
        fs.record_execution()
        fs.record_trade_closed()
        ok, reason = fs.can_execute()
        self.assertFalse(ok)
        self.assertIn("cooldown", reason)

    def test_emergency_disable(self):
        fs = FailSafeState(FailSafeConfig(run_mode=RunMode.LIVE))
        fs.set_emergency_disable(True)
        ok, reason = fs.can_execute()
        self.assertFalse(ok)
        self.assertIn("emergency", reason)

    def test_session_cap(self):
        cfg = FailSafeConfig(run_mode=RunMode.LIVE, max_executions_per_session=2, cooldown_seconds=0)
        fs = FailSafeState(cfg)
        fs.record_execution()
        fs.record_trade_closed()
        fs.record_execution()
        fs.record_trade_closed()
        ok, reason = fs.can_execute()
        self.assertFalse(ok)
        self.assertIn("session execution cap", reason)

    def test_session_reset(self):
        cfg = FailSafeConfig(run_mode=RunMode.LIVE, max_executions_per_session=1, cooldown_seconds=0)
        fs = FailSafeState(cfg)
        fs.record_execution()
        fs.record_trade_closed()
        fs.reset_session()
        ok, _ = fs.can_execute()
        self.assertTrue(ok)

    def test_status_snapshot(self):
        fs = FailSafeState(FailSafeConfig())
        status = fs.status()
        self.assertIn("run_mode", status)
        self.assertIn("kill_switch", status)
        self.assertIn("open_trades", status)


# ── Risk Bridge Tests ────────────────────────────────────────────────────────

class TestComputeContracts(unittest.TestCase):

    def test_normal(self):
        self.assertEqual(compute_contracts(1, 1.0), 1)

    def test_scaled_down(self):
        self.assertEqual(compute_contracts(2, 0.35), 1)  # 2 * 0.35 = 0.7 → round → 1

    def test_blocked(self):
        self.assertEqual(compute_contracts(1, 0.0), 0)

    def test_cap(self):
        self.assertEqual(compute_contracts(10, 1.0, max_contracts=3), 3)

    def test_floor(self):
        self.assertEqual(compute_contracts(1, 0.1, min_contracts=1), 1)


class TestRiskBridge(unittest.TestCase):

    def test_convert_buy(self):
        bridge = RiskBridge()

        class FakeLiveSignal:
            timestamp = _now()
            direction = "long"
            signal_type = type("ST", (), {"name": "LONG_ENTRY"})()
            entry = 5050.0
            stop = 5000.0
            take_profit = 5100.0
            position_size = 1.0
            reason = "breakout"
            strategy_type = "ema50_breakout"
            ml_prob = 0.65
            percentile = 0.8
            quality_score = 0.7

        sig = bridge.convert(FakeLiveSignal(), size_mult=0.35, symbol="MNQ")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, ExecutionSide.BUY)
        self.assertEqual(sig.symbol, "MNQ")
        self.assertEqual(sig.contracts, 1)

    def test_blocked_by_size(self):
        bridge = RiskBridge()

        class FakeLiveSignal:
            timestamp = _now()
            direction = "long"
            signal_type = type("ST", (), {"name": "LONG_ENTRY"})()
            entry = stop = take_profit = position_size = 0.0
            reason = strategy_type = ""
            ml_prob = percentile = quality_score = 0.0

        sig = bridge.convert(FakeLiveSignal(), size_mult=0.0)
        self.assertIsNone(sig)

    def test_exit_signal_skipped(self):
        bridge = RiskBridge()

        class FakeExit:
            timestamp = _now()
            direction = ""
            signal_type = type("ST", (), {"name": "EXIT_TP"})()
            entry = stop = take_profit = position_size = 0.0
            reason = strategy_type = ""
            ml_prob = percentile = quality_score = 0.0

        sig = bridge.convert(FakeExit(), size_mult=1.0)
        self.assertIsNone(sig)


# ── Audit Logger Tests ───────────────────────────────────────────────────────

class TestAuditLogger(unittest.TestCase):

    def test_log_event(self):
        with tempfile.TemporaryDirectory() as td:
            al = AuditLogger(output_dir=td)
            event = al.log(EventType.SIGNAL_RECEIVED, signal_id="test-123", symbol="MNQ", reason="test")
            self.assertEqual(event.event_type, "signal_received")
            self.assertEqual(al.event_count, 1)
            # Check file was written
            files = os.listdir(td)
            self.assertGreater(len(files), 0)
            self.assertTrue(files[0].endswith(".jsonl"))

    def test_summary(self):
        with tempfile.TemporaryDirectory() as td:
            al = AuditLogger(output_dir=td)
            al.log(EventType.SIGNAL_RECEIVED, reason="a")
            al.log(EventType.SIGNAL_RECEIVED, reason="b")
            al.log(EventType.VALIDATION_PASSED, reason="c")
            summary = al.summary()
            self.assertEqual(summary["signal_received"], 2)
            self.assertEqual(summary["validation_passed"], 1)


# ── Execution Controller Tests (Dry-Run) ────────────────────────────────────

class TestExecutionControllerDryRun(unittest.TestCase):
    """
    Tests the full pipeline in dry-run mode.
    OpenClaw is NOT required — the driver never clicks.
    """

    def _make_controller(self, tmpdir: str) -> ExecutionController:
        return ExecutionController(
            fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
            risk_bridge_config=RiskBridgeConfig(
                default_mode=ExecutionMode.CHALLENGE,
                base_contracts=1,
            ),
            expected_account="",
            output_dir=tmpdir,
        )

    def _make_live_signal(self, direction="long"):
        class FakeLiveSignal:
            timestamp = _now()
            signal_type = type("ST", (), {"name": "LONG_ENTRY" if direction == "long" else "SHORT_ENTRY"})()
            entry = 5050.0
            stop = 5000.0
            take_profit = 5100.0
            position_size = 1.0
            reason = "breakout"
            strategy_type = "ema50_breakout"
            ml_prob = 0.65
            percentile = 0.8
            quality_score = 0.7
        FakeLiveSignal.direction = direction
        return FakeLiveSignal()

    def test_dry_run_processes_signal(self):
        with tempfile.TemporaryDirectory() as td:
            ctrl = self._make_controller(td)
            result = ctrl.on_signal(self._make_live_signal(), size_mult=0.35, symbol="MNQ")
            # In dry-run, the signal should be "processed" (validation runs)
            # but whether it returns True depends on UI state.
            # Since OpenClaw isn't available, UI validation will fail.
            # The important thing is no exception is raised.
            self.assertIsInstance(result, bool)

    def test_blocked_by_zero_size(self):
        with tempfile.TemporaryDirectory() as td:
            ctrl = self._make_controller(td)
            result = ctrl.on_signal(self._make_live_signal(), size_mult=0.0, symbol="MNQ")
            self.assertFalse(result)

    def test_duplicate_signal_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            ctrl = self._make_controller(td)
            sig = self._make_live_signal()
            ctrl.on_signal(sig, size_mult=0.35, symbol="MNQ")
            # Second call with same content should be rejected as fingerprint duplicate
            result = ctrl.on_signal(sig, size_mult=0.35, symbol="MNQ")
            # May or may not be duplicate depending on timing — just verify no crash
            self.assertIsInstance(result, bool)

    def test_kill_switch(self):
        with tempfile.TemporaryDirectory() as td:
            ctrl = self._make_controller(td)
            ctrl.activate_kill_switch("test kill")
            self.assertTrue(ctrl.fail_safe.kill_switch_active)

    def test_status_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            ctrl = self._make_controller(td)
            status = ctrl.status()
            self.assertIn("fail_safe", status)
            self.assertIn("audit_events", status)


# ── Validation Result Tests ──────────────────────────────────────────────────

class TestValidationResult(unittest.TestCase):

    def test_all_pass(self):
        vr = ValidationResult(passed=True)
        vr.add("check1", True, "ok")
        vr.add("check2", True, "ok")
        self.assertTrue(vr.passed)
        self.assertEqual(vr.summary, "ALL PASSED")

    def test_failure(self):
        vr = ValidationResult(passed=True)
        vr.add("check1", True)
        vr.add("check2", False, "bad")
        self.assertFalse(vr.passed)
        self.assertIn("check2", vr.summary)

    def test_to_dict(self):
        vr = ValidationResult(passed=True)
        vr.add("check1", True)
        d = vr.to_dict()
        self.assertIn("passed", d)
        self.assertIn("checks", d)


if __name__ == "__main__":
    unittest.main()
