"""
Tests for the execution adapter interface and OpenClaw adapter.

Covers:
- ExecutionAdapter ABC enforcement
- OpenClawAdapter interface compliance
- OrderResult / FillResult dataclasses
- Adapter injection into ExecutionController
- Controller uses adapter.name in logs
- set_run_mode propagates to adapter.dry_run
- Custom adapter swappability
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field

from execution.adapter import (
    ExecutionAdapter,
    FillResult,
    OrderResult,
)
from execution.execution_controller import ExecutionController
from execution.fail_safes import FailSafeConfig, RunMode
from execution.openclaw_adapter import OpenClawAdapter
from execution.signal_schema import ExecutionSignal
from execution.validators import PostTradeState, UIState, ValidationResult


# ══════════════════════════════════════════════════════════════════════════════
#  Stub adapter for testing swappability
# ══════════════════════════════════════════════════════════════════════════════

class StubAdapter(ExecutionAdapter):
    """Minimal concrete adapter for testing the interface contract."""

    def __init__(self) -> None:
        self._dry_run = True
        self.calls: list[str] = []

    def validate_context(self, signal):
        self.calls.append("validate_context")
        return UIState(
            window_found=True,
            window_title="StubBroker",
            account_label="STUB-ACCT-001",
            active_symbol="MNQ",
            quantity_value=1,
            buy_button_visible=True,
            sell_button_visible=True,
        )

    def place_order(self, signal):
        self.calls.append("place_order")
        return OrderResult(
            success=True,
            action="stub_market_order",
            detail="stubbed",
            order_id="ORD-001",
        )

    def confirm_fill(self, signal, timeout_ms=5000):
        self.calls.append("confirm_fill")
        post = PostTradeState(
            position_detected=True,
            position_side="long",
            position_size=1,
            fill_price=21000.0,
            order_status="filled",
            time_elapsed_ms=50,
        )
        v = ValidationResult(passed=True)
        return FillResult(confirmed=True, post_state=post, validation=v, elapsed_ms=50)

    def get_position_state(self, symbol=""):
        self.calls.append("get_position_state")
        return UIState()

    def emergency_stop(self):
        self.calls.append("emergency_stop")
        return OrderResult(success=True, action="flatten_all", detail="stubbed")

    def take_screenshot(self, label):
        self.calls.append("take_screenshot")
        return True

    @property
    def dry_run(self):
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value):
        self._dry_run = value

    @property
    def name(self):
        return "stub"


# ══════════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestABCEnforcement(unittest.TestCase):
    """ExecutionAdapter can't be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        with self.assertRaises(TypeError):
            ExecutionAdapter()  # type: ignore[abstract]


class TestOrderResult(unittest.TestCase):

    def test_defaults(self):
        r = OrderResult(success=True)
        self.assertTrue(r.success)
        self.assertEqual(r.action, "")
        self.assertEqual(r.order_id, "")
        self.assertEqual(r.elapsed_ms, 0)
        self.assertIsInstance(r.metadata, dict)

    def test_with_values(self):
        r = OrderResult(
            success=False,
            action="buy_click",
            detail="window not found",
            elapsed_ms=120,
            order_id="ORD-XYZ",
            metadata={"retries": 2},
        )
        self.assertFalse(r.success)
        self.assertEqual(r.order_id, "ORD-XYZ")
        self.assertEqual(r.metadata["retries"], 2)


class TestFillResult(unittest.TestCase):

    def test_confirmed(self):
        post = PostTradeState(position_detected=True, position_side="long", position_size=1)
        v = ValidationResult(passed=True)
        r = FillResult(confirmed=True, post_state=post, validation=v, elapsed_ms=80)
        self.assertTrue(r.confirmed)
        self.assertTrue(r.post_state.position_detected)
        self.assertTrue(r.validation.passed)

    def test_not_confirmed(self):
        post = PostTradeState()
        v = ValidationResult(passed=False)
        r = FillResult(confirmed=False, post_state=post, validation=v)
        self.assertFalse(r.confirmed)
        self.assertEqual(r.elapsed_ms, 0)


class TestOpenClawAdapter(unittest.TestCase):
    """OpenClawAdapter implements the full interface."""

    def test_name(self):
        adapter = OpenClawAdapter(dry_run=True)
        self.assertEqual(adapter.name, "openclaw")

    def test_dry_run_default(self):
        adapter = OpenClawAdapter()
        self.assertTrue(adapter.dry_run)

    def test_dry_run_toggle(self):
        adapter = OpenClawAdapter(dry_run=True)
        adapter.dry_run = False
        self.assertFalse(adapter.dry_run)

    def test_validate_context_returns_ui_state(self):
        adapter = OpenClawAdapter(dry_run=True)
        result = adapter.validate_context(None)
        self.assertIsInstance(result, UIState)

    def test_place_order_returns_order_result(self):
        adapter = OpenClawAdapter(dry_run=True)
        # Build a minimal signal (dry_run + no OpenClaw = will fail at place_order)
        result = adapter.place_order(_fake_signal())
        self.assertIsInstance(result, OrderResult)
        # Dry-run OpenClaw returns success for quantity set but then
        # returns success for click too (dry-run click)
        self.assertTrue(result.success)

    def test_confirm_fill_returns_fill_result(self):
        adapter = OpenClawAdapter(dry_run=True)
        result = adapter.confirm_fill(_fake_signal(), timeout_ms=100)
        self.assertIsInstance(result, FillResult)
        # Without OpenClaw, position won't be detected
        self.assertFalse(result.confirmed)

    def test_get_position_state_returns_ui_state(self):
        adapter = OpenClawAdapter(dry_run=True)
        result = adapter.get_position_state("MNQ")
        self.assertIsInstance(result, UIState)

    def test_emergency_stop_returns_order_result(self):
        adapter = OpenClawAdapter(dry_run=True)
        result = adapter.emergency_stop()
        self.assertIsInstance(result, OrderResult)
        self.assertFalse(result.success)  # stub — not implemented for OpenClaw

    def test_take_screenshot_returns_bool(self):
        adapter = OpenClawAdapter(dry_run=True)
        result = adapter.take_screenshot("test_label")
        self.assertIsInstance(result, bool)

    def test_driver_property(self):
        adapter = OpenClawAdapter(dry_run=True)
        self.assertIsNotNone(adapter.driver)

    def test_is_execution_adapter(self):
        adapter = OpenClawAdapter(dry_run=True)
        self.assertIsInstance(adapter, ExecutionAdapter)


class TestStubAdapterInterface(unittest.TestCase):
    """StubAdapter proves custom adapters work."""

    def test_is_execution_adapter(self):
        adapter = StubAdapter()
        self.assertIsInstance(adapter, ExecutionAdapter)

    def test_all_methods_callable(self):
        adapter = StubAdapter()
        adapter.validate_context(None)
        adapter.place_order(None)
        adapter.confirm_fill(None)
        adapter.get_position_state()
        adapter.emergency_stop()
        adapter.take_screenshot("test")
        self.assertEqual(len(adapter.calls), 6)

    def test_name(self):
        self.assertEqual(StubAdapter().name, "stub")


class TestControllerAdapterInjection(unittest.TestCase):
    """ExecutionController accepts and uses injected adapters."""

    def test_default_adapter_is_openclaw(self):
        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                output_dir=td,
            )
            self.assertIsInstance(ctrl.adapter, OpenClawAdapter)
            self.assertEqual(ctrl.adapter.name, "openclaw")

    def test_custom_adapter_injected(self):
        stub = StubAdapter()
        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                adapter=stub,
                output_dir=td,
            )
            self.assertIs(ctrl.adapter, stub)
            self.assertEqual(ctrl.adapter.name, "stub")

    def test_set_run_mode_propagates_to_adapter(self):
        stub = StubAdapter()
        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                adapter=stub,
                output_dir=td,
            )
            self.assertTrue(stub.dry_run)
            ctrl.set_run_mode(RunMode.LIVE)
            self.assertFalse(stub.dry_run)
            ctrl.set_run_mode(RunMode.DRY_RUN)
            self.assertTrue(stub.dry_run)

    def test_adapter_property_exposed(self):
        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                output_dir=td,
            )
            self.assertIsNotNone(ctrl.adapter)

    def test_driver_property_backward_compat(self):
        """ctrl.driver still works — returns underlying driver from adapter."""
        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                output_dir=td,
            )
            # OpenClawAdapter exposes .driver property
            self.assertIsNotNone(ctrl.driver)


class TestStubAdapterSignalFlow(unittest.TestCase):
    """Signals flow through a custom adapter when injected."""

    def _make_controller(self, adapter, td):
        from execution.risk_bridge import RiskBridgeConfig
        return ExecutionController(
            fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
            risk_bridge_config=RiskBridgeConfig(
                base_contracts=1, min_contracts=1, max_contracts=2,
            ),
            adapter=adapter,
            output_dir=td,
        )

    def test_dry_run_uses_validate_context(self):
        """In dry-run, adapter.validate_context is called for pre-trade check."""
        stub = StubAdapter()
        with tempfile.TemporaryDirectory() as td:
            ctrl = self._make_controller(stub, td)
            signal = _fake_live_signal()
            ctrl.on_signal(signal, size_mult=0.5)
            # validate_context should have been called
            self.assertIn("validate_context", stub.calls)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fake_signal():
    """Build an ExecutionSignal for testing adapter return types."""
    import datetime
    from execution.signal_schema import ExecutionMode, ExecutionSide, create_signal_id
    return ExecutionSignal(
        signal_id=create_signal_id(),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        symbol="MNQ",
        side=ExecutionSide.BUY,
        contracts=1,
        mode=ExecutionMode.CHALLENGE,
        stop_loss=20950.0,
        take_profit=21100.0,
    )


class _FakeLiveSignal:
    """Minimal LiveSignal stub."""
    def __init__(self):
        import datetime
        self.direction = "long"
        self.symbol = "MNQ"
        self.signal_type = None
        self.timestamp = datetime.datetime.now(datetime.timezone.utc)
        self.entry = 21000.0
        self.stop = 20950.0
        self.take_profit = 21100.0
        self.position_size = 1.0
        self.reason = "test"
        self.strategy_type = "ema50_breakout"
        self.ml_prob = 0.70
        self.percentile = 0.75
        self.quality_score = 0.65


def _fake_live_signal():
    return _FakeLiveSignal()


if __name__ == "__main__":
    unittest.main()
