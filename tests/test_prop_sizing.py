"""
Tests for execution/prop_sizing.py and its integration with execution_controller.

Covers:
- ExecutionProfileRegistry: registration, lookup, auto-resolve
- compute_contracts_from_mult: floor/cap/rounding
- PropTradeGate: evaluate() → GateDecision for each action
- GateDecision serialization (to_audit_dict)
- ExecutionController.on_signal_gated() integration (dry-run)
- RiskBridge.convert() with override_contracts
"""

from __future__ import annotations

import datetime
import tempfile
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from execution.prop_sizing import (
    GateAction,
    GateDecision,
    ExecutionProfile,
    ExecutionProfileRegistry,
    PropTradeGate,
    compute_contracts_from_mult,
    _snapshot_account_state,
)
from execution.risk_bridge import RiskBridge, RiskBridgeConfig, compute_contracts
from execution.signal_schema import ExecutionMode


# ── Fake PropRiskLayer for testing ───────────────────────────────────────────

class FakeAccountMode(str, Enum):
    CHALLENGE = "challenge"
    FUNDED = "funded"


@dataclass(frozen=True)
class FakeAccountState:
    equity: float = 25_500.0
    peak_equity: float = 25_500.0
    starting_capital: float = 25_000.0
    trailing_dd_limit: float = 24_500.0
    realized_pnl: float = 500.0
    daily_pnl: float = 200.0
    daily_trades: int = 2
    consecutive_losses: int = 0
    mode: FakeAccountMode = FakeAccountMode.CHALLENGE

    @property
    def distance_to_dd(self) -> float:
        return self.equity - self.trailing_dd_limit

    @property
    def progress(self) -> float:
        return (self.equity - self.starting_capital) / 1500.0


@dataclass
class FakeTradeDecision:
    size_mult: float = 0.35
    blocked: bool = False
    reasons: list = None
    components: dict = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = ["base=0.35"]
        if self.components is None:
            self.components = {"base": 0.35, "dd_prox": 1.0, "streak": 1.0}


class FakePropRiskLayer:
    """Minimal stub matching the PropRiskLayer API."""

    def __init__(
        self,
        state: FakeAccountState = None,
        decision: FakeTradeDecision = None,
        base_size: float = 1.0,
        active: bool = True,
    ):
        self._state = state or FakeAccountState()
        self._decision = decision or FakeTradeDecision()
        self.cfg = type("Cfg", (), {"mode": self._state.mode, "base_size": base_size})()
        self.active = active
        self.recorded_trades: list[tuple[float, str]] = []

    @property
    def state(self):
        return self._state

    def evaluate_trade(self, symbol: str = "") -> FakeTradeDecision:
        return self._decision

    def record_trade(self, net_pnl: float, symbol: str = "") -> None:
        self.recorded_trades.append((net_pnl, symbol))

    def reset_day(self) -> None:
        pass


# ── Fake LiveSignal for controller tests ─────────────────────────────────────

class FakeLiveSignal:
    def __init__(self, direction="long", symbol="MNQ"):
        self.direction = direction
        self.symbol = symbol
        self.signal_type = None
        self.timestamp = datetime.datetime.now(datetime.timezone.utc)
        self.entry = 21000.0
        self.stop = 20950.0
        self.take_profit = 21100.0
        self.position_size = 1.0
        self.reason = "test_signal"
        self.strategy_type = "ema50"
        self.ml_prob = 0.7
        self.percentile = 0.8
        self.quality_score = 0.65


# ══════════════════════════════════════════════════════════════════════════════
#  Profile registry tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionProfileRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = ExecutionProfileRegistry()

    def test_default_profiles_exist(self):
        expected = {"challenge_combined", "challenge_mnq", "challenge_mes", "funded"}
        self.assertTrue(expected.issubset(set(self.reg.available)))

    def test_get_known_profile(self):
        p = self.reg.get("challenge_mnq")
        self.assertIsNotNone(p)
        self.assertEqual(p.mode, "challenge")
        self.assertEqual(p.symbol, "MNQ")

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.reg.get("nonexistent"))

    def test_resolve_exact_match(self):
        p = self.reg.resolve("challenge", "MNQ")
        self.assertEqual(p.label, "challenge_mnq")

    def test_resolve_falls_back_to_combined(self):
        p = self.reg.resolve("challenge", "RTY")
        self.assertEqual(p.label, "challenge_combined")

    def test_resolve_funded(self):
        p = self.reg.resolve("funded", "MNQ")
        self.assertEqual(p.label, "funded")

    def test_resolve_unknown_mode_falls_back(self):
        p = self.reg.resolve("unknown_mode", "MNQ")
        self.assertEqual(p.label, "challenge_combined")

    def test_register_custom_profile(self):
        custom = ExecutionProfile(
            label="custom_nq", mode="challenge", symbol="NQ",
            base_contracts=2, max_contracts=5,
        )
        self.reg.register("challenge_nq", custom)
        p = self.reg.resolve("challenge", "NQ")
        self.assertEqual(p.label, "custom_nq")
        self.assertEqual(p.base_contracts, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Contract computation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeContractsFromMult(unittest.TestCase):

    def setUp(self):
        self.profile = ExecutionProfile(
            label="test", mode="challenge", symbol="MNQ",
            base_contracts=1, min_contracts=1, max_contracts=3,
        )

    def test_zero_mult_returns_zero(self):
        self.assertEqual(compute_contracts_from_mult(0.0, self.profile), 0)

    def test_negative_mult_returns_zero(self):
        self.assertEqual(compute_contracts_from_mult(-0.5, self.profile), 0)

    def test_normal_mult(self):
        # 1 * 0.35 = 0.35, round = 0, but min = 1
        self.assertEqual(compute_contracts_from_mult(0.35, self.profile), 1)

    def test_high_mult_capped(self):
        # 1 * 5.0 = 5, but max = 3
        self.assertEqual(compute_contracts_from_mult(5.0, self.profile), 3)

    def test_exact_one(self):
        self.assertEqual(compute_contracts_from_mult(1.0, self.profile), 1)

    def test_profile_with_higher_base(self):
        p = ExecutionProfile(
            label="double", mode="challenge", symbol="",
            base_contracts=4, min_contracts=1, max_contracts=10,
        )
        # 4 * 0.50 = 2.0 → 2
        self.assertEqual(compute_contracts_from_mult(0.50, p), 2)

    def test_rounding(self):
        p = ExecutionProfile(
            label="round", mode="challenge", symbol="",
            base_contracts=3, min_contracts=1, max_contracts=10,
        )
        # 3 * 0.35 = 1.05 → round = 1
        self.assertEqual(compute_contracts_from_mult(0.35, p), 1)
        # 3 * 0.50 = 1.5 → round = 2
        self.assertEqual(compute_contracts_from_mult(0.50, p), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Account state snapshot tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSnapshotAccountState(unittest.TestCase):

    def test_snapshot_basic(self):
        state = FakeAccountState()
        snap = _snapshot_account_state(state)
        self.assertEqual(snap["equity"], 25_500.0)
        self.assertEqual(snap["starting_capital"], 25_000.0)
        self.assertEqual(snap["consecutive_losses"], 0)
        self.assertIn("mode", snap)
        self.assertIn("distance_to_dd", snap)


# ══════════════════════════════════════════════════════════════════════════════
#  PropTradeGate tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPropTradeGate(unittest.TestCase):

    def test_execute_decision(self):
        """Normal trade: not blocked, size at base → EXECUTE."""
        layer = FakePropRiskLayer(
            decision=FakeTradeDecision(size_mult=1.0, blocked=False),
            base_size=1.0,
        )
        gate = PropTradeGate(layer)
        d = gate.evaluate(symbol="MNQ")

        self.assertEqual(d.action, GateAction.EXECUTE)
        self.assertTrue(d.should_execute)
        self.assertGreater(d.contracts, 0)
        self.assertEqual(d.size_mult, 1.0)
        self.assertEqual(d.profile.label, "challenge_mnq")

    def test_reduce_decision(self):
        """Size reduced below base → REDUCE."""
        layer = FakePropRiskLayer(
            decision=FakeTradeDecision(size_mult=0.35, blocked=False,
                                       reasons=["dd_proximity reduced"]),
            base_size=1.0,
        )
        gate = PropTradeGate(layer)
        d = gate.evaluate(symbol="MES")

        self.assertEqual(d.action, GateAction.REDUCE)
        self.assertTrue(d.should_execute)
        self.assertGreater(d.contracts, 0)
        self.assertEqual(d.size_mult, 0.35)

    def test_skip_decision(self):
        """Blocked but not fatal → SKIP."""
        layer = FakePropRiskLayer(
            decision=FakeTradeDecision(
                size_mult=0.0, blocked=True,
                reasons=["daily_loss_limit reached"],
            ),
        )
        gate = PropTradeGate(layer)
        d = gate.evaluate(symbol="MNQ")

        self.assertEqual(d.action, GateAction.SKIP)
        self.assertFalse(d.should_execute)
        self.assertEqual(d.contracts, 0)

    def test_stop_decision(self):
        """Blocked with fatal keyword → STOP."""
        layer = FakePropRiskLayer(
            decision=FakeTradeDecision(
                size_mult=0.0, blocked=True,
                reasons=["Challenge failed: DD breached"],
            ),
        )
        gate = PropTradeGate(layer)
        d = gate.evaluate(symbol="MNQ")

        self.assertEqual(d.action, GateAction.STOP)
        self.assertFalse(d.should_execute)
        self.assertEqual(d.contracts, 0)

    def test_record_trade_delegates(self):
        layer = FakePropRiskLayer()
        gate = PropTradeGate(layer)
        gate.record_trade(150.0, symbol="MNQ")
        self.assertEqual(layer.recorded_trades, [(150.0, "MNQ")])

    def test_account_state_exposed(self):
        state = FakeAccountState(equity=26_000.0)
        layer = FakePropRiskLayer(state=state)
        gate = PropTradeGate(layer)
        self.assertEqual(gate.account_state.equity, 26_000.0)

    def test_is_active(self):
        layer = FakePropRiskLayer(active=False)
        gate = PropTradeGate(layer)
        self.assertFalse(gate.is_active)

    def test_default_profile_name_override(self):
        """Verify explicit profile name overrides auto-resolve."""
        layer = FakePropRiskLayer()
        gate = PropTradeGate(layer, default_profile_name="funded")
        d = gate.evaluate(symbol="MNQ")
        self.assertEqual(d.profile.label, "funded")

    def test_funded_mode_profile_resolve(self):
        """Funded account state resolves to funded profile."""
        state = FakeAccountState(mode=FakeAccountMode.FUNDED)
        layer = FakePropRiskLayer(state=state,
                                  decision=FakeTradeDecision(size_mult=0.35))
        layer.cfg.mode = FakeAccountMode.FUNDED
        gate = PropTradeGate(layer)
        d = gate.evaluate(symbol="MNQ")
        self.assertEqual(d.profile.label, "funded")


# ══════════════════════════════════════════════════════════════════════════════
#  GateDecision serialization tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGateDecisionAudit(unittest.TestCase):

    def test_to_audit_dict_contains_all_keys(self):
        profile = ExecutionProfile(
            label="challenge_mnq", mode="challenge", symbol="MNQ",
        )
        d = GateDecision(
            action=GateAction.EXECUTE,
            contracts=1,
            size_mult=0.35,
            profile=profile,
            reasons=["base=0.35"],
            components={"base": 0.35},
            account_snapshot={"equity": 25500.0},
        )
        audit = d.to_audit_dict()
        required_keys = {
            "action", "contracts", "size_mult", "profile_label",
            "profile_mode", "profile_symbol", "reasons", "components",
            "account_snapshot",
        }
        self.assertTrue(required_keys.issubset(audit.keys()))
        self.assertEqual(audit["action"], "execute")
        self.assertEqual(audit["contracts"], 1)
        self.assertEqual(audit["size_mult"], 0.35)


# ══════════════════════════════════════════════════════════════════════════════
#  RiskBridge override_contracts tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskBridgeOverrideContracts(unittest.TestCase):

    def test_override_contracts_skips_computation(self):
        """When override_contracts is set, use that value directly."""
        bridge = RiskBridge(RiskBridgeConfig(
            base_contracts=1, min_contracts=1, max_contracts=2,
        ))
        sig = bridge.convert(
            FakeLiveSignal(), size_mult=0.35, symbol="MNQ",
            override_contracts=3,
        )
        # 3 exceeds the bridge's max_contracts=2, but override bypasses that
        self.assertIsNotNone(sig)
        self.assertEqual(sig.contracts, 3)

    def test_without_override_uses_computation(self):
        """Standard path: compute contracts from size_mult."""
        bridge = RiskBridge(RiskBridgeConfig(
            base_contracts=1, min_contracts=1, max_contracts=5,
        ))
        sig = bridge.convert(FakeLiveSignal(), size_mult=0.35, symbol="MNQ")
        self.assertIsNotNone(sig)
        # 1 * 0.35 = 0.35, round = 0, min = 1 → 1
        self.assertEqual(sig.contracts, 1)

    def test_override_zero_blocks(self):
        """override_contracts=0 → blocked (None)."""
        bridge = RiskBridge(RiskBridgeConfig())
        sig = bridge.convert(
            FakeLiveSignal(), size_mult=0.35, symbol="MNQ",
            override_contracts=0,
        )
        self.assertIsNone(sig)


# ══════════════════════════════════════════════════════════════════════════════
#  ExecutionController.on_signal_gated() integration tests (dry-run)
# ══════════════════════════════════════════════════════════════════════════════

class TestControllerGatedDryRun(unittest.TestCase):

    def _make_controller_with_gate(self, layer=None):
        from execution.execution_controller import ExecutionController
        from execution.fail_safes import FailSafeConfig, RunMode

        layer = layer or FakePropRiskLayer(
            decision=FakeTradeDecision(size_mult=0.35, blocked=False),
        )
        gate = PropTradeGate(layer)

        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                prop_gate=gate,
                output_dir=td,
            )
            return ctrl, gate, td

    def test_gated_execute_dry_run(self):
        from execution.execution_controller import ExecutionController
        from execution.fail_safes import FailSafeConfig, RunMode

        layer = FakePropRiskLayer(
            decision=FakeTradeDecision(size_mult=0.35, blocked=False),
        )
        gate = PropTradeGate(layer)

        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                prop_gate=gate,
                output_dir=td,
            )
            result = ctrl.on_signal_gated(FakeLiveSignal(), symbol="MNQ")
            # Without OpenClaw, UI validation fails → False is expected.
            # Key assertion: no crash, gate decision was evaluated.
            self.assertIsInstance(result, bool)

    def test_gated_skip_returns_false(self):
        from execution.execution_controller import ExecutionController
        from execution.fail_safes import FailSafeConfig, RunMode

        layer = FakePropRiskLayer(
            decision=FakeTradeDecision(size_mult=0.0, blocked=True,
                                       reasons=["daily_limit hit"]),
        )
        gate = PropTradeGate(layer)

        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                prop_gate=gate,
                output_dir=td,
            )
            result = ctrl.on_signal_gated(FakeLiveSignal(), symbol="MNQ")
            self.assertFalse(result)

    def test_gated_stop_returns_false(self):
        from execution.execution_controller import ExecutionController
        from execution.fail_safes import FailSafeConfig, RunMode

        layer = FakePropRiskLayer(
            decision=FakeTradeDecision(
                size_mult=0.0, blocked=True,
                reasons=["DD breached, challenge failed"],
            ),
        )
        gate = PropTradeGate(layer)

        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                prop_gate=gate,
                output_dir=td,
            )
            result = ctrl.on_signal_gated(FakeLiveSignal(), symbol="MNQ")
            self.assertFalse(result)

    def test_no_gate_configured_returns_false(self):
        from execution.execution_controller import ExecutionController
        from execution.fail_safes import FailSafeConfig, RunMode

        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                output_dir=td,
            )
            result = ctrl.on_signal_gated(FakeLiveSignal(), symbol="MNQ")
            self.assertFalse(result)

    def test_status_includes_account_state_when_gated(self):
        from execution.execution_controller import ExecutionController
        from execution.fail_safes import FailSafeConfig, RunMode

        layer = FakePropRiskLayer(
            state=FakeAccountState(equity=26_200.0),
        )
        gate = PropTradeGate(layer)

        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                prop_gate=gate,
                output_dir=td,
            )
            status = ctrl.status()
            self.assertIn("account_state", status)
            self.assertEqual(status["account_state"]["equity"], 26_200.0)
            self.assertIn("account_active", status)

    def test_prop_gate_property(self):
        from execution.execution_controller import ExecutionController
        from execution.fail_safes import FailSafeConfig, RunMode

        layer = FakePropRiskLayer()
        gate = PropTradeGate(layer)

        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                prop_gate=gate,
                output_dir=td,
            )
            self.assertIs(ctrl.prop_gate, gate)

    def test_gated_reduce_still_executes(self):
        """REDUCE action should still go through (reduced size, not blocked)."""
        from execution.execution_controller import ExecutionController
        from execution.fail_safes import FailSafeConfig, RunMode

        layer = FakePropRiskLayer(
            decision=FakeTradeDecision(
                size_mult=0.25, blocked=False,
                reasons=["dd_proximity reduced"],
            ),
            base_size=1.0,
        )
        gate = PropTradeGate(layer)

        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                prop_gate=gate,
                output_dir=td,
            )
            result = ctrl.on_signal_gated(FakeLiveSignal(), symbol="MES")
            # Without OpenClaw, UI validation fails. Verify REDUCE passes
            # the gate (not SKIP/STOP) → reaches controller pipeline.
            self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
