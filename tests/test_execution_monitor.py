"""
Tests for execution/monitor.py — ExecutionMonitorState.

Covers:
- Event ingestion and classification
- Alert generation from different event types
- Snapshot structure and content
- Counter tracking (signals, attempts, confirms, skips, blocks)
- Attachment state
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass

from execution.monitor import (
    AlertLevel,
    ExecutionMonitorState,
    MonitorAlert,
)

try:
    import fastapi
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ── Fake AuditEvent ──────────────────────────────────────────────────────────

@dataclass
class FakeAuditEvent:
    timestamp: str = "2026-04-18T10:00:00Z"
    event_type: str = "signal_received"
    signal_id: str = "sig-001"
    symbol: str = "MNQ"
    side: str = "BUY"
    contracts: int = 1
    mode: str = "challenge"
    reason: str = "test"
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side,
            "contracts": self.contracts,
            "mode": self.mode,
            "reason": self.reason,
            "details": self.details,
        }


# ── Fake AuditLogger ─────────────────────────────────────────────────────────

class FakeAuditLogger:
    def __init__(self):
        self._events = []

    @property
    def events(self):
        return list(self._events)

    @property
    def event_count(self):
        return len(self._events)

    def add(self, event: FakeAuditEvent):
        self._events.append(event)

    def summary(self):
        counts = {}
        for e in self._events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts


# ── Fake FailSafe ─────────────────────────────────────────────────────────────

class FakeFailSafe:
    kill_switch_active = False

    def status(self):
        return {
            "run_mode": "dry_run",
            "kill_switch": self.kill_switch_active,
            "kill_reason": "",
            "emergency_disable": False,
            "open_trades": 0,
            "max_open_trades": 3,
            "session_executions": 0,
            "max_session_executions": 50,
            "cooldown_seconds": 10,
            "last_execution": None,
        }

    def reset_kill_switch(self):
        self.kill_switch_active = False


# ── Fake Controller ──────────────────────────────────────────────────────────

class FakeController:
    def __init__(self):
        self.audit_logger = FakeAuditLogger()
        self.fail_safe = FakeFailSafe()
        self._prop_gate = None

    # Alias so existing test helpers still work via ctrl.audit
    @property
    def audit(self):
        return self.audit_logger

    def status(self):
        return {
            "fail_safe": self.fail_safe.status(),
            "registry_seen": 0,
            "positions": {},
            "audit_events": self.audit_logger.event_count,
            "audit_summary": self.audit_logger.summary(),
        }

    def activate_kill_switch(self, reason):
        self.fail_safe.kill_switch_active = True


# ══════════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionMonitorState(unittest.TestCase):

    def test_initial_state(self):
        m = ExecutionMonitorState()
        self.assertFalse(m.is_attached)
        snap = m.snapshot()
        self.assertFalse(snap["attached"])
        self.assertIsNone(snap["controller"])
        self.assertEqual(snap["counters"]["signals_seen"], 0)

    def test_attach(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)
        self.assertTrue(m.is_attached)
        snap = m.snapshot()
        self.assertTrue(snap["attached"])
        self.assertIsNotNone(snap["controller"])

    def test_ingest_signal_received(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(
            event_type="signal_received",
            signal_id="sig-001",
            symbol="MNQ",
        ))
        snap = m.snapshot()
        self.assertEqual(snap["counters"]["signals_seen"], 1)
        self.assertIsNotNone(snap["last_signal"])
        self.assertEqual(snap["last_signal"]["signal_id"], "sig-001")

    def test_ingest_gate_decision(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(
            event_type="signal_received",
            details={
                "gate_decision": {
                    "action": "reduce",
                    "contracts": 1,
                    "size_mult": 0.35,
                    "profile_label": "challenge_mnq",
                    "profile_mode": "challenge",
                    "profile_symbol": "MNQ",
                    "reasons": ["dd proximity"],
                    "components": {"base": 0.35},
                    "account_snapshot": {},
                }
            }
        ))
        snap = m.snapshot()
        self.assertIsNotNone(snap["last_gate_decision"])
        self.assertEqual(snap["last_gate_decision"]["action"], "reduce")

    def test_ingest_execution_confirmed(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(event_type="execution_confirmed", signal_id="sig-002"))
        snap = m.snapshot()
        self.assertEqual(snap["counters"]["executions_confirmed"], 1)
        self.assertIsNotNone(snap["last_execution_result"])
        self.assertEqual(snap["last_execution_result"]["status"], "confirmed")

    def test_ingest_dry_run(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(event_type="dry_run", signal_id="sig-003"))
        snap = m.snapshot()
        self.assertEqual(snap["counters"]["executions_confirmed"], 1)
        self.assertEqual(snap["last_execution_result"]["status"], "dry_run")

    def test_skip_counter(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(event_type="signal_expired"))
        ctrl.audit.add(FakeAuditEvent(event_type="signal_duplicate"))
        snap = m.snapshot()
        self.assertEqual(snap["counters"]["trades_skipped"], 2)

    def test_block_counter(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(event_type="validation_failed", reason="window not found"))
        ctrl.audit.add(FakeAuditEvent(event_type="kill_switch_activated", reason="operator kill"))
        snap = m.snapshot()
        self.assertEqual(snap["counters"]["trades_blocked"], 2)

    def test_alerts_generated(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(event_type="validation_failed", reason="wrong tab"))
        ctrl.audit.add(FakeAuditEvent(event_type="kill_switch_activated", reason="emergency"))
        snap = m.snapshot()
        alerts = snap["alerts"]
        self.assertEqual(len(alerts), 2)
        # Newest first
        self.assertEqual(alerts[0]["event_type"], "kill_switch_activated")
        self.assertEqual(alerts[0]["level"], "critical")
        self.assertEqual(alerts[1]["event_type"], "validation_failed")
        self.assertEqual(alerts[1]["level"], "warning")

    def test_alert_count(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(event_type="execution_rejected"))
        m.snapshot()  # trigger ingest
        self.assertEqual(m.alert_count, 1)
        self.assertEqual(m.critical_count, 1)

    def test_error_tracking(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(
            event_type="post_trade_mismatch",
            reason="wrong side filled",
            signal_id="sig-err",
        ))
        snap = m.snapshot()
        self.assertIsNotNone(snap["last_error"])
        self.assertEqual(snap["last_error"]["event_type"], "post_trade_mismatch")
        self.assertEqual(snap["last_error"]["signal_id"], "sig-err")

    def test_alerts_by_level(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(event_type="signal_expired"))          # warning
        ctrl.audit.add(FakeAuditEvent(event_type="kill_switch_activated"))    # critical
        ctrl.audit.add(FakeAuditEvent(event_type="cooldown_active"))         # info
        m.snapshot()  # trigger ingest
        crits = m.alerts_by_level("critical")
        self.assertEqual(len(crits), 1)
        warns = m.alerts_by_level("warning")
        self.assertEqual(len(warns), 1)
        infos = m.alerts_by_level("info")
        self.assertEqual(len(infos), 1)

    def test_push_signal_event(self):
        m = ExecutionMonitorState()
        m.push_signal_event({
            "event_type": "signal_received",
            "signal_id": "push-001",
            "symbol": "MES",
            "timestamp": "2026-04-18T12:00:00Z",
        })
        # No controller attached — push still works
        snap = m.snapshot()
        self.assertEqual(snap["counters"]["signals_seen"], 1)

    def test_incremental_ingest(self):
        """Second snapshot doesn't re-process old events."""
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        ctrl.audit.add(FakeAuditEvent(event_type="signal_received"))
        m.snapshot()
        self.assertEqual(m._signals_seen, 1)

        # Add one more event
        ctrl.audit.add(FakeAuditEvent(event_type="execution_attempted"))
        m.snapshot()
        self.assertEqual(m._signals_seen, 1)  # not double-counted
        self.assertEqual(m._executions_attempted, 1)

    def test_controller_status_in_snapshot(self):
        m = ExecutionMonitorState()
        ctrl = FakeController()
        m.attach(ctrl)

        snap = m.snapshot()
        self.assertIn("fail_safe", snap["controller"])
        self.assertEqual(snap["controller"]["fail_safe"]["run_mode"], "dry_run")


@unittest.skipUnless(HAS_FASTAPI, "FastAPI not installed")
class TestDashboardAppExecEndpoints(unittest.TestCase):
    """Integration test: verify the FastAPI endpoints respond."""

    def test_exec_endpoint_no_monitor(self):
        from dashboard.app import create_app
        from dashboard.state import DashboardState
        from fastapi.testclient import TestClient

        app = create_app(DashboardState())
        client = TestClient(app)

        resp = client.get("/api/exec/snapshot")
        self.assertEqual(resp.status_code, 503)

    def test_exec_endpoint_with_monitor(self):
        from dashboard.app import create_app
        from dashboard.state import DashboardState
        from fastapi.testclient import TestClient

        monitor = ExecutionMonitorState()
        app = create_app(DashboardState(), exec_monitor=monitor)
        client = TestClient(app)

        resp = client.get("/api/exec/snapshot")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("counters", data)
        self.assertIn("alerts", data)

    def test_exec_html_page(self):
        from dashboard.app import create_app
        from dashboard.state import DashboardState
        from fastapi.testclient import TestClient

        monitor = ExecutionMonitorState()
        app = create_app(DashboardState(), exec_monitor=monitor)
        client = TestClient(app)

        resp = client.get("/exec")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Apex Execution Monitor", resp.text)

    def test_kill_switch_endpoints(self):
        from dashboard.app import create_app
        from dashboard.state import DashboardState
        from fastapi.testclient import TestClient

        monitor = ExecutionMonitorState()
        ctrl = FakeController()
        monitor.attach(ctrl)
        app = create_app(DashboardState(), exec_monitor=monitor)
        client = TestClient(app)

        # Activate kill switch
        resp = client.post("/api/exec/kill", json={"reason": "test kill"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(ctrl.fail_safe.kill_switch_active)

        # Reset kill switch
        resp = client.post("/api/exec/unkill")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(ctrl.fail_safe.kill_switch_active)

    def test_alerts_endpoint_filtered(self):
        from dashboard.app import create_app
        from dashboard.state import DashboardState
        from fastapi.testclient import TestClient

        monitor = ExecutionMonitorState()
        ctrl = FakeController()
        monitor.attach(ctrl)

        # Inject some events
        ctrl.audit.add(FakeAuditEvent(event_type="validation_failed"))
        ctrl.audit.add(FakeAuditEvent(event_type="kill_switch_activated"))
        monitor.ingest_audit_events()

        app = create_app(DashboardState(), exec_monitor=monitor)
        client = TestClient(app)

        resp = client.get("/api/exec/alerts?level=critical")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["alerts"]), 1)
        self.assertEqual(data["alerts"][0]["level"], "critical")


if __name__ == "__main__":
    unittest.main()
