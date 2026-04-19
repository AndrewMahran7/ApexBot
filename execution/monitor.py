"""
Execution Monitor — operator-visible state for the OpenClaw execution layer.

Reads from an ExecutionController and its AuditLogger to provide a
structured snapshot of:
- Latest signals and gate decisions
- Fail-safe / kill-switch state
- Validation pass/fail history
- Execution results and errors
- Alert classification (stale, duplicate, mismatch, etc.)

This module does NOT modify execution behavior.  It only reads.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

MAX_EVENTS = 200
MAX_ALERTS = 100


# ── Alert levels ─────────────────────────────────────────────────────────────

class AlertLevel:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Alert classification rules: event_type -> (level, short_label)
_ALERT_MAP: dict[str, tuple[str, str]] = {
    "signal_expired":       (AlertLevel.WARNING,  "Stale signal"),
    "signal_duplicate":     (AlertLevel.WARNING,  "Duplicate signal"),
    "validation_failed":    (AlertLevel.WARNING,  "Validation failed"),
    "execution_rejected":   (AlertLevel.CRITICAL, "Execution rejected"),
    "execution_timeout":    (AlertLevel.CRITICAL, "Confirmation timeout"),
    "post_trade_mismatch":  (AlertLevel.CRITICAL, "Post-trade mismatch"),
    "fail_safe_triggered":  (AlertLevel.WARNING,  "Fail-safe triggered"),
    "kill_switch_activated": (AlertLevel.CRITICAL, "KILL SWITCH"),
    "cooldown_active":      (AlertLevel.INFO,     "Cooldown active"),
    "error":                (AlertLevel.CRITICAL, "Error"),
}


@dataclass
class MonitorAlert:
    """Classified alert for operator display."""
    timestamp: str
    level: str
    label: str
    event_type: str
    signal_id: str
    symbol: str
    reason: str


class ExecutionMonitorState:
    """
    Thread-safe state collector for the execution monitor panel.

    Attach to an ExecutionController after init:
        monitor = ExecutionMonitorState()
        monitor.attach(controller)

    The FastAPI endpoints read from monitor.snapshot().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._controller = None

        # Ring buffers for classified alerts (derived from audit events)
        self._alerts: deque[MonitorAlert] = deque(maxlen=MAX_ALERTS)

        # Latest signal tracking
        self._last_signal: Optional[dict] = None
        self._last_gate_decision: Optional[dict] = None
        self._last_execution_result: Optional[dict] = None
        self._last_error: Optional[dict] = None

        # Counters
        self._signals_seen = 0
        self._executions_attempted = 0
        self._executions_confirmed = 0
        self._trades_skipped = 0
        self._trades_blocked = 0

    def attach(self, controller) -> None:
        """Attach to an ExecutionController instance."""
        self._controller = controller
        logger.info("ExecutionMonitor attached to controller")

    @property
    def is_attached(self) -> bool:
        return self._controller is not None

    # ── Event ingestion (called by controller hooks or polling) ──────

    def ingest_audit_events(self) -> None:
        """
        Pull new events from the controller's audit logger and classify.

        Call this periodically (e.g. every poll cycle) or after each
        signal processing.
        """
        if self._controller is None:
            return

        audit = self._controller.audit_logger
        events = audit.events  # returns a copy

        with self._lock:
            # Process only events beyond what we've already seen
            start = self._signals_seen_from_audit if hasattr(self, "_signals_seen_from_audit") else 0
            new_events = events[start:]
            self._signals_seen_from_audit = len(events)

            for ev in new_events:
                self._process_event(ev)

    def _process_event(self, event) -> None:
        """Classify a single AuditEvent and update state."""
        ev_dict = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        et = ev_dict.get("event_type", "")

        # Track latest signal
        if et == "signal_received":
            self._last_signal = ev_dict
            self._signals_seen += 1
            # Check for gate decision in details
            details = ev_dict.get("details", {})
            if isinstance(details, dict) and "gate_decision" in details:
                self._last_gate_decision = details["gate_decision"]

        # Track execution attempts
        if et == "execution_attempted":
            self._executions_attempted += 1

        # Track confirmed executions
        if et == "execution_confirmed":
            self._executions_confirmed += 1
            self._last_execution_result = {
                "status": "confirmed",
                "timestamp": ev_dict.get("timestamp", ""),
                "signal_id": ev_dict.get("signal_id", ""),
                "symbol": ev_dict.get("symbol", ""),
                "side": ev_dict.get("side", ""),
                "contracts": ev_dict.get("contracts", 0),
                "reason": ev_dict.get("reason", ""),
            }

        # Track dry-runs as "executions" too
        if et == "dry_run":
            self._executions_confirmed += 1
            self._last_execution_result = {
                "status": "dry_run",
                "timestamp": ev_dict.get("timestamp", ""),
                "signal_id": ev_dict.get("signal_id", ""),
                "symbol": ev_dict.get("symbol", ""),
                "side": ev_dict.get("side", ""),
                "contracts": ev_dict.get("contracts", 0),
                "reason": ev_dict.get("reason", ""),
            }

        # Track skips
        if et in ("signal_expired", "signal_duplicate", "cooldown_active"):
            self._trades_skipped += 1

        # Track blocks
        if et in ("validation_failed", "fail_safe_triggered",
                   "kill_switch_activated", "execution_rejected"):
            self._trades_blocked += 1

        # Track errors
        if et in ("error", "execution_rejected", "execution_timeout",
                   "post_trade_mismatch"):
            self._last_error = {
                "timestamp": ev_dict.get("timestamp", ""),
                "event_type": et,
                "signal_id": ev_dict.get("signal_id", ""),
                "symbol": ev_dict.get("symbol", ""),
                "reason": ev_dict.get("reason", ""),
            }

        # Classify as alert if applicable
        if et in _ALERT_MAP:
            level, label = _ALERT_MAP[et]
            alert = MonitorAlert(
                timestamp=ev_dict.get("timestamp", ""),
                level=level,
                label=label,
                event_type=et,
                signal_id=ev_dict.get("signal_id", ""),
                symbol=ev_dict.get("symbol", ""),
                reason=ev_dict.get("reason", ""),
            )
            self._alerts.append(alert)

    # ── Manual event push (for direct integration) ───────────────────

    def push_signal_event(self, event_dict: dict) -> None:
        """Push a pre-built event dict (from controller callback)."""
        with self._lock:
            # Wrap in a minimal object with to_dict()
            class _Ev:
                def to_dict(self_inner):
                    return event_dict
            self._process_event(_Ev())

    # ── Snapshot for API ─────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Full monitor state snapshot for the /api/exec/snapshot endpoint.

        Returns a JSON-safe dict with all operator-visible fields.
        """
        # Pull latest from controller
        self.ingest_audit_events()

        with self._lock:
            result: dict = {
                "attached": self.is_attached,
                "counters": {
                    "signals_seen": self._signals_seen,
                    "executions_attempted": self._executions_attempted,
                    "executions_confirmed": self._executions_confirmed,
                    "trades_skipped": self._trades_skipped,
                    "trades_blocked": self._trades_blocked,
                },
                "last_signal": self._last_signal,
                "last_gate_decision": self._last_gate_decision,
                "last_execution_result": self._last_execution_result,
                "last_error": self._last_error,
                "alerts": [
                    {
                        "timestamp": a.timestamp,
                        "level": a.level,
                        "label": a.label,
                        "event_type": a.event_type,
                        "signal_id": a.signal_id,
                        "symbol": a.symbol,
                        "reason": a.reason,
                    }
                    for a in reversed(self._alerts)  # newest first
                ],
            }

        # Merge live controller status if attached
        if self._controller is not None:
            try:
                ctrl_status = self._controller.status()
                result["controller"] = ctrl_status
            except Exception as e:
                logger.warning("Failed to read controller status: %s", e)
                result["controller"] = {"error": str(e)}
        else:
            result["controller"] = None

        return result

    def alerts_by_level(self, level: str) -> list[dict]:
        """Return alerts filtered by level."""
        with self._lock:
            return [
                {
                    "timestamp": a.timestamp,
                    "level": a.level,
                    "label": a.label,
                    "event_type": a.event_type,
                    "signal_id": a.signal_id,
                    "symbol": a.symbol,
                    "reason": a.reason,
                }
                for a in reversed(self._alerts)
                if a.level == level
            ]

    @property
    def alert_count(self) -> int:
        with self._lock:
            return len(self._alerts)

    @property
    def critical_count(self) -> int:
        with self._lock:
            return sum(1 for a in self._alerts if a.level == AlertLevel.CRITICAL)
