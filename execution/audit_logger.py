"""
Structured audit logging for the OpenClaw execution layer.

Writes:
- JSON-lines log file per day (results/openclaw_execution/YYYY-MM-DD.jsonl)
- Human-readable log via standard logging
- Screenshots on failure (if screenshot_fn is provided)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    SIGNAL_RECEIVED = "signal_received"
    SIGNAL_EXPIRED = "signal_expired"
    SIGNAL_DUPLICATE = "signal_duplicate"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    EXECUTION_ATTEMPTED = "execution_attempted"
    EXECUTION_CONFIRMED = "execution_confirmed"
    EXECUTION_REJECTED = "execution_rejected"
    EXECUTION_TIMEOUT = "execution_timeout"
    POST_TRADE_MISMATCH = "post_trade_mismatch"
    FAIL_SAFE_TRIGGERED = "fail_safe_triggered"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    COOLDOWN_ACTIVE = "cooldown_active"
    DRY_RUN = "dry_run"
    SCREENSHOT_SAVED = "screenshot_saved"
    ERROR = "error"


@dataclass
class AuditEvent:
    """A single auditable event in the execution pipeline."""

    timestamp: str  # ISO format
    event_type: str
    signal_id: str = ""
    symbol: str = ""
    side: str = ""
    contracts: int = 0
    mode: str = ""
    reason: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side,
            "contracts": self.contracts,
            "mode": self.mode,
            "reason": self.reason,
        }
        if self.details:
            d["details"] = self.details
        return d


class AuditLogger:
    """
    Persistent JSON-lines audit logger for the execution layer.

    Parameters
    ----------
    output_dir : str or Path
        Directory for log files.  Created if missing.
    screenshot_fn : callable, optional
        Function(filepath: str) → None that saves a screenshot.
        Called on validation/execution failures.
    """

    def __init__(
        self,
        output_dir: str | Path = "results/openclaw_execution",
        screenshot_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._screenshot_fn = screenshot_fn
        self._events: list[AuditEvent] = []
        logger.info("AuditLogger initialized -> %s", self._output_dir.resolve())

    def _log_file_for_date(self, dt: datetime.datetime) -> Path:
        date_str = dt.strftime("%Y-%m-%d")
        return self._output_dir / f"{date_str}.jsonl"

    def log(
        self,
        event_type: EventType,
        signal_id: str = "",
        symbol: str = "",
        side: str = "",
        contracts: int = 0,
        mode: str = "",
        reason: str = "",
        details: Optional[dict] = None,
    ) -> AuditEvent:
        """Record an audit event.  Written to JSONL and in-memory list."""
        now = datetime.datetime.now(datetime.timezone.utc)
        event = AuditEvent(
            timestamp=now.isoformat(),
            event_type=event_type.value,
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            contracts=contracts,
            mode=mode,
            reason=reason,
            details=details or {},
        )
        self._events.append(event)

        # Write to JSONL
        log_file = self._log_file_for_date(now)
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError as e:
            logger.error("Failed to write audit log to %s: %s", log_file, e)

        # Human-readable log
        logger.info(
            "[%s] signal=%s symbol=%s side=%s qty=%d mode=%s — %s",
            event_type.value,
            signal_id[:8] if signal_id else "N/A",
            symbol,
            side,
            contracts,
            mode,
            reason,
        )
        return event

    def log_from_signal(
        self,
        event_type: EventType,
        signal: Any,
        reason: str = "",
        details: Optional[dict] = None,
    ) -> AuditEvent:
        """Convenience: extract fields from an ExecutionSignal."""
        return self.log(
            event_type=event_type,
            signal_id=getattr(signal, "signal_id", ""),
            symbol=getattr(signal, "symbol", ""),
            side=getattr(signal, "side", "").value if hasattr(getattr(signal, "side", ""), "value") else str(getattr(signal, "side", "")),
            contracts=getattr(signal, "contracts", 0),
            mode=getattr(signal, "mode", "").value if hasattr(getattr(signal, "mode", ""), "value") else str(getattr(signal, "mode", "")),
            reason=reason,
            details=details,
        )

    def save_screenshot(self, label: str) -> Optional[str]:
        """
        Capture a screenshot if a screenshot function is registered.

        Returns the saved filepath or None.
        """
        if self._screenshot_fn is None:
            logger.debug("No screenshot function registered — skipping capture for %s", label)
            return None

        now = datetime.datetime.now(datetime.timezone.utc)
        filename = f"screenshot_{now.strftime('%Y%m%d_%H%M%S')}_{label}.png"
        filepath = str(self._output_dir / filename)
        try:
            self._screenshot_fn(filepath)
            self.log(EventType.SCREENSHOT_SAVED, reason=label, details={"path": filepath})
            return filepath
        except Exception as e:
            logger.error("Screenshot capture failed for %s: %s", label, e)
            return None

    @property
    def events(self) -> list[AuditEvent]:
        """In-memory event list (read-only copy)."""
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def summary(self) -> dict[str, int]:
        """Count events by type."""
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts
