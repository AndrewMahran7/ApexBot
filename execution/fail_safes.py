"""
Fail-safe controls for the OpenClaw execution layer.

Provides:
- Kill switch (manual emergency stop)
- Cooldown timer between executions
- One-open-trade-max enforcement
- Duplicate signal suppression (via SignalRegistry)
- Dry-run / sim / live mode toggle
- Emergency disable flag
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RunMode(str, Enum):
    DRY_RUN = "dry_run"       # All validation, no clicks
    SIM = "sim"               # Clicks on sim/paper account only
    LIVE = "live"             # Clicks on eval/funded account


@dataclass
class FailSafeConfig:
    """All fail-safe parameters in one place."""

    # Run mode
    run_mode: RunMode = RunMode.DRY_RUN

    # Trade limits
    max_open_trades: int = 1
    cooldown_seconds: int = 60  # Minimum gap between executions

    # Kill switch
    kill_switch_enabled: bool = False  # Start in safe state

    # Emergency disable — if True, nothing executes at all
    emergency_disable: bool = False

    # Duplicate suppression cooldown (for fingerprint-based dedup)
    duplicate_cooldown_seconds: int = 120

    # Max executions per session (safety cap)
    max_executions_per_session: int = 20

    # Post-trade confirmation timeout
    confirmation_timeout_seconds: int = 10

    # Screenshot on failure
    screenshot_on_failure: bool = True


class FailSafeState:
    """
    Runtime state for fail-safe enforcement.

    All checks are explicit — nothing is hidden.
    """

    def __init__(self, config: FailSafeConfig) -> None:
        self.config = config
        self._open_trade_count: int = 0
        self._last_execution_time: Optional[datetime.datetime] = None
        self._session_execution_count: int = 0
        self._kill_switch_active: bool = config.kill_switch_enabled
        self._kill_reason: str = ""
        logger.info(
            "FailSafeState initialized: mode=%s, max_open=%d, cooldown=%ds, kill=%s",
            config.run_mode.value,
            config.max_open_trades,
            config.cooldown_seconds,
            self._kill_switch_active,
        )

    # ── Queries ──────────────────────────────────────────────────────────────

    def can_execute(self, now: Optional[datetime.datetime] = None) -> tuple[bool, str]:
        """
        Check ALL fail-safe conditions.

        Returns (allowed, reason).
        If not allowed, reason explains which check failed.
        """
        now = now or datetime.datetime.now(datetime.timezone.utc)

        if self.config.emergency_disable:
            return False, "emergency_disable flag is active"

        if self._kill_switch_active:
            return False, f"kill switch active: {self._kill_reason}"

        if self.config.run_mode == RunMode.DRY_RUN:
            return False, "dry_run mode — would execute but clicks are disabled"

        if self._open_trade_count >= self.config.max_open_trades:
            return False, f"max open trades reached ({self._open_trade_count}/{self.config.max_open_trades})"

        if self._session_execution_count >= self.config.max_executions_per_session:
            return False, f"session execution cap reached ({self._session_execution_count}/{self.config.max_executions_per_session})"

        if self._last_execution_time is not None:
            elapsed = (now - self._last_execution_time).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                remaining = self.config.cooldown_seconds - elapsed
                return False, f"cooldown active ({remaining:.0f}s remaining)"

        return True, "all fail-safes passed"

    @property
    def is_dry_run(self) -> bool:
        return self.config.run_mode == RunMode.DRY_RUN

    @property
    def is_sim(self) -> bool:
        return self.config.run_mode == RunMode.SIM

    @property
    def is_live(self) -> bool:
        return self.config.run_mode == RunMode.LIVE

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def open_trade_count(self) -> int:
        return self._open_trade_count

    @property
    def session_execution_count(self) -> int:
        return self._session_execution_count

    @property
    def last_execution_time(self) -> Optional[datetime.datetime]:
        return self._last_execution_time

    # ── Mutations ────────────────────────────────────────────────────────────

    def record_execution(self, now: Optional[datetime.datetime] = None) -> None:
        """Record that an execution click was performed."""
        now = now or datetime.datetime.now(datetime.timezone.utc)
        self._last_execution_time = now
        self._session_execution_count += 1
        self._open_trade_count += 1
        logger.info(
            "Execution recorded: session_count=%d, open_trades=%d",
            self._session_execution_count,
            self._open_trade_count,
        )

    def record_trade_closed(self) -> None:
        """Record that a trade was closed (reduces open count)."""
        if self._open_trade_count > 0:
            self._open_trade_count -= 1
        logger.info("Trade closed. open_trades=%d", self._open_trade_count)

    def set_open_trade_count(self, count: int) -> None:
        """Update open trade count from external position check."""
        self._open_trade_count = count

    def activate_kill_switch(self, reason: str) -> None:
        """Emergency stop — blocks all future executions until reset."""
        self._kill_switch_active = True
        self._kill_reason = reason
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def reset_kill_switch(self) -> None:
        """Manually re-enable trading after operator review."""
        self._kill_switch_active = False
        self._kill_reason = ""
        logger.warning("Kill switch reset by operator")

    def set_emergency_disable(self, enabled: bool) -> None:
        """Toggle the emergency disable flag."""
        self.config.emergency_disable = enabled
        logger.warning("Emergency disable set to %s", enabled)

    def set_run_mode(self, mode: RunMode) -> None:
        """Change the run mode (dry_run → sim → live)."""
        old = self.config.run_mode
        self.config.run_mode = mode
        logger.warning("Run mode changed: %s → %s", old.value, mode.value)

    def reset_session(self) -> None:
        """Reset session counters (start of new trading day)."""
        self._session_execution_count = 0
        self._last_execution_time = None
        logger.info("Session counters reset")

    # ── Inspection ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Full status snapshot for logging/display."""
        return {
            "run_mode": self.config.run_mode.value,
            "kill_switch": self._kill_switch_active,
            "kill_reason": self._kill_reason,
            "emergency_disable": self.config.emergency_disable,
            "open_trades": self._open_trade_count,
            "max_open_trades": self.config.max_open_trades,
            "session_executions": self._session_execution_count,
            "max_session_executions": self.config.max_executions_per_session,
            "cooldown_seconds": self.config.cooldown_seconds,
            "last_execution": self._last_execution_time.isoformat() if self._last_execution_time else None,
        }
