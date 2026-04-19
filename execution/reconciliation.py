"""
Continuous position reconciliation loop.

Reads Tradovate UI position state at a configurable interval and compares
against the internal PositionMonitor / FailSafeState.  On any mismatch the
kill switch is activated immediately.

Usage:
    loop = ReconciliationLoop(
        driver=driver,
        fail_safe=fail_safe,
        position_monitor=position_monitor,
        on_alert=my_callback,
    )
    loop.start()
    ...
    loop.stop()

Design:
    - Daemon thread — dies automatically when main thread exits.
    - Does NOT touch execution logic.  Read-only monitoring.
    - Thread-safe: all shared state is accessed under a lock.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from execution.fail_safes import FailSafeState
from execution.validators import PositionMonitor, UIState

logger = logging.getLogger(__name__)


# ── Mismatch types ───────────────────────────────────────────────────────────

class MismatchType:
    MISSING_POSITION = "missing_position"
    UNEXPECTED_POSITION = "unexpected_position"
    WRONG_SIZE = "wrong_size"
    WRONG_SIDE = "wrong_side"
    UI_READ_FAILED = "ui_read_failed"


@dataclass
class ReconciliationResult:
    """Outcome of a single reconciliation check."""

    timestamp: str
    matched: bool
    mismatches: list[str] = field(default_factory=list)
    expected: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.matched:
            return "positions reconciled — no mismatches"
        return f"MISMATCH: {'; '.join(self.mismatches)}"


# ── Core reconciliation logic (pure function, easy to test) ──────────────────


def reconcile_positions(
    expected: dict[str, dict],
    ui_state: UIState,
) -> ReconciliationResult:
    """
    Compare expected positions (from PositionMonitor) against live UI state.

    Parameters
    ----------
    expected : dict
        ``{symbol: {"side": "long"|"short", "size": int}}`` from
        ``PositionMonitor.open_positions``.
    ui_state : UIState
        Fresh snapshot from ``OpenClawDriver.read_ui_state()``.

    Returns
    -------
    ReconciliationResult
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    mismatches: list[str] = []

    # Build observed state from UI
    observed: dict[str, dict] = {}
    if ui_state.has_open_position and ui_state.open_position_size > 0:
        symbol = ui_state.active_symbol or "UNKNOWN"
        observed[symbol] = {
            "side": ui_state.open_position_side,
            "size": ui_state.open_position_size,
        }

    # Detect missing positions (expected but not observed)
    for sym, exp in expected.items():
        if sym not in observed:
            mismatches.append(
                f"{MismatchType.MISSING_POSITION}: expected {exp['side']} "
                f"{exp['size']}x {sym} but UI shows no position"
            )
        else:
            obs = observed[sym]
            if obs["side"] != exp["side"]:
                mismatches.append(
                    f"{MismatchType.WRONG_SIDE}: {sym} expected {exp['side']}, "
                    f"observed {obs['side']}"
                )
            if obs["size"] != exp["size"]:
                mismatches.append(
                    f"{MismatchType.WRONG_SIZE}: {sym} expected {exp['size']}x, "
                    f"observed {obs['size']}x"
                )

    # Detect unexpected positions (observed but not expected)
    for sym in observed:
        if sym not in expected:
            obs = observed[sym]
            mismatches.append(
                f"{MismatchType.UNEXPECTED_POSITION}: UI shows {obs['side']} "
                f"{obs['size']}x {sym} but no position expected"
            )

    return ReconciliationResult(
        timestamp=now,
        matched=len(mismatches) == 0,
        mismatches=mismatches,
        expected=expected,
        observed=observed,
    )


# ── Background loop ─────────────────────────────────────────────────────────


class ReconciliationLoop:
    """
    Daemon thread that polls Tradovate position state and reconciles
    against the internal PositionMonitor.

    On mismatch → activates kill switch and fires on_alert callback.
    """

    def __init__(
        self,
        driver,
        fail_safe: FailSafeState,
        position_monitor: PositionMonitor,
        on_alert: Optional[Callable[[str, str], None]] = None,
        interval_seconds: float = 3.0,
        max_consecutive_failures: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        driver
            Object with a ``read_ui_state() -> UIState`` method
            (typically ``OpenClawDriver``).
        fail_safe
            Shared FailSafeState — kill switch is activated on mismatch.
        position_monitor
            Shared PositionMonitor — source of expected positions.
        on_alert
            ``(level: str, message: str) -> None`` callback.
        interval_seconds
            Polling interval in seconds (2-5 recommended).
        max_consecutive_failures
            If the UI read fails this many times in a row, treat it
            as a mismatch and activate the kill switch.
        """
        self._driver = driver
        self._fail_safe = fail_safe
        self._position_monitor = position_monitor
        self._on_alert = on_alert
        self._interval = interval_seconds
        self._max_failures = max_consecutive_failures

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures: int = 0
        self._check_count: int = 0
        self._mismatch_count: int = 0
        self._last_result: Optional[ReconciliationResult] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background reconciliation thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ReconciliationLoop already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="reconciliation-loop",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "ReconciliationLoop started (interval=%.1fs, max_failures=%d)",
            self._interval,
            self._max_failures,
        )

    def stop(self) -> None:
        """Signal the loop to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            if self._thread.is_alive():
                logger.warning("ReconciliationLoop thread did not exit cleanly")
            self._thread = None
        logger.info("ReconciliationLoop stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Status snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Thread-safe status snapshot for dashboard / logging."""
        with self._lock:
            return {
                "running": self.running,
                "check_count": self._check_count,
                "mismatch_count": self._mismatch_count,
                "consecutive_failures": self._consecutive_failures,
                "interval_seconds": self._interval,
                "last_result": (
                    {
                        "timestamp": self._last_result.timestamp,
                        "matched": self._last_result.matched,
                        "mismatches": self._last_result.mismatches,
                        "expected": self._last_result.expected,
                        "observed": self._last_result.observed,
                    }
                    if self._last_result
                    else None
                ),
            }

    # ── Internal loop ────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main loop body — runs in the daemon thread."""
        logger.info("Reconciliation loop thread started")
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Unhandled error in reconciliation tick")
                with self._lock:
                    self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    self._handle_persistent_failure(
                        "unhandled exception in reconciliation loop"
                    )

            self._stop_event.wait(timeout=self._interval)

    def _tick(self) -> None:
        """Single reconciliation check."""
        # 1. Read UI state
        ui_state = self._driver.read_ui_state()

        # 2. Check for read errors
        if ui_state.read_errors or not ui_state.window_found:
            with self._lock:
                self._consecutive_failures += 1
                self._check_count += 1

            errors = ui_state.read_errors or ["window not found"]
            logger.warning(
                "Reconciliation UI read failed (%d/%d): %s",
                self._consecutive_failures,
                self._max_failures,
                errors,
            )

            if self._consecutive_failures >= self._max_failures:
                self._handle_persistent_failure(
                    f"UI read failed {self._consecutive_failures} consecutive times: "
                    f"{errors}"
                )
            return

        # 3. Read expected from position monitor
        expected = self._position_monitor.open_positions

        # 4. Reconcile
        result = reconcile_positions(expected, ui_state)

        # 5. Store result
        with self._lock:
            self._check_count += 1
            self._last_result = result
            if result.matched:
                self._consecutive_failures = 0
            else:
                self._mismatch_count += 1

        # 6. Act on mismatch
        if not result.matched:
            summary = result.summary()
            logger.critical("RECONCILIATION MISMATCH: %s", summary)
            self._fail_safe.activate_kill_switch(
                f"reconciliation mismatch: {summary}"
            )
            if self._on_alert:
                self._on_alert("CRITICAL", f"RECONCILIATION MISMATCH — {summary}")
        else:
            logger.debug(
                "Reconciliation OK (check #%d): expected=%s, observed=%s",
                self._check_count,
                result.expected,
                result.observed,
            )

    def _handle_persistent_failure(self, reason: str) -> None:
        """Kill switch when UI reads fail repeatedly."""
        logger.critical("Persistent reconciliation failure: %s", reason)
        self._fail_safe.activate_kill_switch(f"reconciliation failure: {reason}")
        if self._on_alert:
            self._on_alert(
                "CRITICAL",
                f"RECONCILIATION FAILURE — {reason}",
            )
