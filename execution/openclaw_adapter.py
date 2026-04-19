"""
OpenClaw execution adapter — wraps the existing OpenClawDriver behind the
ExecutionAdapter interface.

This is the current production adapter for Tradovate UI automation.
All OpenClaw-specific logic stays here; the controller only sees the
adapter interface.
"""

from __future__ import annotations

import logging
from typing import Optional

from execution.adapter import ExecutionAdapter, FillResult, OrderResult
from execution.openclaw_driver import OpenClawDriver
from execution.signal_schema import ExecutionSignal
from execution.validators import PostTradeState, PostTradeValidator, UIState

logger = logging.getLogger(__name__)


class OpenClawAdapter(ExecutionAdapter):
    """
    ExecutionAdapter backed by OpenClaw + Tradovate UI.

    Delegates to the existing OpenClawDriver for all UI interaction.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self._driver = OpenClawDriver(dry_run=dry_run)
        self._post_validator = PostTradeValidator()

    # ── Interface implementation ─────────────────────────────────────────

    def validate_context(self, signal: ExecutionSignal) -> UIState:
        return self._driver.read_ui_state()

    def place_order(self, signal: ExecutionSignal) -> OrderResult:
        # Step 1: set quantity
        qty_result = self._driver.set_quantity(signal.contracts)
        if not qty_result.success:
            return OrderResult(
                success=False,
                action="qty_set",
                detail=f"failed to set quantity: {qty_result.detail}",
                elapsed_ms=qty_result.elapsed_ms,
            )

        # Step 2: click buy or sell
        if signal.side.value == "BUY":
            click_result = self._driver.click_buy()
        else:
            click_result = self._driver.click_sell()

        return OrderResult(
            success=click_result.success,
            action=click_result.action,
            detail=click_result.detail,
            elapsed_ms=qty_result.elapsed_ms + click_result.elapsed_ms,
        )

    def confirm_fill(
        self,
        signal: ExecutionSignal,
        timeout_ms: int = 5000,
    ) -> FillResult:
        post_state = self._driver.read_post_trade_state(timeout_ms=timeout_ms)
        validation = self._post_validator.validate(signal, post_state)
        return FillResult(
            confirmed=validation.passed,
            post_state=post_state,
            validation=validation,
            elapsed_ms=post_state.time_elapsed_ms,
        )

    def get_position_state(self, symbol: str = "") -> UIState:
        return self._driver.read_ui_state()

    def emergency_stop(self) -> OrderResult:
        # OpenClaw has no flatten-all; this is a best-effort stub.
        # In production, you would close the position via UI clicks.
        logger.warning("emergency_stop called — OpenClaw adapter has no flatten-all")
        return OrderResult(
            success=False,
            action="emergency_stop",
            detail="not implemented for OpenClaw UI adapter",
        )

    def take_screenshot(self, label: str) -> bool:
        return self._driver.take_screenshot(label)

    @property
    def dry_run(self) -> bool:
        return self._driver.dry_run

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        self._driver.dry_run = value

    @property
    def name(self) -> str:
        return "openclaw"

    # ── Expose raw driver for backward compat / harness ──────────────────

    @property
    def driver(self) -> OpenClawDriver:
        """Access the underlying OpenClawDriver (for legacy code paths)."""
        return self._driver
