"""
Execution adapter interface — abstraction layer between the controller
and any broker/execution backend.

Current implementation:  OpenClawAdapter  (Tradovate UI via OpenClaw)
Future implementation:   BrokerAPIAdapter (direct API, not built yet)

The controller depends ONLY on this interface.  Strategy and risk layers
never touch the adapter.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Optional

from execution.signal_schema import ExecutionSignal
from execution.validators import PostTradeState, UIState, ValidationResult

logger = logging.getLogger(__name__)


# ── Adapter result types ─────────────────────────────────────────────────────

@dataclass
class OrderResult:
    """
    Outcome of a place_order() call.

    Replaces the ClickResult that was specific to OpenClaw.
    Both UI-click and API adapters return this same type.
    """
    success: bool
    action: str = ""          # e.g. "buy_click", "api_market_buy"
    detail: str = ""          # human-readable explanation
    elapsed_ms: int = 0
    order_id: str = ""        # broker-assigned order ID (empty for UI clicks)
    metadata: dict = field(default_factory=dict)


@dataclass
class FillResult:
    """
    Outcome of a confirm_fill() call.

    Wraps whatever the adapter returns when checking post-trade state.
    """
    confirmed: bool
    post_state: PostTradeState       # always populated (may have .position_detected=False)
    validation: ValidationResult     # post-trade validation against expected signal
    elapsed_ms: int = 0


# ── Abstract adapter interface ───────────────────────────────────────────────

class ExecutionAdapter(abc.ABC):
    """
    Abstract execution adapter.

    Any broker backend must implement these five operations.
    The controller calls them in this order:

        1. validate_context(signal)  — can we trade this right now?
        2. place_order(signal)       — submit the order
        3. confirm_fill(signal)      — verify execution
        4. get_position_state()      — current positions
        5. emergency_stop()          — kill everything

    Plus a screenshot helper and a dry_run flag.
    """

    @abc.abstractmethod
    def validate_context(
        self,
        signal: ExecutionSignal,
    ) -> UIState:
        """
        Read the current execution context and return a UIState snapshot.

        For UI adapters: reads the broker window.
        For API adapters: queries account state, symbol availability, etc.

        The returned UIState is fed into PreTradeValidator.
        """
        ...

    @abc.abstractmethod
    def place_order(
        self,
        signal: ExecutionSignal,
    ) -> OrderResult:
        """
        Submit an order for the given signal.

        For UI adapters: set quantity + click buy/sell.
        For API adapters: POST market order to broker endpoint.

        Returns OrderResult with success/failure and metadata.
        """
        ...

    @abc.abstractmethod
    def confirm_fill(
        self,
        signal: ExecutionSignal,
        timeout_ms: int = 5000,
    ) -> FillResult:
        """
        Confirm that the order was filled.

        For UI adapters: poll Tradovate position panel.
        For API adapters: poll order status endpoint.

        Returns FillResult with post-trade state and validation.
        """
        ...

    @abc.abstractmethod
    def get_position_state(
        self,
        symbol: str = "",
    ) -> UIState:
        """
        Get current position state (for monitoring / preflight checks).

        Returns a UIState snapshot.  For API adapters, synthesize the
        relevant fields from account/position queries.
        """
        ...

    @abc.abstractmethod
    def emergency_stop(self) -> OrderResult:
        """
        Emergency: flatten all positions and cancel pending orders.

        Returns OrderResult indicating whether the stop was successful.
        """
        ...

    @abc.abstractmethod
    def take_screenshot(self, label: str) -> bool:
        """
        Capture a screenshot or state dump for audit purposes.

        UI adapters: capture window screenshot.
        API adapters: dump current order book / account state to file.

        Returns True if saved successfully.
        """
        ...

    @property
    @abc.abstractmethod
    def dry_run(self) -> bool:
        """Whether this adapter is in dry-run mode (no real orders)."""
        ...

    @dry_run.setter
    @abc.abstractmethod
    def dry_run(self, value: bool) -> None:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable adapter name, e.g. 'openclaw', 'broker_api'."""
        ...
