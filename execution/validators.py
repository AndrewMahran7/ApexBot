"""
Pre-trade and post-trade validation for the OpenClaw execution layer.

Pre-trade checks (before clicking):
- Tradovate window is present
- Correct account selected
- Correct instrument tab active
- Correct quantity set
- No unexpected existing position
- No pending order conflict
- Signal is still fresh
- Symbol / side match

Post-trade checks (after clicking):
- Position was actually opened
- Side matches expected
- Size matches expected
- Duplicate / unexpected position detection
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

from execution.signal_schema import ExecutionSignal

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    checks: list[dict] = field(default_factory=list)
    # Each check: {"name": str, "passed": bool, "detail": str}

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    @property
    def failed_checks(self) -> list[dict]:
        return [c for c in self.checks if not c["passed"]]

    @property
    def summary(self) -> str:
        failed = self.failed_checks
        if not failed:
            return "ALL PASSED"
        names = [c["name"] for c in failed]
        return f"FAILED: {', '.join(names)}"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "checks": self.checks,
        }


@dataclass
class UIState:
    """
    Snapshot of observed Tradovate UI state from OpenClaw.

    Populated by the openclaw_driver before validation is run.
    All fields are Optional because any read might fail.
    """

    window_found: bool = False
    window_title: str = ""
    account_label: Optional[str] = None
    active_symbol: Optional[str] = None
    quantity_value: Optional[int] = None
    atm_template_name: Optional[str] = None
    has_open_position: bool = False
    open_position_side: Optional[str] = None  # "long" / "short" / None
    open_position_size: int = 0
    pending_orders: int = 0
    buy_button_visible: bool = False
    sell_button_visible: bool = False
    read_errors: list[str] = field(default_factory=list)


class PreTradeValidator:
    """
    Validates that the Tradovate UI is in the correct state before execution.
    """

    def __init__(
        self,
        expected_account_fragment: str = "",
        allowed_symbols: Optional[set[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        expected_account_fragment : str
            Substring that must appear in the account label (e.g. "APEX-" or account number).
        allowed_symbols : set of str, optional
            If provided, only these symbols are allowed.
        """
        self._expected_account = expected_account_fragment
        self._allowed_symbols = allowed_symbols or {"MES", "MNQ", "RTY", "NQ", "ES"}

    def validate(
        self,
        signal: ExecutionSignal,
        ui_state: UIState,
        now: Optional[datetime.datetime] = None,
    ) -> ValidationResult:
        """
        Run all pre-trade checks.

        Returns a ValidationResult.  If .passed is False, do NOT execute.
        """
        result = ValidationResult(passed=True)
        now = now or datetime.datetime.now(datetime.timezone.utc)

        # 1. Window present
        result.add(
            "window_present",
            ui_state.window_found,
            ui_state.window_title if ui_state.window_found else "Tradovate window not found",
        )

        # 2. Correct account
        if self._expected_account:
            acct_ok = (
                ui_state.account_label is not None
                and self._expected_account.lower() in ui_state.account_label.lower()
            )
            result.add(
                "correct_account",
                acct_ok,
                f"expected '{self._expected_account}' in '{ui_state.account_label}'",
            )
        else:
            result.add("correct_account", True, "no account filter configured")

        # 3. Correct instrument active
        symbol_ok = (
            ui_state.active_symbol is not None
            and ui_state.active_symbol.upper().startswith(signal.symbol.upper())
        )
        result.add(
            "correct_symbol",
            symbol_ok,
            f"expected={signal.symbol}, found={ui_state.active_symbol}",
        )

        # 4. Symbol in allowed list
        result.add(
            "symbol_allowed",
            signal.symbol.upper() in {s.upper() for s in self._allowed_symbols},
            f"symbol={signal.symbol}, allowed={self._allowed_symbols}",
        )

        # 5. Correct quantity
        qty_ok = ui_state.quantity_value is not None and ui_state.quantity_value == signal.contracts
        result.add(
            "correct_quantity",
            qty_ok,
            f"expected={signal.contracts}, found={ui_state.quantity_value}",
        )

        # 6. No unexpected existing position
        result.add(
            "no_unexpected_position",
            not ui_state.has_open_position,
            f"open_position={ui_state.has_open_position}, side={ui_state.open_position_side}, size={ui_state.open_position_size}",
        )

        # 7. No pending order conflict
        result.add(
            "no_pending_orders",
            ui_state.pending_orders == 0,
            f"pending_orders={ui_state.pending_orders}",
        )

        # 8. Signal still fresh
        expired = signal.is_expired(now)
        result.add(
            "signal_fresh",
            not expired,
            f"expires_at={signal.expires_at.isoformat()}, now={now.isoformat()}",
        )

        # 9. Buy / Sell button visible
        if signal.side.value == "BUY":
            result.add("action_button_visible", ui_state.buy_button_visible, "BUY button visibility")
        else:
            result.add("action_button_visible", ui_state.sell_button_visible, "SELL button visibility")

        # 10. ATM template check (if specified)
        if signal.atm_template:
            atm_ok = (
                ui_state.atm_template_name is not None
                and signal.atm_template.lower() in ui_state.atm_template_name.lower()
            )
            result.add(
                "atm_template_active",
                atm_ok,
                f"expected='{signal.atm_template}', found='{ui_state.atm_template_name}'",
            )

        # 11. No UI read errors
        result.add(
            "no_read_errors",
            len(ui_state.read_errors) == 0,
            "; ".join(ui_state.read_errors) if ui_state.read_errors else "clean",
        )

        logger.info(
            "Pre-trade validation: %s (%d/%d checks passed)",
            result.summary,
            sum(1 for c in result.checks if c["passed"]),
            len(result.checks),
        )
        return result


@dataclass
class PostTradeState:
    """
    Observed state after execution click.

    Populated by the openclaw_driver after the click.
    """

    position_detected: bool = False
    position_side: Optional[str] = None  # "long" / "short"
    position_size: int = 0
    fill_price: Optional[float] = None
    order_status: Optional[str] = None  # "filled" / "working" / "rejected"
    time_elapsed_ms: int = 0
    read_errors: list[str] = field(default_factory=list)


class PostTradeValidator:
    """
    Validates that execution produced the intended result.
    """

    def validate(
        self,
        signal: ExecutionSignal,
        post_state: PostTradeState,
    ) -> ValidationResult:
        """
        Run post-trade checks.

        If validation fails, an alert should be raised.
        """
        result = ValidationResult(passed=True)

        # 1. Position was actually opened
        result.add(
            "position_opened",
            post_state.position_detected,
            f"detected={post_state.position_detected}, status={post_state.order_status}",
        )

        # 2. Side matches
        expected_side = "long" if signal.side.value == "BUY" else "short"
        side_ok = (
            post_state.position_side is not None
            and post_state.position_side.lower() == expected_side
        )
        result.add(
            "side_matches",
            side_ok,
            f"expected={expected_side}, found={post_state.position_side}",
        )

        # 3. Size matches
        size_ok = post_state.position_size == signal.contracts
        result.add(
            "size_matches",
            size_ok,
            f"expected={signal.contracts}, found={post_state.position_size}",
        )

        # 4. No read errors during confirmation
        result.add(
            "no_read_errors",
            len(post_state.read_errors) == 0,
            "; ".join(post_state.read_errors) if post_state.read_errors else "clean",
        )

        logger.info(
            "Post-trade validation: %s (%d/%d checks passed)",
            result.summary,
            sum(1 for c in result.checks if c["passed"]),
            len(result.checks),
        )
        return result


class PositionMonitor:
    """
    Detects duplicate or unexpected positions that may indicate a UI desync.

    Triggers kill-switch behavior when anomalies are detected.
    """

    def __init__(self, expected_max_positions: int = 1) -> None:
        self._expected_max = expected_max_positions
        self._known_positions: dict[str, dict] = {}  # symbol → {side, size}

    def update(self, symbol: str, side: Optional[str], size: int) -> list[str]:
        """
        Update known position state and return a list of anomaly descriptions.

        Returns empty list if everything is normal.
        """
        anomalies: list[str] = []

        if size == 0:
            self._known_positions.pop(symbol, None)
        else:
            self._known_positions[symbol] = {"side": side, "size": size}

        # Check total open positions
        total = len(self._known_positions)
        if total > self._expected_max:
            anomalies.append(
                f"unexpected position count: {total} > {self._expected_max} "
                f"(positions: {self._known_positions})"
            )

        # Check for same-symbol multiple positions (shouldn't happen with ATM)
        if size > 0 and symbol in self._known_positions:
            known = self._known_positions[symbol]
            if known["size"] != size:
                anomalies.append(
                    f"position size changed unexpectedly for {symbol}: "
                    f"known={known['size']}, observed={size}"
                )

        for anomaly in anomalies:
            logger.warning("Position anomaly: %s", anomaly)

        return anomalies

    def clear(self) -> None:
        self._known_positions.clear()

    @property
    def open_positions(self) -> dict[str, dict]:
        return dict(self._known_positions)
