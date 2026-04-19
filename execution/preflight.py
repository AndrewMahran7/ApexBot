"""
Pre-flight validation — blocks system startup unless all conditions are met.

Checks:
1. Tradovate window detected
2. Correct account selected
3. Correct symbol tab active
4. ATM template selected
5. No open positions
6. Buy and Sell buttons visible
7. Kill switch is OFF
8. No read errors from OpenClaw

Usage (standalone):
    python -m execution.preflight --symbol MNQ --account APEX

Usage (integrated):
    from execution.preflight import run_preflight
    result = run_preflight(expected_symbol="MNQ", expected_account="APEX")
    if not result.passed:
        sys.exit(1)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

from execution.openclaw_driver import OpenClawDriver
from execution.validators import UIState

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Outcome of the pre-flight validation."""

    passed: bool = True
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    @property
    def failed_checks(self) -> list[dict]:
        return [c for c in self.checks if not c["passed"]]

    def print_report(self) -> None:
        """Print a human-readable pre-flight report to stdout."""
        print()
        print("=" * 60)
        print("  APEX PRE-FLIGHT VALIDATION")
        print("=" * 60)

        for c in self.checks:
            icon = "PASS" if c["passed"] else "FAIL"
            print(f"  [{icon}]  {c['name']}")
            if c["detail"]:
                print(f"         {c['detail']}")

        print("-" * 60)
        if self.passed:
            print("  RESULT: SYSTEM READY")
        else:
            failed = [c["name"] for c in self.failed_checks]
            print(f"  RESULT: BLOCKED — {len(failed)} check(s) failed")
            print(f"  Failed: {', '.join(failed)}")
        print("=" * 60)
        print()


def run_preflight(
    expected_symbol: str = "",
    expected_account: str = "",
    fail_safe_state=None,
    dry_run: bool = True,
) -> PreflightResult:
    """
    Run all pre-flight checks.

    Parameters
    ----------
    expected_symbol : str
        Symbol that must be active in Tradovate (e.g. "MNQ", "MES").
        Empty string skips the symbol check.
    expected_account : str
        Substring that must appear in the Tradovate account label.
        Empty string skips the account check.
    fail_safe_state : FailSafeState, optional
        If provided, checks the kill switch and emergency disable flags.
    dry_run : bool
        Passed to OpenClawDriver (read-only, no clicks regardless).

    Returns
    -------
    PreflightResult
        .passed is True only if ALL checks succeed.
    """
    result = PreflightResult()

    # ── Check 1: OpenClaw available and Tradovate window found ───────────
    driver = OpenClawDriver(dry_run=True)  # Always read-only for preflight
    ui = driver.read_ui_state()

    result.add(
        "tradovate_window",
        ui.window_found,
        ui.window_title if ui.window_found else "Tradovate window not found — is the app running?",
    )

    if not ui.window_found:
        # Remaining checks are meaningless without a window.
        return result

    # ── Check 2: Correct account ─────────────────────────────────────────
    if expected_account:
        acct_ok = (
            ui.account_label is not None
            and expected_account.lower() in ui.account_label.lower()
        )
        result.add(
            "correct_account",
            acct_ok,
            f"expected '{expected_account}' in '{ui.account_label}'"
            if not acct_ok else f"account: {ui.account_label}",
        )
    else:
        result.add(
            "correct_account",
            True,
            f"no account filter (detected: {ui.account_label})",
        )

    # ── Check 3: Correct symbol tab active ───────────────────────────────
    if expected_symbol:
        symbol_ok = (
            ui.active_symbol is not None
            and expected_symbol.upper() in ui.active_symbol.upper()
        )
        result.add(
            "correct_symbol",
            symbol_ok,
            f"expected '{expected_symbol}', found '{ui.active_symbol}'"
            if not symbol_ok else f"symbol: {ui.active_symbol}",
        )
    else:
        result.add(
            "correct_symbol",
            True,
            f"no symbol filter (detected: {ui.active_symbol})",
        )

    # ── Check 4: ATM template selected ───────────────────────────────────
    atm_ok = ui.atm_template_name is not None and ui.atm_template_name.strip() != ""
    result.add(
        "atm_template",
        atm_ok,
        f"template: {ui.atm_template_name}" if atm_ok
        else "no ATM template selected — stops/targets will NOT be placed",
    )

    # ── Check 5: No open positions ───────────────────────────────────────
    result.add(
        "no_open_positions",
        not ui.has_open_position,
        "flat" if not ui.has_open_position
        else f"OPEN POSITION: {ui.open_position_side} x{ui.open_position_size}",
    )

    # ── Check 6: Buy and Sell buttons visible ────────────────────────────
    result.add(
        "buy_button_visible",
        ui.buy_button_visible,
        "Buy button detected" if ui.buy_button_visible
        else "Buy button NOT found — check DOM/chart trader layout",
    )
    result.add(
        "sell_button_visible",
        ui.sell_button_visible,
        "Sell button detected" if ui.sell_button_visible
        else "Sell button NOT found — check DOM/chart trader layout",
    )

    # ── Check 7: Kill switch is OFF ──────────────────────────────────────
    if fail_safe_state is not None:
        kill_off = not fail_safe_state.kill_switch_active
        result.add(
            "kill_switch_off",
            kill_off,
            "kill switch is OFF" if kill_off
            else f"KILL SWITCH IS ON — reset before trading",
        )
        # Also check emergency disable
        emerg_off = not fail_safe_state.config.emergency_disable
        result.add(
            "emergency_disable_off",
            emerg_off,
            "emergency disable is OFF" if emerg_off
            else "EMERGENCY DISABLE IS ON",
        )
    else:
        result.add("kill_switch_off", True, "no fail-safe state provided (assumed OFF)")

    # ── Check 8: No read errors ──────────────────────────────────────────
    result.add(
        "no_read_errors",
        len(ui.read_errors) == 0,
        "clean read" if not ui.read_errors
        else f"{len(ui.read_errors)} error(s): {'; '.join(ui.read_errors)}",
    )

    return result


# ── CLI entry point ──────────────────────────────────────────────────────────

def _cli_main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Apex pre-flight validation")
    p.add_argument("--symbol", default="", help="Expected active symbol (e.g. MNQ)")
    p.add_argument("--account", default="", help="Expected account label substring")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)

    result = run_preflight(
        expected_symbol=args.symbol,
        expected_account=args.account,
    )
    result.print_report()
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(_cli_main())
