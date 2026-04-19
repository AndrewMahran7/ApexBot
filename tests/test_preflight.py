"""Tests for execution.preflight module."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import MagicMock, patch

import pytest

from execution.preflight import PreflightResult, run_preflight
from execution.validators import UIState


# ── PreflightResult unit tests ───────────────────────────────────────────────

class TestPreflightResult:
    def test_all_pass(self):
        r = PreflightResult()
        r.add("check_a", True, "ok")
        r.add("check_b", True, "ok")
        assert r.passed
        assert len(r.failed_checks) == 0

    def test_one_fail(self):
        r = PreflightResult()
        r.add("check_a", True, "ok")
        r.add("check_b", False, "bad")
        assert not r.passed
        assert len(r.failed_checks) == 1
        assert r.failed_checks[0]["name"] == "check_b"

    def test_multiple_fail(self):
        r = PreflightResult()
        r.add("a", False, "x")
        r.add("b", False, "y")
        r.add("c", True, "z")
        assert not r.passed
        assert len(r.failed_checks) == 2

    def test_print_report_pass(self, capsys):
        r = PreflightResult()
        r.add("window", True, "found")
        r.print_report()
        out = capsys.readouterr().out
        assert "SYSTEM READY" in out
        assert "PASS" in out

    def test_print_report_fail(self, capsys):
        r = PreflightResult()
        r.add("window", False, "not found")
        r.print_report()
        out = capsys.readouterr().out
        assert "BLOCKED" in out
        assert "FAIL" in out


# ── Helper to build a UIState ────────────────────────────────────────────────

def _good_ui(**overrides) -> UIState:
    """Return a UIState that passes all checks."""
    defaults = dict(
        window_found=True,
        window_title="Tradovate Trader",
        account_label="APEX-99999PA",
        active_symbol="MNQ Z5",
        quantity_value=1,
        atm_template_name="APEX_MNQ_10SL",
        has_open_position=False,
        open_position_side=None,
        open_position_size=0,
        pending_orders=0,
        buy_button_visible=True,
        sell_button_visible=True,
        read_errors=[],
    )
    defaults.update(overrides)
    return UIState(**defaults)


# ── run_preflight integration tests ──────────────────────────────────────────

class TestRunPreflight:
    """Tests that mock the OpenClawDriver to inject controlled UIState."""

    def _run(self, ui: UIState, **kwargs) -> PreflightResult:
        """Run preflight with a mocked driver returning the given UIState."""
        mock_driver = MagicMock()
        mock_driver.read_ui_state.return_value = ui

        with patch("execution.preflight.OpenClawDriver", return_value=mock_driver):
            return run_preflight(**kwargs)

    # ── All-pass scenario ────────────────────────────────────────────────

    def test_all_pass(self):
        result = self._run(
            _good_ui(),
            expected_symbol="MNQ",
            expected_account="APEX",
        )
        assert result.passed
        assert len(result.failed_checks) == 0

    def test_all_pass_no_filters(self):
        result = self._run(_good_ui())
        assert result.passed

    # ── Individual failure scenarios ─────────────────────────────────────

    def test_fail_window_not_found(self):
        result = self._run(_good_ui(window_found=False))
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "tradovate_window" in failed

    def test_fail_wrong_account(self):
        result = self._run(
            _good_ui(account_label="SIM-12345"),
            expected_account="APEX",
        )
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "correct_account" in failed

    def test_fail_wrong_symbol(self):
        result = self._run(
            _good_ui(active_symbol="MES Z5"),
            expected_symbol="MNQ",
        )
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "correct_symbol" in failed

    def test_fail_no_atm(self):
        result = self._run(_good_ui(atm_template_name=None))
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "atm_template" in failed

    def test_fail_atm_empty_string(self):
        result = self._run(_good_ui(atm_template_name="  "))
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "atm_template" in failed

    def test_fail_open_position(self):
        result = self._run(
            _good_ui(has_open_position=True, open_position_side="long", open_position_size=2),
        )
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "no_open_positions" in failed

    def test_fail_buy_button_hidden(self):
        result = self._run(_good_ui(buy_button_visible=False))
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "buy_button_visible" in failed

    def test_fail_sell_button_hidden(self):
        result = self._run(_good_ui(sell_button_visible=False))
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "sell_button_visible" in failed

    def test_fail_read_errors(self):
        result = self._run(_good_ui(read_errors=["qty field: timeout"]))
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "no_read_errors" in failed

    # ── Kill switch / fail-safe ──────────────────────────────────────────

    def test_fail_kill_switch_on(self):
        fs = MagicMock()
        fs.kill_switch_active = True
        fs.config.emergency_disable = False
        result = self._run(_good_ui(), fail_safe_state=fs)
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "kill_switch_off" in failed

    def test_fail_emergency_disable(self):
        fs = MagicMock()
        fs.kill_switch_active = False
        fs.config.emergency_disable = True
        result = self._run(_good_ui(), fail_safe_state=fs)
        assert not result.passed
        failed = [c["name"] for c in result.failed_checks]
        assert "emergency_disable_off" in failed

    def test_pass_fail_safe_clean(self):
        fs = MagicMock()
        fs.kill_switch_active = False
        fs.config.emergency_disable = False
        result = self._run(
            _good_ui(),
            expected_symbol="MNQ",
            expected_account="APEX",
            fail_safe_state=fs,
        )
        assert result.passed

    # ── Multiple failures at once ────────────────────────────────────────

    def test_multiple_failures(self):
        result = self._run(
            _good_ui(
                buy_button_visible=False,
                sell_button_visible=False,
                has_open_position=True,
                atm_template_name="",
            ),
        )
        assert not result.passed
        assert len(result.failed_checks) == 4

    # ── Early exit when window not found ─────────────────────────────────

    def test_window_not_found_short_circuits(self):
        """When window is missing, remaining checks should not run."""
        result = self._run(
            _good_ui(window_found=False),
            expected_symbol="MNQ",
            expected_account="APEX",
        )
        assert not result.passed
        # Only the window check should be present
        assert len(result.checks) == 1
        assert result.checks[0]["name"] == "tradovate_window"

    # ── Account/symbol filters are optional ──────────────────────────────

    def test_skip_account_filter(self):
        result = self._run(
            _good_ui(account_label="ANYTHING"),
            expected_account="",
        )
        acct_check = next(c for c in result.checks if c["name"] == "correct_account")
        assert acct_check["passed"]

    def test_skip_symbol_filter(self):
        result = self._run(
            _good_ui(active_symbol="ES Z5"),
            expected_symbol="",
        )
        sym_check = next(c for c in result.checks if c["name"] == "correct_symbol")
        assert sym_check["passed"]
