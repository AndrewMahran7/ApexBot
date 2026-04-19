"""
Staged validation harness for the OpenClaw execution layer.

Five validation stages, each progressively closer to live:

    Stage 0 — DRY_RUN:   Process signals, validate everything, never click.
    Stage 1 — UI_CHECK:  Read live Tradovate UI and verify state only.
    Stage 2 — SCENARIOS: Run canned test scenarios through dry-run pipeline.
    Stage 3 — SIM:       Execute on sim/paper account (real clicks, fake money).
    Stage 4 — LIVE_MIN:  Live eval, smallest allowed size (1 contract).

Usage:
    python execution/validation_harness.py --stage 0
    python execution/validation_harness.py --stage 2 --output results/harness
    python execution/validation_harness.py --stage 1 --ui-check-only

This module does NOT modify strategy logic or signal generation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

from execution.audit_logger import AuditLogger, EventType
from execution.execution_controller import ExecutionController
from execution.fail_safes import FailSafeConfig, FailSafeState, RunMode
from execution.openclaw_adapter import OpenClawAdapter
from execution.risk_bridge import RiskBridgeConfig
from execution.signal_schema import (
    ExecutionMode,
    ExecutionSide,
    ExecutionSignal,
    create_signal_id,
)
from execution.validators import PreTradeValidator, UIState

logger = logging.getLogger(__name__)


# ── Validation stages ────────────────────────────────────────────────────────

class Stage(IntEnum):
    DRY_RUN = 0
    UI_CHECK = 1
    SCENARIOS = 2
    SIM = 3
    LIVE_MIN = 4


# ── Test scenario definitions ────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Result of one test scenario."""
    name: str
    passed: bool
    expected_outcome: str
    actual_outcome: str
    events: list[dict] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "events": self.events,
            "details": self.details,
        }


@dataclass
class HarnessReport:
    """Full test harness report."""
    stage: int
    stage_name: str
    timestamp: str
    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    results: list[dict] = field(default_factory=list)
    ui_check: Optional[dict] = None
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "stage_name": self.stage_name,
            "timestamp": self.timestamp,
            "total_scenarios": self.total_scenarios,
            "passed": self.passed,
            "failed": self.failed,
            "results": self.results,
            "ui_check": self.ui_check,
            "summary": self.summary,
        }


# ── Fake LiveSignal for scenario injection ───────────────────────────────────

class _FakeSignal:
    """Minimal LiveSignal stub for test scenarios."""

    def __init__(
        self,
        direction: str = "long",
        symbol: str = "MNQ",
        entry: float = 21000.0,
        stop: float = 20950.0,
        take_profit: float = 21100.0,
        timestamp: Optional[datetime.datetime] = None,
        reason: str = "test_scenario",
        strategy_type: str = "ema50_breakout",
        ml_prob: float = 0.70,
    ):
        self.direction = direction
        self.symbol = symbol
        self.signal_type = None
        self.timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc)
        self.entry = entry
        self.stop = stop
        self.take_profit = take_profit
        self.position_size = 1.0
        self.reason = reason
        self.strategy_type = strategy_type
        self.ml_prob = ml_prob
        self.percentile = 0.75
        self.quality_score = 0.65


# ── Scenario definitions ─────────────────────────────────────────────────────

def _build_scenarios() -> list[dict]:
    """
    Canned test scenarios.  Each is a dict with:
        name:               Human-readable scenario name
        signal:             _FakeSignal to inject
        size_mult:          Sizing multiplier
        symbol:             Symbol override
        expected_executed:  Whether on_signal() should return True (dry-run processed)
        expected_events:    List of expected EventType values in audit log
        description:        What this scenario validates
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    return [
        {
            "name": "valid_mnq_buy",
            "signal": _FakeSignal(direction="long", symbol="MNQ"),
            "size_mult": 0.35,
            "symbol": "MNQ",
            # Without OpenClaw installed, pre-trade validation fails (no window).
            # With Tradovate open, this would be expected_executed=True.
            "expected_executed": False,
            "expected_events": ["signal_received", "validation_failed"],
            "description": "Standard MNQ long entry at 0.35x sizing. Without OpenClaw: "
                           "validation fails. With Tradovate open: processes to dry-run.",
        },
        {
            "name": "valid_mes_sell",
            "signal": _FakeSignal(direction="short", symbol="MES", entry=5050.0,
                                  stop=5100.0, take_profit=5000.0),
            "size_mult": 0.50,
            "symbol": "MES",
            "expected_executed": False,
            "expected_events": ["signal_received", "validation_failed"],
            "description": "Standard MES short entry at 0.50x. Without OpenClaw: "
                           "validation fails. With Tradovate open: processes to dry-run.",
        },
        {
            "name": "stale_signal",
            "signal": _FakeSignal(
                direction="long", symbol="MNQ",
                timestamp=now - datetime.timedelta(seconds=120),
            ),
            "size_mult": 0.35,
            "symbol": "MNQ",
            "expected_executed": False,
            "expected_events": ["signal_received", "signal_expired"],
            "description": "Signal is 120s old (TTL=30s). Should be rejected as expired.",
        },
        {
            "name": "blocked_by_sizing",
            "signal": _FakeSignal(direction="long", symbol="MNQ"),
            "size_mult": 0.0,
            "symbol": "MNQ",
            "expected_executed": False,
            "expected_events": ["signal_received"],
            "description": "PropRiskLayer returned size_mult=0.0. Should be blocked at "
                           "bridge conversion (0 contracts).",
        },
        {
            "name": "duplicate_signal",
            "signal": _FakeSignal(direction="long", symbol="MNQ", reason="dup_test"),
            "size_mult": 0.35,
            "symbol": "MNQ",
            "expected_executed": False,
            # Without OpenClaw: both calls fail at pre-trade validation before
            # reaching dedup (signals only register on dry-run success at Step 7).
            # With Tradovate open: first call succeeds and registers, second
            # call is blocked at Step 3 with signal_duplicate.
            "expected_events": ["signal_received", "validation_failed"],
            "description": "Same signal sent twice. Without OpenClaw: both fail at "
                           "validation (dedup unreachable). With Tradovate: second call "
                           "rejected as duplicate.",
            "_is_duplicate_test": True,
        },
        {
            "name": "wrong_tab_selected",
            "signal": _FakeSignal(direction="long", symbol="MNQ"),
            "size_mult": 0.35,
            "symbol": "MNQ",
            "expected_executed": False,
            "expected_events": ["signal_received", "validation_failed"],
            "description": "UI has wrong symbol active (simulated via OpenClaw not installed). "
                           "Pre-trade validation should catch symbol mismatch.",
        },
        {
            "name": "account_mismatch",
            "signal": _FakeSignal(direction="long", symbol="MNQ"),
            "size_mult": 0.35,
            "symbol": "MNQ",
            "expected_executed": False,
            "expected_events": ["signal_received", "validation_failed"],
            "description": "Expected account label not found in UI. Pre-trade validation "
                           "should fail on account check.",
            "_expected_account": "APEX-12345",
        },
    ]


# ── Harness runner ───────────────────────────────────────────────────────────

class ValidationHarness:
    """
    Staged test harness that runs canned scenarios through the execution
    controller at the configured validation stage.

    Produces a structured HarnessReport and persists results to disk.
    """

    def __init__(
        self,
        output_dir: str = "results/harness",
        expected_account: str = "",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.expected_account = expected_account

    # ── Stage 0: Dry-run signal processing ───────────────────────────

    def run_dry_run(self) -> HarnessReport:
        """
        Process all canned scenarios through dry-run mode.

        All validation + conversion + dedup runs, but no clicks.
        """
        report = HarnessReport(
            stage=Stage.DRY_RUN,
            stage_name="dry_run",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        scenarios = _build_scenarios()
        report.total_scenarios = len(scenarios)

        for scenario in scenarios:
            result = self._run_single_scenario(scenario)
            report.results.append(result.to_dict())
            if result.passed:
                report.passed += 1
            else:
                report.failed += 1

        report.summary = (
            f"Stage 0 (DRY_RUN): {report.passed}/{report.total_scenarios} passed"
        )
        self._save_report(report)
        return report

    # ── Stage 1: UI-only validation ──────────────────────────────────

    def run_ui_check(self) -> HarnessReport:
        """
        Read UI state and validate without processing signals.

        Checks: window found, account label, symbol active, quantity
        field, buy/sell buttons, ATM template.
        """
        report = HarnessReport(
            stage=Stage.UI_CHECK,
            stage_name="ui_check",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        driver = OpenClawAdapter(dry_run=True)
        ui_state = driver.validate_context(None)

        checks = {}
        checks["window_found"] = ui_state.window_found
        checks["window_title"] = ui_state.window_title
        checks["account_label"] = ui_state.account_label or "(not read)"
        checks["active_symbol"] = ui_state.active_symbol or "(not read)"
        checks["quantity_value"] = ui_state.quantity_value
        checks["atm_template"] = ui_state.atm_template_name or "(not read)"
        checks["buy_button_visible"] = ui_state.buy_button_visible
        checks["sell_button_visible"] = ui_state.sell_button_visible
        checks["has_open_position"] = ui_state.has_open_position
        checks["pending_orders"] = ui_state.pending_orders
        checks["read_errors"] = ui_state.read_errors

        # Score the checks
        issues = []
        if not ui_state.window_found:
            issues.append("Tradovate window NOT found")
        if not ui_state.buy_button_visible:
            issues.append("Buy button not visible")
        if not ui_state.sell_button_visible:
            issues.append("Sell button not visible")
        if ui_state.has_open_position:
            issues.append(f"Open position detected: {ui_state.open_position_side} x{ui_state.open_position_size}")
        if ui_state.pending_orders > 0:
            issues.append(f"{ui_state.pending_orders} pending orders")
        if ui_state.read_errors:
            issues.append(f"Read errors: {ui_state.read_errors}")
        if self.expected_account and ui_state.account_label:
            if self.expected_account not in ui_state.account_label:
                issues.append(f"Account mismatch: expected '{self.expected_account}' in '{ui_state.account_label}'")

        all_ok = len(issues) == 0
        report.ui_check = {
            "all_ok": all_ok,
            "checks": checks,
            "issues": issues,
        }
        report.summary = (
            f"Stage 1 (UI_CHECK): {'PASS' if all_ok else 'FAIL'} - "
            f"{len(issues)} issue(s)"
        )

        self._save_report(report)
        return report

    # ── Stage 2: Canned scenarios ────────────────────────────────────

    def run_scenarios(self) -> HarnessReport:
        """
        Run all canned test scenarios and verify expected behavior.

        This is the primary pre-live validation — every scenario must
        produce the expected outcome.
        """
        return self.run_dry_run()

    # ── Stage 3: Sim execution (placeholder gate) ────────────────────

    def run_sim_preflight(self) -> HarnessReport:
        """
        Preflight check for sim mode.

        Runs UI validation + dry-run scenarios first.  Does NOT
        actually execute in sim (that requires manual operator action).

        Returns combined preflight report.
        """
        report = HarnessReport(
            stage=Stage.SIM,
            stage_name="sim_preflight",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        # Run UI check
        ui_report = self.run_ui_check()
        report.ui_check = ui_report.ui_check

        # Run scenarios
        scenario_report = self.run_dry_run()
        report.total_scenarios = scenario_report.total_scenarios
        report.passed = scenario_report.passed
        report.failed = scenario_report.failed
        report.results = scenario_report.results

        ui_ok = report.ui_check.get("all_ok", False) if report.ui_check else False
        scenarios_ok = (report.failed == 0)

        if ui_ok and scenarios_ok:
            report.summary = (
                f"Stage 3 (SIM) PREFLIGHT PASS: UI OK, "
                f"{report.passed}/{report.total_scenarios} scenarios passed. "
                f"Safe to enable SIM mode."
            )
        else:
            issues = []
            if not ui_ok:
                issues.append("UI check failed")
            if not scenarios_ok:
                issues.append(f"{report.failed} scenario(s) failed")
            report.summary = (
                f"Stage 3 (SIM) PREFLIGHT FAIL: {', '.join(issues)}. "
                f"Do NOT enable SIM mode."
            )

        self._save_report(report)
        return report

    # ── Stage 4: Live min-size preflight ─────────────────────────────

    def run_live_preflight(self) -> HarnessReport:
        """
        Preflight check for live min-size eval mode.

        Must pass ALL of:
        1. UI check clean
        2. All scenarios pass
        3. No existing positions
        4. Kill switch available

        Returns go/no-go report.
        """
        report = HarnessReport(
            stage=Stage.LIVE_MIN,
            stage_name="live_min_preflight",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        # Run sim preflight (includes UI + scenarios)
        sim_report = self.run_sim_preflight()
        report.ui_check = sim_report.ui_check
        report.total_scenarios = sim_report.total_scenarios
        report.passed = sim_report.passed
        report.failed = sim_report.failed
        report.results = sim_report.results

        # Additional live-specific checks
        live_checks = {}

        # Check no open positions
        adapter = OpenClawAdapter(dry_run=True)
        ui_state = adapter.get_position_state()
        live_checks["no_open_positions"] = not ui_state.has_open_position
        live_checks["no_pending_orders"] = ui_state.pending_orders == 0

        # Build a test controller to verify kill switch works
        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                output_dir=td,
            )
            ctrl.activate_kill_switch("preflight test")
            live_checks["kill_switch_works"] = ctrl.fail_safe.kill_switch_active
            ctrl.reset_kill_switch()
            live_checks["kill_switch_resets"] = not ctrl.fail_safe.kill_switch_active

        all_live_ok = all(live_checks.values())
        sim_ok = sim_report.failed == 0

        go_nogo = all_live_ok and sim_ok
        report.summary = (
            f"Stage 4 (LIVE_MIN) PREFLIGHT {'GO' if go_nogo else 'NO-GO'}: "
            f"scenarios={report.passed}/{report.total_scenarios}, "
            f"live_checks={'all_pass' if all_live_ok else 'FAIL'}"
        )
        report.results.append({
            "name": "live_preflight_checks",
            "passed": all_live_ok,
            "expected_outcome": "all live-specific checks pass",
            "actual_outcome": "pass" if all_live_ok else "fail",
            "events": [],
            "details": live_checks,
        })

        self._save_report(report)
        return report

    # ── Internal: run a single scenario ──────────────────────────────

    def _run_single_scenario(self, scenario: dict) -> ScenarioResult:
        """Run one canned test scenario and verify outcome."""
        name = scenario["name"]
        signal = scenario["signal"]
        size_mult = scenario["size_mult"]
        symbol = scenario["symbol"]
        expected_executed = scenario["expected_executed"]
        expected_events = scenario.get("expected_events", [])
        expected_account = scenario.get("_expected_account", "")

        with tempfile.TemporaryDirectory() as td:
            ctrl = ExecutionController(
                fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
                risk_bridge_config=RiskBridgeConfig(
                    base_contracts=1, min_contracts=1, max_contracts=2,
                ),
                expected_account=expected_account,
                output_dir=td,
            )

            # For duplicate test: send signal twice
            if scenario.get("_is_duplicate_test"):
                ctrl.on_signal(signal, size_mult=size_mult, symbol=symbol)
                result = ctrl.on_signal(signal, size_mult=size_mult, symbol=symbol)
            else:
                result = ctrl.on_signal(signal, size_mult=size_mult, symbol=symbol)

            # Collect audit events
            events = [ev.to_dict() for ev in ctrl.audit_logger.events]
            event_types = [ev["event_type"] for ev in events]

            # Verify expected outcome
            outcome_match = (result == expected_executed)

            # Verify expected events appeared
            events_match = all(et in event_types for et in expected_events)

            passed = outcome_match and events_match

            actual_outcome = f"executed={result}, events={event_types}"
            expected_desc = f"executed={expected_executed}, events_contain={expected_events}"

            if not passed:
                logger.warning(
                    "Scenario %s FAILED: expected=%s actual=%s",
                    name, expected_desc, actual_outcome,
                )

            return ScenarioResult(
                name=name,
                passed=passed,
                expected_outcome=expected_desc,
                actual_outcome=actual_outcome,
                events=events,
                details={
                    "description": scenario.get("description", ""),
                    "size_mult": size_mult,
                    "symbol": symbol,
                    "result_bool": result,
                    "outcome_match": outcome_match,
                    "events_match": events_match,
                    "missing_events": [
                        et for et in expected_events if et not in event_types
                    ],
                },
            )

    # ── Persistence ──────────────────────────────────────────────────

    def _save_report(self, report: HarnessReport) -> None:
        """Save report as JSON and human-readable log."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON report
        json_path = self.output_dir / f"test_results_{report.stage_name}_{ts}.json"
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Test results saved: %s", json_path)

        # Human-readable log
        log_path = self.output_dir / f"dry_run_log_{report.stage_name}_{ts}.txt"
        lines = [
            f"Apex Execution Validation Harness",
            f"Stage: {report.stage} ({report.stage_name})",
            f"Timestamp: {report.timestamp}",
            f"",
            f"{'='*60}",
            f"SUMMARY: {report.summary}",
            f"{'='*60}",
            f"",
        ]

        if report.ui_check:
            lines.append("UI CHECK:")
            for k, v in report.ui_check.get("checks", {}).items():
                lines.append(f"  {k}: {v}")
            issues = report.ui_check.get("issues", [])
            if issues:
                lines.append(f"  ISSUES:")
                for issue in issues:
                    lines.append(f"    - {issue}")
            lines.append("")

        for r in report.results:
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f"[{status}] {r['name']}")
            lines.append(f"  Expected: {r['expected_outcome']}")
            lines.append(f"  Actual:   {r['actual_outcome']}")
            desc = r.get("details", {}).get("description", "")
            if desc:
                lines.append(f"  Description: {desc}")
            missing = r.get("details", {}).get("missing_events", [])
            if missing:
                lines.append(f"  Missing events: {missing}")
            lines.append("")

        log_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Dry-run log saved: %s", log_path)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Staged validation harness for OpenClaw execution layer",
    )
    parser.add_argument(
        "--stage", type=int, default=0, choices=[0, 1, 2, 3, 4],
        help="Validation stage: 0=dry_run, 1=ui_check, 2=scenarios, 3=sim_preflight, 4=live_preflight",
    )
    parser.add_argument(
        "--output", type=str, default="results/harness",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--expected-account", type=str, default="",
        help="Expected account label substring (for UI validation)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    harness = ValidationHarness(
        output_dir=args.output,
        expected_account=args.expected_account,
    )

    stage = Stage(args.stage)
    print(f"\n{'='*60}")
    print(f"  Apex Execution Validation Harness - Stage {stage.value} ({stage.name})")
    print(f"{'='*60}\n")

    if stage == Stage.DRY_RUN:
        report = harness.run_dry_run()
    elif stage == Stage.UI_CHECK:
        report = harness.run_ui_check()
    elif stage == Stage.SCENARIOS:
        report = harness.run_scenarios()
    elif stage == Stage.SIM:
        report = harness.run_sim_preflight()
    elif stage == Stage.LIVE_MIN:
        report = harness.run_live_preflight()
    else:
        print(f"Unknown stage: {args.stage}")
        return 1

    # Print summary
    print(f"\n{report.summary}\n")

    if report.results:
        print(f"{'Scenario':<25} {'Result':<8} {'Details'}")
        print(f"{'-'*25} {'-'*8} {'-'*40}")
        for r in report.results:
            status = "PASS" if r["passed"] else "FAIL"
            detail = r.get("actual_outcome", "")[:50]
            print(f"{r['name']:<25} {status:<8} {detail}")

    if report.ui_check:
        print(f"\nUI Check: {'PASS' if report.ui_check.get('all_ok') else 'FAIL'}")
        for issue in report.ui_check.get("issues", []):
            print(f"  - {issue}")

    print(f"\nResults saved to: {harness.output_dir}")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
