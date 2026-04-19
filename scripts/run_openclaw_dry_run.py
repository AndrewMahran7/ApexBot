"""
Dry-run runner for the OpenClaw execution layer.

This script demonstrates and tests the full execution pipeline
WITHOUT actually clicking anything.  Use this to verify:

1. Signal conversion works correctly
2. Pre-trade validation logic catches problems
3. Fail-safes behave as expected
4. Audit logging produces correct output
5. The pipeline can be switched from dry-run → sim → live

Usage:
    # Basic dry run with synthetic signals
    python run_openclaw_dry_run.py

    # Dry run with specific symbol and mode
    python run_openclaw_dry_run.py --symbol MNQ --mode challenge

    # Show what live mode would do (still dry-run, just shows config)
    python run_openclaw_dry_run.py --show-live-config

    # Run against actual Tradovate window (read-only, no clicks)
    python run_openclaw_dry_run.py --read-ui
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from execution.signal_schema import (
    ExecutionMode,
    ExecutionSide,
    ExecutionSignal,
    create_signal_id,
)
from execution.fail_safes import FailSafeConfig, RunMode
from execution.risk_bridge import RiskBridgeConfig
from execution.execution_controller import ExecutionController
from execution.openclaw_driver import OpenClawDriver
from execution.validators import PreTradeValidator, UIState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/openclaw_execution/dry_run.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ── Synthetic signal generators ──────────────────────────────────────────────

def make_synthetic_live_signal(direction: str = "long", symbol: str = "MNQ"):
    """Create a fake LiveSignal-like object for testing."""

    class SyntheticLiveSignal:
        pass

    sig = SyntheticLiveSignal()
    sig.timestamp = datetime.datetime.now(datetime.timezone.utc)
    sig.direction = direction

    class FakeSignalType:
        name = "LONG_ENTRY" if direction == "long" else "SHORT_ENTRY"

    sig.signal_type = FakeSignalType()
    sig.entry = 5050.25
    sig.stop = 4990.0 if direction == "long" else 5110.0
    sig.take_profit = 5140.0 if direction == "long" else 4960.0
    sig.position_size = 1.0
    sig.reason = f"synthetic {direction} for dry-run test"
    sig.strategy_type = "ema50_breakout"
    sig.ml_prob = 0.65
    sig.percentile = 0.75
    sig.quality_score = 0.70
    sig.symbol = symbol
    return sig


# ── Test scenarios ───────────────────────────────────────────────────────────

def run_scenario_basic(ctrl: ExecutionController, symbol: str) -> None:
    """Scenario 1: Basic signal processing."""
    print("\n" + "=" * 60)
    print("SCENARIO 1: Basic dry-run signal processing")
    print("=" * 60)

    sig = make_synthetic_live_signal("long", symbol)
    result = ctrl.on_signal(sig, size_mult=0.35, symbol=symbol)
    print(f"  Result: {'PROCESSED' if result else 'BLOCKED'}")
    print(f"  Audit events: {ctrl.audit_logger.event_count}")
    print(f"  Summary: {ctrl.audit_logger.summary()}")


def run_scenario_duplicate(ctrl: ExecutionController, symbol: str) -> None:
    """Scenario 2: Duplicate signal rejection."""
    print("\n" + "=" * 60)
    print("SCENARIO 2: Duplicate signal rejection")
    print("=" * 60)

    sig = make_synthetic_live_signal("long", symbol)
    r1 = ctrl.on_signal(sig, size_mult=0.35, symbol=symbol)
    r2 = ctrl.on_signal(sig, size_mult=0.35, symbol=symbol)
    print(f"  First call:  {'PROCESSED' if r1 else 'BLOCKED'}")
    print(f"  Second call: {'PROCESSED' if r2 else 'BLOCKED (duplicate)'}")


def run_scenario_blocked_size(ctrl: ExecutionController, symbol: str) -> None:
    """Scenario 3: Zero-size blocking."""
    print("\n" + "=" * 60)
    print("SCENARIO 3: Blocked by zero sizing")
    print("=" * 60)

    sig = make_synthetic_live_signal("short", symbol)
    result = ctrl.on_signal(sig, size_mult=0.0, symbol=symbol)
    print(f"  Result: {'PROCESSED' if result else 'BLOCKED (size=0)'}")


def run_scenario_kill_switch(ctrl: ExecutionController, symbol: str) -> None:
    """Scenario 4: Kill switch behavior."""
    print("\n" + "=" * 60)
    print("SCENARIO 4: Kill switch activation and reset")
    print("=" * 60)

    ctrl.activate_kill_switch("test kill switch")
    sig = make_synthetic_live_signal("long", symbol)
    r1 = ctrl.on_signal(sig, size_mult=0.35, symbol=symbol)
    print(f"  With kill switch: {'PROCESSED' if r1 else 'BLOCKED'}")

    ctrl.reset_kill_switch()
    r2 = ctrl.on_signal(sig, size_mult=0.35, symbol=symbol)
    print(f"  After reset:      {'PROCESSED' if r2 else 'BLOCKED'}")


def run_scenario_stale_signal(ctrl: ExecutionController, symbol: str) -> None:
    """Scenario 5: Stale signal rejection."""
    print("\n" + "=" * 60)
    print("SCENARIO 5: Stale signal rejection")
    print("=" * 60)

    sig = make_synthetic_live_signal("long", symbol)
    sig.timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)
    result = ctrl.on_signal(sig, size_mult=0.35, symbol=symbol)
    print(f"  Result: {'PROCESSED' if result else 'BLOCKED (stale)'}")


def run_scenario_read_ui(symbol: str) -> None:
    """Scenario 6: Read Tradovate UI state (no clicks)."""
    print("\n" + "=" * 60)
    print("SCENARIO 6: Read Tradovate UI state")
    print("=" * 60)

    driver = OpenClawDriver(dry_run=True)
    ui = driver.read_ui_state()

    print(f"  Window found:   {ui.window_found}")
    print(f"  Window title:   {ui.window_title}")
    print(f"  Account:        {ui.account_label}")
    print(f"  Active symbol:  {ui.active_symbol}")
    print(f"  Quantity:       {ui.quantity_value}")
    print(f"  ATM template:   {ui.atm_template_name}")
    print(f"  Has position:   {ui.has_open_position}")
    print(f"  Buy visible:    {ui.buy_button_visible}")
    print(f"  Sell visible:   {ui.sell_button_visible}")
    print(f"  Read errors:    {ui.read_errors}")


def run_scenario_show_config(symbol: str, mode: str) -> None:
    """Print the configuration that would be used for live execution."""
    print("\n" + "=" * 60)
    print("LIVE CONFIGURATION PREVIEW")
    print("=" * 60)

    fs_config = FailSafeConfig(
        run_mode=RunMode.LIVE,
        max_open_trades=1,
        cooldown_seconds=60,
        max_executions_per_session=20,
        confirmation_timeout_seconds=10,
        screenshot_on_failure=True,
    )

    rb_config = RiskBridgeConfig(
        default_mode=ExecutionMode.CHALLENGE if mode == "challenge" else ExecutionMode.FUNDED,
        base_contracts=1,
        min_contracts=1,
        max_contracts=2,
        signal_ttl_seconds=30,
    )

    print(f"\n  Fail-Safe Config:")
    print(f"    Run mode:           {fs_config.run_mode.value}")
    print(f"    Max open trades:    {fs_config.max_open_trades}")
    print(f"    Cooldown:           {fs_config.cooldown_seconds}s")
    print(f"    Max session execs:  {fs_config.max_executions_per_session}")
    print(f"    Confirm timeout:    {fs_config.confirmation_timeout_seconds}s")
    print(f"    Screenshot on fail: {fs_config.screenshot_on_failure}")

    print(f"\n  Risk Bridge Config:")
    print(f"    Mode:               {rb_config.default_mode.value}")
    print(f"    Base contracts:     {rb_config.base_contracts}")
    print(f"    Min contracts:      {rb_config.min_contracts}")
    print(f"    Max contracts:      {rb_config.max_contracts}")
    print(f"    Signal TTL:         {rb_config.signal_ttl_seconds}s")

    print(f"\n  To go live, set: run_mode=RunMode.LIVE")
    print(f"  WARNING: Live mode will click Buy/Sell buttons in Tradovate!")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenClaw execution layer dry-run")
    parser.add_argument("--symbol", default="MNQ", help="Symbol to test (default: MNQ)")
    parser.add_argument("--mode", default="challenge", choices=["challenge", "funded"])
    parser.add_argument("--show-live-config", action="store_true", help="Show live configuration")
    parser.add_argument("--read-ui", action="store_true", help="Read actual Tradovate UI state")
    parser.add_argument("--account", default="", help="Expected account label fragment")
    parser.add_argument("--output-dir", default="results/openclaw_execution")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  OPENCLAW EXECUTION LAYER — DRY RUN")
    print(f"  Symbol: {args.symbol}")
    print(f"  Mode:   {args.mode}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    if args.show_live_config:
        run_scenario_show_config(args.symbol, args.mode)
        return

    if args.read_ui:
        run_scenario_read_ui(args.symbol)
        return

    # Create controller in dry-run mode
    ctrl = ExecutionController(
        fail_safe_config=FailSafeConfig(
            run_mode=RunMode.DRY_RUN,
            max_open_trades=1,
            cooldown_seconds=60,
        ),
        risk_bridge_config=RiskBridgeConfig(
            default_mode=ExecutionMode.CHALLENGE if args.mode == "challenge" else ExecutionMode.FUNDED,
            base_contracts=1,
        ),
        expected_account=args.account,
        output_dir=args.output_dir,
    )

    # Run all scenarios
    run_scenario_basic(ctrl, args.symbol)

    # Reset for next scenario (need fresh controller for dedup tests)
    ctrl.reset_session()
    run_scenario_duplicate(ctrl, args.symbol)

    ctrl.reset_session()
    run_scenario_blocked_size(ctrl, args.symbol)

    ctrl.reset_session()
    run_scenario_kill_switch(ctrl, args.symbol)

    ctrl.reset_session()
    run_scenario_stale_signal(ctrl, args.symbol)

    # Final summary
    print("\n" + "=" * 60)
    print("DRY RUN COMPLETE")
    print("=" * 60)
    status = ctrl.status()
    print(f"  Total audit events: {status['audit_events']}")
    print(f"  Event breakdown:    {json.dumps(status['audit_summary'], indent=4)}")
    print(f"\n  Logs saved to: {args.output_dir}/")
    print(f"\n  Next steps:")
    print(f"    1. Review logs in {args.output_dir}/")
    print(f"    2. Run with --read-ui to test Tradovate detection")
    print(f"    3. Run with --show-live-config to preview live settings")
    print(f"    4. When ready: change RunMode.DRY_RUN → RunMode.SIM")


if __name__ == "__main__":
    main()
