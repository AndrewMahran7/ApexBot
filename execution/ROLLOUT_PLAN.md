# OpenClaw Execution Layer — Staged Rollout Plan

## Overview

The execution layer has 3 modes: **DRY_RUN**, **SIM**, **LIVE**.
Each stage must be fully verified before advancing.

---

## Stage 1: Dry Run (No Clicks)

**Goal**: Verify signal conversion, validation logic, fail-safes, and logging.

### Steps

```bash
# 1. Run automated unit tests (56 tests)
.\venv\Scripts\python.exe -m pytest tests/test_execution.py -v

# 2. Run dry-run scenarios with synthetic signals
.\venv\Scripts\python.exe run_openclaw_dry_run.py --symbol MNQ --mode challenge

# 3. Review audit logs
cat results/openclaw_execution/*.jsonl | python -m json.tool --no-ensure-ascii
```

### Verify

- [ ] All 56 unit tests pass
- [ ] Audit JSONL files are written correctly
- [ ] Signal expiration rejects old signals
- [ ] Duplicate signal IDs are rejected
- [ ] Duplicate fingerprints within cooldown are rejected
- [ ] Kill switch blocks all signals
- [ ] Kill switch reset allows signals again
- [ ] Zero-size signals are blocked
- [ ] Session cap enforcement works
- [ ] Cooldown timer enforcement works

### Pass criteria: All checkboxes above verified.

---

## Stage 2: UI Read Test (Read-Only, No Clicks)

**Goal**: Verify OpenClaw can read the Tradovate window state.

### Prerequisites

- [ ] OpenClaw installed: `pip install openclaw`
- [ ] Tradovate desktop app open and logged in
- [ ] A chart/DOM trader tab active with MNQ or MES

### Steps

```bash
# Read UI state without clicking anything
.\venv\Scripts\python.exe run_openclaw_dry_run.py --read-ui
```

### Verify

- [ ] `window_found: True`
- [ ] `active_symbol` matches the open chart
- [ ] `quantity_value` reads correctly
- [ ] `account_label` contains expected account identifier
- [ ] `atm_template_name` reads if ATM is configured
- [ ] `buy_button_visible: True`
- [ ] `sell_button_visible: True`
- [ ] `read_errors` is empty

### Troubleshooting

If elements aren't found, the `_ELEMENTS` dict in `execution/openclaw_driver.py`
may need adjustment for your Tradovate version. The UI element patterns are
role/name-based and may vary by layout.

### Pass criteria: All UI fields read correctly with 0 read errors.

---

## Stage 3: Dry Run with Live UI State

**Goal**: Run the full validation pipeline reading real UI state, but still no clicks.

### Steps

```python
from execution.execution_controller import ExecutionController
from execution.fail_safes import FailSafeConfig, RunMode
from execution.risk_bridge import RiskBridgeConfig
from execution.signal_schema import ExecutionMode

ctrl = ExecutionController(
    fail_safe_config=FailSafeConfig(run_mode=RunMode.DRY_RUN),
    risk_bridge_config=RiskBridgeConfig(
        default_mode=ExecutionMode.CHALLENGE,
        base_contracts=1,
    ),
    expected_account="APEX",  # Your account ID fragment
    output_dir="results/openclaw_execution",
)

# Create a synthetic signal matching the active chart
# Then call ctrl.on_signal(...) and check the audit log
```

### Verify

- [ ] Pre-trade validation passes all checks (window, symbol, qty, account)
- [ ] DRY_RUN event is logged with correct "would do" description
- [ ] Signal is registered in the dedup registry
- [ ] No actual click occurs

### Pass criteria: ValidationResult shows ALL PASSED in dry-run mode.

---

## Stage 4: Sim Account Execution

**Goal**: Test actual clicking on a sim/paper account.

### Prerequisites

- [ ] Tradovate connected to a **SIM** account
- [ ] ATM template preconfigured with SL/TP
- [ ] MNQ (or MES) chart/DOM trader active
- [ ] Quantity set to 1 manually as baseline

### Steps

```python
from execution.fail_safes import RunMode

# IMPORTANT: Verify you are on a SIM account before proceeding
ctrl.set_run_mode(RunMode.SIM)

# Send a test signal
result = ctrl.on_signal(live_signal, size_mult=0.35, symbol="MNQ")
```

### Verify

- [ ] Quantity was set correctly in the UI
- [ ] Buy/Sell button was clicked
- [ ] Post-trade validation confirms: position detected, correct side, correct size
- [ ] Execution confirmed event in audit log
- [ ] ATM template placed SL/TP automatically (check Tradovate orders panel)
- [ ] Cooldown prevents immediate second execution

### Cleanup

- Close any open sim positions manually
- Review `results/openclaw_execution/*.jsonl` for the full audit trail

### Pass criteria: Full round-trip confirmed on sim with correct side/size.

---

## Stage 5: Eval Account — Smallest Safe Size

**Goal**: First real execution on eval account.

### Prerequisites

- [ ] Stage 4 fully verified on sim
- [ ] Tradovate connected to eval account
- [ ] ATM template verified on eval
- [ ] Expected account fragment updated in config
- [ ] `base_contracts=1`, `max_contracts=1` in RiskBridgeConfig
- [ ] `max_open_trades=1`, `cooldown_seconds=120` in FailSafeConfig
- [ ] Market is open and liquid

### Steps

```python
ctrl = ExecutionController(
    fail_safe_config=FailSafeConfig(
        run_mode=RunMode.LIVE,
        max_open_trades=1,
        cooldown_seconds=120,      # 2 min between trades
        max_executions_per_session=3,  # Very conservative cap
        screenshot_on_failure=True,
    ),
    risk_bridge_config=RiskBridgeConfig(
        default_mode=ExecutionMode.CHALLENGE,
        base_contracts=1,
        max_contracts=1,  # HARD CAP at 1 contract
    ),
    expected_account="APEX-XXXXX",  # Your eval account
)

# Wire to strategy pipeline
# risk.on_approved = lambda sig: ctrl.on_signal(sig, size_mult=decision.size_mult, symbol="MNQ")
```

### Verify after first trade

- [ ] Position opened with correct side
- [ ] Position size is exactly 1 contract
- [ ] ATM bracket (SL + TP) visible in orders
- [ ] Audit log shows full trace: received → validated → attempted → confirmed
- [ ] No duplicate orders
- [ ] Cooldown prevents immediate re-entry

### Emergency procedures

1. **Wrong trade opened**: Kill switch → `ctrl.activate_kill_switch("wrong trade")`
2. **UI desync detected**: Close position manually in Tradovate, then kill switch
3. **Multiple positions**: Kill switch auto-triggered by PositionMonitor

### Pass criteria: 3 consecutive trades execute correctly with full audit trail.

---

## Ongoing Monitoring

Once live:

1. Review `results/openclaw_execution/YYYY-MM-DD.jsonl` daily
2. Check for `validation_failed` or `post_trade_mismatch` events
3. Monitor `fail_safe_triggered` frequency
4. Verify no `kill_switch_activated` events (unless intentional)
5. Periodically run `ctrl.status()` to inspect system state

---

## Rollback Procedure

At any point:

```python
# Immediate stop
ctrl.activate_kill_switch("rollback")

# Or disable entirely
ctrl.fail_safe.set_emergency_disable(True)

# Or switch back to dry-run
ctrl.set_run_mode(RunMode.DRY_RUN)
```

All of these are logged in the audit trail.
