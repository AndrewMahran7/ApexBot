# Apex Trading System — Operations Guide

Complete setup and operations guide for the Apex MES/MNQ futures trading system.

---

## Table of Contents

1. [System Overview](#1--system-overview)
2. [Requirements](#2--requirements)
3. [Environment Setup](#3--environment-setup)
4. [Tradovate Setup](#4--tradovate-setup)
5. [ATM Configuration](#5--atm-configuration)
6. [OpenClaw Setup](#6--openclaw-setup)
7. [System Configuration](#7--system-configuration)
8. [Running the System](#8--running-the-system)
9. [Dry-Run Mode (Mandatory First Step)](#9--dry-run-mode-mandatory-first-step)
10. [Live Trading Checklist](#10--live-trading-checklist)
11. [Monitoring](#11--monitoring)
12. [Kill Switch](#12--kill-switch)
13. [Common Failure Cases](#13--common-failure-cases)
14. [Daily Workflow](#14--daily-workflow)
15. [Troubleshooting](#15--troubleshooting)

---

## 1 — System Overview

Apex is an automated futures trading system for Micro E-mini S&P 500 (MES) and Micro E-mini Nasdaq 100 (MNQ) contracts. It generates trading signals from price data and executes them via the Tradovate desktop application using OpenClaw for UI automation.

### Architecture

```
Historical Bars / Live Feed
        │
        ▼
  StrategyEngine       (signal generation: ORB breakout, EMA, ML filtering)
        │
        ▼
  PropRiskGate         (optional: prop challenge sizing + gating)
        │
        ▼
  RiskManager          (daily loss limit, max trades, concurrent positions)
        │
        ▼
  Execution Layer      (one of:)
    ├─ PaperEngine         (simulated fills for backtesting / paper trading)
    ├─ TradovateClient     (direct REST API — SIM/demo accounts)
    └─ ExecutionController (OpenClaw → Tradovate UI for eval/funded accounts)
```

### Signal Flow (OpenClaw Live Execution)

```
LiveSignal → RiskBridge → ExecutionSignal → SignalRegistry (dedup)
  → FailSafe checks → PreTradeValidator (UI state) → OpenClawDriver (click)
  → PostTradeValidator (confirmation) → AuditLogger (JSONL record)
```

### Key Design Decisions

- **ATM templates handle SL/TP** — the system only clicks Buy or Sell. Stops and targets are set by the Tradovate ATM template.
- **Read before write** — the system always reads and validates UI state before clicking any button.
- **Every click is logged** — full audit trail in JSONL format with optional screenshots.
- **Three run modes** — `DRY_RUN` (no clicks), `SIM` (sim account clicks), `LIVE` (eval/funded account clicks).

---

## 2 — Requirements

### Software

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Tested on 3.13.x |
| OS | Windows 10/11 | Required for Tradovate desktop + OpenClaw |
| Tradovate | Desktop app | Must be logged in with chart/DOM trader visible |
| OpenClaw | Latest | `pip install openclaw` — UI automation library |
| Git | Any | For cloning the repo |

### Accounts

| Account | Purpose |
|---|---|
| Tradovate SIM | Free demo account for testing execution |
| Tradovate Eval | Prop firm evaluation account (e.g., Apex Trader Funding) |
| Tradovate Funded | Funded account (after passing eval) |
| Databento (optional) | For fetching historical market data |

### Network

- Stable internet connection
- Tradovate ports open (standard HTTPS/WSS)

---

## 3 — Environment Setup

### 3.1 Clone the Repository

```powershell
cd C:\Users\YourUser\Desktop
git clone <repo-url> Apex
cd Apex
```

### 3.2 Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **Important**: Always use the venv Python. The ML model (scikit-learn) must match the version in the venv.

### 3.3 Install Dependencies

```powershell
pip install -r requirements.txt
```

Core dependencies:

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data processing |
| `scikit-learn` | ML model (gradient boosting) |
| `matplotlib` | Plotting (optional) |
| `python-dotenv` | `.env` file loading |
| `httpx` | Tradovate REST API client |
| `fastapi`, `uvicorn` | Dashboard web server |
| `pytz` | Timezone handling |

### 3.4 Install OpenClaw (for live execution)

```powershell
pip install openclaw
```

### 3.5 Configure Environment Variables

Create a `.env` file in the project root:

```ini
# Tradovate API credentials (for TradovateClient direct API mode)
TRADOVATE_USERNAME=your_username
TRADOVATE_PASSWORD=your_password
TRADOVATE_CID=your_cid
TRADOVATE_SECRET=your_secret

# Databento (optional — for fetching new historical data)
DATABENTO_API_KEY=your_databento_key

# Telegram alerts (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

> **Security**: The `.env` file is in `.gitignore` and is never committed.

### 3.6 Verify Installation

```powershell
.\venv\Scripts\python.exe -c "import pandas; import sklearn; import fastapi; print('All packages OK')"
```

Run the test suite:

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -x -q
```

All 713+ tests should pass (5 skips are expected — they require FastAPI test dependencies).

---

## 4 — Tradovate Setup

### 4.1 Login

1. Open the **Tradovate desktop application** (not the web version).
2. Log into the correct account:
   - **SIM account** for testing
   - **Eval account** for prop challenge
   - **Funded account** for live trading

### 4.2 Select Correct Account

1. Click the account selector in the top-left area.
2. Select the specific account you want to trade on.
3. **Verify the account label** — note the exact text (e.g., `APEX-12345PA`). You will need this for the `expected_account` config parameter.

### 4.3 Open Charts

Open chart or DOM trader tabs for each symbol you intend to trade:

- **MNQ** (Micro E-mini Nasdaq 100)
- **MES** (Micro E-mini S&P 500)

### 4.4 Chart / DOM Trader Layout (Critical for OpenClaw)

OpenClaw reads UI elements by their accessibility roles and names. The layout must expose:

1. **Buy button** — must be visible and labeled "Buy" in the DOM trader or chart trader panel.
2. **Sell button** — must be visible and labeled "Sell".
3. **Quantity field** — the spinbutton/input labeled "Qty" must be visible and editable.
4. **ATM dropdown** — the combobox for ATM template selection must be visible.
5. **Symbol tabs** — each instrument should have its own tab. The **active tab** determines which instrument OpenClaw interacts with.

**Recommended layout**:

```
┌──────────────────────────────────────────────────┐
│  [MNQ tab] [MES tab]                             │
│                                                  │
│  ┌─────────────────────┐  ┌───────────────────┐  │
│  │   Chart Area        │  │  DOM / Order Entry │  │
│  │                     │  │                    │  │
│  │                     │  │  Qty: [1]          │  │
│  │                     │  │  ATM: [template ▼] │  │
│  │                     │  │                    │  │
│  │                     │  │  [BUY]   [SELL]    │  │
│  │                     │  │                    │  │
│  │                     │  │  Positions Panel   │  │
│  │                     │  │  Orders Panel      │  │
│  └─────────────────────┘  └───────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 4.5 Tab Selection Rules

- The **active tab** determines the instrument. If the system expects to trade MNQ but the MES tab is active, pre-trade validation will fail.
- **Do not switch tabs manually** while the system is running.
- For multi-symbol trading, the system switches tabs programmatically via `select_symbol()`.

---

## 5 — ATM Configuration

ATM (Advanced Trade Management) templates are how Tradovate automatically places stop-loss and take-profit orders. **The Apex system does NOT place SL/TP orders itself** — it relies entirely on the active ATM template.

### 5.1 Create a New ATM Template

1. In the DOM trader or chart trader, find the **ATM dropdown** (usually above the Buy/Sell buttons).
2. Click the dropdown and select **"Create New"** or the gear icon.
3. Configure:

| Parameter | MNQ Value | MES Value | Notes |
|---|---|---|---|
| Stop Loss | 10–15 ticks | 10–15 ticks | Adjust based on your risk tolerance |
| Take Profit | 15–25 ticks | 15–25 ticks | Adjust for R:R ratio |
| Bracket Type | OCO | OCO | One cancels the other |
| Trailing | Off (recommended) | Off | Prop challenges prefer fixed stops |

4. **Name the template** descriptively (e.g., `APEX_MNQ_10SL_20TP`).
5. Save it.

### 5.2 Select the Template Before Trading

1. Click the ATM dropdown.
2. Select your template by name.
3. **Verify it is highlighted/selected** — this is what OpenClaw reads.

### 5.3 Template Must Match the Symbol

- MNQ tick = $0.50 (0.25 points × $2/point)
- MES tick = $1.25 (0.25 points × $5/point)
- Use separate templates if your SL/TP differ between instruments.

### 5.4 Pre-Trade Verification

The `PreTradeValidator` reads the ATM template name from the UI. You can verify this in dry-run mode before going live:

```powershell
.\venv\Scripts\python.exe -m execution.validation_harness --stage 1 --ui-check-only
```

This reads and prints the current UI state including the ATM template name.

---

## 6 — OpenClaw Setup

### 6.1 What OpenClaw Does

OpenClaw is a Windows accessibility automation library that:
- Finds the Tradovate window by title
- Reads UI element text, values, and states
- Clicks buttons and types values

It does **not** inject code into Tradovate — it uses the Windows UI Automation API.

### 6.2 Installation

```powershell
pip install openclaw
```

### 6.3 Required Permissions

- Tradovate must be running (not minimized to system tray).
- The Tradovate window must be **visible** on screen.
- The Python process needs standard user permissions (no admin required in most cases).
- If running on a remote desktop (RDP), the GUI session must be active and unlocked.

### 6.4 Testing Button Detection

Run the UI state reader:

```powershell
.\venv\Scripts\python.exe -m execution.validation_harness --stage 1
```

Expected output includes:

```
window_found: True
active_symbol: MNQ
quantity_value: 1
account_label: APEX-12345PA
atm_template_name: APEX_MNQ_10SL_20TP
buy_button_visible: True
sell_button_visible: True
read_errors: []
```

### 6.5 Verifying Buttons Work (Sim Only)

On a **SIM account only**, run Stage 4 of the validation harness:

```powershell
.\venv\Scripts\python.exe -m execution.validation_harness --stage 3
```

This sends canned test scenarios through the dry-run pipeline, verifying:
- Signal conversion
- Validation passes
- Dedup works
- Fail-safes fire

### 6.6 Troubleshooting OpenClaw

If elements are not detected, the UI element identifiers in `execution/openclaw_driver.py` may need adjustment. All identifiers are in the `_ELEMENTS` dictionary:

```python
_ELEMENTS = {
    "window_title_pattern": "Tradovate",
    "buy_button_name": "Buy",
    "sell_button_name": "Sell",
    "quantity_field_role": "spinbutton",
    "quantity_field_name": "Qty",
    "atm_dropdown_role": "combobox",
    "atm_dropdown_pattern": "ATM",
    # ... etc
}
```

Use Windows **Accessibility Insights** or `openclaw.inspect()` to find the correct role/name for your Tradovate version.

---

## 7 — System Configuration

### 7.1 Configuration Files

| File | Purpose |
|---|---|
| `config/settings.py` | All dataclass configs: `InstrumentConfig`, `StrategyConfig`, `AdaptiveRegimeConfig`, `BacktestConfig` |
| `.env` | API keys, credentials (not committed) |
| `execution/fail_safes.py` | `FailSafeConfig`: run mode, kill switch, cooldown, session cap |
| `execution/risk_bridge.py` | `RiskBridgeConfig`: contract sizing, ATM template names |
| `risk/prop_challenge.py` | `PropConfig`: profit target, max drawdown, daily limits |
| `risk/risk_manager.py` | `RiskConfig`: max daily loss, max trades, max concurrent |

### 7.2 Challenge Mode vs Funded Mode

**Challenge mode** (`--prop-mode`):
- Conservative sizing (0.35× base)
- Strict daily loss limit ($300)
- Profit target ($1,500)
- Max trailing drawdown ($1,000)
- Breakout-only entries
- Max 4 trades/day
- Consecutive loss halt (5 losses → stop trading)

**Funded mode** (default, no `--prop-mode`):
- Standard sizing
- Wider daily loss limit ($500)
- No profit target
- Standard risk limits

Use `PropConfig.for_challenge()` or `PropConfig.for_funded()` factory methods for pre-tuned settings.

### 7.3 Symbol Configuration

Default instrument is MES. To trade MNQ, set the instrument config:

```python
from config.settings import InstrumentConfig

# MES (default)
instrument = InstrumentConfig(symbol="MES", tick_size=0.25, point_value=5.0)

# MNQ
instrument = InstrumentConfig(symbol="MNQ", tick_size=0.25, point_value=2.0)
```

The `INSTRUMENT_REGISTRY` in `config/settings.py` has pre-defined specs for MES, NQ, and RTY.

### 7.4 Sizing Configuration

Position sizing flows through the `RiskBridge`:

```python
from execution.risk_bridge import RiskBridgeConfig

config = RiskBridgeConfig(
    base_contracts=1,      # base size at 1.0x
    min_contracts=1,       # floor (never 0 unless blocked)
    max_contracts=5,       # hard cap for safety
)
```

In prop challenge mode, the `PropRiskLayer` returns a `size_mult` (e.g., 0.35), and `compute_contracts()` translates that to integer contracts.

### 7.5 Fail-Safe Configuration

```python
from execution.fail_safes import FailSafeConfig, RunMode

config = FailSafeConfig(
    run_mode=RunMode.DRY_RUN,          # DRY_RUN / SIM / LIVE
    max_open_trades=1,                  # max simultaneous positions
    cooldown_seconds=60,                # min gap between executions
    max_executions_per_session=20,      # safety cap per day
    kill_switch_enabled=False,          # starts safe
    emergency_disable=False,            # master override
    duplicate_cooldown_seconds=120,     # dedup window
    confirmation_timeout_seconds=10,    # post-trade read timeout
    screenshot_on_failure=True,         # capture on error
)
```

---

## 8 — Running the System

### 8.1 Pre-Flight Validation (Automatic)

When running in `--mode live`, the system automatically runs a pre-flight validation before starting. This checks all critical conditions and **blocks startup if any check fails**.

The pre-flight runs these checks:

| # | Check | What It Verifies |
|---|---|---|
| 1 | `tradovate_window` | Tradovate desktop app is running and visible |
| 2 | `correct_account` | Account label contains the expected substring |
| 3 | `correct_symbol` | Active symbol tab matches expected instrument |
| 4 | `atm_template` | An ATM template is selected (not blank) |
| 5 | `no_open_positions` | No existing positions from a prior session |
| 6 | `buy_button_visible` | OpenClaw can detect the Buy button |
| 7 | `sell_button_visible` | OpenClaw can detect the Sell button |
| 8 | `kill_switch_off` | Kill switch and emergency disable are OFF |
| 9 | `no_read_errors` | OpenClaw read the UI with zero errors |

**Example output (all pass):**

```
============================================================
  APEX PRE-FLIGHT VALIDATION
============================================================
  [PASS]  tradovate_window
         Tradovate Trader
  [PASS]  correct_account
         account: APEX-99999PA
  [PASS]  correct_symbol
         symbol: MNQ Z5
  [PASS]  atm_template
         template: APEX_MNQ_10SL_20TP
  [PASS]  no_open_positions
         flat
  [PASS]  buy_button_visible
         Buy button detected
  [PASS]  sell_button_visible
         Sell button detected
  [PASS]  kill_switch_off
         kill switch is OFF
  [PASS]  no_read_errors
         clean read
------------------------------------------------------------
  RESULT: SYSTEM READY
============================================================
```

**Example output (failure):**

```
  [FAIL]  correct_symbol
         expected 'MNQ', found 'MES Z5'
  [FAIL]  atm_template
         no ATM template selected — stops/targets will NOT be placed
------------------------------------------------------------
  RESULT: BLOCKED — 2 check(s) failed
  Failed: correct_symbol, atm_template
```

**CLI flags for pre-flight:**

```powershell
# Require specific symbol and account
.\venv\Scripts\python.exe scripts\run_live.py `
    --mode live `
    --expected-symbol MNQ `
    --expected-account APEX-99999

# Skip pre-flight (NOT recommended)
.\venv\Scripts\python.exe scripts\run_live.py `
    --mode live `
    --skip-preflight
```

**Run pre-flight standalone** (without starting the system):

```powershell
.\venv\Scripts\python.exe -m execution.preflight --symbol MNQ --account APEX
```

> **Note**: Pre-flight only runs for `--mode live`. Paper and replay modes skip it since they don't interact with the Tradovate UI.

### 8.2 Paper Mode (Historical Data Replay)

Replay historical bars through the full pipeline with simulated execution:

```powershell
.\venv\Scripts\python.exe scripts\run_live.py `
    --mode paper `
    --data data\mes_4y.csv `
    --dashboard `
    --analytics `
    --log-level INFO
```

### 8.3 Paper Mode with Prop Challenge

```powershell
.\venv\Scripts\python.exe scripts\run_live.py `
    --mode paper `
    --data data\mnq_4y.csv `
    --prop-mode `
    --prop-target 1500 `
    --prop-max-dd 1000 `
    --prop-daily-loss 300 `
    --initial-capital 25000 `
    --dashboard `
    --analytics
```

### 8.4 Live Mode (Tradovate API)

```powershell
.\venv\Scripts\python.exe scripts\run_live.py `
    --mode live `
    --dashboard `
    --analytics `
    --prop-mode
```

> **Warning**: Live mode connects to Tradovate via REST API. Ensure your `.env` credentials are correct and `sim_only=True` is set in `TradovateConfig` unless you explicitly intend to trade on a live account.

### 8.5 Start the Dashboard

The dashboard starts automatically with the `--dashboard` flag. Access it at:

```
http://127.0.0.1:8501
```

Dashboard pages:
- `/` — Strategy PnL, equity curve, open positions, trades table, alerts
- `/exec` — Execution monitor (signals, gate decisions, fail-safe state, alerts)

To use a different port:

```powershell
--dashboard-port 9000
```

### 8.6 Multi-Strategy Mode

Enable intraday strategies (VWAP Bounce, Momentum, Mean Reversion) alongside the primary ORB/EMA strategy:

```powershell
.\venv\Scripts\python.exe scripts\run_live.py `
    --mode paper `
    --data data\mes_4y.csv `
    --multi-strategy `
    --min-quality-score 0.40 `
    --intraday-cooldown 4 `
    --dashboard `
    --analytics
```

---

## 9 — Dry-Run Mode (Mandatory First Step)

**Never go live without completing the dry-run validation sequence.**

### 9.1 Stage 0 — Automated Test Suite

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -x -q
```

All tests must pass.

### 9.1b — Pre-Flight Validation (Standalone)

With Tradovate open, run the pre-flight check without starting the trading system:

```powershell
.\venv\Scripts\python.exe -m execution.preflight --symbol MNQ --account APEX
```

This verifies all 9 checks (window, account, symbol, ATM, positions, buttons, kill switch, read errors) and prints a pass/fail report. Fix any failures before proceeding to Stage 1.

### 9.2 Stage 1 — UI Read Check

With Tradovate open and a chart visible:

```powershell
.\venv\Scripts\python.exe -m execution.validation_harness --stage 1
```

Verify all UI fields are read correctly (window found, symbol, qty, account, ATM, buttons visible, zero read errors).

### 9.3 Stage 2 — Canned Scenario Tests

```powershell
.\venv\Scripts\python.exe -m execution.validation_harness --stage 2 --output results\harness
```

Runs 7 test scenarios through the dry-run pipeline:
- Valid MNQ buy
- Valid MES sell
- Stale signal (should be rejected)
- Zero-size signal (should be blocked)
- Duplicate signal (should be rejected)
- Wrong tab selected
- Account mismatch

Review `results/harness/report.json` for pass/fail details.

### 9.4 Stage 3 — Sim Execution (Real Clicks, Fake Money)

**Only on a SIM account:**

```powershell
.\venv\Scripts\python.exe -m execution.validation_harness --stage 3
```

This physically clicks Buy/Sell on the sim account and verifies:
- Quantity was set correctly
- Button was clicked
- Position appeared in the UI
- Correct side and size
- ATM placed SL/TP orders

### 9.5 Verify Signal → Execution Flow

Run a paper backtest and look at the logs:

```powershell
.\venv\Scripts\python.exe scripts\run_live.py `
    --mode paper `
    --data data\mes_5m.csv `
    --log-level DEBUG

# Review logs
Get-Content logs\signals.log | Select-Object -First 20
Get-Content logs\trades.log | Select-Object -First 20
```

Confirm:
- Signals are generated with correct prices, stops, targets
- Risk manager blocks/allows as configured
- Paper fills match expectations

---

## 10 — Live Trading Checklist

Run through this checklist **every time** before enabling live execution:

```
PRE-TRADE CHECKLIST
═══════════════════

Account & Platform
  [ ] Tradovate desktop app is open
  [ ] Correct account is selected (verify account label text)
  [ ] Account has sufficient margin
  [ ] No pending orders from prior sessions

Symbol & Chart
  [ ] Correct symbol tab is active (MNQ or MES)
  [ ] Chart/DOM trader is visible (not minimized)
  [ ] ATM template is selected and shows correct SL/TP

System State
  [ ] No open positions from prior sessions
  [ ] Kill switch is OFF
  [ ] Emergency disable is OFF
  [ ] Session counters are reset (new session)

Apex System
  [ ] Virtual environment activated
  [ ] Latest code pulled (if applicable)
  [ ] Tests pass (quick: pytest tests/ -x -q)
  [ ] Dashboard is running
  [ ] Dry-run mode verified (UI read test passed)
  [ ] Correct run mode selected (DRY_RUN → SIM → LIVE)

Market Conditions
  [ ] Market is open (futures: 6pm ET Sun – 5pm ET Fri)
  [ ] Regular trading hours for your strategy (9:30 AM – 4:00 PM ET)
  [ ] No known economic events in next 30 minutes (or acceptable)

Risk Parameters
  [ ] Daily loss limit is set
  [ ] Max trades per day is set
  [ ] Max concurrent positions is set
  [ ] Prop mode is ON/OFF as intended
```

---

## 11 — Monitoring

### 11.1 Dashboard

Access at `http://127.0.0.1:8501`:

**Strategy Dashboard** (`/`):
- Real-time equity curve (Chart.js)
- PnL cards: total, realized, unrealized, drawdown
- Open positions table
- Completed trades table
- Risk alerts (drawdown warning at -$300, critical at -$450)

**Execution Monitor** (`/exec`):
- Latest signal and gate decision
- Fail-safe status (mode, kill switch, cooldown, session count)
- Execution results and confirmation status
- Classified alerts: stale signals, duplicates, validation failures

### 11.2 Log Files

| File | Contents |
|---|---|
| `logs/main.log` | All messages (DEBUG and above) |
| `logs/signals.log` | Every strategy signal with prices/sizes |
| `logs/trades.log` | Entry and exit events only |
| `logs/errors.log` | Warnings and errors only |
| `logs/telegram_alerts.log` | Telegram notification attempts |

### 11.3 Audit Trail

The `AuditLogger` writes per-day JSONL files to `results/openclaw_execution/`:

```
results/openclaw_execution/
  audit_2026-04-18.jsonl
  screenshots/
    2026-04-18_buy_click_14-30-22.png
```

Each line is a JSON event with type, timestamp, signal details, and outcome.

### 11.4 What to Watch

| Signal | Meaning | Action |
|---|---|---|
| `KILL SWITCH` alert | Automated or manual emergency stop | Investigate, then reset if safe |
| `Duplicate signal` | Same signal sent within cooldown | Normal — dedup is working |
| `Validation failed` | UI state doesn't match expectations | Check Tradovate layout |
| `Confirmation timeout` | Position didn't appear after click | Check if order filled, check ATM |
| Drawdown warning | PnL down $300+ | Monitor closely |
| Drawdown critical | PnL down $450+ | Consider stopping for the day |

### 11.5 Telegram Alerts (Optional)

If configured in `.env`, the system sends trade alerts to Telegram:

- Entry signals with direction, price, strategy type
- Exit events with PnL
- Risk events (blocks, kill switch)

---

## 12 — Kill Switch

### 12.1 What It Does

The kill switch **immediately blocks all future trade executions**. No new orders will be placed. Existing open positions are NOT automatically closed (by design — automatic position closing during volatile conditions can be worse than holding).

### 12.2 When to Use It

- Unexpected behavior from the system
- Wrong account or symbol detected
- Market flash crash or circuit breaker
- Multiple rapid unexpected fills
- Any situation where you need to stop trading immediately

### 12.3 How to Activate

**Via Dashboard** (recommended):

```
POST http://127.0.0.1:8501/api/exec/kill
```

**Via Code**:

```python
controller._fail_safe.activate_kill_switch("manual: suspicious behavior")
```

**Via Risk Manager** (automatic):

The `RiskManager` automatically activates the kill switch when:
- Daily loss limit is breached
- Kill switch conditions configured in `RiskConfig` are met

The `PropRiskGate` activates when:
- Max trailing drawdown is breached
- Consecutive loss limit is hit

### 12.4 How to Reset

**Via Dashboard**:

```
POST http://127.0.0.1:8501/api/exec/unkill
```

**Via Code**:

```python
controller._fail_safe.reset_kill_switch()
```

> **Important**: The kill switch persists across days. You must explicitly reset it. This is intentional — it prevents accidental re-entry after a bad day.

### 12.5 Emergency Disable

For a harder stop, use the `emergency_disable` flag which blocks everything including read operations:

```python
controller._fail_safe.set_emergency_disable(True)
```

---

## 13 — Common Failure Cases

### 13.1 Wrong Tab Selected

**Symptom**: Pre-trade validation fails with "symbol mismatch". Log shows `active_symbol=MES` when signal expects MNQ.

**Fix**:
1. Click the correct tab in Tradovate.
2. The system will auto-recover on the next signal (validation runs per-signal).
3. For multi-symbol mode, ensure both tabs exist — the driver calls `select_symbol()` to switch.

### 13.2 OpenClaw Not Detecting Buttons

**Symptom**: `buy_button_visible: False`, `sell_button_visible: False`, or `read_errors` list is non-empty.

**Fixes**:
1. Ensure the DOM trader / chart trader panel is visible and not collapsed.
2. Ensure Tradovate is not minimized.
3. If using RDP, the session must be unlocked with an active GUI.
4. Check element names in `_ELEMENTS` dict — Tradovate UI updates may change button labels.
5. Use `openclaw.inspect()` or Windows Accessibility Insights to find correct element identifiers.

### 13.3 Duplicate Trades

**Symptom**: Same signal produces two fills.

**Unlikely** — the system has three layers of dedup:
1. `SignalRegistry` fingerprint-based dedup (120s cooldown)
2. `FailSafeState` cooldown timer (60s between clicks)
3. `FailSafeState` max open trades check

**If it happens**:
1. Check audit log for two `execution_confirmed` events.
2. Activate kill switch.
3. Manually close the duplicate position in Tradovate.
4. Review `duplicate_cooldown_seconds` and `cooldown_seconds` settings.

### 13.4 No Position Opened

**Symptom**: System reports `execution_confirmed` but no position appears.

**Possible causes**:
- Post-trade validation timed out (position took too long to appear in UI).
- ATM template rejected the order (invalid SL/TP for the instrument).
- Insufficient margin.

**Fix**:
1. Check Tradovate order history for rejected orders.
2. Verify ATM template is valid for the symbol/account.
3. Check margin requirements.
4. Increase `confirmation_timeout_seconds` if orders are slow to fill.

### 13.5 ATM Not Placing Stops

**Symptom**: Position opens but no SL/TP orders in the orders panel.

**Fix**:
1. Verify ATM template is **selected** (not just visible).
2. Re-create the ATM template.
3. Manually test by clicking Buy in Tradovate — do SL/TP appear?
4. Some ATM templates require a specific order type (Market vs Limit).

### 13.6 Stale Signals

**Symptom**: Many `signal_expired` events in audit log.

**Cause**: Signal TTL is 30 seconds by default. If the system takes too long to validate and click, the signal expires.

**Fix**:
1. Ensure Tradovate UI is responsive (not lagging).
2. Check for high system load.
3. Increase `DEFAULT_SIGNAL_TTL_SECONDS` in `execution/signal_schema.py` if needed (not recommended beyond 60s).

---

## 14 — Daily Workflow

### 14.1 Pre-Market Setup (Before 9:15 AM ET)

```
1. Start Tradovate desktop app
2. Log into the correct account
3. Open MNQ / MES chart tabs
4. Verify ATM template is selected
5. Verify no open positions or pending orders from yesterday
6. Start the Apex system:

   .\venv\Scripts\Activate.ps1
   .\venv\Scripts\python.exe scripts\run_live.py `
       --mode paper `
       --data data\mes_5m.csv `
       --dashboard `
       --analytics `
       --prop-mode

7. Open dashboard → http://127.0.0.1:8501
8. Verify system health: /api/health returns OK
9. Run a quick UI read check:
   .\venv\Scripts\python.exe -m execution.validation_harness --stage 1
10. Verify all UI fields read correctly
```

### 14.2 Market Open Monitoring (9:30 AM – 10:00 AM ET)

- Watch for the opening range to be established (first 15 minutes).
- Monitor the dashboard for first signals.
- Verify first signal looks reasonable (direction, price, SL, TP).
- If everything looks good and you're in live mode: let it run.

### 14.3 Mid-Day Monitoring (10:00 AM – 3:00 PM ET)

- Check dashboard every 15–30 minutes.
- Watch for:
  - Drawdown approaching limits
  - Unusual number of signals
  - Kill switch events
- Check `logs/errors.log` for any warnings.

### 14.4 End of Day (3:50 PM ET)

- The strategy forces flat at 3:50 PM ET (configurable via `eod_exit_time`).
- After EOD flat, verify all positions are closed.
- Review trades:

```powershell
Get-Content logs\trades.log
```

### 14.5 Post-Market Review

```
1. Check paper trades CSV:
   results/trades.csv

2. Review analytics report:
   Get-Content results/analytics.json | ConvertFrom-Json

3. Review daily PnL, win rate, max drawdown

4. If prop mode: check equity vs target and DD levels

5. Stop the system (Ctrl+C in the terminal)

6. Archive logs if needed:
   Copy-Item logs\* "logs\archive\$(Get-Date -Format 'yyyy-MM-dd')\" -Recurse
```

---

## 15 — Troubleshooting

### 15.1 Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'strategy'`

**Fix**: Run from the project root directory, or use the venv Python:

```powershell
cd C:\Users\andre\Desktop\Apex
.\venv\Scripts\python.exe scripts\run_live.py --mode paper --data data\mes_5m.csv
```

All scripts have a `sys.path` preamble that adds the project root, but it assumes the script is in its correct directory.

### 15.2 OpenClaw Not Working

**Symptom**: `OpenClaw is not installed. Install with: pip install openclaw`

**Fix**:

```powershell
.\venv\Scripts\pip.exe install openclaw
```

**Symptom**: OpenClaw installed but `window_found: False`

**Fixes**:
1. Ensure Tradovate is running (not just installed).
2. The window title must contain "Tradovate". Check `_ELEMENTS["window_title_pattern"]`.
3. If running as a different user or from a service, UI Automation may not have access to the target window.

### 15.3 Tradovate UI Mismatch

**Symptom**: UI elements read as `None` or wrong values.

**Fix**: Tradovate may have updated its UI. Use Accessibility Insights to inspect elements:

1. Download [Accessibility Insights for Windows](https://accessibilityinsights.io/docs/windows/overview/).
2. Run the live inspect tool and hover over Tradovate UI elements.
3. Note the `Name`, `Role`, `AutomationId` values.
4. Update `_ELEMENTS` in `execution/openclaw_driver.py` to match.

### 15.4 Execution Failures

**Symptom**: `on_signal()` returns `False` consistently.

**Debug steps**:

1. Check fail-safe status:
   ```python
   print(controller._fail_safe.status())
   ```
2. Check if kill switch is active.
3. Check if cooldown timer hasn't expired.
4. Check if session execution cap is hit.
5. Check audit log for specific rejection reason.

### 15.5 scikit-learn Version Mismatch

**Symptom**: `ModuleNotFoundError: No module named 'sklearn'` or model loading errors.

**Fix**: Ensure you're using the venv Python (not system Python):

```powershell
.\venv\Scripts\python.exe -c "import sklearn; print(sklearn.__version__)"
```

If the model was trained with a different sklearn version, retrain:

```powershell
.\venv\Scripts\python.exe pipeline\train_model.py
```

### 15.6 FastAPI / Dashboard Not Starting

**Symptom**: `No module named 'fastapi'`

**Fix**:

```powershell
.\venv\Scripts\pip.exe install fastapi uvicorn
```

### 15.7 Data File Not Found

**Symptom**: `Data file not found: data/mes_5m.csv`

**Fix**: Available data files are in the `data/` directory:

```powershell
Get-ChildItem data\*.csv | Select-Object Name, Length
```

Common files:
- `data/mes_4y.csv` — 4 years of MES data (2021–2024)
- `data/mnq_4y.csv` — 4 years of MNQ data
- `data/mes_2025.csv` — 2025 data
- `data/mnq_2025.csv` — 2025 data

---

## Appendix A — CLI Reference

```
.\venv\Scripts\python.exe scripts\run_live.py --help
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--mode` | (required) | `replay`, `paper`, or `live` |
| `--data` | `data/mes_5m.csv` | Path to OHLCV CSV |
| `--start` / `--end` | None | Date filters (YYYY-MM-DD) |
| `--dashboard` | off | Enable web dashboard |
| `--dashboard-port` | 8501 | Dashboard port |
| `--analytics` | off | Enable analytics collection |
| `--prop-mode` | off | Enable prop challenge mode |
| `--ml-threshold` | 0.6 | ML probability threshold |
| `--max-daily-loss` | 500 | Max daily loss ($) |
| `--max-trades-per-day` | 6 | Max trades per day |
| `--max-concurrent` | 3 | Max concurrent positions |
| `--multi-strategy` | off | Enable intraday strategies |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |
| `--expected-symbol` | (none) | Required symbol tab for pre-flight (e.g. MNQ) |
| `--expected-account` | (none) | Required account label substring for pre-flight |
| `--skip-preflight` | off | Skip pre-flight validation (not recommended) |

## Appendix B — File Structure Reference

```
Apex/
├── scripts/                   # Entry points
│   ├── run_live.py            # Main system runner (was main.py)
│   ├── run_backtest.py        # Backtesting entry point
│   ├── run_multi_symbol.py    # Multi-symbol paper trading
│   └── run_paper_live.py      # Paper live runner
├── strategy/                  # Strategy layer
│   ├── orb.py                 # Opening Range Breakout base
│   ├── hybrid_ema_ml.py       # Hybrid EMA + ML strategy
│   ├── adaptive_regime.py     # Adaptive regime breakout
│   ├── strategy_engine.py     # Live signal engine
│   ├── multi_strategy_engine.py
│   ├── intraday_strategies.py
│   ├── paper_engine.py        # Simulated execution
│   └── telegram_alerts.py     # Telegram notifications
├── execution/                 # Execution layer
│   ├── adapter.py             # ExecutionAdapter interface
│   ├── openclaw_adapter.py    # OpenClaw implementation
│   ├── openclaw_driver.py     # Low-level Tradovate UI driver
│   ├── execution_controller.py # Central orchestrator
│   ├── signal_schema.py       # ExecutionSignal, SignalRegistry
│   ├── validators.py          # Pre/post trade validation
│   ├── fail_safes.py          # Kill switch, cooldown, caps
│   ├── risk_bridge.py         # LiveSignal → ExecutionSignal
│   ├── audit_logger.py        # JSONL audit trail
│   ├── prop_sizing.py         # PropTradeGate for execution
│   ├── monitor.py             # Execution monitor state
│   ├── preflight.py           # Pre-flight startup validation
│   ├── validation_harness.py  # 5-stage validation
│   ├── tradovate_client.py    # Tradovate REST API client
│   └── tradovate_multi.py     # Multi-account Tradovate
├── risk/                      # Risk layer
│   ├── risk_manager.py        # Daily loss, trade limits
│   ├── prop_challenge.py      # Prop firm challenge logic
│   ├── prop_risk_layer.py     # Sizing + DD proximity
│   └── portfolio_risk.py      # Multi-symbol portfolio risk
├── data/                      # Data layer
│   ├── loader.py              # CSV/Parquet loader
│   ├── features.py            # Feature engineering
│   ├── labels.py              # Label generation
│   ├── splits.py              # Train/test splits
│   ├── *.csv                  # Historical bar data
│   └── ...
├── pipeline/                  # Data pipeline
│   ├── build_dataset.py       # Dataset builder
│   ├── train_model.py         # ML model training
│   ├── evaluate_model.py      # Model evaluation
│   └── ...
├── backtest/                  # Backtesting engine
│   ├── engine.py              # Bar-by-bar backtest
│   ├── metrics.py             # Trade metrics + CSV export
│   ├── benchmark.py           # Buy-and-hold benchmark
│   └── sweep.py               # Parameter sweep
├── backtests/                 # Backtest runners
│   ├── run_backtest.py        # Main backtest script
│   ├── optimize.py            # Optimization
│   └── ...
├── challenge/                 # Prop challenge simulation
│   ├── simulator.py           # Challenge Monte Carlo
│   ├── monte_carlo.py         # MC engine
│   └── dynamic_monte_carlo.py # Dynamic sizing MC
├── dashboard/                 # Web UI
│   ├── app.py                 # FastAPI app
│   ├── state.py               # DashboardState
│   └── static/                # HTML/JS frontend
├── analytics/                 # Analytics engine
│   └── engine.py              # AnalyticsEngine
├── config/                    # Configuration
│   └── settings.py            # All config dataclasses
├── research/                  # Research / analysis scripts
│   └── analyze_*.py           # One-off analysis scripts
├── tests/                     # Test suite (713+ tests)
├── models/                    # Trained models + results
├── logs/                      # Runtime logs
├── results/                   # Output results
└── docs/                      # Documentation
```

## Appendix C — Execution Layer Staged Rollout

Follow this exact sequence when moving from dry-run to live:

| Stage | Mode | What Happens | Risk |
|---|---|---|---|
| 0 | `DRY_RUN` | Process signals, validate, never click | Zero |
| 1 | `DRY_RUN` | Read live Tradovate UI state | Zero |
| 2 | `DRY_RUN` | Run canned scenarios through pipeline | Zero |
| 3 | `SIM` | Execute on sim account (real clicks, fake money) | Zero (sim) |
| 4 | `LIVE` | Execute on eval account, 1 contract max | Low |

**Never skip stages. Always verify each stage before advancing.**

See `execution/ROLLOUT_PLAN.md` for the detailed checklist.
