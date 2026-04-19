# Apex — Micro Futures Trading System

Algorithmic trading system for CME micro equity-index futures (MES, MNQ). Combines an adaptive regime-aware breakout strategy with prop-firm risk management, Monte Carlo challenge simulation, and automated execution via OpenClaw on Tradovate.

Built in Python. Validated across 8 years of historical data (2017–2025). Designed to survive a prop firm evaluation under realistic execution costs.

> Guiding principle: don't make the system look good — make it behave correctly in reality.

---

## Overview

Apex is a complete trading system stack:

- **Strategy layer** — Adaptive regime-aware EMA breakout strategy that detects market regimes (trending, ranging, dead) and adjusts entry criteria accordingly. No ML required for signal generation.
- **Risk layer** — Prop-aware risk management with dynamic position sizing, trailing drawdown enforcement, daily loss limits, and a kill switch.
- **Execution layer** — Automated trade placement on Tradovate via OpenClaw UI automation, with pre-flight validation, post-trade confirmation, continuous position reconciliation, and a real-time execution monitor.
- **Validation layer** — Multi-year backtesting, walk-forward evaluation, Monte Carlo challenge simulation, and time-filter analysis.

The system targets micro futures (MES, MNQ) because they offer meaningful price action with manageable per-contract risk — suitable for prop firm evaluations with $25K starting capital and $1,000 max drawdown constraints.

---

## Features

- **Adaptive regime strategy** — Regime detection (trending/ranging/dead) via ATR, EMA slope, and range-ratio metrics. Adjustable confirmation scoring for long and short entries.
- **Prop-aware risk layer** — Dynamic position sizing profiles for challenge vs. funded modes. Configurable drawdown caution zones, streak reduction, profit locks, and giveback halts.
- **OpenClaw / Tradovate execution** — Automated UI-driven trade placement with pre-trade and post-trade validation.
- **Pre-flight validation** — 9-point system check before any live session (window, account, symbol, ATM template, positions, buttons, kill switch, read errors).
- **Continuous reconciliation** — Background thread polling Tradovate positions every 2–5 seconds, comparing against internal state, activating kill switch on mismatch.
- **Execution monitor** — Thread-safe state collector with classified alerts (INFO/WARNING/CRITICAL) and dashboard-ready snapshots.
- **Monte Carlo challenge simulation** — Block-sampled simulation (2,000 paths) across multiple sizing profiles to estimate prop challenge pass rates.
- **Multi-year backtesting** — Validated across 8 individual years (2017–2025) with per-symbol, per-year granularity.
- **Time-filter analysis** — Entry-time bucketing to identify and remove negative-edge time windows.

---

## Architecture

```
Market Data (CSV / Databento)
        │
        ▼
┌─────────────────┐
│ AdaptiveRegime   │  Regime detection + EMA breakout signals
│ Strategy         │  (no ML dependency)
└────────┬────────┘
         │ LiveSignal
         ▼
┌─────────────────┐
│ Risk Manager     │  Daily loss limits, position caps, kill switch
│ + PropRiskLayer  │  Dynamic sizing (challenge / funded profiles)
│ + PortfolioRisk  │  Cross-symbol exposure limits
└────────┬────────┘
         │ Approved signal
         ▼
┌─────────────────┐
│ Execution        │  OpenClaw → Tradovate UI automation
│ Controller       │  Pre-trade validation → Click → Post-trade confirmation
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
Audit Log   Reconciliation Loop
(JSON)      (position monitoring, kill switch on mismatch)
```

---

## Validated Results

All results below are from backtests with realistic execution costs: 1 tick slippage per side, $2.25 commission per side, integer contract sizing, $25,000 starting capital per year. **These are not live trading results.**

### Multi-Year Backtest (AdaptiveRegimeStrategy, 2017–2025)

Strategy: Adaptive regime-aware EMA breakout, pure defaults, no ML, no optimization. Stop-order entry at opening-range breakout. MES + MNQ combined.

| Year | Trades | Combined PnL | Best Symbol |
|------|--------|-------------|-------------|
| 2017 | 207 | −$545 | MES |
| 2018 | 224 | −$2,694 | MES |
| 2019 | 217 | +$752 | MES |
| 2021 | 262 | +$7,282 | MNQ |
| 2022 | 291 | +$2,418 | MES |
| 2023 | 219 | +$1,544 | MNQ |
| 2024 | 219 | −$716 | MNQ |
| 2025 | 268 | +$9,189 | MNQ |

| Aggregate Metric | Value |
|-----------------|-------|
| Total trades | 1,907 |
| Profitable years | 5 / 8 (62%) |
| Cumulative PnL | +$17,231 |
| Average annual PnL | +$2,154 |
| Worst single-year drawdown | $3,001 (MNQ 2022) |

**After time-filter refinement** (MNQ `max_entry_time` tightened to 10:30):

| Metric | Before | After |
|--------|--------|-------|
| Cumulative PnL | $20,547 | $21,716 |
| Total trades | 1,904 | 1,901 |
| Profitable years | 5/8 | 5/8 |
| Worst year (2018) | −$2,071 | −$1,610 |
| 2024 (was negative) | −$325 | +$284 |

The time filter removes 3 trades total and improves cumulative PnL by +$1,169 — a minimal, conservative change.

### Prop Challenge Simulation (Monte Carlo)

2,000 simulated challenge paths per scenario. $25,000 starting capital, $1,500 profit target, $1,000 max trailing drawdown. Block-sampled (block_size=7) from real backtest trades.

| Scenario | Source Trades | Pass Rate | Fail Rate | Avg Trades to Pass | Mean Max DD |
|----------|-------------|-----------|-----------|-------------------|-------------|
| Combined baseline (1× size) | 1,901 | 40.7% | 59.3% | 40 | $875 |
| Combined 0.35× fixed | 1,901 | 63.5% | 32.7% | 221 | $748 |
| Combined dynamic challenge | 1,901 | 63.5% | 32.7% | 221 | $748 |
| Combined dynamic extended | 1,901 | 64.2% | 35.8% | 230 | $754 |
| MES baseline | 590 | 61.6% | 38.4% | 118 | $804 |
| MES dynamic challenge | 590 | 69.2% | 4.8% | 286 | $608 |
| MES dynamic extended | 590 | 89.9% | 9.7% | 400 | $633 |
| MNQ baseline | 1,311 | 40.5% | 59.5% | 40 | $889 |
| MNQ dynamic challenge | 1,311 | 68.1% | 28.6% | 214 | $723 |
| MNQ dynamic extended | 1,311 | 73.9% | 26.1% | 220 | $703 |

**Key insight:** Position sizing profile matters more than signal refinement for challenge passing. Reducing from 1× to 0.35× sizing cuts the fail rate from 59% to 33% while improving the pass rate from 41% to 64%. The trade-off is longer time-to-pass (~221 trades vs. ~40).

MES with dynamic extended sizing achieves the highest simulated pass rate (89.9%) due to its lower per-trade variance, though with slower capital deployment (400 avg trades to pass).

### Funded Mode (Simulation)

With 1× sizing and $1,000 max drawdown (no profit target), 0% of simulations reach a meaningful profit target — 78.8% fail, 21.2% are incomplete after 500 trades. Mean max drawdown: $1,003. This confirms that funded mode requires conservative sizing (0.35–0.75×) to survive.

### Current Observations

- MNQ generates the majority of PnL in volatile years (2021, 2025) but has higher variance.
- MES is more consistent but lower-returning; it produces a better challenge simulation profile due to lower per-trade risk.
- The strategy has real losing years (2017, 2018, 2024) — this is expected for a non-curve-fit approach.
- The 0.35× challenge sizing profile is the current recommended configuration for prop challenge attempts.

---

## Current Status

| Component | Status |
|-----------|--------|
| Adaptive regime strategy | Validated — 8 years, 2 symbols |
| Backtesting engine | Production — per-year, per-symbol granularity |
| Prop risk layer | Validated — challenge + funded profiles |
| Monte Carlo simulator | Production — 2,000 sims, block sampling |
| Time-filter analysis | Validated — MNQ 10:30 cutoff applied |
| Execution controller | Built — pre-trade, click, post-trade, reconciliation |
| Pre-flight validation | Built — 9-point check |
| Reconciliation loop | Built — continuous position monitoring |
| Execution monitor | Built — thread-safe alert classification |
| OpenClaw driver | Built — Tradovate UI automation |
| Dashboard | Functional — FastAPI + Chart.js |
| Test suite | 778 tests across 25 modules |
| **Live capital** | **Not deployed — system is in validation phase** |

The system has production-grade execution infrastructure but has not been used with real capital. Current focus is stable execution validation in dry-run / SIM mode before any live deployment.

---

## Repository Structure

```
apex/
├── strategy/              # Signal generation and trade management
│   ├── adaptive_regime.py # Regime-aware EMA breakout strategy
│   ├── strategy_engine.py # Bar-by-bar signal engine
│   ├── paper_engine.py    # Simulated execution with realistic costs
│   ├── risk_manager.py    # Per-symbol risk limits and kill switch
│   ├── portfolio_risk.py  # Multi-symbol exposure coordination
│   ├── prop_risk_layer.py # Dynamic sizing for prop challenges
│   └── prop_challenge.py  # Prop firm evaluation rule enforcement
├── execution/             # Automated trade placement
│   ├── execution_controller.py  # Signal → validation → click → confirm
│   ├── openclaw_adapter.py      # Adapter bridging strategy to OpenClaw
│   ├── openclaw_driver.py       # Low-level Tradovate UI interaction
│   ├── reconciliation.py        # Continuous position reconciliation loop
│   ├── preflight.py             # Pre-session system validation
│   ├── monitor.py               # Real-time execution state and alerts
│   ├── fail_safes.py            # Kill switch, cooldown, dry-run modes
│   ├── validators.py            # Pre-trade and post-trade checks
│   ├── audit_logger.py          # Structured JSON audit trail
│   └── risk_bridge.py           # PropRiskLayer ↔ execution bridge
├── backtest/              # Backtesting infrastructure
│   ├── engine.py          # Bar-by-bar backtest engine
│   ├── metrics.py         # Trade metrics, CSV/JSON export
│   ├── benchmark.py       # Benchmarking (always-long, flat, etc.)
│   └── sweep.py           # Parameter sweep engine
├── backtests/             # Validation and analysis scripts
│   ├── run_adaptive_regime_validation.py
│   ├── run_time_filter_validation.py
│   ├── run_prop_challenge_simulation.py
│   ├── run_prop_risk_layer_validation.py
│   └── ...
├── risk/                  # Risk management modules
├── challenge/             # Prop challenge simulation
│   ├── simulator.py       # Challenge rule engine
│   ├── monte_carlo.py     # Monte Carlo simulation
│   └── dynamic_monte_carlo.py
├── data/                  # Data loading and feature engineering
│   ├── loader.py          # CSV data loading
│   ├── features.py        # Feature engineering
│   └── databento_fetcher.py
├── models/                # ML training and evaluation
├── pipeline/              # Data → train → evaluate pipeline
├── analytics/             # Post-trade analytics
├── dashboard/             # FastAPI + Chart.js monitoring UI
├── scripts/               # Runner scripts
│   ├── run_live.py        # Live/paper mode runner
│   ├── run_paper.py       # Paper trading runner
│   └── run_pipeline.py    # Full pipeline runner
├── research/              # Analysis and investigation scripts
├── results/               # Validation output (JSON summaries)
├── docs/
│   └── OPERATIONS_GUIDE.md
├── tests/                 # 778 tests
├── main.py                # Primary entry point
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- Windows (required for OpenClaw / Tradovate execution)

### Installation

```bash
git clone https://github.com/AndrewMahran7/ApexBot.git
cd ApexBot
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in credentials:

```bash
copy .env.example .env
```

Required for specific features:
- `DATABENTO_API_KEY` — historical data fetching
- `TRADOVATE_*` — Tradovate SIM execution
- `TELEGRAM_*` — real-time trade alerts

### Data

Historical 5-minute bar CSVs are not included (too large). Place them in `data/`:
- `data/mes_*.csv`, `data/mnq_*.csv` (per-year files)
- Fetch via Databento: `python -m pipeline.fetch_data`

---

## Usage

### Run tests

```bash
python -m pytest tests/ -q
```

### Run multi-year backtest validation

```bash
python -m backtests.run_adaptive_regime_validation
```

### Run prop challenge Monte Carlo simulation

```bash
python -m backtests.run_prop_challenge_simulation
```

### Run prop risk layer validation

```bash
python -m backtests.run_prop_risk_layer_validation
```

### Run in dry-run mode (no clicks, full validation)

```bash
python -m scripts.run_live --mode dry_run
```

### Run in SIM mode

```bash
python -m scripts.run_live --mode sim
```

---

## Documentation

- [Operations Guide](docs/OPERATIONS_GUIDE.md) — Pre-flight checks, run modes, monitoring, incident response
- [Execution Rollout Plan](execution/ROLLOUT_PLAN.md) — Staged deployment from dry-run → SIM → live

---

## Safety

- **Always start in dry-run mode.** Verify all pre-flight checks pass before enabling SIM or live execution.
- **Verify Tradovate UI layout.** OpenClaw depends on specific UI element positions. If Tradovate updates its layout, the driver may need adjustment.
- **Pre-flight validation is mandatory.** The system blocks execution if any of the 9 pre-flight checks fail.
- **Kill switch is automatic.** Triggered by daily loss limits, position anomalies, reconciliation mismatches, or persistent UI read failures.
- **Reconciliation runs continuously.** If the background loop detects a position mismatch with internal state, it activates the kill switch immediately.
- **No guarantees of profitability.** Backtest results show real losing years. Challenge simulation pass rates are probabilistic, not deterministic.

---

## Disclaimer

This software is provided for educational and research purposes only. It is **not financial advice**.

- All performance metrics shown are from backtests or Monte Carlo simulations — **no real capital has been traded with this system**.
- Past backtest performance does not guarantee future results. The strategy has real losing years in its validation history.
- Futures trading involves substantial risk of loss. Micro futures (MES, MNQ) carry real financial risk despite lower notional values.
- Monte Carlo pass rates are statistical estimates based on historical trade distributions, not predictions of future challenge outcomes.
- The author is not a registered investment advisor.

**Use at your own risk.**
