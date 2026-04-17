# Apex — Micro Futures Trading System

A production-oriented algorithmic trading system for CME micro equity-index futures (MES, MNQ), built from scratch in Python. Combines EMA-based breakout signals with ML-driven trade filtering, multi-symbol portfolio management, and realistic tick-based execution simulation.

**Current phase: live validation via paper trading + Telegram alerts + manual execution.**

> Guiding principle: don't make the system look good — make it behave correctly in reality.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Strategy Overview](#strategy-overview)
- [Edge Thesis](#edge-thesis)
- [System Architecture](#system-architecture)
- [Current System Status](#current-system-status)
- [Backtesting Results](#backtesting-results)
- [365-Day Validated Results](#365-day-validated-results)
- [Monte Carlo & Robustness](#monte-carlo--robustness)
- [Live Validation](#live-validation-paper-trading)
- [Risk Management](#risk-management)
- [Realistic Futures Execution](#realistic-futures-execution)
- [Execution Workflow](#execution-workflow)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Design Principles](#design-principles)
- [Disclaimer](#disclaimer)

---

## Why This Exists

Most retail algo-trading projects skip the hard parts: realistic execution costs, integer contract sizing, proper drawdown tracking, and honest out-of-sample validation. Apex was built to confront those problems directly.

The goal is a futures trading system that can survive a prop firm evaluation — not by curve-fitting past data, but by maintaining a small, genuine statistical edge under realistic conditions.

---

## Strategy Overview

**Core signal:** EMA breakout during the opening range (9:30–9:45 ET). Price breaks above/below the opening range *and* the EMA line → entry signal.

**ML filter:** A gradient-boosted classifier (trained on 36 engineered features) scores each candidate trade. ML does **not** generate signals — it only filters them. Trades below the ML threshold are rejected.

**Trade management:**
- Stop loss: opening-range low/high (direction-dependent)
- Take profit: reward-risk multiple (default 1.5×)
- End-of-day exit: all positions closed by 15:50 ET

**EMA configurations:** 20, 50, 100-period EMAs evaluated independently. Multi-candidate mode ranks by ML probability and enters the top candidates per day.

---

## Edge Thesis

The edge is structural, not predictive:

1. **Opening-range breakouts** capture the highest-volatility window of the session
2. **EMA confluence** filters for trades aligned with the intraday trend
3. **ML filtering** (walk-forward AUC ≈ 0.61) removes the worst 40–50% of candidates — not by predicting winners, but by rejecting likely losers
4. **Disciplined sizing** (1% risk per trade, integer contracts) keeps drawdowns survivable
5. **Multi-symbol diversification** (MES + MNQ) reduces single-instrument concentration

This is a modest edge. It does not compound into exponential returns. It aims to generate small, consistent profits with controlled risk — enough to pass a prop firm challenge.

---

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Historical  │────▶│  Strategy    │────▶│  Risk        │────▶│  Paper       │
│  Bar Feed    │     │  Engine      │     │  Manager     │     │  Engine      │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                           │                     │                     │
                    ┌──────┘              ┌──────┘              ┌──────┘
                    ▼                     ▼                     ▼
               LiveSignal            RiskEvent             PnLUpdate
               (entry/exit)          (approve/block)       (equity/DD)
                                          │
                                          ▼
                                  ┌──────────────┐
                                  │  Portfolio    │
                                  │  Risk Mgr    │◀── multi-symbol coordination
                                  └──────────────┘

  Optional layers:
  ├── PropChallenge gate (prop firm rules)
  ├── Telegram alerts (real-time notifications)
  ├── Analytics engine (post-trade analysis)
  ├── Dashboard (FastAPI + Chart.js)
  └── Tradovate client (live execution — SIM only)
```

**Signal flow per bar:**
1. `StrategyEngine.on_bar()` → generates `LiveSignal` (entry or exit)
2. `PortfolioRiskManager` → cross-symbol exposure check
3. `RiskManager.on_signal()` → daily loss limits, position caps, kill switch
4. `PaperEngine.on_signal()` → simulates fill with slippage + commission
5. `PnLUpdate` emitted → equity curve, drawdown tracking

---

## Current System Status

| Component | Status |
|---|---|
| Backtesting engine | Production — validated across 4 years of MES data |
| ML model (gradient boosting) | Trained — walk-forward AUC 0.61, 5-fold expanding window |
| Paper trading engine | Production — tick-based PnL, integer contracts |
| Multi-symbol portfolio | Production — MES + MNQ with shared risk limits |
| Risk management | Production — daily loss limits, kill switch, position caps |
| Prop challenge mode | Production — trailing DD, daily profit lock, staged sizing |
| Monte Carlo simulator | Production — block sampling, sensitivity analysis |
| Telegram alerts | Active — real-time trade notifications |
| Tradovate integration | SIM-only — order placement, fill tracking |
| Dashboard | Functional — FastAPI + Chart.js equity display |
| **Live trading** | **Not yet — paper validation in progress** |

**Test coverage:** 528 tests passing across 17 test modules.

---

## Backtesting Results

**Dataset:** MES 5-minute bars, Jan 2021 – Dec 2024 (~4 years).

### Multi-Strategy Backtest (analytics engine, single-symbol MES)

| Metric | Value |
|---|---|
| Total trades | 1,172 |
| Win rate | 43.9% |
| Profit factor | 1.25 |
| Sharpe ratio | 1.31 |
| Total PnL | $1,517 |
| Max drawdown | −$339 |
| Trading days | 211 |

### By Strategy Type

| Strategy | Trades | Win Rate | PnL | Profit Factor |
|---|---|---|---|---|
| ema50_breakout | 119 | 73.1% | $1,610 | 4.04 |
| ema100_breakout | 130 | 55.4% | $1,023 | 2.18 |
| ema20_breakout | 49 | 57.1% | $538 | 2.44 |
| intraday_momentum | 468 | 37.6% | −$730 | 0.66 |
| mean_reversion | 159 | 34.6% | −$554 | 0.41 |
| vwap_bounce | 238 | 40.3% | −$316 | 0.74 |

> **Note:** EMA breakout strategies carry the system. Intraday momentum, mean reversion, and VWAP bounce are net-negative — they remain in the codebase for research but are not used in the primary trading configuration.

### ML-Filtered Backtest (3,045 trades, all EMA configs)

| Metric | Value |
|---|---|
| Total trades | 3,045 |
| Win rate | 38.6% |
| Total PnL | $21,551 |
| Profit factor | 1.24 |
| Sharpe ratio | 1.51 |
| Max drawdown | −$6,069 (12.1%) |
| Initial capital | $25,000 |
| Final equity | $46,551 |
| Return | 86.2% |

### ML Model Performance (Walk-Forward)

| Metric | Value |
|---|---|
| Walk-forward folds | 5 (expanding window) |
| Aggregate AUC | 0.607 |
| Mean fold AUC | 0.593 ± 0.038 |
| Out-of-sample accuracy | 63.8% |
| Total OOS samples | 814 |
| Training period | 2021-01-04 → 2023-05-25 |
| Holdout period | 2024-08-09 → 2024-12-30 |
| Features | 36 |

The ML model provides modest but genuine signal. It is used as a filter (reject low-probability trades), not as the primary signal generator.

---

## 365-Day Validated Results

The primary validation run: 365 trading days, multi-symbol (MES + MNQ), EMA breakout strategy with ML filtering, realistic execution costs.

**These are paper-traded results with simulated tick-based execution — not live capital.**

### Portfolio Summary

| Metric | Value |
|---|---|
| Symbols | MES, MNQ |
| Trading days | 365 |
| Total trades | 237 |
| Trades/day | 0.6 |
| Win rate | 42.6% |
| Avg PnL/trade | $15.25 |
| Total PnL | $3,613.19 |
| Portfolio equity | $13,613.19 |
| Peak equity | $14,485.81 |
| Max drawdown | −$872.62 |
| Strategy | ema50_breakout |
| Status | **STABLE** |

### By Symbol

| Symbol | Trades | Win Rate | Total PnL | Avg PnL | Max Drawdown |
|---|---|---|---|---|---|
| MES | 95 | 38.9% | −$121.59 | −$1.28 | −$373.76 |
| MNQ | 142 | 45.1% | $3,734.77 | $26.30 | −$498.86 |

### Portfolio Risk Events

| Event Type | Count |
|---|---|
| portfolio_corr_reduced | 54 |

> **Key observations:**
> - MNQ carries the portfolio — MES is roughly flat over the validation period
> - The 54 correlation-reduced events indicate the portfolio risk manager is actively limiting correlated entries across MES and MNQ
> - Drawdown stayed within prop firm tolerance (~$873 peak-to-trough)
> - Trade frequency is low (0.6/day) — the system is selective

---

## Monte Carlo & Robustness

Monte Carlo simulation stress-tests the trade distribution to estimate prop challenge pass rates and worst-case drawdowns.

### Baseline Sensitivity (full trade set, 3,045 trades)

| Position Scale | Pass Rate | Fail Rate | Avg Max DD |
|---|---|---|---|
| 0.25× | 0.0% | — | — |
| 0.50× | 11.5% | — | — |
| 0.75× | 39.6% | — | — |
| **1.00×** | **55.8%** | **23.6%** | — |
| 1.25× | 59.9% | — | — |
| 1.50× | 57.0% | — | — |
| 2.00× | 51.1% | — | — |

*2,000 simulations per scale. Prop challenge: $25,000 starting capital, $1,500 profit target, $1,000 max trailing drawdown.*

### Filtered Strategy (EMA breakout only, 263 trades, block_size=5)

| Metric | Value |
|---|---|
| Pass rate | 63.8% |
| Fail rate | 0.0% |
| Avg trades to pass | 240 |
| Source win rate | 55.1% |
| Source mean PnL | $5.66 |

### Multi-Symbol Monte Carlo (37 trades)

| Metric | Value |
|---|---|
| Pass rate | 98.7% |
| Fail rate | 0.2% |
| Avg trades to pass | 85 |
| Source win rate | 46.0% |
| Source mean PnL | $14.75 |

> **Interpretation:** Filtering to EMA breakout strategies and diversifying across symbols dramatically improves the simulated pass rate. The multi-symbol configuration shows the strongest Monte Carlo profile. These are simulated distributions, not guaranteed outcomes.

---

## Live Validation (Paper Trading)

**Status:** Paper trading with real-time Telegram alerts. No real capital deployed yet.

Paper trading replays historical bars through the full execution pipeline (strategy → risk manager → paper engine) with realistic slippage, commission, and integer contract sizing. Results are compared against manual Tradovate SIM fills to validate execution fidelity.

The [365-day validated results](#365-day-validated-results) above represent the most comprehensive paper validation completed to date.

**Important:** These are simulated results. They do not account for real-world factors like partial fills, market impact, latency, or data feed interruptions. Do not treat this as proven live profitability.

---

## Risk Management

### Per-Trade Risk

| Parameter | Default | Description |
|---|---|---|
| `risk_per_trade` | 1% | Percentage of equity risked per trade |
| `max_contracts` | 5 | Hard cap on contracts per trade |
| `slippage_ticks` | 1 | Assumed slippage per side |
| `commission_per_side` | $0.62 | Per-contract per-side commission |

### Daily Risk Limits

| Parameter | Default | Description |
|---|---|---|
| `max_daily_loss` | $500 | Kill switch triggers at this loss |
| `max_trades_per_day` | 6 | Hard cap on entries |
| `max_concurrent_positions` | 3 | Open position limit |
| `max_per_direction` | 2 | Max same-direction positions |
| `kill_switch` | Auto | Blocks all entries after breach, force-closes positions |

### Portfolio Risk (Multi-Symbol)

| Parameter | Default | Description |
|---|---|---|
| `max_total_concurrent` | 3 | Total positions across all symbols |
| `max_same_direction` | 2 | Cross-symbol directional limit |
| `max_total_exposure` | 3.0 | Aggregate exposure cap |
| `correlation_divisor` | 2.0 | Size reduction for correlated entries |

### Prop Challenge Mode

Optional mode that enforces prop firm evaluation rules:

| Parameter | Default |
|---|---|
| Starting capital | $25,000 |
| Profit target | $1,500 |
| Max drawdown (trailing) | $1,000 |
| Daily loss limit | $300 |
| Daily profit lock | $400 |
| Max consecutive losses | 5 → kill switch |

---

## Realistic Futures Execution

A core design goal: execution math must match what happens with a real broker.

### Contract Specifications

| Instrument | Tick Size | Point Value | Tick Value | Description |
|---|---|---|---|---|
| MES | 0.25 | $5.00 | **$1.25** | Micro E-mini S&P 500 |
| MNQ | 0.25 | $2.00 | **$0.50** | Micro E-mini Nasdaq-100 |
| RTY | 0.10 | $5.00 | **$0.50** | Micro E-mini Russell 2000 |

### PnL Calculation

```
ticks = (exit_price - entry_price) / tick_size    # signed
pnl_dollars = ticks × tick_value × contracts      # long
pnl_dollars = -ticks × tick_value × contracts     # short
```

### Contract Sizing (Risk-Based)

```
risk_dollars = equity × risk_per_trade             # default 1%
contracts = floor(risk_dollars / (stop_ticks × tick_value))
contracts = clamp(1, max_contracts)                # integer only
```

### Costs

| Cost | Default | Calculation |
|---|---|---|
| Slippage | 1 tick per side | `slippage_ticks × tick_value × contracts` |
| Commission | $0.62 per side | `commission_per_side × contracts` |

All costs are per-contract and applied at both entry and exit. No fractional contracts anywhere in the system.

---

## Execution Workflow

Current production workflow:

1. Paper engine runs against historical bar data
2. Telegram bot sends real-time entry/exit alerts with contract details
3. Human operator manually places orders on Tradovate SIM
4. Results compared between paper engine output and actual fills
5. Discrepancies logged and investigated

Future (when validated): direct Tradovate SIM execution via API, then transition to live.

---

## Repository Structure

```
apex/
├── config/
│   └── settings.py              # InstrumentConfig, INSTRUMENT_REGISTRY, compute_contracts
├── strategy/
│   ├── hybrid_ema_ml.py         # Core EMA + ML strategy logic
│   ├── strategy_engine.py       # Live-compatible bar-by-bar wrapper
│   ├── paper_engine.py          # Paper trading execution (tick-based PnL)
│   ├── risk_manager.py          # Per-symbol risk limits and kill switch
│   ├── portfolio_risk.py        # Multi-symbol portfolio risk coordination
│   ├── prop_challenge.py        # Prop firm evaluation rules
│   ├── tradovate_client.py      # Tradovate API client (SIM only)
│   ├── tradovate_multi.py       # Multi-symbol Tradovate adapter
│   ├── telegram_alerts.py       # Telegram bot notifications
│   ├── orb.py                   # Opening range breakout signal logic
│   ├── adaptive_regime.py       # Market regime detection
│   ├── regimes.py               # Market regime classification
│   ├── features.py              # Strategy-level feature engineering
│   ├── intraday_strategies.py   # Additional intraday strategies (research)
│   └── multi_strategy_engine.py # Multi-strategy coordinator
├── backtest/
│   ├── engine.py                # Bar-by-bar backtesting engine
│   ├── metrics.py               # Trade metrics, CSV/JSON export, plotting
│   ├── benchmark.py             # Strategy benchmarking (always-long, flat, etc.)
│   └── sweep.py                 # Parameter sweep engine
├── data/
│   ├── loader.py                # CSV data loading
│   ├── features.py              # Feature engineering (36 features)
│   ├── labels.py                # Label generation
│   ├── splits.py                # Time-based train/test/validation/holdout splits
│   ├── storage.py               # SQLite bar storage
│   ├── replay.py                # Bar replay engine
│   ├── bar_api.py               # Bar API interface
│   ├── data_pipeline.py         # Data pipeline orchestration
│   ├── build_dataset.py         # Dataset construction
│   ├── validate.py              # Data validation
│   ├── ema_candidates.py        # EMA candidate generation
│   ├── inspect_data.py          # Data inspection utilities
│   └── databento_fetcher.py     # Databento market data client
├── models/
│   ├── train_model.py           # ML training pipeline
│   ├── evaluate_model.py        # Model evaluation
│   ├── ema_model.pkl            # Trained gradient boosting model
│   └── *.json                   # Training metrics, walk-forward results
├── analytics/
│   └── engine.py                # Post-trade analytics and reporting
├── challenge/
│   ├── simulator.py             # Prop challenge simulator
│   └── monte_carlo.py           # Monte Carlo simulation engine
├── dashboard/
│   ├── app.py                   # FastAPI dashboard
│   └── state.py                 # Dashboard state management
├── tests/                       # 528 tests across 17 modules
├── main.py                      # Primary system runner (replay/paper/live)
├── run_paper_live.py            # Paper trading runner with Telegram
├── run_multi_symbol.py          # Multi-symbol paper trading runner
├── run_backtest.py              # Backtesting CLI
├── walk_forward.py              # Walk-forward ML evaluation
├── optimize.py                  # Parameter optimization
├── train_model.py               # ML training entry point
├── evaluate_model.py            # Model evaluation entry point
├── build_dataset.py             # Dataset building entry point
├── split_dataset.py             # Dataset splitting entry point
├── fetch_data.py                # Market data fetching (Databento)
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/AndrewMahran7/ApexBot.git
cd ApexBot
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables (optional)

Create a `.env` file for API integrations:

```
DATABENTO_API_KEY=your_key
TRADOVATE_USERNAME=your_user
TRADOVATE_PASSWORD=your_pass
TRADOVATE_CID=your_cid
TRADOVATE_SECRET=your_secret
TRADOVATE_APP_ID=your_app_id
TRADOVATE_DEVICE_ID=your_device_id
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Data

Historical data files (CSV, 5-minute bars) are not included in the repository due to size. Place them in `data/`:

- `data/mes_4y.csv` — MES, 2021–2024
- `data/mnq_4y.csv` — MNQ, 2021–2024

Data can be fetched via Databento:
```bash
python fetch_data.py
```

---

## How to Run

### Backtest (single symbol)

```bash
python run_backtest.py --data data/mes_4y.csv --strategy hybrid_ema_ml --ml-threshold 0.55
```

### Paper trading (single symbol)

```bash
python run_paper_live.py --data data/mes_4y.csv --days 30 --bar-delay 0
```

### Paper trading (multi-symbol)

```bash
python run_paper_live.py --symbols MES MNQ \
    --data-MES data/mes_4y.csv \
    --data-MNQ data/mnq_4y.csv \
    --days 30 --bar-delay 0
```

### With prop challenge mode

```bash
python run_paper_live.py --data data/mes_4y.csv --days 30 --prop-mode --bar-delay 0
```

### Monte Carlo simulation

```bash
python -m challenge.monte_carlo --data results/trades.csv --sims 2000 --sensitivity
```

### Walk-forward evaluation

```bash
python walk_forward.py
```

### Run tests

```bash
python -m pytest tests/ --ignore=tests/test_dashboard.py -q
```

---

## Key Scripts

| Script | Purpose |
|---|---|
| `run_paper_live.py` | Primary paper trading runner. Single or multi-symbol, with Telegram alerts. |
| `run_multi_symbol.py` | Multi-symbol runner with portfolio-level risk management. |
| `main.py` | Core system runner. Supports replay, paper, and live modes. |
| `run_backtest.py` | Historical backtesting with split support (train/test/holdout). |
| `walk_forward.py` | Expanding-window walk-forward ML evaluation. |
| `optimize.py` | Grid search parameter optimization. |
| `train_model.py` | ML training pipeline (gradient boosting). |
| `evaluate_model.py` | Model evaluation with leakage checks. |
| `fetch_data.py` | Fetch historical bar data via Databento. |
| `test_telegram.py` | Verify Telegram bot connectivity. |

---

## Current Limitations

- **No live capital deployed.** All results are paper-traded or backtested.
- **MES is roughly flat.** Over the 365-day validation, MNQ carries the portfolio; MES alone is slightly negative.
- **Single strategy family.** Only EMA breakout strategies are profitable; momentum, mean reversion, and VWAP bounce are net-negative.
- **MNQ data quality.** Historical MNQ data is mapped from NQ; micro-specific microstructure effects are not captured.
- **ML model is modest.** Walk-forward AUC of 0.61 provides filtering value but is not strongly predictive.
- **No real-time data feed.** Paper trading replays historical bars; it does not connect to a streaming feed.
- **Tradovate integration is SIM-only.** Live order routing has a safety guard and has not been tested with real capital.
- **No portfolio optimization.** Capital is split equally across symbols — no Kelly criterion or mean-variance allocation.
- **Low trade frequency.** 0.6 trades/day means slow validation and slow capital deployment.

---

## Roadmap

- [ ] Complete extended paper validation with Telegram-alerted manual execution
- [ ] Automated Tradovate SIM execution (remove human from the loop)
- [ ] Real-time data feed integration (Databento live or Tradovate WebSocket)
- [ ] Per-strategy position sizing (scale to strategy-level profit factor)
- [ ] RTY (Micro Russell 2000) as a third symbol
- [ ] Systematic walk-forward model retraining schedule
- [ ] Portfolio-level Kelly sizing
- [ ] Investigate MES underperformance relative to MNQ

---

## Design Principles

1. **Don't assume anything works unless explicitly tested.** 528 tests cover every layer.
2. **Keep modules small and testable.** Each component has a single responsibility.
3. **Log every critical step.** Every entry, exit, risk decision, and state change is logged.
4. **Avoid hidden state.** Configuration is explicit; no module-level side effects.
5. **Make everything reproducible.** Same data + same config = same results.
6. **No silent failures.** Every error path logs or raises — nothing is swallowed.

---

## Disclaimer

This software is provided for educational and research purposes. It is **not financial advice**.

- Past performance (backtested or paper-traded) does not guarantee future results.
- All metrics shown are from backtests or paper simulations — **no real capital has been traded**.
- Futures trading involves substantial risk of loss. Micro futures (MES, MNQ) have lower notional value but still carry real financial risk.
- The ML model provides modest filtering value (AUC ≈ 0.61) — it is not a reliable standalone predictor.
- The 365-day validated results are from simulated paper execution, not live trading.
- The author is not a registered investment advisor.

Use at your own risk.

---

## License

MIT License. See [LICENSE](LICENSE).
