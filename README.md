# Apex — MES Futures Trading System

A complete **Micro E-mini S&P 500 (MES)** futures trading system: backtesting, ML pipeline, live strategy engine, paper trading, risk management, Tradovate execution, real-time monitoring, analytics, and prop firm challenge simulation.

Supports **multiple strategies** with both long and short entries (shorts enabled via `--shorts` flag):

| Strategy | Module | Description |
|----------|--------|-------------|
| **ORB** | `strategy/orb.py` | Opening Range Breakout — trade breakouts above/below the opening range |
| **Adaptive Regime** | `strategy/adaptive_regime.py` | Regime-aware breakout/continuation — classifies market conditions and only trades in favorable regimes |
| **Hybrid EMA-ML** | `strategy/hybrid_ema_ml.py` | Multi-candidate EMA strategy with ML ranking — multiple EMA lengths × entry types with configurable selection strategies |

> **Origin**: Refactored from a live TSLA ORB trading system (Alpaca + Yahoo Finance). All live-trading, stock-specific, and external-API dependencies have been removed.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                            │
│         Modes: replay │ paper │ live    + optional flags        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Data (CSV/Parquet)                                             │
│       │                                                         │
│       ▼                                                         │
│  StrategyEngine          on_bar(bar) → list[LiveSignal]         │
│       │                                                         │
│       ▼                                                         │
│  [PropRiskGate]          --prop-mode (optional challenge gate)  │
│       │                                                         │
│       ▼                                                         │
│  RiskManager             pre-trade risk gate + kill switch      │
│       │                                                         │
│       ▼                                                         │
│  PaperEngine ─or─ TradovateClient    execution layer            │
│       │                                                         │
│       ▼                                                         │
│  [Dashboard]  [Analytics]            optional monitoring        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Signal flow**: `engine → prop_gate.on_signal → risk.on_signal → execution.on_signal`
**Bar order**: `prop.on_bar → risk.on_bar → paper.on_bar → engine.on_bar`

---

## Project Structure

```
Apex/
├── main.py                      # System runner — wires full pipeline (replay/paper/live)
├── run_backtest.py              # Backtest entry point (multi-strategy)
├── optimize.py                  # Parameter grid-search optimizer
├── walk_forward.py              # Walk-forward validation over rolling windows
├── generate_ema_dataset.py      # Build EMA-specific ML dataset
├── train_model.py               # Train ML model (Gradient Boosting)
├── evaluate_model.py            # Evaluate trained model performance
├── analyze_ml_value.py          # Compare ML vs no-ML strategy variants
├── fetch_data.py                # Databento data fetcher CLI
├── build_dataset.py             # CLI: build research dataset (features + labels)
├── split_dataset.py             # CLI: time-based train/val/test/holdout split
├── run_pipeline.py              # End-to-end pipeline runner
├── requirements.txt             # Python dependencies
│
├── config/
│   ├── __init__.py
│   └── settings.py              # All config dataclasses (instrument, strategy, backtest, eval)
│
├── data/
│   ├── __init__.py
│   ├── loader.py                # CSV/Parquet data loader with timezone handling
│   ├── databento_fetcher.py     # Databento Historical API integration
│   ├── validate.py              # OHLCV data validation checks
│   ├── features.py              # Vectorized feature engineering (research/ML)
│   ├── labels.py                # Supervised label generation (future data)
│   ├── splits.py                # Time-based dataset splitting
│   ├── build_dataset.py         # Pipeline orchestrator
│   ├── ema_candidates.py        # EMA candidate generation
│   ├── inspect_data.py          # Quick data inspection utility
│   └── sample_mes_5m.csv        # Sample data template
│
├── strategy/
│   ├── __init__.py
│   ├── orb.py                   # ORB strategy (pure signal generation)
│   ├── adaptive_regime.py       # Adaptive Regime strategy (regime-aware breakout)
│   ├── hybrid_ema_ml.py         # Hybrid EMA-ML multi-candidate strategy
│   ├── strategy_engine.py       # Live strategy engine (streaming bar-by-bar)
│   ├── paper_engine.py          # Paper trading engine (simulated execution)
│   ├── risk_manager.py          # Risk management layer (pre-trade gate)
│   ├── tradovate_client.py      # Tradovate broker execution client
│   ├── prop_challenge.py        # Prop firm challenge mode (equity tracking + risk gate)
│   ├── features.py              # Shared technical feature computations (EMA, ATR, volume)
│   └── regimes.py               # Regime classification logic (TREND/BREAKOUT/RANGE/DEAD)
│
├── backtest/
│   ├── __init__.py
│   ├── engine.py                # Bar-by-bar simulation engine (strategy-agnostic)
│   ├── metrics.py               # Performance statistics, export, and plotting
│   └── benchmark.py             # Benchmark strategies for comparison
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py                   # FastAPI web app (REST API + static frontend)
│   ├── state.py                 # Thread-safe in-memory state store
│   └── static/
│       └── index.html           # Real-time monitoring frontend (Chart.js)
│
├── analytics/
│   ├── __init__.py
│   └── engine.py                # Analytics engine (metrics, breakdowns, reports)
│
├── challenge/
│   ├── __init__.py
│   └── simulator.py             # Prop firm challenge Monte Carlo simulator
│
├── models/                      # Saved ML models and evaluation artifacts
│   ├── ema_model.pkl
│   ├── train_metrics.json
│   ├── evaluation_results.json
│   └── walk_forward_results.json
│
├── tests/                       # 390 tests across 11 test files
│   ├── test_data_pipeline.py
│   ├── test_hybrid_strategy.py
│   ├── test_storage_replay.py
│   ├── test_strategy_engine.py
│   ├── test_paper_engine.py
│   ├── test_risk_manager.py
│   ├── test_tradovate_client.py
│   ├── test_main.py
│   ├── test_dashboard.py
│   ├── test_analytics.py
│   └── test_prop_challenge.py
│
└── results/                     # Output directory (created on run)
    ├── trades.csv
    ├── metrics.json
    ├── equity_curve.csv
    ├── optimization_results.csv
    └── analytics.json
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Databento API key

Get your API key from [databento.com/portal/keys](https://databento.com/portal/keys) and export it:

```bash
export DATABENTO_API_KEY=db-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

On Windows PowerShell:

```powershell
$env:DATABENTO_API_KEY = "db-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3. Fetch historical data

```bash
python fetch_data.py --start 2024-01-02 --end 2024-06-30
```

This pulls 1-minute MES bars from Databento, resamples to 5-minute, and saves to `data/mes_5m.csv`.

**Options:**

| Flag           | Default               | Description                              |
|----------------|-----------------------|------------------------------------------|
| `--start`      | *(required)*          | Start date (YYYY-MM-DD)                  |
| `--end`        | *(required)*          | End date (YYYY-MM-DD, exclusive)         |
| `--output`     | `data/mes_5m.csv`     | Output CSV path                          |
| `--symbol`     | `MES.c.0`             | Databento continuous contract symbol     |
| `--dataset`    | `GLBX.MDP3`           | Databento dataset (CME Globex)           |
| `--timeframe`  | `5min`                | Resample timeframe (`1min`, `5min`, `15min`, `1h`) |
| `--timezone`   | `America/New_York`    | Target timezone for timestamps           |
| `--save-raw`   | off                   | Also save raw 1-minute data              |
| `--api-key`    | `DATABENTO_API_KEY`   | Override env var                         |

**Inspect fetched data:**

```bash
python -m data.inspect_data data/mes_5m.csv
```

### 4. Prepare your data (alternative: manual CSV)

Place your MES historical bar data as a CSV. Expected format:

| Column    | Type     | Description                     |
|-----------|----------|---------------------------------|
| timestamp | datetime | Bar timestamp (see timezone note)|
| open      | float    | Open price                      |
| high      | float    | High price                      |
| low       | float    | Low price                       |
| close     | float    | Close price                     |
| volume    | int/float| Volume                          |

**Timezone handling**:
- If timestamps are **naive** (no timezone info), they're assumed to be in the `--timezone` you specify (default: `America/New_York`).
- If timestamps are **tz-aware**, they're converted to the target timezone.
- Column name aliases accepted: `datetime`, `date`, `time`, `dt` → `timestamp`; `vol` → `volume`.

A sample template is at `data/sample_mes_5m.csv`.

### 5. Run a backtest

**ORB strategy** (default):

```bash
python run_backtest.py --data data/mes_5m.csv --strategy orb
```

With custom parameters:

```bash
python run_backtest.py --data data/mes_5m.csv --strategy orb \
    --rr 1.5 \
    --ema-length 20 \
    --or-end 09:45 \
    --capital 25000 \
    --slippage-ticks 1 \
    --commission 0.62
```

Enable short trades and filters:

```bash
python run_backtest.py --data data/mes_5m.csv --strategy orb \
    --rr 1.5 \
    --shorts \
    --min-range 2.0 \
    --max-entry-time 10:30
```

**Adaptive Regime strategy**:

```bash
python run_backtest.py --data data/mes_5m.csv --strategy adaptive_regime
```

With custom parameters and diagnostics:

```bash
python run_backtest.py --data data/mes_5m.csv --strategy adaptive_regime \
    --rr 2.0 \
    --shorts \
    --ema-length 50 \
    --breakout-buffer 0.5 \
    --min-range 1.0 \
    --min-score 3 \
    --export-diagnostics
```

### 6. Run parameter optimization

**ORB optimization**:

```bash
python optimize.py --data data/mes_5m.csv --strategy orb
```

Sort by a different metric:

```bash
python optimize.py --data data/mes_5m.csv --strategy orb --metric sharpe_ratio --top 20
```

Custom parameter grids:

```bash
python optimize.py --data data/mes_5m.csv --strategy orb \
    --or-minutes 10,15,20 \
    --rr-values 1.0,1.25,1.5,2.0 \
    --ema-values 20,50 \
    --eod-values 15:30,15:50 \
    --min-range-values 0,1.5,2.0,3.0 \
    --max-entry-time-values '',10:15,10:30 \
    --shorts
```

**Adaptive Regime optimization**:

```bash
python optimize.py --data data/mes_5m.csv --strategy adaptive_regime
```

Custom adaptive grid:

```bash
python optimize.py --data data/mes_5m.csv --strategy adaptive_regime \
    --rr-values 1.5,2.0,3.0 \
    --ema-values 20,50 \
    --min-range-values 0.5,1.0,2.0 \
    --breakout-buffer-values 0,0.25,0.5 \
    --min-score-values 2,3 \
    --shorts \
    --metric profit_factor \
    --top 20
```

---

## How the ORB Strategy Works

1. **Opening Range**: During the configured window (default 09:30–09:45 ET), track the highest high and lowest low across all bars.
2. **Long Breakout Entry**: After the opening range closes, if a bar's high exceeds the opening range high, enter long at the breakout level. Stop at OR low, target at entry + RR × range.
3. **Short Breakdown Entry** (optional, `--shorts`): If a bar's low breaks below the opening range low, enter short. Stop at OR high, target at entry − RR × range.
4. **EMA Filter** (optional, on by default): Only enter long if price is above EMA; only enter short if price is below EMA.
5. **Minimum Range Filter** (`--min-range`): Skip the day if the opening range is smaller than the threshold (in points). Filters out low-volatility setups.
6. **Max Entry Time Filter** (`--max-entry-time`): No entries after this time (e.g. `10:30`). Prevents late-day chasing.
7. **Stop Loss**: OR low for longs, OR high for shorts.
8. **Take Profit**: Entry ± (RR × range). Default RR is 1.5.
9. **End-of-Day Exit**: Force flat before close (default 15:50 ET).
10. **One Trade Per Day**: Only one entry attempt per session (configurable).

### Exit Priority (same bar)

- **Longs**: SL checked before TP (conservative — assumes adverse move happens first).
- **Shorts**: SL checked before TP (same logic — high checked before low).
- **EOD**: Always checked first regardless of direction.

### Fill Assumptions

- **Entry**: Assumed filled at the opening range high (breakout level), not at bar close. This is conservative — a real breakout fill may be worse.
- **Stop Loss**: Assumed filled at stop price when bar low penetrates it.
- **Take Profit**: Assumed filled at target price when bar high reaches it.
- **Slippage**: Applied as a dollar cost per side (ticks × tick_size × point_value).
- **No lookahead**: The strategy only sees the current bar and prior state.

---

## How the Adaptive Regime Strategy Works

The Adaptive Regime strategy is a more selective evolution of ORB. Instead of trading every breakout, it first classifies the market into a **regime** and only enters when conditions are favorable.

### Regime Classification

After the opening range closes, the strategy classifies the session into one of four regimes:

| Regime | Description | Trades allowed? |
|--------|-------------|-----------------|
| **TREND** | Strong EMA slope indicates directional bias | Yes — continuation breakouts |
| **BREAKOUT** | Opening range is large relative to recent ATR | Yes — expansion breakouts |
| **RANGE** | No strong trend or expansion signal | No |
| **DEAD** | Very low volatility and/or volume | No |

Classification uses transparent rules based on:
- **EMA slope** — measures trend strength over a configurable lookback
- **OR/ATR ratio** — compares opening range size to recent average true range
- **Relative volume** — current volume vs. rolling average

### Multi-Signal Confirmation

Even when the regime allows trading, entry requires passing a configurable number of confirmation filters (controlled by `--min-score`):

1. **Range size filter** — opening range must be within min/max bounds
2. **EMA direction filter** — price must be above EMA for longs, below for shorts
3. **EMA slope filter** — EMA must be sloping in the trade direction
4. **Volume filter** — volume must exceed a ratio of the rolling average
5. **ATR filter** — ATR must be above a minimum threshold
6. **Timing filter** — entry must occur before the max entry time
7. **Breakout buffer** — price must clear the range by a configurable buffer

Each filter can be individually enabled/disabled. The strategy counts how many pass and only enters if the score meets the minimum.

### Entry Logic

- **Long**: Bar high exceeds opening range high + buffer, regime allows, filters pass
- **Short**: Bar low breaks below opening range low − buffer, regime allows, filters pass

### Risk Management

- **Stop loss**: Opposite side of the opening range
- **Take profit**: Entry ± (reward:risk × range size)
- **End-of-day exit**: Forced flat at configurable time (default 15:50 ET)
- **One trade per day**: Only one entry attempt per session (configurable)

### Diagnostics

Use `--export-diagnostics` to get a per-day CSV showing:
- Detected regime and classification reason
- Opening range high/low/size
- EMA value and slope
- ATR and relative volume
- Whether a trade was taken and in which direction
- Full filter pass/fail breakdown
- Skip reason if no trade was taken

---

## Benchmark Comparison

Since MES is a futures contract, stock-style "buy and hold" is not meaningful. Four benchmarks are available:

1. **Always-Long Benchmark** (default): Enter long at session open every day, exit at EOD. Pays the same commissions and slippage. This answers: "Is the strategy filter adding value over naive directional exposure?"

2. **Flat / No-Trade Benchmark**: Equity stays constant at initial capital. Answers: "Am I better off doing nothing?"

3. **EMA Directional Benchmark**: Enter long at session open only when close > EMA, flat otherwise. A simple trend-following baseline.

4. **Unfiltered ORB**: Same ORB logic but with EMA filter disabled. Available via the optimizer (compare EMA on vs off).

All benchmarks use the same cost model (commissions + slippage) for fair comparison. The equity curve plot overlays all active benchmarks automatically.

---

## Computed Metrics

| Metric               | Description                                  |
|----------------------|----------------------------------------------|
| total_trades         | Number of completed round-trip trades         |
| win_rate_pct         | Percentage of winning trades                  |
| total_pnl_dollars    | Net P&L after commissions and slippage        |
| total_pnl_points     | Total P&L in index points                    |
| profit_factor        | Gross profit / gross loss                     |
| sharpe_ratio         | Annualized Sharpe from daily equity returns   |
| max_drawdown_dollars | Largest peak-to-trough equity decline ($)     |
| max_drawdown_pct     | Largest peak-to-trough decline (%)            |
| expectancy           | Average net P&L per trade                     |
| avg_win / avg_loss   | Average winning/losing trade                  |
| avg_trade_duration   | Mean trade duration in minutes                |
| pnl_by_weekday       | Net P&L broken down by day of week            |
| long_trades          | Count of long trades taken                    |
| short_trades         | Count of short trades taken                   |
| long_win_rate_pct    | Win rate for long trades only                 |
| short_win_rate_pct   | Win rate for short trades only                |
| long_pnl_dollars     | Net P&L from long trades                      |
| short_pnl_dollars    | Net P&L from short trades                     |

---

## MES Instrument Defaults

| Parameter    | Value | Notes                                |
|-------------|-------|--------------------------------------|
| Tick size    | 0.25  | Minimum price increment               |
| Point value  | $5.00 | Dollar value per index point          |
| Contract size| 1     | Default position size                 |
| Symbol       | MES   | Micro E-mini S&P 500                 |

These are configurable in `config/settings.py`.

---

## Output Files

After running a backtest, the `results/` directory contains:

- **trades.csv** — Every trade with entry/exit times, prices, PnL, reason
- **metrics.json** — Full performance statistics
- **equity_curve.csv** — Bar-by-bar equity and drawdown
- **equity_curve.png** — Visual equity curve with benchmark overlay

After optimization:

- **optimization_results.csv** — All parameter combinations with their metrics

---

## Hybrid EMA-ML Strategy

The Hybrid EMA-ML strategy (`strategy/hybrid_ema_ml.py`) combines EMA technical signals with machine learning probability scoring across multiple candidate setups.

### Multi-Candidate Mode

Enable with `--multi-candidate` to run multiple EMA lengths × entry types simultaneously:

```bash
python run_backtest.py --data data/mes_4y.csv --strategy hybrid_ema_ml \
    --multi-candidate \
    --ema-periods 20 50 100 \
    --entry-types breakout pullback momentum \
    --max-trades-per-day 3 \
    --selection-strategy priority
```

**Entry types:**

| Type | Description |
|------|-------------|
| `breakout` | Range + EMA breakout — price clears opening range and EMA |
| `pullback` | Near-EMA pullback — price retraces to EMA zone |
| `momentum` | Strong directional move with volume confirmation |

**Selection strategies** (`--selection-strategy`):

| Strategy | Description |
|----------|-------------|
| `global_ml` | Rank all candidates by ML probability, take top N |
| `priority` | Group by entry type (breakout > momentum > pullback), ML ranks within groups |
| `priority_ml_sizing` | Always enter in priority order, ML only affects position sizing |

Priority order: breakout (0) > momentum (1) > pullback (2).

### ML Pipeline

```bash
# 1. Generate EMA-specific training dataset
python generate_ema_dataset.py --data data/mes_4y.csv

# 2. Train Gradient Boosting model
python train_model.py

# 3. Evaluate on holdout split
python evaluate_model.py

# 4. Run backtest with ML
python run_backtest.py --data data/mes_4y.csv --strategy hybrid_ema_ml \
    --ml-threshold 0.55 --shorts --split holdout

# 5. Walk-forward validation
python walk_forward.py --data data/mes_4y.csv --candidates data/ema_candidates.csv
```

Model is saved at `models/ema_model.pkl` (Gradient Boosting with StandardScaler pipeline).

---

## Main System Runner

`main.py` wires the full live-compatible pipeline with three execution modes:

```bash
# Replay mode — signals only (no execution)
python main.py --mode replay --data data/mes_4y.csv

# Paper mode — simulated execution with full P&L tracking
python main.py --mode paper --data data/mes_4y.csv

# Live mode — real execution via Tradovate
python main.py --mode live --data data/mes_4y.csv
```

### Pipeline Components

| Component | Module | Role |
|-----------|--------|------|
| **StrategyEngine** | `strategy/strategy_engine.py` | Streaming bar-by-bar signal generation |
| **PropRiskGate** | `strategy/prop_challenge.py` | Optional prop firm challenge risk gate |
| **RiskManager** | `strategy/risk_manager.py` | Pre-trade risk limits and kill switch |
| **PaperEngine** | `strategy/paper_engine.py` | Simulated execution (paper mode) |
| **TradovateClient** | `strategy/tradovate_client.py` | Live broker execution (live mode) |
| **Dashboard** | `dashboard/` | Optional real-time web monitoring |
| **Analytics** | `analytics/engine.py` | Optional trade analytics and reporting |

### CLI Flags

| Flag | Description |
|------|-------------|
| `--mode` | `replay`, `paper`, or `live` |
| `--data` | Path to OHLCV CSV/Parquet |
| `--start` / `--end` | Date range filter |
| `--dashboard` | Enable real-time web dashboard |
| `--dashboard-port` | Dashboard port (default 8501) |
| `--analytics` | Enable analytics engine |
| `--analytics-output` | Analytics JSON output path |
| `--prop-mode` | Enable prop firm challenge mode |
| `--prop-target` | Profit target override (default $1,500) |
| `--prop-max-dd` | Max drawdown override (default $1,000) |
| `--prop-daily-loss` | Daily loss limit override (default $300) |
| `--prop-daily-lock` | Daily profit lock override (default $400) |
| `--prop-max-trades` | Max trades/day override (default 4) |

### Logging

When running `main.py`, a `logs/` directory is created with:

- `main.log` — general system events
- `trades.log` — all trade entries and exits
- `signals.log` — all signals generated
- `errors.log` — errors and exceptions

---

## Live Strategy Engine

`strategy/strategy_engine.py` wraps the backtest strategy into a live-compatible streaming interface.

- **`StrategyEngine.on_bar(bar: dict) → list[LiveSignal]`** — processes one bar at a time
- **`LiveSignal`** dataclass: timestamp, direction, signal_type, entry, stop, take_profit, position_size, strategy_type, ml_prob, percentile
- **`EngineState`** — read-only snapshot of internal state (bar_count, current_date, range_set, open_positions)
- **Validation mode**: `engine.validate(bars, reference_signals) → ValidationResult` — cross-checks live engine output against backtest trades
- **`on_signal`** callback for wiring to downstream components (risk, execution)

---

## Paper Trading Engine

`strategy/paper_engine.py` provides simulated execution identical to the backtest engine, consuming `LiveSignal` objects in real-time.

- **Same cost model** as the backtest engine (slippage + commission)
- **Mark-to-market** equity tracking on every bar
- **PnLUpdate** emitted on every state change (bar tick, entry, exit) — drainable via `pending_updates()`
- **Validation**: `compare_with_backtest(trades)` compares paper trades against backtest trades by (entry_time, direction, strategy_type)
- **PaperConfig**: starting_capital, commission_per_side, slippage_ticks, point_value, tick_size

---

## Risk Management Layer

`strategy/risk_manager.py` sits between the StrategyEngine and execution as a pre-trade risk gate.

### Hard Limits

| Parameter | Description |
|-----------|-------------|
| `max_daily_loss` | Max $ loss per day before blocking entries |
| `max_trades_per_day` | Max trade entries per day |
| `max_concurrent_positions` | Max open positions at once |
| `max_position_size` | Max contracts per entry |
| `max_total_exposure` | Max total contracts across all positions |

### Kill Switch

- Blocks all new entries immediately
- Optionally force-closes all open positions (emits EXIT_EOD signals)
- **Persists across days** — operator must explicitly reset via `reset_kill_switch()`
- Triggered manually or by hitting unrealized loss limits

### Behavior

- Daily counters auto-reset on day change
- Unrealized loss checked on `on_bar()` against mark-to-market equity
- All blocks, caps, and kill switch events logged as `RiskEvent` audit trail
- Wiring: `engine → risk.on_signal → risk.on_approved = paper.on_signal`

---

## Tradovate Execution Client

`strategy/tradovate_client.py` provides the same `on_signal(LiveSignal)` interface as PaperEngine — a drop-in replacement for live execution.

### Features

| Feature | Description |
|---------|-------------|
| **Authentication** | OAuth via `/auth/accesstokenrequest` |
| **Bracket orders** | OSO (One-Sends-Other): market entry + SL stop + TP limit |
| **State sync** | Background polling thread for order/position status |
| **Fill tracking** | Polls `executionReport/list` for fills, partial fill support |
| **Safety guard** | `sim_only=True` refuses live environment connections |
| **Emergency** | `liquidate_all()` cancels working orders + liquidates positions |

### Configuration

```python
TradovateConfig(
    username="...",
    password="...",
    app_id="...",
    cid="...",
    sec="...",
    environment="demo",  # "demo" or "live"
    sim_only=True,       # safety guard
)
```

Wiring: `risk.on_approved = tradovate.on_signal`

---

## Monitoring Dashboard

Real-time web dashboard for monitoring live/paper trading sessions.

### Components

| File | Purpose |
|------|---------|
| `dashboard/state.py` | Thread-safe in-memory store for PnL, trades, signals, alerts |
| `dashboard/app.py` | FastAPI app: `GET /`, `/api/snapshot`, `/api/equity`, `/api/health` |
| `dashboard/static/index.html` | Frontend: Chart.js equity curve, PnL cards, positions, trades table, alerts |

### Enable

```bash
python main.py --mode paper --data data/mes_4y.csv --dashboard --dashboard-port 8501
```

### Alerts

| Alert | Trigger |
|-------|---------|
| Drawdown warning | -$300 unrealized |
| Drawdown critical | -$450 unrealized |
| Risk blocked | Entry blocked by RiskManager |
| Kill switch | Kill switch activated |

### Callbacks

- `on_pnl(update)` — PnL updates pushed each bar
- `on_trade(trade)` — new trade notifications
- `on_signal(signal)` — signal generation events
- `on_risk_event(event)` — risk blocks and kill switch events
- `update_open_positions(positions)` — current position state
- `update_risk_state(state)` — risk manager state

---

## Analytics System

`analytics/engine.py` provides thread-safe trade analytics with on-demand metric computation.

### Enable

```bash
python main.py --mode paper --data data/mes_4y.csv --analytics --analytics-output results/analytics.json
```

### AnalyticsReport

| Metric | Description |
|--------|-------------|
| `total_trades` | Total completed trades |
| `win_rate` | Percentage of winning trades |
| `profit_factor` | Gross profit / gross loss |
| `sharpe_ratio` | Annualized Sharpe (√252) |
| `max_drawdown` | Largest peak-to-trough equity decline |
| `long_trades` / `short_trades` | Trades by direction |
| `long_win_rate` / `short_win_rate` | Win rate by direction |
| `strategy_breakdown` | Per-strategy_type metrics (win rate, PF, avg PnL, count) |
| `daily_pnl` | Day-by-day P&L series |

### Features

- Records trades, signals, and risk decisions
- Per-strategy breakdowns (e.g., ema20_breakout vs ema50_pullback)
- Console summary printed after each run
- JSON export via `export_json()`

---

## Prop Firm Challenge Mode

`strategy/prop_challenge.py` implements comprehensive prop firm evaluation rules designed for accounts like a **25K Rithmic Intraday Trail**.

### Enable

```bash
# Basic prop mode
python main.py --mode paper --data data/mes_4y.csv --prop-mode

# Custom parameters
python main.py --mode paper --data data/mes_4y.csv --prop-mode \
    --prop-target 2000 \
    --prop-daily-loss 250 \
    --prop-max-trades 5
```

### Account Rules

| Parameter | Default | Description |
|-----------|---------|-------------|
| Starting capital | $25,000 | Account starting balance |
| Profit target | $1,500 | Pass when equity reaches $26,500 |
| Max drawdown | $1,000 | Trailing intraday drawdown limit |
| Drawdown type | `trailing_intraday` | Threshold rises with peak equity, never falls |

### Risk Controls

| Control | Default | Description |
|---------|---------|-------------|
| Daily loss limit | $300 | Stop trading for the day after $300 loss |
| DD buffer | $200 | Stop trading when within $200 of drawdown limit |
| Max trades/day | 4 | Hard cap on daily entries |
| Max consecutive losses | 3 | Kill switch after 3 losses in a row |

### Trade Filtering

- **Entry types**: Only `breakout` entries allowed (pullback/momentum blocked)
- **ML threshold**: Minimum `ml_prob ≥ 0.60` required
- Exits always pass through (never blocked)

### Staged Position Sizing

| Equity Gain | Size Multiplier | Phase |
|-------------|----------------|-------|
| $0 – $500 | 0.25× | Small — conservative start |
| $500 – $1,200 | 0.50× | Medium — building confidence |
| $1,200 – $1,500 | 0.35× | Lock-in — protect gains near target |

### Profit Protection

| Rule | Default | Description |
|------|---------|-------------|
| Daily profit lock | $400 | Stop trading after +$400 in a single day |
| Giveback threshold | $300 | If daily peak PnL reaches $300... |
| Giveback drop | $200 | ...and drops $200 from that peak, stop for the day |

### Exit Tightening

- **Stop tightening**: Stops moved 15% closer to entry (0.85× original distance)
- **Reward:risk override**: Take-profit set at 1.2× risk (instead of default R:R)

### Kill Switch

Trading halts immediately when any of these conditions are met:

- Trailing drawdown limit breached (FAIL)
- Profit target reached (PASS)
- 3 consecutive losing trades
- Drawdown buffer exhausted (within $200 of limit)

### PropEquityTracker

Tracks real-time equity state:

- `peak_equity` — highest equity seen (only rises)
- `trailing_dd_level` — `peak_equity - max_drawdown` (drawdown floor)
- `daily_pnl` — P&L since day start
- `daily_peak_pnl` — highest daily P&L (for giveback rule)
- `dd_buffer_remaining` — distance from current equity to drawdown limit
- `passed` / `failed` — terminal states
- Auto-resets daily tracking on day change

### Challenge Simulator

Sliding-window Monte Carlo simulation across historical data:

```bash
python -m challenge.simulator --data data/mes_4y.csv --windows 50 --window-days 30 --step-days 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *(required)* | Historical OHLCV data |
| `--windows` | 20 | Number of simulation windows |
| `--window-days` | 30 | Trading days per trial window |
| `--step-days` | 5 | Days to slide between windows |
| `--output` | — | JSON output path |
| `--ml-model` | — | Path to ML model pickle |

**SimulationReport output:**

| Metric | Description |
|--------|-------------|
| `pass_rate` | Percentage of trials that passed |
| `avg_days_to_pass` | Mean trading days to reach target |
| `avg_days_to_fail` | Mean trading days to hit drawdown |
| `avg_max_drawdown` | Average peak-to-trough drawdown across trials |
| `avg_win_rate` | Average trade win rate across trials |
| `max_pass_streak` | Longest consecutive passing windows |
| `max_fail_streak` | Longest consecutive failing windows |
| `total_trials` | Number of windows simulated |

Each trial builds a fresh pipeline (StrategyEngine → PropRiskGate → RiskManager → PaperEngine) and feeds a window of historical bars.

---

## Evaluation Mode (Backtest-Level)

The backtester can also simulate a prop-firm evaluation account with trailing intraday drawdown rules (simpler than the full challenge mode above):

```bash
python run_backtest.py --data data/mes_5m.csv --strategy adaptive_regime --shorts --eval
```

Default eval settings model a **25K Rithmic Intraday Trail** account:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval-capital` | $25,000 | Starting capital |
| `--eval-target` | $1,500 | Profit target (pass at $26,500) |
| `--eval-drawdown` | $1,000 | Max trailing intraday drawdown |

**Behavior**:
- Tracks mark-to-market equity (including unrealized P&L) every bar
- Trailing drawdown threshold rises with peak equity, never falls
- **PASS**: equity reaches `starting_capital + profit_target`
- **FAIL**: equity drops to or below `peak_equity - max_drawdown`
- Simulation stops immediately on PASS or FAIL
- Open positions are force-closed at the current bar's close

> **Note**: For full challenge mode with staged sizing, trade filtering, profit protection, and kill switch logic, use `--prop-mode` via `main.py` instead.

---

## Data Pipeline (Research / ML)

A separate, modular pipeline for building research-grade datasets with features, labels, and time-based splits. Designed for strategy research, model training, and overfitting-resistant evaluation.

### Architecture

```
fetch_data.py          → Raw OHLCV from Databento
       ↓
data/features.py       → Vectorized feature engineering (f_* columns)
       ↓
data/labels.py         → Supervised labels using future data (lbl_* columns)
       ↓
data/build_dataset.py  → Orchestrator: load → features → labels → trade candidates
       ↓
data/splits.py         → Time-based train/val/test/holdout splits
```

> **Note**: `data/features.py` (vectorized, DataFrame-level) is distinct from `strategy/features.py` (streaming, bar-by-bar for live/backtest use). They serve different purposes and coexist.

### Pipeline modules

| Module | Purpose | Key classes |
|--------|---------|-------------|
| `data/features.py` | 30+ vectorized features across price, volume, volatility, time, range, regime-proxy | `FeatureConfig`, `compute_features()` |
| `data/labels.py` | Future-return labels, breakout success (TP/SL sim), false breakout detection, regime labels | `LabelConfig`, `compute_labels()` |
| `data/splits.py` | Contiguous time-based splitting with purge gap, no random shuffling | `SplitConfig`, `time_based_split()` |
| `data/build_dataset.py` | Full pipeline orchestrator with trade candidate extraction | `DatasetConfig`, `build_dataset()` |

### Build a research dataset

```bash
python build_dataset.py --input data/mes_5m.csv
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Raw OHLCV CSV/parquet |
| `--start` | — | Start date filter (YYYY-MM-DD) |
| `--end` | — | End date filter (YYYY-MM-DD) |
| `--output-dir` | `data/` | Base output directory |
| `--min-range` | 0 | Min opening range size (points) |
| `--range-end` | `09:45` | Opening range end time |
| `--no-save` | off | Compute without saving |

Outputs:
- `data/processed/mes_5m.parquet` — validated bars
- `data/features/mes_features.parquet` — full bar-level features + labels
- `data/features/mes_trade_candidates.parquet` — one row per session per direction (long/short)
- `data/features/dataset_summary.json` — stats and distributions

### Split the dataset

```bash
python split_dataset.py --input data/features/mes_trade_candidates.parquet
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Parquet from build step |
| `--output-dir` | `<input_dir>/splits/` | Split output directory |
| `--train` | 60 | Train % |
| `--val` | 20 | Validation % |
| `--test` | 10 | Test % |
| `--holdout` | 10 | Holdout % |
| `--purge-days` | 1 | Gap days between splits |

Outputs per split: `train.parquet`, `val.parquet`, `test.parquet`, `holdout.parquet`.

### Feature columns (`f_*`)

| Group | Examples |
|-------|----------|
| Price/structure | `f_price_ema`, `f_price_ema_dist`, `f_price_ema_slope`, `f_price_rolling_ret_*`, `f_price_gap` |
| Volume | `f_vol_avg`, `f_vol_relative`, `f_vol_expansion` |
| Volatility | `f_vola_tr`, `f_vola_atr`, `f_vola_atr_norm`, `f_vola_realized` |
| Time | `f_time_since_open`, `f_time_since_range`, `f_time_to_close`, `f_time_weekday` |
| Opening range | `f_range_high`, `f_range_low`, `f_range_size`, `f_range_dist_*`, `f_range_vs_atr` |
| Regime proxy | `f_regime_trend_strength`, `f_regime_compression`, `f_regime_breakout_strength` |

### Label columns (`lbl_*`)

| Label | Description |
|-------|-------------|
| `lbl_ret_Nbar` | N-bar forward return (pct) |
| `lbl_long_success_RR` | Long TP/SL hit within session at R:R ratio |
| `lbl_short_success_RR` | Short TP/SL hit within session |
| `lbl_false_breakout_long/short` | Reversal within M bars of breakout |
| `lbl_regime` | Session regime: trend/breakout/range/dead |

---

## Tests

390 tests across 11 test files:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_data_pipeline.py` | 36 | Data loading, features, labels, splits |
| `test_hybrid_strategy.py` | 26 | Hybrid EMA-ML strategy + multi-candidate |
| `test_storage_replay.py` | 48 | Storage and signal replay |
| `test_strategy_engine.py` | 27 | Live strategy engine |
| `test_paper_engine.py` | 39 | Paper trading execution + validation |
| `test_risk_manager.py` | 38 | Risk limits, kill switch, daily resets |
| `test_tradovate_client.py` | 31 | Tradovate auth, orders, fills, sync |
| `test_main.py` | 32 | Pipeline wiring, CLI flags, bar loop |
| `test_dashboard.py` | 29 | Dashboard state, API endpoints, alerts |
| `test_analytics.py` | 46 | Analytics metrics, breakdowns, export |
| `test_prop_challenge.py` | 45 | Prop equity tracker, risk gate, sizing, simulator |

Run tests:

```bash
python -m pytest tests/ -v
```

> **Note**: Tests are run via subprocess internally to avoid pyarrow DLL conflicts. Use `python -m pytest` rather than invoking pytest directly.

---

## Assumptions and Limitations

- **Shorts are opt-in**: All strategies support long and short entries. Use `--shorts` to enable short entries.
- **Bar-based fills**: Fills are simulated within bar OHLC ranges. Intra-bar price path is unknown, so same-bar SL+TP priority is: SL checked first (conservative).
- **No overnight positions**: EOD exit is enforced. The strategy resets daily.
- **No margin modeling**: The system tracks equity but does not model futures margin requirements.
- **Data quality**: The system validates basic schema but does not detect gaps, bad ticks, or exchange holidays. Garbage in = garbage out.
- **Timezone matters**: Make sure your data timestamps match the timezone you specify. Misaligned timestamps will produce incorrect opening ranges.
- **Tradovate sim_only**: The Tradovate client defaults to `sim_only=True` and will refuse live environment connections unless explicitly overridden.
- **ML value caveat**: Analysis showed no-ML (arrival order) outperforms all ML variants on holdout data (Sharpe 1.39 vs 0.81-0.98). ML correlation is weak but positive (+0.06). Arrival order structurally favors ema20_breakout which dominates PnL.

---

## What You Need to Provide

1. **Databento API key** — sign up at [databento.com](https://databento.com) and export `DATABENTO_API_KEY`. Then run `python fetch_data.py` to pull MES data automatically.

2. **Or, manual CSV** — MES historical bar data in CSV format (5-minute bars recommended). Sources:
   - Tradovate (export from platform)
   - Sierra Chart
   - TradeStation
   - Kinetick / NinjaTrader

3. Make sure timestamps are in **Eastern Time** (or specify the correct timezone via `--timezone`).

---

## Overfitting Warning

Parameter optimization is a powerful tool but also the most common way to fool yourself. The optimizer performs an **in-sample** search. To validate:

1. Split your data: use the first 60-70% for optimization, hold out the rest (or use `split_dataset.py` for proper train/val/test/holdout splits)
2. Run the best parameters on the out-of-sample period
3. If performance degrades significantly, the parameters are overfit
4. Prefer parameter sets that are "good enough" across many combinations over the single best one
5. Use evaluation mode (`--eval`) to test if a parameter set can pass a simulated prop-firm account
#   A p e x B o t  
 #   A p e x B o t  
 