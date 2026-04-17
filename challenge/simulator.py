"""
Challenge Simulator
====================

Runs the prop firm challenge on historical data across multiple
date windows to estimate:

    - Pass rate (% of attempts reaching +$1,500 before -$1,000)
    - Average days to pass / fail
    - Drawdown distribution
    - Win/loss streak statistics

Usage:
    python -m challenge.simulator --data data/mes_4y.csv
    python -m challenge.simulator --data data/mes_4y.csv --windows 50

No lookahead bias — each window starts fresh and processes bars
sequentially using the same pipeline as a live run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import InstrumentConfig
from strategy.hybrid_ema_ml import HybridEMAMLConfig
from strategy.paper_engine import PaperEngine, PaperConfig
from strategy.risk_manager import RiskManager, RiskConfig
from strategy.strategy_engine import StrategyEngine, LiveSignal
from strategy.prop_challenge import PropConfig, PropRiskGate
from strategy.orb import SignalType

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------

@dataclass
class TrialResult:
    """Outcome of a single challenge attempt."""
    start_date: str
    end_date: str
    outcome: str          # "passed", "failed", "incomplete"
    days_elapsed: int = 0
    final_equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    prop_events: int = 0


@dataclass
class SimulationReport:
    """Aggregate statistics across all trial windows."""
    total_trials: int = 0
    passed: int = 0
    failed: int = 0
    incomplete: int = 0
    pass_rate: float = 0.0
    avg_days_to_pass: float = 0.0
    avg_days_to_fail: float = 0.0
    avg_max_drawdown: float = 0.0
    median_final_equity: float = 0.0
    avg_trades_per_trial: float = 0.0
    avg_win_rate: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    trials: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_trials": self.total_trials,
                "passed": self.passed,
                "failed": self.failed,
                "incomplete": self.incomplete,
                "pass_rate": round(self.pass_rate, 2),
                "avg_days_to_pass": round(self.avg_days_to_pass, 1),
                "avg_days_to_fail": round(self.avg_days_to_fail, 1),
                "avg_max_drawdown": round(self.avg_max_drawdown, 2),
                "median_final_equity": round(self.median_final_equity, 2),
                "avg_trades_per_trial": round(self.avg_trades_per_trial, 1),
                "avg_win_rate": round(self.avg_win_rate, 1),
                "max_win_streak": self.max_win_streak,
                "max_loss_streak": self.max_loss_streak,
            },
            "trials": self.trials,
        }


# ------------------------------------------------------------------
# Single trial runner
# ------------------------------------------------------------------

def run_trial(
    bars: pd.DataFrame,
    instrument: InstrumentConfig,
    strategy_cfg: HybridEMAMLConfig,
    risk_cfg: RiskConfig,
    prop_cfg: PropConfig,
    paper_cfg: PaperConfig,
) -> TrialResult:
    """
    Run one challenge attempt on a slice of bars.

    Returns a TrialResult with outcome and statistics.
    """
    paper = PaperEngine(instrument=instrument, config=paper_cfg)
    risk = RiskManager(config=risk_cfg, instrument=instrument)

    prop_gate = PropRiskGate(
        config=prop_cfg,
        on_approved=risk.on_signal,
        get_equity=lambda: paper.mark_to_market_equity,
    )
    risk.on_approved = paper.on_signal

    engine = StrategyEngine(
        config=strategy_cfg,
        on_signal=prop_gate.on_signal,
    )

    dates_seen: set[str] = set()

    for ts, row in bars.iterrows():
        bar = {
            "timestamp": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

        prop_gate.on_bar(bar)
        risk.on_bar(bar)
        paper.on_bar(bar)
        engine.on_bar(bar)

        # Track trade closes for consecutive loss detection
        trades = paper.trades
        if trades:
            last_trade = trades[-1]
            # Only notify on newly closed trades
            if len(trades) > len(dates_seen):
                prop_gate.on_trade_closed(last_trade.net_pnl, strategy_type=getattr(last_trade, 'strategy_type', ''))

        dates_seen.add(str(ts.date()) if hasattr(ts, 'date') else str(ts)[:10])

        # Early exit if challenge resolved
        if prop_gate.halted:
            break

    # --- Compute result ---
    tracker = prop_gate.tracker
    trades = paper.trades
    pnls = [t.net_pnl for t in trades]

    if tracker.passed:
        outcome = "passed"
    elif tracker.failed:
        outcome = "failed"
    else:
        outcome = "incomplete"

    # Win/loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0
    for pnl in pnls:
        if pnl > 0:
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)

    # Max drawdown from equity curve
    equity_pts = paper.equity_curve
    if equity_pts:
        equities = np.array([e.equity for e in equity_pts])
        peaks = np.maximum.accumulate(equities)
        max_dd = float(np.min(equities - peaks))
    else:
        max_dd = 0.0

    trading_days = len(dates_seen)
    start_date = str(bars.index[0].date()) if len(bars) > 0 else ""
    end_date = str(bars.index[-1].date()) if len(bars) > 0 else ""

    return TrialResult(
        start_date=start_date,
        end_date=end_date,
        outcome=outcome,
        days_elapsed=trading_days,
        final_equity=tracker.current_equity,
        peak_equity=tracker.peak_equity,
        max_drawdown=max_dd,
        total_trades=len(trades),
        win_count=sum(1 for p in pnls if p > 0),
        loss_count=sum(1 for p in pnls if p <= 0),
        largest_win=max(pnls) if pnls else 0.0,
        largest_loss=min(pnls) if pnls else 0.0,
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        prop_events=len(prop_gate.events),
    )


# ------------------------------------------------------------------
# Multi-window simulation
# ------------------------------------------------------------------

def run_simulation(
    bars: pd.DataFrame,
    instrument: InstrumentConfig,
    strategy_cfg: HybridEMAMLConfig,
    risk_cfg: RiskConfig,
    prop_cfg: PropConfig,
    paper_cfg: PaperConfig,
    *,
    num_windows: int = 20,
    window_trading_days: int = 60,
    step_trading_days: int = 10,
) -> SimulationReport:
    """
    Run the challenge across sliding windows of historical data.

    Parameters
    ----------
    bars : DataFrame
        Full historical OHLCV data.
    num_windows : int
        Number of trial windows to run.
    window_trading_days : int
        Max trading days per trial window.
    step_trading_days : int
        Step size between window starts (in trading days).
    """
    # Group bars by date to slice by trading day
    bars = bars.sort_index()
    dates = sorted(bars.index.normalize().unique())

    if len(dates) < window_trading_days:
        logger.warning("Not enough data: %d trading days < %d window",
                       len(dates), window_trading_days)

    trials: list[TrialResult] = []

    for i in range(num_windows):
        start_idx = i * step_trading_days
        end_idx = start_idx + window_trading_days

        if start_idx >= len(dates):
            break
        if end_idx > len(dates):
            end_idx = len(dates)

        start_date = dates[start_idx]
        end_date = dates[min(end_idx, len(dates) - 1)]

        window_bars = bars[
            (bars.index >= start_date) & (bars.index <= end_date + pd.Timedelta(days=1))
        ]

        if len(window_bars) == 0:
            continue

        logger.info(
            "Trial %d/%d: %s → %s (%d bars)",
            i + 1, num_windows,
            str(start_date.date()), str(end_date.date()),
            len(window_bars),
        )

        result = run_trial(
            window_bars, instrument, strategy_cfg,
            risk_cfg, prop_cfg, paper_cfg,
        )
        trials.append(result)

        logger.info(
            "Trial %d: %s in %d days, equity=%.2f, trades=%d",
            i + 1, result.outcome, result.days_elapsed,
            result.final_equity, result.total_trades,
        )

    return _aggregate(trials)


def _aggregate(trials: list[TrialResult]) -> SimulationReport:
    """Compute aggregate statistics from trial results."""
    if not trials:
        return SimulationReport()

    report = SimulationReport()
    report.total_trials = len(trials)
    report.passed = sum(1 for t in trials if t.outcome == "passed")
    report.failed = sum(1 for t in trials if t.outcome == "failed")
    report.incomplete = sum(1 for t in trials if t.outcome == "incomplete")

    completed = report.passed + report.failed
    report.pass_rate = (report.passed / completed * 100) if completed > 0 else 0.0

    pass_days = [t.days_elapsed for t in trials if t.outcome == "passed"]
    fail_days = [t.days_elapsed for t in trials if t.outcome == "failed"]
    report.avg_days_to_pass = float(np.mean(pass_days)) if pass_days else 0.0
    report.avg_days_to_fail = float(np.mean(fail_days)) if fail_days else 0.0

    dds = [t.max_drawdown for t in trials]
    report.avg_max_drawdown = float(np.mean(dds))

    finals = [t.final_equity for t in trials]
    report.median_final_equity = float(np.median(finals))

    trade_counts = [t.total_trades for t in trials]
    report.avg_trades_per_trial = float(np.mean(trade_counts))

    win_rates = [
        t.win_count / t.total_trades * 100
        if t.total_trades > 0 else 0.0
        for t in trials
    ]
    report.avg_win_rate = float(np.mean(win_rates))

    report.max_win_streak = max(t.max_win_streak for t in trials)
    report.max_loss_streak = max(t.max_loss_streak for t in trials)

    report.trials = [
        {
            "start_date": t.start_date,
            "end_date": t.end_date,
            "outcome": t.outcome,
            "days_elapsed": t.days_elapsed,
            "final_equity": round(t.final_equity, 2),
            "peak_equity": round(t.peak_equity, 2),
            "max_drawdown": round(t.max_drawdown, 2),
            "total_trades": t.total_trades,
            "win_count": t.win_count,
            "loss_count": t.loss_count,
            "largest_win": round(t.largest_win, 2),
            "largest_loss": round(t.largest_loss, 2),
            "max_win_streak": t.max_win_streak,
            "max_loss_streak": t.max_loss_streak,
            "prop_events": t.prop_events,
        }
        for t in trials
    ]

    return report


# ------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------

def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Prop Firm Challenge Simulator",
    )
    p.add_argument("--data", default="data/mes_4y.csv", help="OHLCV CSV path")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--windows", type=int, default=20, help="Number of trial windows")
    p.add_argument("--window-days", type=int, default=60, help="Trading days per window")
    p.add_argument("--step-days", type=int, default=10, help="Step between windows")
    p.add_argument("--output", default="results/challenge_sim.json", help="Output JSON")
    p.add_argument("--ml-model", default="models/ema_model.pkl", help="ML model path")
    p.add_argument("--ml-threshold", type=float, default=0.60)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from data.loader import load_bars

    logger.info("Loading bars from %s", args.data)
    bars = load_bars(args.data, start=args.start, end=args.end)
    logger.info("Loaded %d bars", len(bars))

    instrument = InstrumentConfig()

    strategy_cfg = HybridEMAMLConfig(
        multi_candidate=False,
        ema_periods=(50,),
        entry_types=("breakout",),
        ml_threshold=args.ml_threshold,
        model_path=args.ml_model,
        selection_strategy="global_ml",
        max_trades_per_day=4,
    )

    risk_cfg = RiskConfig(
        max_daily_loss=300.0,
        max_trades_per_day=4,
        max_concurrent_positions=1,
    )

    prop_cfg = PropConfig()

    paper_cfg = PaperConfig(
        initial_capital=prop_cfg.starting_capital,
    )

    report = run_simulation(
        bars, instrument, strategy_cfg, risk_cfg, prop_cfg, paper_cfg,
        num_windows=args.windows,
        window_trading_days=args.window_days,
        step_trading_days=args.step_days,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  Prop Challenge Simulation Results")
    print("=" * 60)
    print(f"  Trials:           {report.total_trials}")
    print(f"  Passed:           {report.passed}")
    print(f"  Failed:           {report.failed}")
    print(f"  Incomplete:       {report.incomplete}")
    print(f"  Pass rate:        {report.pass_rate:.1f}%")
    print(f"  Avg days to pass: {report.avg_days_to_pass:.1f}")
    print(f"  Avg days to fail: {report.avg_days_to_fail:.1f}")
    print(f"  Avg max DD:       ${report.avg_max_drawdown:,.2f}")
    print(f"  Median equity:    ${report.median_final_equity:,.2f}")
    print(f"  Avg trades/trial: {report.avg_trades_per_trial:.1f}")
    print(f"  Avg win rate:     {report.avg_win_rate:.1f}%")
    print(f"  Max win streak:   {report.max_win_streak}")
    print(f"  Max loss streak:  {report.max_loss_streak}")
    print("=" * 60 + "\n")

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Report saved to %s", args.output)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
