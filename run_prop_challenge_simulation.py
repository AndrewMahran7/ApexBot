#!/usr/bin/env python3
"""
Prop Firm Challenge Monte Carlo Simulation
============================================

Simulates prop firm challenge pass/fail outcomes for the AdaptiveRegimeStrategy
under realistic constraints using the existing Monte Carlo engine.

Rules:
  - Starting capital: $25,000
  - Profit target:    +$1,500 (pass at $26,500)
  - Max drawdown:     -$1,000 (trailing intraday)
  - Stop on first PASS or FAIL

Uses block bootstrap (blocks of 7 trades) to preserve streak structure.
Runs 2,000 simulations per scenario (MES, MNQ, Combined).

Usage:
    .\\venv\\Scripts\\python.exe run_prop_challenge_simulation.py
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    AdaptiveRegimeConfig,
    BacktestConfig,
    INSTRUMENT_REGISTRY,
)
from data.loader import load_bars
from backtest.engine import BacktestEngine, Trade
from backtest.metrics import compute_metrics, export_trades_csv
from strategy.adaptive_regime import AdaptiveRegimeStrategy
from challenge.monte_carlo import (
    TradeRecord,
    run_monte_carlo,
    run_sensitivity,
    plot_results,
    print_report,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prop_challenge_sim")
logger.setLevel(logging.INFO)

SYMBOLS = ["MES", "MNQ"]

OUTPUT_DIR = Path("results/prop_challenge_simulation")

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=25_000.0,
    slippage_ticks=1,
    commission_per_side=2.25,
)

# Monte Carlo parameters
N_SIMULATIONS = 2000
MAX_TRADES_PER_SIM = 500  # generous — allow long attempts
BLOCK_SIZE = 7  # preserve streak structure (5-10 range)
SEED = 42
STARTING_CAPITAL = 25_000.0
PROFIT_TARGET = 1_500.0
MAX_DRAWDOWN = 1_000.0
DD_BUFFER = 200.0

# ---------------------------------------------------------------------------
# Data discovery (same as run_adaptive_regime_validation.py)
# ---------------------------------------------------------------------------

YEAR_DATA = {}
for sym in SYMBOLS:
    prefix = sym.lower()
    for year in [2017, 2018, 2019, 2022, 2025]:
        fpath = Path(f"data/{prefix}_{year}.csv")
        if fpath.exists():
            YEAR_DATA[(year, sym)] = {"path": str(fpath), "filter": False}

    fouryr = Path(f"data/{prefix}_4y.csv")
    if fouryr.exists():
        for year in [2021, 2023, 2024]:
            YEAR_DATA[(year, sym)] = {"path": str(fouryr), "filter": True, "year": year}
        if (2022, sym) not in YEAR_DATA:
            YEAR_DATA[(2022, sym)] = {"path": str(fouryr), "filter": True, "year": 2022}


def load_year_data(year: int, symbol: str) -> pd.DataFrame:
    key = (year, symbol)
    if key not in YEAR_DATA:
        raise FileNotFoundError(f"No data available for {symbol} {year}")
    info = YEAR_DATA[key]
    bars = load_bars(info["path"])
    if info.get("filter"):
        y = info["year"]
        start = pd.Timestamp(f"{y}-01-01", tz="America/New_York")
        end = pd.Timestamp(f"{y}-12-31 23:59:59", tz="America/New_York")
        bars = bars[(bars.index >= start) & (bars.index <= end)]
        if bars.empty:
            raise ValueError(f"No bars for {symbol} {year} after filtering")
    return bars


# ---------------------------------------------------------------------------
# Step 1 & 2 — Generate trades from current strategy
# ---------------------------------------------------------------------------

def collect_all_trades() -> dict[str, list[Trade]]:
    """Run AdaptiveRegimeStrategy across all years/symbols, return trades by symbol."""
    trades_by_symbol: dict[str, list[Trade]] = {"MES": [], "MNQ": [], "Combined": []}

    available_years = sorted(set(y for (y, _) in YEAR_DATA.keys()))
    logger.info("Collecting trades from years: %s", available_years)

    for year in available_years:
        for symbol in SYMBOLS:
            key = (year, symbol)
            if key not in YEAR_DATA:
                logger.warning("No data for %s %d — skipped", symbol, year)
                continue

            logger.info("Running %s %d ...", symbol, year)
            bars = load_year_data(year, symbol)
            instrument = INSTRUMENT_REGISTRY[symbol]
            config = AdaptiveRegimeConfig.for_symbol(symbol)
            strategy = AdaptiveRegimeStrategy(config)

            engine = BacktestEngine(
                instrument=instrument,
                strategy_config=config,
                backtest_config=BACKTEST_CONFIG,
                strategy=strategy,
            )
            result = engine.run(bars)
            n = len(result.trades)
            pnl = sum(t.net_pnl for t in result.trades)
            logger.info("  %s %d: %d trades, PnL $%.2f", symbol, year, n, pnl)

            trades_by_symbol[symbol].extend(result.trades)
            trades_by_symbol["Combined"].extend(result.trades)

    # Sort combined by entry_time
    trades_by_symbol["Combined"].sort(key=lambda t: t.entry_time)

    for label, trades in trades_by_symbol.items():
        pnls = [t.net_pnl for t in trades]
        if pnls:
            wins = sum(1 for p in pnls if p > 0)
            logger.info(
                "%s: %d trades, WR %.1f%%, mean $%.2f, total $%.2f",
                label, len(pnls), 100 * wins / len(pnls),
                np.mean(pnls), sum(pnls),
            )
    return trades_by_symbol


# ---------------------------------------------------------------------------
# Step 3-6 — Run Monte Carlo simulations
# ---------------------------------------------------------------------------

def run_scenario(
    label: str,
    trades: list[Trade],
    output_dir: Path,
) -> dict:
    """Run full Monte Carlo simulation for a set of trades and save results."""
    logger.info("=" * 60)
    logger.info("  Scenario: %s (%d trades)", label, len(trades))
    logger.info("=" * 60)

    if len(trades) < 20:
        logger.warning("Too few trades for %s (%d) — skipping", label, len(trades))
        return {"label": label, "skipped": True, "reason": "too few trades"}

    scenario_dir = output_dir / label.lower().replace(" ", "_")
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Save trade list
    export_trades_csv(trades, str(scenario_dir / "trades.csv"))

    # --- Main simulation (block bootstrap) ---
    result = run_monte_carlo(
        trades,
        n_simulations=N_SIMULATIONS,
        max_trades=MAX_TRADES_PER_SIM,
        starting_capital=STARTING_CAPITAL,
        profit_target=PROFIT_TARGET,
        max_drawdown=MAX_DRAWDOWN,
        dd_buffer=DD_BUFFER,
        position_scale=1.0,
        block_size=BLOCK_SIZE,
        n_sample_paths=20,
        seed=SEED,
    )

    print(f"\n{'='*60}")
    print(f"  SCENARIO: {label}")
    print_report(result)

    # --- Sensitivity analysis ---
    sens_result = run_sensitivity(
        trades,
        scales=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
        n_simulations=N_SIMULATIONS,
        max_trades=MAX_TRADES_PER_SIM,
        starting_capital=STARTING_CAPITAL,
        profit_target=PROFIT_TARGET,
        max_drawdown=MAX_DRAWDOWN,
        block_size=BLOCK_SIZE,
        seed=SEED,
    )
    result.sensitivity = sens_result.sensitivity

    # --- Risk diagnostics ---
    risk_diag = compute_risk_diagnostics(result, trades)

    # --- Save everything ---
    # Summary JSON
    summary = result.to_dict()
    summary["label"] = label
    summary["risk_diagnostics"] = risk_diag
    with open(scenario_dir / "simulation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pass/fail distribution CSV
    _save_pass_fail_csv(result, scenario_dir / "pass_fail_distribution.csv")

    # Trades-to-pass histogram CSV
    _save_trades_to_pass_csv(result, scenario_dir / "trades_to_pass_histogram.csv")

    # Plots
    try:
        saved_plots = plot_results(result, output_dir=str(scenario_dir))
        for name, path in saved_plots.items():
            logger.info("  Plot saved: %s → %s", name, path)
    except Exception as e:
        logger.warning("Plot generation failed: %s", e)

    return summary


def compute_risk_diagnostics(result, trades: list[Trade]) -> dict:
    """Compute additional risk diagnostics beyond standard MC output."""
    paths = result.all_paths

    # --- Max losing streak distribution ---
    loss_streaks = [p.max_loss_streak for p in paths]
    loss_streak_arr = np.array(loss_streaks, dtype=float)

    streak_dist = {}
    for threshold in [3, 4, 5, 6, 7, 8, 10]:
        pct = float(np.mean(loss_streak_arr >= threshold))
        streak_dist[f"pct_streak_ge_{threshold}"] = round(pct, 4)

    # --- Probability of hitting drawdown within N trades ---
    early_fail = {}
    for n_trades in [10, 20, 50, 100]:
        fail_within_n = sum(
            1 for p in paths
            if p.outcome == "failed" and p.trades_taken <= n_trades
        )
        early_fail[f"fail_within_{n_trades}_trades"] = round(
            fail_within_n / len(paths), 4
        )

    # --- Source trade statistics ---
    pnls = np.array([t.net_pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    source_stats = {
        "total_trades": len(pnls),
        "win_rate": round(float(len(wins) / len(pnls)), 4) if len(pnls) else 0,
        "avg_win": round(float(np.mean(wins)), 2) if len(wins) else 0,
        "avg_loss": round(float(np.mean(losses)), 2) if len(losses) else 0,
        "largest_win": round(float(np.max(pnls)), 2) if len(pnls) else 0,
        "largest_loss": round(float(np.min(pnls)), 2) if len(pnls) else 0,
        "median_pnl": round(float(np.median(pnls)), 2),
        "std_pnl": round(float(np.std(pnls)), 2),
        "expectancy": round(float(np.mean(pnls)), 2),
        "profit_factor": round(
            float(np.sum(wins) / abs(np.sum(losses))), 2
        ) if len(losses) and np.sum(losses) != 0 else 999.0,
    }

    # --- Outcome breakdown by trades taken ---
    pass_trades = [p.trades_taken for p in paths if p.outcome == "passed"]
    fail_trades = [p.trades_taken for p in paths if p.outcome == "failed"]

    timing = {}
    if pass_trades:
        timing["pass_p10_trades"] = int(np.percentile(pass_trades, 10))
        timing["pass_p25_trades"] = int(np.percentile(pass_trades, 25))
        timing["pass_p75_trades"] = int(np.percentile(pass_trades, 75))
        timing["pass_p90_trades"] = int(np.percentile(pass_trades, 90))
    if fail_trades:
        timing["fail_p10_trades"] = int(np.percentile(fail_trades, 10))
        timing["fail_p25_trades"] = int(np.percentile(fail_trades, 25))
        timing["fail_p75_trades"] = int(np.percentile(fail_trades, 75))
        timing["fail_p90_trades"] = int(np.percentile(fail_trades, 90))

    # --- Peak equity before failure ---
    peak_before_fail = [p.peak_equity for p in paths if p.outcome == "failed"]
    peak_fail_stats = {}
    if peak_before_fail:
        peak_fail_stats = {
            "mean_peak_before_fail": round(float(np.mean(peak_before_fail)), 2),
            "median_peak_before_fail": round(float(np.median(peak_before_fail)), 2),
            "pct_failed_after_profit": round(
                float(np.mean(np.array(peak_before_fail) > STARTING_CAPITAL)), 4
            ),
        }

    return {
        "loss_streak_distribution": streak_dist,
        "early_failure_probability": early_fail,
        "source_trade_stats": source_stats,
        "timing_percentiles": timing,
        "peak_before_fail": peak_fail_stats,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_pass_fail_csv(result, path: Path):
    """Save per-simulation outcome summary."""
    rows = []
    for i, p in enumerate(result.all_paths):
        rows.append({
            "sim_id": i,
            "outcome": p.outcome,
            "trades_taken": p.trades_taken,
            "final_equity": round(p.final_equity, 2),
            "peak_equity": round(p.peak_equity, 2),
            "max_drawdown": round(p.max_drawdown, 2),
            "max_win_streak": p.max_win_streak,
            "max_loss_streak": p.max_loss_streak,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_trades_to_pass_csv(result, path: Path):
    """Save histogram data for trades-to-pass distribution."""
    passed = [p for p in result.all_paths if p.outcome == "passed"]
    if not passed:
        pd.DataFrame({"bin": [], "count": []}).to_csv(path, index=False)
        return

    trades_taken = [p.trades_taken for p in passed]
    bins = list(range(0, max(trades_taken) + 20, 10))
    counts, edges = np.histogram(trades_taken, bins=bins)
    rows = []
    for i in range(len(counts)):
        rows.append({
            "bin_start": int(edges[i]),
            "bin_end": int(edges[i + 1]),
            "count": int(counts[i]),
            "pct": round(float(counts[i] / len(passed)), 4),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_equity_curves_plot(
    combined_result, mes_result, mnq_result, output_dir: Path
):
    """Save equity curve examples: best, worst, median for combined scenario."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping equity curve plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    passed_paths = [p for p in combined_result.all_paths if p.equity_path and p.outcome == "passed"]
    failed_paths = [p for p in combined_result.all_paths if p.equity_path and p.outcome == "failed"]

    target_line = STARTING_CAPITAL + PROFIT_TARGET
    floor_line = STARTING_CAPITAL - MAX_DRAWDOWN

    # Best: fastest pass
    if passed_paths:
        best = min(passed_paths, key=lambda p: p.trades_taken)
        ax = axes[0]
        ax.plot(best.equity_path, color="green", linewidth=1.5)
        ax.axhline(target_line, color="green", linestyle="--", alpha=0.5)
        ax.axhline(floor_line, color="red", linestyle="--", alpha=0.5)
        ax.axhline(STARTING_CAPITAL, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"Best Pass ({best.trades_taken} trades)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity ($)")

    # Worst: most drawdown before pass, or longest fail
    worst_candidates = failed_paths if failed_paths else passed_paths
    if worst_candidates:
        worst = max(worst_candidates, key=lambda p: p.max_drawdown)
        ax = axes[1]
        color = "red" if worst.outcome == "failed" else "orange"
        ax.plot(worst.equity_path, color=color, linewidth=1.5)
        ax.axhline(target_line, color="green", linestyle="--", alpha=0.5)
        ax.axhline(floor_line, color="red", linestyle="--", alpha=0.5)
        ax.axhline(STARTING_CAPITAL, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"Worst ({worst.outcome}, DD ${worst.max_drawdown:.0f})")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity ($)")

    # Median: middle trades-to-pass
    if passed_paths:
        sorted_passed = sorted(passed_paths, key=lambda p: p.trades_taken)
        median_path = sorted_passed[len(sorted_passed) // 2]
        ax = axes[2]
        ax.plot(median_path.equity_path, color="steelblue", linewidth=1.5)
        ax.axhline(target_line, color="green", linestyle="--", alpha=0.5)
        ax.axhline(floor_line, color="red", linestyle="--", alpha=0.5)
        ax.axhline(STARTING_CAPITAL, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"Median Pass ({median_path.trades_taken} trades)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity ($)")

    plt.suptitle("Sample Equity Curves: Best / Worst / Median", fontsize=14)
    plt.tight_layout()
    path = output_dir / "sample_equity_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sample equity curves → %s", path)


# ---------------------------------------------------------------------------
# Step 8 — Final interpretation
# ---------------------------------------------------------------------------

def print_interpretation(scenarios: dict[str, dict]):
    """Print plain-English interpretation of simulation results."""
    print("\n" + "=" * 70)
    print("  FINAL INTERPRETATION — PROP FIRM CHALLENGE FEASIBILITY")
    print("=" * 70)

    combined = scenarios.get("Combined")
    mes = scenarios.get("MES")
    mnq = scenarios.get("MNQ")

    if not combined or combined.get("skipped"):
        print("\n  ERROR: Combined scenario not available.")
        return

    pr = combined["pass_rate"]
    fr = combined["fail_rate"]
    avg_trades = combined["avg_trades_to_pass"]
    med_trades = combined["median_trades_to_pass"]
    dd = combined["drawdown_stats"]
    risk = combined.get("risk_diagnostics", {})

    print(f"\n  Combined MES + MNQ System:")
    print(f"  ─────────────────────────")
    print(f"  Pass Rate:  {pr:.1%}")
    print(f"  Fail Rate:  {fr:.1%}")
    print(f"  Avg trades to pass: {avg_trades:.0f}")
    print(f"  Median trades to pass: {med_trades:.0f}")

    # Per-symbol comparison
    if mes and not mes.get("skipped"):
        print(f"\n  MES Only:   {mes['pass_rate']:.1%} pass rate")
    if mnq and not mnq.get("skipped"):
        print(f"  MNQ Only:   {mnq['pass_rate']:.1%} pass rate")

    # --- Question 1: Is this strategy likely to pass? ---
    print(f"\n  1. IS THIS STRATEGY LIKELY TO PASS THE CHALLENGE?")
    if pr >= 0.70:
        print(f"     YES — {pr:.1%} pass rate is strong. Favorable odds.")
    elif pr >= 0.50:
        print(f"     MAYBE — {pr:.1%} pass rate is marginal. More likely to pass")
        print(f"     than fail, but expect multiple attempts.")
    elif pr >= 0.30:
        print(f"     UNLIKELY — {pr:.1%} pass rate means the challenge is")
        print(f"     asymmetrically unfavorable. Expect several failures.")
    else:
        print(f"     NO — {pr:.1%} pass rate is too low. The $1,000 trailing DD")
        print(f"     is too tight for this strategy's volatility profile.")

    # --- Question 2: Expected attempts ---
    if pr > 0:
        expected_attempts = 1.0 / pr
        print(f"\n  2. EXPECTED NUMBER OF ATTEMPTS TO PASS:")
        print(f"     {expected_attempts:.1f} attempts (geometric distribution)")
        print(f"     P(pass within 3 attempts): {1 - (1-pr)**3:.1%}")
        print(f"     P(pass within 5 attempts): {1 - (1-pr)**5:.1%}")
    else:
        print(f"\n  2. EXPECTED NUMBER OF ATTEMPTS: Infinite (0% pass rate)")

    # --- Question 3: Biggest failure mode ---
    print(f"\n  3. BIGGEST FAILURE MODE:")
    early_fail = risk.get("early_failure_probability", {})
    f10 = early_fail.get("fail_within_10_trades", 0)
    f20 = early_fail.get("fail_within_20_trades", 0)
    f50 = early_fail.get("fail_within_50_trades", 0)
    streak = risk.get("loss_streak_distribution", {})
    peak_fail = risk.get("peak_before_fail", {})

    if f10 > 0.15:
        print(f"     EARLY BLOWOUT — {f10:.1%} of sims fail within 10 trades.")
        print(f"     A single bad streak at the start can end the challenge.")
    elif f20 > 0.25:
        print(f"     EARLY DRAWDOWN — {f20:.1%} fail within first 20 trades.")
    else:
        print(f"     GRADUAL EROSION — failures tend to happen over time,")
        print(f"     not from single catastrophic streaks.")

    pct_fail_after_profit = peak_fail.get("pct_failed_after_profit", 0)
    if pct_fail_after_profit > 0.3:
        print(f"     WARNING: {pct_fail_after_profit:.1%} of failures happen AFTER")
        print(f"     the account was profitable — giving back gains is common.")

    # --- Question 4: Too volatile or too slow? ---
    print(f"\n  4. IS THE SYSTEM TOO VOLATILE OR TOO SLOW?")
    src = risk.get("source_trade_stats", {})
    avg_pnl = src.get("expectancy", 0)
    std_pnl = src.get("std_pnl", 0)
    if std_pnl > 0:
        sharpe_per_trade = avg_pnl / std_pnl
    else:
        sharpe_per_trade = 0

    if med_trades > 200:
        print(f"     TOO SLOW — median {med_trades:.0f} trades to pass suggests")
        print(f"     the edge is thin relative to target. May take months.")
    elif dd["p95_max_dd"] > MAX_DRAWDOWN * 0.95:
        print(f"     TOO VOLATILE — 95th percentile DD (${dd['p95_max_dd']:.0f})")
        print(f"     is nearly at the $1,000 limit. High variance in outcomes.")
    else:
        print(f"     BALANCED — median {med_trades:.0f} trades to pass,")
        print(f"     per-trade Sharpe {sharpe_per_trade:.3f}.")

    # --- Question 5: Position sizing adjustment ---
    print(f"\n  5. SHOULD POSITION SIZING BE ADJUSTED?")
    if combined.get("sensitivity"):
        sens = combined["sensitivity"]
        best_sens = max(sens, key=lambda s: s["pass_rate"])
        current_sens = next((s for s in sens if s["scale"] == 1.0), None)

        if best_sens["scale"] != 1.0:
            print(f"     Current scale (1.0x): {current_sens['pass_rate']:.1%} pass rate" if current_sens else "")
            print(f"     Optimal scale ({best_sens['scale']}x): {best_sens['pass_rate']:.1%} pass rate")
            if best_sens["scale"] < 1.0:
                print(f"     → REDUCE size to {best_sens['scale']}x for better risk/reward")
            else:
                print(f"     → INCREASE size to {best_sens['scale']}x (if risk tolerance allows)")
        else:
            print(f"     Current sizing (1.0x) is already optimal among tested scales.")
    else:
        print(f"     No sensitivity data available.")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Capture all console output
    console_buffer = io.StringIO()

    class TeeWriter:
        def __init__(self, *writers):
            self.writers = writers
        def write(self, text):
            for w in self.writers:
                try:
                    w.write(text)
                except Exception:
                    pass
        def flush(self):
            for w in self.writers:
                try:
                    w.flush()
                except Exception:
                    pass

    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, console_buffer)

    try:
        print("=" * 70)
        print("  PROP FIRM CHALLENGE SIMULATION")
        print(f"  Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Capital: ${STARTING_CAPITAL:,.0f}")
        print(f"  Target:  +${PROFIT_TARGET:,.0f} (equity ${STARTING_CAPITAL + PROFIT_TARGET:,.0f})")
        print(f"  Max DD:  -${MAX_DRAWDOWN:,.0f} (trailing intraday)")
        print(f"  Simulations: {N_SIMULATIONS:,} per scenario")
        print(f"  Block size: {BLOCK_SIZE} trades")
        print(f"  Seed: {SEED}")
        print("=" * 70)

        # Step 1-2: Collect trades
        print("\n[STEP 1-2] Generating trades from AdaptiveRegimeStrategy...")
        trades_by_symbol = collect_all_trades()

        # Step 3-6: Run scenarios
        scenarios: dict[str, dict] = {}

        for label in ["Combined", "MNQ", "MES"]:
            trades = trades_by_symbol.get(label, [])
            if not trades:
                logger.warning("No trades for %s — skipping", label)
                scenarios[label] = {"label": label, "skipped": True}
                continue
            scenarios[label] = run_scenario(label, trades, OUTPUT_DIR)

        # Step 7: Save combined comparison
        _save_comparison(scenarios, OUTPUT_DIR)

        # Custom equity curve plot
        combined_result = None
        if "Combined" in scenarios and not scenarios["Combined"].get("skipped"):
            # Re-run with more sample paths for the plot
            combined_trades = trades_by_symbol["Combined"]
            combined_result = run_monte_carlo(
                combined_trades,
                n_simulations=N_SIMULATIONS,
                max_trades=MAX_TRADES_PER_SIM,
                starting_capital=STARTING_CAPITAL,
                profit_target=PROFIT_TARGET,
                max_drawdown=MAX_DRAWDOWN,
                dd_buffer=DD_BUFFER,
                position_scale=1.0,
                block_size=BLOCK_SIZE,
                n_sample_paths=50,
                seed=SEED,
            )
            _save_equity_curves_plot(
                combined_result,
                scenarios.get("MES"),
                scenarios.get("MNQ"),
                OUTPUT_DIR,
            )

        # Step 8: Interpretation
        print_interpretation(scenarios)

    finally:
        sys.stdout = original_stdout

    # Save console log
    with open(OUTPUT_DIR / "console_log.txt", "w", encoding="utf-8") as f:
        f.write(console_buffer.getvalue())
    logger.info("Console log saved → %s", OUTPUT_DIR / "console_log.txt")
    logger.info("All results saved to %s", OUTPUT_DIR)


def _save_comparison(scenarios: dict[str, dict], output_dir: Path):
    """Save cross-scenario comparison."""
    rows = []
    for label in ["Combined", "MNQ", "MES"]:
        s = scenarios.get(label, {})
        if s.get("skipped"):
            continue
        risk = s.get("risk_diagnostics", {})
        src = risk.get("source_trade_stats", {})
        rows.append({
            "scenario": label,
            "source_trades": s.get("source_trade_count", src.get("total_trades", 0)),
            "source_win_rate": s.get("source_win_rate", src.get("win_rate", 0)),
            "source_mean_pnl": s.get("source_mean_pnl", src.get("expectancy", 0)),
            "pass_rate": s.get("pass_rate", 0),
            "fail_rate": s.get("fail_rate", 0),
            "avg_trades_to_pass": s.get("avg_trades_to_pass", 0),
            "median_trades_to_pass": s.get("median_trades_to_pass", 0),
            "mean_max_dd": s.get("drawdown_stats", {}).get("mean_max_dd", 0),
            "p95_max_dd": s.get("drawdown_stats", {}).get("p95_max_dd", 0),
        })
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "scenario_comparison.csv", index=False)


if __name__ == "__main__":
    main()
