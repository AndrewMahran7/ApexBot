#!/usr/bin/env python3
"""
Prop Risk Layer Validation
============================

Compares challenge pass probability between:
  1. Fixed-size baseline (current system, no risk layer)
  2. Dynamic-sizing with PropRiskLayer (challenge mode)
  3. Funded-mode behavior (separate report)

Uses the same AdaptiveRegimeStrategy trades (no strategy changes)
and the same Monte Carlo framework (block bootstrap, trailing DD).

Usage:
    .\\venv\\Scripts\\python.exe run_prop_risk_layer_validation.py
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))

import datetime
import io
import json
import logging
import math
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
from backtest.metrics import export_trades_csv
from strategy.adaptive_regime import AdaptiveRegimeStrategy
from risk.prop_risk_layer import PropRiskConfig, AccountMode
from challenge.monte_carlo import (
    run_monte_carlo,
    run_sensitivity,
    plot_results,
    print_report,
    MonteCarloResult,
)
from challenge.dynamic_monte_carlo import run_dynamic_monte_carlo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prop_risk_validation")
logger.setLevel(logging.INFO)

SYMBOLS = ["MES", "MNQ"]
OUTPUT_DIR = Path("results/prop_risk_layer_validation")

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=25_000.0,
    slippage_ticks=1,
    commission_per_side=2.25,
)

# MC constants (same as prior simulation for fair comparison)
N_SIMS = 2000
MAX_TRADES = 500
BLOCK_SIZE = 7
SEED = 42
STARTING_CAPITAL = 25_000.0
PROFIT_TARGET = 1_500.0
MAX_DRAWDOWN = 1_000.0
DD_BUFFER = 200.0

# ---------------------------------------------------------------------------
# Data discovery (reused from prior runner)
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
        raise FileNotFoundError(f"No data for {symbol} {year}")
    info = YEAR_DATA[key]
    bars = load_bars(info["path"])
    if info.get("filter"):
        y = info["year"]
        start = pd.Timestamp(f"{y}-01-01", tz="America/New_York")
        end = pd.Timestamp(f"{y}-12-31 23:59:59", tz="America/New_York")
        bars = bars[(bars.index >= start) & (bars.index <= end)]
    return bars


def collect_all_trades() -> dict[str, list[Trade]]:
    """Run AdaptiveRegimeStrategy across all years/symbols."""
    trades_by_symbol: dict[str, list[Trade]] = {"MES": [], "MNQ": [], "Combined": []}
    available_years = sorted(set(y for (y, _) in YEAR_DATA.keys()))

    for year in available_years:
        for symbol in SYMBOLS:
            key = (year, symbol)
            if key not in YEAR_DATA:
                continue
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
            logger.info("  %s %d: %d trades, $%.2f", symbol, year, n, pnl)
            trades_by_symbol[symbol].extend(result.trades)
            trades_by_symbol["Combined"].extend(result.trades)

    trades_by_symbol["Combined"].sort(key=lambda t: t.entry_time)
    for label, trades in trades_by_symbol.items():
        if trades:
            pnls = [t.net_pnl for t in trades]
            logger.info("%s: %d trades, mean $%.2f", label, len(pnls), np.mean(pnls))
    return trades_by_symbol


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def run_baseline(trades: list[Trade], label: str) -> MonteCarloResult:
    """Fixed-size baseline (no risk layer)."""
    return run_monte_carlo(
        trades,
        n_simulations=N_SIMS,
        max_trades=MAX_TRADES,
        starting_capital=STARTING_CAPITAL,
        profit_target=PROFIT_TARGET,
        max_drawdown=MAX_DRAWDOWN,
        dd_buffer=DD_BUFFER,
        position_scale=1.0,
        block_size=BLOCK_SIZE,
        n_sample_paths=20,
        seed=SEED,
    )


def run_dynamic_challenge(trades: list[Trade], label: str) -> MonteCarloResult:
    """Dynamic sizing with PropRiskLayer â€” symbol-aware config."""
    if label == "MES":
        config = PropRiskConfig.for_challenge_mes(
            starting_capital=STARTING_CAPITAL,
            profit_target=PROFIT_TARGET,
            max_drawdown=MAX_DRAWDOWN,
        )
    else:
        config = PropRiskConfig.for_challenge(
            starting_capital=STARTING_CAPITAL,
            profit_target=PROFIT_TARGET,
            max_drawdown=MAX_DRAWDOWN,
        )
    return run_dynamic_monte_carlo(
        trades,
        risk_config=config,
        n_simulations=N_SIMS,
        max_trades=MAX_TRADES,
        dd_buffer=DD_BUFFER,
        block_size=BLOCK_SIZE,
        n_sample_paths=20,
        seed=SEED,
    )


def run_dynamic_challenge_extended(trades: list[Trade], label: str) -> MonteCarloResult:
    """Dynamic sizing with extended 1500-trade horizon to show convergence."""
    if label == "MES":
        config = PropRiskConfig.for_challenge_mes(
            starting_capital=STARTING_CAPITAL,
            profit_target=PROFIT_TARGET,
            max_drawdown=MAX_DRAWDOWN,
        )
    else:
        config = PropRiskConfig.for_challenge(
            starting_capital=STARTING_CAPITAL,
            profit_target=PROFIT_TARGET,
            max_drawdown=MAX_DRAWDOWN,
        )
    return run_dynamic_monte_carlo(
        trades,
        risk_config=config,
        n_simulations=N_SIMS,
        max_trades=1500,  # extended horizon
        dd_buffer=DD_BUFFER,
        block_size=BLOCK_SIZE,
        n_sample_paths=20,
        seed=SEED,
    )


def run_funded_simulation(trades: list[Trade], label: str) -> MonteCarloResult:
    """Funded mode simulation â€” conservative preservation."""
    config = PropRiskConfig.for_funded(
        starting_capital=STARTING_CAPITAL,
        max_drawdown=MAX_DRAWDOWN,
    )
    return run_dynamic_monte_carlo(
        trades,
        risk_config=config,
        n_simulations=N_SIMS,
        max_trades=MAX_TRADES,
        dd_buffer=DD_BUFFER,
        block_size=BLOCK_SIZE,
        n_sample_paths=20,
        seed=SEED,
    )


# ---------------------------------------------------------------------------
# Giveback analysis
# ---------------------------------------------------------------------------

def compute_giveback_rate(result: MonteCarloResult) -> dict:
    """What fraction of failures happened after the account was profitable?"""
    failed = [p for p in result.all_paths if p.outcome == "failed"]
    if not failed:
        return {"total_failures": 0, "failures_after_profit": 0, "giveback_rate": 0.0}
    after_profit = sum(1 for p in failed if p.peak_equity > STARTING_CAPITAL)
    return {
        "total_failures": len(failed),
        "failures_after_profit": after_profit,
        "giveback_rate": round(after_profit / len(failed), 4),
    }


def compute_early_fail_rates(result: MonteCarloResult) -> dict:
    """P(fail) within first N trades."""
    total = len(result.all_paths)
    out = {}
    for n in [10, 20, 50, 100]:
        fails = sum(
            1 for p in result.all_paths
            if p.outcome == "failed" and p.trades_taken <= n
        )
        out[f"fail_within_{n}"] = round(fails / total, 4)
    return out


# ---------------------------------------------------------------------------
# Summary & comparison
# ---------------------------------------------------------------------------

def build_comparison_row(label: str, scenario: str, result: MonteCarloResult) -> dict:
    """Build one row for the comparison table."""
    giveback = compute_giveback_rate(result)
    early = compute_early_fail_rates(result)
    dd = result.drawdown_stats

    return {
        "scenario": f"{label}_{scenario}",
        "label": label,
        "mode": scenario,
        "source_trades": result.source_trade_count,
        "pass_rate": result.pass_rate,
        "fail_rate": result.fail_rate,
        "incomplete_rate": result.incomplete_rate,
        "avg_trades_to_pass": result.avg_trades_to_pass,
        "median_trades_to_pass": result.median_trades_to_pass,
        "avg_trades_to_fail": result.avg_trades_to_fail,
        "mean_max_dd": dd.mean_max_dd,
        "p95_max_dd": dd.p95_max_dd,
        "p99_max_dd": dd.p99_max_dd,
        "mean_loss_streak": dd.mean_largest_loss_streak,
        "giveback_rate": giveback["giveback_rate"],
        "fail_within_10": early.get("fail_within_10", 0),
        "fail_within_20": early.get("fail_within_20", 0),
        "fail_within_50": early.get("fail_within_50", 0),
        "mean_final_equity": result.mean_final_equity,
    }


def print_comparison(rows: list[dict]):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("  BEFORE vs AFTER COMPARISON: Fixed Size vs Dynamic Risk Layer")
    print("=" * 90)

    headers = [
        ("Scenario", "scenario", "<30"),
        ("Pass%", "pass_rate", ">7"),
        ("Fail%", "fail_rate", ">7"),
        ("AvgTrd", "avg_trades_to_pass", ">7"),
        ("MedTrd", "median_trades_to_pass", ">7"),
        ("MeanDD", "mean_max_dd", ">8"),
        ("P95 DD", "p95_max_dd", ">8"),
        ("Givebk", "giveback_rate", ">8"),
        ("F<20", "fail_within_20", ">6"),
    ]

    header_line = "  ".join(f"{h:{fmt}}" for h, _, fmt in headers)
    print(f"\n  {header_line}")
    print("  " + "-" * len(header_line))

    for row in rows:
        vals = []
        for h, key, fmt in headers:
            v = row.get(key, "")
            if isinstance(v, float):
                if "rate" in key or key.startswith("fail_within") or key.startswith("giveb"):
                    v = f"{v:.1%}"
                elif "dd" in key.lower() or "equity" in key.lower():
                    v = f"${v:,.0f}"
                else:
                    v = f"{v:.0f}"
            vals.append(f"{v:{fmt}}")
        print("  " + "  ".join(vals))

    print()


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def print_interpretation(rows: list[dict], funded_result: Optional[MonteCarloResult]):
    """Print final interpretation using actual config values."""
    print("\n" + "=" * 70)
    print("  FINAL INTERPRETATION")
    print("=" * 70)

    # Find baseline and dynamic for Combined
    baseline = next((r for r in rows if "Combined_baseline" in r["scenario"]), None)
    dynamic = next((r for r in rows if "Combined_dynamic_challenge" in r["scenario"]), None)

    if not baseline or not dynamic:
        print("  ERROR: Missing baseline or dynamic Combined results")
        return

    bp = baseline["pass_rate"]
    dp = dynamic["pass_rate"]
    delta = dp - bp

    print(f"\n  1. DYNAMIC SIZING CONFIG")
    cfg_default = PropRiskConfig.for_challenge(
        starting_capital=STARTING_CAPITAL,
        profit_target=PROFIT_TARGET,
        max_drawdown=MAX_DRAWDOWN,
    )
    cfg_mes = PropRiskConfig.for_challenge_mes(
        starting_capital=STARTING_CAPITAL,
        profit_target=PROFIT_TARGET,
        max_drawdown=MAX_DRAWDOWN,
    )
    print(f"     Combined/MNQ: base={cfg_default.base_size:.2f}x, "
          f"DD prox={'ON' if cfg_default.dd_caution_zone > 0 else 'OFF'}, "
          f"daily loss=${cfg_default.daily_loss_limit:.0f}")
    print(f"     MES:          base={cfg_mes.base_size:.2f}x, "
          f"DD prox={cfg_mes.dd_caution_zone:.0f} zone/{cfg_mes.dd_min_size:.2f}x floor, "
          f"daily loss=${cfg_mes.daily_loss_limit:.0f}")

    print(f"\n  2. DESIGN RATIONALE")
    print(f"     - Full sensitivity grid: tested 0.25x, 0.35x, 0.50x, 1.0x per symbol")
    print(f"     - Combined/MNQ optimal: 0.35x flat â†’ 63-68% pass rate")
    print(f"     - MES optimal: 0.50x + DD proximity â†’ 68% @ 500 trades, 88% @ 1500")
    print(f"     - DD proximity HURTS high-variance instruments (MNQ) â€” creates recovery trap")

    print(f"\n  3. DID PASS RATE IMPROVE?")
    print(f"     Baseline:    {bp:.1%}")
    print(f"     Dynamic:     {dp:.1%}")
    print(f"     Delta:       {delta:+.1%}")
    inc = dynamic.get("incomplete_rate", 0)
    if inc > 0.01:
        resolved = dp + dynamic["fail_rate"]
        cond_pass = dp / resolved if resolved > 0 else 0
        print(f"     Incomplete:  {inc:.1%}")
        print(f"     P(pass|resolved): {cond_pass:.1%} (vs baseline {bp:.1%})")
    if delta > 0.02:
        print(f"     YES â€” improvement of {delta:+.1%}")
    elif delta > -0.02:
        print(f"     COMPARABLE â€” within noise ({delta:+.1%})")
    else:
        print(f"     NO â€” pass rate decreased by {abs(delta):.1%}")

    print(f"\n  4. FAILURE ANALYSIS")
    bg = baseline["giveback_rate"]
    dg = dynamic["giveback_rate"]
    print(f"     Giveback rate: {bg:.1%} â†’ {dg:.1%} ({dg - bg:+.1%})")
    bf20 = baseline["fail_within_20"]
    df20 = dynamic["fail_within_20"]
    print(f"     Early fail (<20 trades): {bf20:.1%} â†’ {df20:.1%}")
    bf = baseline["fail_rate"]
    df = dynamic["fail_rate"]
    print(f"     Total fail rate: {bf:.1%} â†’ {df:.1%} ({df - bf:+.1%})")
    bdd = baseline["mean_max_dd"]
    ddd = dynamic["mean_max_dd"]
    print(f"     Mean max DD: ${bdd:,.0f} â†’ ${ddd:,.0f}")
    bp95 = baseline["p95_max_dd"]
    dp95 = dynamic["p95_max_dd"]
    print(f"     P95 DD: ${bp95:,.0f} â†’ ${dp95:,.0f}")

    print(f"\n  5. TIMING")
    bat = baseline["avg_trades_to_pass"]
    dat = dynamic["avg_trades_to_pass"]
    print(f"     Avg trades to pass: {bat:.0f} â†’ {dat:.0f}")
    bmt = baseline["median_trades_to_pass"]
    dmt = dynamic["median_trades_to_pass"]
    print(f"     Median trades to pass: {bmt:.0f} â†’ {dmt:.0f}")

    print(f"\n  6. FUNDED MODE")
    if funded_result:
        fr = funded_result.fail_rate
        fdd = funded_result.drawdown_stats.p95_max_dd
        finc = funded_result.incomplete_rate
        fmfe = funded_result.mean_final_equity
        print(f"     Fail rate: {fr:.1%} (at {MAX_TRADES} trade horizon)")
        print(f"     Incomplete: {finc:.1%}")
        print(f"     P95 DD: ${fdd:,.0f}")
        print(f"     Mean final equity: ${fmfe:,.0f}")
    else:
        print(f"     (not available)")

    print(f"\n  7. RECOMMENDATIONS")
    if dp > bp * 1.05:
        print(f"     USE the dynamic config â€” clear improvement")
    elif dp > bp * 0.95:
        print(f"     CONSIDER the dynamic config â€” similar pass rate with lower DD risk")
    else:
        print(f"     INVESTIGATE further â€” pass rate regressed, tune parameters")
    if dp > 0:
        attempts = 1.0 / dp
        cost_per = 20  # typical challenge fee
        print(f"     Expected attempts to pass: {attempts:.1f} (est. cost: ${math.ceil(attempts) * cost_per:,.0f})")
    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        print("  PROP RISK LAYER VALIDATION")
        print(f"  Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Capital: ${STARTING_CAPITAL:,.0f}  Target: +${PROFIT_TARGET:,.0f}  DD: -${MAX_DRAWDOWN:,.0f}")
        print(f"  Simulations: {N_SIMS:,} per scenario, block size {BLOCK_SIZE}")
        print("=" * 70)

        # --- Collect trades ---
        print("\n[STEP 1] Generating trades from AdaptiveRegimeStrategy...")
        trades_by_symbol = collect_all_trades()

        comparison_rows: list[dict] = []
        all_results: dict[str, MonteCarloResult] = {}

        # --- Run per-scenario: Combined, MES, MNQ ---
        for label in ["Combined", "MES", "MNQ"]:
            trades = trades_by_symbol.get(label, [])
            if not trades or len(trades) < 20:
                logger.warning("Skipping %s â€” too few trades (%d)", label, len(trades))
                continue

            print(f"\n{'='*60}")
            print(f"  SCENARIO: {label} ({len(trades)} trades)")
            print(f"{'='*60}")

            # 1. Fixed-size baseline
            print(f"\n  --- Baseline (fixed size 1.0x) ---")
            baseline = run_baseline(trades, label)
            print_report(baseline)
            all_results[f"{label}_baseline"] = baseline
            comparison_rows.append(build_comparison_row(label, "baseline", baseline))

            # 2. Fixed-size alternatives (optimal sizing search)
            for scale in [0.25, 0.35, 0.50]:
                tag = f"fixed_{scale:.2f}x"
                print(f"\n  --- Fixed size {scale:.2f}x ---")
                fixed_result = run_monte_carlo(
                    trades,
                    n_simulations=N_SIMS,
                    max_trades=MAX_TRADES,
                    starting_capital=STARTING_CAPITAL,
                    profit_target=PROFIT_TARGET,
                    max_drawdown=MAX_DRAWDOWN,
                    dd_buffer=DD_BUFFER,
                    position_scale=scale,
                    block_size=BLOCK_SIZE,
                    n_sample_paths=5,
                    seed=SEED,
                )
                print_report(fixed_result)
                all_results[f"{label}_{tag}"] = fixed_result
                comparison_rows.append(build_comparison_row(label, tag, fixed_result))

            # 3. Dynamic challenge mode
            print(f"\n  --- Dynamic Challenge Mode ---")
            dynamic = run_dynamic_challenge(trades, label)
            print_report(dynamic)
            all_results[f"{label}_dynamic_challenge"] = dynamic
            comparison_rows.append(build_comparison_row(label, "dynamic_challenge", dynamic))

            # 4. Dynamic challenge â€” extended horizon (1500 trades)
            print(f"\n  --- Dynamic Challenge Mode (extended 1500 trades) ---")
            dynamic_ext = run_dynamic_challenge_extended(trades, label)
            print_report(dynamic_ext)
            all_results[f"{label}_dynamic_extended"] = dynamic_ext
            comparison_rows.append(build_comparison_row(label, "dynamic_extended", dynamic_ext))

        # --- Funded mode (Combined only) ---
        combined_trades = trades_by_symbol.get("Combined", [])
        funded_result = None
        if combined_trades and len(combined_trades) >= 20:
            print(f"\n{'='*60}")
            print(f"  FUNDED MODE SIMULATION (Combined, {len(combined_trades)} trades)")
            print(f"{'='*60}")
            funded_result = run_funded_simulation(combined_trades, "Combined")
            print_report(funded_result)
            all_results["Combined_funded"] = funded_result

        # --- Comparison output ---
        print_comparison(comparison_rows)

        # --- Per-scenario plots ---
        for key, result in all_results.items():
            scenario_dir = OUTPUT_DIR / key
            scenario_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_results(result, output_dir=str(scenario_dir))
            except Exception as e:
                logger.warning("Plot failed for %s: %s", key, e)

        # --- Save equity curve comparison plot ---
        _save_comparison_equity_plot(all_results, OUTPUT_DIR)

        # --- Save all data ---
        _save_results(comparison_rows, all_results, funded_result, OUTPUT_DIR)

        # --- Interpretation ---
        print_interpretation(comparison_rows, funded_result)

    finally:
        sys.stdout = original_stdout

    # Console log
    with open(OUTPUT_DIR / "console_log.txt", "w", encoding="utf-8") as f:
        f.write(console_buffer.getvalue())
    logger.info("All results saved to %s", OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_results(
    comparison_rows: list[dict],
    all_results: dict[str, MonteCarloResult],
    funded_result,
    output_dir: Path,
):
    """Save all structured outputs."""
    # Comparison CSV
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "challenge_before_vs_after.csv", index=False
    )

    # Comparison JSON
    summary = {
        "generated": datetime.datetime.now().isoformat(),
        "config": {
            "starting_capital": STARTING_CAPITAL,
            "profit_target": PROFIT_TARGET,
            "max_drawdown": MAX_DRAWDOWN,
            "n_simulations": N_SIMS,
            "block_size": BLOCK_SIZE,
            "seed": SEED,
        },
        "scenarios": comparison_rows,
    }

    # Add per-scenario summaries
    for key, result in all_results.items():
        summary[key] = result.to_dict()

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pass/fail distribution for dynamic challenge
    for key, result in all_results.items():
        scenario_dir = output_dir / key
        scenario_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, p in enumerate(result.all_paths):
            rows.append({
                "sim_id": i,
                "outcome": p.outcome,
                "trades_taken": p.trades_taken,
                "final_equity": round(p.final_equity, 2),
                "peak_equity": round(p.peak_equity, 2),
                "max_drawdown": round(p.max_drawdown, 2),
            })
        pd.DataFrame(rows).to_csv(scenario_dir / "pass_fail_distribution.csv", index=False)

    # Funded mode JSON
    if funded_result:
        with open(output_dir / "funded_mode_summary.json", "w") as f:
            json.dump(funded_result.to_dict(), indent=2, fp=f)

    # Risk layer config documentation
    challenge_cfg = PropRiskConfig.for_challenge()
    funded_cfg = PropRiskConfig.for_funded()
    config_doc = {
        "challenge_mode": {
            "base_size": challenge_cfg.base_size,
            "dd_caution_zone": challenge_cfg.dd_caution_zone,
            "dd_min_size": challenge_cfg.dd_min_size,
            "streak_threshold": challenge_cfg.streak_threshold,
            "streak_reduction_per": challenge_cfg.streak_reduction_per,
            "profit_lock_threshold": challenge_cfg.profit_lock_threshold,
            "profit_lock_size": challenge_cfg.profit_lock_size,
            "giveback_halt_amount": challenge_cfg.giveback_halt_amount,
            "daily_loss_limit": challenge_cfg.daily_loss_limit,
            "progress_zones": challenge_cfg.progress_zones,
            "symbol_challenge_preference": challenge_cfg.symbol_challenge_preference,
        },
        "funded_mode": {
            "base_size": funded_cfg.funded_base_size,
            "dd_caution_zone": funded_cfg.dd_caution_zone,
            "dd_min_size": funded_cfg.dd_min_size,
            "profit_lock_threshold": funded_cfg.funded_profit_lock_threshold,
            "profit_lock_size": funded_cfg.funded_profit_lock_size,
            "giveback_halt_amount": funded_cfg.funded_giveback_halt_amount,
            "daily_loss_limit": funded_cfg.funded_daily_loss_limit,
        },
    }
    with open(output_dir / "risk_layer_config.json", "w") as f:
        json.dump(config_doc, f, indent=2)


def _save_comparison_equity_plot(
    all_results: dict[str, MonteCarloResult],
    output_dir: Path,
):
    """Side-by-side equity path comparison: baseline vs dynamic."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    baseline = all_results.get("Combined_baseline")
    dynamic = all_results.get("Combined_dynamic_challenge")

    if not baseline or not dynamic:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    target = STARTING_CAPITAL + PROFIT_TARGET
    floor = STARTING_CAPITAL - MAX_DRAWDOWN

    for ax, result, title in [
        (axes[0], baseline, "Baseline (Fixed 1.0x)"),
        (axes[1], dynamic, "Dynamic Risk Layer"),
    ]:
        for sp in result.sample_paths:
            if not sp.equity_path:
                continue
            color = "green" if sp.outcome == "passed" else "red"
            ax.plot(sp.equity_path, color=color, alpha=0.4, linewidth=0.8)

        ax.axhline(target, color="green", linestyle="--", linewidth=2, alpha=0.5)
        ax.axhline(floor, color="red", linestyle="--", linewidth=2, alpha=0.5)
        ax.axhline(STARTING_CAPITAL, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity ($)")
        ax.set_title(f"{title}\nPass: {result.pass_rate:.1%}  Fail: {result.fail_rate:.1%}")

    plt.suptitle("Baseline vs Dynamic Risk Layer â€” Sample Equity Paths", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "comparison_equity_paths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
