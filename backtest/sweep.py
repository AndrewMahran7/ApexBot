"""
Lookback Sweep for Ranking-Based Trade Selection
==================================================

Systematically evaluates different rolling-window lookback sizes
for the ML ranking modes (top_n, top_pct).  Runs the same strategy
on the same dataset with all other parameters held constant.

Usage (programmatic):
    from backtest.sweep import run_lookback_sweep
    results = run_lookback_sweep(bars, base_cfg, instrument, bt_cfg, lookbacks)

Data leakage safety:
    Each lookback run creates a FRESH strategy instance with its own
    rolling window.  The window only includes PRIOR session scores —
    the current candidate is never included before the accept/reject
    decision.  Cold-start behaviour is unchanged.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import compute_metrics
from config.settings import InstrumentConfig, StrategyConfig, BacktestConfig
from strategy.hybrid_ema_ml import HybridEMAMLStrategy, HybridEMAMLConfig


def _compute_ranking_diagnostics(decisions: list[dict]) -> dict:
    """
    Compute ranking diagnostics from a strategy's ml_decisions log.

    Returns
    -------
    dict with keys:
        candidates       : total EMA candidates evaluated
        accepted         : number accepted by ML filter
        acceptance_rate  : accepted / candidates (0-1)
        prob_mean        : mean ML probability across all candidates
        prob_std         : std of ML probabilities
        avg_percentile   : average percentile rank of accepted trades
                           within their respective rolling windows
                           (NaN if no ranked acceptances)
        avg_position_size: average position size of accepted trades
    """
    if not decisions:
        return {
            "candidates": 0,
            "accepted": 0,
            "acceptance_rate": 0.0,
            "prob_mean": 0.0,
            "prob_std": 0.0,
            "avg_percentile": float("nan"),
            "avg_position_size": float("nan"),
        }

    n = len(decisions)
    n_accepted = sum(1 for d in decisions if d["accepted"])
    all_probs = [d["ml_prob"] for d in decisions]

    # Percentile of accepted trades within their rolling window.
    # rank=1 means top of window → percentile = 100.
    # rank=window_size → percentile ≈ 0.
    percentiles = []
    sizes = []
    for d in decisions:
        if d["accepted"]:
            sizes.append(d.get("position_size", 1.0))
            if d["window_size"] > 0:
                pctile = (1.0 - (d["rank"] - 1) / d["window_size"]) * 100
                percentiles.append(pctile)

    return {
        "candidates": n,
        "accepted": n_accepted,
        "acceptance_rate": round(n_accepted / n, 4) if n else 0.0,
        "prob_mean": round(float(np.mean(all_probs)), 4),
        "prob_std": round(float(np.std(all_probs)), 4),
        "avg_percentile": round(float(np.mean(percentiles)), 2) if percentiles else float("nan"),
        "avg_position_size": round(float(np.mean(sizes)), 4) if sizes else float("nan"),
    }


def run_lookback_sweep(
    bars: pd.DataFrame,
    base_hybrid_cfg: HybridEMAMLConfig,
    base_strat_cfg: StrategyConfig,
    instrument: InstrumentConfig,
    bt_cfg: BacktestConfig,
    lookbacks: list[int],
    eval_config=None,
) -> list[dict]:
    """
    Run the hybrid EMA+ML strategy across multiple lookback windows.

    Parameters
    ----------
    bars : DataFrame
        OHLCV data (already filtered to the desired split).
    base_hybrid_cfg : HybridEMAMLConfig
        Base config — ml_lookback will be overridden per run.
    base_strat_cfg : StrategyConfig
        For benchmark compatibility (passed to engine).
    instrument : InstrumentConfig
    bt_cfg : BacktestConfig
    lookbacks : list[int]
        Lookback window sizes to test.
    eval_config : EvalConfig, optional

    Returns
    -------
    list[dict]
        One row per lookback with metrics + ranking diagnostics.
    """
    results = []

    for lb in lookbacks:
        # Create fresh config with this lookback
        cfg = copy.copy(base_hybrid_cfg)
        cfg.ml_lookback = lb

        # Fresh strategy — new rolling window with correct maxlen
        strategy = HybridEMAMLStrategy(cfg)

        engine = BacktestEngine(instrument, base_strat_cfg, bt_cfg, strategy=strategy)
        bt_result = engine.run(bars, eval_config=eval_config)

        metrics = compute_metrics(bt_result, bt_cfg)
        diagnostics = _compute_ranking_diagnostics(strategy.ml_decisions)

        row = {
            "lookback": lb,
            # Core metrics
            "trades": metrics["total_trades"],
            "win_rate": metrics["win_rate_pct"],
            "profit_factor": metrics["profit_factor"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown_dollars"],
            "total_return": metrics["total_return_pct"],
            "expectancy": metrics["expectancy"],
            # Ranking diagnostics
            "candidates": diagnostics["candidates"],
            "accepted": diagnostics["accepted"],
            "acceptance_rate": diagnostics["acceptance_rate"],
            "prob_mean": diagnostics["prob_mean"],
            "prob_std": diagnostics["prob_std"],
            "avg_percentile": diagnostics["avg_percentile"],
            "avg_position_size": diagnostics["avg_position_size"],
        }
        results.append(row)

    return results


def print_sweep_table(rows: list[dict]):
    """Print a formatted comparison table for the lookback sweep."""
    if not rows:
        print("No sweep results to display.")
        return

    # Check if position sizing is active (any non-NaN avg_position_size != 1.0)
    has_sizing = any(
        not np.isnan(r.get("avg_position_size", float("nan")))
        and r.get("avg_position_size", 1.0) != 1.0
        for r in rows
    )

    print()
    print("=" * 105)
    print("  LOOKBACK SWEEP RESULTS")
    print("=" * 105)

    header = (
        f"  {'LB':>4s}  {'Trades':>6s}  {'WR%':>6s}  {'PF':>6s}  "
        f"{'Sharpe':>7s}  {'MaxDD':>9s}  {'Ret%':>7s}  {'Expect':>8s}  "
        f"{'Acpt%':>6s}  {'ProbMu':>6s}  {'ProbSD':>6s}  {'AvgPct':>6s}"
    )
    if has_sizing:
        header += f"  {'AvgSz':>6s}"
    sep = "  " + "-" * (101 + (8 if has_sizing else 0))
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        avg_pct = f"{r['avg_percentile']:>5.1f}%" if not np.isnan(r["avg_percentile"]) else "   N/A"
        line = (
            f"  {r['lookback']:>4d}  {r['trades']:>6d}  {r['win_rate']:>5.1f}%  "
            f"{r['profit_factor']:>6.2f}  {r['sharpe_ratio']:>7.2f}  "
            f"${r['max_drawdown']:>7,.0f}  {r['total_return']:>6.1f}%  "
            f"${r['expectancy']:>7,.2f}  "
            f"{r['acceptance_rate']*100:>5.1f}%  "
            f"{r['prob_mean']:>6.3f}  {r['prob_std']:>6.3f}  {avg_pct}"
        )
        if has_sizing:
            avg_sz = r.get("avg_position_size", float("nan"))
            sz_str = f"{avg_sz:>5.3f}" if not np.isnan(avg_sz) else "  N/A"
            line += f"  {sz_str}"
        print(line)

    print(sep)
    print()


def export_sweep_csv(rows: list[dict], path: str):
    """Write sweep results to CSV."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Lookback sweep results saved to {path}")
