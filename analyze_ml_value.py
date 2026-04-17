#!/usr/bin/env python3
"""
ML Value Analysis for Multi-Candidate Trading System
======================================================

Determines whether ML ranking actually improves performance by
comparing selection strategies on the same candidate pool:

  A. No-ML baseline       — take the first N candidates (arrival order)
  B. ML-ranked (global)   — rank by ML probability, take top N
  C. Priority hybrid      — group by type priority (breakout > momentum > pullback),
                             ML ranks within groups
  D. Priority ML-sizing   — priority ordering, always enter, ML for sizing only
  E. Random baseline       — randomly pick N candidates per day

Also produces:
  - Percentile bucket analysis (monotonic improvement test)
  - Correlation(percentile, trade PnL)
  - Strategy-type breakdown for each variant
  - Size-weighted return diagnostics
  - ML contribution metric

Usage:
    python analyze_ml_value.py --data data/mes_4y.csv --split holdout
    python analyze_ml_value.py --data data/mes_4y.csv --start 2024-08-09 --end 2024-12-30
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult, Trade
from backtest.metrics import compute_metrics
from config.settings import InstrumentConfig, StrategyConfig, BacktestConfig
from data.loader import load_bars
from strategy.hybrid_ema_ml import (
    HybridEMAMLConfig,
    HybridEMAMLStrategy,
    STRATEGY_PRIORITY,
    TradeCandidate,
)


# ──────────────────────────────────────────────────────────────
# Strategy variants that override selection logic
# ──────────────────────────────────────────────────────────────

class NoMLStrategy(HybridEMAMLStrategy):
    """Takes the first N candidates in generation order (no ML ranking)."""

    def _rank_and_select(self, candidates: list[TradeCandidate]) -> list[TradeCandidate]:
        if not candidates:
            return []

        # Do NOT sort by ml_prob — keep generation order
        max_new = self.cfg.max_trades_per_day - len(self._open_positions)
        if max_new <= 0:
            for c in candidates:
                self._log_multi_decision(c, accepted=False,
                                         reject_reason="max_trades_per_day reached")
                self._append_prob_window(c.ml_prob, c.direction)
            return []

        selected: list[TradeCandidate] = []
        for cand in candidates:
            # Still compute percentile/size for diagnostics uniformity
            percentile = self._compute_percentile(cand.ml_prob, cand.direction)
            cand.percentile = percentile
            cand.position_size = 1.0  # uniform sizing

            accepted = len(selected) < max_new
            self._log_multi_decision(
                cand, accepted=accepted,
                reject_reason="" if accepted else "max_trades_per_day reached",
            )
            self._append_prob_window(cand.ml_prob, cand.direction)
            if accepted:
                selected.append(cand)

        return selected


class RandomStrategy(HybridEMAMLStrategy):
    """Randomly selects N candidates per day."""

    def __init__(self, config: HybridEMAMLConfig, seed: int = 42):
        super().__init__(config)
        self._rng = random.Random(seed)

    def _rank_and_select(self, candidates: list[TradeCandidate]) -> list[TradeCandidate]:
        if not candidates:
            return []

        # Shuffle randomly
        shuffled = list(candidates)
        self._rng.shuffle(shuffled)

        max_new = self.cfg.max_trades_per_day - len(self._open_positions)
        if max_new <= 0:
            for c in shuffled:
                self._log_multi_decision(c, accepted=False,
                                         reject_reason="max_trades_per_day reached")
                self._append_prob_window(c.ml_prob, c.direction)
            return []

        selected: list[TradeCandidate] = []
        for cand in shuffled:
            percentile = self._compute_percentile(cand.ml_prob, cand.direction)
            cand.percentile = percentile
            cand.position_size = 1.0

            accepted = len(selected) < max_new
            self._log_multi_decision(
                cand, accepted=accepted,
                reject_reason="" if accepted else "max_trades_per_day reached",
            )
            self._append_prob_window(cand.ml_prob, cand.direction)
            if accepted:
                selected.append(cand)

        return selected


class HybridPriorityStrategy(HybridEMAMLStrategy):
    """Priority-based selection: uses 'priority' selection_strategy."""

    def __init__(self, config: HybridEMAMLConfig):
        cfg = copy.copy(config)
        cfg.selection_strategy = "priority"
        super().__init__(cfg)


class PriorityMLSizingStrategy(HybridEMAMLStrategy):
    """Priority ordering, always enter, ML only for sizing."""

    def __init__(self, config: HybridEMAMLConfig):
        cfg = copy.copy(config)
        cfg.selection_strategy = "priority_ml_sizing"
        super().__init__(cfg)


# ──────────────────────────────────────────────────────────────
# Run backtest helper
# ──────────────────────────────────────────────────────────────

def run_variant(
    bars: pd.DataFrame,
    strategy: HybridEMAMLStrategy,
    label: str,
) -> tuple[BacktestResult, dict, list[dict]]:
    """Run a backtest and return (result, metrics, ml_decisions)."""
    inst = InstrumentConfig()
    bt_cfg = BacktestConfig()
    strat_cfg = StrategyConfig(timezone="America/New_York")
    engine = BacktestEngine(inst, strat_cfg, bt_cfg, strategy=strategy)
    result = engine.run(bars)
    metrics = compute_metrics(result, bt_cfg)
    decisions = list(strategy.ml_decisions)
    return result, metrics, decisions


# ──────────────────────────────────────────────────────────────
# Analysis functions
# ──────────────────────────────────────────────────────────────

def print_comparison_table(results: dict[str, dict]):
    """Print side-by-side comparison of the three systems."""
    labels = list(results.keys())
    fields = [
        ("Trades",       "total_trades",        "d"),
        ("Win Rate",     "win_rate_pct",         ".1f", "%"),
        ("Profit Factor","profit_factor",        ".2f"),
        ("Sharpe",       "sharpe_ratio",         ".2f"),
        ("Max DD",       "max_drawdown_dollars", ",.2f", "$"),
        ("Return",       "total_return_pct",     ".1f", "%"),
        ("Net PnL",      "total_pnl_dollars",    ",.2f", "$"),
        ("Expectancy",   "expectancy",           ",.2f", "$"),
    ]

    col_w = 18
    header = f"{'Metric':<18}" + "".join(f"{l:>{col_w}}" for l in labels)
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("  SYSTEM COMPARISON")
    print(sep)
    print(header)
    print("-" * len(header))

    for entry in fields:
        name, key, fmt = entry[0], entry[1], entry[2]
        prefix = entry[3] if len(entry) > 3 else ""
        row = f"{name:<18}"
        for label in labels:
            val = results[label].get(key, 0)
            if fmt == "d":
                val = int(round(val))
            if prefix == "$":
                cell = f"${val:{fmt}}"
            elif prefix == "%":
                cell = f"{val:{fmt}}%"
            else:
                cell = f"{val:{fmt}}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(sep)


def percentile_bucket_analysis(
    result: BacktestResult,
    decisions: list[dict],
) -> list[dict]:
    """
    Group trades by ML percentile bucket and compute stats.

    Matches accepted decisions to trades by timestamp order.
    """
    accepted = [d for d in decisions if d["accepted"]]
    trades = result.trades

    if len(accepted) != len(trades):
        # Fall back: if counts don't match, pair by index
        n = min(len(accepted), len(trades))
        accepted = accepted[:n]
        trades = trades[:n]

    if not trades:
        return []

    # Build paired data
    paired = []
    for dec, trade in zip(accepted, trades):
        paired.append({
            "percentile": dec["percentile"],
            "ml_prob": dec["ml_prob"],
            "net_pnl": trade.net_pnl,
            "win": 1 if trade.net_pnl > 0 else 0,
            "strategy_type": getattr(trade, "strategy_type", ""),
        })

    # Define buckets
    buckets = [
        ("0-20%",   0.0, 0.2),
        ("20-40%",  0.2, 0.4),
        ("40-60%",  0.4, 0.6),
        ("60-80%",  0.6, 0.8),
        ("80-100%", 0.8, 1.01),
    ]

    rows = []
    for label, lo, hi in buckets:
        subset = [p for p in paired if lo <= p["percentile"] < hi]
        n = len(subset)
        if n == 0:
            rows.append({"bucket": label, "trades": 0, "win_rate": 0, "avg_pnl": 0})
            continue
        wins = sum(p["win"] for p in subset)
        avg_pnl = np.mean([p["net_pnl"] for p in subset])
        rows.append({
            "bucket": label,
            "trades": n,
            "win_rate": wins / n * 100,
            "avg_pnl": avg_pnl,
        })

    return rows, paired


def print_percentile_table(rows: list[dict]):
    """Print percentile bucket table."""
    print(f"\n{'=' * 60}")
    print("  PERCENTILE BUCKET ANALYSIS (ML-Ranked System)")
    print("=" * 60)
    print(f"  {'Bucket':<12} {'Trades':>8} {'Win Rate':>10} {'Avg PnL':>12}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*12}")
    for r in rows:
        wr = f"{r['win_rate']:.1f}%"
        pnl = f"${r['avg_pnl']:,.2f}"
        print(f"  {r['bucket']:<12} {r['trades']:>8} {wr:>10} {pnl:>12}")

    # Check monotonicity
    pnls = [r["avg_pnl"] for r in rows if r["trades"] > 0]
    if len(pnls) >= 3:
        increasing = all(pnls[i] <= pnls[i+1] for i in range(len(pnls)-1))
        if increasing:
            print(f"\n  --> Monotonic improvement: YES (higher percentile = higher PnL)")
        else:
            # Check general trend
            if pnls[-1] > pnls[0]:
                print(f"\n  --> General upward trend from bottom to top bucket")
            else:
                print(f"\n  --> No clear monotonic improvement detected")
    print("=" * 60)


def correlation_analysis(paired: list[dict]):
    """Compute and print correlation(percentile, trade PnL)."""
    if len(paired) < 5:
        print("\n  Insufficient data for correlation analysis.")
        return

    pctiles = np.array([p["percentile"] for p in paired])
    pnls = np.array([p["net_pnl"] for p in paired])
    probs = np.array([p["ml_prob"] for p in paired])

    corr_pctile_pnl = float(np.corrcoef(pctiles, pnls)[0, 1])
    corr_prob_pnl = float(np.corrcoef(probs, pnls)[0, 1])

    print(f"\n{'=' * 60}")
    print("  CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"  corr(percentile, PnL) : {corr_pctile_pnl:+.4f}")
    print(f"  corr(ML prob, PnL)    : {corr_prob_pnl:+.4f}")

    if corr_pctile_pnl > 0.05:
        print(f"\n  --> Positive correlation: ML ranking DOES predict better trades")
    elif corr_pctile_pnl < -0.05:
        print(f"\n  --> Negative correlation: ML ranking is COUNTERPRODUCTIVE")
    else:
        print(f"\n  --> Near-zero correlation: ML ranking has MINIMAL predictive value")
    print("=" * 60)


def strategy_type_breakdown(result: BacktestResult, label: str = ""):
    """Print per-strategy_type statistics with size-weighted returns."""
    by_type: dict[str, list[Trade]] = defaultdict(list)
    for t in result.trades:
        st = t.strategy_type or "unknown"
        by_type[st].append(t)

    if not by_type:
        return

    title = f"STRATEGY-TYPE BREAKDOWN ({label})" if label else "STRATEGY-TYPE BREAKDOWN"
    print(f"\n{'=' * 82}")
    print(f"  {title}")
    print("=" * 82)
    print(f"  {'Type':<22} {'Trades':>7} {'Win Rate':>10} {'Avg PnL':>10} "
          f"{'Total PnL':>12} {'Avg Size':>10} {'SzWt PnL':>10}")
    print(f"  {'-'*22} {'-'*7} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

    sorted_types = sorted(by_type.items(), key=lambda x: sum(t.net_pnl for t in x[1]), reverse=True)
    for st, trades in sorted_types:
        n = len(trades)
        wins = sum(1 for t in trades if t.net_pnl > 0)
        wr = wins / n * 100 if n > 0 else 0
        avg_pnl = np.mean([t.net_pnl for t in trades])
        total_pnl = sum(t.net_pnl for t in trades)
        sizes = [getattr(t, "position_size", 1.0) or 1.0 for t in trades]
        avg_size = np.mean(sizes)
        # Size-weighted PnL: PnL * size, normalized by avg size
        sw_pnls = [t.net_pnl * s for t, s in zip(trades, sizes)]
        sw_avg = np.mean(sw_pnls) if sw_pnls else 0
        print(f"  {st:<22} {n:>7} {wr:>9.1f}% ${avg_pnl:>8,.2f} "
              f"${total_pnl:>10,.2f} {avg_size:>9.2f} ${sw_avg:>8,.2f}")

    print("=" * 82)


def ml_contribution_metric(metrics_dict: dict[str, dict]):
    """Compute and print the ML contribution across all variants."""
    labels = list(metrics_dict.keys())

    print(f"\n{'=' * 70}")
    print("  ML CONTRIBUTION METRIC")
    print("=" * 70)

    for label in labels:
        ret = metrics_dict[label]["total_return_pct"]
        sharpe = metrics_dict[label]["sharpe_ratio"]
        print(f"  {label:<28} : {ret:+.2f}% return, Sharpe {sharpe:.2f}")

    # Pairwise comparisons vs No-ML (baseline)
    noml_key = [k for k in labels if "No-ML" in k]
    if noml_key:
        noml = noml_key[0]
        noml_ret = metrics_dict[noml]["total_return_pct"]
        noml_pnl = metrics_dict[noml]["total_pnl_dollars"]
        print(f"\n  {'Variant':<28} {'vs No-ML':>12} {'PnL Lift':>14}")
        print(f"  {'-'*28} {'-'*12} {'-'*14}")
        for label in labels:
            if label == noml:
                continue
            ret = metrics_dict[label]["total_return_pct"]
            pnl = metrics_dict[label]["total_pnl_dollars"]
            print(f"  {label:<28} {ret - noml_ret:>+11.2f}% ${pnl - noml_pnl:>+12,.2f}")

    print("=" * 70)

    # Return improvement of ML-Ranked vs No-ML and Random for backward compat
    ml_key = [k for k in labels if "ML-Ranked" in k or "ML Global" in k]
    rand_key = [k for k in labels if "Random" in k]
    if noml_key and ml_key:
        improvement_noml = (metrics_dict[ml_key[0]]["total_return_pct"]
                            - metrics_dict[noml_key[0]]["total_return_pct"])
    else:
        improvement_noml = 0.0
    if ml_key and rand_key:
        improvement_random = (metrics_dict[ml_key[0]]["total_return_pct"]
                              - metrics_dict[rand_key[0]]["total_return_pct"])
    else:
        improvement_random = 0.0

    return improvement_noml, improvement_random


def print_conclusion(
    metrics_dict: dict[str, dict],
    improvement_vs_noml: float,
    improvement_vs_random: float,
    corr_pctile_pnl: float,
    type_breakdown: dict[str, list[Trade]],
):
    """Print final conclusion."""
    labels = list(metrics_dict.keys())

    print(f"\n{'=' * 70}")
    print("  CONCLUSION")
    print("=" * 70)

    # Find best variant by Sharpe
    best_label = max(labels, key=lambda k: metrics_dict[k]["sharpe_ratio"])
    best_sharpe = metrics_dict[best_label]["sharpe_ratio"]
    best_ret = metrics_dict[best_label]["total_return_pct"]
    print(f"  [*] BEST SYSTEM: {best_label}")
    print(f"      Sharpe={best_sharpe:.2f}, Return={best_ret:+.1f}%")

    # ML ranking assessment
    if improvement_vs_noml > 0.5:
        print(f"\n  [+] ML global ranking ADDS VALUE vs no-ML: {improvement_vs_noml:+.1f}%")
    elif improvement_vs_noml > -0.5:
        print(f"\n  [~] ML global ranking is NEUTRAL vs no-ML: {improvement_vs_noml:+.1f}%")
    else:
        print(f"\n  [-] ML global ranking HURTS vs no-ML: {improvement_vs_noml:+.1f}%")

    # Priority hybrid assessment
    priority_key = [k for k in labels if "Priority" in k and "Sizing" not in k]
    noml_key = [k for k in labels if "No-ML" in k]
    if priority_key and noml_key:
        p_ret = metrics_dict[priority_key[0]]["total_return_pct"]
        n_ret = metrics_dict[noml_key[0]]["total_return_pct"]
        delta = p_ret - n_ret
        if delta > 0.5:
            print(f"  [+] Priority hybrid BEATS no-ML by {delta:+.1f}%")
        elif delta > -0.5:
            print(f"  [~] Priority hybrid is NEUTRAL vs no-ML: {delta:+.1f}%")
        else:
            print(f"  [-] Priority hybrid TRAILS no-ML by {delta:+.1f}%")

    # Ranking correlation
    if corr_pctile_pnl > 0.05:
        print(f"  [+] Ranking IS meaningful: corr(percentile, PnL) = {corr_pctile_pnl:+.4f}")
    elif corr_pctile_pnl > -0.02:
        print(f"  [~] Ranking shows WEAK signal: corr = {corr_pctile_pnl:+.4f}")
    else:
        print(f"  [-] Ranking NOT meaningful: corr = {corr_pctile_pnl:+.4f}")

    # Dominant strategy type
    if type_breakdown:
        total_pnl = sum(sum(t.net_pnl for t in trades) for trades in type_breakdown.values())
        for st, trades in type_breakdown.items():
            st_pnl = sum(t.net_pnl for t in trades)
            if total_pnl != 0 and abs(st_pnl / total_pnl) > 0.5:
                share = st_pnl / total_pnl * 100
                print(f"  [!] System DOMINATED by '{st}' ({share:.0f}% of total PnL)")
                break
        else:
            print(f"  [+] PnL is distributed across strategy types (no single dominant)")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ML Value Analysis")
    p.add_argument("--data", required=True, help="Path to bar data CSV")
    p.add_argument("--split", default=None,
                   choices=["train", "validation", "test", "holdout"],
                   help="Data split to analyze (reads dates from model pkl)")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--ml-model", default="models/ema_model.pkl", help="Model path")
    p.add_argument("--ema-periods", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--entry-types", nargs="+", default=["breakout", "pullback", "momentum"])
    p.add_argument("--max-trades", type=int, default=3, help="Max trades per day")
    p.add_argument("--ml-threshold", type=float, default=0.0,
                   help="ML threshold for selection (0 = accept all above random)")
    p.add_argument("--random-seeds", type=int, default=5,
                   help="Number of random seeds to average (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Load data ---
    tz = "America/New_York"
    print(f"Loading data from {args.data} ...")
    bars = load_bars(args.data, timezone=tz, start=args.start, end=args.end)

    # Filter by split
    if args.split is not None:
        import pickle
        try:
            with open(args.ml_model, "rb") as f:
                model_data = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: Model file {args.ml_model} not found.")
            sys.exit(1)
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as exc:
            print(f"ERROR: Failed to load model from {args.ml_model}: {exc}")
            sys.exit(1)

        sd = model_data.get("split_dates")
        if sd is None:
            print(f"ERROR: Model has no split_dates. Re-train with fixed pipeline.")
            sys.exit(1)
        if args.split not in sd:
            print(f"ERROR: Split '{args.split}' not in model. Available: {list(sd.keys())}")
            sys.exit(1)
        split_start = pd.Timestamp(sd[args.split]["start"]).date()
        split_end = pd.Timestamp(sd[args.split]["end"]).date()
        bar_dates = bars.index.date
        bars = bars[(bar_dates >= split_start) & (bar_dates <= split_end)]
        print(f"Split '{args.split}': {split_start} to {split_end}")

    print(f"Bars: {len(bars)} ({bars.index[0].date()} to {bars.index[-1].date()})")

    # --- Base config ---
    base_cfg = HybridEMAMLConfig(
        multi_candidate=True,
        max_trades_per_day=args.max_trades,
        ema_periods=tuple(args.ema_periods),
        entry_types=tuple(args.entry_types),
        allow_shorts=True,
        model_path=args.ml_model,
        ml_selection_mode="threshold",
        ml_threshold=args.ml_threshold,
        position_sizing_mode="none",
        base_size=1.0,
    )

    print(f"\nConfig: EMA periods={base_cfg.ema_periods}, "
          f"entry types={base_cfg.entry_types}, "
          f"max trades/day={base_cfg.max_trades_per_day}")
    print(f"ML threshold={base_cfg.ml_threshold:.2f}")

    # ── Run A: No-ML ──
    print(f"\n--- Running A: No-ML (first-N selection) ---")
    strat_a = NoMLStrategy(base_cfg)
    result_a, metrics_a, decisions_a = run_variant(bars, strat_a, "No-ML")
    print(f"  Trades: {metrics_a['total_trades']}, Return: {metrics_a['total_return_pct']:.1f}%")

    # ── Run B: ML-Ranked (global) ──
    print(f"--- Running B: ML-Ranked (global) ---")
    strat_b = HybridEMAMLStrategy(base_cfg)
    result_b, metrics_b, decisions_b = run_variant(bars, strat_b, "ML Global")
    print(f"  Trades: {metrics_b['total_trades']}, Return: {metrics_b['total_return_pct']:.1f}%")

    # ── Run C: Priority Hybrid ──
    print(f"--- Running C: Priority Hybrid (breakout > momentum > pullback, ML within) ---")
    strat_c = HybridPriorityStrategy(base_cfg)
    result_c, metrics_c, decisions_c = run_variant(bars, strat_c, "Priority Hybrid")
    print(f"  Trades: {metrics_c['total_trades']}, Return: {metrics_c['total_return_pct']:.1f}%")

    # ── Run D: Priority ML-Sizing ──
    print(f"--- Running D: Priority ML-Sizing (priority order, ML for sizing only) ---")
    strat_d = PriorityMLSizingStrategy(base_cfg)
    result_d, metrics_d, decisions_d = run_variant(bars, strat_d, "Priority ML-Sizing")
    print(f"  Trades: {metrics_d['total_trades']}, Return: {metrics_d['total_return_pct']:.1f}%")

    # ── Run E: Random (averaged over multiple seeds) ──
    print(f"--- Running E: Random ({args.random_seeds} seeds) ---")
    random_metrics_list = []
    random_results_list = []
    for seed in range(args.random_seeds):
        strat_e = RandomStrategy(base_cfg, seed=seed)
        result_e, metrics_e, _ = run_variant(bars, strat_e, f"Random-{seed}")
        random_metrics_list.append(metrics_e)
        random_results_list.append(result_e)

    # Average random metrics
    avg_random_metrics = {}
    for key in metrics_a.keys():
        vals = [m.get(key, 0) for m in random_metrics_list]
        if isinstance(vals[0], (int, float)):
            avg_random_metrics[key] = round(float(np.mean(vals)), 4)
        else:
            avg_random_metrics[key] = vals[0]  # non-numeric: take first
    print(f"  Avg Trades: {avg_random_metrics['total_trades']:.0f}, "
          f"Avg Return: {avg_random_metrics['total_return_pct']:.1f}%")

    # ── Comparison table (all 5 variants) ──
    all_metrics = {
        "A: No-ML": metrics_a,
        "B: ML Global": metrics_b,
        "C: Priority Hybrid": metrics_c,
        "D: Priority Sizing": metrics_d,
        "E: Random": avg_random_metrics,
    }
    print_comparison_table(all_metrics)

    # ── Percentile bucket analysis (on ML system) ──
    bucket_rows, paired = percentile_bucket_analysis(result_b, decisions_b)
    print_percentile_table(bucket_rows)

    # ── Correlation analysis ──
    if paired:
        pctiles = np.array([p["percentile"] for p in paired])
        pnls_arr = np.array([p["net_pnl"] for p in paired])
        if len(paired) >= 5:
            corr_val = float(np.corrcoef(pctiles, pnls_arr)[0, 1])
        else:
            corr_val = 0.0
        correlation_analysis(paired)
    else:
        corr_val = 0.0

    # ── Strategy-type breakdown for each variant ──
    variant_results = {
        "A: No-ML": result_a,
        "B: ML Global": result_b,
        "C: Priority Hybrid": result_c,
        "D: Priority Sizing": result_d,
    }
    for label, res in variant_results.items():
        strategy_type_breakdown(res, label=label)

    # Build type breakdown dict for conclusion (use priority hybrid)
    type_dict: dict[str, list[Trade]] = defaultdict(list)
    for t in result_c.trades:
        type_dict[t.strategy_type or "unknown"].append(t)

    # ── ML contribution metric ──
    improvement_noml, improvement_random = ml_contribution_metric(all_metrics)

    # ── Conclusion ──
    print_conclusion(all_metrics, improvement_noml, improvement_random, corr_val, type_dict)

    # ── Export results ──
    import json
    from pathlib import Path

    out = Path("results")
    out.mkdir(exist_ok=True)

    export = {
        "comparison": {k: {kk: vv for kk, vv in v.items() if kk != "pnl_by_weekday"}
                       for k, v in all_metrics.items()},
        "percentile_buckets": bucket_rows,
        "correlation_percentile_pnl": corr_val,
        "ml_improvement_vs_noml_pct": improvement_noml,
        "ml_improvement_vs_random_pct": improvement_random,
    }
    with open(out / "ml_value_analysis.json", "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"\nResults saved to {out / 'ml_value_analysis.json'}")


if __name__ == "__main__":
    main()
