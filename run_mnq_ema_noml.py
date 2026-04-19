#!/usr/bin/env python3
"""
MNQ EMA Breakout — No ML Filter
=================================

Runs hybrid_ema_ml with ml_threshold=0.0 (accepts ALL EMA breakout
signals regardless of ML probability) on MNQ across multiple years.

Purpose: determine if the raw EMA breakout signal has a real edge
on MNQ without any ML filtering.

Usage:
    python run_mnq_ema_noml.py
"""

import json
import sys
import time
from pathlib import Path

from config.settings import (
    BacktestConfig,
    StrategyConfig,
    INSTRUMENT_REGISTRY,
)
from data.loader import load_bars
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, export_trades_csv, export_metrics_json
from strategy.hybrid_ema_ml import HybridEMAMLStrategy, HybridEMAMLConfig

OUT_DIR = Path("results/mnq_ema_noml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MNQ instrument: $2/point, 0.25 tick
INSTRUMENT = INSTRUMENT_REGISTRY["MNQ"]

INITIAL_CAPITAL = 5000.0
SLIPPAGE_TICKS = 1
COMMISSION = 2.32

# Year → (data_file, start_date, end_date)
YEAR_CONFIG = {
    2017: ("data/mnq_2017.csv", None, None),
    2018: ("data/mnq_2018.csv", None, None),
    2019: ("data/mnq_2019.csv", None, None),
    2024: ("data/mnq_4y.csv", "2024-01-01", "2024-12-31"),
    2025: ("data/mnq_2025.csv", None, None),
}


def run_year(year: int) -> dict:
    data_file, start, end = YEAR_CONFIG[year]
    print(f"  [{year}] Loading {data_file} ...")

    cfg = HybridEMAMLConfig(
        allow_shorts=True,
        ml_threshold=0.0,          # Accept ALL signals — no ML filtering
        model_path="models/ema_model.pkl",
    )
    strategy = HybridEMAMLStrategy(cfg)

    strat_cfg = StrategyConfig(
        shorts_enabled=True,
        ema_enabled=True,
        ema_length=50,
    )
    bt_cfg = BacktestConfig(
        slippage_ticks=SLIPPAGE_TICKS,
        commission_per_side=COMMISSION,
        initial_capital=INITIAL_CAPITAL,
    )

    bars = load_bars(data_file, timezone=strat_cfg.timezone, start=start, end=end)
    print(f"  [{year}] {len(bars)} bars, {bars.index[0].date()} to {bars.index[-1].date()}")

    t0 = time.time()
    engine = BacktestEngine(INSTRUMENT, strat_cfg, bt_cfg, strategy=strategy)
    result = engine.run(bars)
    elapsed = time.time() - t0
    print(f"  [{year}] Done in {elapsed:.1f}s — {len(result.trades)} trades")

    metrics = compute_metrics(result, bt_cfg)

    # Save per-year files
    export_trades_csv(result.trades, str(OUT_DIR / f"{year}_trades.csv"))
    export_metrics_json(metrics, str(OUT_DIR / f"{year}_metrics.json"))

    # Compute stats
    trades = result.trades
    n = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    total_pnl = sum(t.net_pnl for t in trades)
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = [ep.equity for ep in result.equity_curve] if result.equity_curve else [INITIAL_CAPITAL]
    running_peak = equity[0]
    max_dd = 0.0
    for e in equity:
        running_peak = max(running_peak, e)
        dd = e - running_peak
        if dd < max_dd:
            max_dd = dd

    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]

    return {
        "year": year,
        "trades": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": round(wins / n * 100, 1) if n else 0.0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / n, 2) if n else 0.0,
        "profit_factor": round(pf, 2),
        "max_drawdown": round(max_dd, 2),
        "final_equity": round(INITIAL_CAPITAL + total_pnl, 2),
        "long_trades": len(longs),
        "long_wins": sum(1 for t in longs if t.net_pnl > 0),
        "long_pnl": round(sum(t.net_pnl for t in longs), 2),
        "short_trades": len(shorts),
        "short_wins": sum(1 for t in shorts if t.net_pnl > 0),
        "short_pnl": round(sum(t.net_pnl for t in shorts), 2),
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    print("=" * 70)
    print("  MNQ EMA BREAKOUT — NO ML FILTER")
    print("  ml_threshold=0.0 (all EMA breakout signals accepted)")
    print("  Shorts: enabled | EMA: 50 | Causal entry: next-bar open")
    print("  Slippage: 1 tick | Commission: $2.32/side")
    print("=" * 70)

    results = []
    t_total = time.time()

    for year in sorted(YEAR_CONFIG.keys()):
        r = run_year(year)
        results.append(r)

    total_elapsed = time.time() - t_total

    # Print table
    print(f"\n{'='*80}")
    print(f"  MNQ EMA BREAKOUT (NO ML) — RESULTS")
    print(f"{'='*80}")
    print(f"  {'Year':<6} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'Avg PnL':>9} "
          f"{'PF':>6} {'Max DD':>10} {'L/S':>7}")
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*9} {'-'*6} {'-'*10} {'-'*7}")

    for r in results:
        ls = f"{r['long_trades']}/{r['short_trades']}"
        print(f"  {r['year']:<6} {r['trades']:>7} {r['win_rate']:>6.1f}% "
              f"${r['total_pnl']:>9.2f} ${r['avg_pnl']:>8.2f} "
              f"{r['profit_factor']:>6.2f} ${r['max_drawdown']:>9.2f} {ls:>7}")

    # Totals
    all_trades = sum(r["trades"] for r in results)
    all_wins = sum(r["wins"] for r in results)
    all_pnl = sum(r["total_pnl"] for r in results)
    all_wr = all_wins / all_trades * 100 if all_trades else 0
    all_avg = all_pnl / all_trades if all_trades else 0
    gp = sum(r["total_pnl"] for r in results if r["total_pnl"] > 0)
    gl = abs(sum(r["total_pnl"] for r in results if r["total_pnl"] <= 0))
    # overall PF from individual trades
    gp_t = sum(r["total_pnl"] for r in results)  # recompute from trades in detail below
    print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*9} {'-'*6} {'-'*10} {'-'*7}")
    print(f"  {'TOTAL':<6} {all_trades:>7} {all_wr:>6.1f}% "
          f"${all_pnl:>9.2f} ${all_avg:>8.2f}")

    # Direction breakdown
    print(f"\n  --- DIRECTION BREAKDOWN ---")
    print(f"  {'Year':<6} {'L Trades':>8} {'L WR%':>7} {'L PnL':>10} "
          f"{'S Trades':>8} {'S WR%':>7} {'S PnL':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*10} {'-'*8} {'-'*7} {'-'*10}")
    for r in results:
        lwr = round(r["long_wins"] / r["long_trades"] * 100, 1) if r["long_trades"] else 0
        swr = round(r["short_wins"] / r["short_trades"] * 100, 1) if r["short_trades"] else 0
        print(f"  {r['year']:<6} {r['long_trades']:>8} {lwr:>6.1f}% "
              f"${r['long_pnl']:>9.2f} {r['short_trades']:>8} "
              f"{swr:>6.1f}% ${r['short_pnl']:>9.2f}")

    # Profitable years
    prof_years = sum(1 for r in results if r["total_pnl"] > 0)
    print(f"\n  Profitable years: {prof_years}/{len(results)}")
    print(f"  Total runtime: {total_elapsed:.1f}s")

    # Save JSON
    output = {
        "test": "mnq_ema_breakout_no_ml",
        "ml_threshold": 0.0,
        "instrument": "MNQ",
        "initial_capital": INITIAL_CAPITAL,
        "slippage_ticks": SLIPPAGE_TICKS,
        "commission": COMMISSION,
        "results": results,
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Summary: {summary_path}")
    print(f"  Trades & metrics per year in: {OUT_DIR}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
