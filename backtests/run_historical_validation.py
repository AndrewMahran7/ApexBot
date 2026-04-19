import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
#!/usr/bin/env python3
"""
Historical Validation Suite — 2017, 2018, 2019
================================================

Runs the corrected (causal entry) hybrid_ema_ml strategy on MES + MNQ
for each year, saves per-year results, and produces a comparison table.

This script uses BacktestEngine directly (no Telegram, no dashboard,
no multi-symbol routing overhead) for fast execution.

Usage:
    python run_historical_validation.py
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path

from config.settings import (
    InstrumentConfig,
    StrategyConfig,
    BacktestConfig,
    INSTRUMENT_REGISTRY,
)
from data.loader import load_bars
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, export_trades_csv, export_metrics_json
from strategy.hybrid_ema_ml import HybridEMAMLStrategy, HybridEMAMLConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

OUT_DIR = Path("results/historical_validation_2017_2019")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2017, 2018, 2019]
SYMBOLS = ["MES", "MNQ"]

ML_THRESHOLD = 0.55
ML_MODEL = "models/ema_model.pkl"
INITIAL_CAPITAL = 5000.0
SLIPPAGE_TICKS = 1
COMMISSION = 2.32


def build_strategy(symbol: str) -> tuple:
    """Build hybrid_ema_ml strategy + configs for a given symbol."""
    instrument = INSTRUMENT_REGISTRY[symbol]
    cfg = HybridEMAMLConfig(
        allow_shorts=True,
        ml_threshold=ML_THRESHOLD,
        model_path=ML_MODEL,
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
    return strategy, instrument, strat_cfg, bt_cfg


def run_single(year: int, symbol: str) -> dict:
    """Run backtest for one symbol-year. Returns summary dict."""
    data_path = f"data/{symbol.lower()}_{year}.csv"
    print(f"  [{symbol} {year}] Loading {data_path} ...")
    
    strategy, instrument, strat_cfg, bt_cfg = build_strategy(symbol)
    bars = load_bars(data_path, timezone=strat_cfg.timezone)
    print(f"  [{symbol} {year}] {len(bars)} bars, {bars.index[0].date()} to {bars.index[-1].date()}")

    t0 = time.time()
    engine = BacktestEngine(instrument, strat_cfg, bt_cfg, strategy=strategy)
    result = engine.run(bars)
    elapsed = time.time() - t0
    print(f"  [{symbol} {year}] Done in {elapsed:.1f}s — {len(result.trades)} trades")

    metrics = compute_metrics(result, bt_cfg)

    # Save trades CSV
    trades_path = OUT_DIR / f"{year}_{symbol.lower()}_trades.csv"
    export_trades_csv(result.trades, str(trades_path))

    # Save metrics JSON
    metrics_path = OUT_DIR / f"{year}_{symbol.lower()}_metrics.json"
    export_metrics_json(metrics, str(metrics_path))

    # Compute per-strategy breakdown
    by_strategy = {}
    for t in result.trades:
        st = getattr(t, "strategy_type", "unknown") or "unknown"
        if st not in by_strategy:
            by_strategy[st] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_strategy[st]["trades"] += 1
        if t.net_pnl > 0:
            by_strategy[st]["wins"] += 1
        by_strategy[st]["pnl"] += t.net_pnl

    # Compute direction breakdown
    longs = [t for t in result.trades if t.direction == "long"]
    shorts = [t for t in result.trades if t.direction == "short"]
    long_pnl = sum(t.net_pnl for t in longs)
    short_pnl = sum(t.net_pnl for t in shorts)
    long_wins = sum(1 for t in longs if t.net_pnl > 0)
    short_wins = sum(1 for t in shorts if t.net_pnl > 0)

    # Equity curve stats
    equity = [ep.equity for ep in result.equity_curve] if result.equity_curve else [INITIAL_CAPITAL]
    peak = max(equity)
    drawdowns = []
    running_peak = equity[0]
    for e in equity:
        running_peak = max(running_peak, e)
        drawdowns.append(e - running_peak)
    max_dd = min(drawdowns) if drawdowns else 0.0

    # Profit factor
    gross_profit = sum(t.net_pnl for t in result.trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in result.trades if t.net_pnl <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = sum(t.net_pnl for t in result.trades)
    n = len(result.trades)
    wins = sum(1 for t in result.trades if t.net_pnl > 0)
    
    summary = {
        "year": year,
        "symbol": symbol,
        "total_trades": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": (wins / n * 100) if n > 0 else 0.0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / n, 2) if n > 0 else 0.0,
        "profit_factor": round(profit_factor, 2),
        "peak_equity": round(peak, 2),
        "max_drawdown": round(max_dd, 2),
        "final_equity": round(INITIAL_CAPITAL + total_pnl, 2),
        "long_trades": len(longs),
        "long_wins": long_wins,
        "long_wr": round(long_wins / len(longs) * 100, 1) if longs else 0.0,
        "long_pnl": round(long_pnl, 2),
        "short_trades": len(shorts),
        "short_wins": short_wins,
        "short_wr": round(short_wins / len(shorts) * 100, 1) if shorts else 0.0,
        "short_pnl": round(short_pnl, 2),
        "by_strategy": by_strategy,
        "elapsed_sec": round(elapsed, 1),
    }
    return summary


def print_year_summary(year: int, results: list[dict]):
    """Print a per-year summary table."""
    print(f"\n{'='*70}")
    print(f"  {year} VALIDATION RESULTS")
    print(f"{'='*70}")
    
    combined_trades = 0
    combined_wins = 0
    combined_pnl = 0.0
    
    for r in results:
        sym = r["symbol"]
        print(f"\n  [{sym}]")
        print(f"    Trades:      {r['total_trades']} ({r['wins']}W / {r['losses']}L)")
        print(f"    Win rate:    {r['win_rate']:.1f}%")
        print(f"    Total PnL:   ${r['total_pnl']:.2f}")
        print(f"    Avg PnL:     ${r['avg_pnl']:.2f}")
        print(f"    PF:          {r['profit_factor']:.2f}")
        print(f"    Max DD:      ${r['max_drawdown']:.2f}")
        print(f"    Longs:       {r['long_trades']} ({r['long_wr']:.1f}% WR, ${r['long_pnl']:.2f})")
        print(f"    Shorts:      {r['short_trades']} ({r['short_wr']:.1f}% WR, ${r['short_pnl']:.2f})")
        
        for st, stats in sorted(r["by_strategy"].items()):
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
            print(f"    Strategy [{st}]: {stats['trades']} trades, {wr:.1f}% WR, ${stats['pnl']:.2f}")
        
        combined_trades += r["total_trades"]
        combined_wins += r["wins"]
        combined_pnl += r["total_pnl"]
    
    wr = combined_wins / combined_trades * 100 if combined_trades > 0 else 0
    print(f"\n  [COMBINED {year}]")
    print(f"    Total:       {combined_trades} trades, {wr:.1f}% WR, ${combined_pnl:.2f} PnL")


def print_comparison_table(all_results: dict[int, list[dict]]):
    """Print the final cross-year comparison table."""
    print(f"\n{'='*90}")
    print(f"  CROSS-YEAR COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"  {'Year':<6} {'Symbol':<6} {'Trades':>7} {'WR%':>7} {'PnL':>10} "
          f"{'Avg PnL':>9} {'PF':>6} {'Max DD':>10} {'Longs':>6} {'Shorts':>7}")
    print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*9} {'-'*6} {'-'*10} {'-'*6} {'-'*7}")
    
    for year in YEARS:
        for r in all_results[year]:
            print(f"  {r['year']:<6} {r['symbol']:<6} {r['total_trades']:>7} "
                  f"{r['win_rate']:>6.1f}% ${r['total_pnl']:>9.2f} "
                  f"${r['avg_pnl']:>8.2f} {r['profit_factor']:>6.2f} "
                  f"${r['max_drawdown']:>9.2f} {r['long_trades']:>6} {r['short_trades']:>7}")
        # Year totals
        trades = sum(r["total_trades"] for r in all_results[year])
        wins = sum(r["wins"] for r in all_results[year])
        pnl = sum(r["total_pnl"] for r in all_results[year])
        wr = wins / trades * 100 if trades > 0 else 0
        avg = pnl / trades if trades > 0 else 0
        print(f"  {year:<6} {'TOTAL':<6} {trades:>7} {wr:>6.1f}% ${pnl:>9.2f} "
              f"${avg:>8.2f} {'':>6} {'':>10} {'':>6} {'':>7}")
        print()
    
    # Grand totals
    all_trades = sum(r["total_trades"] for yrs in all_results.values() for r in yrs)
    all_wins = sum(r["wins"] for yrs in all_results.values() for r in yrs)
    all_pnl = sum(r["total_pnl"] for yrs in all_results.values() for r in yrs)
    wr = all_wins / all_trades * 100 if all_trades > 0 else 0
    avg = all_pnl / all_trades if all_trades > 0 else 0
    print(f"  {'GRAND':<6} {'TOTAL':<6} {all_trades:>7} {wr:>6.1f}% ${all_pnl:>9.2f} "
          f"${avg:>8.2f}")


def assess_robustness(all_results: dict[int, list[dict]]) -> str:
    """Produce a ROBUST / ACCEPTABLE / FRAGILE assessment."""
    year_pnls = {}
    year_wrs = {}
    for year in YEARS:
        trades = sum(r["total_trades"] for r in all_results[year])
        wins = sum(r["wins"] for r in all_results[year])
        pnl = sum(r["total_pnl"] for r in all_results[year])
        year_pnls[year] = pnl
        year_wrs[year] = wins / trades * 100 if trades > 0 else 0
    
    profitable_years = sum(1 for p in year_pnls.values() if p > 0)
    avg_wr = sum(year_wrs.values()) / len(year_wrs)
    total_pnl = sum(year_pnls.values())
    
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  ROBUSTNESS ASSESSMENT")
    lines.append(f"{'='*70}")
    lines.append(f"  Profitable years:  {profitable_years}/{len(YEARS)}")
    lines.append(f"  Avg win rate:      {avg_wr:.1f}%")
    lines.append(f"  Total PnL:         ${total_pnl:.2f}")
    lines.append(f"  Year PnLs:         " + ", ".join(
        f"{y}: ${p:.2f}" for y, p in year_pnls.items()))
    lines.append(f"  Year WRs:          " + ", ".join(
        f"{y}: {w:.1f}%" for y, w in year_wrs.items()))
    
    # Assessment criteria
    if profitable_years >= 2 and avg_wr >= 35 and total_pnl > 0:
        verdict = "ROBUST"
        reason = "Majority of years profitable with acceptable win rates"
    elif profitable_years >= 1 and avg_wr >= 30:
        verdict = "ACCEPTABLE"
        reason = "Mixed results but not catastrophic; system shows some edge"
    else:
        verdict = "FRAGILE"
        reason = "Insufficient profitability across validation years"
    
    lines.append(f"\n  VERDICT: {verdict}")
    lines.append(f"  Reason:  {reason}")
    lines.append(f"\n  CAVEATS:")
    lines.append(f"  - 2017-2019 is pre-training period (model trained 2021-2025)")
    lines.append(f"  - MES/MNQ micros didn't exist pre-2019; data mapped from ES/NQ")
    lines.append(f"  - Backtest uses BacktestEngine (no portfolio risk cross-symbol limits)")
    lines.append(f"  - ML model is applied out-of-domain (different market regime)")
    lines.append(f"{'='*70}")
    
    return "\n".join(lines), verdict


def main():
    print("=" * 70)
    print("  APEX HISTORICAL VALIDATION SUITE")
    print("  Years: 2017, 2018, 2019 | Symbols: MES, MNQ")
    print("  Strategy: hybrid_ema_ml (causal entry fix applied)")
    print("  ML threshold: %.2f | Shorts: enabled" % ML_THRESHOLD)
    print("=" * 70)
    
    all_results = {}
    t_total = time.time()
    
    for year in YEARS:
        print(f"\n--- {year} ---")
        year_results = []
        for symbol in SYMBOLS:
            summary = run_single(year, symbol)
            year_results.append(summary)
        all_results[year] = year_results
        print_year_summary(year, year_results)
    
    total_elapsed = time.time() - t_total
    
    # Comparison table
    print_comparison_table(all_results)
    
    # Robustness assessment
    assessment_text, verdict = assess_robustness(all_results)
    print(assessment_text)
    
    print(f"\n  Total runtime: {total_elapsed:.1f}s")
    
    # Save everything to JSON
    output = {
        "validation_type": "historical_oos",
        "years": YEARS,
        "symbols": SYMBOLS,
        "ml_threshold": ML_THRESHOLD,
        "ml_model": ML_MODEL,
        "initial_capital_per_symbol": INITIAL_CAPITAL,
        "slippage_ticks": SLIPPAGE_TICKS,
        "commission": COMMISSION,
        "verdict": verdict,
        "results": {},
    }
    for year in YEARS:
        output["results"][str(year)] = all_results[year]
    
    summary_path = OUT_DIR / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"  Summary saved to: {summary_path}")
    print(f"  Trade CSVs and metrics in: {OUT_DIR}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
