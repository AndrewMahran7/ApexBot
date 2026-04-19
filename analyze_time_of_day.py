"""
TASK 1: Measure AdaptiveRegimeStrategy trade performance by time of day.

Groups trades into meaningful session windows and computes per-bucket
metrics across key years for MNQ and MES.
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
from data.loader import load_bars
from config.settings import AdaptiveRegimeConfig, BacktestConfig, INSTRUMENT_REGISTRY
from strategy.adaptive_regime import AdaptiveRegimeStrategy
from backtest.engine import BacktestEngine, Trade
from backtest.metrics import compute_metrics

OUTPUT_DIR = Path("results/adaptive_regime_time_filter_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=25_000.0,
    slippage_ticks=1,
    commission_per_side=2.25,
)

# Time buckets: name -> (start_hour_minute, end_hour_minute) inclusive
# Range ends at 09:45, first possible entry is 09:45 bar
TIME_BUCKETS = [
    ("09:45-10:15", (9, 45), (10, 15)),   # opening breakout — immediate
    ("10:15-10:45", (10, 15), (10, 45)),   # second wave
    ("10:45-11:15", (10, 45), (11, 15)),   # late morning
    ("11:15-12:00", (11, 15), (12, 0)),    # approaching midday
]
# Current max_entry_time is 12:00, so nothing after that

# Coarser buckets also useful
COARSE_BUCKETS = [
    ("09:45-10:00", (9, 45), (10, 0)),    # first 15 min after range
    ("10:00-10:30", (10, 0), (10, 30)),   # mid-morning
    ("10:30-11:00", (10, 30), (11, 0)),   # late morning
    ("11:00-11:30", (11, 0), (11, 30)),   # pre-midday
    ("11:30-12:00", (11, 30), (12, 0)),   # midday
]


def _time_to_minutes(t):
    return t.hour * 60 + t.minute


def _bucket_trade(trade, buckets):
    """Assign a trade to a time bucket based on entry_time."""
    entry_min = _time_to_minutes(trade.entry_time.time())
    for name, (sh, sm), (eh, em) in buckets:
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        if start_min <= entry_min < end_min:
            return name
    return "other"


def load_year_data(symbol, year):
    prefix = symbol.lower()
    files = {
        2017: f"data/{prefix}_2017.csv",
        2018: f"data/{prefix}_2018.csv",
        2019: f"data/{prefix}_2019.csv",
        2022: f"data/{prefix}_2022.csv",
        2025: f"data/{prefix}_2025.csv",
    }
    if year in files:
        return load_bars(files[year])
    bars = load_bars(f"data/{prefix}_4y.csv")
    s = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
    e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="America/New_York")
    return bars[(bars.index >= s) & (bars.index <= e)]


def run_backtest(symbol, year):
    """Run backtest, return (trades, metrics, selectivity)."""
    bars = load_year_data(symbol, year)
    instrument = INSTRUMENT_REGISTRY[symbol]
    cfg = AdaptiveRegimeConfig.for_symbol(symbol)
    strat = AdaptiveRegimeStrategy(cfg)
    engine = BacktestEngine(
        instrument=instrument,
        strategy_config=cfg,
        backtest_config=BACKTEST_CONFIG,
        strategy=strat,
    )
    result = engine.run(bars)
    metrics = compute_metrics(result, BACKTEST_CONFIG)
    return result.trades, metrics


def analyze_buckets(trades, buckets):
    """Group trades by time bucket and compute per-bucket metrics."""
    grouped = defaultdict(list)
    for t in trades:
        bucket = _bucket_trade(t, buckets)
        grouped[bucket].append(t)

    results = []
    for name, _, _ in buckets:
        bucket_trades = grouped.get(name, [])
        n = len(bucket_trades)
        if n == 0:
            results.append({
                "bucket": name, "trades": 0, "win_rate": 0,
                "avg_pnl": 0, "total_pnl": 0, "profit_factor": 0,
                "longs": 0, "shorts": 0,
            })
            continue

        pnls = [t.net_pnl for t in bucket_trades]
        wins = sum(1 for p in pnls if p > 0)
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        longs = sum(1 for t in bucket_trades if t.direction == "long")
        shorts = n - longs

        results.append({
            "bucket": name,
            "trades": n,
            "win_rate": round(wins / n * 100, 1),
            "avg_pnl": round(sum(pnls) / n, 2),
            "total_pnl": round(sum(pnls), 2),
            "profit_factor": round(pf, 2),
            "longs": longs,
            "shorts": shorts,
        })

    # "other" bucket
    other_trades = grouped.get("other", [])
    if other_trades:
        pnls = [t.net_pnl for t in other_trades]
        wins = sum(1 for p in pnls if p > 0)
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p <= 0))
        results.append({
            "bucket": "other",
            "trades": len(other_trades),
            "win_rate": round(wins / len(other_trades) * 100, 1),
            "avg_pnl": round(sum(pnls) / len(other_trades), 2),
            "total_pnl": round(sum(pnls), 2),
            "profit_factor": round(gp / gl if gl > 0 else float("inf"), 2),
            "longs": sum(1 for t in other_trades if t.direction == "long"),
            "shorts": len(other_trades) - sum(1 for t in other_trades if t.direction == "long"),
        })

    return results


def print_bucket_table(rows, title=""):
    if title:
        print(f"\n{title}")
    print(f"  {'Bucket':>14} {'Trades':>7} {'WR%':>6} {'AvgPnL':>9} {'TotalPnL':>10} {'PF':>6} {'L':>4} {'S':>4}")
    print(f"  {'-' * 14} {'-' * 7} {'-' * 6} {'-' * 9} {'-' * 10} {'-' * 6} {'-' * 4} {'-' * 4}")
    for r in rows:
        print(f"  {r['bucket']:>14} {r['trades']:>7} {r['win_rate']:>6.1f} "
              f"{r['avg_pnl']:>9.2f} {r['total_pnl']:>10.2f} {r['profit_factor']:>6.2f} "
              f"{r['longs']:>4} {r['shorts']:>4}")


if __name__ == "__main__":
    all_analysis = {}

    years = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
    symbols = ["MNQ", "MES"]

    # ── COARSE BUCKET ANALYSIS ──
    print("=" * 90)
    print("  TIME-OF-DAY TRADE ANALYSIS (Coarse 30-min buckets)")
    print("=" * 90)

    for symbol in symbols:
        for year in years:
            trades, metrics = run_backtest(symbol, year)
            rows = analyze_buckets(trades, COARSE_BUCKETS)
            title = f"  {symbol} {year} — {metrics['total_trades']} trades, ${metrics['total_pnl_dollars']:.2f} total"
            print_bucket_table(rows, title)
            all_analysis[f"{symbol}_{year}"] = {
                "total_trades": metrics["total_trades"],
                "total_pnl": metrics["total_pnl_dollars"],
                "buckets": rows,
            }

    # ── CROSS-YEAR AGGREGATION ──
    print("\n" + "=" * 90)
    print("  CROSS-YEAR AGGREGATION BY BUCKET")
    print("=" * 90)

    for symbol in symbols:
        # Aggregate across all years
        agg = defaultdict(lambda: {"trades": 0, "total_pnl": 0.0, "wins": 0, "gp": 0.0, "gl": 0.0})

        for year in years:
            key = f"{symbol}_{year}"
            for b in all_analysis[key]["buckets"]:
                name = b["bucket"]
                agg[name]["trades"] += b["trades"]
                agg[name]["total_pnl"] += b["total_pnl"]
                # Recompute wins from stored data
                # We need to re-run for precise aggregation, but approximate:
                agg[name]["wins"] += round(b["win_rate"] / 100 * b["trades"])

        print(f"\n  {symbol} AGGREGATE (all 8 years)")
        print(f"  {'Bucket':>14} {'Trades':>7} {'WR%':>6} {'AvgPnL':>9} {'TotalPnL':>10}")
        print(f"  {'-' * 14} {'-' * 7} {'-' * 6} {'-' * 9} {'-' * 10}")
        for name, _, _ in COARSE_BUCKETS:
            a = agg[name]
            n = a["trades"]
            if n > 0:
                wr = a["wins"] / n * 100
                avg = a["total_pnl"] / n
                print(f"  {name:>14} {n:>7} {wr:>6.1f} {avg:>9.2f} {a['total_pnl']:>10.2f}")

    # ── SAVE ──
    with open(OUTPUT_DIR / "time_bucket_analysis.json", "w") as f:
        json.dump(all_analysis, f, indent=2)
    print(f"\nAnalysis saved to {OUTPUT_DIR / 'time_bucket_analysis.json'}")

    # ── Per-hour breakdown for MNQ (even finer) ──
    print("\n" + "=" * 90)
    print("  MNQ ENTRY HOUR DISTRIBUTION (all years)")
    print("=" * 90)

    hour_agg = defaultdict(lambda: {"trades": 0, "total_pnl": 0.0, "wins": 0})
    for year in years:
        trades, _ = run_backtest("MNQ", year)
        for t in trades:
            h = t.entry_time.hour
            m = t.entry_time.minute
            # Group into 15-min slots
            slot = f"{h:02d}:{(m // 15) * 15:02d}"
            hour_agg[slot]["trades"] += 1
            hour_agg[slot]["total_pnl"] += t.net_pnl
            if t.net_pnl > 0:
                hour_agg[slot]["wins"] += 1

    print(f"  {'Slot':>8} {'Trades':>7} {'WR%':>6} {'AvgPnL':>9} {'TotalPnL':>10}")
    print(f"  {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 9} {'-' * 10}")
    for slot in sorted(hour_agg.keys()):
        a = hour_agg[slot]
        n = a["trades"]
        if n > 0:
            wr = a["wins"] / n * 100
            avg = a["total_pnl"] / n
            print(f"  {slot:>8} {n:>7} {wr:>6.1f} {avg:>9.2f} {a['total_pnl']:>10.2f}")
