#!/usr/bin/env python3
"""
MES Parameter Optimizer
========================

Grid-search over strategy parameters and rank by chosen metric.

Supported strategies:
  - orb              : Opening Range Breakout
  - adaptive_regime  : Regime-Aware Breakout/Continuation

Usage:
    python optimize.py --data data/mes_5m.csv --strategy orb
    python optimize.py --data data/mes_5m.csv --strategy adaptive_regime
    python optimize.py --data data/mes_5m.csv --metric sharpe_ratio
    python optimize.py --data data/mes_5m.csv --metric profit_factor --top 20

========================== WARNING ===============================
Parameter optimization on historical data is prone to overfitting.
A parameter set that looks great in-sample may fail out-of-sample.

Best practices:
  - Split data into in-sample / out-of-sample periods
  - Prefer robust parameters (good across many combinations)
    over the single best combination
  - Be skeptical of Sharpe > 3 or win rates > 70%
  - Test on out-of-sample data before committing capital
  - Fewer parameters = less overfitting risk
==================================================================
"""

import argparse
import csv
import itertools
import logging
import time
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

from config.settings import InstrumentConfig, StrategyConfig, BacktestConfig, AdaptiveRegimeConfig
from data.loader import load_bars
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics
from strategy.orb import ORBStrategy
from strategy.adaptive_regime import AdaptiveRegimeStrategy


# Default parameter grids for ORB — modify these to explore different ranges
ORB_DEFAULT_GRIDS = {
    "or_minutes": [10, 15, 20, 30],         # opening range duration in minutes
    "rr": [1.0, 1.25, 1.5, 2.0],            # reward:risk ratio
    "ema_length": [20, 35, 50, 100],         # EMA period
    "ema_enabled": [True, False],            # EMA filter on/off
    "eod_exit": ["15:30", "15:45", "15:50"], # end-of-day exit time
    "min_range_points": [0],                 # minimum OR size in points
    "max_entry_time": [""],                  # latest entry time (empty = no limit)
    "shorts_enabled": [False],               # short entries on/off
}

# Default parameter grids for Adaptive Regime
ADAPTIVE_DEFAULT_GRIDS = {
    "rr": [1.5, 2.0, 2.5, 3.0],
    "ema_length": [20, 50],
    "ema_enabled": [True],
    "ema_slope_enabled": [True],
    "min_range_points": [1.0, 2.0, 3.0],
    "breakout_buffer": [0.5, 1.0, 1.5],
    "volume_filter_enabled": [True, False],
    "atr_filter_enabled": [True, False],
    "allow_short": [False],
    "min_confirmation_score": [3, 4, 5],
    "min_breakout_strength": [0.25, 0.5, 1.0],
    "eod_exit": ["15:30", "15:50"],
}

SORTABLE_METRICS = [
    "total_pnl_dollars",
    "profit_factor",
    "sharpe_ratio",
    "win_rate_pct",
    "expectancy",
    "max_drawdown_pct",    # lower is better — we negate for sorting
    "total_return_pct",
]


def parse_args():
    p = argparse.ArgumentParser(description="MES Parameter Optimizer")
    p.add_argument("--data", required=True, help="Path to OHLCV CSV or Parquet file")
    p.add_argument("--strategy", default="orb", choices=["orb", "adaptive_regime"],
                   help="Strategy to optimize (default: orb)")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--timezone", default="America/New_York")
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--commission", type=float, default=0.62)
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--metric", default="profit_factor",
                   choices=SORTABLE_METRICS,
                   help="Metric to rank results by")
    p.add_argument("--top", type=int, default=10, help="Show top N results")
    p.add_argument("--results-dir", default="results", help="Output directory")

    # Optional: override individual grids via comma-separated values (ORB)
    p.add_argument("--or-minutes", default=None,
                   help="Comma-separated OR durations to test (e.g. 10,15,20)")
    p.add_argument("--rr-values", default=None,
                   help="Comma-separated R:R values (e.g. 1.5,2.0,3.0)")
    p.add_argument("--ema-values", default=None,
                   help="Comma-separated EMA lengths (e.g. 20,50)")
    p.add_argument("--eod-values", default=None,
                   help="Comma-separated EOD exit times (e.g. 15:30,15:50)")
    p.add_argument("--min-range-values", default=None,
                   help="Comma-separated min range sizes (e.g. 0,1.5,2.0,3.0)")
    p.add_argument("--max-entry-time-values", default=None,
                   help="Comma-separated max entry times (e.g. '',10:15,10:30)")
    p.add_argument("--shorts", action="store_true",
                   help="Include shorts in grid search")

    # Adaptive regime grid overrides
    p.add_argument("--breakout-buffer-values", default=None,
                   help="Comma-separated breakout buffer values (e.g. 0,0.25,0.5)")
    p.add_argument("--min-score-values", default=None,
                   help="Comma-separated min confirmation scores (e.g. 3,4,5)")
    p.add_argument("--min-strength-values", default=None,
                   help="Comma-separated min breakout strength values (e.g. 0.25,0.5,1.0)")

    return p.parse_args()


def build_orb_grids(args) -> dict:
    """Build ORB parameter grids, using CLI overrides if provided."""
    grids = dict(ORB_DEFAULT_GRIDS)

    if args.or_minutes:
        grids["or_minutes"] = [int(x) for x in args.or_minutes.split(",")]
    if args.rr_values:
        grids["rr"] = [float(x) for x in args.rr_values.split(",")]
    if args.ema_values:
        grids["ema_length"] = [int(x) for x in args.ema_values.split(",")]
    if args.eod_values:
        grids["eod_exit"] = [x.strip() for x in args.eod_values.split(",")]
    if args.min_range_values:
        grids["min_range_points"] = [float(x) for x in args.min_range_values.split(",")]
    if args.max_entry_time_values:
        grids["max_entry_time"] = [x.strip() for x in args.max_entry_time_values.split(",")]
    if args.shorts:
        grids["shorts_enabled"] = [True, False]

    return grids


def build_adaptive_grids(args) -> dict:
    """Build Adaptive Regime parameter grids, using CLI overrides if provided."""
    grids = dict(ADAPTIVE_DEFAULT_GRIDS)

    if args.rr_values:
        grids["rr"] = [float(x) for x in args.rr_values.split(",")]
    if args.ema_values:
        grids["ema_length"] = [int(x) for x in args.ema_values.split(",")]
    if args.eod_values:
        grids["eod_exit"] = [x.strip() for x in args.eod_values.split(",")]
    if args.min_range_values:
        grids["min_range_points"] = [float(x) for x in args.min_range_values.split(",")]
    if args.breakout_buffer_values:
        grids["breakout_buffer"] = [float(x) for x in args.breakout_buffer_values.split(",")]
    if args.min_score_values:
        grids["min_confirmation_score"] = [int(x) for x in args.min_score_values.split(",")]
    if args.min_strength_values:
        grids["min_breakout_strength"] = [float(x) for x in args.min_strength_values.split(",")]
    if args.shorts:
        grids["allow_short"] = [True, False]

    return grids


def generate_combinations(grids: dict) -> list[dict]:
    """Generate all parameter combinations from the grid."""
    keys = list(grids.keys())
    combos = []
    for values in itertools.product(*[grids[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def or_minutes_to_end(or_start: str, minutes: int) -> str:
    """Convert opening range start + duration in minutes to end time string."""
    h, m = map(int, or_start.split(":"))
    total = h * 60 + m + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()
    strategy_name = args.strategy

    # Load data once
    logger.info("Loading data from %s ...", args.data)
    bars = load_bars(args.data, timezone=args.timezone, start=args.start, end=args.end)
    logger.info("Loaded %d bars from %s to %s", len(bars), bars.index[0], bars.index[-1])

    instrument = InstrumentConfig(contract_size=args.contracts)
    bt_cfg = BacktestConfig(
        slippage_ticks=args.slippage_ticks,
        commission_per_side=args.commission,
        initial_capital=args.capital,
        results_dir=args.results_dir,
    )

    if strategy_name == "adaptive_regime":
        grids = build_adaptive_grids(args)
    else:
        grids = build_orb_grids(args)

    combos = generate_combinations(grids)
    total = len(combos)
    logger.info("Running %d %s parameter combinations ...", total, strategy_name)

    results = []
    failed = 0
    t0 = time.time()

    for i, params in enumerate(combos, 1):
        if strategy_name == "adaptive_regime":
            strat_cfg = AdaptiveRegimeConfig(
                timezone=args.timezone,
                range_start_time="09:30",
                range_end_time="09:45",
                ema_length=params["ema_length"],
                ema_enabled=params["ema_enabled"],
                ema_slope_enabled=params["ema_slope_enabled"],
                reward_risk=params["rr"],
                end_of_day_exit_time=params["eod_exit"],
                min_range_points=params["min_range_points"],
                breakout_buffer_points=params["breakout_buffer"],
                volume_filter_enabled=params["volume_filter_enabled"],
                atr_filter_enabled=params["atr_filter_enabled"],
                allow_short=params["allow_short"],
                min_confirmation_score=params["min_confirmation_score"],
                min_breakout_strength=params.get("min_breakout_strength", 0.5),
            )
            strategy = AdaptiveRegimeStrategy(strat_cfg)
            engine = BacktestEngine(instrument, strat_cfg, bt_cfg, strategy=strategy)
        else:
            or_end = or_minutes_to_end("09:30", params["or_minutes"])
            strat_cfg = StrategyConfig(
                timezone=args.timezone,
                or_start="09:30",
                or_end=or_end,
                ema_length=params["ema_length"],
                ema_enabled=params["ema_enabled"],
                reward_risk_ratio=params["rr"],
                eod_exit_time=params["eod_exit"],
                min_range_points=params["min_range_points"],
                max_entry_time=params["max_entry_time"],
                shorts_enabled=params["shorts_enabled"],
            )
            strategy = ORBStrategy(strat_cfg)
            engine = BacktestEngine(instrument, strat_cfg, bt_cfg, strategy=strategy)

        try:
            result = engine.run(bars)
            metrics = compute_metrics(result, bt_cfg)

            row = {"strategy": strategy_name, **params, **metrics}
            results.append(row)
        except Exception:
            failed += 1
            logger.error(
                "Combination %d/%d FAILED (params=%s): %s",
                i, total, params, traceback.format_exc().splitlines()[-1],
            )
            continue

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            logger.info(
                "[%d/%d] %.1f runs/sec | last: %d trades, PnL=$%,.0f, PF=%.2f",
                i, total, rate, metrics['total_trades'],
                metrics['total_pnl_dollars'], metrics['profit_factor'],
            )

    elapsed = time.time() - t0
    logger.info("Completed %d runs in %.1fs (%d failed)", total, elapsed, failed)

    # Sort results
    sort_key = args.metric
    reverse = True
    if sort_key == "max_drawdown_pct":
        reverse = False

    results.sort(key=lambda r: r.get(sort_key, 0), reverse=reverse)

    # Print top N
    print(f"\n{'=' * 130}")
    print(f"  TOP {args.top} by {args.metric} ({strategy_name})")
    print(f"{'=' * 130}")

    if strategy_name == "adaptive_regime":
        header = (f"{'Rank':>4} | {'R:R':>4} | {'EMA':>4} | {'EMA?':>5} | {'Slope':>5} | "
                  f"{'MinR':>5} | {'Buf':>4} | {'Vol?':>4} | {'ATR?':>4} | {'Short':>5} | {'MinS':>4} | {'BkSt':>4} | "
                  f"{'Trades':>6} | {'WinR%':>6} | {'PnL$':>9} | "
                  f"{'PF':>6} | {'Sharpe':>7} | {'MaxDD%':>7} | {'Expect':>7}")
        print(header)
        print("-" * 140)

        for rank, r in enumerate(results[:args.top], 1):
            ema_str = "Yes" if r.get("ema_enabled") else "No"
            slope_str = "Yes" if r.get("ema_slope_enabled") else "No"
            vol_str = "Yes" if r.get("volume_filter_enabled") else "No"
            atr_str = "Yes" if r.get("atr_filter_enabled") else "No"
            short_str = "Yes" if r.get("allow_short") else "No"
            bk_str = f"{r.get('min_breakout_strength', 0.5):.1f}"
            print(f"{rank:4d} | {r['rr']:4.1f} | {r['ema_length']:4d} | "
                  f"{ema_str:>5} | {slope_str:>5} | {r['min_range_points']:5.1f} | "
                  f"{r['breakout_buffer']:4.2f} | {vol_str:>4} | {atr_str:>4} | "
                  f"{short_str:>5} | {r['min_confirmation_score']:4d} | {bk_str:>4} | "
                  f"{r['total_trades']:6d} | {r['win_rate_pct']:5.1f}% | "
                  f"${r['total_pnl_dollars']:>8,.0f} | {r['profit_factor']:6.2f} | "
                  f"{r['sharpe_ratio']:7.2f} | {r['max_drawdown_pct']:6.1f}% | "
                  f"${r['expectancy']:>6,.0f}")
    else:
        header = (f"{'Rank':>4} | {'OR(m)':>5} | {'R:R':>4} | {'EMA':>4} | {'EMA?':>5} | "
                  f"{'EOD':>5} | {'MinR':>5} | {'MaxT':>5} | {'Short':>5} | "
                  f"{'Trades':>6} | {'WinR%':>6} | {'PnL$':>9} | "
                  f"{'PF':>6} | {'Sharpe':>7} | {'MaxDD%':>7}")
        print(header)
        print("-" * 130)

        for rank, r in enumerate(results[:args.top], 1):
            ema_str = "Yes" if r.get("ema_enabled") else "No"
            short_str = "Yes" if r.get("shorts_enabled") else "No"
            max_t = r.get("max_entry_time", "") or "--"
            print(f"{rank:4d} | {r['or_minutes']:5d} | {r['rr']:4.1f} | {r['ema_length']:4d} | "
                  f"{ema_str:>5} | {r['eod_exit']:>5} | {r['min_range_points']:5.1f} | "
                  f"{max_t:>5} | {short_str:>5} | {r['total_trades']:6d} | "
                  f"{r['win_rate_pct']:5.1f}% | ${r['total_pnl_dollars']:>8,.0f} | "
                  f"{r['profit_factor']:6.2f} | {r['sharpe_ratio']:7.2f} | "
                  f"{r['max_drawdown_pct']:6.1f}%")

    # Export full results CSV
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = str(out / "optimization_results.csv")

    # Flatten nested dicts for CSV
    flat_results = []
    for r in results:
        flat = {}
        for k, v in r.items():
            if isinstance(v, dict):
                for dk, dv in v.items():
                    flat[f"{k}_{dk}"] = dv
            else:
                flat[k] = v
        flat_results.append(flat)

    if flat_results:
        fieldnames = list(flat_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)

    print(f"\nFull results saved to {csv_path}")
    print(f"\nWARNING: These results are in-sample. Validate on out-of-sample data")
    print(f"   before drawing conclusions. Overfitting is the #1 risk in optimization.")


if __name__ == "__main__":
    main()
