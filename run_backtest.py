#!/usr/bin/env python3
"""
MES Backtest Runner
====================

Run a historical backtest of a strategy on MES futures bar data.

Supported strategies:
  - orb              : Opening Range Breakout
  - adaptive_regime  : Regime-Aware Breakout/Continuation

Usage:
    python run_backtest.py --data data/mes_5m.csv --strategy orb
    python run_backtest.py --data data/mes_5m.csv --strategy adaptive_regime
    python run_backtest.py --data data/mes_5m.csv --no-plot
    python run_backtest.py --data data/mes_5m.csv --start 2024-01-01 --end 2024-06-30
    python run_backtest.py --data data/mes_5m.csv --rr 3.0 --ema-length 20
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd

from config.settings import InstrumentConfig, StrategyConfig, BacktestConfig, AdaptiveRegimeConfig, EvalConfig
from data.loader import load_bars
from backtest.engine import BacktestEngine
from backtest.metrics import (
    compute_metrics,
    export_trades_csv,
    export_equity_csv,
    export_metrics_json,
    plot_equity_curve,
    metrics_to_comparison_row,
    equity_list_to_comparison_row,
    print_benchmark_table,
    print_strategy_comparison,
    export_benchmark_table,
    plot_normalized_comparison,
    plot_drawdown_comparison,
)
from backtest.benchmark import (
    always_long_benchmark, flat_benchmark, ema_directional_benchmark, orb_benchmark,
)
from strategy.orb import ORBStrategy
from strategy.adaptive_regime import AdaptiveRegimeStrategy
from strategy.hybrid_ema_ml import HybridEMAMLStrategy, HybridEMAMLConfig
from backtest.sweep import run_lookback_sweep, print_sweep_table, export_sweep_csv


def parse_args():
    p = argparse.ArgumentParser(description="MES Backtest Runner")
    # Required
    p.add_argument("--data", required=True, help="Path to OHLCV CSV or Parquet file")

    # Strategy selection
    p.add_argument("--strategy", default="orb",
                   choices=["orb", "adaptive_regime", "hybrid_ema_ml"],
                   help="Strategy to backtest (default: orb)")

    # Hybrid EMA+ML specific
    p.add_argument("--ml-threshold", type=float, default=0.6,
                   help="ML probability threshold for hybrid strategy (default: 0.6)")
    p.add_argument("--ml-model", default="models/ema_model.pkl",
                   help="Path to trained ML model (default: models/ema_model.pkl)")
    p.add_argument("--split", default=None,
                   choices=["train", "validation", "test", "holdout"],
                   help="Restrict backtest to a specific data split (uses dates from model pkl)")
    p.add_argument("--ml-selection-mode", default="threshold",
                   choices=["threshold", "top_n", "top_pct"],
                   help="ML trade selection mode (default: threshold)")
    p.add_argument("--ml-top-n", type=int, default=1,
                   help="For top_n mode: accept top N candidates per rolling window (default: 1)")
    p.add_argument("--ml-top-pct", type=float, default=0.30,
                   help="For top_pct mode: accept top X%% of candidates (default: 0.30)")
    p.add_argument("--ml-lookback", type=int, default=20,
                   help="Rolling window size for ranking modes (default: 20)")
    p.add_argument("--ml-top-n-long", type=int, default=None,
                   help="Direction-specific top_n for longs (overrides --ml-top-n)")
    p.add_argument("--ml-top-n-short", type=int, default=None,
                   help="Direction-specific top_n for shorts (overrides --ml-top-n)")
    p.add_argument("--ml-top-pct-long", type=float, default=None,
                   help="Direction-specific top_pct for longs (overrides --ml-top-pct)")
    p.add_argument("--ml-top-pct-short", type=float, default=None,
                   help="Direction-specific top_pct for shorts (overrides --ml-top-pct)")

    # Position sizing
    p.add_argument("--position-sizing-mode", default="none",
                   choices=["none", "linear", "convex", "hybrid"],
                   help="ML percentile-based position sizing mode (default: none)")
    p.add_argument("--base-size", type=float, default=1.0,
                   help="Base (max) position size multiplier (default: 1.0)")

    # Multi-candidate mode
    p.add_argument("--multi-candidate", action="store_true",
                   help="Enable multi-candidate mode: multiple EMA lengths × entry types")
    p.add_argument("--max-trades-per-day", type=int, default=3,
                   help="Max concurrent trades per day in multi-candidate mode (default: 3)")
    p.add_argument("--ema-periods", nargs="+", type=int, default=[50],
                   help="EMA periods to evaluate (default: 50). e.g. --ema-periods 20 50 100")
    p.add_argument("--entry-types", nargs="+", default=["breakout"],
                   choices=["breakout", "pullback", "momentum"],
                   help="Entry types to evaluate (default: breakout). "
                        "e.g. --entry-types breakout pullback momentum")
    p.add_argument("--selection-strategy", default="global_ml",
                   choices=["global_ml", "priority", "priority_ml_sizing"],
                   help="Candidate selection strategy (default: global_ml). "
                        "priority=group by entry type, ML within groups. "
                        "priority_ml_sizing=priority order, ML only for sizing.")
    p.add_argument("--ml-within-group-threshold", type=float, default=0.0,
                   help="Within-group ML threshold for priority mode (default: 0.0 = no filter)")

    # Date range
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")

    # Shared overrides
    p.add_argument("--or-start", default="09:30", help="Opening range start (HH:MM)")
    p.add_argument("--or-end", default="09:45", help="Opening range end (HH:MM)")
    p.add_argument("--ema-length", type=int, default=50, help="EMA period")
    p.add_argument("--no-ema", action="store_true", help="Disable EMA filter")
    p.add_argument("--rr", type=float, default=None, help="Reward:risk ratio")
    p.add_argument("--eod-exit", default="15:50", help="End-of-day exit time (HH:MM)")
    p.add_argument("--timezone", default="America/New_York", help="Session timezone")
    p.add_argument("--shorts", action="store_true", help="Enable short entries")
    p.add_argument("--min-range", type=float, default=None,
                   help="Minimum opening range size in points")
    p.add_argument("--max-entry-time", default=None,
                   help="Latest allowed entry time (HH:MM)")

    # Adaptive regime specific
    p.add_argument("--no-ema-slope", action="store_true", help="Disable EMA slope filter (adaptive)")
    p.add_argument("--ema-slope-lookback", type=int, default=5, help="EMA slope lookback (adaptive)")
    p.add_argument("--max-range", type=float, default=20.0, help="Max range points (adaptive)")
    p.add_argument("--breakout-buffer", type=float, default=1.0, help="Breakout buffer points (adaptive)")
    p.add_argument("--no-volume-filter", action="store_true", help="Disable volume filter (adaptive)")
    p.add_argument("--volume-lookback", type=int, default=20, help="Volume lookback (adaptive)")
    p.add_argument("--volume-threshold", type=float, default=0.8, help="Volume threshold ratio (adaptive)")
    p.add_argument("--no-atr-filter", action="store_true", help="Disable ATR filter (adaptive)")
    p.add_argument("--atr-length", type=int, default=14, help="ATR length (adaptive)")
    p.add_argument("--atr-min", type=float, default=1.0, help="ATR min threshold (adaptive)")
    p.add_argument("--min-score", type=int, default=4, help="Min confirmation score (adaptive)")
    p.add_argument("--long-min-score", type=int, default=None,
                   help="Min score for long entries (overrides --min-score for longs)")
    p.add_argument("--short-min-score", type=int, default=None,
                   help="Min score for short entries (overrides --min-score for shorts)")
    p.add_argument("--short-breakout-buffer", type=float, default=None,
                   help="Breakout buffer for shorts (default: same as --breakout-buffer)")
    p.add_argument("--short-ema-slope-min", type=float, default=None,
                   help="Min abs(EMA slope) for short entries")
    p.add_argument("--min-breakout-strength", type=float, default=0.5,
                   help="Min breakout strength in points (adaptive)")
    p.add_argument("--strict-shorts", action="store_true",
                   help="Enable strict mode for short entries (higher bar)")
    p.add_argument("--export-diagnostics", action="store_true",
                   help="Export per-day regime diagnostics CSV (adaptive)")

    # Cost model
    p.add_argument("--slippage-ticks", type=float, default=1.0, help="Slippage per side in ticks")
    p.add_argument("--commission", type=float, default=0.62, help="Commission per side per contract")
    p.add_argument("--capital", type=float, default=10000.0, help="Initial capital ($)")
    p.add_argument("--contracts", type=int, default=1, help="Number of contracts")

    # Evaluation mode
    p.add_argument("--eval", action="store_true", help="Enable prop-firm evaluation mode")
    p.add_argument("--eval-capital", type=float, default=25000.0,
                   help="Eval starting capital (default: $25,000)")
    p.add_argument("--eval-target", type=float, default=1500.0,
                   help="Eval profit target (default: $1,500)")
    p.add_argument("--eval-drawdown", type=float, default=1000.0,
                   help="Eval max trailing drawdown (default: $1,000)")

    # Sweep mode
    p.add_argument("--sweep-lookback", nargs="+", type=int, default=None,
                   metavar="N",
                   help="Run lookback sweep across multiple window sizes "
                        "(e.g. --sweep-lookback 10 20 40 60). "
                        "Requires --strategy hybrid_ema_ml with top_n or top_pct mode.")

    # Output
    p.add_argument("--results-dir", default="results", help="Output directory")
    p.add_argument("--no-plot", action="store_true", help="Skip equity curve plot")
    p.add_argument("--no-benchmark", action="store_true", help="Skip benchmark calculation")

    return p.parse_args()


def build_orb_strategy(args):
    """Build ORB strategy + config from CLI args."""
    strat_cfg = StrategyConfig(
        timezone=args.timezone,
        or_start=args.or_start,
        or_end=args.or_end,
        ema_length=args.ema_length,
        ema_enabled=not args.no_ema,
        reward_risk_ratio=args.rr if args.rr is not None else 1.5,
        eod_exit_time=args.eod_exit,
        shorts_enabled=args.shorts,
        min_range_points=args.min_range if args.min_range is not None else 0.0,
        max_entry_time=args.max_entry_time if args.max_entry_time is not None else "",
    )
    strategy = ORBStrategy(strat_cfg)
    return strategy, strat_cfg


def build_adaptive_strategy(args):
    """Build Adaptive Regime strategy + config from CLI args."""
    # Only override asymmetric params if user explicitly set them
    kwargs = dict(
        timezone=args.timezone,
        range_start_time=args.or_start,
        range_end_time=args.or_end,
        max_entry_time=args.max_entry_time if args.max_entry_time is not None else "14:00",
        allow_long=True,
        allow_short=args.shorts,
        ema_length=args.ema_length,
        ema_enabled=not args.no_ema,
        ema_slope_enabled=not args.no_ema_slope,
        ema_slope_lookback=args.ema_slope_lookback,
        min_range_points=args.min_range if args.min_range is not None else 2.0,
        max_range_points=args.max_range,
        breakout_buffer_points=args.breakout_buffer,
        volume_filter_enabled=not args.no_volume_filter,
        volume_lookback=args.volume_lookback,
        volume_threshold_ratio=args.volume_threshold,
        atr_filter_enabled=not args.no_atr_filter,
        atr_length=args.atr_length,
        atr_min_threshold=args.atr_min,
        reward_risk=args.rr if args.rr is not None else 2.0,
        end_of_day_exit_time=args.eod_exit,
        min_confirmation_score=args.min_score,
        min_breakout_strength=args.min_breakout_strength,
        strict_shorts=args.strict_shorts,
    )
    if args.long_min_score is not None:
        kwargs['long_min_score'] = args.long_min_score
    if args.short_min_score is not None:
        kwargs['short_min_score'] = args.short_min_score
    if args.short_breakout_buffer is not None:
        kwargs['short_breakout_buffer_points'] = args.short_breakout_buffer
    if args.short_ema_slope_min is not None:
        kwargs['short_ema_slope_min'] = args.short_ema_slope_min

    cfg = AdaptiveRegimeConfig(**kwargs)
    strategy = AdaptiveRegimeStrategy(cfg)
    return strategy, cfg


def export_diagnostics_csv(diagnostics, path: str):
    """Export per-day regime diagnostics to CSV."""
    if not diagnostics:
        return
    rows = []
    for d in diagnostics:
        rows.append({
            "date": d.date.isoformat(),
            "regime": d.regime,
            "regime_reason": d.regime_reason,
            "preferred_direction": d.preferred_direction,
            "or_high": d.or_high,
            "or_low": d.or_low,
            "or_range": round(d.or_range, 2) if d.or_range else "",
            "ema": round(d.ema, 2) if d.ema is not None else "",
            "ema_slope": round(d.ema_slope, 6) if d.ema_slope is not None else "",
            "atr": round(d.atr, 2) if d.atr is not None else "",
            "relative_volume": round(d.relative_volume, 2) if d.relative_volume is not None else "",
            "trade_taken": d.trade_taken,
            "trade_direction": d.trade_direction,
            "entry_price": round(d.entry_price, 2) if d.entry_price is not None else "",
            "stop_loss": round(d.stop_loss, 2) if d.stop_loss is not None else "",
            "take_profit": round(d.take_profit, 2) if d.take_profit is not None else "",
            "exit_reason": d.exit_reason,
            "filter_score": d.filter_score,
            "filter_min_score": d.filter_min_score,
            "breakout_distance": round(d.breakout_distance, 2) if d.breakout_distance else "",
            "breakout_strength": round(d.breakout_strength, 2) if d.breakout_strength else "",
            "ema_slope_value": round(d.ema_slope_value, 6) if d.ema_slope_value else "",
            "filter_detail": d.filter_detail,
            "skip_reason": d.skip_reason,
        })
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Diagnostics saved to {path}")


def build_hybrid_strategy(args):
    """Build Hybrid EMA+ML strategy + config from CLI args."""
    cfg = HybridEMAMLConfig(
        timezone=args.timezone,
        session_open=args.or_start,
        range_start=args.or_start,
        range_end=args.or_end,
        eod_exit_time=args.eod_exit,
        ema_length=args.ema_length,
        reward_risk=args.rr if args.rr is not None else 1.5,
        allow_shorts=args.shorts,
        ml_selection_mode=args.ml_selection_mode,
        ml_threshold=args.ml_threshold,
        ml_top_n=args.ml_top_n,
        ml_top_pct=args.ml_top_pct,
        ml_lookback=args.ml_lookback,
        ml_top_n_long=args.ml_top_n_long,
        ml_top_n_short=args.ml_top_n_short,
        ml_top_pct_long=args.ml_top_pct_long,
        ml_top_pct_short=args.ml_top_pct_short,
        model_path=args.ml_model,
        position_sizing_mode=args.position_sizing_mode,
        base_size=args.base_size,
        multi_candidate=args.multi_candidate,
        max_trades_per_day=args.max_trades_per_day,
        ema_periods=tuple(args.ema_periods),
        entry_types=tuple(args.entry_types),
        selection_strategy=args.selection_strategy,
        ml_within_group_threshold=args.ml_within_group_threshold,
    )
    strategy = HybridEMAMLStrategy(cfg)
    # Build a StrategyConfig for benchmark compatibility
    strat_cfg = StrategyConfig(
        timezone=args.timezone,
        or_start=args.or_start,
        or_end=args.or_end,
        ema_length=args.ema_length,
        ema_enabled=True,
        reward_risk_ratio=args.rr if args.rr is not None else 1.5,
        eod_exit_time=args.eod_exit,
        shorts_enabled=args.shorts,
    )
    return strategy, strat_cfg, cfg


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    strategy_name = args.strategy
    hybrid_cfg = None

    # --- Build strategy ---
    if strategy_name == "adaptive_regime":
        strategy, strat_cfg = build_adaptive_strategy(args)
        tz = strat_cfg.timezone
    elif strategy_name == "hybrid_ema_ml":
        strategy, strat_cfg, hybrid_cfg = build_hybrid_strategy(args)
        tz = strat_cfg.timezone
    else:
        strategy, strat_cfg = build_orb_strategy(args)
        tz = strat_cfg.timezone

    instrument = InstrumentConfig(contract_size=args.contracts)

    bt_cfg = BacktestConfig(
        slippage_ticks=args.slippage_ticks,
        commission_per_side=args.commission,
        initial_capital=args.capital,
        results_dir=args.results_dir,
    )

    # --- Eval mode ---
    eval_cfg = None
    if args.eval:
        eval_cfg = EvalConfig(
            enabled=True,
            starting_capital=args.eval_capital,
            profit_target=args.eval_target,
            max_drawdown=args.eval_drawdown,
        )
        print(f"Evaluation mode: ${eval_cfg.starting_capital:,.0f} capital, "
              f"${eval_cfg.profit_target:,.0f} target, "
              f"${eval_cfg.max_drawdown:,.0f} trailing drawdown")

    # --- Load data ---
    print(f"Loading data from {args.data} ...")
    bars = load_bars(args.data, timezone=tz, start=args.start, end=args.end)
    print(f"Loaded {len(bars)} bars from {bars.index[0]} to {bars.index[-1]}")

    # --- Filter by split dates (for OOS-only backtesting) ---
    if args.split is not None:
        import pickle
        print(f"Filtering bars to '{args.split}' split ...")
        try:
            with open(args.ml_model, "rb") as f:
                model_data = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: Model file {args.ml_model} not found. Cannot filter by split.")
            sys.exit(1)
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as exc:
            print(f"ERROR: Failed to load model from {args.ml_model}: {exc}")
            sys.exit(1)

        split_dates = model_data.get("split_dates")
        if split_dates is None:
            print("ERROR: Model pkl has no split_dates. Re-train with the fixed pipeline.")
            sys.exit(1)
        if args.split not in split_dates:
            print(f"ERROR: Split '{args.split}' not in model. "
                  f"Available: {list(split_dates.keys())}")
            sys.exit(1)

        sd = split_dates[args.split]
        split_start = pd.Timestamp(sd["start"]).date()
        split_end = pd.Timestamp(sd["end"]).date()
        bar_dates = bars.index.date
        bars = bars[(bar_dates >= split_start) & (bar_dates <= split_end)]
        print(f"  Split '{args.split}': {split_start} to {split_end}")
        print(f"  Bars after filter: {len(bars)}")
        if len(bars) == 0:
            print("ERROR: No bars in the selected split range.")
            sys.exit(1)


    # --- Lookback sweep mode ---
    if args.sweep_lookback is not None:
        if strategy_name != "hybrid_ema_ml":
            print("ERROR: --sweep-lookback requires --strategy hybrid_ema_ml")
            sys.exit(1)
        if hybrid_cfg.ml_selection_mode == "threshold":
            print("ERROR: --sweep-lookback requires --ml-selection-mode top_n or top_pct")
            sys.exit(1)

        lookbacks = sorted(args.sweep_lookback)
        print(f"Running lookback sweep: {lookbacks}")
        print(f"  Selection mode : {hybrid_cfg.ml_selection_mode}")
        print(f"  Other params   : held constant")
        print()

        sweep_results = run_lookback_sweep(
            bars=bars,
            base_hybrid_cfg=hybrid_cfg,
            base_strat_cfg=strat_cfg,
            instrument=instrument,
            bt_cfg=bt_cfg,
            lookbacks=lookbacks,
            eval_config=eval_cfg,
        )

        out = Path(bt_cfg.results_dir)
        out.mkdir(parents=True, exist_ok=True)

        print_sweep_table(sweep_results)
        export_sweep_csv(sweep_results, str(out / "lookback_sweep.csv"))
        return

    # --- Run backtest ---
    print(f"Running {strategy_name} backtest ...")
    engine = BacktestEngine(instrument, strat_cfg, bt_cfg, strategy=strategy)
    result = engine.run(bars, eval_config=eval_cfg)

    # --- Compute metrics ---
    metrics = compute_metrics(result, bt_cfg)

    # --- Benchmarks & Comparison ---
    chart_benchmarks = []   # for Charts A/B (excludes always-long)
    dd_benchmarks = []      # for Chart C (excludes flat)
    comparison_rows = []    # for benchmark table
    orb_result = None
    orb_metrics = None
    al_eq = None

    if not args.no_benchmark:
        bench_strat = StrategyConfig(
            timezone=tz,
            or_start=args.or_start,
            or_end=args.or_end,
            eod_exit_time=args.eod_exit,
            ema_length=args.ema_length,
            ema_enabled=not args.no_ema,
            shorts_enabled=args.shorts,
        )
        print("Computing benchmarks ...")

        # ORB as formal benchmark (when running adaptive_regime)
        if strategy_name == "adaptive_regime":
            orb_eq, orb_result = orb_benchmark(bars, instrument, bt_cfg, bench_strat)
            orb_metrics = compute_metrics(orb_result, bt_cfg)
            chart_benchmarks.append((orb_eq, "ORB", "#E91E63"))
            dd_benchmarks.append((orb_eq, "ORB", "#E91E63"))
            comparison_rows.append(metrics_to_comparison_row("ORB", orb_metrics))

        # EMA directional
        ema_eq = ema_directional_benchmark(bars, instrument, bt_cfg, bench_strat,
                                           ema_length=args.ema_length)
        chart_benchmarks.append((ema_eq, "EMA Directional", "#4CAF50"))
        dd_benchmarks.append((ema_eq, "EMA Directional", "#4CAF50"))
        comparison_rows.append(equity_list_to_comparison_row(
            "EMA Directional", ema_eq, bars.index, bt_cfg.initial_capital))

        # Flat (no-trade)
        flat_eq = flat_benchmark(bars, bt_cfg)
        chart_benchmarks.append((flat_eq, "Flat (No Trade)", "#9E9E9E"))
        comparison_rows.append(equity_list_to_comparison_row(
            "Flat (No Trade)", flat_eq, bars.index, bt_cfg.initial_capital))

        # Always-long (separate chart to avoid scale distortion)
        al_eq = always_long_benchmark(bars, instrument, bt_cfg, bench_strat)
        comparison_rows.append(equity_list_to_comparison_row(
            "Always-Long", al_eq, bars.index, bt_cfg.initial_capital))

    # Main strategy comparison row
    main_comparison_row = metrics_to_comparison_row(strategy_name, metrics)

    # --- Export results ---
    out = Path(bt_cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    export_trades_csv(result.trades, str(out / bt_cfg.trades_csv))
    export_metrics_json(metrics, str(out / bt_cfg.metrics_file))
    export_equity_csv(result, str(out / bt_cfg.equity_csv))

    # Export ORB benchmark results (when available)
    if orb_result is not None:
        export_trades_csv(orb_result.trades, str(out / "orb_trades.csv"))
        export_metrics_json(orb_metrics, str(out / "orb_metrics.json"))
        export_equity_csv(orb_result, str(out / "orb_equity_curve.csv"))

    # --- Charts ---
    if not args.no_plot:
        main_equity = [ep.equity for ep in result.equity_curve]
        timestamps = list(bars.index)
        main_color = "#2196F3"

        # Chart B: Raw equity curve (backward-compatible 2-panel)
        result_len = len(result.equity_curve)
        truncated_bm = [(eq[:result_len], n, c) for eq, n, c in chart_benchmarks]
        plot_equity_curve(result, str(out / bt_cfg.plot_file),
                          benchmarks=truncated_bm if truncated_bm else None)

        if chart_benchmarks:
            all_series = [(main_equity, strategy_name, main_color)] + chart_benchmarks

            # Chart A: Normalized comparison (DEFAULT)
            plot_normalized_comparison(
                all_series, timestamps, bt_cfg.initial_capital,
                str(out / "comparison_normalized.png"))

            # Chart C: Drawdown comparison
            dd_series = [(main_equity, strategy_name, main_color)] + dd_benchmarks
            plot_drawdown_comparison(dd_series, timestamps,
                                    str(out / "comparison_drawdown.png"))

            # Chart D: Always-long separate (avoids scale distortion)
            if al_eq is not None:
                al_series = [(main_equity, strategy_name, main_color),
                             (al_eq, "Always-Long", "#FF9800")]
                plot_normalized_comparison(
                    al_series, timestamps, bt_cfg.initial_capital,
                    str(out / "comparison_always_long.png"),
                    title_suffix="vs Always-Long")

    # --- Export diagnostics for adaptive_regime ---
    if strategy_name == "adaptive_regime" and args.export_diagnostics:
        diag_path = str(out / "regime_diagnostics.csv")
        export_diagnostics_csv(strategy.diagnostics, diag_path)

    # --- Print summary ---
    label = strategy_name.upper().replace("_", " ")
    print("\n" + "=" * 60)
    print(f"  MES {label} BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Strategy       : {strategy_name}")
    print(f"  Period         : {bars.index[0].date()} to {bars.index[-1].date()}")
    print(f"  Bars processed : {metrics['bars_processed']:,}")
    print(f"  Total trades   : {metrics['total_trades']}")
    if metrics['long_trades'] or metrics['short_trades']:
        print(f"    Long trades  : {metrics['long_trades']} "
              f"(win {metrics['long_win_rate_pct']:.1f}%, "
              f"PnL ${metrics['long_pnl_dollars']:,.2f})")
        print(f"    Short trades : {metrics['short_trades']} "
              f"(win {metrics['short_win_rate_pct']:.1f}%, "
              f"PnL ${metrics['short_pnl_dollars']:,.2f})")
    print(f"  Win rate       : {metrics['win_rate_pct']:.1f}%")
    print(f"  Net PnL        : ${metrics['total_pnl_dollars']:,.2f} ({metrics['total_pnl_points']:.1f} pts)")
    print(f"  Profit factor  : {metrics['profit_factor']:.2f}")
    print(f"  Sharpe ratio   : {metrics['sharpe_ratio']:.2f}")
    print(f"  Max drawdown   : ${metrics['max_drawdown_dollars']:,.2f} ({metrics['max_drawdown_pct']:.1f}%)")
    print(f"  Avg win        : ${metrics['avg_win']:,.2f}")
    print(f"  Avg loss       : ${metrics['avg_loss']:,.2f}")
    print(f"  Expectancy     : ${metrics['expectancy']:,.2f}")
    print(f"  Final equity   : ${metrics['final_equity']:,.2f}")
    print(f"  Total return   : {metrics['total_return_pct']:.1f}%")
    print(f"  Commissions    : ${metrics['total_commission']:,.2f}")
    print(f"  Slippage cost  : ${metrics['total_slippage']:,.2f}")
    print("=" * 60)

    # Print regime summary for adaptive_regime
    if strategy_name == "adaptive_regime" and hasattr(strategy, 'diagnostics'):
        diags = strategy.diagnostics
        if diags:
            regime_counts = {}
            trade_count = 0
            direction_counts = {"long": 0, "short": 0}
            for d in diags:
                regime_counts[d.regime] = regime_counts.get(d.regime, 0) + 1
                if d.trade_taken:
                    trade_count += 1
                    direction_counts[d.trade_direction] = direction_counts.get(d.trade_direction, 0) + 1
            print(f"\n  Regime breakdown ({len(diags)} days):")
            for regime, count in sorted(regime_counts.items()):
                print(f"    {regime:12s} : {count}")
            print(f"    Traded days  : {trade_count}")
            print(f"    Long entries : {direction_counts.get('long', 0)}")
            print(f"    Short entries: {direction_counts.get('short', 0)}")
            if args.export_diagnostics:
                print(f"\n  Full diagnostics: {out / 'regime_diagnostics.csv'}")

    # Print ML filter summary for hybrid_ema_ml
    if strategy_name == "hybrid_ema_ml" and hasattr(strategy, 'ml_decisions'):
        decisions = strategy.ml_decisions
        if decisions:
            n_signals = len(decisions)
            n_accepted = sum(1 for d in decisions if d['accepted'])
            n_rejected = n_signals - n_accepted
            avg_prob = sum(d['ml_prob'] for d in decisions) / n_signals if n_signals else 0
            acc_probs = [d['ml_prob'] for d in decisions if d['accepted']]
            rej_probs = [d['ml_prob'] for d in decisions if not d['accepted']]
            avg_acc_prob = sum(acc_probs) / len(acc_probs) if acc_probs else 0
            avg_rej_prob = sum(rej_probs) / len(rej_probs) if rej_probs else 0

            mode = hybrid_cfg.ml_selection_mode
            print(f"\n  ML Filter Summary ({mode} mode):")
            print(f"    EMA signals   : {n_signals}")
            print(f"    Accepted      : {n_accepted} ({n_accepted/n_signals*100:.1f}%)")
            print(f"    Rejected      : {n_rejected} ({n_rejected/n_signals*100:.1f}%)")
            if mode == "threshold":
                print(f"    Threshold     : {hybrid_cfg.ml_threshold:.2f}")
            elif mode == "top_n":
                print(f"    Top-N         : {hybrid_cfg.ml_top_n}")
                print(f"    Lookback      : {hybrid_cfg.ml_lookback}")
                if hybrid_cfg.ml_top_n_long is not None:
                    print(f"    Top-N (long)  : {hybrid_cfg.ml_top_n_long}")
                if hybrid_cfg.ml_top_n_short is not None:
                    print(f"    Top-N (short) : {hybrid_cfg.ml_top_n_short}")
            elif mode == "top_pct":
                print(f"    Top-Pct       : {hybrid_cfg.ml_top_pct:.0%}")
                print(f"    Lookback      : {hybrid_cfg.ml_lookback}")
                if hybrid_cfg.ml_top_pct_long is not None:
                    print(f"    Top-Pct (long): {hybrid_cfg.ml_top_pct_long:.0%}")
                if hybrid_cfg.ml_top_pct_short is not None:
                    print(f"    Top-Pct (short): {hybrid_cfg.ml_top_pct_short:.0%}")
            print(f"    Avg prob (all): {avg_prob:.3f}")
            print(f"    Avg prob (acc): {avg_acc_prob:.3f}")
            print(f"    Avg prob (rej): {avg_rej_prob:.3f}")

            # Show ranking summary for ranking modes
            if mode in ("top_n", "top_pct"):
                ranked = [d for d in decisions if d['window_size'] > 0]
                if ranked:
                    avg_rank = sum(d['rank'] for d in ranked) / len(ranked)
                    avg_window = sum(d['window_size'] for d in ranked) / len(ranked)
                    print(f"    Avg rank      : {avg_rank:.1f}")
                    print(f"    Avg window sz : {avg_window:.1f}")

            # Show position sizing diagnostics
            if hybrid_cfg.position_sizing_mode != "none":
                import numpy as _np
                acc_decisions = [d for d in decisions if d['accepted']]
                sizes = [d['position_size'] for d in acc_decisions]
                pctiles = [d['percentile'] for d in acc_decisions]
                if sizes:
                    avg_size = _np.mean(sizes)
                    min_size = min(sizes)
                    max_size = max(sizes)
                    avg_pctile = _np.mean(pctiles)
                    # Size-weighted return
                    trades_list = result.trades
                    if trades_list:
                        sw_pnl = sum(t.net_pnl for t in trades_list)
                        # Weighted win rate: wins weighted by position size
                        total_size = sum(t.position_size for t in trades_list)
                        sw_wins = sum(t.position_size for t in trades_list if t.net_pnl > 0)
                        sw_wr = (sw_wins / total_size * 100) if total_size > 0 else 0
                        # Correlation between percentile and PnL
                        if len(pctiles) == len(trades_list) and len(pctiles) >= 3:
                            pnls = [t.net_pnl for t in trades_list]
                            corr = float(_np.corrcoef(pctiles, pnls)[0, 1])
                        else:
                            corr = float('nan')
                    else:
                        sw_pnl = 0
                        sw_wr = 0
                        corr = float('nan')

                    print(f"\n  Position Sizing Summary ({hybrid_cfg.position_sizing_mode} mode):")
                    print(f"    Base size     : {hybrid_cfg.base_size:.2f}")
                    print(f"    Avg size      : {avg_size:.3f}")
                    print(f"    Min/Max size  : {min_size:.3f} / {max_size:.3f}")
                    print(f"    Avg percentile: {avg_pctile:.3f}")
                    print(f"    Size-wtd PnL  : ${sw_pnl:,.2f}")
                    print(f"    Size-wtd WR   : {sw_wr:.1f}%")
                    if not _np.isnan(corr):
                        print(f"    Pctile↔PnL r  : {corr:+.3f}")

            # Show multi-candidate diagnostics
            if hybrid_cfg.multi_candidate:
                import numpy as _np2
                from collections import Counter

                print(f"\n  Multi-Candidate Summary:")
                print(f"    EMA periods   : {hybrid_cfg.ema_periods}")
                print(f"    Entry types   : {hybrid_cfg.entry_types}")
                print(f"    Max trades/day: {hybrid_cfg.max_trades_per_day}")

                # Candidates per day
                day_cand_counts = Counter()
                day_trade_counts = Counter()
                for d in decisions:
                    day = d['timestamp'].date()
                    day_cand_counts[day] += 1
                    if d['accepted']:
                        day_trade_counts[day] += 1

                n_days = len(day_cand_counts)
                if n_days > 0:
                    all_cand_counts = list(day_cand_counts.values())
                    all_trade_counts = list(day_trade_counts.values())
                    print(f"    Trading days  : {n_days}")
                    print(f"    Avg cands/day : {_np2.mean(all_cand_counts):.1f}")
                    print(f"    Max cands/day : {max(all_cand_counts)}")
                    print(f"    Avg trades/day: {_np2.mean(all_trade_counts):.1f}" if all_trade_counts else "")

                # Win rate by strategy_type
                type_trades: dict[str, list] = {}
                for t in result.trades:
                    st = getattr(t, 'strategy_type', '') or 'unknown'
                    type_trades.setdefault(st, []).append(t)

                if type_trades:
                    print(f"\n  Win Rate by Strategy Type:")
                    for st, trades_list in sorted(type_trades.items()):
                        n_t = len(trades_list)
                        n_w = sum(1 for t in trades_list if t.net_pnl > 0)
                        wr = n_w / n_t * 100 if n_t > 0 else 0
                        pnl = sum(t.net_pnl for t in trades_list)
                        avg_prob_st = [
                            d['ml_prob'] for d in decisions
                            if d.get('strategy_type') == st and d['accepted']
                        ]
                        avg_p = _np2.mean(avg_prob_st) if avg_prob_st else 0
                        print(f"    {st:20s} : {n_t:3d} trades, "
                              f"WR {wr:5.1f}%, PnL ${pnl:>8,.2f}, "
                              f"avg prob {avg_p:.3f}")

    # --- Eval mode results ---
    if result.eval_result is not None:
        er = result.eval_result
        print(f"\n{'=' * 60}")
        print(f"  EVALUATION RESULTS")
        print(f"{'=' * 60}")
        print(f"  Status              : {er.status}")
        if er.pass_timestamp:
            print(f"  Pass timestamp      : {er.pass_timestamp}")
        if er.fail_timestamp:
            print(f"  Fail timestamp      : {er.fail_timestamp}")
        print(f"  Peak equity         : ${er.peak_equity:,.2f}")
        print(f"  Trail threshold     : ${er.trailing_threshold:,.2f}")
        print(f"  Distance to target  : ${er.distance_to_target:,.2f}")
        print(f"  Distance to fail    : ${er.distance_to_fail:,.2f}")
        print(f"  Trades taken        : {er.trades_taken}")
        print(f"  Trading days used   : {er.trading_days_used}")
        print(f"{'=' * 60}")

    # --- Benchmark comparison ---
    if comparison_rows:
        all_rows = [main_comparison_row] + comparison_rows
        print_benchmark_table(all_rows)

        # Pairwise strategy comparisons
        print(f"\n{'=' * 60}")
        print(f"  STRATEGY COMPARISONS")
        print(f"{'=' * 60}")
        for row in comparison_rows:
            if row['name'] not in ('Flat (No Trade)', 'Always-Long'):
                print_strategy_comparison(main_comparison_row, row)

        # Export comparison data
        export_benchmark_table(all_rows,
                               str(out / "benchmark_comparison.csv"),
                               str(out / "benchmark_comparison.json"))

    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
