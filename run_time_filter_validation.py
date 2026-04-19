"""
Full before/after validation of MNQ max_entry_time=10:30.

Runs the full validation suite with:
  BEFORE: AdaptiveRegimeConfig.for_symbol() WITHOUT the max_entry_time override
  AFTER:  AdaptiveRegimeConfig.for_symbol() WITH the max_entry_time="10:30" on MNQ

Saves comprehensive results to results/adaptive_regime_time_filter_validation/
"""
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
from data.loader import load_bars
from config.settings import AdaptiveRegimeConfig, BacktestConfig, INSTRUMENT_REGISTRY
from strategy.adaptive_regime import AdaptiveRegimeStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics

OUTPUT_DIR = Path("results/adaptive_regime_time_filter_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=25_000.0,
    slippage_ticks=1,
    commission_per_side=2.25,
)

YEARS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
SYMBOLS = ["MNQ", "MES"]


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


def run_backtest(symbol, year, max_entry_override=None):
    """Run one backtest. If max_entry_override given, use it; else use for_symbol() default."""
    bars = load_year_data(symbol, year)
    instrument = INSTRUMENT_REGISTRY[symbol]
    cfg = AdaptiveRegimeConfig.for_symbol(symbol)
    if max_entry_override is not None:
        cfg.max_entry_time = max_entry_override

    strat = AdaptiveRegimeStrategy(cfg)
    engine = BacktestEngine(
        instrument=instrument,
        strategy_config=cfg,
        backtest_config=BACKTEST_CONFIG,
        strategy=strat,
    )
    result = engine.run(bars)
    metrics = compute_metrics(result, BACKTEST_CONFIG)
    sel = strat.selectivity

    # Export trades CSV
    return {
        "metrics": metrics,
        "selectivity": sel,
        "trades": result.trades,
        "result": result,
        "diagnostics": strat.diagnostics,
    }


def fmt_row(year, symbol, mode, m, sel):
    """Format a single result row."""
    return {
        "year": year,
        "symbol": symbol,
        "mode": mode,
        "trades": m.get("total_trades", 0),
        "win_rate": round(m.get("win_rate_pct", 0), 1),
        "avg_trade": round(m.get("expectancy", 0), 2),
        "total_pnl": round(m.get("total_pnl_dollars", 0), 2),
        "profit_factor": round(m.get("profit_factor", 0), 2),
        "max_dd": round(m.get("max_drawdown_dollars", 0), 2),
        "sharpe": round(m.get("sharpe_ratio", 0), 2),
        "days_traded": sel["days_traded"],
        "days_with_range": sel["days_with_range"],
    }


if __name__ == "__main__":
    all_rows = []
    all_json = {}

    # ── Run BEFORE (MNQ max_entry=12:00) and AFTER (MNQ max_entry=10:30, current default) ──
    print("Running backtests...")
    for year in YEARS:
        for symbol in SYMBOLS:
            # BEFORE: force MNQ to 12:00 (old default)
            before_entry = "12:00"
            b = run_backtest(symbol, year, max_entry_override=before_entry)
            brow = fmt_row(year, symbol, "BEFORE", b["metrics"], b["selectivity"])
            all_rows.append(brow)
            all_json[f"BEFORE_{symbol}_{year}"] = brow

            # AFTER: use for_symbol() default (MNQ=10:30, MES=12:00)
            a = run_backtest(symbol, year)
            arow = fmt_row(year, symbol, "AFTER", a["metrics"], a["selectivity"])
            all_rows.append(arow)
            all_json[f"AFTER_{symbol}_{year}"] = arow

            # Save trade CSVs for AFTER
            trades = a["trades"]
            if trades:
                csv_path = OUTPUT_DIR / f"{year}_{symbol.lower()}_trades.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["entry_time", "exit_time", "direction", "entry_price",
                                     "exit_price", "pnl_points", "net_pnl", "exit_reason"])
                    for t in trades:
                        writer.writerow([
                            t.entry_time.isoformat(), t.exit_time.isoformat(),
                            t.direction, f"{t.entry_price:.2f}", f"{t.exit_price:.2f}",
                            f"{t.pnl_points:.2f}", f"{t.net_pnl:.2f}", t.exit_reason,
                        ])

            # Save per-year summary JSON for AFTER
            summary = {
                "year": year,
                "symbol": symbol,
                "metrics": a["metrics"],
                "selectivity": dict(a["selectivity"]),
                "regime_breakdown": {},
            }
            for d in a["diagnostics"]:
                r = d.regime
                summary["regime_breakdown"][r] = summary["regime_breakdown"].get(r, 0) + 1
            json_path = OUTPUT_DIR / f"{year}_{symbol.lower()}_summary.json"
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

        print(f"  {year} done.")

    # ── Print comparison table ──
    print("\n" + "=" * 120)
    print("  BEFORE vs AFTER COMPARISON")
    print("=" * 120)
    hdr = (f"  {'Year':>6} {'Sym':>4} {'Mode':>7} {'Tr':>5} {'WR%':>6} {'AvgTr':>8} "
           f"{'TotalPnL':>10} {'PF':>6} {'MaxDD':>9} {'Sh':>6} {'DaysT':>6}")
    print(hdr)
    print(f"  {'-'*6} {'-'*4} {'-'*7} {'-'*5} {'-'*6} {'-'*8} {'-'*10} {'-'*6} {'-'*9} {'-'*6} {'-'*6}")

    for year in YEARS:
        for symbol in SYMBOLS:
            for mode in ["BEFORE", "AFTER"]:
                r = all_json[f"{mode}_{symbol}_{year}"]
                print(f"  {r['year']:>6} {r['symbol']:>4} {r['mode']:>7} {r['trades']:>5} "
                      f"{r['win_rate']:>6.1f} {r['avg_trade']:>8.2f} {r['total_pnl']:>10.2f} "
                      f"{r['profit_factor']:>6.2f} {r['max_dd']:>9.2f} {r['sharpe']:>6.2f} "
                      f"{r['days_traded']:>6}")
        print(f"  {'-'*120}")

    # ── Annual combined summary ──
    print(f"\n  ANNUAL COMBINED (MNQ + MES)")
    print(f"  {'Year':>6} {'BEFORE PnL':>12} {'AFTER PnL':>12} {'Delta':>10} {'BEFORE Tr':>10} {'AFTER Tr':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    before_total = 0
    after_total = 0
    before_trades_total = 0
    after_trades_total = 0

    for year in YEARS:
        bp = sum(all_json[f"BEFORE_{s}_{year}"]["total_pnl"] for s in SYMBOLS)
        ap = sum(all_json[f"AFTER_{s}_{year}"]["total_pnl"] for s in SYMBOLS)
        bt = sum(all_json[f"BEFORE_{s}_{year}"]["trades"] for s in SYMBOLS)
        at = sum(all_json[f"AFTER_{s}_{year}"]["trades"] for s in SYMBOLS)
        delta = ap - bp
        before_total += bp
        after_total += ap
        before_trades_total += bt
        after_trades_total += at
        marker = " <--" if abs(delta) > 200 else ""
        print(f"  {year:>6} {bp:>12.2f} {ap:>12.2f} {delta:>+10.2f} {bt:>10} {at:>10}{marker}")

    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':>6} {before_total:>12.2f} {after_total:>12.2f} "
          f"{after_total - before_total:>+10.2f} {before_trades_total:>10} {after_trades_total:>10}")

    # ── MNQ-only summary ──
    print(f"\n  MNQ ONLY")
    print(f"  {'Year':>6} {'BEFORE PnL':>12} {'AFTER PnL':>12} {'Delta':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
    mnq_b = 0
    mnq_a = 0
    for year in YEARS:
        bp = all_json[f"BEFORE_MNQ_{year}"]["total_pnl"]
        ap = all_json[f"AFTER_MNQ_{year}"]["total_pnl"]
        mnq_b += bp
        mnq_a += ap
        print(f"  {year:>6} {bp:>12.2f} {ap:>12.2f} {ap - bp:>+10.2f}")
    print(f"  {'TOTAL':>6} {mnq_b:>12.2f} {mnq_a:>12.2f} {mnq_a - mnq_b:>+10.2f}")

    # ── Save comparison CSV ──
    csv_path = OUTPUT_DIR / "before_vs_after_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Saved: {csv_path}")

    # ── Save comparison JSON ──
    json_path = OUTPUT_DIR / "before_vs_after_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_json, f, indent=2)
    print(f"  Saved: {json_path}")

    # ── Console log ──
    log_path = OUTPUT_DIR / "validation_console.log"
    import sys
    # Already printed to stdout; save a simple summary
    with open(log_path, "w") as f:
        f.write(f"Time filter validation run\n")
        f.write(f"Change: MNQ max_entry_time 12:00 -> 10:30 (soft filter)\n")
        f.write(f"MES: unchanged (max_entry_time stays 12:00)\n\n")
        f.write(f"BEFORE total: {before_trades_total} trades, ${before_total:.2f}\n")
        f.write(f"AFTER  total: {after_trades_total} trades, ${after_total:.2f}\n")
        f.write(f"Delta: ${after_total - before_total:+.2f}\n")
    print(f"  Saved: {log_path}")
