import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
#!/usr/bin/env python3
"""
AdaptiveRegimeStrategy Historical Validation Suite
====================================================

Runs the AdaptiveRegimeStrategy (unchanged) across all available yearly
datasets from 2017 through 2025 for MES and MNQ. Saves per-year logs,
trade CSVs, summary JSONs, and a combined comparison table.

Usage:
    python run_adaptive_regime_validation.py

No flags needed — uses default AdaptiveRegimeConfig and discovers data
files automatically. No ML, no optimization, no parameter changes.
"""

import csv
import datetime
import io
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from config.settings import (
    AdaptiveRegimeConfig,
    BacktestConfig,
    INSTRUMENT_REGISTRY,
)
from data.loader import load_bars
from backtest.engine import BacktestEngine
from backtest.metrics import (
    compute_metrics,
    export_trades_csv,
    export_metrics_json,
)
from strategy.adaptive_regime import AdaptiveRegimeStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOLS = ["MES", "MNQ"]

# Map years to data file paths.
# Individual yearly files where available; 4y files filtered for 2021-2024.
YEAR_DATA = {}
for sym in SYMBOLS:
    prefix = sym.lower()
    for year in [2017, 2018, 2019, 2022, 2025]:
        fpath = Path(f"data/{prefix}_{year}.csv")
        if fpath.exists():
            YEAR_DATA[(year, sym)] = {"path": str(fpath), "filter": False}

    # 4y files cover 2021-01-03 to 2024-12-30
    fouryr = Path(f"data/{prefix}_4y.csv")
    if fouryr.exists():
        for year in [2021, 2023, 2024]:
            YEAR_DATA[(year, sym)] = {"path": str(fouryr), "filter": True, "year": year}
        # 2022 from 4y only if individual file missing
        if (2022, sym) not in YEAR_DATA:
            YEAR_DATA[(2022, sym)] = {"path": str(fouryr), "filter": True, "year": 2022}

OUTPUT_DIR = Path("results/adaptive_regime_refined_validation")

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=25_000.0,
    slippage_ticks=1,
    commission_per_side=2.25,
)

# Market character annotations for interpretation
MARKET_CHARACTER = {
    2017: "Low vol, grind higher, difficult breakout year",
    2018: "Volatile, regime-shift year (Feb vol-spike, Q4 selloff)",
    2019: "Trending bull year after Q4-2018 selloff",
    2020: "Extreme volatility — COVID crash + rebound (DATA MISSING)",
    2021: "Strong trend, expansion, low-vol breakout-friendly",
    2022: "Bear market, volatile, frequent reversals",
    2023: "Mixed — choppy early, stronger trend later",
    2024: "Mixed — broadening, rotational",
    2025: "Strong trend, recent out-of-sample year",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_year_data(year: int, symbol: str) -> pd.DataFrame:
    """Load bar data for a specific year and symbol."""
    key = (year, symbol)
    if key not in YEAR_DATA:
        raise FileNotFoundError(f"No data available for {symbol} {year}")

    info = YEAR_DATA[key]
    bars = load_bars(info["path"])

    if info.get("filter"):
        # Filter to the requested year from the 4y file
        y = info["year"]
        start = pd.Timestamp(f"{y}-01-01", tz="America/New_York")
        end = pd.Timestamp(f"{y}-12-31 23:59:59", tz="America/New_York")
        bars = bars[(bars.index >= start) & (bars.index <= end)]
        if bars.empty:
            raise ValueError(f"No bars found for {symbol} {year} after filtering")

    return bars


def run_single_backtest(bars: pd.DataFrame, symbol: str) -> dict:
    """Run one backtest and return metrics + trades + result."""
    instrument = INSTRUMENT_REGISTRY[symbol]
    config = AdaptiveRegimeConfig.for_symbol(symbol)  # Instrument-aware defaults
    strategy = AdaptiveRegimeStrategy(config)

    engine = BacktestEngine(
        instrument=instrument,
        strategy_config=config,
        backtest_config=BACKTEST_CONFIG,
        strategy=strategy,
    )

    result = engine.run(bars)

    metrics = compute_metrics(result, BACKTEST_CONFIG)
    # peak_equity not in compute_metrics — add from result
    metrics["peak_equity"] = round(result.peak_equity, 2)
    diagnostics = strategy.diagnostics
    selectivity = dict(strategy.selectivity)

    return {
        "result": result,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "selectivity": selectivity,
        "config": asdict(config),
    }


def format_table(headers: list[str], rows: list[list], align: list[str] = None) -> str:
    """Simple text table formatter."""
    if not rows:
        return "(no data)"
    col_widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        sr = [str(v) for v in row]
        str_rows.append(sr)
        for i, s in enumerate(sr):
            col_widths[i] = max(col_widths[i], len(s))

    if align is None:
        align = ["<"] * len(headers)

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr = "| " + " | ".join(
        f"{h:{a}{w}}" for h, w, a in zip(headers, col_widths, align)
    ) + " |"

    lines = [sep, hdr, sep]
    for sr in str_rows:
        line = "| " + " | ".join(
            f"{s:{a}{w}}" for s, w, a in zip(sr, col_widths, align)
        ) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def regime_summary(diagnostics: list) -> dict:
    """Summarize regime distribution from strategy diagnostics."""
    counts = {}
    for d in diagnostics:
        r = d.regime if hasattr(d, "regime") else "UNKNOWN"
        counts[r] = counts.get(r, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("validation")
    logger.setLevel(logging.INFO)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", OUTPUT_DIR)

    # Discover available years
    available_years = sorted(set(y for (y, _) in YEAR_DATA.keys()))
    logger.info("Available years: %s", available_years)
    logger.info("Symbols: %s", SYMBOLS)

    all_results = {}  # (year, symbol) -> metrics dict
    comparison_rows = []

    for year in available_years:
        year_log_lines = []
        year_log_lines.append(f"=" * 70)
        year_log_lines.append(f"  YEAR {year}  —  {MARKET_CHARACTER.get(year, '')}")
        year_log_lines.append(f"=" * 70)

        year_summary = {"year": year, "symbol_results": {}}

        for symbol in SYMBOLS:
            key = (year, symbol)
            if key not in YEAR_DATA:
                msg = f"  {symbol} {year}: NO DATA AVAILABLE — skipped"
                year_log_lines.append(msg)
                logger.warning(msg)
                continue

            logger.info("Running %s %d ...", symbol, year)
            year_log_lines.append(f"\n--- {symbol} {year} ---")

            try:
                bars = load_year_data(year, symbol)
                year_log_lines.append(f"  Bars loaded: {len(bars)}")
                year_log_lines.append(f"  Date range: {bars.index.min()} to {bars.index.max()}")

                run = run_single_backtest(bars, symbol)
                m = run["metrics"]
                result = run["result"]
                diagnostics = run["diagnostics"]
                sel = run["selectivity"]

                # Store
                all_results[key] = m
                year_summary["symbol_results"][symbol] = m
                year_summary.setdefault("selectivity", {})[symbol] = sel

                # Print key metrics
                year_log_lines.append(f"  Trades:        {m['total_trades']}")
                year_log_lines.append(f"  Win Rate:      {m['win_rate_pct']:.1f}%")
                year_log_lines.append(f"  Avg Trade:     ${m['expectancy']:.2f}")
                year_log_lines.append(f"  Total PnL:     ${m['total_pnl_dollars']:.2f}")
                year_log_lines.append(f"  Profit Factor: {m['profit_factor']:.2f}")
                year_log_lines.append(f"  Max Drawdown:  ${m['max_drawdown_dollars']:.2f}")
                year_log_lines.append(f"  Final Equity:  ${m['final_equity']:.2f}")
                year_log_lines.append(f"  Peak Equity:   ${m['peak_equity']:.2f}")
                year_log_lines.append(f"  Sharpe:        {m['sharpe_ratio']:.2f}")

                # Direction breakdown
                year_log_lines.append(f"  Long trades:   {m.get('long_trades', '?')}, "
                                      f"win {m.get('long_win_rate_pct', '?')}%")
                year_log_lines.append(f"  Short trades:  {m.get('short_trades', '?')}, "
                                      f"win {m.get('short_win_rate_pct', '?')}%")

                # Regime distribution
                reg_counts = regime_summary(diagnostics)
                year_log_lines.append(f"  Regimes: {reg_counts}")

                # Selectivity diagnostics
                year_log_lines.append(f"  Selectivity:")
                year_log_lines.append(f"    Days with range: {sel['days_with_range']}")
                year_log_lines.append(f"    Days traded:     {sel['days_traded']}")
                year_log_lines.append(f"    Entry attempts:  {sel['entry_attempts']}")
                year_log_lines.append(f"    Regime blocks:   {sel['regime_blocks']}")
                year_log_lines.append(f"    Filter blocks:   {sel['filter_blocks']}")
                year_log_lines.append(f"    Entries taken:   {sel['entries_taken']}")
                if sel['days_with_range'] > 0:
                    trade_rate = sel['days_traded'] / sel['days_with_range'] * 100
                    year_log_lines.append(f"    Trade rate:      {trade_rate:.1f}% of range-days")

                # Comparison row
                comparison_rows.append([
                    year, symbol,
                    m["total_trades"],
                    f"{m['win_rate_pct']:.1f}%",
                    f"${m['expectancy']:.2f}",
                    f"${m['total_pnl_dollars']:.2f}",
                    f"{m['profit_factor']:.2f}",
                    f"${m['max_drawdown_dollars']:.2f}",
                    f"${m['final_equity']:.2f}",
                    f"{m['sharpe_ratio']:.2f}",
                ])

                # Save trade CSV
                trades_path = OUTPUT_DIR / f"{year}_{symbol.lower()}_trades.csv"
                export_trades_csv(result.trades, str(trades_path))
                year_log_lines.append(f"  Trades saved: {trades_path}")

            except Exception as e:
                msg = f"  ERROR: {symbol} {year}: {e}"
                year_log_lines.append(msg)
                logger.error(msg, exc_info=True)

        # Save year console log
        log_text = "\n".join(year_log_lines)
        print(log_text)
        log_path = OUTPUT_DIR / f"{year}_console.log"
        log_path.write_text(log_text, encoding="utf-8")

        # Save year summary JSON
        summary_path = OUTPUT_DIR / f"{year}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(year_summary, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Combined comparison table
    # ------------------------------------------------------------------
    headers = ["Year", "Symbol", "Trades", "Win Rate", "Avg Trade",
               "Total PnL", "Profit Factor", "Max DD", "Final Equity", "Sharpe"]
    align = ["<", "<", ">", ">", ">", ">", ">", ">", ">", ">"]

    table_text = format_table(headers, comparison_rows, align)
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE — AdaptiveRegimeStrategy 2017-2025")
    print("=" * 70)
    print(table_text)

    # Annual combined summary
    annual_rows = []
    for year in available_years:
        year_trades = 0
        year_pnl = 0.0
        best_sym = None
        best_pnl = -float("inf")
        worst_sym = None
        worst_pnl = float("inf")

        for sym in SYMBOLS:
            m = all_results.get((year, sym))
            if m:
                year_trades += m["total_trades"]
                year_pnl += m["total_pnl_dollars"]
                if m["total_pnl_dollars"] > best_pnl:
                    best_pnl = m["total_pnl_dollars"]
                    best_sym = sym
                if m["total_pnl_dollars"] < worst_pnl:
                    worst_pnl = m["total_pnl_dollars"]
                    worst_sym = sym

        annual_rows.append([
            year,
            year_trades,
            f"${year_pnl:.2f}",
            best_sym or "N/A",
            worst_sym or "N/A",
            MARKET_CHARACTER.get(year, ""),
        ])

    ann_headers = ["Year", "Total Trades", "Total PnL", "Best Symbol",
                   "Worst Symbol", "Market Character"]
    ann_align = ["<", ">", ">", "<", "<", "<"]
    ann_table = format_table(ann_headers, annual_rows, ann_align)

    print("\n" + "=" * 70)
    print("  ANNUAL COMBINED SUMMARY")
    print("=" * 70)
    print(ann_table)

    # ------------------------------------------------------------------
    # Save comparison files
    # ------------------------------------------------------------------

    # comparison.csv
    comp_csv_path = OUTPUT_DIR / "comparison.csv"
    with open(comp_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(comparison_rows)

    # comparison.json
    comp_json_path = OUTPUT_DIR / "comparison.json"
    comp_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "AdaptiveRegimeStrategy",
        "config": asdict(AdaptiveRegimeConfig()),
        "backtest_config": asdict(BACKTEST_CONFIG),
        "per_symbol": {},
        "annual_combined": [],
    }
    for (year, sym), m in sorted(all_results.items()):
        yr_key = str(year)
        if yr_key not in comp_data["per_symbol"]:
            comp_data["per_symbol"][yr_key] = {}
        comp_data["per_symbol"][yr_key][sym] = m

    for row in annual_rows:
        comp_data["annual_combined"].append({
            "year": row[0],
            "total_trades": row[1],
            "total_pnl": row[2],
            "best_symbol": row[3],
            "worst_symbol": row[4],
            "market_character": row[5],
        })

    with open(comp_json_path, "w") as f:
        json.dump(comp_data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Summary markdown
    # ------------------------------------------------------------------
    md_lines = [
        "# AdaptiveRegimeStrategy Historical Validation (2017-2025)",
        "",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "",
        "## Strategy Config",
        "Pure defaults — no optimization, no ML, no parameter changes.",
        "",
        "## Execution Model",
        "- Stop-order entry at opening range breakout level",
        "- Entry price = OR high/low + buffer (pre-determined before bar)",
        "- All features (EMA, ATR, volume) use standard rolling windows",
        "- One trade per day per symbol",
        "- No ML model involved",
        "",
        "## Data Coverage",
        f"- Years tested: {available_years}",
        "- 2020: NOT AVAILABLE (no data file)",
        "- 2021/2023/2024: extracted from 4-year aggregate files",
        "- Symbols: MES, MNQ",
        "",
        "## Per-Symbol Results",
        "",
        "```",
        table_text,
        "```",
        "",
        "## Annual Combined Summary",
        "",
        "```",
        ann_table,
        "```",
        "",
        "## Regime-Aware Interpretation",
        "",
    ]

    # Compute totals for final classification
    total_pnl_all = 0.0
    total_trades_all = 0
    profitable_years = 0
    worst_dd = 0.0
    yearly_pnls = {}

    for year in available_years:
        year_pnl = 0.0
        year_trades = 0
        year_dd = 0.0
        year_interp_lines = []
        char = MARKET_CHARACTER.get(year, "Unknown")
        md_lines.append(f"### {year} — {char}")

        for sym in SYMBOLS:
            m = all_results.get((year, sym))
            if m:
                year_pnl += m["total_pnl_dollars"]
                year_trades += m["total_trades"]
                year_dd = max(year_dd, abs(m["max_drawdown_dollars"]))

        yearly_pnls[year] = year_pnl
        total_pnl_all += year_pnl
        total_trades_all += year_trades
        if year_pnl > 0:
            profitable_years += 1
        worst_dd = max(worst_dd, year_dd)

        # Auto-interpret
        if year_pnl > 500:
            md_lines.append(f"- **Profitable** (${year_pnl:.0f}). Strategy appears well-suited to this regime.")
        elif year_pnl > 0:
            md_lines.append(f"- **Marginally profitable** (${year_pnl:.0f}). Limited edge.")
        elif year_pnl > -500:
            md_lines.append(f"- **Marginally negative** (${year_pnl:.0f}). Slight drag, not catastrophic.")
        else:
            md_lines.append(f"- **Losing year** (${year_pnl:.0f}). Strategy struggled in this regime.")

        if year_trades == 0:
            md_lines.append(f"- Zero trades — strategy was too restrictive or data issue.")
        elif year_trades < 50:
            md_lines.append(f"- Low trade count ({year_trades}). May indicate overly restrictive filters.")
        elif year_trades > 300:
            md_lines.append(f"- High trade count ({year_trades}). May indicate overtrading.")

        md_lines.append("")

    # ------------------------------------------------------------------
    # Final classification
    # ------------------------------------------------------------------
    n_years = len(available_years)
    pct_profitable = profitable_years / n_years * 100 if n_years > 0 else 0
    avg_annual_pnl = total_pnl_all / n_years if n_years > 0 else 0

    if (pct_profitable >= 75 and total_pnl_all > 0 and worst_dd < 5000
            and avg_annual_pnl > 0):
        classification = "ROBUST"
        class_reason = (
            f"{profitable_years}/{n_years} years profitable, "
            f"cumulative ${total_pnl_all:.0f}, "
            f"worst DD ${worst_dd:.0f}"
        )
    elif (pct_profitable >= 50 and total_pnl_all > -2000):
        classification = "ACCEPTABLE"
        class_reason = (
            f"{profitable_years}/{n_years} years profitable, "
            f"cumulative ${total_pnl_all:.0f}, "
            f"worst DD ${worst_dd:.0f}"
        )
    else:
        classification = "FRAGILE"
        class_reason = (
            f"Only {profitable_years}/{n_years} years profitable, "
            f"cumulative ${total_pnl_all:.0f}, "
            f"worst DD ${worst_dd:.0f}"
        )

    md_lines.extend([
        "## Final Classification",
        "",
        f"**{classification}**",
        "",
        f"Basis: {class_reason}",
        "",
        "### Criteria Used",
        f"- Profitable years: {profitable_years}/{n_years} ({pct_profitable:.0f}%)",
        f"- Cumulative PnL: ${total_pnl_all:.2f}",
        f"- Total trades: {total_trades_all}",
        f"- Worst single-year drawdown: ${worst_dd:.2f}",
        f"- Average annual PnL: ${avg_annual_pnl:.2f}",
        "",
        "### Classification Scale",
        "- ROBUST: ≥75% profitable years, positive cumulative, worst DD < $5000",
        "- ACCEPTABLE: ≥50% profitable years, cumulative > -$2000",
        "- FRAGILE: Below acceptable thresholds",
        "",
        "## Caveats",
        "- 2020 data is missing — a critical volatility year is not tested",
        "- 2021/2023/2024 extracted from 4-year aggregate files (same data, filtered by date)",
        "- MNQ has $2/point (vs MES $5/point) — PnL magnitudes differ",
        "- No transaction cost optimization — uses $2.25/side + 1 tick slippage",
        "- Features include current bar data in EMA/ATR (standard practice, ~2% weight)",
    ])

    summary_md_path = OUTPUT_DIR / "summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Print final classification
    print("\n" + "=" * 70)
    print("  FINAL CLASSIFICATION")
    print("=" * 70)
    print(f"  {classification}")
    print(f"  {class_reason}")
    print()
    print(f"  Profitable years: {profitable_years}/{n_years} ({pct_profitable:.0f}%)")
    print(f"  Cumulative PnL:   ${total_pnl_all:.2f}")
    print(f"  Total trades:     {total_trades_all}")
    print(f"  Worst DD:         ${worst_dd:.2f}")
    print(f"  Avg annual PnL:   ${avg_annual_pnl:.2f}")
    print()
    print(f"  Results saved to: {OUTPUT_DIR}/")
    print(f"  Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
