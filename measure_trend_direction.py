"""Measure impact of trend_direction_only on MNQ across all years.

Compares the new setting (trend_direction_only=True, the new default)
against the old behavior (trend_direction_only=False) to quantify
the trade reduction and PnL change per year.
"""
import json
import os
from pathlib import Path
import pandas as pd
from data.loader import load_bars
from config.settings import AdaptiveRegimeConfig, BacktestConfig, INSTRUMENT_REGISTRY
from strategy.adaptive_regime import AdaptiveRegimeStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics

OUTPUT_DIR = Path("results/adaptive_regime_trend_direction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_CONFIG = BacktestConfig(
    initial_capital=25_000.0,
    slippage_ticks=1,
    commission_per_side=2.25,
)


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
    bars = bars[(bars.index >= s) & (bars.index <= e)]
    if len(bars) == 0:
        raise FileNotFoundError(f"No bars for {symbol} {year}")
    return bars


def run_year(symbol, year, trend_direction_only):
    bars = load_year_data(symbol, year)
    instrument = INSTRUMENT_REGISTRY[symbol]
    cfg = AdaptiveRegimeConfig.for_symbol(symbol)
    cfg.trend_direction_only = trend_direction_only

    strat = AdaptiveRegimeStrategy(cfg)
    engine = BacktestEngine(
        instrument=instrument,
        strategy_config=cfg,
        backtest_config=BACKTEST_CONFIG,
        strategy=strat,
    )
    result = engine.run(bars)
    metrics = compute_metrics(result, BACKTEST_CONFIG)

    # Regime breakdown
    regimes = {}
    trend_dirs = {"long": 0, "short": 0}
    for d in strat.diagnostics:
        regimes[d.regime] = regimes.get(d.regime, 0) + 1
        if d.regime == "TREND" and d.trade_taken:
            trend_dirs[d.trade_direction] = trend_dirs.get(d.trade_direction, 0) + 1

    sel = strat.selectivity
    total = metrics.get("total_trades", 0)
    pnl = metrics.get("total_pnl_dollars", 0)
    wr = metrics.get("win_rate_pct", 0)
    pf = metrics.get("profit_factor", 0)

    return {
        "year": year,
        "symbol": symbol,
        "trend_direction_only": trend_direction_only,
        "trades": total,
        "pnl": round(pnl, 2),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "regimes": regimes,
        "trend_trade_dirs": trend_dirs,
        "days_with_range": sel["days_with_range"],
        "days_traded": sel["days_traded"],
        "entry_attempts": sel["entry_attempts"],
        "regime_blocks": sel["regime_blocks"],
        "filter_blocks": sel["filter_blocks"],
    }


if __name__ == "__main__":
    years = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
    all_results = []

    print("=" * 110)
    print(f"{'Year':>6} {'Mode':>8} {'Trades':>7} {'PnL':>10} {'WR%':>6} {'PF':>6} "
          f"{'TREND':>6} {'BRK':>5} {'RANGE':>6} {'DEAD':>5} {'Days':>6} "
          f"{'TrendL':>7} {'TrendS':>7}")
    print("=" * 110)

    for year in years:
        for tdo in [False, True]:
            r = run_year("MNQ", year, tdo)
            all_results.append(r)

            mode = "NEW" if tdo else "OLD"
            reg = r["regimes"]
            td = r["trend_trade_dirs"]
            print(f"{year:>6} {mode:>8} {r['trades']:>7} {r['pnl']:>10.2f} {r['win_rate']:>6.1f} {r['profit_factor']:>6.2f} "
                  f"{reg.get('TREND', 0):>6} {reg.get('BREAKOUT', 0):>5} {reg.get('RANGE', 0):>6} "
                  f"{reg.get('DEAD', 0):>5} {r['days_traded']:>6} "
                  f"{td.get('long', 0):>7} {td.get('short', 0):>7}")
        print("-" * 110)

    # Summary: cumulative PnL comparison
    old_pnl = sum(r["pnl"] for r in all_results if not r["trend_direction_only"])
    new_pnl = sum(r["pnl"] for r in all_results if r["trend_direction_only"])
    old_trades = sum(r["trades"] for r in all_results if not r["trend_direction_only"])
    new_trades = sum(r["trades"] for r in all_results if r["trend_direction_only"])

    print(f"\nCumulative OLD: {old_trades} trades, ${old_pnl:.2f}")
    print(f"Cumulative NEW: {new_trades} trades, ${new_pnl:.2f}")
    print(f"Trade reduction: {old_trades - new_trades} ({(old_trades - new_trades) / old_trades * 100:.1f}%)")
    print(f"PnL delta: ${new_pnl - old_pnl:+.2f}")

    # Save results
    with open(OUTPUT_DIR / "trend_direction_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/trend_direction_comparison.json")

    # Also run MES for comparison
    print("\n" + "=" * 110)
    print("MES comparison (minimal TREND days expected)")
    print("=" * 110)
    for year in years:
        for tdo in [False, True]:
            r = run_year("MES", year, tdo)
            mode = "NEW" if tdo else "OLD"
            reg = r["regimes"]
            print(f"{year:>6} {mode:>8} {r['trades']:>7} {r['pnl']:>10.2f} {r['win_rate']:>6.1f} {r['profit_factor']:>6.2f} "
                  f"TREND={reg.get('TREND', 0):>3} BRK={reg.get('BREAKOUT', 0):>3}")
        print("-" * 110)
