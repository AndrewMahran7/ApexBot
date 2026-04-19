import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
"""Test impact of raising min_breakout_strength on MNQ trade quality.

Hypothesis: In choppy years (2022, 2024), breakouts are marginal
(barely exceed buffer). Higher strength threshold filters fakeouts
without hurting strong-trend years where breakouts are decisive.
"""
import json
from pathlib import Path
import pandas as pd
from data.loader import load_bars
from config.settings import AdaptiveRegimeConfig, BacktestConfig, INSTRUMENT_REGISTRY
from strategy.adaptive_regime import AdaptiveRegimeStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics


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
    return bars[(bars.index >= s) & (bars.index <= e)]


def run_year(symbol, year, breakout_strength):
    bars = load_year_data(symbol, year)
    instrument = INSTRUMENT_REGISTRY[symbol]
    cfg = AdaptiveRegimeConfig.for_symbol(symbol)
    cfg.min_breakout_strength = breakout_strength
    cfg.trend_direction_only = False

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
    return {
        "trades": metrics.get("total_trades", 0),
        "pnl": round(metrics.get("total_pnl_dollars", 0), 2),
        "wr": round(metrics.get("win_rate_pct", 0), 1),
        "pf": round(metrics.get("profit_factor", 0), 2),
        "days_traded": sel["days_traded"],
        "filter_blocks": sel["filter_blocks"],
    }


if __name__ == "__main__":
    years = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
    strengths = [1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"{'':>6}", end="")
    for s in strengths:
        print(f" |  strength={s:.1f}         ", end="")
    print()

    print(f"{'Year':>6}", end="")
    for _ in strengths:
        print(f" | {'Trds':>5} {'PnL':>9} {'PF':>5}", end="")
    print()
    print("=" * (6 + len(strengths) * 24))

    cumulative = {s: 0.0 for s in strengths}
    cumulative_trades = {s: 0 for s in strengths}

    for year in years:
        print(f"{year:>6}", end="")
        for s in strengths:
            r = run_year("MNQ", year, s)
            cumulative[s] += r["pnl"]
            cumulative_trades[s] += r["trades"]
            print(f" | {r['trades']:>5} {r['pnl']:>9.2f} {r['pf']:>5.2f}", end="")
        print()

    print("=" * (6 + len(strengths) * 24))
    print(f"{'TOTAL':>6}", end="")
    for s in strengths:
        print(f" | {cumulative_trades[s]:>5} {cumulative[s]:>9.2f} {'':>5}", end="")
    print()
