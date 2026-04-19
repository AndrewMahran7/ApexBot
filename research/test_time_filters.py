import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
"""
Test candidate time-of-day filters for AdaptiveRegimeStrategy.

Candidates:
  A) MNQ max_entry=10:30, MES max_entry=12:00 (current)
  B) MNQ max_entry=11:00, MES max_entry=12:00
  C) Both max_entry=11:00
  D) Both max_entry=10:30

Compare cumulative PnL, per-year impact, trade reduction.
"""
import json
from pathlib import Path
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


def run_backtest(symbol, year, max_entry_time):
    bars = load_year_data(symbol, year)
    instrument = INSTRUMENT_REGISTRY[symbol]
    cfg = AdaptiveRegimeConfig.for_symbol(symbol)
    cfg.max_entry_time = max_entry_time

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
        "max_dd": round(metrics.get("max_drawdown_dollars", 0), 2),
        "sharpe": round(metrics.get("sharpe_ratio", 0), 2),
        "days_traded": sel["days_traded"],
    }


if __name__ == "__main__":
    years = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

    # Candidates: (label, MNQ_max_entry, MES_max_entry)
    candidates = [
        ("BASELINE", "12:00", "12:00"),
        ("A_MNQ1030", "10:30", "12:00"),
        ("B_MNQ1100", "11:00", "12:00"),
        ("C_BOTH1100", "11:00", "11:00"),
        ("D_BOTH1030", "10:30", "10:30"),
    ]

    # Collect all results
    all_results = {}

    for label, mnq_time, mes_time in candidates:
        print(f"\n{'='*100}")
        print(f"  CANDIDATE: {label} (MNQ max_entry={mnq_time}, MES max_entry={mes_time})")
        print(f"{'='*100}")
        print(f"  {'Year':>6} {'Sym':>4} {'Tr':>5} {'PnL':>10} {'WR%':>6} {'PF':>6} {'DD':>9} {'Sh':>6}")
        print(f"  {'-'*6} {'-'*4} {'-'*5} {'-'*10} {'-'*6} {'-'*6} {'-'*9} {'-'*6}")

        cum_pnl = {"MNQ": 0, "MES": 0}
        cum_trades = {"MNQ": 0, "MES": 0}

        for year in years:
            for symbol in ["MNQ", "MES"]:
                max_entry = mnq_time if symbol == "MNQ" else mes_time
                r = run_backtest(symbol, year, max_entry)
                cum_pnl[symbol] += r["pnl"]
                cum_trades[symbol] += r["trades"]

                key = f"{label}_{symbol}_{year}"
                all_results[key] = r

                print(f"  {year:>6} {symbol:>4} {r['trades']:>5} {r['pnl']:>10.2f} "
                      f"{r['wr']:>6.1f} {r['pf']:>6.2f} {r['max_dd']:>9.2f} {r['sharpe']:>6.2f}")

        print(f"  {'-'*6} {'-'*4} {'-'*5} {'-'*10} {'-'*6} {'-'*6} {'-'*9} {'-'*6}")
        total = cum_pnl["MNQ"] + cum_pnl["MES"]
        total_tr = cum_trades["MNQ"] + cum_trades["MES"]
        print(f"  {'TOTAL':>6} {'MNQ':>4} {cum_trades['MNQ']:>5} {cum_pnl['MNQ']:>10.2f}")
        print(f"  {'':>6} {'MES':>4} {cum_trades['MES']:>5} {cum_pnl['MES']:>10.2f}")
        print(f"  {'':>6} {'ALL':>4} {total_tr:>5} {total:>10.2f}")

    # Summary comparison
    print(f"\n\n{'='*100}")
    print(f"  CANDIDATE COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Candidate':>14} {'MNQ Tr':>7} {'MNQ PnL':>10} {'MES Tr':>7} {'MES PnL':>10} {'Total Tr':>8} {'Total PnL':>10}")
    print(f"  {'-'*14} {'-'*7} {'-'*10} {'-'*7} {'-'*10} {'-'*8} {'-'*10}")

    for label, mnq_time, mes_time in candidates:
        mnq_pnl = sum(all_results[f"{label}_MNQ_{y}"]["pnl"] for y in years)
        mes_pnl = sum(all_results[f"{label}_MES_{y}"]["pnl"] for y in years)
        mnq_tr = sum(all_results[f"{label}_MNQ_{y}"]["trades"] for y in years)
        mes_tr = sum(all_results[f"{label}_MES_{y}"]["trades"] for y in years)
        print(f"  {label:>14} {mnq_tr:>7} {mnq_pnl:>10.2f} {mes_tr:>7} {mes_pnl:>10.2f} "
              f"{mnq_tr + mes_tr:>8} {mnq_pnl + mes_pnl:>10.2f}")

    # Per-year impact for key years
    print(f"\n  KEY YEAR IMPACT (combined MNQ+MES PnL)")
    print(f"  {'Candidate':>14}", end="")
    for y in [2018, 2021, 2024, 2025]:
        print(f"  {y:>8}", end="")
    print()
    for label, _, _ in candidates:
        print(f"  {label:>14}", end="")
        for y in [2018, 2021, 2024, 2025]:
            total = all_results[f"{label}_MNQ_{y}"]["pnl"] + all_results[f"{label}_MES_{y}"]["pnl"]
            print(f"  {total:>8.0f}", end="")
        print()

    # Save
    with open(OUTPUT_DIR / "time_filter_candidates.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'time_filter_candidates.json'}")
