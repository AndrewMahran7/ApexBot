"""Detailed trade-level comparison: baseline vs MNQ max_entry=10:30.

Shows exactly which trades change, when entries shift, and where PnL
is gained/lost.
"""
from collections import defaultdict
import pandas as pd
from data.loader import load_bars
from config.settings import AdaptiveRegimeConfig, BacktestConfig, INSTRUMENT_REGISTRY
from strategy.adaptive_regime import AdaptiveRegimeStrategy
from backtest.engine import BacktestEngine


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


def run_bt(symbol, year, max_entry):
    bars = load_year_data(symbol, year)
    instrument = INSTRUMENT_REGISTRY[symbol]
    cfg = AdaptiveRegimeConfig.for_symbol(symbol)
    cfg.max_entry_time = max_entry
    strat = AdaptiveRegimeStrategy(cfg)
    engine = BacktestEngine(instrument=instrument, strategy_config=cfg,
                            backtest_config=BACKTEST_CONFIG, strategy=strat)
    result = engine.run(bars)
    return result.trades


if __name__ == "__main__":
    years = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

    for year in years:
        old_trades = run_bt("MNQ", year, "12:00")
        new_trades = run_bt("MNQ", year, "10:30")

        # Index by date
        old_by_date = {t.entry_time.date(): t for t in old_trades}
        new_by_date = {t.entry_time.date(): t for t in new_trades}

        all_dates = sorted(set(old_by_date.keys()) | set(new_by_date.keys()))

        same = 0
        changed_better = 0
        changed_worse = 0
        dropped = 0
        added = 0
        pnl_delta = 0.0
        changed_details = []

        for d in all_dates:
            old_t = old_by_date.get(d)
            new_t = new_by_date.get(d)

            if old_t and new_t:
                if old_t.entry_time == new_t.entry_time:
                    same += 1
                else:
                    delta = new_t.net_pnl - old_t.net_pnl
                    pnl_delta += delta
                    if delta > 0:
                        changed_better += 1
                    else:
                        changed_worse += 1
                    changed_details.append({
                        "date": str(d),
                        "old_time": old_t.entry_time.strftime("%H:%M"),
                        "new_time": new_t.entry_time.strftime("%H:%M"),
                        "old_pnl": round(old_t.net_pnl, 2),
                        "new_pnl": round(new_t.net_pnl, 2),
                        "delta": round(delta, 2),
                        "old_dir": old_t.direction,
                        "new_dir": new_t.direction,
                    })
            elif old_t and not new_t:
                dropped += 1
                pnl_delta -= old_t.net_pnl
            elif new_t and not old_t:
                added += 1
                pnl_delta += new_t.net_pnl

        print(f"=== MNQ {year}: {len(old_trades)} -> {len(new_trades)} trades ===")
        print(f"  Same entry: {same}")
        print(f"  Changed (better): {changed_better}")
        print(f"  Changed (worse):  {changed_worse}")
        print(f"  Dropped: {dropped}")
        print(f"  Added:   {added}")
        print(f"  PnL delta from changes: ${pnl_delta:.2f}")

        if changed_details:
            print(f"  Changed trades:")
            for cd in changed_details[:10]:
                print(f"    {cd['date']} {cd['old_time']}({cd['old_dir'][0]})->{cd['new_time']}({cd['new_dir'][0]}) "
                      f"pnl: ${cd['old_pnl']:.0f} -> ${cd['new_pnl']:.0f} (${cd['delta']:+.0f})")
        print()
