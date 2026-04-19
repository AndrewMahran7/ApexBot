"""Diagnose MNQ TREND overclassification by measuring EMA slope distributions."""
import pandas as pd
import statistics
from data.loader import load_bars
from config.settings import AdaptiveRegimeConfig
from strategy.adaptive_regime import AdaptiveRegimeStrategy


def run_year(year):
    files = {
        2017: "data/mnq_2017.csv",
        2018: "data/mnq_2018.csv",
        2019: "data/mnq_2019.csv",
        2022: "data/mnq_2022.csv",
        2025: "data/mnq_2025.csv",
    }
    if year in files:
        bars = load_bars(files[year])
    else:
        bars = load_bars("data/mnq_4y.csv")
        import pytz
        s = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="America/New_York")
        bars = bars[(bars.index >= s) & (bars.index <= e)]

    cfg = AdaptiveRegimeConfig.for_symbol("MNQ")
    strat = AdaptiveRegimeStrategy(cfg)
    for ts, row in bars.iterrows():
        bar = {
            "timestamp": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        strat.on_bar(bar)
    strat._finalize_day_diagnostic()

    regimes = {}
    slopes_trend = []
    slopes_all = []
    slopes_range = []
    or_atr_trend = []
    for d in strat.diagnostics:
        regimes[d.regime] = regimes.get(d.regime, 0) + 1
        if d.ema_slope is not None:
            slopes_all.append(abs(d.ema_slope))
        if d.regime == "TREND":
            if d.ema_slope is not None:
                slopes_trend.append(abs(d.ema_slope))
            if d.atr and d.or_range:
                or_atr_trend.append(d.or_range / d.atr)
        if d.regime == "RANGE":
            if d.ema_slope is not None:
                slopes_range.append(abs(d.ema_slope))

    sel = strat.selectivity
    print(f"=== MNQ {year} ===")
    print(f"  Regimes: {regimes}")
    et = sel["entries_taken"]
    dt = sel["days_traded"]
    dw = sel["days_with_range"]
    print(f"  Trades: {et}, Days traded: {dt}/{dw}")
    if slopes_all:
        sa = sorted(slopes_all)
        print(f"  All-day |slope|: p25={sa[len(sa)//4]:.3f} med={statistics.median(sa):.3f} p75={sa[3*len(sa)//4]:.3f} max={max(sa):.3f}")
    if slopes_trend:
        st = sorted(slopes_trend)
        print(f"  TREND-day |slope|: min={min(st):.3f} p25={st[len(st)//4]:.3f} med={statistics.median(st):.3f} p75={st[3*len(st)//4]:.3f} max={max(st):.3f}")
    if slopes_range:
        sr = sorted(slopes_range)
        print(f"  RANGE-day |slope|: min={min(sr):.3f} max={max(sr):.3f} med={statistics.median(sr):.3f}")
    if or_atr_trend:
        print(f"  TREND-day OR/ATR: min={min(or_atr_trend):.2f} med={statistics.median(or_atr_trend):.2f} max={max(or_atr_trend):.2f}")

    # Show what happens at different thresholds
    for thresh in [0.50, 1.00, 1.50, 2.00, 3.00]:
        count = sum(1 for s in slopes_all if s >= thresh)
        print(f"  |slope| >= {thresh:.2f}: {count} days ({100*count/len(slopes_all):.0f}%)")
    print()


if __name__ == "__main__":
    for year in [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]:
        run_year(year)
