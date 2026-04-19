"""Measure slope/ATR and slope/price ratios to find robust TREND threshold."""
import pandas as pd
import statistics
from data.loader import load_bars
from config.settings import AdaptiveRegimeConfig
from strategy.adaptive_regime import AdaptiveRegimeStrategy


def run_year(symbol, year):
    files = {
        2017: f"data/{symbol.lower()}_2017.csv",
        2018: f"data/{symbol.lower()}_2018.csv",
        2019: f"data/{symbol.lower()}_2019.csv",
        2022: f"data/{symbol.lower()}_2022.csv",
        2025: f"data/{symbol.lower()}_2025.csv",
    }
    if year in files:
        bars = load_bars(files[year])
    else:
        bars = load_bars(f"data/{symbol.lower()}_4y.csv")
        import pytz
        s = pd.Timestamp(f"{year}-01-01", tz="America/New_York")
        e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="America/New_York")
        bars = bars[(bars.index >= s) & (bars.index <= e)]

    cfg = AdaptiveRegimeConfig.for_symbol(symbol)
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
    slope_atr_all = []
    slope_atr_trend = []
    slope_atr_range = []
    for d in strat.diagnostics:
        regimes[d.regime] = regimes.get(d.regime, 0) + 1
        if d.ema_slope is not None and d.atr and d.atr > 0:
            ratio = abs(d.ema_slope) / d.atr
            slope_atr_all.append(ratio)
            if d.regime == "TREND":
                slope_atr_trend.append(ratio)
            elif d.regime == "RANGE":
                slope_atr_range.append(ratio)

    trend_count = regimes.get("TREND", 0)
    print(f"  {symbol} {year}: TREND={trend_count:3d}", end="")
    if slope_atr_all:
        sa = sorted(slope_atr_all)
        print(f"  allDay slope/ATR: med={statistics.median(sa):.4f} p75={sa[3*len(sa)//4]:.4f}", end="")
    if slope_atr_trend:
        st = sorted(slope_atr_trend)
        print(f"  TREND slope/ATR: min={min(st):.4f} med={statistics.median(st):.4f} p75={st[3*len(st)//4]:.4f}", end="")

    # Count TREND days at different slope/ATR thresholds
    for thresh in [0.005, 0.008, 0.010, 0.012, 0.015]:
        count = sum(1 for r in slope_atr_all if r >= thresh)
        print(f"  >={thresh}:{count}", end="")
    print()
    return slope_atr_all, slope_atr_trend


if __name__ == "__main__":
    print("=== MNQ ===")
    for year in [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]:
        run_year("MNQ", year)

    print("\n=== MES (for comparison) ===")
    for year in [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]:
        run_year("MES", year)
