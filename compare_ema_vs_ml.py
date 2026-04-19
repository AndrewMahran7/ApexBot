"""
Compare EMA-only vs EMA+ML filter on MNQ across all years.

Uses:
  - ml_threshold=0.0 for EMA-only (accept everything)
  - ml_threshold=0.55 for ML-filtered (production-like)

Same instrument, same data, same costs. Only difference is the ML gate.
"""

import json
import sys
import time
from pathlib import Path

from config.settings import BacktestConfig, StrategyConfig, INSTRUMENT_REGISTRY
from data.loader import load_bars
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics
from strategy.hybrid_ema_ml import HybridEMAMLStrategy, HybridEMAMLConfig

OUT_DIR = Path("results/ema_vs_ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUMENT = INSTRUMENT_REGISTRY["MNQ"]
INITIAL_CAPITAL = 5000.0
SLIPPAGE_TICKS = 1
COMMISSION = 2.32

YEAR_CONFIG = {
    2017: ("data/mnq_2017.csv", None, None),
    2018: ("data/mnq_2018.csv", None, None),
    2019: ("data/mnq_2019.csv", None, None),
    2024: ("data/mnq_4y.csv", "2024-01-01", "2024-12-31"),
    2025: ("data/mnq_2025.csv", None, None),
}

ML_THRESHOLD = 0.55


def run_backtest(year, ml_threshold):
    data_file, start, end = YEAR_CONFIG[year]

    cfg = HybridEMAMLConfig(
        allow_shorts=True,
        ml_threshold=ml_threshold,
        model_path="models/ema_model.pkl",
    )
    strategy = HybridEMAMLStrategy(cfg)

    strat_cfg = StrategyConfig(shorts_enabled=True, ema_enabled=True, ema_length=50)
    bt_cfg = BacktestConfig(
        slippage_ticks=SLIPPAGE_TICKS,
        commission_per_side=COMMISSION,
        initial_capital=INITIAL_CAPITAL,
    )

    bars = load_bars(data_file, timezone=strat_cfg.timezone, start=start, end=end)
    engine = BacktestEngine(INSTRUMENT, strat_cfg, bt_cfg, strategy=strategy)
    result = engine.run(bars)

    trades = result.trades
    n = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    total_pnl = sum(t.net_pnl for t in trades)
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_pnl = total_pnl / n if n else 0.0

    return {
        "year": year,
        "trades": n,
        "wins": wins,
        "win_rate": round(wins / n * 100, 1) if n else 0.0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "profit_factor": round(pf, 2),
    }


def main():
    print("=" * 80)
    print("  MNQ: EMA-ONLY vs EMA+ML COMPARISON")
    print(f"  EMA-only: ml_threshold=0.0 | ML-filtered: ml_threshold={ML_THRESHOLD}")
    print(f"  Instrument: MNQ ($2/pt) | EMA-50 | Shorts: ON | Causal entry")
    print(f"  Slippage: {SLIPPAGE_TICKS} tick | Commission: ${COMMISSION}/side")
    print("=" * 80)

    rows = []
    t0 = time.time()

    for year in sorted(YEAR_CONFIG.keys()):
        print(f"\n--- {year} ---")

        print(f"  Running EMA-only (threshold=0.0) ...")
        ema = run_backtest(year, ml_threshold=0.0)
        print(f"    {ema['trades']} trades, WR {ema['win_rate']}%, PnL ${ema['total_pnl']:.2f}")

        print(f"  Running EMA+ML (threshold={ML_THRESHOLD}) ...")
        ml = run_backtest(year, ml_threshold=ML_THRESHOLD)
        print(f"    {ml['trades']} trades, WR {ml['win_rate']}%, PnL ${ml['total_pnl']:.2f}")

        rows.append({
            "year": year,
            "ema_trades": ema["trades"],
            "ema_win_rate": ema["win_rate"],
            "ema_pnl": ema["total_pnl"],
            "ema_avg_pnl": ema["avg_pnl"],
            "ema_pf": ema["profit_factor"],
            "ml_trades": ml["trades"],
            "ml_win_rate": ml["win_rate"],
            "ml_pnl": ml["total_pnl"],
            "ml_avg_pnl": ml["avg_pnl"],
            "ml_pf": ml["profit_factor"],
            "trades_filtered": ema["trades"] - ml["trades"],
            "pnl_delta": round(ml["total_pnl"] - ema["total_pnl"], 2),
            "wr_delta": round(ml["win_rate"] - ema["win_rate"], 1),
        })

    elapsed = time.time() - t0

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  COMPARISON: EMA-ONLY vs EMA+ML (MNQ)")
    print(f"{'='*110}")
    hdr = (f"  {'Year':<6} │ {'EMA Trades':>10} {'EMA WR':>7} {'EMA PnL':>10} "
           f"{'EMA PF':>7} │ {'ML Trades':>10} {'ML WR':>7} {'ML PnL':>10} "
           f"{'ML PF':>7} │ {'Δ PnL':>9} {'Δ WR':>6}")
    print(hdr)
    print(f"  {'─'*6}─┼─{'─'*10}─{'─'*7}─{'─'*10}─{'─'*7}─┼─{'─'*10}─{'─'*7}─{'─'*10}─{'─'*7}─┼─{'─'*9}─{'─'*6}")

    for r in rows:
        print(f"  {r['year']:<6} │ {r['ema_trades']:>10} {r['ema_win_rate']:>6.1f}% "
              f"${r['ema_pnl']:>9.2f} {r['ema_pf']:>7.2f} │ "
              f"{r['ml_trades']:>10} {r['ml_win_rate']:>6.1f}% "
              f"${r['ml_pnl']:>9.2f} {r['ml_pf']:>7.2f} │ "
              f"${r['pnl_delta']:>8.2f} {r['wr_delta']:>+5.1f}%")

    # ── Totals ───────────────────────────────────────────────────────
    t_ema_trades = sum(r["ema_trades"] for r in rows)
    t_ml_trades  = sum(r["ml_trades"] for r in rows)
    t_ema_pnl    = sum(r["ema_pnl"] for r in rows)
    t_ml_pnl     = sum(r["ml_pnl"] for r in rows)
    t_ema_wins   = sum(r["ema_trades"] * r["ema_win_rate"] / 100 for r in rows)
    t_ml_wins    = sum(r["ml_trades"] * r["ml_win_rate"] / 100 for r in rows)
    t_ema_wr     = t_ema_wins / t_ema_trades * 100 if t_ema_trades else 0
    t_ml_wr      = t_ml_wins / t_ml_trades * 100 if t_ml_trades else 0
    t_ema_avg    = t_ema_pnl / t_ema_trades if t_ema_trades else 0
    t_ml_avg     = t_ml_pnl / t_ml_trades if t_ml_trades else 0

    print(f"  {'─'*6}─┼─{'─'*10}─{'─'*7}─{'─'*10}─{'─'*7}─┼─{'─'*10}─{'─'*7}─{'─'*10}─{'─'*7}─┼─{'─'*9}─{'─'*6}")
    print(f"  {'TOTAL':<6} │ {t_ema_trades:>10} {t_ema_wr:>6.1f}% "
          f"${t_ema_pnl:>9.2f} {'':>7} │ "
          f"{t_ml_trades:>10} {t_ml_wr:>6.1f}% "
          f"${t_ml_pnl:>9.2f} {'':>7} │ "
          f"${t_ml_pnl - t_ema_pnl:>8.2f} {t_ml_wr - t_ema_wr:>+5.1f}%")

    # ── Per-trade quality ────────────────────────────────────────────
    print(f"\n  PER-TRADE QUALITY:")
    print(f"  {'':>8} {'EMA Avg':>10} {'ML Avg':>10} {'Improvement':>12}")
    print(f"  {'':>8} {'─'*10} {'─'*10} {'─'*12}")
    for r in rows:
        imp = r["ml_avg_pnl"] - r["ema_avg_pnl"]
        print(f"  {r['year']:>8} ${r['ema_avg_pnl']:>9.2f} ${r['ml_avg_pnl']:>9.2f} "
              f"${imp:>+11.2f}")
    print(f"  {'TOTAL':>8} ${t_ema_avg:>9.2f} ${t_ml_avg:>9.2f} "
          f"${t_ml_avg - t_ema_avg:>+11.2f}")

    # ── Filtering stats ──────────────────────────────────────────────
    print(f"\n  FILTERING STATS:")
    print(f"  {'Year':>8} {'Candidates':>12} {'Accepted':>10} {'Filtered':>10} {'Filter%':>8}")
    for r in rows:
        filt_pct = r["trades_filtered"] / r["ema_trades"] * 100 if r["ema_trades"] else 0
        print(f"  {r['year']:>8} {r['ema_trades']:>12} {r['ml_trades']:>10} "
              f"{r['trades_filtered']:>10} {filt_pct:>7.1f}%")
    total_filt = t_ema_trades - t_ml_trades
    total_filt_pct = total_filt / t_ema_trades * 100 if t_ema_trades else 0
    print(f"  {'TOTAL':>8} {t_ema_trades:>12} {t_ml_trades:>10} "
          f"{total_filt:>10} {total_filt_pct:>7.1f}%")

    # ── ML Verdict ───────────────────────────────────────────────────
    ml_better_years = sum(1 for r in rows if r["pnl_delta"] > 0)
    print(f"\n  VERDICT:")
    print(f"  ML improves PnL in {ml_better_years}/{len(rows)} years")
    print(f"  Total EMA PnL: ${t_ema_pnl:,.2f}  |  Total ML PnL: ${t_ml_pnl:,.2f}")
    print(f"  ML delta: ${t_ml_pnl - t_ema_pnl:+,.2f}")
    if t_ml_pnl > t_ema_pnl:
        print(f"  → ML ADDS VALUE (${t_ml_pnl - t_ema_pnl:+,.2f})")
    else:
        print(f"  → ML DOES NOT ADD VALUE (${t_ml_pnl - t_ema_pnl:+,.2f})")

    print(f"\n  Completed in {elapsed:.1f}s")

    # ── Save ─────────────────────────────────────────────────────────
    summary = {
        "test": "ema_vs_ml_comparison",
        "instrument": "MNQ",
        "ema_threshold": 0.0,
        "ml_threshold": ML_THRESHOLD,
        "per_year": rows,
        "totals": {
            "ema_trades": t_ema_trades,
            "ml_trades": t_ml_trades,
            "ema_pnl": round(t_ema_pnl, 2),
            "ml_pnl": round(t_ml_pnl, 2),
            "pnl_delta": round(t_ml_pnl - t_ema_pnl, 2),
            "ema_avg_pnl": round(t_ema_avg, 2),
            "ml_avg_pnl": round(t_ml_avg, 2),
        },
    }
    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved to {OUT_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
