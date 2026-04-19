"""
Robustness test: SL/TP distance sensitivity (optimized).

Tests +/-10% and +/-20% perturbations to:
  1) Stop-loss distance  (shift range_low/high toward or away from entry)
  2) Take-profit distance (scale reward_risk ratio)
  3) Both simultaneously

Each (sl_factor, tp_factor) pair is computed ONCE, then sliced by year.
"""

from __future__ import annotations
import logging, json
import numpy as np
import pandas as pd
from pathlib import Path

from data.loader import load_bars
from data.features import compute_features, FeatureConfig
from data.ema_candidates import EMACandidateConfig, _identify_ema_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

MES_PT = 5.0
COMMISSION_RT = 4.64
SLIPPAGE = 1.25


def label_with_adjustments(
    candidates: pd.DataFrame,
    full_bars: pd.DataFrame,
    cfg: EMACandidateConfig,
    sl_factor: float = 1.0,
    tp_factor: float = 1.0,
) -> pd.DataFrame:
    eod_t = pd.Timestamp("1970-01-01 15:50").time()
    results = []

    for ts, cand in candidates.iterrows():
        direction = cand["direction"]
        entry_price = cand["entry_price"]
        orig_sl = cand["stop_loss"]
        orig_tp = cand["take_profit"]
        session_date = cand["session_date"]
        entry_time = cand["entry_time"]

        if direction == "long":
            sl_dist = entry_price - orig_sl
            tp_dist = orig_tp - entry_price
        else:
            sl_dist = orig_sl - entry_price
            tp_dist = entry_price - orig_tp

        adj_sl_dist = sl_dist * sl_factor
        adj_tp_dist = tp_dist * tp_factor

        if direction == "long":
            adj_sl = entry_price - adj_sl_dist
            adj_tp = entry_price + adj_tp_dist
        else:
            adj_sl = entry_price + adj_sl_dist
            adj_tp = entry_price - adj_tp_dist

        session_mask = (
            (full_bars["session_date"] == session_date)
            & (full_bars.index > entry_time)
            & (full_bars.index.time <= eod_t)
        )
        future_bars = full_bars.loc[session_mask]

        if future_bars.empty:
            continue

        exit_price = float(future_bars.iloc[-1]["close"])
        exit_reason = "EOD"

        for _, fb in future_bars.iterrows():
            high = float(fb["high"])
            low = float(fb["low"])
            if direction == "long":
                if low <= adj_sl:
                    exit_price, exit_reason = adj_sl, "SL"
                    break
                if high >= adj_tp:
                    exit_price, exit_reason = adj_tp, "TP"
                    break
            else:
                if high >= adj_sl:
                    exit_price, exit_reason = adj_sl, "SL"
                    break
                if low <= adj_tp:
                    exit_price, exit_reason = adj_tp, "TP"
                    break

        pnl_pts = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        pnl_dollars = pnl_pts * MES_PT - COMMISSION_RT - SLIPPAGE

        results.append({
            "pnl_dollars": pnl_dollars,
            "win": 1 if pnl_pts > 0 else 0,
            "exit_reason": exit_reason,
            "year": pd.Timestamp(ts).year,
        })

    return pd.DataFrame(results)


def calc_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    wins = df[df["pnl_dollars"] > 0]
    losses = df[df["pnl_dollars"] <= 0]
    gp = wins["pnl_dollars"].sum() if len(wins) else 0
    gl = abs(losses["pnl_dollars"].sum()) if len(losses) else 1
    return {
        "n": len(df),
        "wr": round(df["win"].mean(), 4),
        "avg_pnl": round(df["pnl_dollars"].mean(), 2),
        "total": round(df["pnl_dollars"].sum(), 2),
        "pf": round(gp / gl, 3) if gl > 0 else 0,
        "tp_pct": round((df["exit_reason"] == "TP").mean() * 100, 1),
        "sl_pct": round((df["exit_reason"] == "SL").mean() * 100, 1),
        "eod_pct": round((df["exit_reason"] == "EOD").mean() * 100, 1),
    }


def print_table(title, results_by_label):
    log.info(f"\n{'=' * 78}")
    log.info(f"  {title}")
    log.info("=" * 78)
    hdr = f"  {'Adj':<8} {'N':>5} {'WR':>7} {'Avg PnL':>10} {'Total':>12} {'PF':>6} {'TP%':>6} {'SL%':>6} {'EOD%':>6}"
    log.info(hdr)
    log.info("  " + "-" * 74)
    for lbl, m in results_by_label.items():
        log.info(
            f"  {lbl:<8} {m['n']:>5} {m['wr']:>6.1%} "
            f"${m['avg_pnl']:>9.2f} ${m['total']:>11.2f} "
            f"{m['pf']:>5.3f} {m['tp_pct']:>5.1f} {m['sl_pct']:>5.1f} {m['eod_pct']:>5.1f}"
        )


def main():
    log.info("=" * 78)
    log.info("  SL/TP ROBUSTNESS TEST")
    log.info("=" * 78)

    log.info("Loading bars and generating candidates ...")
    bars = load_bars("data/mes_4y.csv")
    cfg = EMACandidateConfig()
    feat_cfg = FeatureConfig(
        session_open=cfg.session_open,
        range_start=cfg.range_start,
        range_end=cfg.range_end,
        ema_length=cfg.ema_length,
    )
    featured = compute_features(bars, feat_cfg)
    candidates = _identify_ema_candidates(featured, cfg)
    log.info("  %d candidates", len(candidates))

    perturbations = [0.80, 0.90, 1.00, 1.10, 1.20]
    labels = ["-20%", "-10%", "BASE", "+10%", "+20%"]
    all_results = {}

    # Collect all unique (sl_factor, tp_factor) combos to compute
    tests = {
        "SL sensitivity (TP fixed)":   [(p, 1.0) for p in perturbations],
        "TP sensitivity (SL fixed)":   [(1.0, p) for p in perturbations],
        "Both SL+TP scaled together":  [(p, p) for p in perturbations],
    }

    unique_combos = set()
    for combos in tests.values():
        unique_combos.update(combos)

    # Compute each unique combo once
    cached_dfs = {}
    for i, (sl_f, tp_f) in enumerate(sorted(unique_combos), 1):
        log.info(f"  [{i}/{len(unique_combos)}] Computing SL={sl_f:.2f} TP={tp_f:.2f} ...")
        cached_dfs[(sl_f, tp_f)] = label_with_adjustments(
            candidates, featured, cfg, sl_factor=sl_f, tp_factor=tp_f
        )

    # Print results for each test
    for test_name, combos in tests.items():
        results_by_label = {}
        for (sl_f, tp_f), lbl in zip(combos, labels):
            results_by_label[lbl] = calc_metrics(cached_dfs[(sl_f, tp_f)])
        print_table(test_name, results_by_label)
        all_results[test_name] = results_by_label

    # Per-year breakdown (SL sensitivity, reuses cached)
    log.info(f"\n{'=' * 78}")
    log.info("  PER-YEAR TOTAL PnL ACROSS SL PERTURBATIONS (TP fixed)")
    log.info("=" * 78)

    years = sorted(cached_dfs[(1.0, 1.0)]["year"].unique())
    hdr = f"  {'Year':<6} " + " ".join(f"{lbl:>10}" for lbl in labels)
    log.info(hdr)
    log.info("  " + "-" * (len(hdr) - 2))

    yearly_data = {}
    for yr in years:
        parts = []
        yr_d = {}
        for pct, lbl in zip(perturbations, labels):
            yr_df = cached_dfs[(pct, 1.0)]
            yr_df = yr_df[yr_df["year"] == yr]
            total = yr_df["pnl_dollars"].sum()
            parts.append(f"${total:>9.0f}")
            yr_d[lbl] = round(total, 2)
        log.info(f"  {yr:<6} " + " ".join(parts))
        yearly_data[str(yr)] = yr_d
    all_results["yearly_sl_sensitivity"] = yearly_data

    # Stability assessment
    log.info(f"\n{'=' * 78}")
    log.info("  STABILITY ASSESSMENT")
    log.info("=" * 78)

    base_m = all_results["SL sensitivity (TP fixed)"]["BASE"]
    log.info(f"\n  Base: WR={base_m['wr']:.1%}, Avg PnL=${base_m['avg_pnl']:.2f}, PF={base_m['pf']:.3f}")
    log.info("")

    for test_name in tests:
        res = all_results[test_name]
        wr_vals = [v["wr"] for v in res.values()]
        pnl_vals = [v["avg_pnl"] for v in res.values()]
        pf_vals = [v["pf"] for v in res.values()]

        wr_range = max(wr_vals) - min(wr_vals)
        pnl_range = max(pnl_vals) - min(pnl_vals)
        pf_range = max(pf_vals) - min(pf_vals)

        all_negative = all(p < 0 for p in pnl_vals)
        sign_flip = any(p > 0 for p in pnl_vals) and any(p < 0 for p in pnl_vals)

        if all_negative:
            stability = "CONSISTENTLY NEGATIVE"
        elif sign_flip:
            stability = "FRAGILE (sign flips)"
        else:
            stability = "STABLE"

        log.info(
            f"  {test_name:<35}: WR spread={wr_range:.1%}, "
            f"PnL spread=${pnl_range:.2f}, PF spread={pf_range:.3f}  "
            f"-> {stability}"
        )

    # Save
    out_dir = Path("results/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sl_tp_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
