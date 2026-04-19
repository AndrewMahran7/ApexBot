import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
"""
MNQ vs MES detailed comparison.

Generates EMA candidates for MNQ (2021-2024) using the same pipeline,
then compares trade characteristics, move sizes, volatility, and PnL
against the existing MES dataset.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/mnq_vs_mes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Costs
MES_PT = 5.0   # $/point
MNQ_PT = 2.0   # $/point
COMMISSION = 2.32 * 2  # round-trip
MES_SLIP = 0.25 * MES_PT
MNQ_SLIP = 0.25 * MNQ_PT

# ── Generate MNQ candidates ─────────────────────────────────────────
log.info("Generating MNQ EMA candidates (2021-2024) ...")
from data.ema_candidates import build_ema_candidate_dataset, EMACandidateConfig

mnq_cfg = EMACandidateConfig(allow_shorts=True)
mnq = build_ema_candidate_dataset("data/mnq_4y.csv", cfg=mnq_cfg, save_path=None)
mnq["symbol"] = "MNQ"
mnq["year"] = pd.to_datetime(mnq["session_date"]).dt.year
log.info(f"MNQ: {len(mnq)} candidates")

# ── Load MES candidates ─────────────────────────────────────────────
log.info("Loading MES EMA candidates ...")
mes = pd.read_csv("data/ema_candidates.csv", parse_dates=["session_date"])
mes["symbol"] = "MES"
mes["year"] = mes["session_date"].dt.year
log.info(f"MES: {len(mes)} candidates")

# ── Compute dollar PnL ──────────────────────────────────────────────
mes["pnl_dollars"] = mes["label_pnl_pts"] * MES_PT - MES_SLIP - COMMISSION
mnq["pnl_dollars"] = mnq["label_pnl_pts"] * MNQ_PT - MNQ_SLIP - COMMISSION

# ── Helper ───────────────────────────────────────────────────────────
def compute_stats(df, label, pt_value):
    n = len(df)
    wins = df["label_success"].sum()
    wr = wins / n if n else 0

    pnl = df["pnl_dollars"]
    pnl_pts = df["label_pnl_pts"]
    gp = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    pf = gp / gl if gl > 0 else 0

    # Exit breakdown
    exits = df["label_exit_reason"].value_counts()

    # Winning/losing trade stats
    win_trades = df[df["label_success"] == 1]
    loss_trades = df[df["label_success"] == 0]

    return {
        "symbol": label,
        "n": n,
        "wins": int(wins),
        "win_rate": round(wr, 4),
        "total_pnl": round(pnl.sum(), 2),
        "avg_pnl": round(pnl.mean(), 2),
        "median_pnl": round(pnl.median(), 2),
        "std_pnl": round(pnl.std(), 2),
        "profit_factor": round(pf, 3),
        # Points
        "avg_pnl_pts": round(pnl_pts.mean(), 2),
        "median_pnl_pts": round(pnl_pts.median(), 2),
        "avg_win_pts": round(win_trades["label_pnl_pts"].mean(), 2) if len(win_trades) else 0,
        "avg_loss_pts": round(loss_trades["label_pnl_pts"].mean(), 2) if len(loss_trades) else 0,
        # Range / volatility
        "avg_range_size": round(df["range_size"].mean(), 2),
        "median_range_size": round(df["range_size"].median(), 2),
        "avg_risk_points": round(df["f_risk_points"].mean(), 2),
        "avg_atr": round(df["f_vola_atr"].mean(), 4),
        "avg_range_vs_atr": round(df["f_range_size_vs_atr"].mean(), 4) if "f_range_size_vs_atr" in df else 0,
        # EMA / trend
        "avg_ema_slope": round(df["f_price_ema_slope"].mean(), 6),
        "avg_trend_dir": round(df["f_regime_trend_direction"].mean(), 4),
        # Volume
        "avg_vol_relative": round(df["f_vol_relative"].mean(), 4),
        # Direction
        "pct_long": round((df["direction"] == "long").mean(), 4),
        # Exit breakdown
        "exit_TP": int(exits.get("TP", 0)),
        "exit_SL": int(exits.get("SL", 0)),
        "exit_EOD": int(exits.get("EOD", 0)),
        "exit_TP_pct": round(exits.get("TP", 0) / n * 100, 1),
        "exit_SL_pct": round(exits.get("SL", 0) / n * 100, 1),
        "exit_EOD_pct": round(exits.get("EOD", 0) / n * 100, 1),
        # Cost burden
        "cost_per_trade": round(MES_SLIP + COMMISSION if "MES" in label else MNQ_SLIP + COMMISSION, 2),
        "cost_as_pct_of_range": round(
            (MES_SLIP + COMMISSION if "MES" in label else MNQ_SLIP + COMMISSION)
            / (df["range_size"].mean() * pt_value) * 100, 2
        ),
    }


# ── Overall comparison ───────────────────────────────────────────────
mes_stats = compute_stats(mes, "MES", MES_PT)
mnq_stats = compute_stats(mnq, "MNQ", MNQ_PT)

log.info(f"\n{'='*80}")
log.info("OVERALL COMPARISON: MES vs MNQ (2021-2024)")
log.info(f"{'='*80}")
log.info(f"\n  {'Metric':<30} {'MES':>15} {'MNQ':>15} {'Delta':>12}")
log.info(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*12}")

rows = [
    ("N candidates", mes_stats["n"], mnq_stats["n"], None),
    ("Win rate", mes_stats["win_rate"], mnq_stats["win_rate"], True),
    ("Avg PnL ($)", mes_stats["avg_pnl"], mnq_stats["avg_pnl"], True),
    ("Median PnL ($)", mes_stats["median_pnl"], mnq_stats["median_pnl"], True),
    ("Total PnL ($)", mes_stats["total_pnl"], mnq_stats["total_pnl"], True),
    ("Profit factor", mes_stats["profit_factor"], mnq_stats["profit_factor"], True),
    ("Std PnL ($)", mes_stats["std_pnl"], mnq_stats["std_pnl"], None),
    ("", None, None, None),
    ("Avg PnL (pts)", mes_stats["avg_pnl_pts"], mnq_stats["avg_pnl_pts"], True),
    ("Avg win (pts)", mes_stats["avg_win_pts"], mnq_stats["avg_win_pts"], None),
    ("Avg loss (pts)", mes_stats["avg_loss_pts"], mnq_stats["avg_loss_pts"], None),
    ("", None, None, None),
    ("Avg range (pts)", mes_stats["avg_range_size"], mnq_stats["avg_range_size"], None),
    ("Avg risk/stop (pts)", mes_stats["avg_risk_points"], mnq_stats["avg_risk_points"], None),
    ("Avg ATR", mes_stats["avg_atr"], mnq_stats["avg_atr"], None),
    ("Range / ATR", mes_stats["avg_range_vs_atr"], mnq_stats["avg_range_vs_atr"], None),
    ("", None, None, None),
    ("Avg EMA slope", mes_stats["avg_ema_slope"], mnq_stats["avg_ema_slope"], None),
    ("Avg trend direction", mes_stats["avg_trend_dir"], mnq_stats["avg_trend_dir"], None),
    ("Avg vol relative", mes_stats["avg_vol_relative"], mnq_stats["avg_vol_relative"], None),
    ("% Long", mes_stats["pct_long"], mnq_stats["pct_long"], None),
    ("", None, None, None),
    ("TP %", mes_stats["exit_TP_pct"], mnq_stats["exit_TP_pct"], True),
    ("SL %", mes_stats["exit_SL_pct"], mnq_stats["exit_SL_pct"], False),
    ("EOD %", mes_stats["exit_EOD_pct"], mnq_stats["exit_EOD_pct"], None),
    ("", None, None, None),
    ("Cost/trade ($)", mes_stats["cost_per_trade"], mnq_stats["cost_per_trade"], False),
    ("Cost as % of range$", mes_stats["cost_as_pct_of_range"], mnq_stats["cost_as_pct_of_range"], False),
]

for name, mv, nv, higher in rows:
    if mv is None:
        log.info("")
        continue
    if isinstance(mv, int) and higher is None:
        log.info(f"  {name:<30} {mv:>15} {nv:>15}")
    else:
        delta = nv - mv
        marker = ""
        if higher is not None:
            marker = " ✓" if (delta > 0) == higher else " ✗"
        log.info(f"  {name:<30} {mv:>15.4f} {nv:>15.4f} {delta:>+11.4f}{marker}")


# ── Per-year comparison ──────────────────────────────────────────────
log.info(f"\n{'='*80}")
log.info("PER-YEAR COMPARISON")
log.info(f"{'='*80}")
log.info(f"\n  {'Year':<6} │ {'MES WR':>7} {'MES PnL':>10} {'MES PF':>7} │ "
         f"{'MNQ WR':>7} {'MNQ PnL':>10} {'MNQ PF':>7} │ {'Δ WR':>6} {'Δ PnL':>10}")
log.info(f"  {'─'*6}─┼─{'─'*7}─{'─'*10}─{'─'*7}─┼─{'─'*7}─{'─'*10}─{'─'*7}─┼─{'─'*6}─{'─'*10}")

for year in sorted(mes["year"].unique()):
    m = mes[mes["year"] == year]
    n = mnq[mnq["year"] == year]

    m_wr = m["label_success"].mean()
    n_wr = n["label_success"].mean()
    m_pnl = m["pnl_dollars"].sum()
    n_pnl = n["pnl_dollars"].sum()

    m_gp = m["pnl_dollars"][m["pnl_dollars"] > 0].sum()
    m_gl = abs(m["pnl_dollars"][m["pnl_dollars"] < 0].sum())
    m_pf = m_gp / m_gl if m_gl > 0 else 0

    n_gp = n["pnl_dollars"][n["pnl_dollars"] > 0].sum()
    n_gl = abs(n["pnl_dollars"][n["pnl_dollars"] < 0].sum())
    n_pf = n_gp / n_gl if n_gl > 0 else 0

    log.info(f"  {year:<6} │ {m_wr:>6.1%} ${m_pnl:>9.2f} {m_pf:>7.2f} │ "
             f"{n_wr:>6.1%} ${n_pnl:>9.2f} {n_pf:>7.2f} │ "
             f"{n_wr - m_wr:>+5.1%} ${n_pnl - m_pnl:>+9.2f}")


# ── Structural analysis: why MNQ differs ────────────────────────────
log.info(f"\n{'='*80}")
log.info("STRUCTURAL ANALYSIS")
log.info(f"{'='*80}")

# Range size distribution
log.info(f"\n  Range size distribution (points):")
for label, df_sym in [("MES", mes), ("MNQ", mnq)]:
    rs = df_sym["range_size"]
    log.info(f"  {label}: mean={rs.mean():.2f}, median={rs.median():.2f}, "
             f"std={rs.std():.2f}, p10={rs.quantile(0.1):.1f}, p90={rs.quantile(0.9):.1f}")

# Risk (stop distance) distribution
log.info(f"\n  Risk/stop distance (points):")
for label, df_sym in [("MES", mes), ("MNQ", mnq)]:
    rp = df_sym["f_risk_points"]
    log.info(f"  {label}: mean={rp.mean():.2f}, median={rp.median():.2f}, "
             f"std={rp.std():.2f}, p10={rp.quantile(0.1):.1f}, p90={rp.quantile(0.9):.1f}")

# Dollar risk per trade
log.info(f"\n  Dollar risk per trade:")
mes_risk_d = mes["f_risk_points"] * MES_PT
mnq_risk_d = mnq["f_risk_points"] * MNQ_PT
log.info(f"  MES: mean=${mes_risk_d.mean():.2f}, median=${mes_risk_d.median():.2f}")
log.info(f"  MNQ: mean=${mnq_risk_d.mean():.2f}, median=${mnq_risk_d.median():.2f}")

# Cost burden analysis
log.info(f"\n  Cost burden analysis:")
mes_range_d = mes["range_size"] * MES_PT
mnq_range_d = mnq["range_size"] * MNQ_PT
mes_cost = MES_SLIP + COMMISSION
mnq_cost = MNQ_SLIP + COMMISSION
log.info(f"  MES: range_avg=${mes_range_d.mean():.2f}, cost=${mes_cost:.2f}, "
         f"cost/range={mes_cost / mes_range_d.mean() * 100:.1f}%")
log.info(f"  MNQ: range_avg=${mnq_range_d.mean():.2f}, cost=${mnq_cost:.2f}, "
         f"cost/range={mnq_cost / mnq_range_d.mean() * 100:.1f}%")

# Points needed to break even after costs
mes_be_pts = mes_cost / MES_PT
mnq_be_pts = mnq_cost / MNQ_PT
log.info(f"\n  Points needed to break even (costs only):")
log.info(f"  MES: {mes_be_pts:.2f} pts (avg range {mes['range_size'].mean():.1f} pts, "
         f"BE = {mes_be_pts / mes['range_size'].mean() * 100:.1f}% of range)")
log.info(f"  MNQ: {mnq_be_pts:.2f} pts (avg range {mnq['range_size'].mean():.1f} pts, "
         f"BE = {mnq_be_pts / mnq['range_size'].mean() * 100:.1f}% of range)")

# PnL before vs after costs
log.info(f"\n  PnL before vs after costs:")
mes_raw = mes["label_pnl_pts"] * MES_PT
mnq_raw = mnq["label_pnl_pts"] * MNQ_PT
log.info(f"  MES: raw avg=${mes_raw.mean():.2f}, after costs=${mes['pnl_dollars'].mean():.2f}, "
         f"cost drag=${mes_cost:.2f}/trade")
log.info(f"  MNQ: raw avg=${mnq_raw.mean():.2f}, after costs=${mnq['pnl_dollars'].mean():.2f}, "
         f"cost drag=${mnq_cost:.2f}/trade")

# ── Exit reason x direction cross-tab ────────────────────────────────
log.info(f"\n  Exit reason breakdown by direction:")
for label, df_sym in [("MES", mes), ("MNQ", mnq)]:
    log.info(f"\n  {label}:")
    ct = pd.crosstab(df_sym["direction"], df_sym["label_exit_reason"], normalize="index")
    for d in ["long", "short"]:
        if d in ct.index:
            row = ct.loc[d]
            log.info(f"    {d:>6}: TP={row.get('TP',0):.1%}, SL={row.get('SL',0):.1%}, EOD={row.get('EOD',0):.1%}")

# ── Wins: size comparison ────────────────────────────────────────────
log.info(f"\n  Winning trade size comparison (points):")
for label, df_sym in [("MES", mes), ("MNQ", mnq)]:
    w = df_sym[df_sym["label_success"] == 1]["label_pnl_pts"]
    log.info(f"  {label} wins: mean={w.mean():.2f}, median={w.median():.2f}, std={w.std():.2f}")

log.info(f"\n  Losing trade size comparison (points):")
for label, df_sym in [("MES", mes), ("MNQ", mnq)]:
    l = df_sym[df_sym["label_success"] == 0]["label_pnl_pts"]
    log.info(f"  {label} losses: mean={l.mean():.2f}, median={l.median():.2f}, std={l.std():.2f}")

# ── Statistical test: are MNQ wins genuinely larger? ─────────────────
mes_win_pts = mes[mes["label_success"] == 1]["label_pnl_pts"]
mnq_win_pts = mnq[mnq["label_success"] == 1]["label_pnl_pts"]
t_stat, p_val = sp_stats.ttest_ind(mes_win_pts, mnq_win_pts, equal_var=False)
log.info(f"\n  t-test (MES wins vs MNQ wins in points): t={t_stat:.3f}, p={p_val:.4f}")

mes_loss_pts = mes[mes["label_success"] == 0]["label_pnl_pts"]
mnq_loss_pts = mnq[mnq["label_success"] == 0]["label_pnl_pts"]
t2, p2 = sp_stats.ttest_ind(mes_loss_pts, mnq_loss_pts, equal_var=False)
log.info(f"  t-test (MES losses vs MNQ losses in points): t={t2:.3f}, p={p2:.4f}")

# ── Save ─────────────────────────────────────────────────────────────
summary = {
    "mes": mes_stats,
    "mnq": mnq_stats,
    "per_year": {},
}
for year in sorted(mes["year"].unique()):
    m = mes[mes["year"] == year]
    n = mnq[mnq["year"] == year]
    summary["per_year"][str(year)] = {
        "mes_wr": round(m["label_success"].mean(), 4),
        "mnq_wr": round(n["label_success"].mean(), 4),
        "mes_pnl": round(m["pnl_dollars"].sum(), 2),
        "mnq_pnl": round(n["pnl_dollars"].sum(), 2),
    }

with open(OUT_DIR / "comparison.json", "w") as f:
    json.dump(summary, f, indent=2)
log.info(f"\nSaved to {OUT_DIR / 'comparison.json'}")
