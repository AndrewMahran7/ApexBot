"""
Analyze accepted vs rejected trades from the reduced-16 logistic regression.

For each group: win rate, avg PnL, volatility, trend strength, direction,
and exit reason breakdown.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/baseline_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POINT_VALUE = 5.0
COMMISSION  = 2.32 * 2
SLIPPAGE_PTS = 0.25

# ── Load & split ─────────────────────────────────────────────────────
df = pd.read_csv("data/ema_candidates.csv", parse_dates=["session_date"])
df["year"] = df["session_date"].dt.year
df["f_recent_wr_5"] = df["f_recent_wr_5"].fillna(0.5)
df["f_recent_wr_10"] = df["f_recent_wr_10"].fillna(0.5)

train_df = df[df["year"] <= 2023].copy()
test_df  = df[df["year"] == 2024].copy().reset_index(drop=True)

# ── Reduced 16 features (best model) ────────────────────────────────
feat_cols = [
    "f_risk_points", "f_regime_vol_trend", "f_ema_distance_pct",
    "f_price_ret_12bar", "f_vol_relative", "f_price_ema_dist",
    "f_price_ema_dist_pct", "f_ema_distance", "f_range_high",
    "f_regime_trend_direction", "f_price_ema", "f_range_size_vs_atr",
    "f_price_ema_slope", "f_direction_long", "f_range_vs_atr", "f_range_low",
]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feat_cols].values)
X_test  = scaler.transform(test_df[feat_cols].values)
y_train = train_df["label_success"].values

model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
model.fit(X_train, y_train)

test_df["ml_prob"] = model.predict_proba(X_test)[:, 1]
test_df["pnl_dollars"] = test_df["label_pnl_pts"] * POINT_VALUE - SLIPPAGE_PTS * POINT_VALUE - COMMISSION

log.info(f"Test set: {len(test_df)} trades, prob range [{test_df['ml_prob'].min():.3f}, {test_df['ml_prob'].max():.3f}]")


def group_stats(group_df, label):
    """Compute comprehensive stats for a group of trades."""
    n = len(group_df)
    if n == 0:
        return {}

    wins = group_df["label_success"].sum()
    wr = wins / n
    pnl = group_df["pnl_dollars"]
    pnl_pts = group_df["label_pnl_pts"]

    # PnL stats
    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()
    median_pnl = pnl.median()
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Feature-based stats
    stats = {
        "label": label,
        "n_trades": n,
        "wins": int(wins),
        "win_rate": round(wr, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "median_pnl": round(median_pnl, 2),
        "profit_factor": round(pf, 3),
        "avg_prob": round(group_df["ml_prob"].mean(), 4),
        "avg_risk_points": round(group_df["f_risk_points"].mean(), 2),
        "avg_range_size": round(group_df["range_size"].mean(), 2),
        "avg_ema_slope": round(group_df["f_price_ema_slope"].mean(), 6),
        "avg_trend_strength": round(group_df["f_regime_trend_strength"].mean(), 6) if "f_regime_trend_strength" in group_df else None,
        "avg_trend_direction": round(group_df["f_regime_trend_direction"].mean(), 4),
        "avg_vol_relative": round(group_df["f_vol_relative"].mean(), 4),
        "avg_vol_trend": round(group_df["f_regime_vol_trend"].mean(), 4),
        "avg_atr": round(group_df["f_vola_atr"].mean(), 4) if "f_vola_atr" in group_df else None,
        "avg_atr_ratio": round(group_df["f_regime_atr_ratio"].mean(), 4) if "f_regime_atr_ratio" in group_df else None,
        "avg_ema_distance_pct": round(group_df["f_ema_distance_pct"].mean(), 6),
        "pct_long": round((group_df["direction"] == "long").mean(), 4),
    }

    # Exit reason breakdown
    if "label_exit_reason" in group_df.columns:
        exits = group_df["label_exit_reason"].value_counts()
        for reason in ["TP", "SL", "EOD"]:
            stats[f"exit_{reason}"] = int(exits.get(reason, 0))
            stats[f"exit_{reason}_pct"] = round(exits.get(reason, 0) / n * 100, 1)

    return stats


# ── Analyze at multiple thresholds ───────────────────────────────────
thresholds = [0.35, 0.40, 0.45]

for thresh in thresholds:
    accepted = test_df[test_df["ml_prob"] >= thresh]
    rejected = test_df[test_df["ml_prob"] < thresh]

    a = group_stats(accepted, f"ACCEPTED (prob >= {thresh})")
    r = group_stats(rejected, f"REJECTED (prob < {thresh})")

    log.info(f"\n{'='*80}")
    log.info(f"THRESHOLD: {thresh}")
    log.info(f"{'='*80}")

    log.info(f"\n  {'Metric':<30} {'Accepted':>15} {'Rejected':>15} {'Delta':>12}")
    log.info(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*12}")

    comparisons = [
        ("N trades",          a["n_trades"],          r["n_trades"],          None),
        ("Win rate",           a["win_rate"],           r["win_rate"],           True),
        ("Avg PnL ($)",        a["avg_pnl"],            r["avg_pnl"],            True),
        ("Median PnL ($)",     a["median_pnl"],         r["median_pnl"],         True),
        ("Total PnL ($)",      a["total_pnl"],          r["total_pnl"],          True),
        ("Profit factor",      a["profit_factor"],      r["profit_factor"],      True),
        ("Avg ML prob",        a["avg_prob"],           r["avg_prob"],           None),
        ("Avg risk (pts)",     a["avg_risk_points"],    r["avg_risk_points"],    None),
        ("Avg range size",     a["avg_range_size"],     r["avg_range_size"],     None),
        ("Avg EMA slope",      a["avg_ema_slope"],      r["avg_ema_slope"],      None),
        ("Avg trend direction", a["avg_trend_direction"], r["avg_trend_direction"], None),
        ("Avg vol relative",   a["avg_vol_relative"],   r["avg_vol_relative"],   None),
        ("Avg vol trend",      a["avg_vol_trend"],      r["avg_vol_trend"],      None),
        ("Avg ATR",            a.get("avg_atr"),        r.get("avg_atr"),        None),
        ("Avg ATR ratio",      a.get("avg_atr_ratio"),  r.get("avg_atr_ratio"),  None),
        ("Avg EMA dist %",     a["avg_ema_distance_pct"], r["avg_ema_distance_pct"], None),
        ("% Long",             a["pct_long"],           r["pct_long"],           None),
    ]

    for name, av, rv, higher_better in comparisons:
        if av is None or rv is None:
            continue
        if isinstance(av, int):
            log.info(f"  {name:<30} {av:>15} {rv:>15}")
        else:
            delta = av - rv
            marker = ""
            if higher_better is not None:
                marker = " ✓" if (delta > 0) == higher_better else " ✗"
            log.info(f"  {name:<30} {av:>15.4f} {rv:>15.4f} {delta:>+11.4f}{marker}")

    # Exit breakdowns
    log.info(f"\n  Exit breakdown:")
    log.info(f"  {'Reason':<10} {'Accepted':>15} {'Rejected':>15}")
    for reason in ["TP", "SL", "EOD"]:
        a_n = a.get(f"exit_{reason}", 0)
        a_p = a.get(f"exit_{reason}_pct", 0)
        r_n = r.get(f"exit_{reason}", 0)
        r_p = r.get(f"exit_{reason}_pct", 0)
        log.info(f"  {reason:<10} {a_n:>8} ({a_p:>5.1f}%) {r_n:>8} ({r_p:>5.1f}%)")


# ── Detailed analysis at primary threshold (0.40) ────────────────────
THRESH = 0.40
accepted = test_df[test_df["ml_prob"] >= THRESH].copy()
rejected = test_df[test_df["ml_prob"] < THRESH].copy()

log.info(f"\n{'='*80}")
log.info(f"DEEP DIVE: threshold = {THRESH}")
log.info(f"{'='*80}")

# ── PnL distribution comparison ─────────────────────────────────────
log.info(f"\n  PnL Distribution (points):")
for label, grp in [("Accepted", accepted), ("Rejected", rejected)]:
    pts = grp["label_pnl_pts"]
    log.info(f"  {label}: mean={pts.mean():+.2f}, median={pts.median():+.2f}, "
             f"std={pts.std():.2f}, min={pts.min():.1f}, max={pts.max():.1f}")

# ── Direction breakdown ──────────────────────────────────────────────
log.info(f"\n  Direction breakdown:")
log.info(f"  {'Group':<12} {'Long N':>8} {'Long WR':>8} {'Long PnL':>10} "
         f"{'Short N':>8} {'Short WR':>9} {'Short PnL':>10}")
for label, grp in [("Accepted", accepted), ("Rejected", rejected)]:
    longs = grp[grp["direction"] == "long"]
    shorts = grp[grp["direction"] == "short"]
    l_wr = longs["label_success"].mean() if len(longs) else 0
    s_wr = shorts["label_success"].mean() if len(shorts) else 0
    l_pnl = longs["pnl_dollars"].sum()
    s_pnl = shorts["pnl_dollars"].sum()
    log.info(f"  {label:<12} {len(longs):>8} {l_wr:>7.1%} ${l_pnl:>9.2f} "
             f"{len(shorts):>8} {s_wr:>8.1%} ${s_pnl:>9.2f}")

# ── Probability bins for rejected trades ─────────────────────────────
log.info(f"\n  Rejected trades by probability bin:")
log.info(f"  {'Bin':<15} {'N':>5} {'WR':>7} {'Avg PnL':>10} {'Total PnL':>11}")
bins = [(0.0, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40)]
for lo, hi in bins:
    mask = (rejected["ml_prob"] >= lo) & (rejected["ml_prob"] < hi)
    grp = rejected[mask]
    if len(grp) == 0:
        continue
    wr = grp["label_success"].mean()
    avg = grp["pnl_dollars"].mean()
    total = grp["pnl_dollars"].sum()
    log.info(f"  [{lo:.2f}, {hi:.2f})  {len(grp):>5} {wr:>7.3f} ${avg:>9.2f} ${total:>10.2f}")

# ── What would happen if we ONLY traded the rejected? ────────────────
log.info(f"\n  Counterfactual: What if we ONLY traded rejected trades?")
for label, grp in [("Accepted", accepted), ("Rejected", rejected), ("All", test_df)]:
    n = len(grp)
    wr = grp["label_success"].mean()
    total = grp["pnl_dollars"].sum()
    avg = grp["pnl_dollars"].mean()
    log.info(f"  {label:<12}: {n:>4} trades, WR {wr:.1%}, avg ${avg:.2f}, total ${total:.2f}")

# ── Feature distributions: accepted vs rejected ─────────────────────
log.info(f"\n  Feature distributions (mean ± std):")
key_features = [
    "f_risk_points", "f_ema_distance_pct", "f_price_ema_slope",
    "f_regime_trend_direction", "f_vol_relative", "f_regime_vol_trend",
    "f_price_ret_12bar", "f_range_size_vs_atr", "f_direction_long",
]
log.info(f"  {'Feature':<28} {'Accepted':>20} {'Rejected':>20}")
for f in key_features:
    a_m, a_s = accepted[f].mean(), accepted[f].std()
    r_m, r_s = rejected[f].mean(), rejected[f].std()
    log.info(f"  {f:<28} {a_m:>8.4f} ± {a_s:<8.4f} {r_m:>8.4f} ± {r_s:<8.4f}")

# ── Save ─────────────────────────────────────────────────────────────
summary = {
    "threshold": THRESH,
    "accepted": group_stats(accepted, "accepted"),
    "rejected": group_stats(rejected, "rejected"),
}
out_path = OUT_DIR / "accepted_vs_rejected.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
log.info(f"\nSaved to {out_path}")
