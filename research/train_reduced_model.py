"""
Feature-reduced baseline model.

1. Rank features by combining:
   - Stability score (cross-year MNQ analysis)
   - Avg rank from MES feature analysis (correlation, MI, t-test)
2. Keep top 50% of features
3. Retrain logistic regression (same params, time-based split)
4. Compare to full-feature baseline
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, brier_score_loss,
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/baseline_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load feature rankings from both analyses ─────────────────────────
log.info("Loading feature rankings ...")

# Source 1: Feature stability (MNQ cross-year)
stab = pd.read_csv("results/feature_stability/feature_stability.csv")
stab = stab[["feature", "stability_score", "mean_corr", "sign_consistency"]].copy()
stab["stab_rank"] = stab["stability_score"].rank(ascending=False)

# Source 2: Feature analysis (MES corrected dataset)
rank = pd.read_csv("results/feature_analysis/feature_ranking.csv")
rank = rank[["feature", "avg_rank", "pb_corr", "pb_pval"]].copy()
rank["analysis_rank"] = rank["avg_rank"].rank(ascending=True)  # lower avg_rank = better

# Merge
merged = stab.merge(rank, on="feature", how="inner")
# Combined score: average of both ranks (lower = better)
merged["combined_rank"] = (merged["stab_rank"] + merged["analysis_rank"]) / 2
merged = merged.sort_values("combined_rank")

n_features = len(merged)
n_keep = n_features // 2  # top 50%
top_features = merged.head(n_keep)["feature"].tolist()
bottom_features = merged.tail(n_features - n_keep)["feature"].tolist()

log.info(f"Total features: {n_features}")
log.info(f"Keeping top {n_keep}: {top_features}")
log.info(f"Dropping bottom {n_features - n_keep}: {bottom_features}")

# Print ranking table
log.info(f"\n{'Feature':<30} {'StabScore':>10} {'StabRank':>9} {'AnalRank':>9} "
         f"{'Combined':>9} {'Keep':>5}")
log.info("-" * 80)
for _, row in merged.iterrows():
    keep = "YES" if row["feature"] in top_features else "no"
    log.info(f"{row['feature']:<30} {row['stability_score']:>10.3f} "
             f"{row['stab_rank']:>9.0f} {row['analysis_rank']:>9.0f} "
             f"{row['combined_rank']:>9.1f} {keep:>5}")

# ── Load data ────────────────────────────────────────────────────────
df = pd.read_csv("data/ema_candidates.csv", parse_dates=["session_date"])
df["year"] = df["session_date"].dt.year

train_df = df[df["year"] <= 2023].copy()
test_df  = df[df["year"] == 2024].copy()

# ── Train both models (full vs reduced) ──────────────────────────────
all_feat = [c for c in df.columns if c.startswith("f_") and df[c].nunique() > 1]

configs = {
    "full_33_features": all_feat,
    f"reduced_{n_keep}_features": top_features,
}

results = {}

for label, feat_cols in configs.items():
    log.info(f"\n{'='*60}")
    log.info(f"Training: {label} ({len(feat_cols)} features)")
    log.info(f"{'='*60}")

    X_train = train_df[feat_cols].values
    y_train = train_df["label_success"].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df["label_success"].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(X_train_sc, y_train)

    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Train predictions for overfit check
    y_train_prob = model.predict_proba(X_train_sc)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_prob)

    auc   = roc_auc_score(y_test, y_prob)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_prob)

    log.info(f"ROC AUC:      {auc:.4f}")
    log.info(f"Precision:    {prec:.4f}")
    log.info(f"Recall:       {rec:.4f}")
    log.info(f"F1:           {f1:.4f}")
    log.info(f"Brier:        {brier:.4f}")
    log.info(f"Train AUC:    {train_auc:.4f} (gap: {train_auc - auc:+.4f})")

    log.info(f"\n{classification_report(y_test, y_pred, target_names=['loss','win'])}")

    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5, strategy="uniform")
    cal_data = [{"pred": round(float(pp), 3), "actual": round(float(pt), 3)}
                for pp, pt in zip(prob_pred, prob_true)]
    log.info("Calibration (5 bins):")
    for c in cal_data:
        log.info(f"  pred {c['pred']:.3f} → actual {c['actual']:.3f} "
                 f"(Δ {c['actual'] - c['pred']:+.3f})")

    # Probability distribution
    log.info(f"Prob dist: mean={y_prob.mean():.3f}, std={y_prob.std():.3f}, "
             f"min={y_prob.min():.3f}, max={y_prob.max():.3f}")

    # Threshold sweep
    log.info(f"\n  {'Thresh':>7} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10}")
    pnl_pts = test_df["label_pnl_pts"].values
    pnl_dollars = pnl_pts * 5.0 - 0.25 * 5.0 - 2.32 * 2  # MES: $5/pt

    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = y_prob >= thresh
        n = mask.sum()
        if n == 0:
            continue
        sel_pnl = pnl_dollars[mask]
        sel_wr  = y_test[mask].mean()
        log.info(f"  {thresh:7.2f} {n:7d} {sel_wr:7.3f} ${sel_pnl.mean():8.2f} "
                 f"${sel_pnl.sum():9.2f}")

    # Feature coefficients
    coef = model.coef_[0]
    top_idx = np.argsort(np.abs(coef))[::-1][:10]
    log.info(f"\nTop features (|coef|):")
    for i, idx in enumerate(top_idx):
        log.info(f"  {i+1}. {feat_cols[idx]}: {coef[idx]:+.4f}")

    results[label] = {
        "n_features": len(feat_cols),
        "features": feat_cols,
        "roc_auc": round(auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "train_auc": round(train_auc, 4),
        "overfit_gap": round(train_auc - auc, 4),
        "calibration": cal_data,
    }

# ── Comparison ───────────────────────────────────────────────────────
log.info(f"\n{'='*70}")
log.info("COMPARISON: FULL vs REDUCED FEATURES")
log.info(f"{'='*70}")
log.info(f"{'Metric':<20} {'Full (33)':>12} {'Reduced ({n_keep})':>12} {'Delta':>10}")
log.info("-" * 56)

full = results["full_33_features"]
red  = results[f"reduced_{n_keep}_features"]

for metric in ["roc_auc", "precision", "recall", "f1", "brier_score", "train_auc", "overfit_gap"]:
    v_f = full[metric]
    v_r = red[metric]
    delta = v_r - v_f
    better = "↑" if (delta > 0 and metric != "brier_score" and metric != "overfit_gap") \
             else ("↑" if delta < 0 and metric in ("brier_score", "overfit_gap") else "↓")
    if abs(delta) < 0.0001:
        better = "="
    log.info(f"{metric:<20} {v_f:>12.4f} {v_r:>12.4f} {delta:>+9.4f} {better}")

# ── Save ─────────────────────────────────────────────────────────────
out = {
    "feature_ranking": merged[["feature", "stability_score", "stab_rank",
                                "analysis_rank", "combined_rank"]].to_dict("records"),
    "top_features": top_features,
    "dropped_features": bottom_features,
    "results": results,
}
out_path = OUT_DIR / "reduced_features_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
log.info(f"\nSaved to {out_path}")
