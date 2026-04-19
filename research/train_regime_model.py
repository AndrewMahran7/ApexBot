"""
Retrain with reduced features + new regime features.

Compares three models (same logistic regression, same time split):
  1. Full original (33 features)
  2. Reduced (16 features — top 50% from prior analysis)
  3. Reduced + regime (16 + 6 new regime features)

New regime features:
  - f_regime_atr_ratio: current ATR / rolling median ATR
  - f_regime_atr_percentile: ATR position in recent range (0-1)
  - f_regime_trend_percentile: EMA slope magnitude position (0-1)
  - f_streak_raw: consecutive wins (+) or losses (-) entering this trade
  - f_recent_wr_5: rolling win rate over last 5 trades
  - f_recent_wr_10: rolling win rate over last 10 trades
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

# ── Load data ────────────────────────────────────────────────────────
df = pd.read_csv("data/ema_candidates.csv", parse_dates=["session_date"])
df["year"] = df["session_date"].dt.year

# Fill NaN in rolling win rate features (first row has no history)
df["f_recent_wr_5"] = df["f_recent_wr_5"].fillna(0.5)
df["f_recent_wr_10"] = df["f_recent_wr_10"].fillna(0.5)

train_df = df[df["year"] <= 2023].copy()
test_df  = df[df["year"] == 2024].copy()
y_train  = train_df["label_success"].values
y_test   = test_df["label_success"].values
pnl_pts  = test_df["label_pnl_pts"].values
pnl_dollars = pnl_pts * 5.0 - 0.25 * 5.0 - 2.32 * 2

log.info(f"Train: {len(train_df)} (2021-2023), Test: {len(test_df)} (2024)")
log.info(f"Train WR: {y_train.mean():.3f}, Test WR: {y_test.mean():.3f}")

# ── Feature sets ─────────────────────────────────────────────────────
# All features (excluding constants)
all_feat = [c for c in df.columns if c.startswith("f_") and df[c].nunique() > 1]

# Top 16 from prior analysis
reduced_16 = [
    "f_risk_points", "f_regime_vol_trend", "f_ema_distance_pct",
    "f_price_ret_12bar", "f_vol_relative", "f_price_ema_dist",
    "f_price_ema_dist_pct", "f_ema_distance", "f_range_high",
    "f_regime_trend_direction", "f_price_ema", "f_range_size_vs_atr",
    "f_price_ema_slope", "f_direction_long", "f_range_vs_atr", "f_range_low",
]

# New regime features
new_regime = [
    "f_regime_atr_ratio", "f_regime_atr_percentile",
    "f_regime_trend_percentile",
    "f_streak_raw", "f_recent_wr_5", "f_recent_wr_10",
]

configs = {
    f"full_{len(all_feat)}_features": all_feat,
    f"reduced_16_features": reduced_16,
    f"reduced_16_plus_regime": reduced_16 + new_regime,
}

# ── Train and evaluate ───────────────────────────────────────────────
results = {}

for label, feat_cols in configs.items():
    log.info(f"\n{'='*60}")
    log.info(f"{label} ({len(feat_cols)} features)")
    log.info(f"{'='*60}")

    X_train = train_df[feat_cols].values
    X_test  = test_df[feat_cols].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(X_train_sc, y_train)

    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    y_train_prob = model.predict_proba(X_train_sc)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_prob)
    auc   = roc_auc_score(y_test, y_prob)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_prob)

    log.info(f"ROC AUC:    {auc:.4f}  (train: {train_auc:.4f}, gap: {train_auc - auc:+.4f})")
    log.info(f"Precision:  {prec:.4f}")
    log.info(f"Recall:     {rec:.4f}")
    log.info(f"F1:         {f1:.4f}")
    log.info(f"Brier:      {brier:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['loss','win'])}")

    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5, strategy="uniform")
    log.info("Calibration:")
    for pp, pt in zip(prob_pred, prob_true):
        log.info(f"  pred {pp:.3f} → actual {pt:.3f} (Δ {pt - pp:+.3f})")

    # Threshold sweep with PnL
    log.info(f"\n  {'Thresh':>7} {'N':>5} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'PF':>7}")
    best_pnl = -999999
    best_thresh = 0
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = y_prob >= thresh
        n = mask.sum()
        if n < 3:
            continue
        sel = pnl_dollars[mask]
        wr = y_test[mask].mean()
        gp = sel[sel > 0].sum()
        gl = abs(sel[sel < 0].sum())
        pf = gp / gl if gl > 0 else float("inf")
        log.info(f"  {thresh:7.2f} {n:5d} {wr:7.3f} ${sel.mean():8.2f} "
                 f"${sel.sum():9.2f} {pf:7.2f}")
        if sel.sum() > best_pnl:
            best_pnl = sel.sum()
            best_thresh = thresh

    # Feature coefficients
    coef = model.coef_[0]
    top_idx = np.argsort(np.abs(coef))[::-1][:10]
    log.info(f"\nTop features:")
    for i, idx in enumerate(top_idx):
        log.info(f"  {i+1}. {feat_cols[idx]}: {coef[idx]:+.4f}")

    results[label] = {
        "n_features": len(feat_cols),
        "roc_auc": round(auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "train_auc": round(train_auc, 4),
        "overfit_gap": round(train_auc - auc, 4),
        "best_threshold": best_thresh,
        "best_total_pnl": round(best_pnl, 2),
    }

# ── Comparison table ─────────────────────────────────────────────────
log.info(f"\n{'='*90}")
log.info("COMPARISON")
log.info(f"{'='*90}")
log.info(f"{'Model':<28} {'#Feat':>5} {'AUC':>7} {'Prec':>7} {'Recall':>7} "
         f"{'F1':>7} {'Brier':>7} {'TrAUC':>7} {'Gap':>7}")
log.info("-" * 90)
for name, r in results.items():
    log.info(f"{name:<28} {r['n_features']:>5} {r['roc_auc']:>7.4f} "
             f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
             f"{r['brier_score']:>7.4f} {r['train_auc']:>7.4f} {r['overfit_gap']:>+7.4f}")

# ── Save ─────────────────────────────────────────────────────────────
out_path = OUT_DIR / "regime_features_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
log.info(f"\nSaved to {out_path}")
