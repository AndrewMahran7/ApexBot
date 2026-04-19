"""
Baseline ML model — time-based split, no hyperparameter tuning.
Train: 2021-2023, Test: 2024.
Models: Logistic Regression + small Decision Tree (depth=3).
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/baseline_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
log.info("Loading corrected EMA candidates dataset")
df = pd.read_csv("data/ema_candidates.csv", parse_dates=["session_date"])
df["year"] = df["session_date"].dt.year
log.info(f"Total samples: {len(df)}")

# ── Feature selection ────────────────────────────────────────────────
feat_cols = [c for c in df.columns if c.startswith("f_")]

# Drop constant features (zero variance)
constant = [c for c in feat_cols if df[c].nunique() <= 1]
if constant:
    log.info(f"Dropping {len(constant)} constant features: {constant}")
    feat_cols = [c for c in feat_cols if c not in constant]

log.info(f"Using {len(feat_cols)} features")

# ── Time-based split ────────────────────────────────────────────────
train_df = df[df["year"] <= 2023].copy()
test_df  = df[df["year"] == 2024].copy()

X_train = train_df[feat_cols].values
y_train = train_df["label_success"].values
X_test  = test_df[feat_cols].values
y_test  = test_df["label_success"].values

log.info(f"Train: {len(X_train)} samples (2021-2023), success rate {y_train.mean():.3f}")
log.info(f"Test:  {len(X_test)} samples (2024),      success rate {y_test.mean():.3f}")

# ── Standardize (for logistic regression) ────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Models ───────────────────────────────────────────────────────────
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=42, solver="lbfgs"
    ),
    "decision_tree_d3": DecisionTreeClassifier(
        max_depth=3, random_state=42
    ),
}

results = {}

for name, model in models.items():
    log.info(f"\n{'='*60}")
    log.info(f"Training: {name}")
    log.info(f"{'='*60}")

    # Tree doesn't need scaling; logistic does
    Xtr = X_train_sc if "logistic" in name else X_train
    Xte = X_test_sc  if "logistic" in name else X_test

    model.fit(Xtr, y_train)

    # Predictions
    y_prob = model.predict_proba(Xte)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    log.info(f"ROC AUC:      {auc:.4f}")
    log.info(f"Precision:    {prec:.4f}")
    log.info(f"Recall:       {rec:.4f}")
    log.info(f"F1:           {f1:.4f}")
    log.info(f"Brier score:  {brier:.4f}")
    log.info(f"Confusion matrix:\n{cm}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['loss','win'])}")

    # ── Calibration curve ────────────────────────────────────────────
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5, strategy="uniform")
    cal_data = [
        {"bin_mean_predicted": float(pp), "bin_fraction_positive": float(pt)}
        for pp, pt in zip(prob_pred, prob_true)
    ]
    log.info("Calibration (5 bins, uniform):")
    log.info(f"  {'Pred Prob':>10}  {'Actual Rate':>12}  {'Delta':>8}")
    for row in cal_data:
        delta = row["bin_fraction_positive"] - row["bin_mean_predicted"]
        log.info(f"  {row['bin_mean_predicted']:10.3f}  {row['bin_fraction_positive']:12.3f}  {delta:+8.3f}")

    # ── Predicted probability distribution ───────────────────────────
    log.info(f"Predicted probability distribution on test set:")
    log.info(f"  mean={y_prob.mean():.3f}, std={y_prob.std():.3f}, "
             f"min={y_prob.min():.3f}, max={y_prob.max():.3f}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        n_above = (y_prob >= thresh).sum()
        if n_above > 0:
            precision_at = y_test[y_prob >= thresh].mean()
        else:
            precision_at = 0.0
        log.info(f"  threshold >= {thresh}: {n_above} trades, "
                 f"precision {precision_at:.3f}")

    # ── Train-set AUC (sanity check for overfit) ─────────────────────
    y_train_prob = model.predict_proba(Xtr)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_prob)
    log.info(f"Train AUC:    {train_auc:.4f}  (test AUC: {auc:.4f}, gap: {train_auc - auc:+.4f})")

    # ── Feature importances (for tree) ───────────────────────────────
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:10]
        log.info("Top 10 features (Gini importance):")
        for i, idx in enumerate(top_idx):
            if imp[idx] > 0:
                log.info(f"  {i+1}. {feat_cols[idx]}: {imp[idx]:.4f}")
        feat_importance = {feat_cols[i]: float(imp[i]) for i in range(len(feat_cols)) if imp[i] > 0}
    elif hasattr(model, "coef_"):
        coef = model.coef_[0]
        top_idx = np.argsort(np.abs(coef))[::-1][:10]
        log.info("Top 10 features (absolute coefficient, scaled features):")
        for i, idx in enumerate(top_idx):
            log.info(f"  {i+1}. {feat_cols[idx]}: {coef[idx]:+.4f}")
        feat_importance = {feat_cols[i]: float(coef[i]) for i in range(len(feat_cols))}
    else:
        feat_importance = {}

    results[name] = {
        "roc_auc": round(auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "train_auc": round(train_auc, 4),
        "overfit_gap": round(train_auc - auc, 4),
        "confusion_matrix": cm.tolist(),
        "calibration": cal_data,
        "prob_distribution": {
            "mean": round(float(y_prob.mean()), 4),
            "std": round(float(y_prob.std()), 4),
            "min": round(float(y_prob.min()), 4),
            "max": round(float(y_prob.max()), 4),
        },
        "feature_importance": feat_importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_success_rate": round(float(y_train.mean()), 4),
        "test_success_rate": round(float(y_test.mean()), 4),
    }

# ── Save ─────────────────────────────────────────────────────────────
out_path = OUT_DIR / "baseline_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
log.info(f"\nResults saved to {out_path}")

# ── Summary ──────────────────────────────────────────────────────────
log.info("\n" + "="*60)
log.info("SUMMARY")
log.info("="*60)
log.info(f"{'Model':<25} {'AUC':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Brier':>7} {'TrainAUC':>9} {'Gap':>7}")
log.info("-"*80)
for name, r in results.items():
    log.info(f"{name:<25} {r['roc_auc']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} "
             f"{r['f1']:7.4f} {r['brier_score']:7.4f} {r['train_auc']:9.4f} {r['overfit_gap']:+7.4f}")
