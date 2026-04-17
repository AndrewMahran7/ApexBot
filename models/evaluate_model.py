"""
ML Model Evaluation for EMA Trade Filter
==========================================

Evaluates a trained model on validation and test sets.
Reports classification metrics and probability distributions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from data.ema_candidates import get_feature_columns
from models.train_model import load_model


def evaluate_model(
    model_data: dict,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "models",
    save: bool = True,
    train_medians: dict | None = None,
) -> dict:
    """
    Evaluate model on validation and test sets.

    Parameters
    ----------
    model_data : dict
        Output of load_model() with keys: model, feature_columns
    val_df : pd.DataFrame
        Validation split with features and labels.
    test_df : pd.DataFrame
        Test split with features and labels.
    output_dir : str
        Where to save evaluation results.
    save : bool

    Returns
    -------
    dict with evaluation metrics for val and test sets.
    """
    model = model_data["model"]
    feature_cols = model_data["feature_columns"]
    label_col = "label_success"

    results = {}

    for name, df in [("validation", val_df), ("test", test_df)]:
        if df.empty:
            logger.warning("%s set is empty — skipping evaluation", name)
            results[name] = {}
            continue

        # Clean: fill NaN features with train medians (no leakage)
        X = df[feature_cols].copy()
        for col in feature_cols:
            if col in X.columns:
                med = train_medians.get(col, X[col].median()) if train_medians else X[col].median()
                X[col] = X[col].fillna(med)

        y_true = df[label_col].values
        X_vals = X.values

        y_pred = model.predict(X_vals)
        y_proba = model.predict_proba(X_vals)[:, 1]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.0
            logger.warning("%s: AUC set to 0.0 — only one class present in labels", name)

        cm = confusion_matrix(y_true, y_pred)

        # Probability distribution stats
        proba_stats = {
            "mean": float(np.mean(y_proba)),
            "std": float(np.std(y_proba)),
            "min": float(np.min(y_proba)),
            "max": float(np.max(y_proba)),
            "median": float(np.median(y_proba)),
            "q25": float(np.percentile(y_proba, 25)),
            "q75": float(np.percentile(y_proba, 75)),
        }

        # Distribution by decile
        decile_dist = {}
        for i in range(10):
            lo = i * 0.1
            hi = (i + 1) * 0.1
            count = int(((y_proba >= lo) & (y_proba < hi)).sum())
            decile_dist[f"{lo:.1f}-{hi:.1f}"] = count

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "roc_auc": round(auc, 4),
            "samples": len(y_true),
            "positive_rate": round(float(y_true.mean()), 4),
            "predicted_positive_rate": round(float(y_pred.mean()), 4),
            "confusion_matrix": cm.tolist(),
            "probability_distribution": proba_stats,
            "probability_deciles": decile_dist,
        }

        results[name] = metrics

        # Print results
        print(f"\n{'=' * 50}")
        print(f"  {name.upper()} SET EVALUATION")
        print(f"{'=' * 50}")
        print(f"  Samples    : {len(y_true)}")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  ROC AUC    : {auc:.4f}")
        print(f"  Positive % : {y_true.mean():.1%}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
        print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
        print(f"\n  Probability Distribution:")
        print(f"    Mean: {proba_stats['mean']:.3f}  Std: {proba_stats['std']:.3f}")
        print(f"    Min:  {proba_stats['min']:.3f}  Max: {proba_stats['max']:.3f}")
        print(f"    Q25:  {proba_stats['q25']:.3f}  Med: {proba_stats['median']:.3f}  Q75: {proba_stats['q75']:.3f}")

        # Probability decile table
        print(f"\n  Probability Deciles:")
        for bucket, count in decile_dist.items():
            bar = "#" * (count // max(1, len(y_true) // 50))
            print(f"    [{bucket}] {count:4d} {bar}")

    # Threshold analysis
    if "validation" in results and results["validation"]:
        print(f"\n{'=' * 50}")
        print(f"  THRESHOLD ANALYSIS (Validation Set)")
        print(f"{'=' * 50}")

        X_val = val_df[feature_cols].copy()
        for col in feature_cols:
            if col in X_val.columns:
                med = train_medians.get(col, X_val[col].median()) if train_medians else X_val[col].median()
                X_val[col] = X_val[col].fillna(med)

        y_val_true = val_df[label_col].values
        y_val_proba = model.predict_proba(X_val.values)[:, 1]

        print(f"  {'Threshold':>10s} {'Trades':>7s} {'WinRate':>8s} {'Precision':>10s} {'Recall':>7s}")
        print(f"  {'-' * 50}")

        threshold_analysis = {}
        for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            mask = y_val_proba >= thresh
            n_trades = int(mask.sum())
            if n_trades > 0:
                win_rate = float(y_val_true[mask].mean())
                pred_pos = mask.astype(int)
                p = precision_score(y_val_true, pred_pos, zero_division=0)
                r = recall_score(y_val_true, pred_pos, zero_division=0)
            else:
                win_rate = 0.0
                p = 0.0
                r = 0.0

            threshold_analysis[str(thresh)] = {
                "trades": n_trades,
                "win_rate": round(win_rate, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
            }
            print(f"  {thresh:>10.2f} {n_trades:>7d} {win_rate:>7.1%} {p:>10.4f} {r:>7.4f}")

        results["threshold_analysis"] = threshold_analysis

    # Save results
    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        eval_path = str(out / "evaluation_results.json")
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {eval_path}")

    return results
