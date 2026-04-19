#!/usr/bin/env python3
"""
Walk-Forward Evaluation for EMA + ML Strategy
===============================================

Expanding-window walk-forward: trains on progressively larger windows,
tests on the next unseen fold. This is the gold standard for temporal
ML validation — it simulates what would happen if you retrained
periodically in production.

Usage:
    python walk_forward.py --data data/mes_4y.csv
    python walk_forward.py --data data/mes_4y.csv --folds 5 --min-train-days 200
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from data.ema_candidates import (
    build_ema_candidate_dataset,
    EMACandidateConfig,
    get_feature_columns,
)
from models.train_model import TrainConfig


def walk_forward_evaluate(
    candidates: pd.DataFrame,
    n_folds: int = 5,
    min_train_days: int = 200,
    purge_days: int = 1,
    cfg: TrainConfig | None = None,
) -> dict:
    """
    Expanding-window walk-forward evaluation.

    Splits the data into n_folds contiguous test folds. For each fold,
    trains on all preceding data and evaluates on the fold.

    Returns dict with per-fold and aggregate metrics.
    """
    if cfg is None:
        cfg = TrainConfig()

    label_col = "label_success"
    feature_cols = get_feature_columns(candidates)
    clean = candidates.dropna(subset=[label_col])

    if "session_date" in clean.columns:
        dates = sorted(clean["session_date"].unique())
    else:
        dates = sorted(set(clean.index.date))

    n_dates = len(dates)
    if n_dates < min_train_days + n_folds * 10:
        raise ValueError(
            f"Not enough dates ({n_dates}) for {n_folds}-fold walk-forward "
            f"with min_train={min_train_days}"
        )

    # Compute fold boundaries
    test_dates_per_fold = (n_dates - min_train_days) // n_folds
    if test_dates_per_fold < 5:
        raise ValueError(
            f"Only {test_dates_per_fold} test dates per fold — increase data or reduce folds"
        )

    fold_results = []
    all_oos_preds = []
    all_oos_true = []

    logger.info("Walk-Forward: %d folds, %d total dates, min_train=%dd, purge=%dd",
                n_folds, n_dates, min_train_days, purge_days)
    print(f"\nWalk-Forward: {n_folds} folds, {n_dates} total dates, "
          f"min_train={min_train_days}d, purge={purge_days}d\n")
    print(f"{'Fold':>5s}  {'Train':>12s}  {'Test':>12s}  {'#Train':>7s}  {'#Test':>6s}  "
          f"{'Acc':>6s}  {'AUC':>6s}  {'Win%':>6s}")
    print("-" * 75)

    for fold_idx in range(n_folds):
        # Expanding train window
        train_end_idx = min_train_days + fold_idx * test_dates_per_fold
        test_start_idx = train_end_idx + purge_days
        test_end_idx = test_start_idx + test_dates_per_fold

        if test_end_idx > n_dates:
            test_end_idx = n_dates

        train_dates_set = set(dates[:train_end_idx])
        test_dates_set = set(dates[test_start_idx:test_end_idx])

        if not test_dates_set:
            break

        if "session_date" in clean.columns:
            sd = clean["session_date"]
        else:
            sd = pd.Series(clean.index.date, index=clean.index)

        train_df = clean[sd.isin(train_dates_set)].copy()
        test_df = clean[sd.isin(test_dates_set)].copy()

        if len(train_df) < 50 or len(test_df) < 10:
            continue

        # Fill NaN with train-only medians
        train_medians = {}
        for col in feature_cols:
            if col in train_df.columns:
                train_medians[col] = train_df[col].median()
        for col in feature_cols:
            if col in train_df.columns:
                train_df[col] = train_df[col].fillna(train_medians.get(col, 0.0))
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(train_medians.get(col, 0.0))

        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[label_col].values

        # Train
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=cfg.gb_n_estimators,
                max_depth=cfg.gb_max_depth,
                learning_rate=cfg.gb_learning_rate,
                subsample=cfg.gb_subsample,
                min_samples_leaf=cfg.gb_min_samples_leaf,
                random_state=42,
            )),
        ])
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.0
            logger.warning("Fold %d: AUC set to 0.0 — only one class in test set", fold_idx + 1)

        win_rate = float(y_test.mean())

        train_start_d = sorted(train_dates_set)[0]
        train_end_d = sorted(train_dates_set)[-1]
        test_start_d = sorted(test_dates_set)[0]
        test_end_d = sorted(test_dates_set)[-1]

        fold_info = {
            "fold": fold_idx + 1,
            "train_period": f"{train_start_d} → {train_end_d}",
            "test_period": f"{test_start_d} → {test_end_d}",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4),
            "test_positive_rate": round(win_rate, 4),
        }
        fold_results.append(fold_info)

        all_oos_preds.extend(y_proba.tolist())
        all_oos_true.extend(y_test.tolist())

        print(f"{fold_idx+1:>5d}  {str(train_start_d)[:10]}→{str(train_end_d)[:10]}  "
              f"{str(test_start_d)[:10]}→{str(test_end_d)[:10]}  "
              f"{len(X_train):>7d}  {len(X_test):>6d}  "
              f"{acc:>6.3f}  {auc:>6.3f}  {win_rate:>5.1%}")

    # Aggregate
    if not fold_results:
        logger.warning("No valid folds produced")
        return {}

    all_oos_preds = np.array(all_oos_preds)
    all_oos_true = np.array(all_oos_true)

    try:
        agg_auc = roc_auc_score(all_oos_true, all_oos_preds)
    except ValueError:
        agg_auc = 0.0
        logger.warning("Aggregate AUC set to 0.0 — only one class in pooled OOS")
    agg_acc = accuracy_score(all_oos_true, (all_oos_preds >= 0.5).astype(int))

    print("-" * 75)
    print(f"{'AGG':>5s}  {'':>12s}  {'':>12s}  {'':>7s}  {len(all_oos_true):>6d}  "
          f"{agg_acc:>6.3f}  {agg_auc:>6.3f}  {all_oos_true.mean():>5.1%}")

    # Threshold analysis on pooled OOS predictions
    print(f"\nPooled OOS Threshold Analysis:")
    print(f"  {'Thresh':>7s}  {'Trades':>7s}  {'WinRate':>8s}  {'Filtered%':>10s}")
    for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        mask = all_oos_preds >= t
        n = int(mask.sum())
        wr = float(all_oos_true[mask].mean()) if n > 0 else 0.0
        filt = 1.0 - n / len(all_oos_true)
        print(f"  {t:>7.2f}  {n:>7d}  {wr:>7.1%}  {filt:>9.1%}")

    # Ranking simulation on pooled OOS predictions.
    # Simulates the rolling-window ranking used by the strategy:
    # for each candidate in chronological order, compare its prob
    # against the prior `lookback` probs, accept if top-N or top-pct.
    print(f"\nPooled OOS Ranking Simulation (lookback=20):")
    _ranking_lookback = 20
    for mode_label, top_n, top_pct in [
        ("top_n=1", 1, None), ("top_n=2", 2, None), ("top_n=3", 3, None),
        ("top_n=5", 5, None), ("top_n=7", 7, None),
        ("top_pct=20%", None, 0.20), ("top_pct=30%", None, 0.30),
        ("top_pct=40%", None, 0.40), ("top_pct=50%", None, 0.50),
    ]:
        window = []
        accepted_mask = np.zeros(len(all_oos_preds), dtype=bool)
        for i, prob in enumerate(all_oos_preds):
            if len(window) < (top_n if top_n else 5):
                accepted_mask[i] = True  # cold-start grace
            elif top_n is not None:
                n_higher = sum(1 for p in window[-_ranking_lookback:] if p > prob)
                accepted_mask[i] = (n_higher + 1) <= top_n
            else:
                cutoff = float(np.percentile(
                    window[-_ranking_lookback:], (1.0 - top_pct) * 100
                ))
                accepted_mask[i] = prob >= cutoff
            window.append(prob)

        n = int(accepted_mask.sum())
        wr = float(all_oos_true[accepted_mask].mean()) if n > 0 else 0.0
        filt = 1.0 - n / len(all_oos_true)
        print(f"  {mode_label:>12s}  {n:>7d}  {wr:>7.1%}  {filt:>9.1%}")

    summary = {
        "n_folds": len(fold_results),
        "total_oos_samples": len(all_oos_true),
        "aggregate_accuracy": round(agg_acc, 4),
        "aggregate_roc_auc": round(agg_auc, 4),
        "aggregate_positive_rate": round(float(all_oos_true.mean()), 4),
        "mean_fold_auc": round(np.mean([f["roc_auc"] for f in fold_results]), 4),
        "std_fold_auc": round(np.std([f["roc_auc"] for f in fold_results]), 4),
        "folds": fold_results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation for EMA+ML")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV or Parquet")
    parser.add_argument("--candidates", default=None,
                        help="Pre-built candidates CSV (skip generation)")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--min-train-days", type=int, default=200,
                        help="Minimum training days (default: 200)")
    parser.add_argument("--purge-days", type=int, default=1,
                        help="Purge gap in days between train/test (default: 1)")
    parser.add_argument("--output", default="models/walk_forward_results.json",
                        help="Output JSON path")
    parser.add_argument("--ema-length", type=int, default=50)
    parser.add_argument("--rr", type=float, default=1.5)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Build or load candidates
    if args.candidates:
        print(f"Loading candidates from {args.candidates} ...")
        candidates = pd.read_csv(args.candidates, index_col=0, parse_dates=True)
    else:
        print(f"Generating EMA candidates from {args.data} ...")
        cfg = EMACandidateConfig(ema_length=args.ema_length, reward_risk=args.rr)
        candidates = build_ema_candidate_dataset(args.data, cfg=cfg)

    if candidates.empty:
        logger.error("No candidates generated — exiting")
        sys.exit(1)

    print(f"Candidates: {len(candidates)}")

    train_cfg = TrainConfig()
    results = walk_forward_evaluate(
        candidates,
        n_folds=args.folds,
        min_train_days=args.min_train_days,
        purge_days=args.purge_days,
        cfg=train_cfg,
    )

    if results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out_path), "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
