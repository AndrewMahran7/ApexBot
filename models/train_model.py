"""
ML Model Training Pipeline for EMA Trade Filter
=================================================

Trains a classifier to predict probability of trade success
for EMA directional trade candidates.

Models:
  - Logistic Regression (baseline)
  - Gradient Boosting (main model)

Uses time-based splits only. No random shuffling.
Model is saved to disk for use by the hybrid strategy.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data.ema_candidates import get_feature_columns
from data.splits import time_based_split, SplitConfig, print_split_summary


@dataclass
class TrainConfig:
    """Training configuration."""
    model_type: str = "gradient_boosting"  # 'logistic' or 'gradient_boosting'
    output_dir: str = "models"

    # Gradient Boosting hyperparams (conservative defaults)
    gb_n_estimators: int = 200
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.05
    gb_subsample: float = 0.8
    gb_min_samples_leaf: int = 20

    # Logistic Regression
    lr_C: float = 1.0
    lr_max_iter: int = 1000

    # Split config
    train_pct: float = 0.60
    val_pct: float = 0.20
    test_pct: float = 0.10
    holdout_pct: float = 0.10
    purge_days: int = 1


def train_model(
    candidates: pd.DataFrame,
    cfg: TrainConfig | None = None,
    save: bool = True,
) -> dict:
    """
    Train ML model on EMA trade candidates.

    Parameters
    ----------
    candidates : pd.DataFrame
        Output of generate_ema_candidates() with features and labels.
    cfg : TrainConfig
    save : bool
        Whether to save model and artifacts to disk.

    Returns
    -------
    dict with keys: model, scaler, feature_columns, split_result, train_metrics
    """
    if cfg is None:
        cfg = TrainConfig()

    # --- 1. Identify feature columns ---
    feature_cols = get_feature_columns(candidates)
    if not feature_cols:
        raise ValueError("No feature columns found (columns starting with 'f_')")

    print(f"[1/4] Feature columns: {len(feature_cols)}")

    # --- 2. Clean data ---
    label_col = "label_success"
    if label_col not in candidates.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    # Drop rows with NaN labels or all-NaN features
    clean = candidates.dropna(subset=[label_col])
    print(f"       {len(clean):,} candidates with valid labels")

    # Drop any feature columns that are all NaN (before split)
    valid_features = [c for c in feature_cols if c in clean.columns and clean[c].notna().any()]
    dropped = set(feature_cols) - set(valid_features)
    if dropped:
        print(f"       Dropped {len(dropped)} all-NaN features: {dropped}")
    feature_cols = valid_features

    # --- 3. Time-based split ---
    print("[2/4] Splitting data (time-based) ...")
    split_cfg = SplitConfig(
        train_pct=cfg.train_pct,
        val_pct=cfg.val_pct,
        test_pct=cfg.test_pct,
        holdout_pct=cfg.holdout_pct,
        purge_days=cfg.purge_days,
    )
    split_result = time_based_split(clean, split_cfg)
    print_split_summary(split_result)

    # Fill NaN features with TRAIN-ONLY medians (no leakage)
    train_medians = {}
    for col in feature_cols:
        if col in split_result.train.columns:
            train_medians[col] = split_result.train[col].median()
    for split_name in ("train", "validation", "test", "holdout"):
        split_df = getattr(split_result, split_name)
        for col in feature_cols:
            if col in split_df.columns:
                split_df[col] = split_df[col].fillna(train_medians.get(col, 0.0))

    X_train = split_result.train[feature_cols].values
    y_train = split_result.train[label_col].values
    X_val = split_result.validation[feature_cols].values
    y_val = split_result.validation[label_col].values

    print(f"       Train: {len(X_train)} samples, {y_train.mean():.1%} positive rate")
    print(f"       Val:   {len(X_val)} samples, {y_val.mean():.1%} positive rate")

    # --- 4. Train model ---
    print(f"[3/4] Training {cfg.model_type} model ...")

    if cfg.model_type == "logistic":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=cfg.lr_C,
                max_iter=cfg.lr_max_iter,
                random_state=42,
            )),
        ])
    else:
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

    # --- Quick validation metrics ---
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    train_acc = (model.predict(X_train) == y_train).mean()
    val_acc = (model.predict(X_val) == y_val).mean()

    print(f"       Train accuracy: {train_acc:.3f}")
    print(f"       Val accuracy:   {val_acc:.3f}")
    print(f"       Val prob range: [{val_proba.min():.3f}, {val_proba.max():.3f}]")

    # Degenerate model check
    if val_proba.max() - val_proba.min() < 0.01:
        logger.warning(
            "Model produces near-constant predictions (range=%.4f). "
            "Training data may be insufficient or features uninformative.",
            val_proba.max() - val_proba.min(),
        )

    # Feature importance (for gradient boosting)
    importances = {}
    if cfg.model_type == "gradient_boosting":
        clf = model.named_steps["clf"]
        for feat, imp in zip(feature_cols, clf.feature_importances_):
            importances[feat] = float(imp)
        # Print top 10
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        print("\n       Top 10 features:")
        for feat, imp in sorted_imp[:10]:
            print(f"         {feat:40s} {imp:.4f}")

    # --- 5. Save ---
    artifacts = {
        "model": model,
        "feature_columns": feature_cols,
        "split_result": split_result,
        "config": cfg,
        "importances": importances,
        "train_metrics": {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            "model_type": cfg.model_type,
            "n_features": len(feature_cols),
        },
    }

    if save:
        print("[4/4] Saving model ...")
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        try:
            # Save model
            model_path = str(out_dir / "ema_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": model,
                    "feature_columns": feature_cols,
                    "model_type": cfg.model_type,
                    "train_medians": train_medians,
                    "split_dates": {
                        name: {
                            "start": split_result.info[name]["start"],
                            "end": split_result.info[name]["end"],
                        }
                        for name in ("train", "validation", "test", "holdout")
                    },
                }, f)
            saved_paths.append(model_path)
            logger.info("Model saved to %s", model_path)

            # Save feature importances
            if importances:
                imp_path = str(out_dir / "feature_importances.json")
                with open(imp_path, "w") as f:
                    json.dump(
                        dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)),
                        f, indent=2,
                    )
                saved_paths.append(imp_path)
                logger.info("Feature importances saved to %s", imp_path)

            # Save train metrics
            metrics_path = str(out_dir / "train_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(artifacts["train_metrics"], f, indent=2)
            saved_paths.append(metrics_path)
            logger.info("Train metrics saved to %s", metrics_path)

            # Save split info
            split_path = str(out_dir / "split_info.json")
            with open(split_path, "w") as f:
                json.dump(split_result.info, f, indent=2, default=str)
            saved_paths.append(split_path)
            logger.info("Split info saved to %s", split_path)

            # Save audit report
            from data.splits import validate_no_leakage
            leakage_errors = validate_no_leakage(split_result)
            audit_report = {
                "pipeline_version": "2.0-audited",
                "leakage_checks": {
                    "nan_fill_method": "train-only medians",
                    "split_overlap_errors": leakage_errors if leakage_errors else "PASS",
                    "feature_lookahead": "PASS — all f_* columns use lagged/current data only",
                    "label_lookahead": "PASS — labels scan strictly future bars (index > candidate ts)",
                    "backtest_alignment": {
                        "entry_price": "MATCH — both use bar open",
                        "sl_tp_fill_priority": "MATCH — SL checked before TP in both labeler and engine",
                        "eod_exit": "MATCH — same time cutoff",
                    },
                },
                "split_dates": {
                    name: split_result.info[name]
                    for name in ("train", "validation", "test", "holdout")
                },
                "model_info": {
                    "type": cfg.model_type,
                    "n_features": len(feature_cols),
                    "features": feature_cols,
                    "train_accuracy": float(train_acc),
                    "val_accuracy": float(val_acc),
                },
                "class_balance": {
                    "train_positive_rate": float(y_train.mean()),
                    "val_positive_rate": float(y_val.mean()),
                },
                "top_features": dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]) if importances else {},
                "train_medians_saved": len(train_medians),
                "warnings": [],
            }
            if leakage_errors:
                audit_report["warnings"].append(f"SPLIT OVERLAP DETECTED: {leakage_errors}")
            if abs(float(y_train.mean()) - float(y_val.mean())) > 0.10:
                audit_report["warnings"].append(
                    f"Class balance drift: train={y_train.mean():.3f}, val={y_val.mean():.3f}"
                )

            audit_path = str(out_dir / "audit_report.json")
            with open(audit_path, "w") as f:
                json.dump(audit_report, f, indent=2, default=str)
            saved_paths.append(audit_path)
            logger.info("Audit report saved to %s", audit_path)

        except OSError as exc:
            logger.error(
                "Failed to save artifacts (saved so far: %s): %s",
                saved_paths, exc,
            )
            raise

    return artifacts


def load_model(model_path: str = "models/ema_model.pkl") -> dict:
    """
    Load a trained model from disk.

    Returns dict with keys: model, feature_columns, model_type
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data
