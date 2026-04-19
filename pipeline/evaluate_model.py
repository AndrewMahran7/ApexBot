import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
#!/usr/bin/env python3
"""
CLI: Evaluate trained ML model on validation and test sets.

Usage:
    python evaluate_model.py
    python evaluate_model.py --data data/ema_candidates.csv --model models/ema_model.pkl
"""

import argparse
import logging
import sys

import pandas as pd

from models.train_model import load_model, TrainConfig
from models.evaluate_model import evaluate_model
from data.ema_candidates import get_feature_columns
from data.splits import time_based_split, SplitConfig


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Evaluate ML model on validation and test sets."
    )
    parser.add_argument("--data", default="data/ema_candidates.csv",
                        help="Path to EMA candidates CSV (default: data/ema_candidates.csv)")
    parser.add_argument("--model", default="models/ema_model.pkl",
                        help="Path to trained model (default: models/ema_model.pkl)")
    parser.add_argument("--output-dir", default="models",
                        help="Output directory for evaluation results (default: models/)")

    # Split params (must match training)
    parser.add_argument("--train-pct", type=float, default=0.60)
    parser.add_argument("--val-pct", type=float, default=0.20)
    parser.add_argument("--test-pct", type=float, default=0.10)
    parser.add_argument("--holdout-pct", type=float, default=0.10)

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model} ...")
    try:
        model_data = load_model(args.model)
    except FileNotFoundError:
        print(f"Error: {args.model} not found. Run train_model.py first.")
        sys.exit(1)

    print(f"  Model type: {model_data['model_type']}")
    print(f"  Features:   {len(model_data['feature_columns'])}")

    # Load candidates
    print(f"Loading candidates from {args.data} ...")
    try:
        candidates = pd.read_csv(args.data, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: {args.data} not found. Run generate_ema_dataset.py first.")
        sys.exit(1)

    print(f"Loaded {len(candidates):,} candidates")

    # Clean data (same as training — use train-only medians if available)
    label_col = "label_success"
    clean = candidates.dropna(subset=[label_col])
    feature_cols = model_data["feature_columns"]

    # Split (same config as training for consistency)
    split_cfg = SplitConfig(
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        holdout_pct=args.holdout_pct,
        purge_days=1,
    )
    split_result = time_based_split(clean, split_cfg)

    # Fill NaN features with TRAIN-ONLY medians (no leakage)
    train_medians = model_data.get("train_medians", {})
    if not train_medians:
        # Fallback: compute from train split
        print("  (computing train medians from split — model pkl has no saved medians)")
        for col in feature_cols:
            if col in split_result.train.columns:
                train_medians[col] = split_result.train[col].median()
    for split_name in ("train", "validation", "test", "holdout"):
        split_df = getattr(split_result, split_name)
        for col in feature_cols:
            if col in split_df.columns:
                split_df[col] = split_df[col].fillna(train_medians.get(col, 0.0))

    # Evaluate
    results = evaluate_model(
        model_data=model_data,
        val_df=split_result.validation,
        test_df=split_result.test,
        output_dir=args.output_dir,
        save=True,
        train_medians=train_medians,
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
