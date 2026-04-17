#!/usr/bin/env python3
"""
CLI: Train ML model on EMA trade candidates.

Usage:
    python train_model.py
    python train_model.py --data data/ema_candidates.csv --model gradient_boosting
    python train_model.py --model logistic
"""

import argparse
import logging
import sys

import pandas as pd

from models.train_model import train_model, TrainConfig


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train ML model for EMA trade filtering."
    )
    parser.add_argument("--data", default="data/ema_candidates.csv",
                        help="Path to EMA candidates CSV (default: data/ema_candidates.csv)")
    parser.add_argument("--model", default="gradient_boosting",
                        choices=["logistic", "gradient_boosting"],
                        help="Model type (default: gradient_boosting)")
    parser.add_argument("--output-dir", default="models",
                        help="Output directory for model artifacts (default: models/)")

    # Gradient Boosting hyperparams
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of trees (default: 200)")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Max tree depth (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=0.05,
                        help="Learning rate (default: 0.05)")
    parser.add_argument("--min-samples-leaf", type=int, default=20,
                        help="Min samples per leaf (default: 20)")

    # Split params
    parser.add_argument("--train-pct", type=float, default=0.60)
    parser.add_argument("--val-pct", type=float, default=0.20)
    parser.add_argument("--test-pct", type=float, default=0.10)
    parser.add_argument("--holdout-pct", type=float, default=0.10)

    args = parser.parse_args()

    print(f"Loading candidates from {args.data} ...")
    try:
        candidates = pd.read_csv(args.data, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: {args.data} not found. Run generate_ema_dataset.py first.")
        sys.exit(1)

    print(f"Loaded {len(candidates):,} candidates")

    cfg = TrainConfig(
        model_type=args.model,
        output_dir=args.output_dir,
        gb_n_estimators=args.n_estimators,
        gb_max_depth=args.max_depth,
        gb_learning_rate=args.learning_rate,
        gb_min_samples_leaf=args.min_samples_leaf,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        holdout_pct=args.holdout_pct,
    )

    artifacts = train_model(candidates, cfg)

    print(f"\nTraining complete.")
    print(f"  Model type:    {cfg.model_type}")
    print(f"  Train acc:     {artifacts['train_metrics']['train_accuracy']:.4f}")
    print(f"  Val acc:       {artifacts['train_metrics']['val_accuracy']:.4f}")
    print(f"  Features used: {artifacts['train_metrics']['n_features']}")


if __name__ == "__main__":
    main()
