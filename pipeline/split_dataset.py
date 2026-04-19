import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
#!/usr/bin/env python
"""
CLI: split an existing research dataset into train/val/test/holdout.
=====================================================================

Usage:
    python split_dataset.py --input data/features/mes_trade_candidates.parquet
    python split_dataset.py --input data/features/mes_features.parquet --train 70 --val 15 --test 10 --holdout 5
    python split_dataset.py --input data/features/mes_trade_candidates.parquet --purge-days 2
"""

import argparse
import sys

from data.splits import time_based_split, save_splits, validate_no_leakage, print_split_summary, SplitConfig


def main():
    parser = argparse.ArgumentParser(
        description="Split research dataset into time-based train/val/test/holdout."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to parquet dataset (e.g. data/features/mes_trade_candidates.parquet)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for split outputs (default: same dir as input + /splits/)",
    )
    parser.add_argument(
        "--train", type=float, default=60.0,
        help="Train set percentage (default: 60)",
    )
    parser.add_argument(
        "--val", type=float, default=20.0,
        help="Validation set percentage (default: 20)",
    )
    parser.add_argument(
        "--test", type=float, default=10.0,
        help="Test set percentage (default: 10)",
    )
    parser.add_argument(
        "--holdout", type=float, default=10.0,
        help="Holdout set percentage (default: 10)",
    )
    parser.add_argument(
        "--purge-days", type=int, default=1,
        help="Gap days between splits (default: 1)",
    )

    args = parser.parse_args()

    total = args.train + args.val + args.test + args.holdout
    if abs(total - 100.0) > 0.01:
        print(f"[ERROR] Split percentages must sum to 100, got {total}")
        sys.exit(1)

    import pandas as pd
    from pathlib import Path

    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows loaded")

    # Determine session_date column
    if "session_date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["session_date"] = df.index.date
        else:
            print("[ERROR] Cannot determine session dates. Need 'session_date' column or DatetimeIndex.")
            sys.exit(1)

    cfg = SplitConfig(
        train_pct=args.train / 100.0,
        val_pct=args.val / 100.0,
        test_pct=args.test / 100.0,
        holdout_pct=args.holdout / 100.0,
        purge_days=args.purge_days,
    )

    result = time_based_split(df, cfg)

    # Validate
    is_clean = validate_no_leakage(result)
    if not is_clean:
        print("[WARN] Data leakage detected between splits!")

    print_split_summary(result)

    # Save
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = Path(args.input).parent / "splits"

    save_splits(result, str(out))
    print(f"\nSplits saved to {out}/")

    sys.exit(0)


if __name__ == "__main__":
    main()
