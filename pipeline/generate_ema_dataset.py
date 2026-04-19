import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
#!/usr/bin/env python3
"""
CLI: Generate EMA trade candidate dataset for ML training.

Usage:
    python generate_ema_dataset.py --data data/mes_4y.csv
    python generate_ema_dataset.py --data data/mes_4y.csv --rr 2.0 --ema-length 20
    python generate_ema_dataset.py --data data/mes_4y.csv --output data/ema_candidates.csv
"""

import argparse
import logging
import sys

from data.ema_candidates import build_ema_candidate_dataset, EMACandidateConfig


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Generate EMA trade candidate dataset for ML training."
    )
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV or Parquet file")
    parser.add_argument("--output", default="data/ema_candidates.csv",
                        help="Output CSV path (default: data/ema_candidates.csv)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--ema-length", type=int, default=50, help="EMA period (default: 50)")
    parser.add_argument("--rr", type=float, default=1.5, help="Reward:risk ratio (default: 1.5)")
    parser.add_argument("--no-shorts", action="store_true", help="Disable short candidates")
    parser.add_argument("--min-range", type=float, default=0.0,
                        help="Minimum opening range size in points")

    args = parser.parse_args()

    cfg = EMACandidateConfig(
        ema_length=args.ema_length,
        reward_risk=args.rr,
        allow_shorts=not args.no_shorts,
        min_range_points=args.min_range,
    )

    candidates = build_ema_candidate_dataset(
        input_path=args.data,
        cfg=cfg,
        start=args.start,
        end=args.end,
        save_path=args.output,
    )

    if candidates.empty:
        print("No candidates generated. Check data and parameters.")
        sys.exit(1)

    print(f"\nDone. {len(candidates)} candidates saved to {args.output}")


if __name__ == "__main__":
    main()
