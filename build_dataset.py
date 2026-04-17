#!/usr/bin/env python
"""
CLI: build the full research dataset.
======================================

Usage:
    python build_dataset.py --input data/mes_5m.csv
    python build_dataset.py --input data/mes_5m.csv --start 2022-01-01 --end 2024-01-01
    python build_dataset.py --input data/mes_5m.csv --output-dir data --min-range 2.0
"""

import argparse
import logging
import sys

from data.build_dataset import build_dataset, DatasetConfig


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Build MES research dataset (features + labels + trade candidates)."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to raw OHLCV CSV/parquet (e.g. data/mes_5m.csv)",
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Base directory for outputs (default: data/)",
    )
    parser.add_argument(
        "--min-range", type=float, default=0.0,
        help="Minimum opening-range size in points (default: 0 = no filter)",
    )
    parser.add_argument(
        "--range-end", default="09:45",
        help="Opening range end time HH:MM (default: 09:45)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Compute without saving to disk",
    )

    args = parser.parse_args()

    cfg = DatasetConfig(
        output_dir=args.output_dir,
        min_range_points=args.min_range,
        range_end=args.range_end,
    )

    result = build_dataset(
        input_path=args.input,
        cfg=cfg,
        start=args.start,
        end=args.end,
        save=not args.no_save,
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
