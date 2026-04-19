import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
#!/usr/bin/env python3
"""
CLI entry point for fetching MES historical data from Databento.

Pulls 1-minute OHLCV bars, resamples to the target timeframe (default 5min),
and saves a CSV ready for run_backtest.py.

Requirements:
    pip install databento
    export DATABENTO_API_KEY=db-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Usage:
    python fetch_data.py --start 2024-01-02 --end 2024-06-30
    python fetch_data.py --start 2024-01-02 --end 2024-06-30 --timeframe 15min
    python fetch_data.py --start 2024-01-02 --end 2024-06-30 --save-raw
"""

import argparse
import logging
import sys

from data.databento_fetcher import fetch_and_save, DEFAULT_DATASET, DEFAULT_SYMBOL, DEFAULT_SCHEMA, DEFAULT_STYPE_IN, DEFAULT_TIMEZONE, DEFAULT_RESAMPLE
from data.validate import print_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch MES futures OHLCV data from Databento.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python fetch_data.py --start 2024-01-02 --end 2024-06-30
  python fetch_data.py --start 2024-01-02 --end 2024-06-30 --timeframe 15min
  python fetch_data.py --start 2024-01-02 --end 2024-06-30 --output data/mes_2024_h1.csv
  python fetch_data.py --start 2024-01-02 --end 2024-06-30 --save-raw
""",
    )

    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD, exclusive)")
    p.add_argument("--output", "-o", default="data/mes_5m.csv", help="Output CSV path (default: data/mes_5m.csv)")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL, help=f"Databento symbol (default: {DEFAULT_SYMBOL})")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help=f"Databento dataset (default: {DEFAULT_DATASET})")
    p.add_argument("--schema", default=DEFAULT_SCHEMA, help=f"Databento schema (default: {DEFAULT_SCHEMA})")
    p.add_argument("--stype-in", default=DEFAULT_STYPE_IN, help=f"Symbology type (default: {DEFAULT_STYPE_IN})")
    p.add_argument("--timeframe", default=DEFAULT_RESAMPLE, help=f"Resample timeframe (default: {DEFAULT_RESAMPLE})")
    p.add_argument("--timezone", default=DEFAULT_TIMEZONE, help=f"Target timezone (default: {DEFAULT_TIMEZONE})")
    p.add_argument("--api-key", default=None, help="Databento API key (default: DATABENTO_API_KEY env var)")
    p.add_argument("--save-raw", action="store_true", help="Also save the raw 1-minute data before resampling")
    p.add_argument("--raw-output", default=None, help="Path for raw data (default: <output>_raw_1m.csv)")
    p.add_argument("--no-validate", action="store_true", help="Skip post-fetch validation")

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()

    try:
        df = fetch_and_save(
            start=args.start,
            end=args.end,
            output=args.output,
            symbol=args.symbol,
            dataset=args.dataset,
            schema=args.schema,
            stype_in=args.stype_in,
            timeframe=args.timeframe,
            timezone=args.timezone,
            api_key=args.api_key,
            save_raw_data=args.save_raw,
            raw_output=args.raw_output,
        )
    except EnvironmentError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Databento request failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.no_validate:
        print()
        print_report(df, label=args.output)

    print(f"\nDone. Run your backtest with:")
    print(f"  python run_backtest.py --data {args.output}")


if __name__ == "__main__":
    main()
