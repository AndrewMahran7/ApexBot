"""
Quick inspection utility for OHLCV bar data files.

Usage:
    python -m data.inspect_data data/mes_5m.csv
"""

import sys
import pandas as pd
from pathlib import Path
from data.loader import load_bars
from data.validate import print_report


def inspect(path: str, timezone: str = "America/New_York") -> None:
    """Load a data file and print a concise summary."""
    filepath = Path(path)
    if not filepath.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    df = load_bars(path, timezone=timezone)

    print(f"\n{'=' * 60}")
    print(f"  File:     {filepath.name}")
    print(f"  Rows:     {len(df):,}")
    print(f"  Columns:  {list(df.columns)}")
    print(f"  Timezone: {df.index.tz}")
    print(f"  Range:    {df.index.min()} → {df.index.max()}")

    # Trading days
    days = df.index.normalize().nunique()
    print(f"  Days:     {days}")

    # Bar frequency estimate
    if len(df) > 1:
        median_gap = pd.Series(df.index).diff().median()
        print(f"  Bar freq: ~{median_gap}")

    print(f"{'=' * 60}")

    # Head / tail
    print("\n--- First 5 bars ---")
    print(df.head().to_string())
    print("\n--- Last 5 bars ---")
    print(df.tail().to_string())

    # Sample around 09:30 ET (opening range start)
    morning = df.between_time("09:25", "09:45")
    if not morning.empty:
        first_day = morning.index.normalize()[0]
        day_sample = morning.loc[morning.index.normalize() == first_day]
        print(f"\n--- Sample around 09:30 ({first_day.date()}) ---")
        print(day_sample.to_string())

    # Validation
    print()
    print_report(df, label=filepath.name)
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m data.inspect_data <file> [timezone]")
        sys.exit(1)

    tz = sys.argv[2] if len(sys.argv) > 2 else "America/New_York"
    inspect(sys.argv[1], timezone=tz)
