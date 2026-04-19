import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
#!/usr/bin/env python
"""
Example usage of the MES real-time data pipeline.

Modes:
  --live      Stream from Databento Live (requires DATABENTO_API_KEY)
  --replay    Replay a historical CSV file through the same pipeline

Examples:
    # Live streaming (requires API key)
    python run_pipeline.py --live

    # Replay historical data
    python run_pipeline.py --replay --data data/mes_5m.csv

    # Replay with simulated delay between ticks
    python run_pipeline.py --replay --data data/mes_5m.csv --speed 0.01
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from data.data_pipeline import LivePipeline, ReplayPipeline, Bar, log_bar


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

_collected_bars: list[Bar] = []


def on_bar(bar: Bar) -> None:
    """Log + store each completed bar."""
    log_bar(bar)
    _collected_bars.append(bar)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MES 5-min bar pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Stream from Databento Live")
    mode.add_argument("--replay", action="store_true", help="Replay historical CSV")

    parser.add_argument("--data", type=str, default="data/mes_5m.csv",
                        help="CSV path for --replay mode")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="Seconds between ticks in replay (0=instant)")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Seconds to run in --live mode (None=forever)")
    parser.add_argument("--symbol", type=str, default="MES.c.0",
                        help="Databento symbol for --live mode")

    args = parser.parse_args()

    if args.live:
        pipeline = LivePipeline(
            on_bar=on_bar,
            symbol=args.symbol,
        )
        print(f"Starting live stream for {args.symbol} ...")
        print("Press Ctrl+C to stop.\n")
        pipeline.run(timeout=args.timeout)

    elif args.replay:
        pipeline = ReplayPipeline(
            csv_path=args.data,
            on_bar=on_bar,
            speed=args.speed,
        )
        print(f"Replaying {args.data} ...\n")
        pipeline.run()

    print(f"\nTotal bars collected: {len(_collected_bars)}")
    if _collected_bars:
        first = _collected_bars[0]
        last = _collected_bars[-1]
        print(f"  First: {first.timestamp}  O={first.open:.2f}")
        print(f"  Last:  {last.timestamp}  C={last.close:.2f}")


if __name__ == "__main__":
    main()
