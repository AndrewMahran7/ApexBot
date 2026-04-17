"""
Data validation utilities for OHLCV bar data.

Checks schema, OHLC consistency, timestamp monotonicity, duplicates,
and value ranges. Works on any DataFrame — not Databento-specific.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def validate(df: pd.DataFrame) -> list[str]:
    """
    Run all validation checks on an OHLCV DataFrame.

    The DataFrame should have either:
    - A 'timestamp' column, or
    - A DatetimeIndex named 'timestamp'.

    Returns a list of error strings. Empty list = data is clean.
    """
    errors: list[str] = []

    # --- Schema ---
    cols = set(df.columns)
    if df.index.name == "timestamp":
        cols.add("timestamp")
    missing = REQUIRED_COLUMNS - cols
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors  # can't do further checks

    if df.empty:
        errors.append("DataFrame is empty")
        return errors

    # Resolve timestamp series
    if df.index.name == "timestamp":
        ts = df.index.to_series()
    else:
        ts = df["timestamp"]

    # --- Timestamp checks ---
    if not pd.api.types.is_datetime64_any_dtype(ts):
        errors.append("Timestamp column is not datetime type")
    else:
        if not ts.is_monotonic_increasing:
            errors.append("Timestamps are not monotonically increasing")

        dupes = ts.duplicated().sum()
        if dupes > 0:
            errors.append(f"{dupes} duplicate timestamps found")

    # --- OHLC consistency ---
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    high_violations = ((h < o) | (h < c) | (h < l)).sum()
    if high_violations > 0:
        errors.append(f"{high_violations} bars where high < open/close/low")

    low_violations = ((l > o) | (l > c) | (l > h)).sum()
    if low_violations > 0:
        errors.append(f"{low_violations} bars where low > open/close/high")

    # --- Negative / zero prices ---
    for col_name in ["open", "high", "low", "close"]:
        neg = (df[col_name] <= 0).sum()
        if neg > 0:
            errors.append(f"{neg} non-positive values in '{col_name}'")

    # --- Negative volume ---
    neg_vol = (df["volume"] < 0).sum()
    if neg_vol > 0:
        errors.append(f"{neg_vol} negative volume values")

    # --- NaN checks ---
    for col_name in ["open", "high", "low", "close", "volume"]:
        nan_count = df[col_name].isna().sum()
        if nan_count > 0:
            errors.append(f"{nan_count} NaN values in '{col_name}'")

    return errors


def print_report(df: pd.DataFrame, label: str = "Data") -> bool:
    """
    Validate and print a human-readable report.

    Returns True if data is clean, False otherwise.
    """
    errors = validate(df)
    if not errors:
        logger.info("%s: %d bars, all checks passed.", label, len(df))
        print(f"[OK] {label}: {len(df):,} bars, all checks passed.")
        return True
    else:
        logger.warning("%s: %d issue(s) found: %s", label, len(errors), errors)
        print(f"[FAIL] {label}: {len(errors)} issue(s) found:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return False
