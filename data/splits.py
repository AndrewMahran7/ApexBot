"""
Time-based dataset splitting for MES research.
================================================

Splits a dataset into train / validation / test / holdout by DATE.
No random shuffling — temporal order is strictly preserved to prevent
leakage from future data into training.

Design assumptions:
  - Input DataFrame has a tz-aware DatetimeIndex OR a 'session_date' column
  - Splits are contiguous time blocks
  - A configurable gap (purge) between train→val and val→test prevents
    label leakage from overlapping forward-looking windows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Proportions and purge settings."""
    train_pct: float = 0.60
    val_pct: float = 0.20
    test_pct: float = 0.10
    holdout_pct: float = 0.10

    # Gap in calendar days between splits to avoid label leakage
    purge_days: int = 1

    def __post_init__(self):
        total = self.train_pct + self.val_pct + self.test_pct + self.holdout_pct
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split percentages must sum to 1.0, got {total:.4f}"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Holds the four splits plus metadata."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    holdout: pd.DataFrame
    info: dict  # metadata about each split


def time_based_split(
    df: pd.DataFrame,
    cfg: SplitConfig | None = None,
) -> SplitResult:
    """
    Split *df* into contiguous time blocks.

    Returns a SplitResult with .train, .validation, .test, .holdout DataFrames.
    """
    if cfg is None:
        cfg = SplitConfig()

    # Resolve session dates
    if "session_date" in df.columns:
        dates = sorted(df["session_date"].unique())
    else:
        dates = sorted(df.index.date)
        dates = sorted(set(dates))

    n = len(dates)
    if n < 20:
        raise ValueError(
            f"Only {n} unique dates — need at least 20 for meaningful splits."
        )

    # Calculate cut indices
    purge = cfg.purge_days
    n_train = int(n * cfg.train_pct)
    n_val = int(n * cfg.val_pct)
    n_test = int(n * cfg.test_pct)
    # holdout gets whatever remains

    train_end = n_train
    val_start = train_end + purge
    val_end = val_start + n_val
    test_start = val_end + purge
    test_end = test_start + n_test
    holdout_start = test_end + purge

    # Guard against purge eating everything
    if holdout_start >= n:
        # Reduce purge until it fits
        purge = 0
        val_start = train_end
        val_end = val_start + n_val
        test_start = val_end
        test_end = test_start + n_test
        holdout_start = test_end

    train_dates = set(dates[:train_end])
    val_dates = set(dates[val_start:val_end])
    test_dates = set(dates[test_start:test_end])
    holdout_dates = set(dates[holdout_start:])

    # Build masks
    if "session_date" in df.columns:
        sd = df["session_date"]
    else:
        sd = pd.Series(df.index.date, index=df.index)

    train_df = df[sd.isin(train_dates)].copy()
    val_df = df[sd.isin(val_dates)].copy()
    test_df = df[sd.isin(test_dates)].copy()
    holdout_df = df[sd.isin(holdout_dates)].copy()

    info = {
        "total_dates": n,
        "purge_days": purge,
        "train": _split_info("train", train_df, train_dates),
        "validation": _split_info("validation", val_df, val_dates),
        "test": _split_info("test", test_df, test_dates),
        "holdout": _split_info("holdout", holdout_df, holdout_dates),
    }

    logger.info(
        "Split %d dates: train=%d, val=%d, test=%d, holdout=%d (purge=%d)",
        n, len(train_dates), len(val_dates), len(test_dates),
        len(holdout_dates), purge,
    )

    return SplitResult(
        train=train_df,
        validation=val_df,
        test=test_df,
        holdout=holdout_df,
        info=info,
    )


def save_splits(
    result: SplitResult,
    output_dir: str = "data/splits",
) -> dict[str, str]:
    """Save each split to parquet. Returns dict of split → path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, frame in [
        ("train", result.train),
        ("validation", result.validation),
        ("test", result.test),
        ("holdout", result.holdout),
    ]:
        p = str(out / f"{name}.parquet")
        frame.to_parquet(p)
        paths[name] = p

    return paths


def validate_no_leakage(result: SplitResult) -> list[str]:
    """
    Check that no dates appear in multiple splits and order is preserved.
    Returns list of errors (empty = OK).
    """
    errors: list[str] = []
    logger.info("Validating no leakage across splits ...")

    def _dates(frame):
        if "session_date" in frame.columns:
            return set(frame["session_date"].unique())
        return set(frame.index.date)

    sets = {
        "train": _dates(result.train),
        "validation": _dates(result.validation),
        "test": _dates(result.test),
        "holdout": _dates(result.holdout),
    }

    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                errors.append(
                    f"Date overlap between {names[i]} and {names[j]}: "
                    f"{sorted(overlap)[:5]}..."
                )

    # Check temporal ordering
    def _max_date(frame):
        if frame.empty:
            return None
        if "session_date" in frame.columns:
            return max(frame["session_date"])
        return max(frame.index.date)

    def _min_date(frame):
        if frame.empty:
            return None
        if "session_date" in frame.columns:
            return min(frame["session_date"])
        return min(frame.index.date)

    ordered = [result.train, result.validation, result.test, result.holdout]
    for i in range(len(ordered) - 1):
        mx = _max_date(ordered[i])
        mn = _min_date(ordered[i + 1])
        if mx is not None and mn is not None and mx >= mn:
            errors.append(
                f"Split {names[i]} max date {mx} >= split {names[i+1]} min date {mn}"
            )

    return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_info(name: str, frame: pd.DataFrame, dates: set) -> dict:
    sorted_dates = sorted(dates) if dates else []
    return {
        "name": name,
        "rows": len(frame),
        "dates": len(sorted_dates),
        "start": str(sorted_dates[0]) if sorted_dates else None,
        "end": str(sorted_dates[-1]) if sorted_dates else None,
    }


def print_split_summary(result: SplitResult):
    """Print a human-readable summary of the splits."""
    info = result.info
    print(f"\n{'=' * 65}")
    print(f"  DATASET SPLITS  ({info['total_dates']} total dates, purge={info['purge_days']}d)")
    print(f"{'=' * 65}")
    for key in ("train", "validation", "test", "holdout"):
        s = info[key]
        pct = s["dates"] / info["total_dates"] * 100 if info["total_dates"] > 0 else 0
        print(f"  {s['name']:12s}  {s['dates']:4d} dates  {s['rows']:7,d} rows  "
              f"({pct:4.1f}%)  {s['start']} → {s['end']}")
    print(f"{'=' * 65}")
