"""
Historical bar data loader for MES futures backtesting.

Supports CSV and Parquet. Handles timezone conversion explicitly.
No external API calls — everything comes from local files.
"""

import logging

import pandas as pd
import pytz
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with the dataframe schema. Empty = OK."""
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    if df.empty:
        errors.append("Dataframe is empty")
    return errors


def load_bars(
    path: str,
    timezone: str = "America/New_York",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV bar data from a CSV or Parquet file.

    Parameters
    ----------
    path : str
        Path to CSV or .parquet file.
    timezone : str
        Target timezone for the timestamp column. If timestamps are naive
        they are assumed to already be in this timezone. If they carry a
        tz they are converted.
    start, end : str, optional
        ISO date strings to clip the date range (inclusive).

    Returns
    -------
    pd.DataFrame with tz-aware DatetimeIndex named 'timestamp' and
    float columns open/high/low/close/volume.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept common aliases
    rename_map = {
        "datetime": "timestamp",
        "date": "timestamp",
        "time": "timestamp",
        "dt": "timestamp",
        "vol": "volume",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    errors = validate_schema(df)
    if errors:
        raise ValueError(f"Data validation failed: {'; '.join(errors)}")

    # Parse timestamps — try naive first, fall back to utc=True for mixed offsets
    target_tz = pytz.timezone(timezone)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        except (ValueError, TypeError):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if df["timestamp"].dtype == "object":
        # Mixed offsets (e.g. DST transitions) → re-parse as UTC then convert
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if df["timestamp"].dt.tz is None:
        # Naive timestamps — assume they are already in the target timezone
        df["timestamp"] = df["timestamp"].dt.tz_localize(target_tz)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(target_tz)

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where OHLC is NaN (malformed data)
    before_len = len(df)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    dropped = before_len - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN OHLC values", dropped)

    # Optional date range filter
    if start:
        df = df[df.index >= pd.Timestamp(start, tz=target_tz)]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz=target_tz)]

    if df.empty:
        raise ValueError("No data remaining after filtering. Check date range and timezone.")

    return df
