"""
Databento historical data fetcher for MES futures.

Pulls OHLCV bars from Databento's Historical API, resamples to the target
timeframe, converts timezone, and saves to CSV compatible with data/loader.py.

API key: set DATABENTO_API_KEY environment variable.
"""

import logging
import os

import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET = "GLBX.MDP3"          # CME Globex
DEFAULT_SYMBOL = "MES.c.0"             # Continuous front-month MES
DEFAULT_SCHEMA = "ohlcv-1m"            # 1-minute OHLCV bars
DEFAULT_STYPE_IN = "continuous"
DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_RESAMPLE = "5min"


def _get_client(api_key: Optional[str] = None):
    """Create an authenticated Databento Historical client."""
    try:
        import databento as db
    except ImportError:
        raise ImportError(
            "The 'databento' package is required for data fetching. "
            "Install it with: pip install databento"
        )
    from dotenv import load_dotenv
    load_dotenv()
    key = api_key or os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise EnvironmentError(
            "DATABENTO_API_KEY not set. "
            "Export it as an environment variable or pass --api-key."
        )
    return db.Historical(key)


def fetch_bars(
    start: str,
    end: str,
    symbol: str = DEFAULT_SYMBOL,
    dataset: str = DEFAULT_DATASET,
    schema: str = DEFAULT_SCHEMA,
    stype_in: str = DEFAULT_STYPE_IN,
    api_key: Optional[str] = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Databento and return a clean DataFrame.

    Parameters
    ----------
    start, end : str
        ISO-8601 date strings (e.g. "2024-01-02").
    symbol : str
        Databento symbol (default: MES continuous front-month).
    dataset : str
        Databento dataset ID (default: GLBX.MDP3 for CME Globex).
    schema : str
        Databento schema (default: ohlcv-1m).
    stype_in : str
        Input symbology type (default: continuous).
    api_key : str, optional
        Databento API key. Falls back to DATABENTO_API_KEY env var.
    timezone : str
        Target timezone for timestamps.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.
        Index: tz-aware DatetimeIndex named 'ts_event' in the requested timezone.
    """
    client = _get_client(api_key)

    logger.info("Fetching %s from %s [%s]  %s → %s", symbol, dataset, schema, start, end)

    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=[symbol],
        schema=schema,
        stype_in=stype_in,
        start=start,
        end=end,
    )

    df = data.to_df(pretty_ts=True, price_type="float", tz=timezone)

    if df.empty:
        raise ValueError(
            f"No data returned for {symbol} on {dataset} between {start} and {end}. "
            "Check your date range and Databento entitlements."
        )

    logger.info("Received %s raw bars", f"{len(df):,}")
    return df


def resample_bars(df: pd.DataFrame, timeframe: str = DEFAULT_RESAMPLE) -> pd.DataFrame:
    """
    Resample OHLCV bars to a coarser timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must have open/high/low/close/volume columns and a DatetimeIndex.
    timeframe : str
        Pandas resample rule (e.g. '5min', '15min', '1h').

    Returns
    -------
    pd.DataFrame with resampled OHLCV bars (incomplete bars dropped).
    """
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    before_len = len(resampled)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    dropped = before_len - len(resampled)
    if dropped > 0:
        logger.warning("Resample dropped %d incomplete bars", dropped)

    logger.info("Resampled to %s: %s bars", timeframe, f"{len(resampled):,}")
    return resampled


def normalize_for_loader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape a Databento DataFrame to match the format expected by data/loader.py:
    columns = {timestamp, open, high, low, close, volume}.

    The index becomes a 'timestamp' column.
    """
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.index.name = "timestamp"
    return out


def save_csv(df: pd.DataFrame, output: str) -> Path:
    """Save normalized DataFrame to CSV."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path)
    except OSError as exc:
        logger.error("Failed to save CSV to %s: %s", path, exc)
        raise
    logger.info("Saved %s bars → %s", f"{len(df):,}", path)
    return path


def save_raw(df: pd.DataFrame, output: str) -> Path:
    """Save the full Databento DataFrame (all columns) to CSV for debugging."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path)
    except OSError as exc:
        logger.error("Failed to save raw data to %s: %s", path, exc)
        raise
    logger.info("Saved raw data (%s rows, %d cols) → %s", f"{len(df):,}", len(df.columns), path)
    return path


def fetch_and_save(
    start: str,
    end: str,
    output: str = "data/mes_5m.csv",
    symbol: str = DEFAULT_SYMBOL,
    dataset: str = DEFAULT_DATASET,
    schema: str = DEFAULT_SCHEMA,
    stype_in: str = DEFAULT_STYPE_IN,
    timeframe: str = DEFAULT_RESAMPLE,
    timezone: str = DEFAULT_TIMEZONE,
    api_key: Optional[str] = None,
    save_raw_data: bool = False,
    raw_output: Optional[str] = None,
) -> pd.DataFrame:
    """
    End-to-end: fetch from Databento → resample → normalize → save CSV.

    Returns the final normalized DataFrame.
    """
    raw_df = fetch_bars(
        start=start,
        end=end,
        symbol=symbol,
        dataset=dataset,
        schema=schema,
        stype_in=stype_in,
        api_key=api_key,
        timezone=timezone,
    )

    if save_raw_data:
        raw_path = raw_output or output.replace(".csv", "_raw_1m.csv")
        save_raw(raw_df, raw_path)

    # Resample if the source schema is finer than the target timeframe
    if timeframe and timeframe != "1min":
        resampled = resample_bars(raw_df, timeframe)
    else:
        resampled = raw_df

    normalized = normalize_for_loader(resampled)
    save_csv(normalized, output)
    return normalized
