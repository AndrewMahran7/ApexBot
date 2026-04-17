"""
Dataset builder — orchestrates the full pipeline.
====================================================

  raw OHLCV (from fetch_data.py)
        → validated bars
        → features (data/features.py)
        → labels (data/labels.py)
        → trade candidates (filtered rows)
        → saved artefacts

This module is the single import for anyone who wants to build the
complete research dataset programmatically.  CLI wrappers live in
build_dataset.py and split_dataset.py at the project root.

Designed for >= 3 years of 5-minute MES data (~75 000+ bars).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.loader import load_bars
from data.validate import validate as validate_data
from data.features import compute_features, FeatureConfig
from data.labels import compute_labels, LabelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Master knobs for the dataset build."""
    # Input
    timezone: str = "America/New_York"

    # Feature / label sub-configs
    feature_cfg: FeatureConfig | None = None
    label_cfg: LabelConfig | None = None

    # Session times (used for trade-candidate extraction)
    range_end: str = "09:45"
    eod_exit_time: str = "15:50"
    session_open: str = "09:30"

    # Trade candidate filters
    min_range_points: float = 0.0   # 0 = no filter

    # Output directory
    output_dir: str = "data"

    def __post_init__(self):
        if self.feature_cfg is None:
            self.feature_cfg = FeatureConfig(
                range_end=self.range_end,
                session_open=self.session_open,
            )
        if self.label_cfg is None:
            self.label_cfg = LabelConfig(eod_exit_time=self.eod_exit_time)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DatasetResult:
    """Everything produced by a build run."""
    bars: pd.DataFrame               # validated OHLCV
    features: pd.DataFrame           # bars + feature columns
    labeled: pd.DataFrame            # features + label columns
    trade_candidates: pd.DataFrame   # filtered rows (one per setup)
    summary: dict                     # human-readable stats
    paths: dict                       # written file paths


def build_dataset(
    input_path: str,
    cfg: DatasetConfig | None = None,
    start: str | None = None,
    end: str | None = None,
    save: bool = True,
) -> DatasetResult:
    """
    Run the full pipeline: load → validate → features → labels → candidates.

    Parameters
    ----------
    input_path : str
        Path to CSV/parquet produced by fetch_data.py.
    cfg : DatasetConfig, optional
    start, end : str, optional
        Date range filter (YYYY-MM-DD).
    save : bool
        Whether to write parquet artefacts to disk.
    """
    if cfg is None:
        cfg = DatasetConfig()

    # --- 1. Load & validate ---
    logger.info("[1/5] Loading %s ...", input_path)
    bars = load_bars(input_path, timezone=cfg.timezone, start=start, end=end)
    errors = validate_data(bars)
    if errors:
        logger.warning("Validation issues in raw data:")
        for e in errors:
            logger.warning("  - %s", e)
    logger.info("       %s bars loaded, %s to %s",
               f"{len(bars):,}", bars.index[0].date(), bars.index[-1].date())

    # --- 2. Features ---
    logger.info("[2/5] Computing features ...")
    featured = compute_features(bars, cfg.feature_cfg)
    n_features = len([c for c in featured.columns if c.startswith("f_")])
    logger.info("       %d feature columns added", n_features)

    # --- 3. Labels ---
    logger.info("[3/5] Computing labels ...")
    labeled = compute_labels(featured, cfg.label_cfg)
    n_labels = len([c for c in labeled.columns if c.startswith("lbl_")])
    logger.info("       %d label columns added", n_labels)

    # --- 4. Trade candidates ---
    logger.info("[4/5] Extracting trade candidates ...")
    candidates = _extract_trade_candidates(labeled, cfg)
    logger.info("       %s trade candidates extracted", f"{len(candidates):,}")

    # --- 5. Summary & save ---
    summary = _compute_summary(bars, featured, labeled, candidates)

    paths: dict = {}
    if save:
        logger.info("[5/5] Saving outputs ...")
        paths = _save_outputs(bars, featured, candidates, summary, cfg)
    else:
        logger.info("[5/5] Skipping save (save=False)")

    _print_summary(summary)

    return DatasetResult(
        bars=bars,
        features=featured,
        labeled=labeled,
        trade_candidates=candidates,
        summary=summary,
        paths=paths,
    )


# ---------------------------------------------------------------------------
# Trade candidate extraction
# ---------------------------------------------------------------------------

def _extract_trade_candidates(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """
    Build a dataset where each row = a potential trade setup at a specific bar.

    A trade candidate is any bar that is:
      - after the opening range close
      - before EOD exit
      - within RTH
      - where the opening range has been established

    We include BOTH long and short directions as separate rows.
    """
    range_end_t = pd.Timestamp(f"1970-01-01 {cfg.range_end}").time()
    eod_t = pd.Timestamp(f"1970-01-01 {cfg.eod_exit_time}").time()

    # Filter to eligible bars
    mask = (
        df["f_range_size"].notna()
        & (df.index.time >= range_end_t)
        & (df.index.time < eod_t)
    )

    if cfg.min_range_points > 0:
        mask = mask & (df["f_range_size"] >= cfg.min_range_points)

    eligible = df.loc[mask].copy()

    if eligible.empty:
        return pd.DataFrame()

    # Take the FIRST eligible bar per session (the decision point)
    first_idx = eligible.groupby("session_date").head(1).index
    candidates = eligible.loc[first_idx].copy()

    # Add direction columns — one row per direction
    # We'll duplicate each row for long and short
    long_cands = candidates.copy()
    long_cands["direction"] = "long"

    short_cands = candidates.copy()
    short_cands["direction"] = "short"

    combined = pd.concat([long_cands, short_cands], axis=0).sort_index()

    # Add filter pass/fail flags
    _add_filter_flags(combined)

    return combined


def _add_filter_flags(df: pd.DataFrame):
    """Add boolean filter columns for research analysis."""
    # Range size OK
    df["filter_range_size"] = df["f_range_size"].notna() & (df["f_range_size"] > 0)

    # EMA filter: long = close > EMA, short = close < EMA
    if "f_price_ema" in df.columns:
        df["filter_ema"] = np.where(
            df["direction"] == "long",
            df["close"] > df["f_price_ema"],
            df["close"] < df["f_price_ema"],
        )
    else:
        df["filter_ema"] = True

    # EMA slope filter: long = slope > 0, short = slope < 0
    if "f_price_ema_slope" in df.columns:
        df["filter_ema_slope"] = np.where(
            df["direction"] == "long",
            df["f_price_ema_slope"] > 0,
            df["f_price_ema_slope"] < 0,
        )
    else:
        df["filter_ema_slope"] = True

    # Volume filter
    if "f_vol_relative" in df.columns:
        df["filter_volume"] = df["f_vol_relative"] >= 0.8
    else:
        df["filter_volume"] = True

    # ATR filter
    if "f_vola_atr" in df.columns:
        df["filter_atr"] = df["f_vola_atr"] >= 1.0
    else:
        df["filter_atr"] = True

    # Breakout distance: close beyond range boundary
    if "f_range_high" in df.columns and "f_range_low" in df.columns:
        df["filter_breakout"] = np.where(
            df["direction"] == "long",
            df["close"] > df["f_range_high"],
            df["close"] < df["f_range_low"],
        )
    else:
        df["filter_breakout"] = True


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _compute_summary(bars, featured, labeled, candidates) -> dict:
    n_sessions = bars.index.date
    unique_sessions = sorted(set(n_sessions))

    regime_dist = {}
    if "lbl_regime" in labeled.columns:
        # One regime per session
        per_session = labeled.groupby("session_date")["lbl_regime"].first()
        regime_dist = per_session.value_counts().to_dict()

    long_cands = candidates[candidates["direction"] == "long"] if "direction" in candidates.columns else candidates
    short_cands = candidates[candidates["direction"] == "short"] if "direction" in candidates.columns else pd.DataFrame()

    feat_cols = [c for c in featured.columns if c.startswith("f_")]
    label_cols = [c for c in labeled.columns if c.startswith("lbl_")]

    # Feature stats
    feat_stats = {}
    for c in feat_cols:
        s = featured[c].dropna()
        if len(s) > 0:
            feat_stats[c] = {
                "mean": round(float(s.mean()), 6),
                "std": round(float(s.std()), 6),
                "min": round(float(s.min()), 6),
                "max": round(float(s.max()), 6),
                "pct_nan": round(float(featured[c].isna().mean() * 100), 2),
            }

    return {
        "bars": len(bars),
        "sessions": len(unique_sessions),
        "date_range": f"{unique_sessions[0]} to {unique_sessions[-1]}",
        "feature_columns": len(feat_cols),
        "label_columns": len(label_cols),
        "trade_candidates": len(candidates),
        "long_candidates": len(long_cands),
        "short_candidates": len(short_cands),
        "regime_distribution": regime_dist,
        "feature_stats": feat_stats,
    }


def _print_summary(summary: dict):
    print(f"\n{'=' * 60}")
    print(f"  DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Bars              : {summary['bars']:,}")
    print(f"  Sessions          : {summary['sessions']}")
    print(f"  Date range        : {summary['date_range']}")
    print(f"  Feature columns   : {summary['feature_columns']}")
    print(f"  Label columns     : {summary['label_columns']}")
    print(f"  Trade candidates  : {summary['trade_candidates']:,}")
    print(f"    Long            : {summary['long_candidates']:,}")
    print(f"    Short           : {summary['short_candidates']:,}")
    if summary["regime_distribution"]:
        print(f"  Regime distribution:")
        for regime, count in sorted(summary["regime_distribution"].items()):
            print(f"    {regime:12s} : {count}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def _save_outputs(bars, featured, candidates, summary, cfg: DatasetConfig) -> dict:
    base = Path(cfg.output_dir)
    paths = {}

    # Processed bars
    processed_dir = base / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    p = str(processed_dir / "mes_5m.parquet")
    bars.to_parquet(p)
    paths["processed_bars"] = p
    print(f"  → {p}")

    # Features (full bar-level dataset with features + labels)
    feat_dir = base / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    p = str(feat_dir / "mes_features.parquet")
    featured.to_parquet(p)
    paths["features"] = p
    print(f"  → {p}")

    # Trade candidates
    p = str(feat_dir / "mes_trade_candidates.parquet")
    candidates.to_parquet(p)
    paths["trade_candidates"] = p
    print(f"  → {p}")

    # Summary JSON
    summary_path = str(feat_dir / "dataset_summary.json")
    # Make summary JSON-serializable
    serializable = {k: v for k, v in summary.items() if k != "feature_stats"}
    serializable["feature_count"] = len(summary.get("feature_stats", {}))
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    paths["summary"] = summary_path
    print(f"  → {summary_path}")

    return paths
