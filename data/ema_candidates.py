"""
EMA Trade Candidate Dataset Builder
=====================================

Generates a dataset where each row is an EMA directional trade candidate
with features and labels for ML model training.

Each session produces at most one candidate. The EMA directional logic
mirrors the benchmark in backtest/benchmark.py:
  - At first session bar: if close > EMA → long candidate
  - If close < EMA → short candidate (when enabled)
  - Otherwise → no candidate for the day

Labels use the SAME stop/target/exit logic as the backtester:
  - Long: SL = opening range low, TP = entry + RR * range, EOD exit
  - Short: SL = opening range high, TP = entry - RR * range, EOD exit

Features use ONLY past/current data. Labels use ONLY future data.
No lookahead bias.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from data.loader import load_bars
from data.features import compute_features, FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class EMACandidateConfig:
    """Configuration for EMA candidate generation."""
    timezone: str = "America/New_York"
    session_open: str = "09:30"
    range_start: str = "09:30"
    range_end: str = "09:45"
    eod_exit_time: str = "15:50"
    ema_length: int = 50
    reward_risk: float = 1.5
    allow_shorts: bool = True
    min_range_points: float = 0.0


def generate_ema_candidates(
    df: pd.DataFrame,
    cfg: EMACandidateConfig | None = None,
) -> pd.DataFrame:
    """
    Generate EMA trade candidate dataset from OHLCV bars.

    Parameters
    ----------
    df : pd.DataFrame
        Must have tz-aware DatetimeIndex and columns: open, high, low, close, volume.
    cfg : EMACandidateConfig, optional

    Returns
    -------
    pd.DataFrame where each row is one trade candidate with features + label.
    """
    if cfg is None:
        cfg = EMACandidateConfig()

    # Step 1: Compute vectorized features (no lookahead)
    feat_cfg = FeatureConfig(
        session_open=cfg.session_open,
        range_start=cfg.range_start,
        range_end=cfg.range_end,
        ema_length=cfg.ema_length,
    )
    featured = compute_features(df, feat_cfg)

    # Step 2: Identify EMA trade candidates (one per session)
    candidates = _identify_ema_candidates(featured, cfg)

    if candidates.empty:
        logger.warning("No EMA candidates found")
        return pd.DataFrame()

    # Step 3: Label each candidate using forward-looking data (same logic as backtester)
    candidates = _label_candidates(candidates, featured, cfg)

    # Step 4: Collect feature columns
    candidates = _collect_features(candidates)

    return candidates


def _identify_ema_candidates(df: pd.DataFrame, cfg: EMACandidateConfig) -> pd.DataFrame:
    """
    For each session, find the first bar after range close and check EMA condition.
    Mirrors the EMA directional benchmark logic.
    """
    range_end_t = pd.Timestamp(f"1970-01-01 {cfg.range_end}").time()
    eod_t = pd.Timestamp(f"1970-01-01 {cfg.eod_exit_time}").time()

    # Eligible bars: after range close, before EOD, range established
    mask = (
        df["f_range_size"].notna()
        & (df.index.time >= range_end_t)
        & (df.index.time < eod_t)
    )
    if cfg.min_range_points > 0:
        mask = mask & (df["f_range_size"] >= cfg.min_range_points)

    eligible = df.loc[mask]
    if eligible.empty:
        return pd.DataFrame()

    # First eligible bar per session = the decision point
    first_per_session = eligible.groupby("session_date").head(1)

    rows = []
    for ts, bar in first_per_session.iterrows():
        ema_val = bar.get("f_price_ema")
        if pd.isna(ema_val):
            continue  # EMA not yet available

        close = bar["close"]
        range_high = bar["f_range_high"]
        range_low = bar["f_range_low"]
        range_size = bar["f_range_size"]

        if pd.isna(range_high) or pd.isna(range_low) or range_size <= 0:
            continue

        # EMA directional logic
        if close > ema_val:
            direction = "long"
            entry_price = float(bar["open"])  # enter at open of decision bar
            stop_loss = float(range_low)
            take_profit = entry_price + cfg.reward_risk * float(range_size)
        elif close < ema_val and cfg.allow_shorts:
            direction = "short"
            entry_price = float(bar["open"])
            stop_loss = float(range_high)
            take_profit = entry_price - cfg.reward_risk * float(range_size)
        else:
            continue  # no signal

        rows.append({
            "timestamp": ts,
            "session_date": bar["session_date"],
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "range_high": float(range_high),
            "range_low": float(range_low),
            "range_size": float(range_size),
            # Carry all feature columns forward
            **{c: bar[c] for c in bar.index if c.startswith("f_")},
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.set_index("timestamp", inplace=True)
    return result


def _label_candidates(
    candidates: pd.DataFrame,
    full_bars: pd.DataFrame,
    cfg: EMACandidateConfig,
) -> pd.DataFrame:
    """
    Label each candidate: did TP hit before SL?
    Uses the SAME logic as the backtest engine (bar-by-bar forward scan).

    label_success = 1  →  TP hit before SL
    label_success = 0  →  SL hit before TP (or EOD exit was a loss)
    label_pnl_pts      →  signed PnL in points
    label_exit_reason   →  'TP', 'SL', or 'EOD'
    """
    eod_t = pd.Timestamp(f"1970-01-01 {cfg.eod_exit_time}").time()

    label_success = []
    label_pnl = []
    label_exit_reason = []

    for ts, cand in candidates.iterrows():
        direction = cand["direction"]
        entry_price = cand["entry_price"]
        stop_loss = cand["stop_loss"]
        take_profit = cand["take_profit"]
        session_date = cand["session_date"]

        # Get remaining bars in this session after the candidate bar
        session_mask = (
            (full_bars["session_date"] == session_date)
            & (full_bars.index > ts)
            & (full_bars.index.time <= eod_t)
        )
        future_bars = full_bars.loc[session_mask]

        if future_bars.empty:
            label_success.append(np.nan)
            label_pnl.append(np.nan)
            label_exit_reason.append("NO_DATA")
            logger.debug("No future bars for candidate at %s — labeled NO_DATA", ts)
            continue

        hit_tp = False
        hit_sl = False
        exit_price = float(future_bars.iloc[-1]["close"])  # default: EOD
        exit_reason = "EOD"

        for _, fb in future_bars.iterrows():
            high = float(fb["high"])
            low = float(fb["low"])

            if direction == "long":
                if low <= stop_loss:
                    hit_sl = True
                    exit_price = stop_loss
                    exit_reason = "SL"
                    break
                if high >= take_profit:
                    hit_tp = True
                    exit_price = take_profit
                    exit_reason = "TP"
                    break
            else:  # short
                if high >= stop_loss:
                    hit_sl = True
                    exit_price = stop_loss
                    exit_reason = "SL"
                    break
                if low <= take_profit:
                    hit_tp = True
                    exit_price = take_profit
                    exit_reason = "TP"
                    break

        # Compute PnL
        if direction == "long":
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price

        # Label: 1 if profitable, 0 otherwise
        if hit_tp:
            success = 1
        elif hit_sl:
            success = 0
        else:
            # EOD exit: label based on PnL sign
            success = 1 if pnl_pts > 0 else 0

        label_success.append(success)
        label_pnl.append(pnl_pts)
        label_exit_reason.append(exit_reason)

    candidates["label_success"] = label_success
    candidates["label_pnl_pts"] = label_pnl
    candidates["label_exit_reason"] = label_exit_reason

    # Drop rows where labeling failed
    candidates = candidates.dropna(subset=["label_success"])
    candidates["label_success"] = candidates["label_success"].astype(int)

    return candidates


def _collect_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all ML-relevant feature columns are present and add
    derived features specific to the trade candidate context.
    """
    # Add candidate-specific features
    if "f_price_ema" in candidates.columns:
        candidates["f_ema_distance"] = candidates["entry_price"] - candidates["f_price_ema"]
        candidates["f_ema_distance_pct"] = (
            candidates["f_ema_distance"] / candidates["f_price_ema"]
        )

    # Risk/reward context
    if "range_size" in candidates.columns:
        candidates["f_risk_points"] = candidates.apply(
            lambda r: abs(r["entry_price"] - r["stop_loss"]), axis=1
        )
        candidates["f_range_vs_atr"] = np.where(
            candidates.get("f_vola_atr", pd.Series(dtype=float)).notna(),
            candidates["range_size"] / candidates["f_vola_atr"].replace(0, np.nan),
            np.nan,
        )

    # Direction encoded
    candidates["f_direction_long"] = (candidates["direction"] == "long").astype(int)

    return candidates


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of ML feature columns from a candidate DataFrame."""
    return [c for c in df.columns if c.startswith("f_")]


def build_ema_candidate_dataset(
    input_path: str,
    cfg: EMACandidateConfig | None = None,
    start: str | None = None,
    end: str | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    End-to-end: load data → generate candidates → optionally save.

    Parameters
    ----------
    input_path : Path to OHLCV CSV/parquet.
    cfg : EMACandidateConfig
    start, end : Date range filters.
    save_path : If provided, save candidates to this CSV path.

    Returns
    -------
    pd.DataFrame of EMA trade candidates with features and labels.
    """
    if cfg is None:
        cfg = EMACandidateConfig()

    bars = load_bars(input_path, timezone=cfg.timezone, start=start, end=end)
    print(f"Loaded {len(bars):,} bars ({bars.index[0].date()} to {bars.index[-1].date()})")

    candidates = generate_ema_candidates(bars, cfg)
    print(f"Generated {len(candidates):,} EMA trade candidates")

    if not candidates.empty:
        n_long = (candidates["direction"] == "long").sum()
        n_short = (candidates["direction"] == "short").sum()
        n_success = candidates["label_success"].sum()
        print(f"  Long: {n_long}, Short: {n_short}")
        print(f"  Success rate: {n_success / len(candidates) * 100:.1f}%")

        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            candidates.to_csv(save_path)
            print(f"  Saved to {save_path}")

    return candidates
