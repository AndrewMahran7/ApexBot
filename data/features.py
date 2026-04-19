"""
Vectorized feature engineering for MES research datasets.
==========================================================

Computes interpretable features from OHLCV bar data for supervised
learning and strategy research.  All features use ONLY past/current data
— no lookahead.  Future information belongs in data/labels.py only.

Input : pd.DataFrame with tz-aware DatetimeIndex ('timestamp') and
        columns open, high, low, close, volume.
Output: Same DataFrame with feature columns appended.

Design for >= 3 years of 5-minute MES bars (~75k+ rows).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """All feature-engineering knobs in one place."""
    # Session times (Eastern)
    session_open: str = "09:30"
    session_close: str = "16:00"
    range_start: str = "09:30"
    range_end: str = "09:45"

    # EMA
    ema_length: int = 50

    # ATR / volatility
    atr_length: int = 14
    realized_vol_window: int = 20

    # Volume
    volume_avg_window: int = 20

    # Rolling returns / range
    rolling_return_window: int = 12   # 12 × 5 min = 1 hour
    rolling_range_window: int = 14

    # Compression lookback for regime features
    compression_window: int = 6       # bars before breakout to measure


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Append all feature columns to *df* (in-place) and return it.

    Categories added (prefixed for clarity):
      f_price_*    — price / structure features
      f_vol_*      — volume features
      f_vola_*     — volatility features
      f_time_*     — time-of-day features
      f_regime_*   — regime-proxy features
      f_range_*    — opening-range features (per-session)
    """
    if cfg is None:
        cfg = FeatureConfig()

    df = df.copy()

    # Make sure numeric
    for c in ("open", "high", "low", "close", "volume"):
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        after_na = df[c].isna().sum()
        coerced = after_na - before_na
        if coerced > 0:
            logger.warning("Column '%s': %d non-numeric values coerced to NaN", c, coerced)

    # --- Session / day helpers ---
    _add_session_columns(df, cfg)

    # --- Price / structure ---
    _add_price_features(df, cfg)

    # --- Volume ---
    _add_volume_features(df, cfg)

    # --- Volatility ---
    _add_volatility_features(df, cfg)

    # --- Time ---
    _add_time_features(df, cfg)

    # --- Opening-range per session ---
    _add_range_features(df, cfg)

    # --- Regime proxies ---
    _add_regime_features(df, cfg)

    return df


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _add_session_columns(df: pd.DataFrame, cfg: FeatureConfig):
    """Add session_date, session_bar_idx, is_rth columns."""
    df["session_date"] = df.index.date

    open_t = pd.Timestamp(f"1970-01-01 {cfg.session_open}").time()
    close_t = pd.Timestamp(f"1970-01-01 {cfg.session_close}").time()
    df["is_rth"] = (df.index.time >= open_t) & (df.index.time < close_t)

    # Bar index within session (0-based)
    df["session_bar_idx"] = df.groupby("session_date").cumcount()


# ---------------------------------------------------------------------------
# Price / structure features
# ---------------------------------------------------------------------------

def _add_price_features(df: pd.DataFrame, cfg: FeatureConfig):
    c = df["close"]

    # EMA and distance
    df["f_price_ema"] = c.ewm(span=cfg.ema_length, adjust=False).mean()
    df["f_price_ema_dist"] = c - df["f_price_ema"]
    df["f_price_ema_dist_pct"] = df["f_price_ema_dist"] / df["f_price_ema"]

    # EMA slope (change per bar over lookback)
    df["f_price_ema_slope"] = df["f_price_ema"].diff(5) / 5.0

    # Rolling returns
    w = cfg.rolling_return_window
    df[f"f_price_ret_{w}bar"] = c.pct_change(w)

    # Rolling range (avg bar range)
    df["f_price_bar_range"] = df["high"] - df["low"]
    df["f_price_rolling_range"] = df["f_price_bar_range"].rolling(cfg.rolling_range_window).mean()

    # Gap from prior session close
    # Use session_date to group; gap = today open - yesterday close
    session_close = df.groupby("session_date")["close"].transform("last")
    session_open = df.groupby("session_date")["open"].transform("first")
    prev_session_close = session_close.groupby(df["session_date"]).first().shift(1)
    prev_close_map = prev_session_close.to_dict()
    df["_prev_session_close"] = df["session_date"].map(prev_close_map)
    df["f_price_gap"] = session_open - df["_prev_session_close"]
    df["f_price_gap_pct"] = df["f_price_gap"] / df["_prev_session_close"]
    df.drop(columns=["_prev_session_close"], inplace=True)


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------

def _add_volume_features(df: pd.DataFrame, cfg: FeatureConfig):
    v = df["volume"].astype(float)
    w = cfg.volume_avg_window

    df["f_vol_avg"] = v.rolling(w).mean()
    df["f_vol_relative"] = v / df["f_vol_avg"].replace(0, np.nan)

    # Volume expansion: current bar vol vs prior bar vol
    df["f_vol_expansion"] = v / v.shift(1).replace(0, np.nan)


# ---------------------------------------------------------------------------
# Volatility features
# ---------------------------------------------------------------------------

def _add_volatility_features(df: pd.DataFrame, cfg: FeatureConfig):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)

    # True range
    tr = pd.concat([
        h - l,
        (h - pc).abs(),
        (l - pc).abs(),
    ], axis=1).max(axis=1)
    df["f_vola_tr"] = tr

    # ATR
    df["f_vola_atr"] = tr.rolling(cfg.atr_length).mean()

    # ATR normalised (ATR / close)
    df["f_vola_atr_norm"] = df["f_vola_atr"] / c

    # Realised volatility (std of log returns over window)
    log_ret = np.log(c / c.shift(1))
    df["f_vola_realized"] = log_ret.rolling(cfg.realized_vol_window).std() * np.sqrt(252 * 78)
    # 78 ≈ number of 5-min bars per session


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame, cfg: FeatureConfig):
    open_t = pd.Timestamp(f"1970-01-01 {cfg.session_open}").time()
    range_end_t = pd.Timestamp(f"1970-01-01 {cfg.range_end}").time()
    close_t = pd.Timestamp(f"1970-01-01 {cfg.session_close}").time()

    idx = df.index

    # Minutes since session open
    open_minutes = int(open_t.hour) * 60 + int(open_t.minute)
    bar_minutes = idx.hour * 60 + idx.minute
    df["f_time_minutes_since_open"] = bar_minutes - open_minutes

    # Minutes since range close
    range_end_mins = int(range_end_t.hour) * 60 + int(range_end_t.minute)
    df["f_time_minutes_since_range_close"] = bar_minutes - range_end_mins

    # Minutes to session close
    close_mins = int(close_t.hour) * 60 + int(close_t.minute)
    df["f_time_minutes_to_close"] = close_mins - bar_minutes

    # Weekday (0=Mon .. 4=Fri)
    df["f_time_weekday"] = idx.weekday


# ---------------------------------------------------------------------------
# Opening-range features (per-session, no lookahead)
# ---------------------------------------------------------------------------

def _add_range_features(df: pd.DataFrame, cfg: FeatureConfig):
    """
    For each session compute the opening range and add per-bar features
    relative to that range.  Bars before range close get NaN for these.
    """
    range_start_t = pd.Timestamp(f"1970-01-01 {cfg.range_start}").time()
    range_end_t = pd.Timestamp(f"1970-01-01 {cfg.range_end}").time()

    # Identify range bars
    in_range = (df.index.time >= range_start_t) & (df.index.time < range_end_t)

    # Per-session range high / low
    range_high = df.loc[in_range].groupby("session_date")["high"].max()
    range_low = df.loc[in_range].groupby("session_date")["low"].min()

    df["f_range_high"] = df["session_date"].map(range_high)
    df["f_range_low"] = df["session_date"].map(range_low)
    df["f_range_size"] = df["f_range_high"] - df["f_range_low"]

    # Null out for bars inside the range (they shouldn't have breakout features)
    df.loc[in_range, ["f_range_high", "f_range_low", "f_range_size"]] = np.nan

    # Breakout distance from range boundary
    df["f_range_dist_above"] = df["close"] - df["f_range_high"]
    df["f_range_dist_below"] = df["f_range_low"] - df["close"]

    # Range size normalised by ATR
    df["f_range_size_vs_atr"] = df["f_range_size"] / df["f_vola_atr"].replace(0, np.nan)


# ---------------------------------------------------------------------------
# Regime-proxy features (all backward-looking)
# ---------------------------------------------------------------------------

def _add_regime_features(df: pd.DataFrame, cfg: FeatureConfig):
    """
    Simple, transparent regime-proxy features.  These do NOT assign a
    discrete label — see data/labels.py for that.
    """
    # Trend proxy: sign and magnitude of EMA slope
    slope = df["f_price_ema_slope"]
    df["f_regime_trend_strength"] = slope.abs()
    df["f_regime_trend_direction"] = np.sign(slope)

    # Compression: ratio of recent range to ATR (low = compressed)
    recent_range = df["f_price_bar_range"].rolling(cfg.compression_window).mean()
    df["f_regime_compression"] = recent_range / df["f_vola_atr"].replace(0, np.nan)

    # Breakout strength: current bar range / ATR
    df["f_regime_breakout_strength"] = df["f_price_bar_range"] / df["f_vola_atr"].replace(0, np.nan)

    # Volume-weighted trend proxy
    df["f_regime_vol_trend"] = df["f_vol_relative"] * df["f_regime_trend_direction"]

    # --- Volatility regime (ATR percentile over rolling window) ---
    atr = df["f_vola_atr"]
    atr_roll_50 = atr.rolling(50, min_periods=20).median()
    # ATR ratio: current ATR vs recent median — >1 = high vol, <1 = low vol
    df["f_regime_atr_ratio"] = atr / atr_roll_50.replace(0, np.nan)
    # ATR percentile rank (0-1) over trailing 50 bars (vectorized)
    atr_roll_min = atr.rolling(50, min_periods=20).min()
    atr_roll_max = atr.rolling(50, min_periods=20).max()
    atr_range = (atr_roll_max - atr_roll_min).replace(0, np.nan)
    df["f_regime_atr_percentile"] = (atr - atr_roll_min) / atr_range

    # --- Trend regime (EMA slope magnitude percentile, vectorized) ---
    abs_slope = slope.abs()
    slope_roll_min = abs_slope.rolling(50, min_periods=20).min()
    slope_roll_max = abs_slope.rolling(50, min_periods=20).max()
    slope_range = (slope_roll_max - slope_roll_min).replace(0, np.nan)
    df["f_regime_trend_percentile"] = (abs_slope - slope_roll_min) / slope_range
