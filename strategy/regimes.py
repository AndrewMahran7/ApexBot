"""
Regime classification for MES futures.
=======================================

Classifies market conditions into simple, interpretable regimes
using transparent rules based on price action and volume features.

Regimes:
  - TREND:    strong directional bias, EMA slope significant
  - BREAKOUT: range expansion relative to recent volatility
  - RANGE:    contained price action, no clear direction
  - DEAD:     very low volatility / volume, not worth trading

All thresholds are configurable for grid-search optimization.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from strategy.features import FeatureSnapshot


class Regime(Enum):
    TREND = auto()
    BREAKOUT = auto()
    RANGE = auto()
    DEAD = auto()


@dataclass
class RegimeDiagnostics:
    """Detailed breakdown of how the regime was classified."""
    regime: Regime
    or_range: float = 0.0
    atr: Optional[float] = None
    ema_slope: Optional[float] = None
    relative_volume: Optional[float] = None
    or_atr_ratio: Optional[float] = None
    is_trending: bool = False
    is_breakout: bool = False
    is_dead: bool = False
    reason: str = ""


def classify_regime(
    or_range: float,
    features: FeatureSnapshot,
    ema_slope_threshold: float = 0.15,
    range_ratio_threshold: float = 1.5,
    volume_ratio_threshold: float = 1.0,
    dead_atr_ratio: float = 0.4,
) -> RegimeDiagnostics:
    """
    Classify the current session regime using opening range and features.

    Logic (evaluated in priority order):
      1. DEAD  — OR range is tiny relative to ATR AND volume is below average
      2. TREND — EMA slope is significant (strong directional bias)
      3. BREAKOUT — OR range is large relative to ATR AND volume >= threshold
      4. RANGE — default fallback

    Parameters
    ----------
    or_range : float
        Size of the opening range in points.
    features : FeatureSnapshot
        Current technical features from FeatureEngine.
    ema_slope_threshold : float
        Minimum |EMA slope| to classify as trending (default 0.15).
    range_ratio_threshold : float
        Minimum OR/ATR ratio to classify as breakout (default 1.5).
    volume_ratio_threshold : float
        Minimum relative volume to avoid DEAD classification and
        to confirm BREAKOUT.
    dead_atr_ratio : float
        OR/ATR ratio below this = dead day candidate.

    Returns
    -------
    RegimeDiagnostics
    """
    diag = RegimeDiagnostics(regime=Regime.RANGE, or_range=or_range)
    diag.atr = features.atr
    diag.ema_slope = features.ema_slope
    diag.relative_volume = features.relative_volume

    # Compute OR/ATR ratio if ATR available
    or_atr_ratio: Optional[float] = None
    if features.atr is not None and features.atr > 0:
        or_atr_ratio = or_range / features.atr
    diag.or_atr_ratio = or_atr_ratio

    # --- Check DEAD ---
    is_dead = False
    if or_atr_ratio is not None and or_atr_ratio < dead_atr_ratio:
        is_dead = True
    if features.relative_volume is not None and features.relative_volume < volume_ratio_threshold:
        # Low volume reinforces dead classification
        if is_dead or (or_atr_ratio is not None and or_atr_ratio < dead_atr_ratio * 1.5):
            is_dead = True
    diag.is_dead = is_dead

    if is_dead:
        diag.regime = Regime.DEAD
        diag.reason = (f"DEAD: OR/ATR={or_atr_ratio:.2f} (<{dead_atr_ratio})"
                       if or_atr_ratio else "DEAD: no ATR data")
        if features.relative_volume is not None:
            diag.reason += f", relVol={features.relative_volume:.2f} (<{volume_ratio_threshold})"
        return diag

    # --- Check TREND ---
    is_trending = False
    if features.ema_slope is not None:
        if abs(features.ema_slope) >= ema_slope_threshold:
            is_trending = True
    diag.is_trending = is_trending

    # --- Check BREAKOUT ---
    # Requires range expansion AND volume confirmation
    is_breakout = False
    if or_atr_ratio is not None and or_atr_ratio >= range_ratio_threshold:
        # Volume must confirm the expansion (or be unavailable)
        vol_confirmed = (features.relative_volume is None or
                         features.relative_volume >= volume_ratio_threshold)
        if vol_confirmed:
            is_breakout = True
    diag.is_breakout = is_breakout

    # Threshold detail for diagnostics
    slope_str = f"slope={features.ema_slope:.4f}" if features.ema_slope is not None else "slope=N/A"
    ratio_str = f"OR/ATR={or_atr_ratio:.2f}" if or_atr_ratio is not None else "OR/ATR=N/A"
    vol_str = f"relVol={features.relative_volume:.2f}" if features.relative_volume is not None else "relVol=N/A"

    # Priority: TREND if slope is strong, BREAKOUT if range expanding, else RANGE
    if is_trending and is_breakout:
        diag.regime = Regime.BREAKOUT  # trending + expanding = breakout day
        diag.reason = (f"BREAKOUT+TREND: {slope_str} (>={ema_slope_threshold}), "
                       f"{ratio_str} (>={range_ratio_threshold}), {vol_str}")
    elif is_trending:
        diag.regime = Regime.TREND
        diag.reason = f"TREND: {slope_str} (>={ema_slope_threshold}), {ratio_str}, {vol_str}"
    elif is_breakout:
        diag.regime = Regime.BREAKOUT
        diag.reason = f"BREAKOUT: {ratio_str} (>={range_ratio_threshold}), {vol_str}"
    else:
        diag.regime = Regime.RANGE
        diag.reason = f"RANGE: {slope_str} (<{ema_slope_threshold}), {ratio_str} (<{range_ratio_threshold}), {vol_str}"

    return diag
