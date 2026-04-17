"""
Label generation for MES supervised-learning datasets.
========================================================

Labels use FUTURE data — they are targets for prediction, never features.
Each function clearly marks which columns it produces and how far ahead
it looks.

Conventions:
  - Label columns are prefixed ``lbl_`` so they can be identified
    and excluded from feature sets automatically.
  - NaN means the label cannot be computed (e.g. not enough future bars).
  - All functions operate on the full DataFrame in a vectorized way.
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
class LabelConfig:
    """All label-generation knobs."""
    # Breakout success: TP/SL based on opening-range size × R:R
    reward_risk_ratios: list[float] | None = None   # e.g. [1.5, 2.0, 3.0]

    # Future return horizons (in bars)
    return_horizons: list[int] | None = None         # e.g. [6, 12, 24, 78]

    # False breakout: how many bars to check for reversal
    false_breakout_bars: int = 6   # 6 × 5min = 30 min

    # End-of-day exit time (bars after this cannot start a trade)
    eod_exit_time: str = "15:50"

    # MES point value for dollar labels
    point_value: float = 5.0

    def __post_init__(self):
        if self.reward_risk_ratios is None:
            self.reward_risk_ratios = [1.5, 2.0, 3.0]
        if self.return_horizons is None:
            self.return_horizons = [6, 12, 24, 78]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_labels(df: pd.DataFrame, cfg: LabelConfig | None = None) -> pd.DataFrame:
    """
    Append all label columns to *df* and return it.

    Requires that ``f_range_high``, ``f_range_low``, ``f_range_size``
    have already been computed by data/features.py.
    """
    if cfg is None:
        cfg = LabelConfig()

    df = df.copy()

    _add_future_returns(df, cfg)
    _add_breakout_success(df, cfg)
    _add_false_breakout(df, cfg)
    _add_regime_label(df, cfg)

    return df


# ---------------------------------------------------------------------------
# Future return labels
# ---------------------------------------------------------------------------

def _add_future_returns(df: pd.DataFrame, cfg: LabelConfig):
    """
    lbl_ret_Nbar : simple return over next N bars
    lbl_ret_Nbar_pts : return in index points
    """
    c = df["close"]
    for n in cfg.return_horizons:
        future_close = c.shift(-n)
        df[f"lbl_ret_{n}bar"] = (future_close - c) / c
        df[f"lbl_ret_{n}bar_pts"] = future_close - c


# ---------------------------------------------------------------------------
# Breakout success labels
# ---------------------------------------------------------------------------

def _add_breakout_success(df: pd.DataFrame, cfg: LabelConfig):
    """
    For each bar AFTER the opening range, determine whether a hypothetical
    long or short breakout would have hit TP before SL (or vice versa)
    using the remaining bars in the session.

    Produces:
      lbl_long_success_RR  : 1 = TP hit first, 0 = SL hit first, NaN = neither
      lbl_short_success_RR : same for short direction
      lbl_long_pnl_pts_RR  : signed PnL in points (including EOD exit)
      lbl_short_pnl_pts_RR : same for short

    This scans forward within the same session only (no overnight).
    """
    required = {"f_range_high", "f_range_low", "f_range_size", "session_date"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("_add_breakout_success skipped — missing columns: %s", missing)
        return  # features not available; skip

    for rr in cfg.reward_risk_ratios:
        rr_str = str(rr).replace(".", "p")
        long_succ = f"lbl_long_success_{rr_str}"
        short_succ = f"lbl_short_success_{rr_str}"
        long_pnl = f"lbl_long_pnl_pts_{rr_str}"
        short_pnl = f"lbl_short_pnl_pts_{rr_str}"

        df[long_succ] = np.nan
        df[short_succ] = np.nan
        df[long_pnl] = np.nan
        df[short_pnl] = np.nan

        eod_t = pd.Timestamp(f"1970-01-01 {cfg.eod_exit_time}").time()

        # Group by session to avoid scanning across days
        for date, grp in df.groupby("session_date"):
            # Only bars after range is set AND before EOD
            mask = grp["f_range_size"].notna() & (grp.index.time < eod_t)
            idxs = grp.loc[mask].index

            if len(idxs) == 0:
                continue

            range_high = grp["f_range_high"].dropna().iloc[0] if grp["f_range_high"].notna().any() else np.nan
            range_low = grp["f_range_low"].dropna().iloc[0] if grp["f_range_low"].notna().any() else np.nan
            range_size = range_high - range_low if not (np.isnan(range_high) or np.isnan(range_low)) else np.nan

            if np.isnan(range_size) or range_size <= 0:
                continue

            long_entry = range_high
            long_sl = range_low
            long_tp = long_entry + rr * range_size

            short_entry = range_low
            short_sl = range_high
            short_tp = short_entry - rr * range_size

            # Get future highs/lows/closes for the session
            session_highs = grp["high"].values
            session_lows = grp["low"].values
            session_closes = grp["close"].values
            session_positions = grp.index

            pos_map = {ts: i for i, ts in enumerate(session_positions)}

            for ts in idxs:
                i = pos_map[ts]
                # Look forward within session
                future_highs = session_highs[i + 1:]
                future_lows = session_lows[i + 1:]

                if len(future_highs) == 0:
                    continue

                last_close = session_closes[-1]

                # --- Long ---
                tp_hit_long = np.where(future_highs >= long_tp)[0]
                sl_hit_long = np.where(future_lows <= long_sl)[0]
                tp_bar_l = tp_hit_long[0] if len(tp_hit_long) > 0 else len(future_highs) + 1
                sl_bar_l = sl_hit_long[0] if len(sl_hit_long) > 0 else len(future_highs) + 1

                if tp_bar_l < sl_bar_l:
                    df.at[ts, long_succ] = 1.0
                    df.at[ts, long_pnl] = long_tp - long_entry
                elif sl_bar_l < tp_bar_l:
                    df.at[ts, long_succ] = 0.0
                    df.at[ts, long_pnl] = long_sl - long_entry
                else:
                    # Neither hit → EOD exit
                    df.at[ts, long_succ] = np.nan
                    df.at[ts, long_pnl] = last_close - long_entry

                # --- Short ---
                tp_hit_short = np.where(future_lows <= short_tp)[0]
                sl_hit_short = np.where(future_highs >= short_sl)[0]
                tp_bar_s = tp_hit_short[0] if len(tp_hit_short) > 0 else len(future_lows) + 1
                sl_bar_s = sl_hit_short[0] if len(sl_hit_short) > 0 else len(future_lows) + 1

                if tp_bar_s < sl_bar_s:
                    df.at[ts, short_succ] = 1.0
                    df.at[ts, short_pnl] = short_entry - short_tp
                elif sl_bar_s < tp_bar_s:
                    df.at[ts, short_succ] = 0.0
                    df.at[ts, short_pnl] = short_entry - short_sl
                else:
                    df.at[ts, short_succ] = np.nan
                    df.at[ts, short_pnl] = short_entry - last_close


# ---------------------------------------------------------------------------
# False breakout labels
# ---------------------------------------------------------------------------

def _add_false_breakout(df: pd.DataFrame, cfg: LabelConfig):
    """
    lbl_false_breakout_long  : 1 if price breaks above range high then
                                falls back below within M bars
    lbl_false_breakout_short : analogous for short
    """
    required = {"f_range_high", "f_range_low"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("_add_false_breakout skipped — missing columns: %s", missing)
        return

    m = cfg.false_breakout_bars

    df["lbl_false_breakout_long"] = np.nan
    df["lbl_false_breakout_short"] = np.nan

    range_high = df["f_range_high"]
    range_low = df["f_range_low"]
    close = df["close"]

    # For each bar that is above range high (potential long breakout)
    above = close > range_high
    for idx_pos in np.where(above.values)[0]:
        end_pos = min(idx_pos + m, len(df) - 1)
        future_closes = close.iloc[idx_pos + 1: end_pos + 1]
        rh = range_high.iloc[idx_pos]
        if pd.isna(rh) or len(future_closes) == 0:
            continue
        # False breakout = any future close falls back below range high
        if (future_closes < rh).any():
            df.iloc[idx_pos, df.columns.get_loc("lbl_false_breakout_long")] = 1.0
        else:
            df.iloc[idx_pos, df.columns.get_loc("lbl_false_breakout_long")] = 0.0

    below = close < range_low
    for idx_pos in np.where(below.values)[0]:
        end_pos = min(idx_pos + m, len(df) - 1)
        future_closes = close.iloc[idx_pos + 1: end_pos + 1]
        rl = range_low.iloc[idx_pos]
        if pd.isna(rl) or len(future_closes) == 0:
            continue
        if (future_closes > rl).any():
            df.iloc[idx_pos, df.columns.get_loc("lbl_false_breakout_short")] = 1.0
        else:
            df.iloc[idx_pos, df.columns.get_loc("lbl_false_breakout_short")] = 0.0


# ---------------------------------------------------------------------------
# Regime label (discrete classification target)
# ---------------------------------------------------------------------------

def _add_regime_label(df: pd.DataFrame, cfg: LabelConfig):
    """
    lbl_regime : categorical label per session
      - 'trend'     : session moved > 1 ATR in one direction
      - 'breakout'  : opening range breakout occurred AND held
      - 'range'     : session range < 1 ATR
      - 'dead'      : nearly no movement / very low volume

    Uses INTRA-SESSION future data (end-of-session outcomes) so this
    is a label, not a feature.
    """
    if "f_vola_atr" not in df.columns or "session_date" not in df.columns:
        missing = [c for c in ("f_vola_atr", "session_date") if c not in df.columns]
        logger.warning("_add_regime_label skipped — missing columns: %s", missing)
        return

    # Compute per-session metrics
    sessions = df.groupby("session_date").agg(
        session_high=("high", "max"),
        session_low=("low", "min"),
        session_open=("open", "first"),
        session_close=("close", "last"),
        session_volume=("volume", "sum"),
        median_atr=("f_vola_atr", "median"),
    )

    sessions["session_range"] = sessions["session_high"] - sessions["session_low"]
    sessions["session_move"] = (sessions["session_close"] - sessions["session_open"]).abs()

    def _classify(row):
        atr = row["median_atr"]
        if pd.isna(atr) or atr <= 0:
            return "range"

        range_ratio = row["session_range"] / atr
        move_ratio = row["session_move"] / atr

        if range_ratio < 0.4:
            return "dead"
        if move_ratio > 1.0:
            return "trend"
        if range_ratio > 1.0 and move_ratio > 0.5:
            return "breakout"
        return "range"

    sessions["lbl_regime"] = sessions.apply(_classify, axis=1)
    regime_map = sessions["lbl_regime"].to_dict()
    df["lbl_regime"] = df["session_date"].map(regime_map)
