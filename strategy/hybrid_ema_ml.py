"""
Hybrid EMA + ML Strategy for MES Futures
==========================================

Combines EMA directional signals with ML-based trade filtering.

Selection modes:
  threshold : Accept if ML prob >= fixed threshold (original mode).
  top_n     : Accept if ML prob ranks in the top N among a rolling
              window of past candidate scores.  Fully causal — only
              compares against probabilities already observed.
  top_pct   : Accept if ML prob >= the (1 - pct) percentile of the
              rolling window.  E.g. top_pct=0.30 keeps the top 30%.

Leakage safety (ranking modes):
  Each session produces at most ONE EMA candidate (first bar after
  range close).  The rolling window stores probabilities from prior
  sessions only — the current candidate's score is compared against
  that window, then appended AFTER the accept/reject decision.  No
  future information is ever used.

Plugs into the existing backtest engine via Signal/SignalType protocol.
Uses the same execution logic, costs, and slippage.
"""

from __future__ import annotations

import datetime
import logging
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from strategy.orb import SignalType, Signal, _parse_time
from strategy.features import FeatureEngine, FeatureSnapshot
from data.ema_candidates import get_feature_columns

# Priority ordering for entry types (lower = higher priority)
STRATEGY_PRIORITY: dict[str, int] = {
    "breakout": 0,
    "momentum": 1,
    "pullback": 2,
}


@dataclass
class HybridEMAMLConfig:
    """Configuration for the hybrid EMA + ML strategy."""
    timezone: str = "America/New_York"
    session_open: str = "09:30"
    range_start: str = "09:30"
    range_end: str = "09:45"
    eod_exit_time: str = "15:50"

    # EMA
    ema_length: int = 50

    # Risk management
    reward_risk: float = 1.5

    # Direction
    allow_shorts: bool = True

    # --- ML selection ---
    # Mode: "threshold", "top_n", or "top_pct"
    ml_selection_mode: str = "threshold"

    # threshold mode
    ml_threshold: float = 0.6

    # top_n mode: accept if prob ranks in top N of rolling window
    ml_top_n: int = 1

    # top_pct mode: accept if prob >= (1-pct) percentile of window
    # e.g. 0.30 = keep the top 30% of candidates
    ml_top_pct: float = 0.30

    # Rolling window size for ranking modes (number of past candidates)
    ml_lookback: int = 20

    # Direction-specific overrides (None = use global setting)
    ml_top_n_long: int | None = None
    ml_top_n_short: int | None = None
    ml_top_pct_long: float | None = None
    ml_top_pct_short: float | None = None

    # Model path
    model_path: str = "models/ema_model.pkl"

    # --- Position sizing ---
    # "none" = binary accept/reject (backward compatible)
    # "linear", "convex", "hybrid" = continuous sizing based on percentile
    position_sizing_mode: str = "none"
    base_size: float = 1.0          # maximum position scale factor

    # --- Multi-candidate mode ---
    multi_candidate: bool = False
    max_trades_per_day: int = 3
    ema_periods: tuple[int, ...] = (50,)
    entry_types: tuple[str, ...] = ("breakout",)

    # --- Selection strategy ---
    # "global_ml"          : Sort all candidates by ML prob, apply selection (original)
    # "priority"           : Group by entry_type priority, ML ranks within groups
    # "priority_ml_sizing" : Priority ordering, always enter, ML only for sizing
    selection_strategy: str = "global_ml"

    # Within-group ML threshold (used by "priority" mode only)
    # 0.0 = no within-group filtering
    ml_within_group_threshold: float = 0.0


@dataclass
class TradeCandidate:
    """A scored trade opportunity before selection."""
    ema_length: int
    entry_type: str          # 'breakout', 'pullback', 'momentum'
    direction: str           # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    features: dict
    ml_prob: float = 0.5
    percentile: float = 0.5
    position_size: float = 1.0
    timestamp: Optional[datetime.datetime] = None
    strategy_type: str = ""  # "ema50_breakout"


def _no_signal(close: float, ts: datetime.datetime) -> Signal:
    return Signal(signal_type=SignalType.NONE, price=close, timestamp=ts)


class HybridEMAMLStrategy:
    """
    EMA directional + ML filter strategy.

    Processes bars one at a time. Returns Signal objects
    consumed by the backtest engine.

    Selection modes:
      threshold — accept if prob >= fixed value
      top_n    — accept if prob ranks top-N in rolling window of past scores
      top_pct  — accept if prob >= (1-pct) percentile of rolling window
    """

    def __init__(self, config: HybridEMAMLConfig):
        self.cfg = config
        self._validate_config()

        # Parse times
        self._range_start = _parse_time(config.range_start)
        self._range_end = _parse_time(config.range_end)
        self._eod_exit = _parse_time(config.eod_exit_time)
        self._session_open = _parse_time(config.session_open)

        # Feature engine (same as used in data pipeline)
        self._features = FeatureEngine(
            ema_length=config.ema_length,
            ema_slope_lookback=5,
            atr_length=14,
            volume_lookback=20,
        )

        # Multi-candidate: one FeatureEngine per EMA length
        if config.multi_candidate:
            self._feature_engines: dict[int, FeatureEngine] = {
                length: FeatureEngine(
                    ema_length=length, ema_slope_lookback=5,
                    atr_length=14, volume_lookback=20,
                )
                for length in config.ema_periods
            }
        else:
            self._feature_engines = {config.ema_length: self._features}

        # Opening range state
        self.opening_high: Optional[float] = None
        self.opening_low: Optional[float] = None
        self.range_set: bool = False

        # Trade state (single-candidate mode)
        self.trade_taken: bool = False
        self.in_position: bool = False
        self.direction: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.tp: Optional[float] = None
        self.sl: Optional[float] = None

        # Multi-position state (multi-candidate mode)
        self._open_positions: dict[str, dict] = {}  # position_id -> state

        # Day tracking
        self._current_date: Optional[datetime.date] = None
        self._decided_today: bool = False

        # Rolling data for feature computation
        self._bar_count: int = 0
        self._close_history: deque = deque(maxlen=100)
        self._high_history: deque = deque(maxlen=100)
        self._low_history: deque = deque(maxlen=100)
        self._volume_history: deque = deque(maxlen=100)

        # ML model (loaded lazily)
        self._model = None
        self._feature_columns: list[str] = []
        self._model_loaded = False

        # Rolling probability windows for ranking modes.
        # Stores probabilities from PRIOR sessions only.
        # The current candidate is compared against this window,
        # then appended after the accept/reject decision.
        lookback = max(config.ml_lookback, 1)
        self._prob_window_all: deque = deque(maxlen=lookback)
        self._prob_window_long: deque = deque(maxlen=lookback)
        self._prob_window_short: deque = deque(maxlen=lookback)

        # Diagnostics
        self.ml_decisions: list[dict] = []

        logger.info(
            "HybridEMAMLStrategy initialized: selection_strategy=%s, "
            "multi_candidate=%s, ema_periods=%s, entry_types=%s",
            config.selection_strategy,
            config.multi_candidate,
            config.ema_periods,
            config.entry_types,
        )

    def _validate_config(self):
        """Validate configuration at construction time — fail fast."""
        cfg = self.cfg
        valid_selection = ("global_ml", "priority", "priority_ml_sizing")
        if cfg.selection_strategy not in valid_selection:
            raise ValueError(
                f"Invalid selection_strategy='{cfg.selection_strategy}'. "
                f"Valid: {valid_selection}"
            )
        valid_sizing = ("none", "linear", "convex", "hybrid")
        if cfg.position_sizing_mode not in valid_sizing:
            raise ValueError(
                f"Invalid position_sizing_mode='{cfg.position_sizing_mode}'. "
                f"Valid: {valid_sizing}"
            )
        valid_ml_mode = ("threshold", "top_n", "top_pct")
        if cfg.ml_selection_mode not in valid_ml_mode:
            raise ValueError(
                f"Invalid ml_selection_mode='{cfg.ml_selection_mode}'. "
                f"Valid: {valid_ml_mode}"
            )
        if cfg.max_trades_per_day < 1:
            raise ValueError(f"max_trades_per_day must be >= 1, got {cfg.max_trades_per_day}")
        if cfg.ml_threshold < 0 or cfg.ml_threshold > 1:
            raise ValueError(f"ml_threshold must be in [0, 1], got {cfg.ml_threshold}")
        for period in cfg.ema_periods:
            if period < 1:
                raise ValueError(f"EMA period must be >= 1, got {period}")
        valid_entry_types = {"breakout", "pullback", "momentum"}
        for et in cfg.entry_types:
            if et not in valid_entry_types:
                raise ValueError(f"Invalid entry_type='{et}'. Valid: {valid_entry_types}")

    def _ensure_model_loaded(self):
        """Load ML model from disk on first use."""
        if self._model_loaded:
            return

        path = self.cfg.model_path
        logger.info("Loading ML model from %s", path)
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ML model not found at {path}. "
                f"Run train_model.py first."
            )
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"Failed to load ML model from {path}: {exc}"
            ) from exc

        if "model" not in data or "feature_columns" not in data:
            raise ValueError(
                f"ML model pkl at {path} missing required keys. "
                f"Expected 'model' and 'feature_columns', got: {list(data.keys())}"
            )

        self._model = data["model"]
        self._feature_columns = data["feature_columns"]
        self._model_loaded = True
        logger.info(
            "Model loaded: %d feature columns",
            len(self._feature_columns),
        )

    def on_bar(self, bar: dict):
        """
        Process a single OHLCV bar.

        Returns Signal (single-candidate mode) or list[Signal]
        (multi-candidate mode).
        """
        ts: datetime.datetime = bar["timestamp"]
        bar_date = ts.date()
        bar_time = ts.time()
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        open_px = float(bar["open"])
        volume = float(bar["volume"])

        # New day reset
        if self._current_date is None or bar_date != self._current_date:
            self._reset_day(bar_date)

        # Update features every bar
        self._features.update(high, low, close, volume)
        self._bar_count += 1
        self._close_history.append(close)
        self._high_history.append(high)
        self._low_history.append(low)
        self._volume_history.append(volume)

        # Update all feature engines (multi-candidate mode)
        if self.cfg.multi_candidate:
            for length, eng in self._feature_engines.items():
                if eng is not self._features:  # avoid double-update
                    eng.update(high, low, close, volume)

        # Build opening range
        if bar_time >= self._range_start and bar_time < self._range_end:
            self._update_opening_range(high, low)
            if self.cfg.multi_candidate:
                return [_no_signal(close, ts)]
            return _no_signal(close, ts)

        # Finalize range
        if bar_time >= self._range_end and not self.range_set:
            if self.opening_high is not None and self.opening_low is not None:
                self.range_set = True

        # Dispatch to multi-candidate or single-candidate path
        if self.cfg.multi_candidate:
            return self._on_bar_multi(bar, ts, bar_time, close, high, low, open_px, volume)

        # Exit logic (before entry)
        if self.in_position:
            if bar_time >= self._eod_exit:
                return self._exit(close, ts, SignalType.EXIT_EOD, "End-of-day exit")

            if self.direction == "long":
                if self.sl is not None and low <= self.sl:
                    return self._exit(self.sl, ts, SignalType.EXIT_SL,
                                      f"Stop loss hit ({self.sl:.2f})")
                if self.tp is not None and high >= self.tp:
                    return self._exit(self.tp, ts, SignalType.EXIT_TP,
                                      f"Take profit hit ({self.tp:.2f})")
            else:  # short
                if self.sl is not None and high >= self.sl:
                    return self._exit(self.sl, ts, SignalType.EXIT_SL,
                                      f"Stop loss hit ({self.sl:.2f})")
                if self.tp is not None and low <= self.tp:
                    return self._exit(self.tp, ts, SignalType.EXIT_TP,
                                      f"Take profit hit ({self.tp:.2f})")

        # Entry logic: first bar after range close, once per day
        if self._can_enter() and not self._decided_today and bar_time >= self._range_end:
            self._decided_today = True

            snap = self._features.snapshot
            if snap.ema is None:
                return _no_signal(close, ts)

            or_range = self.opening_high - self.opening_low
            if or_range <= 0:
                return _no_signal(close, ts)

            # EMA directional signal
            if close > snap.ema:
                direction = "long"
                entry_px = open_px
                sl = self.opening_low
                tp = entry_px + self.cfg.reward_risk * or_range
            elif close < snap.ema and self.cfg.allow_shorts:
                direction = "short"
                entry_px = open_px
                sl = self.opening_high
                tp = entry_px - self.cfg.reward_risk * or_range
            else:
                return _no_signal(close, ts)

            # ML filter
            self._ensure_model_loaded()
            features = self._extract_features(bar, snap, direction)
            prob = self._predict_proba(features)

            # --- Selection decision ---
            accepted, rank, window_size, reject_reason = self._selection_decision(
                prob, direction
            )

            # --- Percentile & position sizing ---
            percentile = self._compute_percentile(prob, direction)
            position_size = self._compute_position_size(percentile)

            # If sizing mode is active and size rounds to zero, skip trade
            if self.cfg.position_sizing_mode != "none" and position_size <= 0:
                accepted = False
                reject_reason = reject_reason or f"position_size=0 (pctile={percentile:.2f})"

            # Record decision for diagnostics (before updating window)
            self.ml_decisions.append({
                "timestamp": ts,
                "direction": direction,
                "ema_signal": True,
                "ml_prob": prob,
                "selection_mode": self.cfg.ml_selection_mode,
                "accepted": accepted,
                "rank": rank,
                "window_size": window_size,
                "reject_reason": reject_reason,
                "percentile": percentile,
                "position_size": position_size,
                # Mode-specific context
                "threshold": self.cfg.ml_threshold if self.cfg.ml_selection_mode == "threshold" else None,
                "top_n": self._effective_top_n(direction) if self.cfg.ml_selection_mode == "top_n" else None,
                "top_pct": self._effective_top_pct(direction) if self.cfg.ml_selection_mode == "top_pct" else None,
                "lookback": self.cfg.ml_lookback if self.cfg.ml_selection_mode != "threshold" else None,
            })

            # Always append to rolling window AFTER the decision (causal)
            self._prob_window_all.append(prob)
            if direction == "long":
                self._prob_window_long.append(prob)
            else:
                self._prob_window_short.append(prob)

            if accepted:
                if direction == "long":
                    return self._enter_long(entry_px, sl, tp, ts, or_range, prob, position_size)
                else:
                    return self._enter_short(entry_px, sl, tp, ts, or_range, prob, position_size)

        return _no_signal(close, ts)

    def reset(self):
        """Full reset for a new backtest run."""
        self.opening_high = None
        self.opening_low = None
        self.range_set = False
        self.trade_taken = False
        self.in_position = False
        self.direction = None
        self.entry_price = None
        self.tp = None
        self.sl = None
        self._open_positions.clear()
        self._current_date = None
        self._decided_today = False
        self._features.reset()
        for eng in self._feature_engines.values():
            if eng is not self._features:
                eng.reset()
        self._bar_count = 0
        self._close_history.clear()
        self._high_history.clear()
        self._low_history.clear()
        self._volume_history.clear()
        self._prob_window_all.clear()
        self._prob_window_long.clear()
        self._prob_window_short.clear()
        self.ml_decisions = []

    # ------------------------------------------------------------------
    # Selection logic  (threshold / top_n / top_pct)
    # ------------------------------------------------------------------

    def _selection_decision(
        self, prob: float, direction: str
    ) -> tuple[bool, int, int, str]:
        """
        Decide whether to accept this candidate.

        Returns (accepted, rank, window_size, reject_reason).
        rank = 1 means highest in window; 0 if not applicable.
        reject_reason = "" if accepted.

        Leakage guarantee: the rolling window contains only scores from
        PRIOR sessions. The current score is NOT in the window during
        this comparison. It is appended by the caller after this returns.
        """
        mode = self.cfg.ml_selection_mode

        if mode == "threshold":
            accepted = prob >= self.cfg.ml_threshold
            return (accepted, 0, 0,
                    "" if accepted else f"prob {prob:.3f} < threshold {self.cfg.ml_threshold}")

        # For ranking modes, choose the direction-specific window if
        # direction-specific settings are configured, else use the
        # combined window.
        window = self._get_ranking_window(direction)
        window_size = len(window)

        if mode == "top_n":
            return self._decide_top_n(prob, direction, window, window_size)
        elif mode == "top_pct":
            return self._decide_top_pct(prob, direction, window, window_size)
        else:
            raise ValueError(f"Unknown ml_selection_mode: {mode}")

    def _decide_top_n(
        self, prob: float, direction: str,
        window: deque, window_size: int,
    ) -> tuple[bool, int, int, str]:
        """
        Accept if prob would rank in the top N among the rolling window.

        Cold-start: if window has fewer entries than top_n, accept
        (we don't have enough history to be selective).
        """
        top_n = self._effective_top_n(direction)

        if window_size < top_n:
            # Not enough history — accept (cold-start grace period)
            return (True, 1, window_size,
                    "")

        # Rank: count how many past scores are strictly greater than prob.
        # rank=1 means prob is the highest (or tied for highest).
        n_higher = sum(1 for p in window if p > prob)
        rank = n_higher + 1  # 1-indexed

        accepted = rank <= top_n
        reason = "" if accepted else (
            f"rank {rank}/{window_size} exceeds top_n={top_n}"
        )
        return (accepted, rank, window_size, reason)

    def _decide_top_pct(
        self, prob: float, direction: str,
        window: deque, window_size: int,
    ) -> tuple[bool, int, int, str]:
        """
        Accept if prob >= the (1 - top_pct) percentile of the window.

        E.g. top_pct=0.30 means keep the top 30% → accept if prob >=
        the 70th percentile of the window.

        Cold-start: if window has < 5 entries, accept.
        """
        top_pct = self._effective_top_pct(direction)
        min_window = 5

        if window_size < min_window:
            return (True, 1, window_size, "")

        # Compute the percentile cutoff from past scores
        cutoff = float(np.percentile(list(window), (1.0 - top_pct) * 100))

        n_higher = sum(1 for p in window if p > prob)
        rank = n_higher + 1

        accepted = prob >= cutoff
        reason = "" if accepted else (
            f"prob {prob:.3f} < cutoff {cutoff:.3f} "
            f"(top {top_pct:.0%} of {window_size} window)"
        )
        return (accepted, rank, window_size, reason)

    def _get_ranking_window(self, direction: str) -> deque:
        """
        Return the appropriate rolling window for ranking.

        Uses direction-specific window if direction-specific overrides
        are configured; otherwise uses the combined window.
        """
        cfg = self.cfg
        if direction == "long" and (cfg.ml_top_n_long is not None or cfg.ml_top_pct_long is not None):
            return self._prob_window_long
        if direction == "short" and (cfg.ml_top_n_short is not None or cfg.ml_top_pct_short is not None):
            return self._prob_window_short
        return self._prob_window_all

    def _effective_top_n(self, direction: str) -> int:
        """Resolve direction-specific top_n or fall back to global."""
        cfg = self.cfg
        if direction == "long" and cfg.ml_top_n_long is not None:
            return cfg.ml_top_n_long
        if direction == "short" and cfg.ml_top_n_short is not None:
            return cfg.ml_top_n_short
        return cfg.ml_top_n

    def _effective_top_pct(self, direction: str) -> float:
        """Resolve direction-specific top_pct or fall back to global."""
        cfg = self.cfg
        if direction == "long" and cfg.ml_top_pct_long is not None:
            return cfg.ml_top_pct_long
        if direction == "short" and cfg.ml_top_pct_short is not None:
            return cfg.ml_top_pct_short
        return cfg.ml_top_pct

    # ------------------------------------------------------------------
    # Percentile & position sizing
    # ------------------------------------------------------------------

    def _compute_percentile(self, prob: float, direction: str) -> float:
        """
        Compute the percentile of *prob* relative to the rolling window.

        Returns a value in [0, 1].  1.0 = higher than all past scores.
        0.0 = lower than all past scores.

        Uses ONLY prior session scores (causal).  During cold-start
        (empty window), returns 0.5 (neutral).
        """
        window = self._get_ranking_window(direction)
        if len(window) == 0:
            return 0.5  # cold-start: treat as average
        n_below = sum(1 for p in window if p < prob)
        n_equal = sum(1 for p in window if p == prob)
        # Average rank for ties (midpoint method)
        percentile = (n_below + 0.5 * n_equal) / len(window)
        return max(0.0, min(1.0, percentile))

    def _compute_position_size(self, percentile: float) -> float:
        """
        Map a percentile ∈ [0, 1] to a position size.

        Modes:
          none   — always base_size (binary accept/reject handles filtering)
          linear — base_size * percentile
          convex — base_size * percentile²  (favours top trades)
          hybrid — 0 below 50th pctile, linear ramp above

        Return value is clamped to [0, base_size].
        """
        mode = self.cfg.position_sizing_mode
        base = self.cfg.base_size

        if mode == "none":
            return base

        if mode == "linear":
            size = base * percentile
        elif mode == "convex":
            size = base * (percentile ** 2)
        elif mode == "hybrid":
            if percentile < 0.5:
                size = 0.0
            else:
                size = base * (percentile - 0.5) * 2.0
        else:
            raise ValueError(
                f"Unknown position_sizing_mode: '{mode}'. "
                f"Valid: none, linear, convex, hybrid"
            )

        # Clamp: no negative, no larger than base
        return max(0.0, min(base, size))

    # ------------------------------------------------------------------
    # Multi-candidate mode
    # ------------------------------------------------------------------

    def _on_bar_multi(
        self, bar: dict, ts, bar_time, close, high, low, open_px, volume,
    ) -> list[Signal]:
        """Multi-candidate path: multiple positions, candidate ranking."""
        signals: list[Signal] = []

        # --- Exit checks for all open positions ---
        for pos_id in list(self._open_positions):
            pos = self._open_positions[pos_id]
            exit_sig = self._check_exit_multi(pos_id, pos, bar)
            if exit_sig is not None:
                signals.append(exit_sig)
                del self._open_positions[pos_id]

        # --- Entry: generate + rank candidates on decision bar ---
        if (
            not self._decided_today
            and bar_time >= self._range_end
            and self.range_set
        ):
            self._decided_today = True

            or_range = self.opening_high - self.opening_low
            if or_range <= 0:
                return signals or [_no_signal(close, ts)]

            self._ensure_model_loaded()

            candidates = self._generate_candidates(bar, or_range)
            selected = self._rank_and_select(candidates)

            logger.debug(
                "%s: %d candidates generated, %d selected (strategy=%s)",
                ts.date(), len(candidates), len(selected),
                self.cfg.selection_strategy,
            )

            for cand in selected:
                pos_id = f"{cand.strategy_type}_{ts.date().isoformat()}"
                sig_type = (
                    SignalType.LONG_ENTRY if cand.direction == "long"
                    else SignalType.SHORT_ENTRY
                )
                sig = Signal(
                    signal_type=sig_type,
                    price=cand.entry_price,
                    timestamp=ts,
                    reason=(
                        f"Multi {cand.strategy_type} "
                        f"(prob={cand.ml_prob:.3f}, size={cand.position_size:.2f})"
                    ),
                    entry_price=cand.entry_price,
                    stop_loss=cand.stop_loss,
                    take_profit=cand.take_profit,
                    position_size=cand.position_size,
                    position_id=pos_id,
                    strategy_type=cand.strategy_type,
                )
                signals.append(sig)
                self._open_positions[pos_id] = {
                    "direction": cand.direction,
                    "entry_price": cand.entry_price,
                    "sl": cand.stop_loss,
                    "tp": cand.take_profit,
                    "position_size": cand.position_size,
                    "strategy_type": cand.strategy_type,
                }

        return signals or [_no_signal(close, ts)]

    def _check_exit_multi(
        self, pos_id: str, pos: dict, bar: dict,
    ) -> Optional[Signal]:
        """Check TP/SL/EOD exits for a single open position."""
        ts = bar["timestamp"]
        bar_time = ts.time()
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        direction = pos["direction"]
        sl = pos["sl"]
        tp = pos["tp"]

        # EOD exit
        if bar_time >= self._eod_exit:
            return Signal(
                signal_type=SignalType.EXIT_EOD,
                price=close,
                timestamp=ts,
                reason=f"End-of-day exit ({pos_id})",
                entry_price=pos["entry_price"],
                stop_loss=sl,
                take_profit=tp,
                position_id=pos_id,
                strategy_type=pos.get("strategy_type", ""),
            )

        if direction == "long":
            if sl is not None and low <= sl:
                return Signal(
                    signal_type=SignalType.EXIT_SL,
                    price=sl, timestamp=ts,
                    reason=f"Stop loss hit ({sl:.2f}) [{pos_id}]",
                    entry_price=pos["entry_price"],
                    stop_loss=sl, take_profit=tp,
                    position_id=pos_id,
                    strategy_type=pos.get("strategy_type", ""),
                )
            if tp is not None and high >= tp:
                return Signal(
                    signal_type=SignalType.EXIT_TP,
                    price=tp, timestamp=ts,
                    reason=f"Take profit hit ({tp:.2f}) [{pos_id}]",
                    entry_price=pos["entry_price"],
                    stop_loss=sl, take_profit=tp,
                    position_id=pos_id,
                    strategy_type=pos.get("strategy_type", ""),
                )
        else:  # short
            if sl is not None and high >= sl:
                return Signal(
                    signal_type=SignalType.EXIT_SL,
                    price=sl, timestamp=ts,
                    reason=f"Stop loss hit ({sl:.2f}) [{pos_id}]",
                    entry_price=pos["entry_price"],
                    stop_loss=sl, take_profit=tp,
                    position_id=pos_id,
                    strategy_type=pos.get("strategy_type", ""),
                )
            if tp is not None and low <= tp:
                return Signal(
                    signal_type=SignalType.EXIT_TP,
                    price=tp, timestamp=ts,
                    reason=f"Take profit hit ({tp:.2f}) [{pos_id}]",
                    entry_price=pos["entry_price"],
                    stop_loss=sl, take_profit=tp,
                    position_id=pos_id,
                    strategy_type=pos.get("strategy_type", ""),
                )

        return None

    def _generate_candidates(
        self, bar: dict, or_range: float,
    ) -> list[TradeCandidate]:
        """
        Generate trade candidates for all EMA lengths × entry types.

        Each candidate has its own entry/SL/TP logic and features.
        """
        ts = bar["timestamp"]
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        open_px = float(bar["open"])
        volume = float(bar["volume"])

        candidates: list[TradeCandidate] = []

        for ema_length in self.cfg.ema_periods:
            eng = self._feature_engines[ema_length]
            snap = eng.snapshot
            if snap.ema is None:
                continue

            for entry_type in self.cfg.entry_types:
                new_cands = self._build_entry_candidates(
                    ema_length, entry_type, snap,
                    bar, or_range, open_px, close, high, low, volume, ts,
                )
                candidates.extend(new_cands)

        return candidates

    def _build_entry_candidates(
        self,
        ema_length: int,
        entry_type: str,
        snap: FeatureSnapshot,
        bar: dict,
        or_range: float,
        open_px: float,
        close: float,
        high: float,
        low: float,
        volume: float,
        ts,
    ) -> list[TradeCandidate]:
        """
        Build candidates for one EMA length + entry type combination.

        Entry types:
          breakout  — existing logic: close above/below EMA → enter at open
          pullback  — close near EMA (within 1 ATR): buy/sell at EMA level
          momentum  — strong move (volume expansion) in EMA direction
        """
        ema = snap.ema
        atr = snap.atr or or_range  # fallback to range if ATR not ready

        results: list[TradeCandidate] = []

        if entry_type == "breakout":
            # Long: close > EMA
            if close > ema:
                entry_px = open_px
                sl = self.opening_low
                tp = entry_px + self.cfg.reward_risk * or_range
                results.append(self._make_candidate(
                    ema_length, "breakout", "long",
                    entry_px, sl, tp, bar, snap, ts,
                ))
            # Short: close < EMA
            if close < ema and self.cfg.allow_shorts:
                entry_px = open_px
                sl = self.opening_high
                tp = entry_px - self.cfg.reward_risk * or_range
                results.append(self._make_candidate(
                    ema_length, "breakout", "short",
                    entry_px, sl, tp, bar, snap, ts,
                ))

        elif entry_type == "pullback":
            # Pullback long: close above EMA but within 1 ATR
            dist = close - ema
            if 0 < dist < atr:
                entry_px = close
                sl = ema - 0.5 * atr
                tp = entry_px + self.cfg.reward_risk * atr
                results.append(self._make_candidate(
                    ema_length, "pullback", "long",
                    entry_px, sl, tp, bar, snap, ts,
                ))
            # Pullback short: close below EMA but within 1 ATR
            dist = ema - close
            if 0 < dist < atr and self.cfg.allow_shorts:
                entry_px = close
                sl = ema + 0.5 * atr
                tp = entry_px - self.cfg.reward_risk * atr
                results.append(self._make_candidate(
                    ema_length, "pullback", "short",
                    entry_px, sl, tp, bar, snap, ts,
                ))

        elif entry_type == "momentum":
            # Momentum long: close > EMA, bar range > ATR, volume expansion
            vol_expansion = (
                volume / list(self._volume_history)[-2]
                if len(self._volume_history) >= 2
                   and list(self._volume_history)[-2] > 0
                else 1.0
            )
            bar_range = high - low
            if close > ema and bar_range > atr and vol_expansion > 1.2:
                entry_px = open_px
                sl = low  # tight SL at bar low
                tp = entry_px + self.cfg.reward_risk * bar_range
                results.append(self._make_candidate(
                    ema_length, "momentum", "long",
                    entry_px, sl, tp, bar, snap, ts,
                ))
            # Momentum short: close < EMA, bar range > ATR, volume expansion
            if (
                close < ema
                and bar_range > atr
                and vol_expansion > 1.2
                and self.cfg.allow_shorts
            ):
                entry_px = open_px
                sl = high  # tight SL at bar high
                tp = entry_px - self.cfg.reward_risk * bar_range
                results.append(self._make_candidate(
                    ema_length, "momentum", "short",
                    entry_px, sl, tp, bar, snap, ts,
                ))

        return results

    def _make_candidate(
        self,
        ema_length: int,
        entry_type: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        bar: dict,
        snap: FeatureSnapshot,
        ts,
    ) -> TradeCandidate:
        """Create a scored TradeCandidate with ML probability."""
        strategy_type = f"ema{ema_length}_{entry_type}"
        features = self._extract_features(bar, snap, direction)
        prob = self._predict_proba(features)

        return TradeCandidate(
            ema_length=ema_length,
            entry_type=entry_type,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            features=features,
            ml_prob=prob,
            timestamp=ts,
            strategy_type=strategy_type,
        )

    def _rank_and_select(
        self, candidates: list[TradeCandidate],
    ) -> list[TradeCandidate]:
        """
        Rank candidates and apply selection logic.

        Dispatches to the appropriate method based on selection_strategy:
          global_ml          — original: sort all by ML prob, filter globally
          priority           — group by entry_type priority, ML ranks within groups
          priority_ml_sizing — priority ordering, always enter, ML only for sizing
        """
        if not candidates:
            return []

        strategy = self.cfg.selection_strategy
        if strategy == "priority":
            return self._select_priority(candidates)
        elif strategy == "priority_ml_sizing":
            return self._select_priority_ml_sizing(candidates)
        else:
            return self._select_global_ml(candidates)

    def _select_global_ml(
        self, candidates: list[TradeCandidate],
    ) -> list[TradeCandidate]:
        """Original selection: sort all by ML prob descending, apply filter."""
        # Sort by ML probability descending
        candidates.sort(key=lambda c: c.ml_prob, reverse=True)
        logger.debug("select_global_ml: %d candidates", len(candidates))

        # Slots available (account for already-open positions today)
        max_new = self.cfg.max_trades_per_day - len(self._open_positions)
        if max_new <= 0:
            # Log all as rejected
            for c in candidates:
                self._log_multi_decision(c, accepted=False,
                                         reject_reason="max_trades_per_day reached")
                self._append_prob_window(c.ml_prob, c.direction)
            return []

        selected: list[TradeCandidate] = []

        for cand in candidates:
            if len(selected) >= max_new:
                self._log_multi_decision(cand, accepted=False,
                                         reject_reason="max_trades_per_day reached")
                self._append_prob_window(cand.ml_prob, cand.direction)
                continue

            # Apply selection decision (threshold / top_n / top_pct)
            accepted, rank, window_size, reject_reason = self._selection_decision(
                cand.ml_prob, cand.direction,
            )

            # Compute percentile & position sizing
            percentile = self._compute_percentile(cand.ml_prob, cand.direction)
            position_size = self._compute_position_size(percentile)

            if self.cfg.position_sizing_mode != "none" and position_size <= 0:
                accepted = False
                reject_reason = reject_reason or f"position_size=0 (pctile={percentile:.2f})"

            cand.percentile = percentile
            cand.position_size = position_size

            self._log_multi_decision(
                cand, accepted=accepted,
                rank=rank, window_size=window_size,
                reject_reason=reject_reason,
            )

            # Append to rolling window AFTER decision (causal)
            self._append_prob_window(cand.ml_prob, cand.direction)

            if accepted:
                selected.append(cand)

        return selected

    def _order_by_priority(
        self, candidates: list[TradeCandidate],
    ) -> list[TradeCandidate]:
        """
        Order candidates by entry_type priority, then ML prob within groups.

        Priority: breakout (0) > momentum (1) > pullback (2).
        Within each priority group, sort by ml_prob descending.
        """
        groups: dict[str, list[TradeCandidate]] = {}
        for c in candidates:
            groups.setdefault(c.entry_type, []).append(c)

        # Sort within each group by ml_prob descending
        for et in groups:
            groups[et].sort(key=lambda c: c.ml_prob, reverse=True)

        # Build ordered list: priority order, ML-ranked within groups
        ordered: list[TradeCandidate] = []
        for et in sorted(groups.keys(),
                         key=lambda e: STRATEGY_PRIORITY.get(e, 99)):
            ordered.extend(groups[et])
        return ordered

    def _select_priority(
        self, candidates: list[TradeCandidate],
    ) -> list[TradeCandidate]:
        """
        Priority-based selection: group by entry_type, sort by priority,
        ML ranks within groups, optional within-group threshold filter.
        """
        ordered = self._order_by_priority(candidates)

        max_new = self.cfg.max_trades_per_day - len(self._open_positions)
        if max_new <= 0:
            for c in ordered:
                self._log_multi_decision(c, accepted=False,
                                         reject_reason="max_trades_per_day reached")
                self._append_prob_window(c.ml_prob, c.direction)
            return []

        threshold = self.cfg.ml_within_group_threshold
        selected: list[TradeCandidate] = []

        for cand in ordered:
            if len(selected) >= max_new:
                self._log_multi_decision(cand, accepted=False,
                                         reject_reason="max_trades_per_day reached")
                self._append_prob_window(cand.ml_prob, cand.direction)
                continue

            # Within-group threshold filter
            accepted = True
            reject_reason = ""
            if threshold > 0 and cand.ml_prob < threshold:
                accepted = False
                reject_reason = (
                    f"below within-group threshold "
                    f"({cand.ml_prob:.3f} < {threshold:.2f})"
                )

            # Compute percentile & position sizing
            percentile = self._compute_percentile(cand.ml_prob, cand.direction)
            position_size = self._compute_position_size(percentile)

            if self.cfg.position_sizing_mode != "none" and position_size <= 0:
                accepted = False
                reject_reason = reject_reason or f"position_size=0 (pctile={percentile:.2f})"

            cand.percentile = percentile
            cand.position_size = position_size

            self._log_multi_decision(
                cand, accepted=accepted,
                reject_reason=reject_reason,
            )
            self._append_prob_window(cand.ml_prob, cand.direction)

            if accepted:
                selected.append(cand)

        return selected

    def _select_priority_ml_sizing(
        self, candidates: list[TradeCandidate],
    ) -> list[TradeCandidate]:
        """
        Priority ordering, always enter (no ML filtering), ML only for sizing.

        Breakout candidates always taken first. Position size is driven by
        ML percentile regardless of position_sizing_mode config.
        """
        ordered = self._order_by_priority(candidates)

        max_new = self.cfg.max_trades_per_day - len(self._open_positions)
        if max_new <= 0:
            for c in ordered:
                self._log_multi_decision(c, accepted=False,
                                         reject_reason="max_trades_per_day reached")
                self._append_prob_window(c.ml_prob, c.direction)
            return []

        selected: list[TradeCandidate] = []

        for cand in ordered:
            if len(selected) >= max_new:
                self._log_multi_decision(cand, accepted=False,
                                         reject_reason="max_trades_per_day reached")
                self._append_prob_window(cand.ml_prob, cand.direction)
                continue

            # Always accept — no ML filtering
            percentile = self._compute_percentile(cand.ml_prob, cand.direction)
            # ML-driven sizing: linear ramp on percentile
            base = self.cfg.base_size
            position_size = max(0.1 * base, base * percentile)

            cand.percentile = percentile
            cand.position_size = position_size

            self._log_multi_decision(
                cand, accepted=True,
                reject_reason="",
            )
            self._append_prob_window(cand.ml_prob, cand.direction)
            selected.append(cand)

        return selected

    def _log_multi_decision(
        self, cand: TradeCandidate, accepted: bool,
        rank: int = 0, window_size: int = 0,
        reject_reason: str = "",
    ):
        """Record a multi-candidate decision for diagnostics."""
        self.ml_decisions.append({
            "timestamp": cand.timestamp,
            "direction": cand.direction,
            "ema_signal": True,
            "ml_prob": cand.ml_prob,
            "selection_mode": self.cfg.ml_selection_mode,
            "accepted": accepted,
            "rank": rank,
            "window_size": window_size,
            "reject_reason": reject_reason,
            "percentile": cand.percentile,
            "position_size": cand.position_size,
            "strategy_type": cand.strategy_type,
            "ema_length": cand.ema_length,
            "entry_type": cand.entry_type,
            "threshold": self.cfg.ml_threshold if self.cfg.ml_selection_mode == "threshold" else None,
            "top_n": self._effective_top_n(cand.direction) if self.cfg.ml_selection_mode == "top_n" else None,
            "top_pct": self._effective_top_pct(cand.direction) if self.cfg.ml_selection_mode == "top_pct" else None,
            "lookback": self.cfg.ml_lookback if self.cfg.ml_selection_mode != "threshold" else None,
        })

    def _append_prob_window(self, prob: float, direction: str):
        """Append probability to rolling windows (causal)."""
        self._prob_window_all.append(prob)
        if direction == "long":
            self._prob_window_long.append(prob)
        else:
            self._prob_window_short.append(prob)

    # ------------------------------------------------------------------
    # Feature extraction for ML prediction
    # ------------------------------------------------------------------

    def _extract_features(self, bar: dict, snap: FeatureSnapshot, direction: str) -> dict:
        """
        Build a feature dict matching the columns used in training.
        Uses ONLY current/past data — no lookahead.
        """
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        open_px = float(bar["open"])
        volume = float(bar["volume"])
        ts = bar["timestamp"]

        or_range = (self.opening_high - self.opening_low) if (
            self.opening_high is not None and self.opening_low is not None
        ) else 0.0

        # NOTE: entry_px uses open_px for ALL entry types.  For pullback
        # entries the actual fill is at close, which makes f_ema_distance,
        # f_ema_distance_pct, and f_risk_points slightly inaccurate for
        # those candidates.  The ML model was trained with this same logic,
        # so changing it here would create a train/inference mismatch.
        entry_px = open_px

        features = {}

        # Price/EMA features
        features["f_price_ema"] = snap.ema if snap.ema is not None else close
        features["f_price_ema_dist"] = close - features["f_price_ema"]
        features["f_price_ema_dist_pct"] = (
            features["f_price_ema_dist"] / features["f_price_ema"]
            if features["f_price_ema"] != 0 else 0
        )
        features["f_price_ema_slope"] = snap.ema_slope if snap.ema_slope is not None else 0.0

        # Bar range
        features["f_price_bar_range"] = high - low

        # Rolling range (from feature engine)
        features["f_price_rolling_range"] = snap.rolling_range if snap.rolling_range is not None else high - low

        # Rolling return
        if len(self._close_history) >= 13:
            features[f"f_price_ret_12bar"] = (
                close / list(self._close_history)[-13] - 1
            )
        else:
            features[f"f_price_ret_12bar"] = 0.0

        # Gap (approximation: today open vs yesterday close)
        features["f_price_gap"] = 0.0
        features["f_price_gap_pct"] = 0.0

        # Volume
        features["f_vol_avg"] = (
            np.mean(list(self._volume_history)) if self._volume_history else volume
        )
        features["f_vol_relative"] = snap.relative_volume if snap.relative_volume is not None else 1.0
        features["f_vol_expansion"] = (
            volume / list(self._volume_history)[-2]
            if len(self._volume_history) >= 2 and list(self._volume_history)[-2] > 0
            else 1.0
        )

        # Volatility
        features["f_vola_tr"] = max(
            high - low,
            abs(high - (list(self._close_history)[-2] if len(self._close_history) >= 2 else close)),
            abs(low - (list(self._close_history)[-2] if len(self._close_history) >= 2 else close)),
        )
        features["f_vola_atr"] = snap.atr if snap.atr is not None else features["f_vola_tr"]
        features["f_vola_atr_norm"] = (
            features["f_vola_atr"] / close if close > 0 else 0
        )
        features["f_vola_realized"] = 0.0  # Approximate; not critical

        # Time features
        open_minutes = 9 * 60 + 30
        bar_minutes = ts.hour * 60 + ts.minute
        features["f_time_minutes_since_open"] = bar_minutes - open_minutes
        features["f_time_minutes_since_range_close"] = bar_minutes - (9 * 60 + 45)
        features["f_time_minutes_to_close"] = (16 * 60) - bar_minutes
        features["f_time_weekday"] = ts.weekday()

        # Range features
        features["f_range_high"] = self.opening_high if self.opening_high is not None else close
        features["f_range_low"] = self.opening_low if self.opening_low is not None else close
        features["f_range_size"] = or_range
        features["f_range_dist_above"] = close - features["f_range_high"]
        features["f_range_dist_below"] = features["f_range_low"] - close
        features["f_range_size_vs_atr"] = (
            or_range / features["f_vola_atr"]
            if features["f_vola_atr"] > 0 else 0
        )

        # Regime features
        ema_slope = features["f_price_ema_slope"]
        features["f_regime_trend_strength"] = abs(ema_slope)
        features["f_regime_trend_direction"] = np.sign(ema_slope)
        features["f_regime_compression"] = (
            features["f_price_bar_range"] / features["f_vola_atr"]
            if features["f_vola_atr"] > 0 else 0
        )
        features["f_regime_breakout_strength"] = features["f_regime_compression"]
        features["f_regime_vol_trend"] = (
            features["f_vol_relative"] * features["f_regime_trend_direction"]
        )

        # Candidate-specific features
        features["f_ema_distance"] = entry_px - features["f_price_ema"]
        features["f_ema_distance_pct"] = (
            features["f_ema_distance"] / features["f_price_ema"]
            if features["f_price_ema"] != 0 else 0
        )
        features["f_risk_points"] = abs(entry_px - (
            self.opening_low if direction == "long" else self.opening_high
        ))
        features["f_range_vs_atr"] = features["f_range_size_vs_atr"]
        features["f_direction_long"] = 1 if direction == "long" else 0

        return features

    def _predict_proba(self, features: dict) -> float:
        """Get probability of success from ML model."""
        if self._model is None:
            logger.warning("_predict_proba called but model is None; returning 0.5")
            return 0.5

        # Build feature vector in the right column order
        missing = [col for col in self._feature_columns if col not in features]
        if missing:
            logger.debug("Missing features (defaulting to 0): %s", missing)

        feature_vec = [features.get(col, 0.0) for col in self._feature_columns]

        X = np.array([feature_vec])
        try:
            proba = self._model.predict_proba(X)[0, 1]
        except Exception as exc:
            logger.error("predict_proba failed: %s", exc)
            raise
        return float(proba)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_day(self, new_date: datetime.date):
        self.opening_high = None
        self.opening_low = None
        self.range_set = False
        self.trade_taken = False
        self._decided_today = False
        if self.in_position:
            self.in_position = False
            self.direction = None
            self.entry_price = None
            self.tp = None
            self.sl = None
        # Multi-candidate: clear any residual position state on new day
        # (should already be empty due to EOD exit, but safety net)
        if self._open_positions:
            logger.warning(
                "Orphan positions at day reset (%s): %s",
                new_date, list(self._open_positions.keys()),
            )
        self._open_positions.clear()
        self._current_date = new_date

    def _update_opening_range(self, high: float, low: float):
        if self.opening_high is None:
            self.opening_high = high
            self.opening_low = low
        else:
            self.opening_high = max(self.opening_high, high)
            self.opening_low = min(self.opening_low, low)

    def _can_enter(self) -> bool:
        if not self.range_set:
            return False
        if self.in_position:
            return False
        if self.trade_taken:
            return False
        return True

    def _enter_long(self, entry_px, sl, tp, ts, or_range, prob, position_size=1.0) -> Signal:
        self.in_position = True
        self.trade_taken = True
        self.direction = "long"
        self.entry_price = entry_px
        self.sl = sl
        self.tp = tp
        stype = f"ema{self.cfg.ema_length}_breakout"
        return Signal(
            signal_type=SignalType.LONG_ENTRY,
            price=entry_px,
            timestamp=ts,
            reason=f"Hybrid EMA+ML long (range={or_range:.2f}, prob={prob:.3f}, size={position_size:.2f})",
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
            position_size=position_size,
            strategy_type=stype,
        )

    def _enter_short(self, entry_px, sl, tp, ts, or_range, prob, position_size=1.0) -> Signal:
        self.in_position = True
        self.trade_taken = True
        self.direction = "short"
        self.entry_price = entry_px
        self.sl = sl
        self.tp = tp
        stype = f"ema{self.cfg.ema_length}_breakout"
        return Signal(
            signal_type=SignalType.SHORT_ENTRY,
            price=entry_px,
            timestamp=ts,
            reason=f"Hybrid EMA+ML short (range={or_range:.2f}, prob={prob:.3f}, size={position_size:.2f})",
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
            position_size=position_size,
            strategy_type=stype,
        )

    def _exit(self, price, ts, sig_type, reason) -> Signal:
        stype = f"ema{self.cfg.ema_length}_breakout"
        sig = Signal(
            signal_type=sig_type,
            price=price,
            timestamp=ts,
            reason=reason,
            entry_price=self.entry_price,
            stop_loss=self.sl,
            take_profit=self.tp,
            strategy_type=stype,
        )
        self.in_position = False
        self.direction = None
        self.entry_price = None
        self.tp = None
        self.sl = None
        return sig
