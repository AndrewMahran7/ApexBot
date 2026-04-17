"""
Adaptive Regime Breakout Strategy for MES Futures
==================================================

A regime-aware breakout/continuation strategy that classifies each
session into market regimes and only trades when conditions are
favorable. Supports both long and short entries with multi-signal
confirmation.

Design:
  - Classify session into TREND / BREAKOUT / RANGE / DEAD regime
  - Only allow entries in TREND and BREAKOUT regimes
  - Require multiple confirmations (EMA, volume, ATR, range size)
  - Conservative bar-based execution: no lookahead, no hidden info
  - Emits Signal objects consumed by the backtest engine

All parameters come from AdaptiveRegimeConfig. No hardcoded values.
"""

from __future__ import annotations
import datetime
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

from config.settings import AdaptiveRegimeConfig
from strategy.orb import SignalType, Signal, _parse_time
from strategy.features import FeatureEngine, FeatureSnapshot
from strategy.regimes import Regime, RegimeDiagnostics, classify_regime


def _no_signal(close: float, ts: datetime.datetime) -> Signal:
    return Signal(signal_type=SignalType.NONE, price=close, timestamp=ts)


@dataclass
class FilterResult:
    """Diagnostic record of which filters passed/failed for a potential entry."""
    direction: str = ""          # 'long' or 'short'
    range_size_ok: bool = False
    ema_filter_ok: bool = False
    ema_slope_ok: bool = False
    volume_ok: bool = False
    atr_ok: bool = False
    timing_ok: bool = False
    breakout_cleared: bool = False
    breakout_strength_ok: bool = False
    regime_allows: bool = False
    score: int = 0               # number of passed filters
    min_score: int = 0           # required minimum
    breakout_strength: float = 0.0  # how far price exceeded trigger level
    ema_slope_value: float = 0.0    # actual EMA slope for diagnostics

    @property
    def passed(self) -> bool:
        return self.regime_allows and self.breakout_strength_ok and self.score >= self.min_score

    def summary(self) -> str:
        checks = [
            f"regime={'PASS' if self.regime_allows else 'FAIL'}",
            f"range={'PASS' if self.range_size_ok else 'FAIL'}",
            f"ema={'PASS' if self.ema_filter_ok else 'FAIL'}",
            f"slope={'PASS' if self.ema_slope_ok else 'FAIL'}({self.ema_slope_value:+.3f})",
            f"vol={'PASS' if self.volume_ok else 'FAIL'}",
            f"atr={'PASS' if self.atr_ok else 'FAIL'}",
            f"time={'PASS' if self.timing_ok else 'FAIL'}",
            f"breakout={'PASS' if self.breakout_cleared else 'FAIL'}",
            f"bkStr={'PASS' if self.breakout_strength_ok else 'FAIL'}({self.breakout_strength:.2f})",
            f"score={self.score}/{self.min_score}",
        ]
        return " | ".join(checks)


@dataclass
class DayDiagnostic:
    """Per-day diagnostic record for research inspection."""
    date: datetime.date
    regime: str = ""
    regime_reason: str = ""
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    or_range: float = 0.0
    ema: Optional[float] = None
    ema_slope: Optional[float] = None
    atr: Optional[float] = None
    relative_volume: Optional[float] = None
    trade_taken: bool = False
    trade_direction: str = ""
    filter_detail: str = ""
    skip_reason: str = ""
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: str = ""
    filter_score: int = 0
    filter_min_score: int = 0
    breakout_distance: float = 0.0
    breakout_strength: float = 0.0
    ema_slope_value: float = 0.0
    preferred_direction: str = ""


class AdaptiveRegimeStrategy:
    """
    Regime-aware breakout strategy that processes bars one at a time.

    Call `on_bar()` with each new bar dict. Returns a Signal.
    After the run, inspect `diagnostics` for per-day regime/filter info.
    """

    def __init__(self, config: AdaptiveRegimeConfig):
        self.cfg = config

        # Parse times once
        self._range_start = _parse_time(config.range_start_time)
        self._range_end = _parse_time(config.range_end_time)
        self._eod_exit = _parse_time(config.end_of_day_exit_time)
        self._max_entry = _parse_time(config.max_entry_time) if config.max_entry_time else None

        # Feature engine
        self._features = FeatureEngine(
            ema_length=config.ema_length,
            ema_slope_lookback=config.ema_slope_lookback,
            atr_length=config.atr_length,
            volume_lookback=config.volume_lookback,
        )

        # Opening range state
        self.opening_high: Optional[float] = None
        self.opening_low: Optional[float] = None
        self.range_set: bool = False

        # Trade state
        self.trade_taken: bool = False
        self.in_position: bool = False
        self.direction: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.tp: Optional[float] = None
        self.sl: Optional[float] = None

        # Day tracking
        self._current_date: Optional[datetime.date] = None
        self._regime_diag: Optional[RegimeDiagnostics] = None
        self._day_diagnosed: bool = False

        # Research diagnostics
        self.diagnostics: list[DayDiagnostic] = []
        self._current_day_diag: Optional[DayDiagnostic] = None

        # Direction priority (set per-session during regime classification)
        self._preferred_directions: list[str] = ["long", "short"]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_bar(self, bar: dict) -> Signal:
        """
        Process a single OHLCV bar and return a Signal.

        bar keys: timestamp (datetime), open, high, low, close, volume
        """
        ts: datetime.datetime = bar["timestamp"]
        bar_date = ts.date()
        bar_time = ts.time()
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        volume = float(bar["volume"])

        # --- New day reset ---
        if self._current_date is None or bar_date != self._current_date:
            self._finalize_day_diagnostic()
            self._reset_day(bar_date)

        # --- Update features every bar ---
        self._features.update(high, low, close, volume)

        # --- Build opening range ---
        if bar_time >= self._range_start and bar_time < self._range_end:
            self._update_opening_range(high, low)
            return _no_signal(close, ts)

        # --- Finalize range ---
        if bar_time >= self._range_end and not self.range_set:
            if self.opening_high is not None and self.opening_low is not None:
                self.range_set = True
                self._classify_regime()

        # --- Exit logic (before entry, conservative) ---
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

        # --- Entry logic ---
        if self._can_enter():
            or_range = self.opening_high - self.opening_low
            if or_range <= 0:
                return _no_signal(close, ts)

            features = self._features.snapshot
            buf_long = self.cfg.breakout_buffer_points

            # Resolve short-side buffer (asymmetric)
            if self.cfg.strict_shorts:
                buf_short = self.cfg.strict_short_buffer
            elif self.cfg.short_breakout_buffer_points is not None:
                buf_short = self.cfg.short_breakout_buffer_points
            else:
                buf_short = self.cfg.breakout_buffer_points

            # Check directions in preferred order (set by regime classifier
            # based on EMA slope — avoids always-long-first bias)
            for direction in self._preferred_directions:
                if direction == "long" and self.cfg.allow_long and high > (self.opening_high + buf_long):
                    bdist = high - (self.opening_high + buf_long)
                    filt = self._evaluate_filters("long", or_range, features, bar_time, bdist)
                    if filt.passed:
                        entry_px = self.opening_high + buf_long
                        sl = self.opening_low
                        tp = entry_px + (self.cfg.reward_risk * or_range)
                        self._record_trade_diagnostic("long", filt, entry_px, sl, tp, bdist)
                        return self._enter_long(entry_px, sl, tp, ts, or_range, filt)
                    else:
                        self._record_skip_diagnostic("long", filt)

                elif direction == "short" and self.cfg.allow_short and low < (self.opening_low - buf_short):
                    bdist = (self.opening_low - buf_short) - low
                    filt = self._evaluate_filters("short", or_range, features, bar_time, bdist)
                    if filt.passed:
                        entry_px = self.opening_low - buf_short
                        sl = self.opening_high
                        tp = entry_px - (self.cfg.reward_risk * or_range)
                        self._record_trade_diagnostic("short", filt, entry_px, sl, tp, bdist)
                        return self._enter_short(entry_px, sl, tp, ts, or_range, filt)
                    else:
                        self._record_skip_diagnostic("short", filt)

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
        self._current_date = None
        self._regime_diag = None
        self._day_diagnosed = False
        self._features.reset()
        self.diagnostics = []
        self._current_day_diag = None
        self._preferred_directions = ["long", "short"]

    # ------------------------------------------------------------------
    # Filter evaluation
    # ------------------------------------------------------------------

    def _evaluate_filters(
        self, direction: str, or_range: float,
        features: FeatureSnapshot, bar_time: datetime.time,
        breakout_distance: float = 0.0,
    ) -> FilterResult:
        """
        Check all confirmation filters and return a scored result.

        Uses asymmetric min_score / buffer depending on direction.
        Weights EMA alignment and slope more heavily in scoring.
        """
        cfg = self.cfg

        # Resolve per-direction min_score
        if direction == "short" and cfg.strict_shorts:
            min_score = cfg.strict_short_min_score
        elif direction == "short" and cfg.short_min_score is not None:
            min_score = cfg.short_min_score
        elif direction == "long" and cfg.long_min_score is not None:
            min_score = cfg.long_min_score
        else:
            min_score = cfg.min_confirmation_score

        filt = FilterResult(direction=direction, min_score=min_score)

        # 1. Regime filter (binary gate — not scored)
        regime = self._regime_diag.regime if self._regime_diag else Regime.RANGE
        filt.regime_allows = regime in (Regime.TREND, Regime.BREAKOUT)

        score = 0

        # 2. Range size filter (weight: 1)
        if cfg.min_range_points <= or_range <= cfg.max_range_points:
            filt.range_size_ok = True
            score += 1

        # 3. EMA direction filter (weight: 2 — key signal)
        if cfg.ema_enabled and features.ema is not None:
            if direction == "long" and (self.opening_high + cfg.breakout_buffer_points) > features.ema:
                filt.ema_filter_ok = True
                score += 2
            elif direction == "short" and (self.opening_low - cfg.breakout_buffer_points) < features.ema:
                filt.ema_filter_ok = True
                score += 2
            # If EMA not available yet, limited credit
        elif not cfg.ema_enabled:
            filt.ema_filter_ok = True
            score += 1
        elif features.ema is None:
            filt.ema_filter_ok = True
            score += 1

        # 4. EMA slope filter (weight: 2 — key signal)
        ema_slope_val = features.ema_slope if features.ema_slope is not None else 0.0
        filt.ema_slope_value = ema_slope_val

        # Resolve per-direction slope minimum
        short_slope_min = 0.0
        if direction == "short":
            if cfg.strict_shorts and cfg.strict_short_ema_slope_min:
                short_slope_min = cfg.strict_short_ema_slope_min
            elif cfg.short_ema_slope_min is not None:
                short_slope_min = cfg.short_ema_slope_min

        if cfg.ema_slope_enabled and features.ema_slope is not None:
            if direction == "long" and features.ema_slope > 0:
                filt.ema_slope_ok = True
                score += 2
            elif direction == "short" and features.ema_slope < 0:
                if abs(features.ema_slope) >= short_slope_min:
                    filt.ema_slope_ok = True
                    score += 2
        elif not cfg.ema_slope_enabled:
            filt.ema_slope_ok = True
            score += 1
        elif features.ema_slope is None:
            filt.ema_slope_ok = True
            score += 1

        # 5. Volume filter (weight: 1)
        if cfg.volume_filter_enabled and features.relative_volume is not None:
            if features.relative_volume >= cfg.volume_threshold_ratio:
                filt.volume_ok = True
                score += 1
        elif not cfg.volume_filter_enabled:
            filt.volume_ok = True
            score += 1
        elif features.relative_volume is None:
            filt.volume_ok = True
            score += 1

        # 6. ATR filter (weight: 1)
        if cfg.atr_filter_enabled and features.atr is not None:
            if features.atr >= cfg.atr_min_threshold:
                filt.atr_ok = True
                score += 1
        elif not cfg.atr_filter_enabled:
            filt.atr_ok = True
            score += 1
        elif features.atr is None:
            filt.atr_ok = True
            score += 1

        # 7. Timing filter (weight: 1)
        if self._max_entry is not None:
            if bar_time <= self._max_entry:
                filt.timing_ok = True
                score += 1
        else:
            filt.timing_ok = True
            score += 1

        # 8. Breakout cleared buffer (weight: 1)
        filt.breakout_cleared = True  # already checked in on_bar
        score += 1

        filt.score = score

        # 9. Breakout strength gate (binary, not scored)
        filt.breakout_strength = breakout_distance
        if breakout_distance >= cfg.min_breakout_strength:
            filt.breakout_strength_ok = True

        return filt

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def _classify_regime(self):
        """Classify regime once per day after range is set."""
        if self._day_diagnosed:
            return

        or_range = (self.opening_high - self.opening_low) if (
            self.opening_high is not None and self.opening_low is not None
        ) else 0.0

        features = self._features.snapshot
        self._regime_diag = classify_regime(
            or_range=or_range,
            features=features,
            ema_slope_threshold=self.cfg.regime_ema_slope_threshold,
            range_ratio_threshold=self.cfg.regime_range_ratio_threshold,
            volume_ratio_threshold=self.cfg.regime_volume_ratio_threshold,
            dead_atr_ratio=self.cfg.regime_dead_atr_ratio,
        )
        self._day_diagnosed = True
        logger.debug(
            "Regime classified: %s (%s) | OR=%.2f | ema=%.4f slope=%.6f",
            self._regime_diag.regime.name,
            self._regime_diag.reason,
            or_range,
            features.ema if features.ema is not None else 0.0,
            features.ema_slope if features.ema_slope is not None else 0.0,
        )

        # Set direction priority based on EMA slope.
        # In a downtrend, check short first so shorts aren't blocked
        # by the long-entry check consuming the one-trade-per-day slot.
        if features.ema_slope is not None and features.ema_slope < 0:
            self._preferred_directions = ["short", "long"]
        else:
            self._preferred_directions = ["long", "short"]

        # Record diagnostics
        if self._current_day_diag is not None:
            diag = self._current_day_diag
            diag.regime = self._regime_diag.regime.name
            diag.regime_reason = self._regime_diag.reason
            diag.or_high = self.opening_high
            diag.or_low = self.opening_low
            diag.or_range = or_range
            diag.ema = features.ema
            diag.ema_slope = features.ema_slope
            diag.atr = features.atr
            diag.relative_volume = features.relative_volume
            diag.preferred_direction = self._preferred_directions[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_day(self, new_date: datetime.date):
        logger.debug("Day reset -> %s (was %s, in_position=%s)", new_date, self._current_date, self.in_position)
        self.opening_high = None
        self.opening_low = None
        self.range_set = False
        self.trade_taken = False
        self._regime_diag = None
        self._day_diagnosed = False
        if self.in_position:
            self.in_position = False
            self.direction = None
            self.entry_price = None
            self.tp = None
            self.sl = None
        self._current_date = new_date
        self._current_day_diag = DayDiagnostic(date=new_date)

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
        if self.cfg.one_trade_per_day and self.trade_taken:
            return False
        if self.opening_high is None or self.opening_low is None:
            return False
        return True

    def _enter_long(self, entry_px, sl, tp, ts, or_range, filt: FilterResult) -> Signal:
        self.in_position = True
        self.trade_taken = True
        self.direction = "long"
        self.entry_price = entry_px
        self.sl = sl
        self.tp = tp
        regime_name = self._regime_diag.regime.name if self._regime_diag else "?"
        logger.debug(
            "LONG entry: px=%.2f sl=%.2f tp=%.2f regime=%s score=%d",
            entry_px, sl, tp, regime_name, filt.score,
        )
        return Signal(
            signal_type=SignalType.LONG_ENTRY,
            price=entry_px,
            timestamp=ts,
            reason=f"Adaptive long breakout [{regime_name}] (range={or_range:.2f}, score={filt.score})",
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
        )

    def _enter_short(self, entry_px, sl, tp, ts, or_range, filt: FilterResult) -> Signal:
        self.in_position = True
        self.trade_taken = True
        self.direction = "short"
        self.entry_price = entry_px
        self.sl = sl
        self.tp = tp
        regime_name = self._regime_diag.regime.name if self._regime_diag else "?"
        logger.debug(
            "SHORT entry: px=%.2f sl=%.2f tp=%.2f regime=%s score=%d",
            entry_px, sl, tp, regime_name, filt.score,
        )
        return Signal(
            signal_type=SignalType.SHORT_ENTRY,
            price=entry_px,
            timestamp=ts,
            reason=f"Adaptive short breakdown [{regime_name}] (range={or_range:.2f}, score={filt.score})",
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
        )

    def _exit(self, price, ts, sig_type, reason) -> Signal:
        logger.debug("EXIT %s @ %.2f: %s", sig_type.name, price, reason)
        if self._current_day_diag is not None:
            self._current_day_diag.exit_reason = reason
        sig = Signal(
            signal_type=sig_type,
            price=price,
            timestamp=ts,
            reason=reason,
            entry_price=self.entry_price,
            stop_loss=self.sl,
            take_profit=self.tp,
        )
        self.in_position = False
        self.direction = None
        self.entry_price = None
        self.tp = None
        self.sl = None
        return sig

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def _record_trade_diagnostic(self, direction: str, filt: FilterResult,
                                entry_px=None, sl=None, tp=None, breakout_dist=0.0):
        if self._current_day_diag is not None:
            self._current_day_diag.trade_taken = True
            self._current_day_diag.trade_direction = direction
            self._current_day_diag.filter_detail = filt.summary()
            self._current_day_diag.entry_price = entry_px
            self._current_day_diag.stop_loss = sl
            self._current_day_diag.take_profit = tp
            self._current_day_diag.filter_score = filt.score
            self._current_day_diag.filter_min_score = filt.min_score
            self._current_day_diag.breakout_distance = breakout_dist
            self._current_day_diag.breakout_strength = filt.breakout_strength
            self._current_day_diag.ema_slope_value = filt.ema_slope_value
            self._current_day_diag.preferred_direction = self._preferred_directions[0]

    def _record_skip_diagnostic(self, direction: str, filt: FilterResult):
        if self._current_day_diag is not None and not self._current_day_diag.trade_taken:
            self._current_day_diag.skip_reason = f"{direction}: {filt.summary()}"
            self._current_day_diag.filter_score = filt.score
            self._current_day_diag.filter_min_score = filt.min_score
            self._current_day_diag.breakout_strength = filt.breakout_strength
            self._current_day_diag.ema_slope_value = filt.ema_slope_value
            self._current_day_diag.preferred_direction = self._preferred_directions[0]

    def _finalize_day_diagnostic(self):
        if self._current_day_diag is not None:
            if not self._current_day_diag.regime:
                self._current_day_diag.regime = "NO_RANGE"
                self._current_day_diag.skip_reason = "Range never set"
            self.diagnostics.append(self._current_day_diag)
            if len(self.diagnostics) > 5000:
                logger.warning("Diagnostics list exceeds 5000 entries, trimming oldest")
                self.diagnostics = self.diagnostics[-5000:]
