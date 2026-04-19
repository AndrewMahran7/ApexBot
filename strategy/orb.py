"""
Opening Range Breakout (ORB) Strategy for MES Futures
=====================================================

Pure signal-generation logic with NO broker or execution dependency.
Receives one bar at a time and emits signals that the backtest engine
(or a future live adapter) can act on.

Ported from the original TSLA ORB strategy with these changes:
  - Removed all Alpaca API calls
  - Removed Yahoo Finance price lookups
  - Removed share-quantity / buying-power logic
  - Decisions use ONLY the bar data passed in (no lookahead)
  - All parameters come from StrategyConfig (no hardcoded values)
  - Returns structured Signal objects instead of executing orders
"""

from __future__ import annotations
import datetime
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from config.settings import StrategyConfig

logger = logging.getLogger(__name__)


class SignalType(Enum):
    NONE = auto()
    LONG_ENTRY = auto()
    SHORT_ENTRY = auto()
    EXIT_TP = auto()
    EXIT_SL = auto()
    EXIT_EOD = auto()


@dataclass
class Signal:
    signal_type: SignalType
    price: float                   # the bar close that triggered the signal
    timestamp: datetime.datetime
    reason: str = ""
    # Levels attached for the engine to use
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # Position sizing (1.0 = full base size, 0.5 = half, etc.)
    position_size: float = 1.0
    # Multi-position tracking
    position_id: str = ""       # Identifies position (for multi-position exit matching)
    strategy_type: str = ""     # e.g. "ema50_breakout"
    # Quality scoring (0.0 = unscored, higher = better)
    quality_score: float = 0.0
    # Causal audit trail — when was the direction decided?
    decision_time: Optional[datetime.datetime] = None


class ORBStrategy:
    """
    Stateful ORB strategy that processes bars one at a time.

    Call `on_bar()` with each new bar dict. It returns a Signal
    describing what (if anything) should happen.
    """

    def __init__(self, config: StrategyConfig):
        self.cfg = config

        # Parse time strings once
        self._or_start = _parse_time(config.or_start)
        self._or_end = _parse_time(config.or_end)
        self._eod_exit = _parse_time(config.eod_exit_time)
        self._max_entry = _parse_time(config.max_entry_time) if config.max_entry_time else None

        # Opening range state
        self.opening_high: Optional[float] = None
        self.opening_low: Optional[float] = None
        self.range_set: bool = False

        # Trade-of-the-day state
        self.trade_taken: bool = False
        self.in_position: bool = False
        self.direction: Optional[str] = None   # 'long' or 'short'
        self.entry_price: Optional[float] = None
        self.tp: Optional[float] = None
        self.sl: Optional[float] = None

        # Current trading day tracker
        self._current_date: Optional[datetime.date] = None

        # EMA state
        self._close_prices: deque = deque(maxlen=config.ema_length)
        self._ema: Optional[float] = None
        self._ema_alpha: float = 2.0 / (config.ema_length + 1)

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

        # --- New day reset ---
        if self._current_date is None or bar_date != self._current_date:
            self._reset_day(bar_date)

        # --- Update EMA with bar close ---
        self._update_ema(close)

        # --- Build / finalize opening range ---
        if bar_time >= self._or_start and bar_time < self._or_end:
            self._update_opening_range(high, low)
            return _no_signal(close, ts)

        if bar_time >= self._or_end and not self.range_set:
            if self.opening_high is not None and self.opening_low is not None:
                self.range_set = True

        # --- Exit logic (checked BEFORE entry so same-bar exit is possible) ---
        if self.in_position:
            # End-of-day forced exit
            if bar_time >= self._eod_exit:
                return self._exit(close, ts, SignalType.EXIT_EOD, "End-of-day exit")

            if self.direction == "long":
                # Stop loss: triggered if bar low penetrates SL
                if self.sl is not None and low <= self.sl:
                    fill_price = self.sl
                    return self._exit(fill_price, ts, SignalType.EXIT_SL,
                                      f"Stop loss hit ({self.sl:.2f})")
                # Take profit: triggered if bar high reaches TP
                if self.tp is not None and high >= self.tp:
                    fill_price = self.tp
                    return self._exit(fill_price, ts, SignalType.EXIT_TP,
                                      f"Take profit hit ({self.tp:.2f})")
            else:  # short
                # Stop loss: triggered if bar high penetrates SL (above entry)
                if self.sl is not None and high >= self.sl:
                    fill_price = self.sl
                    return self._exit(fill_price, ts, SignalType.EXIT_SL,
                                      f"Stop loss hit ({self.sl:.2f})")
                # Take profit: triggered if bar low reaches TP (below entry)
                if self.tp is not None and low <= self.tp:
                    fill_price = self.tp
                    return self._exit(fill_price, ts, SignalType.EXIT_TP,
                                      f"Take profit hit ({self.tp:.2f})")

        # --- Entry logic ---
        if self._can_enter():
            or_range = self.opening_high - self.opening_low
            if or_range <= 0:
                return _no_signal(close, ts)

            # Min range filter
            if self.cfg.min_range_points > 0 and or_range < self.cfg.min_range_points:
                return _no_signal(close, ts)

            # Max entry time filter
            if self._max_entry is not None and bar_time > self._max_entry:
                return _no_signal(close, ts)

            # Long entry: bar high breaks above opening range high
            if high > self.opening_high:
                entry_px = self.opening_high
                sl = self.opening_low
                tp = entry_px + (self.cfg.reward_risk_ratio * or_range)

                # EMA filter: entry price must be above EMA for longs
                if self.cfg.ema_enabled and self._ema is not None:
                    if entry_px <= self._ema:
                        logger.debug("Long rejected: entry %.2f <= EMA %.2f", entry_px, self._ema)
                        return _no_signal(close, ts)

                return self._enter_long(entry_px, sl, tp, ts, or_range)

            # Short entry: bar low breaks below opening range low
            if self.cfg.shorts_enabled and low < self.opening_low:
                entry_px = self.opening_low
                sl = self.opening_high
                tp = entry_px - (self.cfg.reward_risk_ratio * or_range)

                # EMA filter: entry price must be below EMA for shorts
                if self.cfg.ema_enabled and self._ema is not None:
                    if entry_px >= self._ema:
                        logger.debug("Short rejected: entry %.2f >= EMA %.2f", entry_px, self._ema)
                        return _no_signal(close, ts)

                return self._enter_short(entry_px, sl, tp, ts, or_range)

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
        self._close_prices.clear()
        self._ema = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_day(self, new_date: datetime.date):
        logger.debug("ORB day reset: %s", new_date)
        self.opening_high = None
        self.opening_low = None
        self.range_set = False
        self.trade_taken = False
        if self.in_position:
            self.in_position = False
            self.direction = None
            self.entry_price = None
            self.tp = None
            self.sl = None
        self._current_date = new_date

    def _update_ema(self, close: float):
        self._close_prices.append(close)
        if self._ema is None:
            if len(self._close_prices) >= 10:
                self._ema = sum(list(self._close_prices)[-10:]) / 10
        else:
            self._ema = (close * self._ema_alpha) + (self._ema * (1 - self._ema_alpha))

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

    def _enter_long(self, entry_px, sl, tp, ts, or_range) -> Signal:
        self.in_position = True
        self.trade_taken = True
        self.direction = "long"
        self.entry_price = entry_px
        self.sl = sl
        self.tp = tp
        logger.debug("ORB LONG entry @ %.2f  SL=%.2f TP=%.2f  range=%.2f", entry_px, sl, tp, or_range)
        return Signal(
            signal_type=SignalType.LONG_ENTRY,
            price=entry_px,
            timestamp=ts,
            reason=f"ORB long breakout (range={or_range:.2f})",
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
        )

    def _enter_short(self, entry_px, sl, tp, ts, or_range) -> Signal:
        self.in_position = True
        self.trade_taken = True
        self.direction = "short"
        self.entry_price = entry_px
        self.sl = sl
        self.tp = tp
        logger.debug("ORB SHORT entry @ %.2f  SL=%.2f TP=%.2f  range=%.2f", entry_px, sl, tp, or_range)
        return Signal(
            signal_type=SignalType.SHORT_ENTRY,
            price=entry_px,
            timestamp=ts,
            reason=f"ORB short breakdown (range={or_range:.2f})",
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
        )

    def _exit(self, price, ts, sig_type, reason) -> Signal:
        logger.debug("ORB exit %s @ %.2f: %s", sig_type.name, price, reason)
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


def _no_signal(price: float, ts: datetime.datetime) -> Signal:
    return Signal(signal_type=SignalType.NONE, price=price, timestamp=ts)


def _parse_time(s: str) -> datetime.time:
    parts = s.split(":")
    return datetime.time(int(parts[0]), int(parts[1]))
