"""
Intraday Strategy Suite for Prop Challenge
===========================================

Three complementary intraday strategies that generate trades throughout
the trading day (not just at the opening range):

  1. VWAPBounce    — Long near VWAP support, short near resistance
  2. IntradayMomentum — Breakout beyond short-term range with volume
  3. MeanReversion — Fade extreme moves using RSI / VWAP deviation

Each strategy:
  - Processes one bar at a time (same interface as ORBStrategy)
  - Returns Signal objects compatible with StrategyEngine
  - Tracks its own open positions and emits SL/TP/EOD exits
  - Only trades during RTH (09:30–15:50 ET)

Usage:
    from strategy.intraday_strategies import (
        VWAPBounce, IntradayMomentum, MeanReversion,
        IntradayConfig,
    )
    cfg = IntradayConfig()
    strat = VWAPBounce(cfg)
    signals = strat.on_bar(bar_dict)
"""

from __future__ import annotations

import datetime
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from strategy.orb import Signal, SignalType

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class IntradayConfig:
    """Shared configuration for all intraday strategies."""

    timezone: str = "America/New_York"
    session_open: str = "09:30"
    eod_exit_time: str = "15:50"
    entry_start: str = "10:00"         # Don't trade in the opening range window
    entry_end: str = "15:30"           # No new entries after this

    # VWAP Bounce
    vwap_band_mult: float = 0.3        # VWAP ± mult * ATR for bounce zone
    vwap_reward_risk: float = 1.5

    # Intraday Momentum
    momentum_lookback: int = 3         # Bars for short-term range
    momentum_breakout_mult: float = 1.2  # Range multiplier for breakout
    momentum_volume_mult: float = 1.3  # Volume > mult * avg → confirmed
    momentum_reward_risk: float = 1.5

    # Mean Reversion
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    reversion_reward_risk: float = 1.2
    vwap_deviation_threshold: float = 1.5  # ATR units from VWAP

    # ATR
    atr_period: int = 20

    # Risk per trade (SL distance in ATR units)
    sl_atr_mult: float = 1.0

    # Position sizing (base)
    base_size: float = 1.0

    # Max concurrent per strategy
    max_positions_per_strategy: int = 1

    # Cooldown: min bars between entries for same strategy
    entry_cooldown_bars: int = 6       # 30 min at 5-min bars

    # Quality scoring
    min_quality_score: float = 0.6     # Minimum quality to accept entry


def _parse_time(t: str) -> datetime.time:
    parts = t.split(":")
    return datetime.time(int(parts[0]), int(parts[1]))


# ------------------------------------------------------------------
# Shared indicator state
# ------------------------------------------------------------------

class _IndicatorState:
    """Rolling indicators shared across intraday strategies."""

    def __init__(self, atr_period: int = 20, rsi_period: int = 14):
        self._atr_period = atr_period
        self._rsi_period = rsi_period

        # Price history
        self._highs: deque[float] = deque(maxlen=atr_period + 1)
        self._lows: deque[float] = deque(maxlen=atr_period + 1)
        self._closes: deque[float] = deque(maxlen=max(atr_period, rsi_period) + 1)
        self._volumes: deque[float] = deque(maxlen=atr_period + 1)

        # VWAP state (resets daily)
        self._vwap_cum_vp: float = 0.0  # cumulative volume * price
        self._vwap_cum_vol: float = 0.0  # cumulative volume
        self._vwap: float = 0.0
        self._current_date: Optional[datetime.date] = None

        # ATR
        self._atr: float = 0.0
        self._tr_values: deque[float] = deque(maxlen=atr_period)

        # RSI
        self._rsi: float = 50.0
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._rsi_initialized: bool = False
        self._rsi_bar_count: int = 0

        # Volume average
        self._avg_volume: float = 0.0

        self._bar_count: int = 0

    def update(self, bar: dict) -> None:
        """Update all indicators with a new bar."""
        ts = bar["timestamp"]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        volume = float(bar["volume"])
        bar_date = ts.date() if hasattr(ts, 'date') else ts

        # Reset VWAP on new day
        if self._current_date != bar_date:
            self._vwap_cum_vp = 0.0
            self._vwap_cum_vol = 0.0
            self._current_date = bar_date

        # Store prices
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._volumes.append(volume)
        self._bar_count += 1

        # VWAP
        typical = (high + low + close) / 3.0
        self._vwap_cum_vp += typical * volume
        self._vwap_cum_vol += volume
        if self._vwap_cum_vol > 0:
            self._vwap = self._vwap_cum_vp / self._vwap_cum_vol

        # ATR (True Range)
        if len(self._closes) >= 2:
            prev_close = self._closes[-2]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        else:
            tr = high - low
        self._tr_values.append(tr)
        if len(self._tr_values) >= self._atr_period:
            self._atr = sum(self._tr_values) / len(self._tr_values)

        # Volume average
        if len(self._volumes) >= self._atr_period:
            self._avg_volume = sum(self._volumes) / len(self._volumes)

        # RSI
        self._update_rsi(close)

    def _update_rsi(self, close: float) -> None:
        if len(self._closes) < 2:
            return

        change = close - self._closes[-2]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._rsi_bar_count += 1

        period = self._rsi_period
        if self._rsi_bar_count < period:
            self._avg_gain += gain
            self._avg_loss += loss
            return

        if not self._rsi_initialized:
            self._avg_gain = (self._avg_gain + gain) / period
            self._avg_loss = (self._avg_loss + loss) / period
            self._rsi_initialized = True
        else:
            self._avg_gain = (self._avg_gain * (period - 1) + gain) / period
            self._avg_loss = (self._avg_loss * (period - 1) + loss) / period

        if self._avg_loss == 0:
            self._rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._rsi = 100.0 - (100.0 / (1.0 + rs))

    @property
    def vwap(self) -> float:
        return self._vwap

    @property
    def atr(self) -> float:
        return self._atr

    @property
    def rsi(self) -> float:
        return self._rsi

    @property
    def avg_volume(self) -> float:
        return self._avg_volume

    @property
    def ready(self) -> bool:
        """True when enough bars have been processed for indicators."""
        return (
            self._bar_count >= self._atr_period
            and self._atr > 0
            and self._rsi_initialized
        )

    @property
    def recent_highs(self) -> list[float]:
        return list(self._highs)

    @property
    def recent_lows(self) -> list[float]:
        return list(self._lows)

    @property
    def recent_closes(self) -> list[float]:
        return list(self._closes)


# ------------------------------------------------------------------
# Quality scoring
# ------------------------------------------------------------------

def compute_quality_score(
    indicators: _IndicatorState,
    bar: dict,
    direction: str,
) -> float:
    """Compute a quality score in [0, 1] for an intraday entry signal.

    Components (equal weight, each in [0, 1]):
      1. VWAP distance  — closer to VWAP = higher (better mean-level trade)
      2. Volume spike   — volume relative to average (capped at 2x)
      3. Trend strength — recent close slope aligned with direction
      4. Volatility     — moderate ATR vs recent average (bell curve)

    Returns 0.0 if indicators are not ready.
    """
    if not indicators.ready or indicators.atr <= 0:
        return 0.0

    close = float(bar["close"])
    volume = float(bar["volume"])
    vwap = indicators.vwap
    atr = indicators.atr
    avg_vol = indicators.avg_volume

    # 1. VWAP distance: normalised by ATR, closer = better (max score at 0 ATR away)
    vwap_dist = abs(close - vwap) / atr if atr > 0 else 3.0
    vwap_score = max(0.0, 1.0 - vwap_dist / 3.0)  # 0 at 3+ ATR away

    # 2. Volume spike: ratio of current volume to average (capped contribution)
    if avg_vol > 0 and volume > 0:
        vol_ratio = volume / avg_vol
        vol_score = min(1.0, vol_ratio / 2.0)  # 1.0 at 2x avg or more
    else:
        vol_score = 0.3  # neutral when no volume data

    # 3. Trend strength: slope of recent closes aligned with trade direction
    closes = indicators.recent_closes
    if len(closes) >= 5:
        # Simple linear slope over last 5 bars, normalised by ATR
        recent = closes[-5:]
        slope = (recent[-1] - recent[0]) / (4 * atr) if atr > 0 else 0.0
        if direction == "long":
            trend_score = min(1.0, max(0.0, 0.5 + slope))
        else:
            trend_score = min(1.0, max(0.0, 0.5 - slope))
    else:
        trend_score = 0.5  # neutral

    # 4. Volatility context: prefer moderate ATR (not too flat, not too wild)
    # Use ratio of current ATR to a reference (e.g. close * 0.002 ~ typical 5min)
    if close > 0:
        atr_pct = atr / close
        # Bell curve centered at 0.15% (typical 5-min MES ATR/close)
        ideal_atr_pct = 0.0015
        vol_deviation = abs(atr_pct - ideal_atr_pct) / ideal_atr_pct
        volatility_score = max(0.0, 1.0 - vol_deviation / 3.0)
    else:
        volatility_score = 0.5

    # Weighted average (equal weights)
    score = 0.25 * vwap_score + 0.25 * vol_score + 0.25 * trend_score + 0.25 * volatility_score
    return round(min(1.0, max(0.0, score)), 4)


# ------------------------------------------------------------------
# Position tracker (shared logic for all strategies)
# ------------------------------------------------------------------

@dataclass
class _OpenPosition:
    position_id: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_type: str
    entry_time: datetime.datetime
    size: float = 1.0


class _PositionTracker:
    """Tracks open positions and generates exit signals."""

    def __init__(self, eod_exit_time: str = "15:50"):
        self._positions: dict[str, _OpenPosition] = {}
        self._eod_time = _parse_time(eod_exit_time)

    def add(self, pos: _OpenPosition) -> None:
        self._positions[pos.position_id] = pos

    def remove(self, position_id: str) -> Optional[_OpenPosition]:
        return self._positions.pop(position_id, None)

    @property
    def count(self) -> int:
        return len(self._positions)

    @property
    def positions(self) -> dict[str, _OpenPosition]:
        return self._positions

    def check_exits(self, bar: dict) -> list[Signal]:
        """Check all open positions for SL/TP/EOD exits."""
        exits: list[Signal] = []
        ts = bar["timestamp"]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        bar_time = ts.time() if hasattr(ts, 'time') else ts

        to_close: list[str] = []

        for pid, pos in self._positions.items():
            exit_signal = None

            # EOD exit
            if bar_time >= self._eod_time:
                exit_signal = Signal(
                    signal_type=SignalType.EXIT_EOD,
                    price=close,
                    timestamp=ts,
                    reason=f"EOD exit ({pos.strategy_type})",
                    position_id=pid,
                    strategy_type=pos.strategy_type,
                    position_size=pos.size,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                )
            # Stop loss
            elif (pos.direction == "long" and low <= pos.stop_loss):
                exit_signal = Signal(
                    signal_type=SignalType.EXIT_SL,
                    price=pos.stop_loss,
                    timestamp=ts,
                    reason=f"Stop loss hit ({pos.stop_loss:.2f})",
                    position_id=pid,
                    strategy_type=pos.strategy_type,
                    position_size=pos.size,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                )
            elif (pos.direction == "short" and high >= pos.stop_loss):
                exit_signal = Signal(
                    signal_type=SignalType.EXIT_SL,
                    price=pos.stop_loss,
                    timestamp=ts,
                    reason=f"Stop loss hit ({pos.stop_loss:.2f})",
                    position_id=pid,
                    strategy_type=pos.strategy_type,
                    position_size=pos.size,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                )
            # Take profit
            elif (pos.direction == "long" and high >= pos.take_profit):
                exit_signal = Signal(
                    signal_type=SignalType.EXIT_TP,
                    price=pos.take_profit,
                    timestamp=ts,
                    reason=f"Take profit ({pos.take_profit:.2f})",
                    position_id=pid,
                    strategy_type=pos.strategy_type,
                    position_size=pos.size,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                )
            elif (pos.direction == "short" and low <= pos.take_profit):
                exit_signal = Signal(
                    signal_type=SignalType.EXIT_TP,
                    price=pos.take_profit,
                    timestamp=ts,
                    reason=f"Take profit ({pos.take_profit:.2f})",
                    position_id=pid,
                    strategy_type=pos.strategy_type,
                    position_size=pos.size,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                )

            if exit_signal is not None:
                exits.append(exit_signal)
                to_close.append(pid)

        for pid in to_close:
            self._positions.pop(pid, None)

        return exits


# ------------------------------------------------------------------
# VWAP Bounce Strategy
# ------------------------------------------------------------------

class VWAPBounce:
    """
    Long when price bounces off VWAP support; short off VWAP resistance.

    Entry:
      - Price within VWAP ± band (band = vwap_band_mult * ATR)
      - Bar reversal candle pattern (close back toward VWAP)
    Exit:
      - SL: 1 ATR beyond entry
      - TP: reward_risk * SL distance toward VWAP
      - EOD: forced close at 15:50
    """

    def __init__(self, config: IntradayConfig, indicators: _IndicatorState):
        self._cfg = config
        self._ind = indicators
        self._tracker = _PositionTracker(config.eod_exit_time)
        self._entry_start = _parse_time(config.entry_start)
        self._entry_end = _parse_time(config.entry_end)
        self._eod_time = _parse_time(config.eod_exit_time)
        self._current_date: Optional[datetime.date] = None
        self._daily_entries: int = 0
        self._last_entry_bar: int = 0
        self._bar_count: int = 0
        self._name = "vwap_bounce"

    def on_bar(self, bar: dict) -> list[Signal]:
        self._bar_count += 1
        ts = bar["timestamp"]
        bar_date = ts.date() if hasattr(ts, 'date') else ts
        bar_time = ts.time() if hasattr(ts, 'time') else ts

        # Daily reset
        if self._current_date != bar_date:
            self._current_date = bar_date
            self._daily_entries = 0

        signals: list[Signal] = []

        # Check exits first
        exits = self._tracker.check_exits(bar)
        signals.extend(exits)

        # Entry conditions
        if not self._can_enter(bar_time):
            return signals

        ind = self._ind
        if not ind.ready or ind.atr <= 0:
            return signals

        close = float(bar["close"])
        low = float(bar["low"])
        high = float(bar["high"])
        opn = float(bar["open"])
        vwap = ind.vwap
        atr = ind.atr
        band = self._cfg.vwap_band_mult * atr

        # Long bounce: price dips below VWAP-band then closes above VWAP-band
        # (wicked below support, closed back above)
        if (low <= vwap - band * 0.5
                and close > vwap - band
                and close < vwap
                and close > opn):  # bullish candle
            entry_px = close
            sl = entry_px - self._cfg.sl_atr_mult * atr
            sl_dist = entry_px - sl
            tp = entry_px + self._cfg.vwap_reward_risk * sl_dist
            pos_id = f"{self._name}_long_{bar_date}"
            sig = self._make_entry(
                ts, "long", entry_px, sl, tp, pos_id,
                f"VWAP bounce long (vwap={vwap:.2f}, atr={atr:.2f})",
                bar=bar,
            )
            if sig:
                signals.append(sig)

        # Short bounce: price spikes above VWAP+band then closes below
        elif (high >= vwap + band * 0.5
              and close < vwap + band
              and close > vwap
              and close < opn):  # bearish candle
            entry_px = close
            sl = entry_px + self._cfg.sl_atr_mult * atr
            sl_dist = sl - entry_px
            tp = entry_px - self._cfg.vwap_reward_risk * sl_dist
            pos_id = f"{self._name}_short_{bar_date}"
            sig = self._make_entry(
                ts, "short", entry_px, sl, tp, pos_id,
                f"VWAP bounce short (vwap={vwap:.2f}, atr={atr:.2f})",
                bar=bar,
            )
            if sig:
                signals.append(sig)

        return signals

    def _can_enter(self, bar_time: datetime.time) -> bool:
        if bar_time < self._entry_start or bar_time >= self._entry_end:
            return False
        if self._tracker.count >= self._cfg.max_positions_per_strategy:
            return False
        if (self._bar_count - self._last_entry_bar) < self._cfg.entry_cooldown_bars:
            return False
        return True

    def _make_entry(
        self, ts, direction, entry_px, sl, tp, pos_id, reason, bar=None,
    ) -> Optional[Signal]:
        qscore = compute_quality_score(self._ind, bar, direction) if bar else 0.0
        sig_type = SignalType.LONG_ENTRY if direction == "long" else SignalType.SHORT_ENTRY
        self._tracker.add(_OpenPosition(
            position_id=pos_id,
            direction=direction,
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
            strategy_type=self._name,
            entry_time=ts,
            size=self._cfg.base_size,
        ))
        self._daily_entries += 1
        self._last_entry_bar = self._bar_count
        logger.debug("VWAP bounce entry: %s @ %.2f sl=%.2f tp=%.2f q=%.3f", direction, entry_px, sl, tp, qscore)
        return Signal(
            signal_type=sig_type,
            price=entry_px,
            timestamp=ts,
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
            position_size=self._cfg.base_size,
            strategy_type=self._name,
            reason=reason,
            position_id=pos_id,
            quality_score=qscore,
        )

    def reset(self) -> None:
        self._tracker = _PositionTracker(self._cfg.eod_exit_time)
        self._current_date = None
        self._daily_entries = 0
        self._last_entry_bar = 0
        self._bar_count = 0


# ------------------------------------------------------------------
# Intraday Momentum Strategy
# ------------------------------------------------------------------

class IntradayMomentum:
    """
    Breakout beyond short-term intraday range with volume confirmation.

    Entry:
      - Close breaks above/below N-bar high/low
      - Volume > mult * average volume
    Exit:
      - SL: 1 ATR
      - TP: reward_risk * SL distance
      - EOD
    """

    def __init__(self, config: IntradayConfig, indicators: _IndicatorState):
        self._cfg = config
        self._ind = indicators
        self._tracker = _PositionTracker(config.eod_exit_time)
        self._entry_start = _parse_time(config.entry_start)
        self._entry_end = _parse_time(config.entry_end)
        self._current_date: Optional[datetime.date] = None
        self._daily_entries: int = 0
        self._last_entry_bar: int = 0
        self._bar_count: int = 0
        self._name = "intraday_momentum"

    def on_bar(self, bar: dict) -> list[Signal]:
        self._bar_count += 1
        ts = bar["timestamp"]
        bar_date = ts.date() if hasattr(ts, 'date') else ts
        bar_time = ts.time() if hasattr(ts, 'time') else ts

        if self._current_date != bar_date:
            self._current_date = bar_date
            self._daily_entries = 0

        signals: list[Signal] = []

        # Check exits
        exits = self._tracker.check_exits(bar)
        signals.extend(exits)

        # Entry conditions
        if not self._can_enter(bar_time):
            return signals

        ind = self._ind
        if not ind.ready or ind.atr <= 0:
            return signals

        lookback = self._cfg.momentum_lookback
        if len(ind.recent_highs) < lookback + 1:
            return signals

        close = float(bar["close"])
        volume = float(bar["volume"])
        atr = ind.atr

        # Recent range (excluding current bar)
        recent_highs = ind.recent_highs[-(lookback + 1):-1]
        recent_lows = ind.recent_lows[-(lookback + 1):-1]
        range_high = max(recent_highs)
        range_low = min(recent_lows)
        range_size = range_high - range_low

        # Volume confirmation
        vol_ok = (ind.avg_volume > 0 and
                  volume > self._cfg.momentum_volume_mult * ind.avg_volume)

        # Breakout threshold
        threshold = range_size * self._cfg.momentum_breakout_mult

        # Long breakout
        if close > range_high + threshold * 0.1 and vol_ok:
            entry_px = close
            sl = entry_px - self._cfg.sl_atr_mult * atr
            sl_dist = entry_px - sl
            tp = entry_px + self._cfg.momentum_reward_risk * sl_dist
            pos_id = f"{self._name}_long_{bar_date}_{self._bar_count}"
            sig = self._make_entry(
                ts, "long", entry_px, sl, tp, pos_id,
                f"Momentum breakout long (range_high={range_high:.2f}, vol={volume:.0f})",
                bar=bar,
            )
            if sig:
                signals.append(sig)

        # Short breakout
        elif close < range_low - threshold * 0.1 and vol_ok:
            entry_px = close
            sl = entry_px + self._cfg.sl_atr_mult * atr
            sl_dist = sl - entry_px
            tp = entry_px - self._cfg.momentum_reward_risk * sl_dist
            pos_id = f"{self._name}_short_{bar_date}_{self._bar_count}"
            sig = self._make_entry(
                ts, "short", entry_px, sl, tp, pos_id,
                f"Momentum breakout short (range_low={range_low:.2f}, vol={volume:.0f})",
                bar=bar,
            )
            if sig:
                signals.append(sig)

        return signals

    def _can_enter(self, bar_time: datetime.time) -> bool:
        if bar_time < self._entry_start or bar_time >= self._entry_end:
            return False
        if self._tracker.count >= self._cfg.max_positions_per_strategy:
            return False
        if (self._bar_count - self._last_entry_bar) < self._cfg.entry_cooldown_bars:
            return False
        return True

    def _make_entry(self, ts, direction, entry_px, sl, tp, pos_id, reason, bar=None):
        qscore = compute_quality_score(self._ind, bar, direction) if bar else 0.0
        sig_type = SignalType.LONG_ENTRY if direction == "long" else SignalType.SHORT_ENTRY
        self._tracker.add(_OpenPosition(
            position_id=pos_id,
            direction=direction,
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
            strategy_type=self._name,
            entry_time=ts,
            size=self._cfg.base_size,
        ))
        self._daily_entries += 1
        self._last_entry_bar = self._bar_count
        logger.debug("Momentum entry: %s @ %.2f sl=%.2f tp=%.2f q=%.3f", direction, entry_px, sl, tp, qscore)
        return Signal(
            signal_type=sig_type,
            price=entry_px,
            timestamp=ts,
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
            position_size=self._cfg.base_size,
            strategy_type=self._name,
            reason=reason,
            position_id=pos_id,
            quality_score=qscore,
        )

    def reset(self) -> None:
        self._tracker = _PositionTracker(self._cfg.eod_exit_time)
        self._current_date = None
        self._daily_entries = 0
        self._last_entry_bar = 0
        self._bar_count = 0


# ------------------------------------------------------------------
# Mean Reversion Strategy
# ------------------------------------------------------------------

class MeanReversion:
    """
    Fade extreme intraday moves using RSI and VWAP deviation.

    Entry:
      - RSI oversold/overbought
      - Price far from VWAP (> threshold * ATR)
    Exit:
      - SL: 1 ATR
      - TP: reversion toward VWAP (reward_risk * SL)
      - EOD
    """

    def __init__(self, config: IntradayConfig, indicators: _IndicatorState):
        self._cfg = config
        self._ind = indicators
        self._tracker = _PositionTracker(config.eod_exit_time)
        self._entry_start = _parse_time(config.entry_start)
        self._entry_end = _parse_time(config.entry_end)
        self._current_date: Optional[datetime.date] = None
        self._daily_entries: int = 0
        self._last_entry_bar: int = 0
        self._bar_count: int = 0
        self._name = "mean_reversion"

    def on_bar(self, bar: dict) -> list[Signal]:
        self._bar_count += 1
        ts = bar["timestamp"]
        bar_date = ts.date() if hasattr(ts, 'date') else ts
        bar_time = ts.time() if hasattr(ts, 'time') else ts

        if self._current_date != bar_date:
            self._current_date = bar_date
            self._daily_entries = 0

        signals: list[Signal] = []

        # Check exits
        exits = self._tracker.check_exits(bar)
        signals.extend(exits)

        # Entry conditions
        if not self._can_enter(bar_time):
            return signals

        ind = self._ind
        if not ind.ready or ind.atr <= 0:
            return signals

        close = float(bar["close"])
        opn = float(bar["open"])
        vwap = ind.vwap
        atr = ind.atr
        rsi = ind.rsi
        deviation = (close - vwap) / atr if atr > 0 else 0.0

        # Long mean reversion: oversold + price far below VWAP
        if (rsi <= self._cfg.rsi_oversold
                and deviation <= -self._cfg.vwap_deviation_threshold
                and close > opn):  # reversal candle (bullish)
            entry_px = close
            sl = entry_px - self._cfg.sl_atr_mult * atr
            sl_dist = entry_px - sl
            tp = entry_px + self._cfg.reversion_reward_risk * sl_dist
            pos_id = f"{self._name}_long_{bar_date}_{self._bar_count}"
            sig = self._make_entry(
                ts, "long", entry_px, sl, tp, pos_id,
                f"Mean reversion long (rsi={rsi:.1f}, dev={deviation:.2f})",
                bar=bar,
            )
            if sig:
                signals.append(sig)

        # Short mean reversion: overbought + price far above VWAP
        elif (rsi >= self._cfg.rsi_overbought
              and deviation >= self._cfg.vwap_deviation_threshold
              and close < opn):  # reversal candle (bearish)
            entry_px = close
            sl = entry_px + self._cfg.sl_atr_mult * atr
            sl_dist = sl - entry_px
            tp = entry_px - self._cfg.reversion_reward_risk * sl_dist
            pos_id = f"{self._name}_short_{bar_date}_{self._bar_count}"
            sig = self._make_entry(
                ts, "short", entry_px, sl, tp, pos_id,
                f"Mean reversion short (rsi={rsi:.1f}, dev={deviation:.2f})",
                bar=bar,
            )
            if sig:
                signals.append(sig)

        return signals

    def _can_enter(self, bar_time: datetime.time) -> bool:
        if bar_time < self._entry_start or bar_time >= self._entry_end:
            return False
        if self._tracker.count >= self._cfg.max_positions_per_strategy:
            return False
        if (self._bar_count - self._last_entry_bar) < self._cfg.entry_cooldown_bars:
            return False
        return True

    def _make_entry(self, ts, direction, entry_px, sl, tp, pos_id, reason, bar=None):
        qscore = compute_quality_score(self._ind, bar, direction) if bar else 0.0
        sig_type = SignalType.LONG_ENTRY if direction == "long" else SignalType.SHORT_ENTRY
        self._tracker.add(_OpenPosition(
            position_id=pos_id,
            direction=direction,
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
            strategy_type=self._name,
            entry_time=ts,
            size=self._cfg.base_size,
        ))
        self._daily_entries += 1
        self._last_entry_bar = self._bar_count
        logger.debug("Mean reversion entry: %s @ %.2f sl=%.2f tp=%.2f q=%.3f", direction, entry_px, sl, tp, qscore)
        return Signal(
            signal_type=sig_type,
            price=entry_px,
            timestamp=ts,
            entry_price=entry_px,
            stop_loss=sl,
            take_profit=tp,
            position_size=self._cfg.base_size,
            strategy_type=self._name,
            reason=reason,
            position_id=pos_id,
            quality_score=qscore,
        )

    def reset(self) -> None:
        self._tracker = _PositionTracker(self._cfg.eod_exit_time)
        self._current_date = None
        self._daily_entries = 0
        self._last_entry_bar = 0
        self._bar_count = 0


# ------------------------------------------------------------------
# Multi-Strategy Runner
# ------------------------------------------------------------------

class IntradayStrategyRunner:
    """
    Aggregates signals from multiple intraday strategies.

    Shares a single indicator state across all strategies to avoid
    redundant computation.
    """

    def __init__(self, config: IntradayConfig | None = None):
        self._cfg = config or IntradayConfig()
        self._indicators = _IndicatorState(
            atr_period=self._cfg.atr_period,
            rsi_period=self._cfg.rsi_period,
        )
        self._strategies = [
            VWAPBounce(self._cfg, self._indicators),
            IntradayMomentum(self._cfg, self._indicators),
            MeanReversion(self._cfg, self._indicators),
        ]

    def on_bar(self, bar: dict) -> list[Signal]:
        """Process one bar through all strategies, return aggregated signals."""
        # Update shared indicators first
        self._indicators.update(bar)

        all_signals: list[Signal] = []
        for strat in self._strategies:
            sigs = strat.on_bar(bar)
            all_signals.extend(sigs)
        return all_signals

    def reset(self) -> None:
        self._indicators = _IndicatorState(
            atr_period=self._cfg.atr_period,
            rsi_period=self._cfg.rsi_period,
        )
        for strat in self._strategies:
            strat._ind = self._indicators
            strat.reset()
