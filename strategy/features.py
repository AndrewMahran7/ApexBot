"""
Shared technical feature computations for MES strategies.
==========================================================

Stateful feature calculators that process bars one at a time.
No lookahead — each computation uses only current and past data.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureSnapshot:
    """All computed features at a single point in time."""
    ema: Optional[float] = None
    ema_slope: Optional[float] = None        # change per bar over lookback
    atr: Optional[float] = None
    relative_volume: Optional[float] = None  # current vol / rolling avg
    rolling_range: Optional[float] = None    # avg high-low over lookback


class FeatureEngine:
    """
    Computes streaming technical features bar-by-bar.

    Call `update(high, low, close, volume)` with each new bar.
    Then read `.snapshot` for the latest values.
    """

    def __init__(
        self,
        ema_length: int = 50,
        ema_slope_lookback: int = 5,
        atr_length: int = 14,
        volume_lookback: int = 20,
    ):
        self.ema_length = ema_length
        self.ema_slope_lookback = ema_slope_lookback
        self.atr_length = atr_length
        self.volume_lookback = volume_lookback

        # EMA state
        self._ema: Optional[float] = None
        self._ema_alpha: float = 2.0 / (ema_length + 1)
        self._ema_history: deque[float] = deque(maxlen=ema_slope_lookback + 1)
        self._close_count: int = 0

        # ATR state
        self._tr_values: deque[float] = deque(maxlen=atr_length)
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None

        # Volume state
        self._volumes: deque[float] = deque(maxlen=volume_lookback)

        # Rolling range
        self._ranges: deque[float] = deque(maxlen=atr_length)

        # Latest snapshot
        self.snapshot = FeatureSnapshot()

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar and update all features."""
        self._update_ema(close)
        self._update_atr(high, low, close)
        self._update_volume(volume)
        self._update_rolling_range(high, low)
        self._build_snapshot()

    def reset(self):
        """Full reset for a new backtest run."""
        self._ema = None
        self._ema_history.clear()
        self._close_count = 0
        self._tr_values.clear()
        self._prev_close = None
        self._atr = None
        self._volumes.clear()
        self._ranges.clear()
        self.snapshot = FeatureSnapshot()

    # ------------------------------------------------------------------

    def _update_ema(self, close: float):
        self._close_count += 1
        if self._ema is None:
            if self._close_count >= self.ema_length:
                # bootstrap: not exact SMA seed, but good enough for streaming
                self._ema = close
            elif self._close_count >= 10:
                self._ema = close  # start early with rough seed
        else:
            self._ema = close * self._ema_alpha + self._ema * (1 - self._ema_alpha)

        if self._ema is not None:
            self._ema_history.append(self._ema)

    def _update_atr(self, high: float, low: float, close: float):
        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        else:
            tr = high - low
        self._prev_close = close
        self._tr_values.append(tr)

        if len(self._tr_values) >= self.atr_length:
            self._atr = sum(self._tr_values) / len(self._tr_values)

    def _update_volume(self, volume: float):
        self._volumes.append(volume)

    def _update_rolling_range(self, high: float, low: float):
        self._ranges.append(high - low)

    def _build_snapshot(self):
        snap = FeatureSnapshot()
        snap.ema = self._ema

        # EMA slope: average change per bar over lookback
        if len(self._ema_history) >= 2:
            oldest = self._ema_history[0]
            newest = self._ema_history[-1]
            n = len(self._ema_history) - 1
            snap.ema_slope = (newest - oldest) / n if n > 0 else 0.0

        snap.atr = self._atr

        # Relative volume: latest bar vs rolling average
        if len(self._volumes) >= 2:
            avg_vol = sum(list(self._volumes)[:-1]) / (len(self._volumes) - 1)
            if avg_vol > 0:
                snap.relative_volume = self._volumes[-1] / avg_vol

        if self._ranges:
            snap.rolling_range = sum(self._ranges) / len(self._ranges)

        self.snapshot = snap
