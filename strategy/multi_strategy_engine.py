"""
Multi-Strategy Engine
=====================

Wraps the existing StrategyEngine (HybridEMAML) and the new intraday
strategies (VWAP Bounce, Intraday Momentum, Mean Reversion) into one
engine that:

  - Processes each bar through ALL sub-strategies.
  - Converts all Signal objects to LiveSignal via the standard converter.
  - Dispatches them through a single on_signal callback chain.

This allows the prop gate, risk manager, and paper engine to operate
on a unified signal stream exactly as before.

Usage:
    from strategy.multi_strategy_engine import MultiStrategyEngine
    engine = MultiStrategyEngine(
        strategy_cfg=hybrid_cfg,
        intraday_cfg=IntradayConfig(),
        on_signal=callback,
    )
    engine.on_bar(bar_dict)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from strategy.hybrid_ema_ml import HybridEMAMLConfig
from strategy.strategy_engine import StrategyEngine, LiveSignal, _signal_to_live
from strategy.orb import Signal, SignalType
from strategy.intraday_strategies import IntradayConfig, IntradayStrategyRunner

logger = logging.getLogger(__name__)


class MultiStrategyEngine:
    """
    Aggregates signals from the existing EMA/ML engine and
    the new intraday strategies.

    All LiveSignals pass through the same ``on_signal`` callback
    so the prop gate / risk manager see a unified stream.

    Parameters
    ----------
    strategy_cfg : HybridEMAMLConfig
        Config for the original strategy.
    intraday_cfg : IntradayConfig
        Config for the three new intraday strategies.
    on_signal : callable, optional
        Unified callback for all LiveSignal objects.
    enable_hybrid : bool
        If True, run the original HybridEMAML engine too (default True).
    enable_intraday : bool
        If True, run the intraday strategies (default True).
    """

    def __init__(
        self,
        strategy_cfg: HybridEMAMLConfig,
        intraday_cfg: IntradayConfig | None = None,
        on_signal: Optional[Callable[[LiveSignal], None]] = None,
        enable_hybrid: bool = True,
        enable_intraday: bool = True,
        max_intraday_entries_per_bar: int = 2,
    ) -> None:
        self._on_signal = on_signal
        self._enable_hybrid = enable_hybrid
        self._enable_intraday = enable_intraday
        self._max_intraday_per_bar = max_intraday_entries_per_bar

        self._bar_count: int = 0
        self._signal_log: list[LiveSignal] = []

        # Original EMA/ML engine — delegates signal dispatch to us
        self._hybrid_engine: Optional[StrategyEngine] = None
        if enable_hybrid:
            self._hybrid_engine = StrategyEngine(
                config=strategy_cfg,
                on_signal=self._dispatch,
            )

        # Intraday strategies
        self._intraday_runner: Optional[IntradayStrategyRunner] = None
        if enable_intraday:
            self._intraday_runner = IntradayStrategyRunner(
                config=intraday_cfg or IntradayConfig(),
            )

        logger.info(
            "MultiStrategyEngine initialised: hybrid=%s intraday=%s",
            enable_hybrid, enable_intraday,
        )

    # ---- public API ---------------------------------------------------

    def on_bar(self, bar: dict) -> list[LiveSignal]:
        """
        Process one bar through all sub-strategies and return signals.
        """
        self._bar_count += 1
        all_signals: list[LiveSignal] = []

        # Run the original engine (it dispatches via _dispatch internally)
        if self._hybrid_engine is not None:
            hybrid_signals = self._hybrid_engine.on_bar(bar)
            all_signals.extend(hybrid_signals)

        # Run the intraday strategies
        if self._intraday_runner is not None:
            raw_signals = self._intraday_runner.on_bar(bar)

            # Separate entry signals from exit signals
            entry_signals = []
            exit_signals = []
            for sig in raw_signals:
                if sig.signal_type == SignalType.NONE:
                    continue
                if sig.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
                    entry_signals.append(sig)
                else:
                    exit_signals.append(sig)

            # Always dispatch exit signals
            for sig in exit_signals:
                ls = _signal_to_live(sig)
                self._signal_log.append(ls)
                all_signals.append(ls)
                if self._on_signal is not None:
                    self._on_signal(ls)

            # Limit intraday entries: quality filter + pick top N per bar
            if entry_signals:
                # Get quality threshold from config
                min_q = 0.0
                if self._intraday_runner is not None:
                    min_q = self._intraday_runner._cfg.min_quality_score

                # Filter by quality threshold
                qualified = [s for s in entry_signals if s.quality_score >= min_q]

                if qualified:
                    # Priority map: strategies with historically better edge first
                    _PRIORITY = {
                        "vwap_bounce": 3,
                        "intraday_momentum": 2,
                        "mean_reversion": 1,
                    }

                    def _entry_score(s: Signal) -> tuple[int, float, float]:
                        """Sort key: (priority, quality_score, reward/risk)."""
                        sl_dist = abs(s.price - s.stop_loss) if s.stop_loss else 1.0
                        tp_dist = abs(s.take_profit - s.price) if s.take_profit else 1.0
                        rr = tp_dist / sl_dist if sl_dist > 0 else 0.0
                        prio = _PRIORITY.get(s.strategy_type, 0)
                        return (prio, s.quality_score, rr)

                    qualified.sort(key=_entry_score, reverse=True)
                    best = qualified[:self._max_intraday_per_bar]

                    for sig in best:
                        ls = _signal_to_live(sig)
                        self._signal_log.append(ls)
                        all_signals.append(ls)

                        logger.info(
                            "IntraSignal: %s %s @ %.2f (sl=%.2f tp=%.2f q=%.3f) [%s] %s",
                            ls.signal_type.name,
                            ls.direction,
                            ls.entry,
                            ls.stop,
                            ls.take_profit,
                            sig.quality_score,
                            ls.strategy_type,
                            ls.reason,
                        )

                        # Dispatch through the shared callback
                        if self._on_signal is not None:
                            self._on_signal(ls)

        return all_signals

    def reset(self) -> None:
        """Full reset for a new session/run."""
        if self._hybrid_engine is not None:
            self._hybrid_engine.reset()
        if self._intraday_runner is not None:
            self._intraday_runner.reset()
        self._bar_count = 0
        self._signal_log.clear()
        logger.info("MultiStrategyEngine reset")

    # ---- internal -----------------------------------------------------

    def _dispatch(self, sig: LiveSignal) -> None:
        """Dispatch a signal from the hybrid engine through our callback."""
        self._signal_log.append(sig)
        if self._on_signal is not None:
            self._on_signal(sig)

    @property
    def signal_log(self) -> list[LiveSignal]:
        return list(self._signal_log)

    @property
    def bar_count(self) -> int:
        return self._bar_count
