"""
Live-Compatible Strategy Engine
================================

Wraps the existing HybridEMAMLStrategy in a streaming-friendly engine
that can consume bars one at a time from any source (live feed, replay,
or direct push).

Responsibilities:
  1. Input — Accept streaming 5-minute bars (dict or Bar objects).
  2. State — Maintain rolling features, EMA values, candidate history,
     and ML probability windows (all delegated to the underlying strategy).
  3. Candidate generation — For each bar, generate candidates with
     ML probability and percentile scoring (same logic as backtest).
  4. Selection — Priority-based (breakout > momentum > pullback) with
     ML ranking within groups and max-N trade selection.
  5. Output — Emit LiveSignal objects with all execution-ready fields.
  6. Validation — Replay mode that compares engine output against
     saved backtest signals for reproducibility checking.

The engine does NOT execute orders.  It produces signals consumed by
an execution adapter (broker gateway, paper-trade sim, etc.).
"""

from __future__ import annotations

import datetime
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from strategy.hybrid_ema_ml import (
    HybridEMAMLConfig,
    HybridEMAMLStrategy,
    TradeCandidate,
)
from strategy.orb import Signal, SignalType

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Output dataclass
# ------------------------------------------------------------------

@dataclass
class LiveSignal:
    """Execution-ready signal emitted by the strategy engine."""

    timestamp: datetime.datetime
    direction: str                  # "long" or "short"
    signal_type: SignalType
    entry: float
    stop: float
    take_profit: float
    position_size: float
    strategy_type: str              # e.g. "ema50_breakout"
    reason: str = ""
    position_id: str = ""
    ml_prob: float = 0.0
    percentile: float = 0.0
    quality_score: float = 0.0

    @property
    def is_entry(self) -> bool:
        return self.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY)

    @property
    def is_exit(self) -> bool:
        return self.signal_type in (
            SignalType.EXIT_TP,
            SignalType.EXIT_SL,
            SignalType.EXIT_EOD,
        )


def _signal_to_live(sig: Signal) -> LiveSignal:
    """Convert an internal Signal to a LiveSignal."""
    if sig.signal_type == SignalType.LONG_ENTRY:
        direction = "long"
    elif sig.signal_type == SignalType.SHORT_ENTRY:
        direction = "short"
    elif sig.signal_type in (SignalType.EXIT_TP, SignalType.EXIT_SL, SignalType.EXIT_EOD):
        direction = ""
    else:
        direction = ""

    # For exits, LiveSignal.entry carries the exit (fill) price.
    # sig.price = exit fill price; sig.entry_price = original entry.
    if sig.signal_type in (SignalType.EXIT_TP, SignalType.EXIT_SL, SignalType.EXIT_EOD):
        entry = sig.price
    else:
        entry = sig.entry_price if sig.entry_price is not None else sig.price

    return LiveSignal(
        timestamp=sig.timestamp,
        direction=direction,
        signal_type=sig.signal_type,
        entry=entry,
        stop=sig.stop_loss if sig.stop_loss is not None else 0.0,
        take_profit=sig.take_profit if sig.take_profit is not None else 0.0,
        position_size=sig.position_size,
        strategy_type=sig.strategy_type,
        reason=sig.reason,
        position_id=sig.position_id,
        quality_score=getattr(sig, 'quality_score', 0.0),
    )


# ------------------------------------------------------------------
# Engine state snapshot (for inspection / serialisation)
# ------------------------------------------------------------------

@dataclass
class EngineState:
    """Read-only snapshot of the engine's internal state."""

    bar_count: int
    current_date: Optional[datetime.date]
    range_set: bool
    opening_high: Optional[float]
    opening_low: Optional[float]
    open_positions: int
    decided_today: bool
    prob_window_size: int


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------

@dataclass
class ValidationMismatch:
    """One signal that differs between engine output and reference."""

    bar_index: int
    timestamp: datetime.datetime
    field: str
    engine_value: object
    reference_value: object

    def __str__(self) -> str:
        return (
            f"Bar {self.bar_index} ({self.timestamp}): "
            f"{self.field} engine={self.engine_value} ref={self.reference_value}"
        )


@dataclass
class ValidationResult:
    """Summary of a validation run."""

    total_bars: int = 0
    engine_signals: int = 0
    reference_signals: int = 0
    matched: int = 0
    mismatches: list[ValidationMismatch] = field(default_factory=list)
    extra_engine: list[LiveSignal] = field(default_factory=list)
    missing_engine: list[LiveSignal] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            len(self.mismatches) == 0
            and len(self.extra_engine) == 0
            and len(self.missing_engine) == 0
        )


# ------------------------------------------------------------------
# Strategy Engine
# ------------------------------------------------------------------

class StrategyEngine:
    """
    Live-compatible wrapper around HybridEMAMLStrategy.

    Consumes one bar at a time (streaming or replay) and emits
    LiveSignal objects.  The underlying strategy handles all state:
    rolling features, EMA values, candidate history, ML probability
    window, opening range, and position tracking.

    Parameters
    ----------
    config : HybridEMAMLConfig
        Strategy configuration.  For live use, set ``multi_candidate=True``
        with the desired ``ema_periods``, ``entry_types``, and
        ``selection_strategy``.
    on_signal : callable, optional
        If provided, called with each emitted LiveSignal (entries and
        exits).  Useful for wiring to an execution adapter.
    """

    def __init__(
        self,
        config: HybridEMAMLConfig,
        on_signal: Optional[Callable[[LiveSignal], None]] = None,
    ) -> None:
        self._config = config
        self._strategy = HybridEMAMLStrategy(config)
        self._on_signal = on_signal

        self._bar_count: int = 0
        self._signal_log: list[LiveSignal] = []

        logger.info(
            "StrategyEngine initialised: multi_candidate=%s, "
            "selection=%s, ema_periods=%s, entry_types=%s",
            config.multi_candidate,
            config.selection_strategy,
            config.ema_periods,
            config.entry_types,
        )

    # ---- public API ---------------------------------------------------

    def on_bar(self, bar: dict) -> list[LiveSignal]:
        """
        Process a single OHLCV bar and return any generated signals.

        Parameters
        ----------
        bar : dict
            Must contain keys: ``timestamp``, ``open``, ``high``,
            ``low``, ``close``, ``volume``.  ``timestamp`` must be a
            tz-aware datetime.

        Returns
        -------
        list[LiveSignal]
            Zero or more signals (entries and/or exits).
        """
        self._bar_count += 1

        raw = self._strategy.on_bar(bar)

        # Normalise to list
        if isinstance(raw, Signal):
            signals = [raw]
        elif isinstance(raw, list):
            signals = raw
        else:
            logger.error(
                "Strategy.on_bar() returned unexpected type %s at bar %d",
                type(raw).__name__,
                self._bar_count,
            )
            return []

        # Filter NONE, convert, dispatch
        live_signals: list[LiveSignal] = []
        for sig in signals:
            if sig.signal_type == SignalType.NONE:
                continue

            ls = _signal_to_live(sig)

            # Enrich with ML diagnostics from the strategy's decision log
            if ls.is_entry and self._strategy.ml_decisions:
                decision = self._find_ml_decision(ls)
                ls.ml_prob = decision.get("ml_prob", 0.0)
                ls.percentile = decision.get("percentile", 0.0)

            live_signals.append(ls)
            self._signal_log.append(ls)

            logger.info(
                "Signal: %s %s @ %.2f (sl=%.2f tp=%.2f size=%.2f) [%s] %s",
                ls.signal_type.name,
                ls.direction,
                ls.entry,
                ls.stop,
                ls.take_profit,
                ls.position_size,
                ls.strategy_type,
                ls.reason,
            )

            if self._on_signal is not None:
                self._on_signal(ls)

        return live_signals

    def reset(self) -> None:
        """Full reset for a new session/run."""
        self._strategy.reset()
        self._bar_count = 0
        self._signal_log.clear()
        logger.info("StrategyEngine reset")

    @property
    def state(self) -> EngineState:
        """Read-only snapshot of current engine state."""
        s = self._strategy
        return EngineState(
            bar_count=self._bar_count,
            current_date=s._current_date,
            range_set=s.range_set,
            opening_high=s.opening_high,
            opening_low=s.opening_low,
            open_positions=len(s._open_positions),
            decided_today=s._decided_today,
            prob_window_size=len(s._prob_window_all),
        )

    @property
    def signal_log(self) -> list[LiveSignal]:
        """All signals emitted since last reset (read-only copy)."""
        return list(self._signal_log)

    @property
    def ml_decisions(self) -> list[dict]:
        """Proxy to the underlying strategy's ML decision log."""
        return self._strategy.ml_decisions

    def _find_ml_decision(self, sig: LiveSignal) -> dict:
        """Find the ML decision matching a specific signal.

        In multi-candidate mode each candidate records a decision with
        ``strategy_type``.  We search backwards through the decision log
        for the matching entry.  Falls back to the most recent decision
        if no strategy_type match is found.
        """
        decisions = self._strategy.ml_decisions
        if not decisions:
            return {}

        # Walk backwards for efficiency — the matching decision was
        # just appended moments ago.
        for d in reversed(decisions):
            if d.get("strategy_type") == sig.strategy_type:
                return d

        # Fallback: single-candidate mode has no strategy_type key
        return decisions[-1]

    # ---- Validation mode -----------------------------------------------

    def validate(
        self,
        bars: list[dict],
        reference_signals: list[LiveSignal],
        *,
        price_tolerance: float = 0.01,
    ) -> ValidationResult:
        """
        Replay bars and compare output against reference signals.

        The engine is **reset** before replaying.  Each bar is fed
        through ``on_bar`` and the resulting signals are matched
        against ``reference_signals`` by timestamp + signal_type.

        Parameters
        ----------
        bars : list[dict]
            Historical bars in chronological order (same format as
            ``on_bar`` expects).
        reference_signals : list[LiveSignal]
            Expected signals (e.g. from a backtest run).
        price_tolerance : float
            Maximum allowed absolute difference for price fields
            (entry, stop, take_profit) before flagging a mismatch.

        Returns
        -------
        ValidationResult
        """
        self.reset()

        result = ValidationResult(total_bars=len(bars))

        # Replay
        engine_signals: list[LiveSignal] = []
        for i, bar in enumerate(bars):
            sigs = self.on_bar(bar)
            for s in sigs:
                engine_signals.append(s)

        result.engine_signals = len(engine_signals)
        result.reference_signals = len(reference_signals)

        # Build lookup: (timestamp, signal_type, strategy_type) -> signal
        # strategy_type is included so multi-candidate signals at the
        # same timestamp don't silently overwrite each other.
        ref_lookup: dict[tuple, LiveSignal] = {}
        for rs in reference_signals:
            key = (rs.timestamp, rs.signal_type, rs.strategy_type)
            ref_lookup[key] = rs

        eng_lookup: dict[tuple, LiveSignal] = {}
        for es in engine_signals:
            key = (es.timestamp, es.signal_type, es.strategy_type)
            eng_lookup[key] = es

        # Match
        all_keys = set(ref_lookup.keys()) | set(eng_lookup.keys())
        for key in sorted(all_keys, key=lambda k: k[0]):
            ts, sig_type, _strat_type = key
            eng_sig = eng_lookup.get(key)
            ref_sig = ref_lookup.get(key)

            if eng_sig is None:
                result.missing_engine.append(ref_sig)
                continue
            if ref_sig is None:
                result.extra_engine.append(eng_sig)
                continue

            # Compare fields
            result.matched += 1
            bar_idx = next(
                (i for i, b in enumerate(bars) if b["timestamp"] == ts),
                -1,
            )

            for fld, eng_val, ref_val in [
                ("direction", eng_sig.direction, ref_sig.direction),
                ("entry", eng_sig.entry, ref_sig.entry),
                ("stop", eng_sig.stop, ref_sig.stop),
                ("take_profit", eng_sig.take_profit, ref_sig.take_profit),
                ("strategy_type", eng_sig.strategy_type, ref_sig.strategy_type),
            ]:
                if isinstance(eng_val, float) and isinstance(ref_val, float):
                    if abs(eng_val - ref_val) > price_tolerance:
                        result.mismatches.append(ValidationMismatch(
                            bar_index=bar_idx,
                            timestamp=ts,
                            field=fld,
                            engine_value=eng_val,
                            reference_value=ref_val,
                        ))
                elif eng_val != ref_val:
                    result.mismatches.append(ValidationMismatch(
                        bar_index=bar_idx,
                        timestamp=ts,
                        field=fld,
                        engine_value=eng_val,
                        reference_value=ref_val,
                    ))

            # Position size: wider tolerance (ML percentile can drift)
            if abs(eng_sig.position_size - ref_sig.position_size) > 0.05:
                result.mismatches.append(ValidationMismatch(
                    bar_index=bar_idx,
                    timestamp=ts,
                    field="position_size",
                    engine_value=eng_sig.position_size,
                    reference_value=ref_sig.position_size,
                ))

        if result.passed:
            logger.info(
                "Validation PASSED: %d bars, %d signals matched",
                result.total_bars,
                result.matched,
            )
        else:
            logger.warning(
                "Validation FAILED: %d mismatches, %d extra, %d missing",
                len(result.mismatches),
                len(result.extra_engine),
                len(result.missing_engine),
            )

        return result


# ------------------------------------------------------------------
# Convenience: build engine from backtest results for validation
# ------------------------------------------------------------------

def _map_exit_reason(reason: str) -> SignalType:
    """Map a Trade.exit_reason string to a SignalType.

    Handles both engine format (\"Take Profit\", \"Stop Loss\", \"End of Day\")
    and strategy format (\"Take profit hit ...\", \"Stop loss hit ...\",
    \"End-of-day exit\").
    """
    lower = reason.lower()
    if lower.startswith("take profit"):
        return SignalType.EXIT_TP
    if lower.startswith("stop loss"):
        return SignalType.EXIT_SL
    if lower.startswith("end of day") or lower.startswith("end-of-day"):
        return SignalType.EXIT_EOD
    logger.warning("Unknown exit reason '%s', defaulting to EXIT_EOD", reason)
    return SignalType.EXIT_EOD


def build_reference_signals(
    trades: list,
    *,
    include_entries: bool = True,
    include_exits: bool = True,
) -> list[LiveSignal]:
    """
    Convert backtest Trade objects into LiveSignal references.

    Parameters
    ----------
    trades : list[backtest.engine.Trade]
        Completed trades from a backtest run.
    include_entries, include_exits : bool
        Which signal types to include.

    Returns
    -------
    list[LiveSignal]
    """
    signals: list[LiveSignal] = []

    for t in trades:
        if include_entries:
            sig_type = (
                SignalType.LONG_ENTRY if t.direction == "long"
                else SignalType.SHORT_ENTRY
            )
            signals.append(LiveSignal(
                timestamp=t.entry_time,
                direction=t.direction,
                signal_type=sig_type,
                entry=t.entry_price,
                stop=t.stop_loss,
                take_profit=t.take_profit,
                position_size=t.position_size,
                strategy_type=getattr(t, "strategy_type", ""),
            ))

        if include_exits:
            exit_sig_type = _map_exit_reason(t.exit_reason)

            signals.append(LiveSignal(
                timestamp=t.exit_time,
                direction=t.direction,
                signal_type=exit_sig_type,
                entry=t.entry_price,
                stop=t.stop_loss,
                take_profit=t.take_profit,
                position_size=t.position_size,
                strategy_type=getattr(t, "strategy_type", ""),
            ))

    return signals
