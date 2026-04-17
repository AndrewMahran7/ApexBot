"""
Risk Management Layer
======================

Sits between StrategyEngine and PaperEngine (or any execution adapter)
to enforce hard limits on trading activity.

Responsibilities:
  1. Hard limits — max daily loss, max trades per day, max concurrent positions.
  2. Kill switch — stop all new trades; optionally force-close open positions.
  3. Position limits — cap position size, prevent overexposure.
  4. Logging — every risk event is logged with full context.

Wiring:
    risk = RiskManager(config, instrument)
    engine = StrategyEngine(cfg, on_signal=risk.on_signal)
    risk.on_approved = paper.on_signal

    for bar in bars:
        risk.on_bar(bar)          # update daily P&L tracking
        paper.on_bar(bar)
        engine.on_bar(bar)        # signals flow through risk → paper
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from config.settings import InstrumentConfig
from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class RiskConfig:
    """Risk management parameters."""

    max_daily_loss: float = 500.0
    """Maximum realised + unrealised loss allowed per day (positive number)."""

    max_trades_per_day: int = 6
    """Maximum number of entries allowed per calendar day."""

    max_concurrent_positions: int = 3
    """Maximum open positions at any time."""

    max_position_size: float = 1.0
    """Hard cap on position_size field of any single entry signal."""

    max_total_exposure: float = 3.0
    """Sum of position_size across all open positions must not exceed this."""

    kill_switch_close_positions: bool = True
    """When kill switch triggers, emit EXIT_EOD for every open position."""


# ------------------------------------------------------------------
# Risk event record
# ------------------------------------------------------------------

@dataclass
class RiskEvent:
    """Immutable record of a risk decision."""

    timestamp: datetime.datetime
    event_type: str          # "blocked", "capped", "kill_switch", "reset"
    reason: str
    signal: Optional[LiveSignal] = None
    details: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Risk Manager
# ------------------------------------------------------------------

class RiskManager:
    """
    Pre-trade risk gate that filters and adjusts signals.

    Parameters
    ----------
    config : RiskConfig
        Risk limits.
    instrument : InstrumentConfig
        Contract spec for exposure calculations.
    on_approved : callable, optional
        Downstream handler for approved (possibly adjusted) signals.
    """

    def __init__(
        self,
        config: RiskConfig,
        instrument: InstrumentConfig,
        on_approved: Optional[Callable[[LiveSignal], None]] = None,
    ) -> None:
        self._cfg = config
        self._inst = instrument
        self.on_approved = on_approved

        # --- daily tracking ---
        self._current_date: Optional[datetime.date] = None
        self._daily_entries: int = 0
        self._daily_realized_pnl: float = 0.0

        # --- position tracking ---
        # pos_id -> {direction, entry_price, position_size, entry_time}
        self._open_positions: dict[str, dict] = {}

        # --- kill switch ---
        self._killed: bool = False

        # --- audit trail ---
        self._events: list[RiskEvent] = []

        # --- last bar for unrealised P&L ---
        self._last_bar: Optional[dict] = None

        logger.info(
            "RiskManager initialised: max_daily_loss=%.2f, max_trades=%d, "
            "max_concurrent=%d, max_size=%.2f, max_exposure=%.2f",
            config.max_daily_loss,
            config.max_trades_per_day,
            config.max_concurrent_positions,
            config.max_position_size,
            config.max_total_exposure,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_bar(self, bar: dict) -> None:
        """
        Update daily tracking state.  Call BEFORE engine.on_bar().
        Handles day-change reset and unrealised-PnL kill switch check.
        """
        self._last_bar = bar
        ts = bar["timestamp"]
        bar_date = ts.date() if isinstance(ts, datetime.datetime) else ts

        if self._current_date is None or bar_date != self._current_date:
            self._reset_day(bar_date)

        close = float(bar["close"])
        unrealized = self._unrealized_pnl(close)
        logger.debug(
            "Risk on_bar: date=%s close=%.2f realized=%.2f "
            "unrealized=%.2f open=%d killed=%s",
            bar_date, close, self._daily_realized_pnl,
            unrealized, len(self._open_positions), self._killed,
        )

        # Check kill switch on unrealised loss every bar
        if not self._killed:
            total_loss = self._daily_realized_pnl + self._unrealized_pnl(
                float(bar["close"]),
            )
            if total_loss <= -self._cfg.max_daily_loss:
                self._trigger_kill_switch(
                    ts,
                    f"Daily loss {total_loss:.2f} breached "
                    f"limit -{self._cfg.max_daily_loss:.2f}",
                )

    def on_signal(self, signal: LiveSignal) -> None:
        """
        Risk gate for incoming signals.

        Entry signals are checked against all limits then forwarded
        (possibly with capped size) to ``on_approved``.
        Exit signals are always forwarded and used to update internal
        position tracking.
        """
        if signal.is_exit:
            self._handle_exit(signal)
            self._forward(signal)
            return

        if not signal.is_entry:
            self._forward(signal)
            return

        # --- entry signal risk checks ---

        if self._killed:
            self._block(signal, "kill_switch_active")
            return

        if self._daily_entries >= self._cfg.max_trades_per_day:
            self._block(signal, "max_trades_per_day")
            return

        if len(self._open_positions) >= self._cfg.max_concurrent_positions:
            self._block(signal, "max_concurrent_positions")
            return

        # Position size cap
        adjusted = signal
        if signal.position_size > self._cfg.max_position_size:
            adjusted = self._cap_size(signal, self._cfg.max_position_size)

        # Total exposure cap
        current_exposure = sum(
            p["position_size"] for p in self._open_positions.values()
        )
        remaining = self._cfg.max_total_exposure - current_exposure
        if remaining <= 0:
            self._block(signal, "max_total_exposure")
            return
        if adjusted.position_size > remaining:
            adjusted = self._cap_size(adjusted, remaining)

        # Daily loss pre-check (realised only — don't block on unrealised
        # because on_bar already handles the kill switch for that)
        if self._daily_realized_pnl <= -self._cfg.max_daily_loss:
            self._block(signal, "daily_loss_limit")
            return

        # --- approved ---
        self._record_entry(adjusted)
        self._forward(adjusted)

    def activate_kill_switch(self, ts: datetime.datetime, reason: str) -> None:
        """Manually trigger the kill switch from external code."""
        self._trigger_kill_switch(ts, f"manual: {reason}")

    def reset_kill_switch(self) -> None:
        """Re-enable trading after operator review."""
        if not self._killed:
            return
        self._killed = False
        event = RiskEvent(
            timestamp=datetime.datetime.now(tz=datetime.timezone.utc),
            event_type="reset",
            reason="kill_switch_reset",
        )
        self._events.append(event)
        logger.info("Kill switch RESET — trading re-enabled")

    def reset(self) -> None:
        """Full reset for a new session."""
        self._current_date = None
        self._daily_entries = 0
        self._daily_realized_pnl = 0.0
        self._open_positions.clear()
        self._killed = False
        self._events.clear()
        self._last_bar = None
        logger.info("RiskManager full reset")

    # ---- Read-only properties -----------------------------------------

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def daily_entries(self) -> int:
        return self._daily_entries

    @property
    def daily_realized_pnl(self) -> float:
        return self._daily_realized_pnl

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def total_exposure(self) -> float:
        return sum(p["position_size"] for p in self._open_positions.values())

    @property
    def events(self) -> list[RiskEvent]:
        """Read-only copy of all risk events."""
        return list(self._events)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_day(self, new_date: datetime.date) -> None:
        """Roll counters for a new trading day."""
        if self._current_date is not None:
            logger.info(
                "Day rollover %s → %s: entries=%d realized=%.2f",
                self._current_date, new_date,
                self._daily_entries, self._daily_realized_pnl,
            )
        self._current_date = new_date
        self._daily_entries = 0
        self._daily_realized_pnl = 0.0

        # Kill switch persists across days — operator must explicitly reset
        if self._killed:
            logger.warning(
                "Kill switch still active on day change to %s", new_date,
            )

    def _trigger_kill_switch(
        self, ts: datetime.datetime, reason: str,
    ) -> None:
        """Activate the kill switch and optionally close positions."""
        self._killed = True

        event = RiskEvent(
            timestamp=ts,
            event_type="kill_switch",
            reason=reason,
            details={
                "daily_realized_pnl": self._daily_realized_pnl,
                "open_positions": len(self._open_positions),
            },
        )
        self._events.append(event)
        logger.error(
            "KILL SWITCH ACTIVATED at %s: %s "
            "(realized=%.2f, open=%d)",
            ts, reason,
            self._daily_realized_pnl,
            len(self._open_positions),
        )

        if self._cfg.kill_switch_close_positions and self._open_positions:
            self._force_close_all(ts)

    def _force_close_all(self, ts: datetime.datetime) -> None:
        """Emit EXIT_EOD for every open position."""
        if self._last_bar is None:
            logger.error(
                "Force-close requested at %s but no bar received yet — "
                "cannot determine exit price; skipping",
                ts,
            )
            return

        close_price = float(self._last_bar["close"])
        for pos_id, pos in list(self._open_positions.items()):
            exit_sig = LiveSignal(
                timestamp=ts,
                direction=pos["direction"],
                signal_type=SignalType.EXIT_EOD,
                entry=close_price,
                stop=0.0,
                take_profit=0.0,
                position_size=pos["position_size"],
                strategy_type=pos.get("strategy_type", ""),
                reason="kill_switch_force_close",
                position_id=pos_id,
            )

            # Compute PnL for daily loss tracking
            if pos["direction"] == "short":
                pnl_points = pos["entry_price"] - close_price
            else:
                pnl_points = close_price - pos["entry_price"]
            pnl_dollars = (
                pnl_points
                * self._inst.point_value
                * self._inst.contract_size
                * pos["position_size"]
            )
            self._daily_realized_pnl += pnl_dollars

            logger.warning(
                "Force-closing position %s (%s) at %s: "
                "pnl=%.2f daily_realized=%.2f",
                pos_id, pos["direction"], ts,
                pnl_dollars, self._daily_realized_pnl,
            )
            # Update internal tracking
            self._open_positions.pop(pos_id)
            self._forward(exit_sig)

    def _record_entry(self, sig: LiveSignal) -> None:
        """Track an approved entry internally."""
        pos_id = sig.position_id or "_single"
        self._open_positions[pos_id] = {
            "direction": sig.direction,
            "entry_price": sig.entry,
            "position_size": sig.position_size,
            "entry_time": sig.timestamp,
            "strategy_type": sig.strategy_type,
        }
        self._daily_entries += 1
        logger.info(
            "RISK APPROVED entry %s %s @ %.2f size=%.2f "
            "(daily_entries=%d/%d, open=%d/%d, exposure=%.2f/%.2f)",
            pos_id, sig.direction, sig.entry, sig.position_size,
            self._daily_entries, self._cfg.max_trades_per_day,
            len(self._open_positions), self._cfg.max_concurrent_positions,
            self.total_exposure, self._cfg.max_total_exposure,
        )

    def _handle_exit(self, sig: LiveSignal) -> None:
        """Update internal tracking on exit."""
        pos_id = sig.position_id or None
        if not pos_id and self._open_positions:
            pos_id = next(iter(self._open_positions))

        if pos_id and pos_id in self._open_positions:
            pos = self._open_positions.pop(pos_id)

            fill_price = sig.entry  # exit price carried in LiveSignal.entry
            if pos["direction"] == "short":
                pnl_points = pos["entry_price"] - fill_price
            else:
                pnl_points = fill_price - pos["entry_price"]
            pnl_dollars = (
                pnl_points
                * self._inst.point_value
                * self._inst.contract_size
                * pos["position_size"]
            )
            self._daily_realized_pnl += pnl_dollars

            logger.info(
                "RISK tracked exit %s: pnl=%.2f daily_realized=%.2f",
                pos_id, pnl_dollars, self._daily_realized_pnl,
            )
        else:
            logger.debug(
                "Exit signal for untracked position %s at %s — passing through",
                pos_id, sig.timestamp,
            )

    def _block(self, sig: LiveSignal, reason: str) -> None:
        """Record and log a blocked entry signal."""
        event = RiskEvent(
            timestamp=sig.timestamp,
            event_type="blocked",
            reason=reason,
            signal=sig,
            details={
                "daily_entries": self._daily_entries,
                "open_positions": len(self._open_positions),
                "daily_realized_pnl": self._daily_realized_pnl,
                "total_exposure": self.total_exposure,
            },
        )
        self._events.append(event)
        logger.warning(
            "RISK BLOCKED %s %s at %s: %s "
            "(entries=%d, open=%d, realized=%.2f)",
            sig.position_id or "_single",
            sig.direction,
            sig.timestamp,
            reason,
            self._daily_entries,
            len(self._open_positions),
            self._daily_realized_pnl,
        )

    def _cap_size(self, sig: LiveSignal, new_size: float) -> LiveSignal:
        """Return a copy of the signal with capped position_size."""
        event = RiskEvent(
            timestamp=sig.timestamp,
            event_type="capped",
            reason="position_size_capped",
            signal=sig,
            details={
                "original_size": sig.position_size,
                "capped_size": new_size,
            },
        )
        self._events.append(event)
        logger.info(
            "RISK CAPPED %s size %.2f → %.2f (limit=%.2f)",
            sig.position_id or "_single",
            sig.position_size, new_size,
            self._cfg.max_position_size,
        )
        return LiveSignal(
            timestamp=sig.timestamp,
            direction=sig.direction,
            signal_type=sig.signal_type,
            entry=sig.entry,
            stop=sig.stop,
            take_profit=sig.take_profit,
            position_size=new_size,
            strategy_type=sig.strategy_type,
            reason=sig.reason,
            position_id=sig.position_id,
            ml_prob=sig.ml_prob,
            percentile=sig.percentile,
        )

    def _unrealized_pnl(self, current_price: float) -> float:
        """Sum unrealised P&L across all tracked open positions."""
        total = 0.0
        for pos in self._open_positions.values():
            if pos["direction"] == "long":
                pts = current_price - pos["entry_price"]
            else:
                pts = pos["entry_price"] - current_price
            total += (
                pts
                * self._inst.point_value
                * self._inst.contract_size
                * pos["position_size"]
            )
        return total

    def _forward(self, signal: LiveSignal) -> None:
        """Send signal to downstream handler."""
        if self.on_approved is not None:
            try:
                self.on_approved(signal)
            except Exception as e:
                logger.error(
                    "on_approved callback error at %s: %s",
                    signal.timestamp, e, exc_info=True,
                )
