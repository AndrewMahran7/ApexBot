"""
Prop Firm Challenge Mode
=========================

Enforces strict prop firm account rules on top of the normal risk manager.

Account rules (configurable):
    Starting balance:  $25,000
    Profit target:     +$1,500
    Max drawdown:      -$1,000  (intraday trailing)
    Max size:          4 minis / 40 micros

Core principle:
    Maximize probability of reaching the profit target BEFORE hitting
    the trailing drawdown.  Every decision prioritises survival and
    controlled gains over long-term profitability.

Components:
    PropConfig          — all challenge parameters in one place
    PropEquityTracker   — real-time equity / trailing DD tracking
    PropRiskGate        — pre-signal filter with challenge-specific rules
    PropPositionSizer   — staged sizing based on equity progress

Wiring:
    The gate sits between StrategyEngine and the normal RiskManager.
    Signals flow:  engine → prop_gate.on_signal → risk.on_signal → paper

    prop_gate.on_bar() is called each bar (before engine.on_bar) to
    update equity state from PaperEngine.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class PropConfig:
    """Prop firm challenge parameters."""

    enabled: bool = True

    # Account
    starting_capital: float = 25_000.0
    profit_target: float = 1_500.0
    max_drawdown: float = 1_000.0
    drawdown_type: str = "trailing_intraday"

    # Position limits
    max_minis: int = 4
    max_micros: int = 40

    # Daily risk
    daily_loss_limit: float = 300.0
    daily_profit_lock: float = 400.0

    # Trailing DD safety buffer — stop trading when equity is within
    # this many dollars of the trailing drawdown level
    dd_buffer: float = 200.0

    # Max trades per day
    max_trades_per_day: int = 4

    # No-giveback rule: if daily PnL was above this and drops by
    # more than giveback_drop, stop trading for the day
    giveback_threshold: float = 300.0
    giveback_drop: float = 200.0

    # Kill switch: consecutive losses before halting
    max_consecutive_losses: int = 3

    # Trade filtering
    allowed_entry_types: tuple[str, ...] = ("breakout",)
    min_ml_prob: float = 0.60

    # Tighter exits
    reward_risk_override: float = 1.2
    stop_tightening_pct: float = 0.85  # SL distance = 85% of original

    # Position sizing tiers (keyed by equity gain from start)
    # Each tier: (min_gain, max_gain, size_multiplier)
    sizing_tiers: tuple[tuple[float, float, float], ...] = (
        (0.0, 500.0, 0.25),       # cautious ramp
        (500.0, 1200.0, 0.50),    # medium
        (1200.0, 1500.0, 0.35),   # protect gains / one push
    )


# ------------------------------------------------------------------
# Event record
# ------------------------------------------------------------------

@dataclass
class PropEvent:
    """Immutable record of a prop challenge decision."""

    timestamp: datetime.datetime
    event_type: str
    # "prop_blocked", "prop_stopped_day", "prop_dd_warning",
    # "prop_target_reached", "prop_drawdown_breach",
    # "prop_size_adjusted", "prop_filter_blocked"
    reason: str
    details: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Equity tracker
# ------------------------------------------------------------------

class PropEquityTracker:
    """
    Tracks intraday trailing drawdown and equity progress.

    Trailing DD rule:
        trailing_dd_level = peak_equity - max_drawdown

    The peak is updated on every mark-to-market (including intraday
    unrealised moves).  If equity touches or breaches the DD level,
    the challenge is failed.
    """

    def __init__(self, config: PropConfig) -> None:
        self._cfg = config
        self._starting_capital = config.starting_capital

        self._current_equity: float = config.starting_capital
        self._peak_equity: float = config.starting_capital
        self._trailing_dd_level: float = config.starting_capital - config.max_drawdown

        # Daily tracking
        self._day_start_equity: float = config.starting_capital
        self._daily_peak_pnl: float = 0.0
        self._daily_pnl: float = 0.0
        self._current_date: Optional[datetime.date] = None

        # Status
        self._passed: bool = False
        self._failed: bool = False

        logger.info(
            "PropEquityTracker: start=%.2f target=+%.2f dd=-%.2f "
            "trailing_dd_level=%.2f",
            config.starting_capital, config.profit_target,
            config.max_drawdown, self._trailing_dd_level,
        )

    def update(self, equity: float, ts: datetime.datetime) -> None:
        """Update equity and check pass/fail conditions."""
        bar_date = ts.date() if isinstance(ts, datetime.datetime) else ts

        # Day change
        if self._current_date is None or bar_date != self._current_date:
            self._day_start_equity = self._current_equity
            self._daily_peak_pnl = 0.0
            self._daily_pnl = 0.0
            self._current_date = bar_date

        self._current_equity = equity
        self._daily_pnl = equity - self._day_start_equity
        self._daily_peak_pnl = max(self._daily_peak_pnl, self._daily_pnl)

        # Update peak and trailing DD level
        if equity > self._peak_equity:
            self._peak_equity = equity
            self._trailing_dd_level = equity - self._cfg.max_drawdown

        # Check pass
        gain = equity - self._starting_capital
        if gain >= self._cfg.profit_target:
            self._passed = True

        # Check fail
        if equity <= self._trailing_dd_level:
            self._failed = True

    def reset_day(self) -> None:
        """Explicit day reset (called by gate on day change)."""
        self._day_start_equity = self._current_equity
        self._daily_peak_pnl = 0.0
        self._daily_pnl = 0.0

    # ---- Properties ----

    @property
    def current_equity(self) -> float:
        return self._current_equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def trailing_dd_level(self) -> float:
        return self._trailing_dd_level

    @property
    def equity_gain(self) -> float:
        return self._current_equity - self._starting_capital

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def daily_peak_pnl(self) -> float:
        return self._daily_peak_pnl

    @property
    def dd_buffer_remaining(self) -> float:
        """Dollars between current equity and trailing DD level."""
        return self._current_equity - self._trailing_dd_level

    @property
    def passed(self) -> bool:
        return self._passed

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def active(self) -> bool:
        return not self._passed and not self._failed


# ------------------------------------------------------------------
# Position sizer
# ------------------------------------------------------------------

class PropPositionSizer:
    """
    Staged position sizing based on equity progress.

    Uses PropConfig.sizing_tiers to determine multiplier based on
    current gain from starting capital.  Returns a multiplier applied
    to the signal's position_size field.
    """

    def __init__(self, config: PropConfig) -> None:
        self._tiers = config.sizing_tiers

    def compute(self, equity_gain: float) -> float:
        """Return size multiplier for the given equity gain."""
        for min_gain, max_gain, mult in self._tiers:
            if min_gain <= equity_gain < max_gain:
                return mult
        # Beyond all tiers — use the last tier
        if self._tiers:
            return self._tiers[-1][2]
        return 1.0


# ------------------------------------------------------------------
# Risk gate
# ------------------------------------------------------------------

class PropRiskGate:
    """
    Pre-signal filter enforcing prop firm challenge rules.

    Sits between StrategyEngine and the normal RiskManager.
    Blocks or adjusts signals based on challenge state.

    Parameters
    ----------
    config : PropConfig
        Challenge parameters.
    on_approved : callable
        Next handler in the chain (typically risk.on_signal).
    get_equity : callable
        Returns current mark-to-market equity from PaperEngine.
    """

    def __init__(
        self,
        config: PropConfig,
        on_approved: Optional[Callable[[LiveSignal], None]] = None,
        get_equity: Optional[Callable[[], float]] = None,
    ) -> None:
        self._cfg = config
        self.on_approved = on_approved
        self._get_equity = get_equity

        self._tracker = PropEquityTracker(config)
        self._sizer = PropPositionSizer(config)

        # Daily state
        self._current_date: Optional[datetime.date] = None
        self._daily_entries: int = 0
        self._consecutive_losses: int = 0
        self._day_stopped: bool = False
        self._halted: bool = False  # permanent halt (DD breach or target)

        # Events
        self._events: list[PropEvent] = []

        logger.info(
            "PropRiskGate initialised: target=+%.2f, dd=-%.2f, "
            "daily_loss=%.2f, daily_profit_lock=%.2f, "
            "max_trades=%d, allowed=%s",
            config.profit_target, config.max_drawdown,
            config.daily_loss_limit, config.daily_profit_lock,
            config.max_trades_per_day,
            config.allowed_entry_types,
        )

    # ------------------------------------------------------------------
    # Bar-level update
    # ------------------------------------------------------------------

    def on_bar(self, bar: dict) -> None:
        """
        Update prop state from the latest bar.  Call BEFORE engine.on_bar().

        Reads equity from PaperEngine (via get_equity callback) and
        updates the tracker.
        """
        ts = bar["timestamp"]
        bar_date = ts.date() if isinstance(ts, datetime.datetime) else ts

        # Day change
        if self._current_date is None or bar_date != self._current_date:
            self._current_date = bar_date
            self._daily_entries = 0
            self._day_stopped = False
            self._tracker.reset_day()
            logger.debug("Prop day reset: %s", bar_date)

        # Get current equity
        if self._get_equity is not None:
            equity = self._get_equity()
        else:
            equity = self._tracker.current_equity

        self._tracker.update(equity, ts)

        # --- Check halt conditions ---

        # Challenge passed
        if self._tracker.passed and not self._halted:
            self._halted = True
            self._record("prop_target_reached",
                         f"Profit target +${self._cfg.profit_target:.0f} reached",
                         ts, {"equity": equity,
                              "gain": self._tracker.equity_gain})
            logger.info("PROP: Target reached at equity=%.2f", equity)

        # Challenge failed — DD breach
        if self._tracker.failed and not self._halted:
            self._halted = True
            self._record("prop_drawdown_breach",
                         f"Trailing drawdown breached at equity={equity:.2f}",
                         ts, {"equity": equity,
                              "dd_level": self._tracker.trailing_dd_level})
            logger.error("PROP: Drawdown breach at equity=%.2f", equity)

        # --- Daily stop conditions ---

        # Daily loss limit
        if (self._tracker.daily_pnl <= -self._cfg.daily_loss_limit
                and not self._day_stopped):
            self._day_stopped = True
            self._record("prop_stopped_day",
                         f"Daily loss ${self._tracker.daily_pnl:.2f} "
                         f"breached limit -${self._cfg.daily_loss_limit:.0f}",
                         ts, {"daily_pnl": self._tracker.daily_pnl})
            logger.warning("PROP: Daily loss limit hit: %.2f",
                           self._tracker.daily_pnl)

        # Daily profit lock
        if (self._tracker.daily_pnl >= self._cfg.daily_profit_lock
                and not self._day_stopped):
            self._day_stopped = True
            self._record("prop_stopped_day",
                         f"Daily profit ${self._tracker.daily_pnl:.2f} "
                         f"locked at +${self._cfg.daily_profit_lock:.0f}",
                         ts, {"daily_pnl": self._tracker.daily_pnl})
            logger.info("PROP: Daily profit lock: %.2f",
                        self._tracker.daily_pnl)

        # No-giveback rule
        if (self._tracker.daily_peak_pnl >= self._cfg.giveback_threshold
                and (self._tracker.daily_peak_pnl - self._tracker.daily_pnl)
                >= self._cfg.giveback_drop
                and not self._day_stopped):
            self._day_stopped = True
            self._record(
                "prop_stopped_day",
                f"No-giveback: peak +${self._tracker.daily_peak_pnl:.2f} "
                f"dropped to +${self._tracker.daily_pnl:.2f}",
                ts, {"daily_peak": self._tracker.daily_peak_pnl,
                     "daily_pnl": self._tracker.daily_pnl},
            )
            logger.warning("PROP: No-giveback rule triggered")

        # DD buffer warning
        if self._tracker.dd_buffer_remaining <= self._cfg.dd_buffer:
            if not self._day_stopped:
                self._day_stopped = True
                self._record(
                    "prop_dd_warning",
                    f"DD buffer only ${self._tracker.dd_buffer_remaining:.2f} "
                    f"remaining (limit ${self._cfg.dd_buffer:.0f})",
                    ts, {"buffer": self._tracker.dd_buffer_remaining,
                         "dd_level": self._tracker.trailing_dd_level},
                )
                logger.warning("PROP: DD buffer critical: %.2f",
                               self._tracker.dd_buffer_remaining)

    # ------------------------------------------------------------------
    # Signal filtering
    # ------------------------------------------------------------------

    def on_signal(self, signal: LiveSignal) -> None:
        """
        Filter signals through prop challenge rules.

        Exit signals always pass through.  Entry signals are checked
        against all prop constraints.
        """
        # Exits always pass
        if signal.is_exit:
            self._forward(signal)
            return

        if not signal.is_entry:
            self._forward(signal)
            return

        # --- Entry checks ---
        ts = signal.timestamp

        # Halted (passed or failed)
        if self._halted:
            self._block(signal, "challenge_halted",
                        "Challenge is over (passed or failed)")
            return

        # Day stopped
        if self._day_stopped:
            self._block(signal, "day_stopped",
                        "Trading stopped for the day")
            return

        # Max trades per day
        if self._daily_entries >= self._cfg.max_trades_per_day:
            self._block(signal, "prop_max_trades",
                        f"Max {self._cfg.max_trades_per_day} trades/day")
            return

        # Consecutive losses kill switch
        if self._consecutive_losses >= self._cfg.max_consecutive_losses:
            self._day_stopped = True
            self._block(signal, "consecutive_losses",
                        f"{self._consecutive_losses} consecutive losses")
            return

        # Entry type filter
        entry_type = self._extract_entry_type(signal.strategy_type)
        if entry_type not in self._cfg.allowed_entry_types:
            self._block(signal, "prop_filter_blocked",
                        f"Entry type '{entry_type}' not allowed "
                        f"(only {self._cfg.allowed_entry_types})")
            return

        # ML confidence filter
        if signal.ml_prob < self._cfg.min_ml_prob:
            self._block(signal, "prop_filter_blocked",
                        f"ML prob {signal.ml_prob:.3f} < "
                        f"min {self._cfg.min_ml_prob:.3f}")
            return

        # --- Adjust signal ---

        # Position sizing
        gain = self._tracker.equity_gain
        size_mult = self._sizer.compute(gain)
        new_size = signal.position_size * size_mult

        # Tighten stops and targets
        adjusted = self._tighten_exits(signal, new_size)

        if abs(new_size - signal.position_size) > 0.001:
            self._record("prop_size_adjusted",
                         f"Size {signal.position_size:.3f} → "
                         f"{new_size:.3f} (gain=${gain:.2f}, "
                         f"mult={size_mult:.2f})",
                         ts, {"original_size": signal.position_size,
                              "new_size": new_size,
                              "equity_gain": gain,
                              "multiplier": size_mult})

        self._daily_entries += 1
        self._forward(adjusted)

    def on_trade_closed(self, net_pnl: float) -> None:
        """
        Notify gate that a trade has closed.

        Used to track consecutive losses for kill switch.
        """
        if net_pnl > 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            logger.debug("PROP: consecutive losses = %d",
                         self._consecutive_losses)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def tracker(self) -> PropEquityTracker:
        return self._tracker

    @property
    def events(self) -> list[PropEvent]:
        return list(self._events)

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def day_stopped(self) -> bool:
        return self._day_stopped

    @property
    def daily_entries(self) -> int:
        return self._daily_entries

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tighten_exits(self, sig: LiveSignal, new_size: float) -> LiveSignal:
        """Return a copy of the signal with tighter SL/TP and adjusted size."""
        entry = sig.entry
        orig_sl_dist = abs(entry - sig.stop)
        orig_tp_dist = abs(sig.take_profit - entry)

        # Tighten stop
        new_sl_dist = orig_sl_dist * self._cfg.stop_tightening_pct

        # Tighten target — use override RR
        new_tp_dist = new_sl_dist * self._cfg.reward_risk_override

        if sig.direction == "long":
            new_stop = entry - new_sl_dist
            new_tp = entry + new_tp_dist
        else:
            new_stop = entry + new_sl_dist
            new_tp = entry - new_tp_dist

        return LiveSignal(
            timestamp=sig.timestamp,
            direction=sig.direction,
            signal_type=sig.signal_type,
            entry=sig.entry,
            stop=new_stop,
            take_profit=new_tp,
            position_size=new_size,
            strategy_type=sig.strategy_type,
            reason=sig.reason,
            position_id=sig.position_id,
            ml_prob=sig.ml_prob,
            percentile=sig.percentile,
        )

    @staticmethod
    def _extract_entry_type(strategy_type: str) -> str:
        """Extract entry type from strategy_type like 'ema50_breakout'."""
        parts = strategy_type.split("_", 1)
        return parts[1] if len(parts) > 1 else strategy_type

    def _block(self, sig: LiveSignal, event_type: str, reason: str) -> None:
        self._record(event_type, reason, sig.timestamp, {
            "direction": sig.direction,
            "strategy_type": sig.strategy_type,
            "ml_prob": sig.ml_prob,
            "entry": sig.entry,
        })
        logger.info("PROP BLOCKED %s %s @ %s: %s",
                     sig.direction, sig.strategy_type, sig.timestamp, reason)

    def _record(self, event_type: str, reason: str,
                ts: datetime.datetime, details: dict | None = None) -> None:
        self._events.append(PropEvent(
            timestamp=ts,
            event_type=event_type,
            reason=reason,
            details=details or {},
        ))

    def _forward(self, signal: LiveSignal) -> None:
        if self.on_approved is not None:
            try:
                self.on_approved(signal)
            except Exception as e:
                logger.error("PropRiskGate on_approved error: %s", e,
                             exc_info=True)
