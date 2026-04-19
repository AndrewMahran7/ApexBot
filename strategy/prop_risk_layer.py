"""
Prop-Aware Risk Layer
======================

A clean, configurable execution/risk layer that sits above the strategy
and below the execution engine. Tracks account state, applies dynamic
sizing rules, and enforces profit-lock behavior.

Designed for two modes:
  - **challenge**: maximize P(pass) under trailing DD constraint
  - **funded**: preserve capital, limit give-backs after funding

Architecture:
  AccountState  — immutable snapshot of account health
  RiskPolicy    — stateless sizing/gating rules given an AccountState
  PropRiskLayer — stateful tracker that updates AccountState per-trade
                  and delegates sizing decisions to RiskPolicy

Usage (Monte Carlo integration):
    layer = PropRiskLayer(config)
    for pnl in sampled_pnls:
        decision = layer.evaluate_trade(raw_pnl)
        if decision.blocked:
            continue
        scaled_pnl = raw_pnl * decision.size_mult
        layer.record_trade(scaled_pnl)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class AccountMode(str, Enum):
    CHALLENGE = "challenge"
    FUNDED = "funded"


@dataclass
class PropRiskConfig:
    """All configurable knobs for the prop risk layer.

    Sensible defaults for a $25K / +$1,500 / -$1,000 challenge.
    """

    # --- Account parameters ---
    mode: AccountMode = AccountMode.CHALLENGE
    starting_capital: float = 25_000.0
    profit_target: float = 1_500.0
    max_drawdown: float = 1_000.0

    # --- Base sizing ---
    base_size: float = 1.0  # default position_size multiplier

    # --- Drawdown proximity sizing ---
    # When distance_to_dd < dd_caution_zone, scale size down linearly.
    # At the edge (distance=0) size = dd_min_size.
    dd_caution_zone: float = 600.0  # dollars from DD limit
    dd_min_size: float = 0.25       # floor size when near DD

    # --- Losing streak sizing ---
    # After N consecutive losses, apply reduction.
    streak_threshold: int = 3       # after 3 losses, start reducing
    streak_reduction_per: float = 0.15  # reduce 15% per loss beyond threshold
    streak_min_size: float = 0.25   # floor for streak reduction

    # --- Progress-based sizing (challenge mode) ---
    # Divide the challenge into zones.  Columns: (min_progress%, max_progress%, size_mult)
    # progress = equity_gain / profit_target  (0 → 1.0 = passed)
    progress_zones: list[tuple[float, float, float]] = field(default_factory=lambda: [
        (-999.0, 0.0, 0.60),    # underwater: trade smaller
        (0.0, 0.33, 0.75),      # early gains: moderate
        (0.33, 0.70, 1.0),      # safe zone: full size
        (0.70, 0.90, 0.80),     # near target: tighten to protect
        (0.90, 999.0, 0.50),    # very close: small, just need a nudge
    ])

    # --- Profit-lock / capital-protection ---
    # Once equity reaches starting + lock_threshold, reduce size
    profit_lock_threshold: float = 800.0   # lock once +$800
    profit_lock_size: float = 0.50         # trade at 50% when locked

    # After reaching peak, if we give back more than this, stop trading
    giveback_halt_amount: float = 500.0    # halt if peak - equity > $500
    giveback_halt_enabled: bool = True

    # --- Daily controls ---
    daily_loss_limit: float = 400.0        # stop trading after -$400 in a day
    daily_profit_cap: float = 0.0          # 0 = disabled; stop after +X in a day
    max_trades_per_day: int = 6

    # --- Compound floor ---
    # Prevents extreme compounding when multiple sizing rules stack.
    # Final size = max(compound product, min_compound_size)
    min_compound_size: float = 0.25

    # --- Symbol-aware scaling ---
    # Per-symbol size cap (relative to base_size). Missing symbol = 1.0
    symbol_size_caps: dict[str, float] = field(default_factory=lambda: {
        "MES": 1.0,
        "MNQ": 0.80,
    })
    # Symbol preference in challenge mode: prioritize symbols with lower variance
    symbol_challenge_preference: dict[str, float] = field(default_factory=lambda: {
        "MES": 1.0,
        "MNQ": 0.75,
    })

    # --- Funded mode overrides ---
    funded_base_size: float = 0.75
    funded_profit_lock_threshold: float = 2_000.0
    funded_profit_lock_size: float = 0.50
    funded_giveback_halt_amount: float = 1_000.0
    funded_daily_loss_limit: float = 500.0
    funded_max_trades_per_day: int = 8

    @classmethod
    def for_challenge(cls, **overrides) -> PropRiskConfig:
        """Factory for challenge mode — OPTIMAL config based on MC validation.

        Sensitivity analysis (2,000 sims, 500 max trades) showed:
          Combined: 0.35x → 63.5%, 0.25x → 56.8%, 0.50x → 52.8%, 1.0x → 40.7%
          MES:      0.50x → 72.5%,  dynamic 0.50x + DD prox → 68.0% (88.2% @1500 trades)
          MNQ:      0.35x → 68.1%, 0.25x → 63.4%, 0.50x → 57.4%

        Optimal combined/MNQ: 0.35x flat sizing + daily loss limit.
        DD proximity HURTS for MNQ (creates recovery trap at DD edge).
        For MES-only trading, use for_challenge_mes() instead.
        """
        defaults = {
            "mode": AccountMode.CHALLENGE,
            "base_size": 0.35,
            "dd_caution_zone": 0.0,                # disabled — hurts MNQ
            "dd_min_size": 0.35,                    # same as base (no scaling)
            "profit_lock_threshold": 999_999.0,     # disabled
            "profit_lock_size": 1.0,
            "giveback_halt_enabled": False,
            "daily_loss_limit": 999_999.0,          # disabled for MC fairness
            "streak_threshold": 99,                 # disabled
            "min_compound_size": 0.25,
            "max_trades_per_day": 999,              # disabled for MC fairness
            "progress_zones": [
                (-999.0, 999.0, 1.0),               # flat
            ],
            "symbol_challenge_preference": {
                "MES": 1.0,
                "MNQ": 1.0,
            },
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_challenge_mes(cls, **overrides) -> PropRiskConfig:
        """Factory for MES-only challenge mode.

        MES has lower variance ($115 std vs MNQ $206), so DD proximity
        helps without creating a recovery trap.
        At 500 trades: 68% pass (4.7% fail). At 1500 trades: 88% pass.
        Compared to MES fixed 0.50x: 72.5% pass but 9.9% fail.
        Use this when you have time for 300+ trades (typically 4+ months).
        """
        defaults = {
            "mode": AccountMode.CHALLENGE,
            "base_size": 0.50,
            "dd_caution_zone": 500.0,
            "dd_min_size": 0.15,
            "profit_lock_threshold": 999_999.0,
            "profit_lock_size": 1.0,
            "giveback_halt_enabled": False,
            "daily_loss_limit": 999_999.0,          # disabled for MC fairness
            "streak_threshold": 99,
            "min_compound_size": 0.15,
            "max_trades_per_day": 999,              # disabled for MC fairness
            "progress_zones": [
                (-999.0, 999.0, 1.0),
            ],
            "symbol_challenge_preference": {
                "MES": 1.0,
                "MNQ": 1.0,
            },
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_funded(cls, **overrides) -> PropRiskConfig:
        """Factory for funded mode with conservative defaults.

        In funded mode, the goal is capital preservation (no profit target).
        Uses 0.35x base sizing (same as challenge) with no DD proximity
        to avoid the recovery trap seen in MC testing.
        """
        defaults = {
            "mode": AccountMode.FUNDED,
            "base_size": 0.35,
            "profit_target": 999_999.0,  # no pass target
            "dd_caution_zone": 0.0,      # disabled
            "dd_min_size": 0.35,
            "profit_lock_threshold": 999_999.0,
            "profit_lock_size": 1.0,
            "giveback_halt_amount": 1_000.0,
            "giveback_halt_enabled": False,
            "daily_loss_limit": 999_999.0,
            "max_trades_per_day": 999,
            "min_compound_size": 0.25,
        }
        defaults.update(overrides)
        return cls(**defaults)


# ---------------------------------------------------------------------------
# Account State (immutable snapshot)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AccountState:
    """Immutable snapshot of account health at a point in time."""
    equity: float
    peak_equity: float
    starting_capital: float
    trailing_dd_limit: float     # peak - max_drawdown; equity below this = fail
    realized_pnl: float         # cumulative realized (from starting capital)
    daily_pnl: float
    daily_trades: int
    consecutive_losses: int
    mode: AccountMode

    @property
    def equity_gain(self) -> float:
        """Profit from starting capital."""
        return self.equity - self.starting_capital

    @property
    def distance_to_dd(self) -> float:
        """Dollars above the trailing DD limit (>0 = safe)."""
        return self.equity - self.trailing_dd_limit

    @property
    def distance_to_target(self) -> float:
        """Dollars remaining to profit target. Negative = already passed."""
        return (self.starting_capital + 1_500.0) - self.equity  # overridden in layer

    @property
    def progress(self) -> float:
        """Fraction of profit target achieved (0 → 1)."""
        target = 1_500.0  # default, overridden at layer level
        if target <= 0:
            return 1.0
        return self.equity_gain / target

    @property
    def passed(self) -> bool:
        return self.equity >= self.starting_capital + 1_500.0

    @property
    def failed(self) -> bool:
        return self.equity <= self.trailing_dd_limit


# ---------------------------------------------------------------------------
# Trade Decision
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    """Output of the risk layer's evaluate_trade method."""
    size_mult: float      # final position size multiplier (0 = blocked)
    blocked: bool         # True if trade should not be taken
    reasons: list[str]    # human-readable explanation of adjustments
    components: dict      # breakdown of sizing factors

    @property
    def effective_size(self) -> float:
        return 0.0 if self.blocked else self.size_mult


# ---------------------------------------------------------------------------
# Risk Policy (stateless rules)
# ---------------------------------------------------------------------------

class RiskPolicy:
    """Stateless sizing / gating rules. Given config + state → decision."""

    def __init__(self, config: PropRiskConfig):
        self.cfg = config

    def evaluate(
        self,
        state: AccountState,
        symbol: str = "",
        raw_pnl: float = 0.0,
    ) -> TradeDecision:
        """Decide whether to take a trade and at what size."""
        reasons: list[str] = []
        components: dict = {}
        blocked = False

        # --- Hard blocks ---
        if state.mode == AccountMode.CHALLENGE:
            if state.passed:
                return TradeDecision(0.0, True, ["PASSED — challenge complete"], {})
            if state.failed:
                return TradeDecision(0.0, True, ["FAILED — DD breached"], {})

        # Daily limits
        if state.daily_trades >= self.cfg.max_trades_per_day:
            return TradeDecision(0.0, True, ["max daily trades reached"], {})

        limit = self._daily_loss_limit()
        if state.daily_pnl <= -limit:
            return TradeDecision(0.0, True, [f"daily loss limit (-${limit:.0f})"], {})

        if self.cfg.daily_profit_cap > 0 and state.daily_pnl >= self.cfg.daily_profit_cap:
            return TradeDecision(0.0, True, [f"daily profit cap (+${self.cfg.daily_profit_cap:.0f})"], {})

        # Giveback halt
        if self.cfg.giveback_halt_enabled:
            giveback = state.peak_equity - state.equity
            halt_amt = self._giveback_halt()
            if giveback >= halt_amt and state.equity_gain > 0:
                return TradeDecision(
                    0.0, True,
                    [f"giveback halt: gave back ${giveback:.0f} from peak"],
                    {},
                )

        # --- Size computation ---
        size = self._base_size()

        # 1. Progress-based zone sizing (challenge mode)
        if state.mode == AccountMode.CHALLENGE:
            progress = state.equity_gain / self.cfg.profit_target if self.cfg.profit_target > 0 else 0.0
            zone_mult = self._progress_zone_mult(progress)
            components["progress_zone"] = round(zone_mult, 3)
            if zone_mult != 1.0:
                reasons.append(f"progress zone ({progress:.0%}): {zone_mult:.2f}x")
            size *= zone_mult

        # 2. DD proximity scaling
        dd_mult = self._dd_proximity_mult(state.distance_to_dd)
        components["dd_proximity"] = round(dd_mult, 3)
        if dd_mult < 1.0:
            reasons.append(f"DD proximity (${state.distance_to_dd:.0f} buffer): {dd_mult:.2f}x")
        size *= dd_mult

        # 3. Losing streak reduction
        streak_mult = self._streak_mult(state.consecutive_losses)
        components["streak"] = round(streak_mult, 3)
        if streak_mult < 1.0:
            reasons.append(f"losing streak ({state.consecutive_losses}): {streak_mult:.2f}x")
        size *= streak_mult

        # 4. Profit-lock reduction
        lock_mult = self._profit_lock_mult(state.equity_gain)
        components["profit_lock"] = round(lock_mult, 3)
        if lock_mult < 1.0:
            reasons.append(f"profit lock (+${state.equity_gain:.0f}): {lock_mult:.2f}x")
        size *= lock_mult

        # 5. Symbol-aware cap
        sym_mult = self._symbol_mult(symbol, state)
        components["symbol"] = round(sym_mult, 3)
        if sym_mult < 1.0:
            reasons.append(f"symbol cap ({symbol}): {sym_mult:.2f}x")
        size *= sym_mult

        # Compound floor: prevent extreme stacking of multipliers
        size = max(size, self.cfg.min_compound_size)
        components["final"] = round(size, 3)
        components["base"] = round(self._base_size(), 3)

        if not reasons:
            reasons.append(f"base size: {size:.2f}x")

        return TradeDecision(
            size_mult=round(size, 4),
            blocked=blocked,
            reasons=reasons,
            components=components,
        )

    # --- Internal rule implementations ---

    def _base_size(self) -> float:
        if self.cfg.mode == AccountMode.FUNDED:
            return self.cfg.funded_base_size
        return self.cfg.base_size

    def _daily_loss_limit(self) -> float:
        if self.cfg.mode == AccountMode.FUNDED:
            return self.cfg.funded_daily_loss_limit
        return self.cfg.daily_loss_limit

    def _giveback_halt(self) -> float:
        if self.cfg.mode == AccountMode.FUNDED:
            return self.cfg.funded_giveback_halt_amount
        return self.cfg.giveback_halt_amount

    def _progress_zone_mult(self, progress: float) -> float:
        """Look up sizing multiplier for current progress zone."""
        for lo, hi, mult in self.cfg.progress_zones:
            if lo <= progress < hi:
                return mult
        return 1.0

    def _dd_proximity_mult(self, distance_to_dd: float) -> float:
        """Linear scaling when close to DD limit."""
        if distance_to_dd >= self.cfg.dd_caution_zone:
            return 1.0
        if distance_to_dd <= 0:
            return self.cfg.dd_min_size
        # Linear interpolation
        t = distance_to_dd / self.cfg.dd_caution_zone
        return self.cfg.dd_min_size + t * (1.0 - self.cfg.dd_min_size)

    def _streak_mult(self, consecutive_losses: int) -> float:
        """Reduce size after losing streak."""
        if consecutive_losses < self.cfg.streak_threshold:
            return 1.0
        excess = consecutive_losses - self.cfg.streak_threshold
        reduction = 1.0 - (excess + 1) * self.cfg.streak_reduction_per
        return max(reduction, self.cfg.streak_min_size)

    def _profit_lock_mult(self, equity_gain: float) -> float:
        """Reduce size once profit threshold is reached."""
        threshold = self.cfg.profit_lock_threshold
        lock_size = self.cfg.profit_lock_size
        if self.cfg.mode == AccountMode.FUNDED:
            threshold = self.cfg.funded_profit_lock_threshold
            lock_size = self.cfg.funded_profit_lock_size
        if equity_gain >= threshold:
            return lock_size
        return 1.0

    def _symbol_mult(self, symbol: str, state: AccountState) -> float:
        """Symbol-aware sizing cap."""
        if not symbol:
            return 1.0
        if state.mode == AccountMode.CHALLENGE:
            return self.cfg.symbol_challenge_preference.get(symbol, 1.0)
        return self.cfg.symbol_size_caps.get(symbol, 1.0)


# ---------------------------------------------------------------------------
# Prop Risk Layer (stateful tracker)
# ---------------------------------------------------------------------------

class PropRiskLayer:
    """Stateful account tracker that wraps RiskPolicy.

    Updates equity/peak/trailing-DD after each trade and provides
    per-trade sizing decisions.
    """

    def __init__(self, config: PropRiskConfig):
        self.cfg = config
        self.policy = RiskPolicy(config)

        # Mutable state
        self._equity = config.starting_capital
        self._peak_equity = config.starting_capital
        self._trailing_dd_limit = config.starting_capital - config.max_drawdown
        self._realized_pnl = 0.0
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._total_trades = 0
        self._halted = False
        self._halt_reason = ""

    def reset(self):
        """Reset to initial state."""
        self.__init__(self.cfg)

    def reset_day(self):
        """Reset daily counters (call at start of each trading day)."""
        self._daily_pnl = 0.0
        self._daily_trades = 0

    @property
    def state(self) -> AccountState:
        """Current account state snapshot."""
        return AccountState(
            equity=self._equity,
            peak_equity=self._peak_equity,
            starting_capital=self.cfg.starting_capital,
            trailing_dd_limit=self._trailing_dd_limit,
            realized_pnl=self._realized_pnl,
            daily_pnl=self._daily_pnl,
            daily_trades=self._daily_trades,
            consecutive_losses=self._consecutive_losses,
            mode=self.cfg.mode,
        )

    @property
    def passed(self) -> bool:
        return self._equity >= self.cfg.starting_capital + self.cfg.profit_target

    @property
    def failed(self) -> bool:
        return self._equity <= self._trailing_dd_limit

    @property
    def active(self) -> bool:
        return not self.passed and not self.failed and not self._halted

    def evaluate_trade(self, symbol: str = "", raw_pnl: float = 0.0) -> TradeDecision:
        """Evaluate whether the next trade should be taken and at what size.

        Parameters
        ----------
        symbol : str
            Instrument symbol (e.g. "MES", "MNQ") for symbol-aware scaling.
        raw_pnl : float
            Expected or historical PnL for this trade (used for diagnostics only,
            NOT for decision-making — no future information).

        Returns
        -------
        TradeDecision
        """
        if self._halted:
            return TradeDecision(0.0, True, [f"halted: {self._halt_reason}"], {})
        return self.policy.evaluate(self.state, symbol=symbol)

    def record_trade(self, net_pnl: float, symbol: str = ""):
        """Record a completed trade and update all state.

        Must be called AFTER evaluate_trade decides to take the trade
        and AFTER PnL has been scaled by the decision's size_mult.
        """
        # Update equity
        self._equity += net_pnl
        self._realized_pnl += net_pnl
        self._daily_pnl += net_pnl
        self._daily_trades += 1
        self._total_trades += 1

        # Update peak & trailing DD
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity
            self._trailing_dd_limit = self._peak_equity - self.cfg.max_drawdown

        # Update streak
        if net_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Check terminal conditions
        if self.cfg.mode == AccountMode.CHALLENGE:
            if self.passed:
                self._halted = True
                self._halt_reason = "PASSED"
            elif self.failed:
                self._halted = True
                self._halt_reason = "FAILED"

    def get_summary(self) -> dict:
        """Return a summary of the current layer state."""
        return {
            "mode": self.cfg.mode.value,
            "equity": round(self._equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "trailing_dd_limit": round(self._trailing_dd_limit, 2),
            "equity_gain": round(self._equity - self.cfg.starting_capital, 2),
            "distance_to_dd": round(self._equity - self._trailing_dd_limit, 2),
            "distance_to_target": round(
                (self.cfg.starting_capital + self.cfg.profit_target) - self._equity, 2
            ),
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_trades": self._daily_trades,
            "total_trades": self._total_trades,
            "passed": self.passed,
            "failed": self.failed,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
        }
