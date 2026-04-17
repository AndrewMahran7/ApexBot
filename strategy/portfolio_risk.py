"""
Portfolio-Level Risk Manager
=============================

Enforces account-wide constraints across multiple symbols:
  1. Max total concurrent positions
  2. Max same-direction positions
  3. Max total exposure cap
  4. Correlation reduction for same-direction equity-index signals
  5. Signal prioritization by ML quality

Sits above per-symbol RiskManagers.  Does NOT replace them — it adds
a cross-symbol layer that checks portfolio-wide limits before
forwarding signals to the per-symbol risk gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from strategy.strategy_engine import LiveSignal

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskConfig:
    """Portfolio-level risk parameters."""
    max_total_concurrent: int = 3
    max_same_direction: int = 2
    max_total_exposure: float = 3.0
    max_exposure_per_direction: float = 2.0
    correlation_divisor: float = 2.0


@dataclass
class PortfolioRiskEvent:
    """Record of a portfolio-level risk decision."""
    event_type: str
    symbol: str
    strategy_type: str = ""
    direction: str = ""
    details: dict = field(default_factory=dict)


class PortfolioRiskManager:
    """
    Account-level risk gate across multiple symbols.

    Tracks open positions from all symbols and enforces
    portfolio-wide constraints before forwarding to per-symbol
    risk managers.

    Parameters
    ----------
    config : PortfolioRiskConfig
        Portfolio-wide limits.
    """

    def __init__(self, config: PortfolioRiskConfig) -> None:
        self._cfg = config
        # (symbol, pos_id) -> {direction, position_size}
        self._open_positions: dict[tuple[str, str], dict] = {}
        self._events: list[PortfolioRiskEvent] = []

        logger.info(
            "PortfolioRiskManager: max_concurrent=%d, max_same_dir=%d, "
            "max_exposure=%.1f, max_dir_exposure=%.1f, corr_divisor=%.1f",
            config.max_total_concurrent,
            config.max_same_direction,
            config.max_total_exposure,
            config.max_exposure_per_direction,
            config.correlation_divisor,
        )

    # ------------------------------------------------------------------
    # Signal ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _composite_score(signal: LiveSignal) -> float:
        """Composite quality score: weighted blend of ml_prob and quality_score."""
        ml = signal.ml_prob or 0.0
        qs = signal.quality_score or 0.0
        # ML probability is primary (70%), quality_score secondary (30%)
        return 0.7 * ml + 0.3 * qs

    def rank_signals(
        self, signals: list[tuple[str, LiveSignal]]
    ) -> list[tuple[str, LiveSignal]]:
        """Rank entry signals by composite quality score (descending)."""
        ranked = sorted(
            signals,
            key=lambda x: self._composite_score(x[1]),
            reverse=True,
        )
        if len(ranked) > 1:
            logger.info(
                "PORTFOLIO RANKING: %d signals at same timestamp — "
                "order: %s",
                len(ranked),
                ", ".join(
                    f"{sym}/{sig.strategy_type}(score={self._composite_score(sig):.3f})"
                    for sym, sig in ranked
                ),
            )
        return ranked

    # ------------------------------------------------------------------
    # Entry check
    # ------------------------------------------------------------------

    def check_entry(
        self, symbol: str, signal: LiveSignal
    ) -> Optional[LiveSignal]:
        """
        Check if an entry signal can proceed given portfolio state.

        Returns the signal (possibly with reduced size) or None if blocked.
        """
        # Total concurrent check
        if len(self._open_positions) >= self._cfg.max_total_concurrent:
            self._block(symbol, signal, "portfolio_max_concurrent")
            return None

        # Same-direction check
        dir_count = sum(
            1 for p in self._open_positions.values()
            if p["direction"] == signal.direction
        )
        if dir_count >= self._cfg.max_same_direction:
            self._block(symbol, signal, "portfolio_max_same_direction")
            return None

        # Total exposure check
        current_exposure = sum(
            p["position_size"] for p in self._open_positions.values()
        )
        remaining = self._cfg.max_total_exposure - current_exposure
        if remaining <= 0:
            self._block(symbol, signal, "portfolio_max_exposure")
            return None

        # Per-direction exposure check
        dir_exposure = sum(
            p["position_size"] for p in self._open_positions.values()
            if p["direction"] == signal.direction
        )
        dir_remaining = self._cfg.max_exposure_per_direction - dir_exposure
        if dir_remaining <= 0:
            self._block(symbol, signal, "portfolio_max_direction_exposure")
            return None
        remaining = min(remaining, dir_remaining)

        # Correlation reduction: if another symbol already has a position
        # in the same direction, reduce size
        other_same_dir = sum(
            1 for (sym, _), p in self._open_positions.items()
            if sym != symbol and p["direction"] == signal.direction
        )
        adjusted = signal
        if other_same_dir > 0:
            reduced_size = signal.position_size / self._cfg.correlation_divisor
            adjusted = _clone_signal(signal, position_size=reduced_size)
            self._events.append(PortfolioRiskEvent(
                event_type="portfolio_corr_reduced",
                symbol=symbol,
                strategy_type=signal.strategy_type,
                direction=signal.direction,
                details={
                    "original_size": signal.position_size,
                    "reduced_size": reduced_size,
                },
            ))

        # Exposure cap on adjusted size
        if adjusted.position_size > remaining:
            adjusted = _clone_signal(adjusted, position_size=remaining)

        return adjusted

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    def record_entry(self, symbol: str, signal: LiveSignal) -> None:
        """Track a new open position."""
        pos_key = (symbol, signal.position_id or "_single")
        self._open_positions[pos_key] = {
            "direction": signal.direction,
            "position_size": signal.position_size,
        }

    def record_exit(self, symbol: str, signal: LiveSignal) -> None:
        """Remove a closed position."""
        pos_key = (symbol, signal.position_id or "_single")
        self._open_positions.pop(pos_key, None)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def total_exposure(self) -> float:
        return sum(p["position_size"] for p in self._open_positions.values())

    @property
    def events(self) -> list[PortfolioRiskEvent]:
        return list(self._events)

    def positions_for_symbol(self, symbol: str) -> int:
        return sum(1 for (s, _) in self._open_positions if s == symbol)

    def reset(self) -> None:
        """Full reset for a new session."""
        self._open_positions.clear()
        self._events.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _block(
        self, symbol: str, signal: LiveSignal, reason: str,
    ) -> None:
        self._events.append(PortfolioRiskEvent(
            event_type=reason,
            symbol=symbol,
            strategy_type=signal.strategy_type,
            direction=signal.direction,
        ))
        logger.info(
            "PORTFOLIO BLOCKED: %s/%s %s -- %s",
            symbol, signal.strategy_type, signal.direction, reason,
        )


# ------------------------------------------------------------------
# Helper: clone a LiveSignal with overrides
# ------------------------------------------------------------------

def _clone_signal(sig: LiveSignal, **overrides) -> LiveSignal:
    """Create a copy of a LiveSignal with field overrides."""
    return LiveSignal(
        timestamp=overrides.get("timestamp", sig.timestamp),
        direction=overrides.get("direction", sig.direction),
        signal_type=overrides.get("signal_type", sig.signal_type),
        entry=overrides.get("entry", sig.entry),
        stop=overrides.get("stop", sig.stop),
        take_profit=overrides.get("take_profit", sig.take_profit),
        position_size=overrides.get("position_size", sig.position_size),
        strategy_type=overrides.get("strategy_type", sig.strategy_type),
        reason=overrides.get("reason", sig.reason),
        position_id=overrides.get("position_id", sig.position_id),
        ml_prob=overrides.get("ml_prob", sig.ml_prob),
        percentile=overrides.get("percentile", sig.percentile),
        quality_score=overrides.get("quality_score", sig.quality_score),
    )
