"""
Execution signal contract — machine-readable format for OpenClaw execution.

Every signal entering the execution layer must be an ExecutionSignal.
Signals are converted from the upstream LiveSignal + PropRiskLayer decision.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default TTL ──────────────────────────────────────────────────────────────

DEFAULT_SIGNAL_TTL_SECONDS = 30  # signals older than this are stale


class ExecutionSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ExecutionMode(str, Enum):
    CHALLENGE = "challenge"
    FUNDED = "funded"


@dataclass(frozen=True)
class ExecutionSignal:
    """
    Immutable, self-describing signal ready for the execution layer.

    Fields
    ------
    signal_id : str
        Unique ID (UUID4).  Duplicate IDs are rejected by the controller.
    timestamp : datetime.datetime
        When the signal was generated (tz-aware).
    symbol : str
        Instrument symbol (e.g. "MNQ", "MES").
    side : ExecutionSide
        BUY or SELL.
    contracts : int
        Number of contracts to trade (already risk-sized).
    mode : ExecutionMode
        Account operating mode — affects risk profile.
    stop_loss : float
        Absolute stop-loss price.
    take_profit : float
        Absolute take-profit price.
    reason : str
        Human-readable reason string from strategy.
    strategy_type : str
        e.g. "ema50_breakout".
    confidence : float
        ML probability or quality score (0.0-1.0).
    position_size_mult : float
        Sizing multiplier from PropRiskLayer.
    entry_price : float
        Expected entry price at signal time.
    ttl_seconds : int
        Time-to-live — signal expires this many seconds after *timestamp*.
    atm_template : str
        Name of the Tradovate ATM template to use (empty = default).
    metadata : dict
        Arbitrary extra info (ml_prob, percentile, etc.).
    expires_at : datetime.datetime
        Precomputed expiration time (timestamp + ttl).
    fingerprint : str
        Content-based hash for dedup even if signal_id differs.
    """

    signal_id: str
    timestamp: datetime.datetime
    symbol: str
    side: ExecutionSide
    contracts: int
    mode: ExecutionMode
    stop_loss: float
    take_profit: float
    reason: str = ""
    strategy_type: str = ""
    confidence: float = 0.0
    position_size_mult: float = 1.0
    entry_price: float = 0.0
    ttl_seconds: int = DEFAULT_SIGNAL_TTL_SECONDS
    atm_template: str = ""
    metadata: dict = field(default_factory=dict)

    # Computed at creation
    expires_at: datetime.datetime = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        # Compute expiration
        exp = self.timestamp + datetime.timedelta(seconds=self.ttl_seconds)
        object.__setattr__(self, "expires_at", exp)

        # Content-based fingerprint (symbol + side + SL + TP + contracts + strategy)
        content = f"{self.symbol}|{self.side.value}|{self.contracts}|{self.stop_loss:.4f}|{self.take_profit:.4f}|{self.strategy_type}"
        fp = hashlib.sha256(content.encode()).hexdigest()[:16]
        object.__setattr__(self, "fingerprint", fp)

    def is_expired(self, now: Optional[datetime.datetime] = None) -> bool:
        """Check whether the signal has exceeded its TTL."""
        now = now or datetime.datetime.now(datetime.timezone.utc)
        if self.timestamp.tzinfo is None or now.tzinfo is None:
            # Naive comparison — strip tz
            return now.replace(tzinfo=None) > self.expires_at.replace(tzinfo=None)
        return now > self.expires_at

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict for logging."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side.value,
            "contracts": self.contracts,
            "mode": self.mode.value,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reason": self.reason,
            "strategy_type": self.strategy_type,
            "confidence": self.confidence,
            "position_size_mult": self.position_size_mult,
            "entry_price": self.entry_price,
            "ttl_seconds": self.ttl_seconds,
            "atm_template": self.atm_template,
            "expires_at": self.expires_at.isoformat(),
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
        }


class SignalRegistry:
    """
    Thread-safe registry that tracks seen signal IDs and fingerprints.

    Prevents:
    - duplicate signal_id processing
    - content-identical signals (same fingerprint within cooldown window)
    """

    def __init__(self, fingerprint_cooldown_seconds: int = 120) -> None:
        self._seen_ids: set[str] = set()
        self._fingerprints: dict[str, datetime.datetime] = {}  # fp → last seen time
        self._cooldown = datetime.timedelta(seconds=fingerprint_cooldown_seconds)

    def is_duplicate(self, signal: ExecutionSignal) -> tuple[bool, str]:
        """
        Check if this signal is a duplicate.

        Returns (is_dup, reason).
        """
        if signal.signal_id in self._seen_ids:
            return True, f"duplicate signal_id={signal.signal_id}"

        now = signal.timestamp
        if signal.fingerprint in self._fingerprints:
            last = self._fingerprints[signal.fingerprint]
            if (now - last) < self._cooldown:
                return True, (
                    f"duplicate fingerprint={signal.fingerprint} "
                    f"within cooldown ({(now - last).total_seconds():.0f}s < {self._cooldown.total_seconds():.0f}s)"
                )

        return False, ""

    def register(self, signal: ExecutionSignal) -> None:
        """Mark a signal as seen."""
        self._seen_ids.add(signal.signal_id)
        self._fingerprints[signal.fingerprint] = signal.timestamp
        logger.debug("Registered signal_id=%s fp=%s", signal.signal_id, signal.fingerprint)

    def clear(self) -> None:
        """Reset the registry."""
        self._seen_ids.clear()
        self._fingerprints.clear()

    @property
    def seen_count(self) -> int:
        return len(self._seen_ids)


def create_signal_id() -> str:
    """Generate a unique signal ID."""
    return str(uuid.uuid4())
