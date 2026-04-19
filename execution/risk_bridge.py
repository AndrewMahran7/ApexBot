"""
Bridge between the upstream risk/sizing layer and the execution signal format.

Converts LiveSignal + PropRiskLayer decision → ExecutionSignal.

Rules:
- Risk decisions are made UPSTREAM (PropRiskLayer / RiskManager)
- This bridge only TRANSLATES — it does NOT add risk logic
- Sizing multiplier is converted to integer contracts
- Challenge vs. funded mode is propagated from PropRiskConfig
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from execution.signal_schema import (
    DEFAULT_SIGNAL_TTL_SECONDS,
    ExecutionMode,
    ExecutionSide,
    ExecutionSignal,
    create_signal_id,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskBridgeConfig:
    """Configuration for how upstream sizing translates to contracts."""

    # Default account mode
    default_mode: ExecutionMode = ExecutionMode.CHALLENGE

    # Base contracts (before sizing multiplier)
    base_contracts: int = 1

    # Minimum contracts (floor — never go to 0 unless blocked)
    min_contracts: int = 1

    # Maximum contracts (hard cap for safety)
    max_contracts: int = 5

    # Signal TTL
    signal_ttl_seconds: int = DEFAULT_SIGNAL_TTL_SECONDS

    # Default ATM template names per mode
    challenge_atm_template: str = ""
    funded_atm_template: str = ""

    # Symbol to instrument-symbol mapping (for Tradovate contract naming)
    symbol_map: dict[str, str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.symbol_map is None:
            self.symbol_map = {}


def compute_contracts(
    base_contracts: int,
    size_mult: float,
    min_contracts: int = 1,
    max_contracts: int = 5,
) -> int:
    """
    Convert a sizing multiplier to integer contracts.

    Parameters
    ----------
    base_contracts : int
        Base number of contracts at 1.0x.
    size_mult : float
        Multiplier from PropRiskLayer (e.g. 0.35).
    min_contracts : int
        Floor (1 unless trade is blocked, i.e. size_mult=0).
    max_contracts : int
        Hard cap.

    Returns
    -------
    int
        Number of contracts to trade.
    """
    if size_mult <= 0:
        return 0

    raw = base_contracts * size_mult
    contracts = max(min_contracts, min(max_contracts, round(raw)))
    logger.debug(
        "compute_contracts: base=%d × mult=%.2f = %.2f → %d (min=%d, max=%d)",
        base_contracts, size_mult, raw, contracts, min_contracts, max_contracts,
    )
    return contracts


class RiskBridge:
    """
    Converts upstream LiveSignal + sizing decision into an ExecutionSignal.

    Usage:
        bridge = RiskBridge(config)
        exec_signal = bridge.convert(live_signal, size_mult=0.35, symbol="MNQ")
    """

    def __init__(self, config: Optional[RiskBridgeConfig] = None) -> None:
        self.config = config or RiskBridgeConfig()
        logger.info(
            "RiskBridge initialized: mode=%s, base_contracts=%d, min=%d, max=%d",
            self.config.default_mode.value,
            self.config.base_contracts,
            self.config.min_contracts,
            self.config.max_contracts,
        )

    def convert(
        self,
        live_signal,  # LiveSignal from strategy_engine
        size_mult: float = 1.0,
        symbol: str = "",
        mode: Optional[ExecutionMode] = None,
        atm_template: str = "",
        extra_metadata: Optional[dict] = None,
        override_contracts: Optional[int] = None,
    ) -> Optional[ExecutionSignal]:
        """
        Convert a LiveSignal to an ExecutionSignal.

        Parameters
        ----------
        live_signal
            LiveSignal object from the strategy pipeline.
        size_mult : float
            Position sizing multiplier from PropRiskLayer (0.0 = blocked).
        symbol : str
            Override symbol (if empty, attempts to extract from live_signal).
        mode : ExecutionMode, optional
            Override mode (defaults to config.default_mode).
        atm_template : str
            Override ATM template name.
        extra_metadata : dict, optional
            Additional metadata to include.
        override_contracts : int, optional
            If provided, skip compute_contracts() and use this value directly.
            Used by PropTradeGate which pre-computes contracts via its own
            ExecutionProfile.

        Returns
        -------
        ExecutionSignal or None
            None if the trade is blocked (size_mult=0 or contracts=0).
        """
        # Determine symbol
        sym = symbol or getattr(live_signal, "symbol", "") or "MES"
        sym = self.config.symbol_map.get(sym, sym)

        # Determine side
        sig_type = getattr(live_signal, "signal_type", None)
        direction = getattr(live_signal, "direction", "")

        if sig_type is not None and hasattr(sig_type, "name"):
            if "LONG" in sig_type.name:
                side = ExecutionSide.BUY
            elif "SHORT" in sig_type.name:
                side = ExecutionSide.SELL
            else:
                logger.debug("Skipping non-entry signal: %s", sig_type)
                return None
        elif direction:
            side = ExecutionSide.BUY if direction.lower() == "long" else ExecutionSide.SELL
        else:
            logger.warning("Cannot determine side from signal: %s", live_signal)
            return None

        # Compute contracts (or use pre-computed value from PropTradeGate)
        if override_contracts is not None:
            contracts = override_contracts
            logger.debug("Using override_contracts=%d from PropTradeGate", contracts)
        else:
            contracts = compute_contracts(
                base_contracts=self.config.base_contracts,
                size_mult=size_mult,
                min_contracts=self.config.min_contracts,
                max_contracts=self.config.max_contracts,
            )
        if contracts == 0:
            logger.info("Trade blocked by sizing: size_mult=%.2f -> 0 contracts", size_mult)
            return None

        # Mode
        effective_mode = mode or self.config.default_mode

        # ATM template
        if not atm_template:
            if effective_mode == ExecutionMode.CHALLENGE:
                atm_template = self.config.challenge_atm_template
            else:
                atm_template = self.config.funded_atm_template

        # Build metadata
        metadata = {
            "ml_prob": getattr(live_signal, "ml_prob", 0.0),
            "percentile": getattr(live_signal, "percentile", 0.0),
            "quality_score": getattr(live_signal, "quality_score", 0.0),
            "position_size_raw": getattr(live_signal, "position_size", 1.0),
            "size_mult_applied": size_mult,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        signal = ExecutionSignal(
            signal_id=create_signal_id(),
            timestamp=getattr(live_signal, "timestamp", None) or __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            symbol=sym,
            side=side,
            contracts=contracts,
            mode=effective_mode,
            stop_loss=getattr(live_signal, "stop", 0.0),
            take_profit=getattr(live_signal, "take_profit", 0.0),
            reason=getattr(live_signal, "reason", ""),
            strategy_type=getattr(live_signal, "strategy_type", ""),
            confidence=getattr(live_signal, "ml_prob", 0.0) or getattr(live_signal, "quality_score", 0.0),
            position_size_mult=size_mult,
            entry_price=getattr(live_signal, "entry", 0.0),
            ttl_seconds=self.config.signal_ttl_seconds,
            atm_template=atm_template,
            metadata=metadata,
        )

        logger.info(
            "Converted LiveSignal -> ExecutionSignal: id=%s sym=%s side=%s contracts=%d mode=%s",
            signal.signal_id[:8],
            signal.symbol,
            signal.side.value,
            signal.contracts,
            signal.mode.value,
        )
        return signal
