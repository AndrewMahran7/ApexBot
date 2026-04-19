"""
Mode-aware prop risk integration for the OpenClaw execution layer.

Bridges strategy/prop_risk_layer.py into execution/ so that:
- Account mode (challenge/funded) drives sizing and gating
- Symbol-specific profiles select the right PropRiskConfig
- Trade gating produces EXECUTE / REDUCE / SKIP / STOP decisions
- Contract count is computed from fractional multipliers
- Every decision is captured with a full audit snapshot

This module does NOT contain risk logic — it delegates to PropRiskLayer.
It TRANSLATES risk decisions into execution-layer terms.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ── Gate decision enum ───────────────────────────────────────────────────────

class GateAction(str, Enum):
    EXECUTE = "execute"          # proceed at computed size
    REDUCE = "reduce"            # proceed at reduced size (sizing rules kicked in)
    SKIP = "skip"                # do not trade this signal
    STOP = "stop"                # halt all trading (account state critical)


# ── Mode-aware execution profile ─────────────────────────────────────────────

@dataclass
class ExecutionProfile:
    """
    Full execution configuration for one account-mode + symbol combination.

    Created by ExecutionProfileRegistry.get().
    """

    # Identity
    label: str                          # e.g. "challenge_mnq", "funded"
    mode: str                           # "challenge" or "funded"
    symbol: str                         # "MNQ", "MES", or "" for combined

    # Contract sizing
    base_contracts: int = 1             # contracts at 1.0x multiplier
    min_contracts: int = 1              # floor (0 only if blocked)
    max_contracts: int = 2              # hard cap per trade

    # ATM template
    atm_template: str = ""

    # Logging
    log_label: str = ""                 # human-readable label for logs

    def __post_init__(self) -> None:
        if not self.log_label:
            self.log_label = self.label


class ExecutionProfileRegistry:
    """
    Registry of named execution profiles.

    Provides mode + symbol -> ExecutionProfile lookup with factory
    methods for the standard Apex configurations.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ExecutionProfile] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the standard profiles based on validation results."""

        # ── Challenge profiles ───────────────────────────────────────────
        # Combined challenge (MNQ or MES): 0.35x flat optimal
        self._profiles["challenge_combined"] = ExecutionProfile(
            label="challenge_combined",
            mode="challenge",
            symbol="",
            base_contracts=1,
            min_contracts=1,
            max_contracts=2,
            atm_template="",
            log_label="Challenge (Combined 0.35x)",
        )

        # MNQ-specific challenge: 0.35x flat optimal
        self._profiles["challenge_mnq"] = ExecutionProfile(
            label="challenge_mnq",
            mode="challenge",
            symbol="MNQ",
            base_contracts=1,
            min_contracts=1,
            max_contracts=2,
            atm_template="",
            log_label="Challenge (MNQ 0.35x)",
        )

        # MES-specific challenge: 0.50x + DD proximity optimal
        self._profiles["challenge_mes"] = ExecutionProfile(
            label="challenge_mes",
            mode="challenge",
            symbol="MES",
            base_contracts=1,
            min_contracts=1,
            max_contracts=3,       # MES lower variance allows slightly more
            atm_template="",
            log_label="Challenge (MES 0.50x+DDprox)",
        )

        # ── Funded profiles ──────────────────────────────────────────────
        self._profiles["funded"] = ExecutionProfile(
            label="funded",
            mode="funded",
            symbol="",
            base_contracts=1,
            min_contracts=1,
            max_contracts=3,
            atm_template="",
            log_label="Funded (0.35x conservative)",
        )

    def register(self, name: str, profile: ExecutionProfile) -> None:
        """Register a custom profile."""
        self._profiles[name] = profile
        logger.info("Registered execution profile: %s", name)

    def get(self, name: str) -> Optional[ExecutionProfile]:
        """Look up a profile by name."""
        return self._profiles.get(name)

    def resolve(self, mode: str, symbol: str = "") -> ExecutionProfile:
        """
        Auto-resolve the best profile for a mode + symbol combination.

        Priority:
        1. Exact match: "{mode}_{symbol.lower()}"
        2. Mode + combined: "{mode}_combined"
        3. Mode alone: "{mode}"
        4. Fallback: challenge_combined
        """
        if symbol:
            key = f"{mode}_{symbol.lower()}"
            if key in self._profiles:
                return self._profiles[key]

        key = f"{mode}_combined"
        if key in self._profiles:
            return self._profiles[key]

        if mode in self._profiles:
            return self._profiles[mode]

        logger.warning("No profile for mode=%s symbol=%s, using challenge_combined", mode, symbol)
        return self._profiles["challenge_combined"]

    @property
    def available(self) -> list[str]:
        return list(self._profiles.keys())


# ── Gate decision output ─────────────────────────────────────────────────────

@dataclass
class GateDecision:
    """
    Complete decision output combining prop risk layer + execution profile.

    Contains everything needed to execute or skip a trade, plus full
    audit context.
    """

    action: GateAction
    contracts: int                      # 0 if skipped/stopped
    size_mult: float                    # raw multiplier from PropRiskLayer
    profile: ExecutionProfile           # which profile was used
    reasons: list[str]                  # human-readable explanations
    components: dict                    # sizing factor breakdown

    # Account state snapshot at decision time
    account_snapshot: dict = field(default_factory=dict)

    @property
    def should_execute(self) -> bool:
        return self.action in (GateAction.EXECUTE, GateAction.REDUCE)

    def to_audit_dict(self) -> dict:
        """Full audit record for this decision."""
        return {
            "action": self.action.value,
            "contracts": self.contracts,
            "size_mult": round(self.size_mult, 4),
            "profile_label": self.profile.label,
            "profile_mode": self.profile.mode,
            "profile_symbol": self.profile.symbol,
            "reasons": self.reasons,
            "components": self.components,
            "account_snapshot": self.account_snapshot,
        }


# ── Trade gate ───────────────────────────────────────────────────────────────

def compute_contracts_from_mult(
    size_mult: float,
    profile: ExecutionProfile,
) -> int:
    """
    Convert a fractional sizing multiplier to integer contracts.

    Parameters
    ----------
    size_mult : float
        Multiplier from PropRiskLayer (e.g. 0.35). 0 = blocked.
    profile : ExecutionProfile
        Contains base/min/max contracts for this mode+symbol.

    Returns
    -------
    int
        Contract count. 0 only if blocked.
    """
    if size_mult <= 0:
        return 0

    raw = profile.base_contracts * size_mult
    contracts = max(profile.min_contracts, min(profile.max_contracts, round(raw)))

    logger.debug(
        "compute_contracts: base=%d x mult=%.3f = %.3f -> %d (min=%d, max=%d) [%s]",
        profile.base_contracts, size_mult, raw, contracts,
        profile.min_contracts, profile.max_contracts, profile.label,
    )
    return contracts


def _snapshot_account_state(account_state) -> dict:
    """Serialize an AccountState to a JSON-safe dict."""
    return {
        "equity": round(getattr(account_state, "equity", 0.0), 2),
        "peak_equity": round(getattr(account_state, "peak_equity", 0.0), 2),
        "starting_capital": round(getattr(account_state, "starting_capital", 0.0), 2),
        "trailing_dd_limit": round(getattr(account_state, "trailing_dd_limit", 0.0), 2),
        "realized_pnl": round(getattr(account_state, "realized_pnl", 0.0), 2),
        "daily_pnl": round(getattr(account_state, "daily_pnl", 0.0), 2),
        "daily_trades": getattr(account_state, "daily_trades", 0),
        "consecutive_losses": getattr(account_state, "consecutive_losses", 0),
        "mode": str(getattr(account_state, "mode", "")),
        "distance_to_dd": round(getattr(account_state, "distance_to_dd", 0.0), 2),
        "progress": round(getattr(account_state, "progress", 0.0), 4),
    }


class PropTradeGate:
    """
    Trade gating layer that combines PropRiskLayer + ExecutionProfile.

    Usage:
        gate = PropTradeGate(prop_layer, registry)
        decision = gate.evaluate(symbol="MNQ")
        if decision.should_execute:
            controller.on_signal(signal, decision=decision)

    The gate does NOT modify the PropRiskLayer — it only reads from it.
    record_trade() must be called externally after confirmed execution.
    """

    def __init__(
        self,
        prop_layer,                                # PropRiskLayer instance
        profile_registry: Optional[ExecutionProfileRegistry] = None,
        default_profile_name: str = "",            # override auto-resolve
    ) -> None:
        self._layer = prop_layer
        self._registry = profile_registry or ExecutionProfileRegistry()
        self._default_profile_name = default_profile_name
        logger.info(
            "PropTradeGate initialized: mode=%s, profiles=%s",
            getattr(getattr(prop_layer, "cfg", None), "mode", "unknown"),
            self._registry.available,
        )

    def evaluate(self, symbol: str = "") -> GateDecision:
        """
        Evaluate whether the next trade should be taken.

        Parameters
        ----------
        symbol : str
            Instrument symbol (e.g. "MNQ", "MES").

        Returns
        -------
        GateDecision
            Contains action, contracts, reasons, and full audit context.
        """
        # Get the account state snapshot
        state = self._layer.state

        # Resolve execution profile
        mode_str = state.mode.value if hasattr(state.mode, "value") else str(state.mode)
        if self._default_profile_name:
            profile = self._registry.get(self._default_profile_name)
            if profile is None:
                profile = self._registry.resolve(mode_str, symbol)
        else:
            profile = self._registry.resolve(mode_str, symbol)

        # Ask PropRiskLayer for sizing decision
        trade_decision = self._layer.evaluate_trade(symbol=symbol)

        # Snapshot account state
        snapshot = _snapshot_account_state(state)

        # Determine gate action
        if trade_decision.blocked:
            # Distinguish STOP vs SKIP
            reasons_lower = " ".join(trade_decision.reasons).lower()
            if any(kw in reasons_lower for kw in ("failed", "halted", "passed", "dd breached")):
                action = GateAction.STOP
            else:
                action = GateAction.SKIP

            decision = GateDecision(
                action=action,
                contracts=0,
                size_mult=trade_decision.size_mult,
                profile=profile,
                reasons=trade_decision.reasons,
                components=trade_decision.components,
                account_snapshot=snapshot,
            )
            logger.info(
                "Trade gate: %s [%s] symbol=%s reasons=%s",
                action.value, profile.log_label, symbol, trade_decision.reasons,
            )
            return decision

        # Trade allowed — compute contracts
        contracts = compute_contracts_from_mult(trade_decision.size_mult, profile)
        if contracts == 0:
            decision = GateDecision(
                action=GateAction.SKIP,
                contracts=0,
                size_mult=trade_decision.size_mult,
                profile=profile,
                reasons=["sizing resulted in 0 contracts"],
                components=trade_decision.components,
                account_snapshot=snapshot,
            )
            logger.info("Trade gate: SKIP (0 contracts) [%s] symbol=%s", profile.log_label, symbol)
            return decision

        # Was the size reduced from base?
        base_size = getattr(self._layer.cfg, "base_size", 1.0)
        if trade_decision.size_mult < base_size - 0.001:
            action = GateAction.REDUCE
        else:
            action = GateAction.EXECUTE

        decision = GateDecision(
            action=action,
            contracts=contracts,
            size_mult=trade_decision.size_mult,
            profile=profile,
            reasons=trade_decision.reasons,
            components=trade_decision.components,
            account_snapshot=snapshot,
        )

        logger.info(
            "Trade gate: %s [%s] symbol=%s mult=%.3f -> %d contracts, reasons=%s",
            action.value, profile.log_label, symbol,
            trade_decision.size_mult, contracts, trade_decision.reasons,
        )
        return decision

    def record_trade(self, net_pnl: float, symbol: str = "") -> None:
        """Record a completed trade in the PropRiskLayer."""
        self._layer.record_trade(net_pnl, symbol=symbol)

    def reset_day(self) -> None:
        """Reset daily counters."""
        self._layer.reset_day()

    @property
    def account_state(self):
        """Current account state from the prop layer."""
        return self._layer.state

    @property
    def is_active(self) -> bool:
        """Whether the account is still actively trading."""
        return getattr(self._layer, "active", True)

    @property
    def profile_registry(self) -> ExecutionProfileRegistry:
        return self._registry
