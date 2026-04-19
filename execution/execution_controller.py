"""
Execution controller — orchestrates the full signal-to-trade pipeline.

Signal flow (with PropTradeGate):
    LiveSignal → PropTradeGate.evaluate(symbol)
        → GateDecision (EXECUTE / REDUCE / SKIP / STOP)
        → RiskBridge.convert(signal, size_mult, contracts)
        → ExecutionSignal
        → SignalRegistry (dedup)
        → FailSafe checks
        → PreTradeValidator (UI state)
        → OpenClawDriver (click)
        → PostTradeValidator (confirmation)
        → AuditLogger (record everything)

The controller processes ONE signal at a time and never queues.
"""

from __future__ import annotations

import datetime
import logging
import threading
from typing import Callable, Optional

from execution.adapter import ExecutionAdapter, FillResult, OrderResult
from execution.audit_logger import AuditLogger, EventType
from execution.fail_safes import FailSafeConfig, FailSafeState, RunMode
from execution.openclaw_adapter import OpenClawAdapter
from execution.prop_sizing import GateAction, GateDecision, PropTradeGate
from execution.risk_bridge import RiskBridge, RiskBridgeConfig
from execution.signal_schema import (
    ExecutionMode,
    ExecutionSignal,
    SignalRegistry,
)
from execution.validators import (
    PositionMonitor,
    PostTradeValidator,
    PreTradeValidator,
    UIState,
)

logger = logging.getLogger(__name__)


class ExecutionController:
    """
    Central orchestrator for the OpenClaw execution layer.

    Receives LiveSignals from the strategy pipeline, converts them
    to ExecutionSignals, validates, and optionally clicks.

    Thread-safe: uses a lock to ensure one signal at a time.

    Parameters
    ----------
    fail_safe_config : FailSafeConfig
        Controls run mode, kill switch, cooldown, etc.
    risk_bridge_config : RiskBridgeConfig
        Sizing translation config.
    adapter : ExecutionAdapter, optional
        Execution backend.  Defaults to OpenClawAdapter if not provided.
        Swap this to use a different broker backend.
    prop_gate : PropTradeGate, optional
        If provided, on_signal_gated() uses it for sizing/gating.
    expected_account : str
        Substring to match in Tradovate account label.
    allowed_symbols : set of str
        Symbols allowed for trading.
    output_dir : str
        Directory for audit logs and screenshots.
    on_alert : callable, optional
        Called with (severity: str, message: str) for critical events.
        Wire to Telegram or similar.
    """

    def __init__(
        self,
        fail_safe_config: Optional[FailSafeConfig] = None,
        risk_bridge_config: Optional[RiskBridgeConfig] = None,
        adapter: Optional[ExecutionAdapter] = None,
        prop_gate: Optional[PropTradeGate] = None,
        expected_account: str = "",
        allowed_symbols: Optional[set[str]] = None,
        output_dir: str = "results/openclaw_execution",
        on_alert: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        # Config
        self._fs_config = fail_safe_config or FailSafeConfig()
        self._rb_config = risk_bridge_config or RiskBridgeConfig()

        # Execution adapter — defaults to OpenClaw if not provided
        is_dry_run = self._fs_config.run_mode == RunMode.DRY_RUN
        self._adapter: ExecutionAdapter = adapter or OpenClawAdapter(dry_run=is_dry_run)

        # Components
        self._fail_safe = FailSafeState(self._fs_config)
        self._registry = SignalRegistry(
            fingerprint_cooldown_seconds=self._fs_config.duplicate_cooldown_seconds
        )
        self._bridge = RiskBridge(self._rb_config)
        self._pre_validator = PreTradeValidator(
            expected_account_fragment=expected_account,
            allowed_symbols=allowed_symbols,
        )
        self._post_validator = PostTradeValidator()
        self._position_monitor = PositionMonitor(
            expected_max_positions=self._fs_config.max_open_trades
        )
        self._audit = AuditLogger(
            output_dir=output_dir,
            screenshot_fn=self._adapter.take_screenshot,
        )
        self._on_alert = on_alert
        self._prop_gate = prop_gate

        # Thread safety
        self._lock = threading.Lock()
        self._processing = False

        logger.info(
            "ExecutionController initialized: mode=%s, adapter=%s, account=%s, symbols=%s, prop_gate=%s",
            self._fs_config.run_mode.value,
            self._adapter.name,
            expected_account or "(any)",
            allowed_symbols or "(default)",
            "yes" if prop_gate else "no",
        )

    # ── Primary entry point ──────────────────────────────────────────────────

    def on_signal(
        self,
        live_signal,
        size_mult: float = 1.0,
        symbol: str = "",
        mode: Optional[ExecutionMode] = None,
        _gate_decision: Optional[GateDecision] = None,
    ) -> bool:
        """
        Process a LiveSignal from the strategy pipeline.

        Parameters
        ----------
        live_signal
            LiveSignal from StrategyEngine -> RiskManager.
        size_mult : float
            Sizing multiplier from PropRiskLayer.evaluate_trade().
        symbol : str
            Instrument symbol override.
        mode : ExecutionMode, optional
            Account mode override.
        _gate_decision : GateDecision, optional
            Internal: passed by on_signal_gated() with full gate context.

        Returns
        -------
        bool
            True if execution was attempted (or would be in dry-run).
            False if blocked at any stage.
        """
        with self._lock:
            if self._processing:
                logger.warning("Already processing a signal — rejecting concurrent call")
                return False
            self._processing = True

        try:
            return self._process_signal(live_signal, size_mult, symbol, mode, _gate_decision)
        finally:
            with self._lock:
                self._processing = False

    def _process_signal(
        self,
        live_signal,
        size_mult: float,
        symbol: str,
        mode: Optional[ExecutionMode],
        gate_decision: Optional[GateDecision] = None,
    ) -> bool:
        now = datetime.datetime.now(datetime.timezone.utc)

        # ── Step 1: Convert to ExecutionSignal ───────────────────────────────
        override_contracts = gate_decision.contracts if gate_decision is not None else None
        exec_signal = self._bridge.convert(
            live_signal, size_mult=size_mult, symbol=symbol, mode=mode,
            override_contracts=override_contracts,
        )
        if exec_signal is None:
            self._audit.log(
                EventType.SIGNAL_RECEIVED,
                reason="blocked by sizing (0 contracts)",
                details={"size_mult": size_mult},
            )
            return False

        # Merge gate decision audit data into signal details
        signal_details = exec_signal.to_dict()
        if gate_decision is not None:
            signal_details["gate_decision"] = gate_decision.to_audit_dict()

        self._audit.log_from_signal(
            EventType.SIGNAL_RECEIVED, exec_signal,
            reason=exec_signal.reason,
            details=signal_details,
        )

        # ── Step 2: Check expiration ─────────────────────────────────────────
        if exec_signal.is_expired(now):
            self._audit.log_from_signal(
                EventType.SIGNAL_EXPIRED, exec_signal,
                reason=f"signal expired at {exec_signal.expires_at.isoformat()}",
            )
            return False

        # ── Step 3: Check duplicates ─────────────────────────────────────────
        is_dup, dup_reason = self._registry.is_duplicate(exec_signal)
        if is_dup:
            self._audit.log_from_signal(
                EventType.SIGNAL_DUPLICATE, exec_signal,
                reason=dup_reason,
            )
            return False

        # ── Step 4: Fail-safe checks ─────────────────────────────────────────
        can_exec, fs_reason = self._fail_safe.can_execute(now)

        # In dry-run mode, we continue through validation but don't click
        is_dry_run = self._fail_safe.is_dry_run
        if not can_exec and not is_dry_run:
            event_type = (
                EventType.KILL_SWITCH_ACTIVATED
                if self._fail_safe.kill_switch_active
                else EventType.FAIL_SAFE_TRIGGERED
                if "emergency" in fs_reason
                else EventType.COOLDOWN_ACTIVE
                if "cooldown" in fs_reason
                else EventType.FAIL_SAFE_TRIGGERED
            )
            self._audit.log_from_signal(event_type, exec_signal, reason=fs_reason)
            return False

        # ── Step 5: Read execution context ─────────────────────────────────
        ui_state = self._adapter.validate_context(exec_signal)

        # ── Step 6: Pre-trade validation ─────────────────────────────────────
        pre_result = self._pre_validator.validate(exec_signal, ui_state, now)

        if not pre_result.passed:
            self._audit.log_from_signal(
                EventType.VALIDATION_FAILED, exec_signal,
                reason=pre_result.summary,
                details=pre_result.to_dict(),
            )
            if self._fs_config.screenshot_on_failure:
                self._audit.save_screenshot(f"pre_validation_fail_{exec_signal.signal_id[:8]}")
            self._alert("WARNING", f"Pre-trade validation failed: {pre_result.summary}")
            return False

        self._audit.log_from_signal(
            EventType.VALIDATION_PASSED, exec_signal,
            reason="pre-trade checks passed",
            details=pre_result.to_dict(),
        )

        # ── Step 7: Execute or dry-run ───────────────────────────────────────
        if is_dry_run:
            self._audit.log_from_signal(
                EventType.DRY_RUN, exec_signal,
                reason=f"DRY RUN — would {exec_signal.side.value} {exec_signal.contracts}x {exec_signal.symbol}",
                details={
                    "fail_safe_reason": fs_reason,
                    "ui_state": {
                        "window": ui_state.window_found,
                        "symbol": ui_state.active_symbol,
                        "quantity": ui_state.quantity_value,
                        "atm": ui_state.atm_template_name,
                    },
                },
            )
            self._registry.register(exec_signal)
            return True

        # ── Step 7b: Actual execution via adapter ────────────────────────────
        self._audit.log_from_signal(
            EventType.EXECUTION_ATTEMPTED, exec_signal,
            reason=f"placing {exec_signal.side.value} {exec_signal.contracts}x {exec_signal.symbol} via {self._adapter.name}",
        )

        order_result = self._adapter.place_order(exec_signal)
        if not order_result.success:
            self._audit.log_from_signal(
                EventType.EXECUTION_REJECTED, exec_signal,
                reason=f"order failed: {order_result.detail}",
                details={"action": order_result.action, "order_id": order_result.order_id},
            )
            self._audit.save_screenshot(f"order_fail_{exec_signal.signal_id[:8]}")
            self._alert("ERROR", f"Order failed: {order_result.detail}")
            return False

        # ── Step 8: Post-trade confirmation via adapter ──────────────────────
        fill_result = self._adapter.confirm_fill(
            exec_signal,
            timeout_ms=self._fs_config.confirmation_timeout_seconds * 1000,
        )
        post_state = fill_result.post_state
        post_result = fill_result.validation

        if not post_result.passed:
            self._audit.log_from_signal(
                EventType.POST_TRADE_MISMATCH, exec_signal,
                reason=post_result.summary,
                details=post_result.to_dict(),
            )
            self._audit.save_screenshot(f"post_validation_fail_{exec_signal.signal_id[:8]}")
            self._alert("CRITICAL", f"Post-trade mismatch: {post_result.summary}")

            # Check for anomalies
            anomalies = self._position_monitor.update(
                exec_signal.symbol,
                post_state.position_side,
                post_state.position_size,
            )
            if anomalies:
                self._fail_safe.activate_kill_switch(
                    f"Position anomaly: {'; '.join(anomalies)}"
                )
                self._audit.log_from_signal(
                    EventType.KILL_SWITCH_ACTIVATED, exec_signal,
                    reason=f"position anomaly detected: {anomalies}",
                )
                self._alert("CRITICAL", f"KILL SWITCH — position anomaly: {anomalies}")
            return False

        # ── Step 9: Success ──────────────────────────────────────────────────
        self._audit.log_from_signal(
            EventType.EXECUTION_CONFIRMED, exec_signal,
            reason=f"confirmed {exec_signal.side.value} {exec_signal.contracts}x {exec_signal.symbol}",
            details={
                "fill_price": post_state.fill_price,
                "elapsed_ms": post_state.time_elapsed_ms,
                **post_result.to_dict(),
            },
        )

        self._fail_safe.record_execution(now)
        self._registry.register(exec_signal)
        self._position_monitor.update(
            exec_signal.symbol, post_state.position_side, post_state.position_size
        )

        logger.info(
            "EXECUTION CONFIRMED: %s %dx %s (fill=%s, elapsed=%dms)",
            exec_signal.side.value,
            exec_signal.contracts,
            exec_signal.symbol,
            post_state.fill_price,
            post_state.time_elapsed_ms,
        )
        return True

    # ── Gated entry point (uses PropTradeGate) ────────────────────────────

    def on_signal_gated(
        self,
        live_signal,
        symbol: str = "",
    ) -> bool:
        """
        Process a LiveSignal using the integrated PropTradeGate.

        The gate determines whether to trade, at what size, and in which
        mode.  This is the preferred entry point when a PropRiskLayer is
        configured.

        Parameters
        ----------
        live_signal
            LiveSignal from StrategyEngine -> RiskManager.
        symbol : str
            Instrument symbol (e.g. "MNQ", "MES").

        Returns
        -------
        bool
            True if execution was attempted (or would be in dry-run).
        """
        if self._prop_gate is None:
            logger.error("on_signal_gated called but no PropTradeGate configured")
            return False

        # Evaluate the trade gate
        decision = self._prop_gate.evaluate(symbol=symbol)

        # Log the gate decision regardless of outcome
        self._audit.log(
            EventType.SIGNAL_RECEIVED,
            symbol=symbol,
            reason=f"gate={decision.action.value}, mult={decision.size_mult:.3f}, "
                   f"contracts={decision.contracts}, profile={decision.profile.label}",
            details=decision.to_audit_dict(),
        )

        # Check if the gate says STOP
        if decision.action == GateAction.STOP:
            self._audit.log(
                EventType.FAIL_SAFE_TRIGGERED,
                symbol=symbol,
                reason=f"STOP: {', '.join(decision.reasons)}",
                details=decision.to_audit_dict(),
            )
            self._alert("CRITICAL", f"Trade gate STOP: {decision.reasons}")
            return False

        # Check if the gate says SKIP
        if decision.action == GateAction.SKIP:
            self._audit.log(
                EventType.SIGNAL_EXPIRED,  # reuse — semantically "skipped"
                symbol=symbol,
                reason=f"SKIP: {', '.join(decision.reasons)}",
                details=decision.to_audit_dict(),
            )
            return False

        # EXECUTE or REDUCE — proceed with the computed contracts
        mode = (
            ExecutionMode.FUNDED
            if decision.profile.mode == "funded"
            else ExecutionMode.CHALLENGE
        )

        return self.on_signal(
            live_signal,
            size_mult=decision.size_mult,
            symbol=symbol,
            mode=mode,
            _gate_decision=decision,
        )

    # ── Control methods ──────────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str) -> None:
        """Emergency stop."""
        self._fail_safe.activate_kill_switch(reason)
        self._audit.log(
            EventType.KILL_SWITCH_ACTIVATED,
            reason=reason,
        )
        self._alert("CRITICAL", f"Kill switch activated: {reason}")

    def reset_kill_switch(self) -> None:
        """Resume trading after operator review."""
        self._fail_safe.reset_kill_switch()
        self._audit.log(EventType.FAIL_SAFE_TRIGGERED, reason="kill switch reset by operator")

    def set_run_mode(self, mode: RunMode) -> None:
        """Change run mode."""
        self._fail_safe.set_run_mode(mode)
        self._adapter.dry_run = (mode == RunMode.DRY_RUN)

    def record_trade_closed(self) -> None:
        """Notify that a trade was closed externally."""
        self._fail_safe.record_trade_closed()

    def reset_session(self) -> None:
        """Reset for new trading day."""
        self._fail_safe.reset_session()
        self._registry.clear()
        self._position_monitor.clear()
        self._audit.log(EventType.FAIL_SAFE_TRIGGERED, reason="session reset")

    # ── Alert helper ─────────────────────────────────────────────────────────

    def _alert(self, severity: str, message: str) -> None:
        """Send an alert through the registered callback."""
        logger.log(
            logging.CRITICAL if severity == "CRITICAL" else logging.WARNING,
            "[ALERT %s] %s", severity, message,
        )
        if self._on_alert:
            try:
                self._on_alert(severity, message)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

    # ── Status / inspection ──────────────────────────────────────────────────

    def status(self) -> dict:
        """Full status snapshot."""
        s = {
            "fail_safe": self._fail_safe.status(),
            "registry_seen": self._registry.seen_count,
            "positions": self._position_monitor.open_positions,
            "audit_events": self._audit.event_count,
            "audit_summary": self._audit.summary(),
        }
        if self._prop_gate is not None:
            from execution.prop_sizing import _snapshot_account_state
            s["account_state"] = _snapshot_account_state(self._prop_gate.account_state)
            s["account_active"] = self._prop_gate.is_active
        return s

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit

    @property
    def fail_safe(self) -> FailSafeState:
        return self._fail_safe

    @property
    def adapter(self) -> ExecutionAdapter:
        return self._adapter

    @property
    def driver(self):
        """Backward-compat: returns underlying driver if adapter exposes one."""
        return getattr(self._adapter, "driver", self._adapter)

    @property
    def prop_gate(self) -> Optional[PropTradeGate]:
        return self._prop_gate
