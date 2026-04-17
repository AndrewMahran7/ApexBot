"""
Dashboard State Store
======================

Thread-safe in-memory state that collects PnL snapshots, trades,
signals, and alerts from the running pipeline.  The FastAPI app
reads from this store; the pipeline writes to it via callbacks.

No hidden state — everything is explicit, logged, and inspectable.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum items retained in ring buffers
MAX_EQUITY_POINTS = 50_000
MAX_TRADES = 5_000
MAX_SIGNALS = 5_000
MAX_ALERTS = 1_000


@dataclass
class Alert:
    """A dashboard alert (drawdown, risk trigger, etc.)."""
    timestamp: str
    level: str        # "warning", "critical"
    category: str     # "drawdown", "risk", "kill_switch"
    message: str


class DashboardState:
    """
    Thread-safe state store for the monitoring dashboard.

    Written to by pipeline callbacks (on_pnl, on_trade, on_signal,
    on_risk_event).  Read from by FastAPI endpoints.
    """

    def __init__(
        self,
        drawdown_warn: float = -300.0,
        drawdown_critical: float = -450.0,
    ) -> None:
        self._lock = threading.Lock()

        # Thresholds
        self._dd_warn = drawdown_warn
        self._dd_critical = drawdown_critical

        # --- PnL state ---
        self._equity: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._realized_pnl: float = 0.0
        self._drawdown: float = 0.0
        self._peak_equity: float = 0.0
        self._open_position_count: int = 0
        self._last_update: Optional[str] = None

        # --- Ring buffers ---
        self._equity_curve: deque[dict] = deque(maxlen=MAX_EQUITY_POINTS)
        self._trades: deque[dict] = deque(maxlen=MAX_TRADES)
        self._signals: deque[dict] = deque(maxlen=MAX_SIGNALS)
        self._alerts: deque[dict] = deque(maxlen=MAX_ALERTS)

        # --- Open positions (pos_id -> dict) ---
        self._open_positions: dict[str, dict] = {}

        # --- Risk state ---
        self._risk_killed: bool = False
        self._risk_daily_entries: int = 0
        self._risk_events_count: int = 0

        logger.info(
            "DashboardState initialised: dd_warn=%.0f, dd_critical=%.0f",
            drawdown_warn, drawdown_critical,
        )

    # ------------------------------------------------------------------
    # Write API (called by pipeline callbacks)
    # ------------------------------------------------------------------

    def on_pnl(self, update) -> None:
        """Receive a PnLUpdate from PaperEngine."""
        with self._lock:
            ts_str = str(update.timestamp)
            self._equity = update.equity
            self._unrealized_pnl = update.unrealized_pnl
            self._realized_pnl = update.realized_pnl
            self._drawdown = update.drawdown
            self._open_position_count = update.open_position_count
            self._peak_equity = max(self._peak_equity, update.equity)
            self._last_update = ts_str

            self._equity_curve.append({
                "timestamp": ts_str,
                "equity": round(update.equity, 2),
                "drawdown": round(update.drawdown, 2),
            })

            # Drawdown alerts
            if update.drawdown <= self._dd_critical:
                self._add_alert_locked(
                    ts_str, "critical", "drawdown",
                    f"Critical drawdown: ${update.drawdown:,.2f}",
                )
            elif update.drawdown <= self._dd_warn:
                self._add_alert_locked(
                    ts_str, "warning", "drawdown",
                    f"Drawdown warning: ${update.drawdown:,.2f}",
                )

        logger.debug("Dashboard PnL update: equity=%.2f dd=%.2f", update.equity, update.drawdown)

    def on_trade(self, trade) -> None:
        """Receive a completed Trade from PaperEngine."""
        with self._lock:
            self._trades.append({
                "entry_time": str(trade.entry_time),
                "exit_time": str(trade.exit_time),
                "direction": trade.direction,
                "entry_price": round(trade.entry_price, 2),
                "exit_price": round(trade.exit_price, 2),
                "net_pnl": round(trade.net_pnl, 2),
                "exit_reason": trade.exit_reason,
                "strategy_type": trade.strategy_type,
                "position_size": trade.position_size,
            })
        logger.debug("Dashboard trade: %s %s pnl=%.2f", trade.direction, trade.exit_reason, trade.net_pnl)

    def on_signal(self, signal) -> None:
        """Receive a LiveSignal from the strategy engine."""
        with self._lock:
            self._signals.append({
                "timestamp": str(signal.timestamp),
                "signal_type": signal.signal_type.name,
                "direction": signal.direction,
                "entry": round(signal.entry, 2),
                "stop": round(signal.stop, 2),
                "take_profit": round(signal.take_profit, 2),
                "position_size": signal.position_size,
                "strategy_type": signal.strategy_type,
            })

    def on_risk_event(self, event) -> None:
        """Receive a RiskEvent from the RiskManager."""
        with self._lock:
            self._risk_events_count += 1

            if event.event_type == "kill_switch":
                self._risk_killed = True
                self._add_alert_locked(
                    str(event.timestamp), "critical", "kill_switch",
                    f"Kill switch triggered: {event.reason}",
                )
            elif event.event_type == "blocked":
                self._add_alert_locked(
                    str(event.timestamp), "warning", "risk",
                    f"Trade blocked: {event.reason}",
                )
        logger.debug("Dashboard risk event: %s %s", event.event_type, event.reason)

    def update_risk_state(self, risk_manager) -> None:
        """Pull latest state from RiskManager."""
        with self._lock:
            self._risk_killed = risk_manager.killed
            self._risk_daily_entries = risk_manager.daily_entries

    def update_open_positions(self, paper_engine) -> None:
        """Pull latest open positions from PaperEngine."""
        with self._lock:
            raw = paper_engine.open_positions
            self._open_positions = {}
            for pos_id, pos in raw.items():
                self._open_positions[pos_id] = {
                    "position_id": pos_id,
                    "direction": pos.get("direction", ""),
                    "entry_price": round(pos.get("entry_price", 0.0), 2),
                    "entry_time": str(pos.get("entry_time", "")),
                    "position_size": pos.get("position_size", 0.0),
                    "strategy_type": pos.get("strategy_type", ""),
                }

    # ------------------------------------------------------------------
    # Read API (called by FastAPI endpoints)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Full dashboard snapshot as a JSON-serialisable dict."""
        with self._lock:
            return {
                "pnl": {
                    "equity": round(self._equity, 2),
                    "unrealized_pnl": round(self._unrealized_pnl, 2),
                    "realized_pnl": round(self._realized_pnl, 2),
                    "drawdown": round(self._drawdown, 2),
                    "peak_equity": round(self._peak_equity, 2),
                    "open_position_count": self._open_position_count,
                    "last_update": self._last_update,
                },
                "risk": {
                    "killed": self._risk_killed,
                    "daily_entries": self._risk_daily_entries,
                    "total_events": self._risk_events_count,
                },
                "open_positions": list(self._open_positions.values()),
                "recent_trades": list(self._trades)[-50:],
                "alerts": list(self._alerts)[-50:],
            }

    def equity_curve(self, last_n: int | None = None) -> list[dict]:
        """Return equity curve points (optionally last N)."""
        with self._lock:
            curve = list(self._equity_curve)
            if last_n is not None:
                curve = curve[-last_n:]
            return curve

    def trade_count(self) -> int:
        with self._lock:
            return len(self._trades)

    def alert_count(self) -> int:
        with self._lock:
            return len(self._alerts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_alert_locked(self, ts: str, level: str, category: str, message: str) -> None:
        """Add an alert (caller must hold self._lock)."""
        alert = {"timestamp": ts, "level": level, "category": category, "message": message}
        self._alerts.append(alert)
        logger.warning("ALERT [%s] %s: %s", level, category, message)
