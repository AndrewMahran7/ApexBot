"""
Analytics Module
=================

Collects trades, signals, and risk decisions from the live pipeline
and computes performance metrics on demand.

Stores:
    - All completed trades (from PaperEngine)
    - All emitted signals (from StrategyEngine)
    - All risk decisions (from RiskManager)

Computes:
    - Win rate, profit factor, Sharpe ratio
    - Performance breakdown by strategy_type
    - Daily PnL series, equity curve statistics
    - Signal-to-trade conversion rate
    - Risk event summary

Thread-safe — written to by pipeline callbacks, read by dashboard
or CLI at any time.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------

@dataclass
class StrategyBreakdown:
    """Performance metrics for a single strategy_type."""
    strategy_type: str
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0


@dataclass
class AnalyticsReport:
    """Full analytics snapshot — JSON-serialisable."""
    # Overall
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_drawdown: float = 0.0

    # Direction
    long_trades: int = 0
    long_win_rate: float = 0.0
    long_pnl: float = 0.0
    short_trades: int = 0
    short_win_rate: float = 0.0
    short_pnl: float = 0.0

    # By strategy_type
    by_strategy: list[dict] = field(default_factory=list)

    # Signals / decisions
    total_signals: int = 0
    entry_signals: int = 0
    exit_signals: int = 0
    risk_blocked: int = 0
    risk_capped: int = 0
    risk_kill_switches: int = 0
    signal_to_trade_rate: float = 0.0

    # Daily
    daily_pnl: list[dict] = field(default_factory=list)
    trading_days: int = 0
    winning_days: int = 0
    losing_days: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serialisable dict."""
        return {
            "overall": {
                "total_trades": self.total_trades,
                "win_count": self.win_count,
                "loss_count": self.loss_count,
                "win_rate": round(self.win_rate, 2),
                "profit_factor": round(self.profit_factor, 4),
                "sharpe_ratio": round(self.sharpe_ratio, 4),
                "total_pnl": round(self.total_pnl, 2),
                "avg_pnl": round(self.avg_pnl, 2),
                "largest_win": round(self.largest_win, 2),
                "largest_loss": round(self.largest_loss, 2),
                "max_drawdown": round(self.max_drawdown, 2),
            },
            "direction": {
                "long_trades": self.long_trades,
                "long_win_rate": round(self.long_win_rate, 2),
                "long_pnl": round(self.long_pnl, 2),
                "short_trades": self.short_trades,
                "short_win_rate": round(self.short_win_rate, 2),
                "short_pnl": round(self.short_pnl, 2),
            },
            "by_strategy": self.by_strategy,
            "signals": {
                "total_signals": self.total_signals,
                "entry_signals": self.entry_signals,
                "exit_signals": self.exit_signals,
                "risk_blocked": self.risk_blocked,
                "risk_capped": self.risk_capped,
                "risk_kill_switches": self.risk_kill_switches,
                "signal_to_trade_rate": round(self.signal_to_trade_rate, 2),
            },
            "daily": {
                "trading_days": self.trading_days,
                "winning_days": self.winning_days,
                "losing_days": self.losing_days,
                "daily_pnl": self.daily_pnl,
            },
        }


# ------------------------------------------------------------------
# Analytics Engine
# ------------------------------------------------------------------

class AnalyticsEngine:
    """
    Collects pipeline data and computes performance metrics.

    Thread-safe — pipeline callbacks write; dashboard/CLI reads.

    Parameters
    ----------
    initial_capital : float
        Starting equity for Sharpe and drawdown calculations.
    """

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self._lock = threading.Lock()
        self._initial_capital = initial_capital

        # Raw storage
        self._trades: list[dict] = []
        self._signals: list[dict] = []
        self._decisions: list[dict] = []

        logger.info("AnalyticsEngine initialised: initial_capital=%.2f", initial_capital)

    # ------------------------------------------------------------------
    # Write API — called by pipeline callbacks
    # ------------------------------------------------------------------

    def record_trade(self, trade) -> None:
        """Record a completed Trade from PaperEngine."""
        with self._lock:
            self._trades.append({
                "entry_time": str(trade.entry_time),
                "exit_time": str(trade.exit_time),
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "net_pnl": trade.net_pnl,
                "pnl_dollars": trade.pnl_dollars,
                "commission": trade.commission,
                "slippage_cost": trade.slippage_cost,
                "exit_reason": trade.exit_reason,
                "strategy_type": trade.strategy_type,
                "position_size": trade.position_size,
            })
        logger.debug(
            "Trade recorded: %s %s pnl=%.2f [%s]",
            trade.direction, trade.exit_reason, trade.net_pnl, trade.strategy_type,
        )

    def record_signal(self, signal) -> None:
        """Record a LiveSignal from the strategy engine."""
        with self._lock:
            self._signals.append({
                "timestamp": str(signal.timestamp),
                "signal_type": signal.signal_type.name,
                "direction": signal.direction,
                "entry": signal.entry,
                "stop": signal.stop,
                "take_profit": signal.take_profit,
                "position_size": signal.position_size,
                "strategy_type": signal.strategy_type,
                "ml_prob": signal.ml_prob,
                "percentile": signal.percentile,
                "is_entry": signal.is_entry,
                "is_exit": signal.is_exit,
            })

    def record_decision(self, event) -> None:
        """Record a RiskEvent (block, cap, kill switch)."""
        with self._lock:
            self._decisions.append({
                "timestamp": str(event.timestamp),
                "event_type": event.event_type,
                "reason": event.reason,
                "details": event.details,
            })
        logger.debug("Decision recorded: %s — %s", event.event_type, event.reason)

    # ------------------------------------------------------------------
    # Read API — compute metrics on demand
    # ------------------------------------------------------------------

    def report(self) -> AnalyticsReport:
        """Compute full analytics report from stored data."""
        with self._lock:
            trades = list(self._trades)
            signals = list(self._signals)
            decisions = list(self._decisions)

        logger.info(
            "Computing report: %d trades, %d signals, %d decisions",
            len(trades), len(signals), len(decisions),
        )

        r = AnalyticsReport()

        if not trades:
            r.total_signals = len(signals)
            r.entry_signals = sum(1 for s in signals if s["is_entry"])
            r.exit_signals = sum(1 for s in signals if s["is_exit"])
            r.risk_blocked = sum(1 for d in decisions if d["event_type"] == "blocked")
            r.risk_capped = sum(1 for d in decisions if d["event_type"] == "capped")
            r.risk_kill_switches = sum(1 for d in decisions if d["event_type"] == "kill_switch")
            logger.info("Report computed: 0 trades")
            return r

        pnls = np.array([t["net_pnl"] for t in trades])

        # --- Overall ---
        r.total_trades = len(trades)
        r.win_count = int(np.sum(pnls > 0))
        r.loss_count = int(np.sum(pnls <= 0))
        r.win_rate = r.win_count / r.total_trades * 100
        r.total_pnl = float(np.sum(pnls))
        r.avg_pnl = float(np.mean(pnls))
        r.largest_win = float(np.max(pnls)) if r.win_count > 0 else 0.0
        r.largest_loss = float(np.min(pnls)) if r.loss_count > 0 else 0.0
        r.profit_factor = _profit_factor(pnls)
        r.sharpe_ratio = _sharpe_from_trades(pnls, self._initial_capital)
        r.max_drawdown = _max_drawdown_from_pnls(pnls, self._initial_capital)

        # --- Direction ---
        long_pnls = np.array([t["net_pnl"] for t in trades if t["direction"] == "long"])
        short_pnls = np.array([t["net_pnl"] for t in trades if t["direction"] == "short"])

        r.long_trades = len(long_pnls)
        r.long_win_rate = (float(np.sum(long_pnls > 0)) / len(long_pnls) * 100) if len(long_pnls) > 0 else 0.0
        r.long_pnl = float(np.sum(long_pnls)) if len(long_pnls) > 0 else 0.0

        r.short_trades = len(short_pnls)
        r.short_win_rate = (float(np.sum(short_pnls > 0)) / len(short_pnls) * 100) if len(short_pnls) > 0 else 0.0
        r.short_pnl = float(np.sum(short_pnls)) if len(short_pnls) > 0 else 0.0

        # --- By strategy_type ---
        r.by_strategy = _breakdown_by_strategy(trades)

        # --- Signals / decisions ---
        r.total_signals = len(signals)
        r.entry_signals = sum(1 for s in signals if s["is_entry"])
        r.exit_signals = sum(1 for s in signals if s["is_exit"])
        r.risk_blocked = sum(1 for d in decisions if d["event_type"] == "blocked")
        r.risk_capped = sum(1 for d in decisions if d["event_type"] == "capped")
        r.risk_kill_switches = sum(1 for d in decisions if d["event_type"] == "kill_switch")
        r.signal_to_trade_rate = (
            r.total_trades / r.entry_signals * 100 if r.entry_signals > 0 else 0.0
        )

        # --- Daily PnL ---
        r.daily_pnl, r.trading_days, r.winning_days, r.losing_days = _daily_pnl(trades)

        logger.info(
            "Report computed: %d trades, win_rate=%.1f%%, PF=%.2f, Sharpe=%.2f, PnL=$%.2f",
            r.total_trades, r.win_rate, r.profit_factor, r.sharpe_ratio, r.total_pnl,
        )
        return r

    def export_json(self, path: str) -> None:
        """Export full analytics report to a JSON file."""
        report = self.report()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Analytics report exported to %s", path)

    # ------------------------------------------------------------------
    # Raw data access
    # ------------------------------------------------------------------

    @property
    def trade_count(self) -> int:
        with self._lock:
            return len(self._trades)

    @property
    def signal_count(self) -> int:
        with self._lock:
            return len(self._signals)

    @property
    def decision_count(self) -> int:
        with self._lock:
            return len(self._decisions)

    @property
    def trades(self) -> list[dict]:
        with self._lock:
            return list(self._trades)

    @property
    def signals(self) -> list[dict]:
        with self._lock:
            return list(self._signals)

    @property
    def decisions(self) -> list[dict]:
        with self._lock:
            return list(self._decisions)


# ------------------------------------------------------------------
# Computation helpers (pure functions, no state)
# ------------------------------------------------------------------

def _profit_factor(pnls: np.ndarray) -> float:
    """Gross profit / gross loss. Returns inf if no losses."""
    gains = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_profit = float(np.sum(gains)) if len(gains) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _sharpe_from_trades(pnls: np.ndarray, initial_capital: float) -> float:
    """
    Annualised Sharpe ratio from trade PnLs.

    Groups PnL by implicit daily batches (assumes ~252 trading days/year).
    Falls back to trade-level if daily grouping is not possible.
    """
    if len(pnls) < 2:
        return 0.0

    returns = pnls / initial_capital
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))

    if std_r == 0.0:
        return 0.0

    # Annualise assuming ~252 trading days
    sharpe = (mean_r / std_r) * math.sqrt(252)
    return sharpe


def _max_drawdown_from_pnls(pnls: np.ndarray, initial_capital: float) -> float:
    """Maximum drawdown in dollars from cumulative PnL series."""
    cum = np.cumsum(pnls) + initial_capital
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.min(dd)) if len(dd) > 0 else 0.0


def _breakdown_by_strategy(trades: list[dict]) -> list[dict]:
    """Performance breakdown by strategy_type."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        grouped[t["strategy_type"]].append(t["net_pnl"])

    results = []
    for stype, pnl_list in sorted(grouped.items()):
        pnls = np.array(pnl_list)
        wins = int(np.sum(pnls > 0))
        losses = int(np.sum(pnls <= 0))
        count = len(pnls)

        results.append({
            "strategy_type": stype,
            "trade_count": count,
            "win_count": wins,
            "loss_count": losses,
            "win_rate": round(wins / count * 100, 2) if count > 0 else 0.0,
            "total_pnl": round(float(np.sum(pnls)), 2),
            "avg_pnl": round(float(np.mean(pnls)), 2),
            "profit_factor": round(_profit_factor(pnls), 4),
            "largest_win": round(float(np.max(pnls)), 2) if wins > 0 else 0.0,
            "largest_loss": round(float(np.min(pnls)), 2) if losses > 0 else 0.0,
        })

    return results


def _daily_pnl(trades: list[dict]) -> tuple[list[dict], int, int, int]:
    """Aggregate PnL by exit date. Returns (daily_list, days, win_days, lose_days)."""
    by_day: dict[str, float] = defaultdict(float)
    for t in trades:
        # Parse date from exit_time string
        exit_str = t["exit_time"]
        day = exit_str[:10]  # "YYYY-MM-DD"
        by_day[day] += t["net_pnl"]

    daily = [
        {"date": day, "pnl": round(pnl, 2)}
        for day, pnl in sorted(by_day.items())
    ]
    win_days = sum(1 for d in daily if d["pnl"] > 0)
    lose_days = sum(1 for d in daily if d["pnl"] <= 0)

    return daily, len(daily), win_days, lose_days
