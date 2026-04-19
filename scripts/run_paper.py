#!/usr/bin/env python3
"""
Live Paper Trading Runner
==========================

Steps through historical bars simulating a real-time feed, executing the
full pipeline: StrategyEngine ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ RiskManager ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ PaperEngine.

Outputs:
    logs/paper_trading.log   ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â formatted real-time trade/signal log
    results/paper_trades.csv ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â all executed paper trades

Usage:
    python run_paper_live.py --data data/mes_4y.csv --days 2
    python run_paper_live.py --data data/mes_4y.csv --days 2 --bar-delay 0.05
    python run_paper_live.py --data data/mes_4y.csv --days 5 --prop-mode
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))

import argparse
import csv
import datetime
import logging
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import InstrumentConfig, BacktestConfig, INSTRUMENT_REGISTRY
from data.loader import load_bars
from strategy.hybrid_ema_ml import HybridEMAMLConfig
from strategy.strategy_engine import StrategyEngine, LiveSignal
from strategy.multi_strategy_engine import MultiStrategyEngine
from strategy.intraday_strategies import IntradayConfig
from strategy.paper_engine import PaperEngine, PaperConfig, PnLUpdate
from risk.risk_manager import RiskManager, RiskConfig, RiskEvent
from risk.prop_challenge import PropConfig, PropRiskGate
from risk.portfolio_risk import PortfolioRiskManager, PortfolioRiskConfig
from analytics.engine import AnalyticsEngine
from backtest.metrics import export_trades_csv
import csv

logger = logging.getLogger(__name__)

# Dedicated logger for the paper_trading.log
paper_logger = logging.getLogger("apex.paper_live")


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

def setup_paper_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configure logging with a dedicated paper_trading.log handler."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â real-time paper trading output
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper()))
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%H:%M:%S",
    ))
    console.addFilter(_PaperFilter())
    root.addHandler(console)

    # Full debug log
    all_handler = logging.FileHandler(log_path / "main.log", mode="w")
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(fmt)
    root.addHandler(all_handler)

    # Paper trading log ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the key output file
    paper_handler = logging.FileHandler(log_path / "paper_trading.log", mode="w")
    paper_handler.setLevel(logging.INFO)
    paper_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    paper_handler.addFilter(_PaperFilter())
    root.addHandler(paper_handler)

    # Errors
    error_handler = logging.FileHandler(log_path / "errors.log", mode="w")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(fmt)
    root.addHandler(error_handler)


class _PaperFilter(logging.Filter):
    """Pass only records from the paper live logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("apex.paper_live")


# ------------------------------------------------------------------
# Signal tracking for formatted output
# ------------------------------------------------------------------

class LivePaperTracker:
    """
    Tracks signals, risk decisions, trades, and equity for
    formatted real-time console output.
    """

    def __init__(self) -> None:
        self.signals_today: list[LiveSignal] = []
        self.blocked_today: list[tuple[LiveSignal, str]] = []
        self.trades_today: list = []
        self.current_date: Optional[datetime.date] = None
        self.daily_stats: dict[str, dict] = {}  # date_str -> stats

        # Counters for syncing
        self._last_trade_count: int = 0
        self._last_risk_event_count: int = 0

    def on_new_day(self, date: datetime.date) -> None:
        """Reset daily counters on day change."""
        if self.current_date is not None and self.current_date != date:
            # Save stats for the previous day
            self._save_daily_stats()
        self.current_date = date
        self.signals_today.clear()
        self.blocked_today.clear()
        self.trades_today.clear()

    def _save_daily_stats(self) -> None:
        """Save summary for the completed day."""
        if self.current_date is None:
            return
        key = self.current_date.isoformat()
        self.daily_stats[key] = {
            "signals": len(self.signals_today),
            "blocked": len(self.blocked_today),
            "trades": len(self.trades_today),
        }

    def finalize(self) -> None:
        """Save stats for the last day."""
        self._save_daily_stats()


# ------------------------------------------------------------------
# Pipeline builder (mirrors main.py but simpler)
# ------------------------------------------------------------------

def build_live_pipeline(
    args: argparse.Namespace,
    tracker: LivePaperTracker,
    telegram_alerter=None,
) -> dict:
    """
    Build the full pipeline with signal/trade callbacks that feed
    the tracker for formatted output.
    """
    instrument = InstrumentConfig()

    strategy_cfg = HybridEMAMLConfig(
        multi_candidate=len(args.ema_periods) > 1 or len(args.entry_types) > 1,
        ema_periods=tuple(args.ema_periods),
        entry_types=tuple(args.entry_types),
        ml_threshold=args.ml_threshold,
        model_path=args.ml_model,
        selection_strategy=args.selection_strategy,
        max_trades_per_day=args.max_trades_per_day,
    )

    risk_cfg = RiskConfig(
        max_daily_loss=args.max_daily_loss,
        max_trades_per_day=args.max_trades_per_day,
        max_concurrent_positions=args.max_concurrent,
    )

    paper_cfg = PaperConfig(
        slippage_ticks=args.slippage_ticks,
        commission_per_side=args.commission,
        initial_capital=args.initial_capital,
    )

    # Analytics
    analytics = AnalyticsEngine(initial_capital=args.initial_capital)

    # PnL callback
    def _on_pnl(update: PnLUpdate) -> None:
        pass  # We log PnL explicitly in the bar loop

    paper = PaperEngine(
        instrument=instrument,
        config=paper_cfg,
        on_update=_on_pnl,
    )

    risk = RiskManager(config=risk_cfg, instrument=instrument)

    # Prop mode
    prop_gate = None
    if args.prop_mode:
        prop_cfg = PropConfig(
            starting_capital=args.initial_capital,
            profit_target=args.prop_target,
            max_drawdown=args.prop_max_dd,
            daily_loss_limit=args.prop_daily_loss,
            daily_profit_lock=args.prop_daily_lock,
            max_trades_per_day=args.prop_max_trades,
            allowed_entry_types=tuple(args.prop_allowed_entries),
            max_consecutive_losses=args.prop_consecutive_losses,
            min_ml_prob=args.prop_min_ml_prob,
        )
        prop_gate = PropRiskGate(config=prop_cfg)
        risk_cfg = RiskConfig(
            max_daily_loss=args.prop_daily_loss,
            max_trades_per_day=args.max_trades_per_day,
            max_concurrent_positions=args.max_concurrent,
            max_per_direction=2,
        )
        risk = RiskManager(config=risk_cfg, instrument=instrument)

    # Execution callback ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â logs formatted trade output
    def _on_execution(sig: LiveSignal) -> None:
        paper.on_signal(sig)
        # Telegram entry alert (single-symbol mode)
        if telegram_alerter is not None and sig.is_entry:
            risk_amt = abs(sig.entry - sig.stop) if sig.stop else 0
            reward_amt = abs(sig.take_profit - sig.entry) if sig.take_profit else 0
            rr = reward_amt / risk_amt if risk_amt > 0 else 0.0
            telegram_alerter.send_entry_alert(
                symbol=instrument.symbol,
                direction=sig.direction,
                entry=sig.entry,
                stop=sig.stop,
                target=sig.take_profit,
                size=sig.position_size,
                strategy_type=sig.strategy_type,
                ml_prob=sig.ml_prob,
                quality_score=sig.quality_score,
                timestamp=sig.timestamp,
                rr_ratio=rr,
                open_positions=paper.open_position_count,
            )

    risk.on_approved = _on_execution

    # Signal callback chain ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â risk ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ execution
    risk_entry = risk.on_signal
    if prop_gate is not None:
        prop_gate.on_approved = risk.on_signal
        risk_entry = prop_gate.on_signal

    def _on_signal(sig: LiveSignal) -> None:
        analytics.record_signal(sig)
        tracker.signals_today.append(sig)
        risk_entry(sig)

    # Build engine
    intraday_cfg = None
    enable_hybrid = True
    if args.multi_strategy:
        intraday_cfg = IntradayConfig(
            vwap_band_mult=args.vwap_band_mult,
            momentum_lookback=args.momentum_lookback,
            rsi_oversold=args.rsi_oversold,
            rsi_overbought=args.rsi_overbought,
            entry_start=args.intraday_entry_start,
            entry_end=args.intraday_entry_end,
            entry_cooldown_bars=args.intraday_cooldown,
            min_quality_score=args.min_quality_score,
        )
        enable_hybrid = not args.disable_hybrid

    if intraday_cfg is not None:
        engine = MultiStrategyEngine(
            strategy_cfg=strategy_cfg,
            intraday_cfg=intraday_cfg,
            on_signal=_on_signal,
            enable_hybrid=enable_hybrid,
            enable_intraday=True,
            max_intraday_entries_per_bar=getattr(args, 'max_intraday_per_bar', 2),
        )
    else:
        engine = StrategyEngine(
            config=strategy_cfg,
            on_signal=_on_signal,
        )

    # Wire prop equity callback
    if prop_gate is not None:
        prop_gate._get_equity = lambda: paper.mark_to_market_equity

    return {
        "engine": engine,
        "risk": risk,
        "paper": paper,
        "analytics": analytics,
        "prop_gate": prop_gate,
        "tracker": tracker,
    }


# ------------------------------------------------------------------
# Real-time bar loop with formatted output
# ------------------------------------------------------------------

def run_paper_live(
    bars: pd.DataFrame,
    pipeline: dict,
    bar_delay: float = 0.1,
    telegram_alerter=None,
) -> dict:
    """
    Step through bars simulating a live feed.

    Prints formatted real-time logs for every signal, trade,
    equity update, and risk decision.

    Returns summary dict with key metrics.
    """
    engine = pipeline["engine"]
    risk: RiskManager = pipeline["risk"]
    paper: PaperEngine = pipeline["paper"]
    analytics: AnalyticsEngine = pipeline["analytics"]
    prop_gate: Optional[PropRiskGate] = pipeline.get("prop_gate")
    tracker: LivePaperTracker = pipeline["tracker"]

    total = len(bars)
    shutdown = False

    def _handle_sigint(signum, frame):
        nonlocal shutdown
        paper_logger.info("Shutdown signal received ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â stopping after current bar")
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)

    paper_logger.info("=" * 60)
    paper_logger.info("  PAPER TRADING ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â LIVE SIMULATION")
    paper_logger.info("  Bars: %d | Delay: %.2fs/bar", total, bar_delay)
    paper_logger.info("  Capital: $%.2f", paper.equity)
    paper_logger.info("=" * 60)

    prev_trade_count = 0
    prev_risk_count = 0
    prev_date = None
    day_count = 0
    anomalies = []

    for i, (ts, row) in enumerate(bars.iterrows()):
        if shutdown:
            paper_logger.info("Shutdown ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â stopping at bar %d / %d", i, total)
            break

        bar = {
            "timestamp": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

        bar_date = ts.date() if hasattr(ts, 'date') else ts
        bar_time = ts.strftime("%H:%M") if hasattr(ts, 'strftime') else str(ts)

        # Day change
        if bar_date != prev_date:
            if prev_date is not None:
                _log_day_summary(tracker, paper, risk, prev_date, paper_logger)
            tracker.on_new_day(bar_date)
            day_count += 1
            paper_logger.info("")
            paper_logger.info(
                "=== DAY %d: %s ================================",
                day_count, bar_date,
            )
            prev_date = bar_date

        # Feed bar through pipeline (order matters)
        if prop_gate is not None:
            prop_gate.on_bar(bar)
        risk.on_bar(bar)
        paper.on_bar(bar)

        # Capture signal count before engine processes
        pre_signal_count = len(tracker.signals_today)

        engine.on_bar(bar)

        # Sync prop trades
        if prop_gate is not None:
            _sync_prop(prop_gate, paper, tracker)

        # Sync analytics
        _sync_analytics(analytics, paper, risk)

        # Check for new signals generated this bar
        new_signals = tracker.signals_today[pre_signal_count:]
        for sig in new_signals:
            _log_signal_formatted(sig, bar_time, paper, risk, prev_risk_count, paper_logger)

        # Check for new risk events
        new_risk_events = risk.events[prev_risk_count:]
        for evt in new_risk_events:
            if evt.signal is not None:
                _log_risk_event(evt, bar_time, paper_logger)
        prev_risk_count = len(risk.events)

        # Check for new trades (closed)
        new_trades = paper.trades[prev_trade_count:]
        for t in new_trades:
            tracker.trades_today.append(t)
            _log_trade_close(t, paper_logger)
            # Telegram exit alert (single-symbol)
            if telegram_alerter is not None:
                telegram_alerter.send_exit_alert(
                    symbol=paper._inst.symbol,
                    direction=t.direction,
                    exit_price=t.exit_price,
                    exit_reason=t.exit_reason,
                    net_pnl=t.net_pnl,
                    timestamp=t.exit_time,
                )
        prev_trade_count = len(paper.trades)

        # Periodic equity log (every 5 bars or on position change)
        if i % 5 == 0 or new_trades or new_signals:
            mtm = paper.mark_to_market_equity
            dd = mtm - paper.peak_equity
            if abs(dd) > 0.01 or paper.open_position_count > 0:
                paper_logger.info(
                    "[%s] Equity: $%.2f | DD: $%.2f | Open: %d",
                    bar_time, mtm, dd, paper.open_position_count,
                )

        # Anomaly detection
        if paper.open_position_count > risk._cfg.max_concurrent_positions:
            msg = f"ANOMALY: {paper.open_position_count} positions exceeds max {risk._cfg.max_concurrent_positions}"
            paper_logger.warning(msg)
            anomalies.append(msg)

        # Pace the simulation
        if bar_delay > 0:
            time.sleep(bar_delay)

    # Final day summary
    if prev_date is not None:
        _log_day_summary(tracker, paper, risk, prev_date, paper_logger)
    tracker.finalize()

    # Build summary
    trades = paper.trades
    total_pnl = sum(t.net_pnl for t in trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    win_rate = (wins / len(trades) * 100) if trades else 0.0
    trading_days = day_count

    summary = {
        "total_bars": i + 1 if not shutdown else i,
        "trading_days": trading_days,
        "total_trades": len(trades),
        "trades_per_day": len(trades) / trading_days if trading_days > 0 else 0,
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": total_pnl / len(trades) if trades else 0,
        "final_equity": paper.mark_to_market_equity,
        "peak_equity": paper.peak_equity,
        "max_drawdown": paper.mark_to_market_equity - paper.peak_equity,
        "risk_events": len(risk.events),
        "anomalies": anomalies,
    }

    # Per-strategy breakdown
    by_strategy = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        st = getattr(t, 'strategy_type', 'unknown')
        by_strategy[st]["trades"] += 1
        by_strategy[st]["pnl"] += t.net_pnl
        if t.net_pnl > 0:
            by_strategy[st]["wins"] += 1
    summary["by_strategy"] = dict(by_strategy)

    # Prop event breakdown
    prop_gate_ref = pipeline.get("prop_gate")
    if prop_gate_ref is not None:
        prop_events = prop_gate_ref.events
        blocked_by_reason = defaultdict(int)
        for e in prop_events:
            blocked_by_reason[e.event_type] += 1
        summary["prop_events_total"] = len(prop_events)
        summary["prop_blocked_reasons"] = dict(blocked_by_reason)
    else:
        summary["prop_events_total"] = 0
        summary["prop_blocked_reasons"] = {}

    return summary


# ------------------------------------------------------------------
# Formatted log helpers
# ------------------------------------------------------------------

def _log_signal_formatted(
    sig: LiveSignal,
    bar_time: str,
    paper: PaperEngine,
    risk: RiskManager,
    prev_risk_count: int,
    log: logging.Logger,
) -> None:
    """Log a signal in the user-requested format."""
    # Check if this signal was blocked by risk (new risk events since last check)
    new_risk_events = risk.events[prev_risk_count:]
    was_blocked = False
    block_reason = ""
    for evt in new_risk_events:
        if evt.signal is not None and evt.event_type == "blocked":
            if (evt.signal.strategy_type == sig.strategy_type
                    and evt.signal.timestamp == sig.timestamp):
                was_blocked = True
                block_reason = evt.reason
                break

    strategy_name = sig.strategy_type.replace("_", " ")
    if sig.is_entry:
        if was_blocked:
            log.info(
                "[%s] %s -> FILTERED (%s)",
                bar_time, strategy_name, block_reason,
            )
        else:
            log.info(
                "[%s] %s -> EXECUTED %s contracts=%d @ %.2f (sl=%.2f tp=%.2f)",
                bar_time, strategy_name, sig.direction,
                max(1, int(sig.position_size)), sig.entry, sig.stop, sig.take_profit,
            )
    elif sig.is_exit:
        log.info(
            "[%s] %s -> EXIT %s @ %.2f [%s]",
            bar_time, strategy_name, sig.signal_type.name,
            sig.entry, sig.reason,
        )


def _log_risk_event(
    evt: RiskEvent,
    bar_time: str,
    log: logging.Logger,
) -> None:
    """Log a risk event."""
    if evt.event_type == "blocked":
        strategy = evt.signal.strategy_type if evt.signal else "unknown"
        log.info(
            "[%s] RISK BLOCKED: %s -- %s",
            bar_time, strategy.replace("_", " "), evt.reason,
        )
    elif evt.event_type == "kill_switch":
        log.warning("[%s] KILL SWITCH ACTIVATED: %s", bar_time, evt.reason)
    elif evt.event_type == "capped":
        log.info("[%s] RISK CAPPED: %s", bar_time, evt.reason)


def _log_trade_close(trade, log: logging.Logger) -> None:
    """Log a closed trade with PnL."""
    pnl_symbol = "+" if trade.net_pnl >= 0 else ""
    log.info(
        "  -> CLOSED %s contracts=%d [%s] %s$%.2f (%s)",
        trade.direction, getattr(trade, 'contracts', 1),
        getattr(trade, 'strategy_type', ''),
        pnl_symbol, trade.net_pnl, trade.exit_reason,
    )


def _log_day_summary(
    tracker: LivePaperTracker,
    paper: PaperEngine,
    risk: RiskManager,
    date: datetime.date,
    log: logging.Logger,
) -> None:
    """Print end-of-day summary."""
    trades = tracker.trades_today
    total_pnl = sum(t.net_pnl for t in trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)

    log.info("")
    log.info("-- Day Summary: %s --", date)
    log.info(
        "  Trades: %d (%dW / %dL) | PnL: $%.2f",
        len(trades), wins, len(trades) - wins, total_pnl,
    )
    log.info(
        "  Signals: %d | Blocked: %d",
        len(tracker.signals_today), len(tracker.blocked_today),
    )
    log.info(
        "  Equity: $%.2f | Peak: $%.2f | DD: $%.2f",
        paper.mark_to_market_equity,
        paper.peak_equity,
        paper.mark_to_market_equity - paper.peak_equity,
    )
    log.info(
        "  Open positions: %d",
        paper.open_position_count,
    )


# ------------------------------------------------------------------
# Sync helpers
# ------------------------------------------------------------------

_prop_trade_cursor: int = 0


def _sync_prop(
    prop: PropRiskGate,
    paper: PaperEngine,
    tracker: LivePaperTracker,
) -> None:
    """Notify prop gate about newly closed trades."""
    global _prop_trade_cursor
    trades = paper.trades
    for t in trades[_prop_trade_cursor:]:
        prop.on_trade_closed(
            t.net_pnl,
            strategy_type=getattr(t, 'strategy_type', ''),
        )
    _prop_trade_cursor = len(trades)


def _sync_analytics(
    analytics: AnalyticsEngine,
    paper: PaperEngine,
    risk: RiskManager,
) -> None:
    """Push newly completed trades and risk decisions to analytics."""
    paper_trades = paper.trades
    ana_count = analytics.trade_count
    for t in paper_trades[ana_count:]:
        analytics.record_trade(t)

    risk_events = risk.events
    dec_count = analytics.decision_count
    for e in risk_events[dec_count:]:
        analytics.record_decision(e)


# ------------------------------------------------------------------
# Final report
# ------------------------------------------------------------------

def print_final_report(summary: dict) -> None:
    """Print the final paper trading report."""
    paper_logger.info("")
    paper_logger.info("=" * 60)
    paper_logger.info("  PAPER TRADING ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â FINAL REPORT")
    paper_logger.info("=" * 60)
    paper_logger.info("  Trading days:    %d", summary["trading_days"])
    paper_logger.info("  Total trades:    %d", summary["total_trades"])
    paper_logger.info("  Trades/day:      %.1f", summary["trades_per_day"])
    paper_logger.info("  Win rate:        %.1f%%", summary["win_rate"])
    paper_logger.info("  Avg PnL/trade:   $%.2f", summary["avg_pnl_per_trade"])
    paper_logger.info("  Total PnL:       $%.2f", summary["total_pnl"])
    paper_logger.info("  Final equity:    $%.2f", summary["final_equity"])
    paper_logger.info("  Peak equity:     $%.2f", summary["peak_equity"])
    paper_logger.info("  Max drawdown:    $%.2f", summary["max_drawdown"])
    paper_logger.info("  Risk events:     %d", summary["risk_events"])

    # Per-strategy breakdown
    by_strategy = summary.get("by_strategy", {})
    if by_strategy:
        paper_logger.info("")
        paper_logger.info("  --- BY STRATEGY ---")
        for st, stats in sorted(by_strategy.items()):
            n = stats["trades"]
            w = stats["wins"]
            pnl = stats["pnl"]
            wr = (w / n * 100) if n > 0 else 0
            avg = pnl / n if n > 0 else 0
            paper_logger.info(
                "  [%s] %d trades, %.1f%% WR, $%.2f total, $%.2f avg",
                st, n, wr, pnl, avg,
            )

    # Prop event breakdown
    prop_total = summary.get("prop_events_total", 0)
    prop_reasons = summary.get("prop_blocked_reasons", {})
    if prop_total > 0:
        paper_logger.info("")
        paper_logger.info("  --- PROP GATE EVENTS (%d total) ---", prop_total)
        for reason, count in sorted(prop_reasons.items(), key=lambda x: -x[1]):
            paper_logger.info("    %s: %d", reason, count)

    if summary["anomalies"]:
        paper_logger.info("")
        paper_logger.warning("  ANOMALIES DETECTED:")
        for a in summary["anomalies"]:
            paper_logger.warning("    - %s", a)
    else:
        paper_logger.info("  Anomalies:       None")

    paper_logger.info("")
    if summary["total_trades"] > 0 and not summary["anomalies"]:
        paper_logger.info("  STATUS: STABLE ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â System operating correctly")
    elif summary["anomalies"]:
        paper_logger.info("  STATUS: ANOMALIES DETECTED ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Review required")
    else:
        paper_logger.info("  STATUS: NO TRADES ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Check strategy parameters")
    paper_logger.info("=" * 60)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

# ==================================================================
# Multi-symbol support
# ==================================================================


@dataclass
class SymbolPipeline:
    """All components for a single symbol ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â fully independent state."""
    symbol: str
    instrument: InstrumentConfig
    engine: object  # StrategyEngine or MultiStrategyEngine
    risk: RiskManager
    paper: PaperEngine
    analytics: AnalyticsEngine
    prop_gate: Optional[PropRiskGate] = None
    tracker: Optional[LivePaperTracker] = None


class MultiSymbolRouter:
    """
    Coordinates per-symbol pipelines with shared portfolio risk.

    Flow per timestamp:
      1. Each symbol's risk + paper get on_bar
      2. Each symbol's engine generates signals
         - Exit signals forward immediately
         - Entry signals collected for portfolio ranking
      3. Portfolio risk ranks and filters entries

    When *tradovate_adapter* is provided, approved signals are also
    routed to Tradovate SIM for real order execution alongside the
    local paper engine (dual tracking).

    When *telegram_alerter* is provided, approved entry signals
    trigger a Telegram alert after execution.
    """

    def __init__(
        self,
        pipelines: dict[str, SymbolPipeline],
        portfolio_risk: PortfolioRiskManager,
        tradovate_adapter=None,
        telegram_alerter=None,
    ) -> None:
        self._pipelines = pipelines
        self._portfolio_risk = portfolio_risk
        self._pending_entries: list[tuple[str, LiveSignal]] = []
        self._tradovate_adapter = tradovate_adapter
        self._telegram_alerter = telegram_alerter

        # Wire callbacks per symbol
        for sym, pipe in pipelines.items():
            pipe.risk.on_approved = self._make_execution_tracker(sym)
            pipe.engine._on_signal = self._make_signal_handler(sym)

    def process_bars(self, bars_by_symbol: dict[str, dict]) -> None:
        """Process one timestamp across all symbols in lockstep."""
        self._pending_entries.clear()

        for sym, bar in bars_by_symbol.items():
            pipe = self._pipelines[sym]
            if pipe.prop_gate is not None:
                pipe.prop_gate.on_bar(bar)
            pipe.risk.on_bar(bar)
            pipe.paper.on_bar(bar)
            pipe.engine.on_bar(bar)

            # Sync prop gate
            if pipe.prop_gate is not None:
                _sync_prop_for_symbol(pipe.prop_gate, pipe.paper, pipe)

        # Rank and filter entries through portfolio risk
        self._flush_entries()

    def total_equity(self) -> float:
        return sum(p.paper.mark_to_market_equity for p in self._pipelines.values())

    def _make_signal_handler(self, symbol: str):
        """Exits forward immediately; entries collected for portfolio ranking."""
        pipe = self._pipelines[symbol]

        def handler(signal: LiveSignal) -> None:
            if pipe.analytics:
                pipe.analytics.record_signal(signal)
            if pipe.tracker:
                pipe.tracker.signals_today.append(signal)
            if signal.is_exit:
                pipe.risk.on_signal(signal)
            elif signal.is_entry:
                self._pending_entries.append((symbol, signal))
            else:
                pipe.risk.on_signal(signal)
        return handler

    def _make_execution_tracker(self, symbol: str):
        """Wraps paper.on_signal to track portfolio positions.

        When a Tradovate adapter is attached, signals are also
        forwarded to Tradovate SIM for real order execution.
        Entry signals trigger a Telegram alert when alerter is attached.
        """
        paper = self._pipelines[symbol].paper
        adapter = self._tradovate_adapter
        alerter = self._telegram_alerter

        def tracker(signal: LiveSignal) -> None:
            if signal.is_entry:
                self._portfolio_risk.record_entry(symbol, signal)
            elif signal.is_exit:
                self._portfolio_risk.record_exit(symbol, signal)
            # Local paper tracking (always runs)
            paper.on_signal(signal)
            # Tradovate SIM execution (when attached)
            if adapter is not None:
                adapter.route_signal(symbol, signal)
            # Telegram entry alert
            if alerter is not None and signal.is_entry:
                risk = abs(signal.entry - signal.stop) if signal.stop else 0
                reward = abs(signal.take_profit - signal.entry) if signal.take_profit else 0
                rr = reward / risk if risk > 0 else 0.0
                alerter.send_entry_alert(
                    symbol=symbol,
                    direction=signal.direction,
                    entry=signal.entry,
                    stop=signal.stop,
                    target=signal.take_profit,
                    size=signal.position_size,
                    strategy_type=signal.strategy_type,
                    ml_prob=signal.ml_prob,
                    quality_score=signal.quality_score,
                    timestamp=signal.timestamp,
                    rr_ratio=rr,
                    open_positions=self._portfolio_risk.open_position_count,
                )
        return tracker

    def _flush_entries(self) -> None:
        """Rank pending entries by ML quality and filter through portfolio risk."""
        if not self._pending_entries:
            return
        ranked = self._portfolio_risk.rank_signals(self._pending_entries)
        for sym, sig in ranked:
            approved = self._portfolio_risk.check_entry(sym, sig)
            if approved is not None:
                # Route through prop gate if present, else direct to risk
                pipe = self._pipelines[sym]
                if pipe.prop_gate is not None:
                    pipe.prop_gate.on_signal(approved)
                else:
                    pipe.risk.on_signal(approved)


def _sync_prop_for_symbol(
    prop: PropRiskGate,
    paper: PaperEngine,
    pipe: SymbolPipeline,
) -> None:
    """Notify prop gate about newly closed trades for a symbol."""
    cursor_attr = "_prop_trade_cursor"
    cursor = getattr(pipe, cursor_attr, 0)
    trades = paper.trades
    for t in trades[cursor:]:
        prop.on_trade_closed(
            t.net_pnl,
            strategy_type=getattr(t, "strategy_type", ""),
        )
    setattr(pipe, cursor_attr, len(trades))


def build_multi_pipeline(
    args: argparse.Namespace,
    symbols: list[str],
    tradovate_adapter=None,
    telegram_alerter=None,
) -> tuple[MultiSymbolRouter, PortfolioRiskManager, dict[str, SymbolPipeline]]:
    """
    Build fully independent per-symbol pipelines with shared portfolio risk.

    Each symbol gets its own StrategyEngine, RiskManager, PaperEngine,
    AnalyticsEngine ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â no shared indicator or EMA state.
    """
    strategy_cfg = HybridEMAMLConfig(
        multi_candidate=len(args.ema_periods) > 1 or len(args.entry_types) > 1,
        ema_periods=tuple(args.ema_periods),
        entry_types=tuple(args.entry_types),
        ml_threshold=args.ml_threshold,
        model_path=args.ml_model,
        selection_strategy=args.selection_strategy,
        max_trades_per_day=args.max_trades_per_day,
    )

    risk_cfg = RiskConfig(
        max_daily_loss=args.max_daily_loss,
        max_trades_per_day=args.max_trades_per_day,
        max_concurrent_positions=args.max_concurrent,
        max_per_direction=2,
    )

    per_symbol_capital = args.initial_capital / len(symbols)
    paper_cfg = PaperConfig(
        slippage_ticks=args.slippage_ticks,
        commission_per_side=args.commission,
        initial_capital=per_symbol_capital,
    )

    portfolio_cfg = PortfolioRiskConfig(
        max_total_concurrent=args.portfolio_max_concurrent,
        max_same_direction=args.portfolio_max_same_dir,
        max_total_exposure=args.portfolio_max_exposure,
        correlation_divisor=args.correlation_divisor,
    )
    portfolio_risk = PortfolioRiskManager(portfolio_cfg)

    pipelines: dict[str, SymbolPipeline] = {}
    for sym in symbols:
        instrument = INSTRUMENT_REGISTRY.get(sym)
        if instrument is None:
            paper_logger.error("Unknown symbol: %s ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â not in INSTRUMENT_REGISTRY", sym)
            sys.exit(1)

        # Prop mode (per-symbol)
        prop_gate = None
        sym_risk_cfg = risk_cfg
        if args.prop_mode:
            prop_cfg = PropConfig(
                starting_capital=per_symbol_capital,
                profit_target=args.prop_target / len(symbols),
                max_drawdown=args.prop_max_dd / len(symbols),
                daily_loss_limit=args.prop_daily_loss / len(symbols),
                daily_profit_lock=args.prop_daily_lock / len(symbols),
                max_trades_per_day=args.prop_max_trades,
                allowed_entry_types=tuple(args.prop_allowed_entries),
                max_consecutive_losses=args.prop_consecutive_losses,
                min_ml_prob=args.prop_min_ml_prob,
            )
            prop_gate = PropRiskGate(config=prop_cfg)
            sym_risk_cfg = RiskConfig(
                max_daily_loss=args.prop_daily_loss / len(symbols),
                max_trades_per_day=args.max_trades_per_day,
                max_concurrent_positions=args.max_concurrent,
                max_per_direction=2,
            )

        risk = RiskManager(config=sym_risk_cfg, instrument=instrument)
        paper = PaperEngine(instrument=instrument, config=paper_cfg)
        analytics = AnalyticsEngine(initial_capital=per_symbol_capital)
        tracker = LivePaperTracker()

        # Build engine ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â multi-strategy or EMA-only, per symbol
        intraday_cfg = None
        enable_hybrid = True
        if args.multi_strategy:
            intraday_cfg = IntradayConfig(
                vwap_band_mult=args.vwap_band_mult,
                momentum_lookback=args.momentum_lookback,
                rsi_oversold=args.rsi_oversold,
                rsi_overbought=args.rsi_overbought,
                entry_start=args.intraday_entry_start,
                entry_end=args.intraday_entry_end,
                entry_cooldown_bars=args.intraday_cooldown,
                min_quality_score=args.min_quality_score,
            )
            enable_hybrid = not args.disable_hybrid

        if intraday_cfg is not None:
            engine = MultiStrategyEngine(
                strategy_cfg=strategy_cfg,
                intraday_cfg=intraday_cfg,
                on_signal=None,  # wired by router
                enable_hybrid=enable_hybrid,
                enable_intraday=True,
                max_intraday_entries_per_bar=getattr(args, "max_intraday_per_bar", 2),
            )
        else:
            engine = StrategyEngine(
                config=strategy_cfg,
                on_signal=None,  # wired by router
            )

        # Wire prop gate into risk chain
        if prop_gate is not None:
            prop_gate.on_approved = risk.on_signal
            prop_gate._get_equity = lambda p=paper: p.mark_to_market_equity

        pipe = SymbolPipeline(
            symbol=sym,
            instrument=instrument,
            engine=engine,
            risk=risk,
            paper=paper,
            analytics=analytics,
            prop_gate=prop_gate,
            tracker=tracker,
        )
        pipelines[sym] = pipe

    router = MultiSymbolRouter(pipelines, portfolio_risk, tradovate_adapter, telegram_alerter)
    return router, portfolio_risk, pipelines


# ------------------------------------------------------------------
# Multi-symbol bar loop
# ------------------------------------------------------------------

def run_paper_multi_symbol(
    bars_by_symbol: dict[str, pd.DataFrame],
    router: MultiSymbolRouter,
    portfolio_risk: PortfolioRiskManager,
    pipelines: dict[str, SymbolPipeline],
    bar_delay: float = 0.0,
    telegram_alerter=None,
) -> dict:
    """
    Step through merged bars across all symbols in lockstep.

    Timestamps are merged and sorted so all symbols process the
    same moment before advancing.
    """
    # Merge all timestamps
    all_timestamps = set()
    for df in bars_by_symbol.values():
        all_timestamps.update(df.index)
    all_timestamps = sorted(all_timestamps)

    total_bars = len(all_timestamps)
    symbols = list(bars_by_symbol.keys())
    shutdown = False

    def _handle_sigint(signum, frame):
        nonlocal shutdown
        paper_logger.info("Shutdown signal received ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â stopping after current bar")
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)

    paper_logger.info("=" * 60)
    paper_logger.info("  MULTI-SYMBOL PAPER TRADING ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â LIVE SIMULATION")
    paper_logger.info("  Symbols: %s", ", ".join(symbols))
    paper_logger.info("  Merged timestamps: %d", total_bars)
    paper_logger.info("  Delay: %.2fs/bar", bar_delay)
    for sym in symbols:
        pipe = pipelines[sym]
        paper_logger.info("  %s capital: $%.2f", sym, pipe.paper.equity)
    paper_logger.info("=" * 60)

    prev_date = None
    day_count = 0
    prev_trade_counts = {sym: 0 for sym in symbols}
    prev_risk_counts = {sym: 0 for sym in symbols}

    for i, ts in enumerate(all_timestamps):
        if shutdown:
            paper_logger.info("Shutdown at bar %d / %d", i, total_bars)
            break

        bar_date = ts.date() if hasattr(ts, "date") else ts
        bar_time = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)

        # Day change
        if bar_date != prev_date:
            if prev_date is not None:
                _log_multi_day_summary(pipelines, portfolio_risk, prev_date, symbols)
            day_count += 1
            for pipe in pipelines.values():
                pipe.tracker.on_new_day(bar_date)
            paper_logger.info("")
            paper_logger.info(
                "=== DAY %d: %s ================================",
                day_count, bar_date,
            )
            prev_date = bar_date

        # Build bars dict for this timestamp
        bars_at_ts: dict[str, dict] = {}
        for sym in symbols:
            df = bars_by_symbol[sym]
            if ts in df.index:
                row = df.loc[ts]
                bars_at_ts[sym] = {
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }

        # Process all symbols at this timestamp
        router.process_bars(bars_at_ts)

        # Log new signals, trades, risk events per symbol
        for sym in symbols:
            pipe = pipelines[sym]

            # New trades
            cur_trades = pipe.paper.trades
            for t in cur_trades[prev_trade_counts[sym]:]:
                pipe.tracker.trades_today.append(t)
                pnl_sym = "+" if t.net_pnl >= 0 else ""
                paper_logger.info(
                    "[%s] %s %s -> CLOSED %s %s$%.2f (%s)",
                    bar_time, sym,
                    getattr(t, "strategy_type", "").replace("_", " "),
                    t.direction, pnl_sym, t.net_pnl, t.exit_reason,
                )
                # Telegram exit alert
                if telegram_alerter is not None:
                    telegram_alerter.send_exit_alert(
                        symbol=sym,
                        direction=t.direction,
                        exit_price=t.exit_price,
                        exit_reason=t.exit_reason,
                        net_pnl=t.net_pnl,
                        timestamp=t.exit_time,
                    )
            prev_trade_counts[sym] = len(cur_trades)

            # New risk events
            cur_risk = pipe.risk.events
            for evt in cur_risk[prev_risk_counts[sym]:]:
                if evt.event_type == "blocked" and evt.signal:
                    paper_logger.info(
                        "[%s] %s RISK BLOCKED: %s -- %s",
                        bar_time, sym,
                        evt.signal.strategy_type.replace("_", " "),
                        evt.reason,
                    )
            prev_risk_counts[sym] = len(cur_risk)

            # Sync analytics
            _sync_analytics(pipe.analytics, pipe.paper, pipe.risk)

        # Periodic equity log
        if i % 5 == 0:
            total_eq = router.total_equity()
            port_open = portfolio_risk.open_position_count
            if port_open > 0 or i % 25 == 0:
                paper_logger.info(
                    "[%s] Portfolio: $%.2f | Open: %d",
                    bar_time, total_eq, port_open,
                )

        if bar_delay > 0:
            time.sleep(bar_delay)

    # Final day summary
    if prev_date is not None:
        _log_multi_day_summary(pipelines, portfolio_risk, prev_date, symbols)
    for pipe in pipelines.values():
        pipe.tracker.finalize()

    return _build_multi_summary(pipelines, portfolio_risk, symbols, day_count)


def _log_multi_day_summary(
    pipelines: dict[str, SymbolPipeline],
    portfolio_risk: PortfolioRiskManager,
    date,
    symbols: list[str],
) -> None:
    paper_logger.info("")
    paper_logger.info("-- Day Summary: %s --", date)
    for sym in symbols:
        pipe = pipelines[sym]
        trades = pipe.tracker.trades_today
        total_pnl = sum(t.net_pnl for t in trades)
        wins = sum(1 for t in trades if t.net_pnl > 0)
        paper_logger.info(
            "  %s: %d trades (%dW/%dL) PnL=$%.2f equity=$%.2f open=%d",
            sym, len(trades), wins, len(trades) - wins, total_pnl,
            pipe.paper.mark_to_market_equity, pipe.paper.open_position_count,
        )
    total_eq = sum(p.paper.mark_to_market_equity for p in pipelines.values())
    paper_logger.info("  Portfolio: $%.2f", total_eq)


def _build_multi_summary(
    pipelines: dict[str, SymbolPipeline],
    portfolio_risk: PortfolioRiskManager,
    symbols: list[str],
    day_count: int,
) -> dict:
    """Aggregate metrics from all symbol pipelines."""
    all_trades = []
    by_symbol = {}
    total_pnl = 0.0

    for sym in symbols:
        pipe = pipelines[sym]
        trades = pipe.paper.trades
        sym_pnl = sum(t.net_pnl for t in trades)
        sym_wins = sum(1 for t in trades if t.net_pnl > 0)
        by_symbol[sym] = {
            "trades": len(trades),
            "wins": sym_wins,
            "losses": len(trades) - sym_wins,
            "pnl": sym_pnl,
            "avg_pnl": sym_pnl / len(trades) if trades else 0.0,
            "win_rate": (sym_wins / len(trades) * 100) if trades else 0.0,
            "equity": pipe.paper.mark_to_market_equity,
            "peak_equity": pipe.paper.peak_equity,
            "drawdown": pipe.paper.mark_to_market_equity - pipe.paper.peak_equity,
        }
        for t in trades:
            all_trades.append((sym, t))
        total_pnl += sym_pnl

    total_equity = sum(p.paper.mark_to_market_equity for p in pipelines.values())
    peak_equity = sum(p.paper.peak_equity for p in pipelines.values())
    total_count = len(all_trades)
    total_wins = sum(1 for _, t in all_trades if t.net_pnl > 0)
    max_dd = sum(
        pipelines[s].paper.mark_to_market_equity - pipelines[s].paper.peak_equity
        for s in symbols
    )

    # By strategy across all symbols
    by_strategy = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for _, t in all_trades:
        st = getattr(t, "strategy_type", "unknown")
        by_strategy[st]["trades"] += 1
        by_strategy[st]["pnl"] += t.net_pnl
        if t.net_pnl > 0:
            by_strategy[st]["wins"] += 1

    # Portfolio risk events
    port_events = portfolio_risk.events
    port_by_type = defaultdict(int)
    for e in port_events:
        port_by_type[e.event_type] += 1

    # Prop events
    prop_total = 0
    prop_reasons: dict[str, int] = defaultdict(int)
    for pipe in pipelines.values():
        if pipe.prop_gate is not None:
            for e in pipe.prop_gate.events:
                prop_total += 1
                prop_reasons[e.event_type] += 1

    return {
        "trading_days": day_count,
        "symbols": symbols,
        "total_trades": total_count,
        "trades_per_day": total_count / day_count if day_count > 0 else 0,
        "wins": total_wins,
        "losses": total_count - total_wins,
        "win_rate": (total_wins / total_count * 100) if total_count > 0 else 0.0,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": total_pnl / total_count if total_count > 0 else 0,
        "final_equity": total_equity,
        "peak_equity": peak_equity,
        "max_drawdown": max_dd,
        "by_symbol": by_symbol,
        "by_strategy": dict(by_strategy),
        "portfolio_events_total": len(port_events),
        "portfolio_events_by_type": dict(port_by_type),
        "prop_events_total": prop_total,
        "prop_blocked_reasons": dict(prop_reasons),
        "anomalies": [],
        "all_trades": all_trades,
        "risk_events": sum(len(p.risk.events) for p in pipelines.values()),
    }


def print_multi_report(summary: dict) -> None:
    """Print the multi-symbol final report."""
    symbols = summary["symbols"]
    paper_logger.info("")
    paper_logger.info("=" * 60)
    paper_logger.info("  MULTI-SYMBOL PAPER TRADING ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â FINAL REPORT")
    paper_logger.info("=" * 60)
    paper_logger.info("  Symbols:         %s", ", ".join(symbols))
    paper_logger.info("  Trading days:    %d", summary["trading_days"])
    paper_logger.info("  Total trades:    %d", summary["total_trades"])
    paper_logger.info("  Trades/day:      %.1f", summary["trades_per_day"])
    paper_logger.info("  Win rate:        %.1f%%", summary["win_rate"])
    paper_logger.info("  Avg PnL/trade:   $%.2f", summary["avg_pnl_per_trade"])
    paper_logger.info("  Total PnL:       $%.2f", summary["total_pnl"])
    paper_logger.info("  Portfolio equity: $%.2f", summary["final_equity"])
    paper_logger.info("  Peak equity:     $%.2f", summary["peak_equity"])
    paper_logger.info("  Max drawdown:    $%.2f", summary["max_drawdown"])

    paper_logger.info("")
    paper_logger.info("  --- BY SYMBOL ---")
    for sym, stats in summary["by_symbol"].items():
        paper_logger.info(
            "  [%s] %d trades, %.1f%% WR, $%.2f total, $%.2f avg, DD $%.2f",
            sym, stats["trades"], stats["win_rate"],
            stats["pnl"], stats["avg_pnl"], stats["drawdown"],
        )

    by_strategy = summary.get("by_strategy", {})
    if by_strategy:
        paper_logger.info("")
        paper_logger.info("  --- BY STRATEGY ---")
        for st, stats in sorted(by_strategy.items()):
            n = stats["trades"]
            w = stats["wins"]
            pnl = stats["pnl"]
            wr = (w / n * 100) if n > 0 else 0
            avg = pnl / n if n > 0 else 0
            paper_logger.info(
                "  [%s] %d trades, %.1f%% WR, $%.2f total, $%.2f avg",
                st, n, wr, pnl, avg,
            )

    port_total = summary.get("portfolio_events_total", 0)
    if port_total > 0:
        paper_logger.info("")
        paper_logger.info("  --- PORTFOLIO RISK EVENTS (%d total) ---", port_total)
        for reason, count in sorted(
            summary.get("portfolio_events_by_type", {}).items(), key=lambda x: -x[1]
        ):
            paper_logger.info("    %s: %d", reason, count)

    prop_total = summary.get("prop_events_total", 0)
    if prop_total > 0:
        paper_logger.info("")
        paper_logger.info("  --- PROP GATE EVENTS (%d total) ---", prop_total)
        for reason, count in sorted(
            summary.get("prop_blocked_reasons", {}).items(), key=lambda x: -x[1]
        ):
            paper_logger.info("    %s: %d", reason, count)

    paper_logger.info("")
    if summary["total_trades"] > 0:
        paper_logger.info("  STATUS: STABLE ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â System operating correctly")
    else:
        paper_logger.info("  STATUS: NO TRADES ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Check strategy parameters")
    paper_logger.info("=" * 60)


def export_multi_trades(all_trades: list[tuple[str, object]], path: str) -> None:
    """Export trades from all symbols to a single CSV with symbol column."""
    if not all_trades:
        return
    fieldnames = [
        "symbol", "entry_time", "exit_time", "direction", "contracts",
        "entry_price", "exit_price", "stop_loss", "take_profit",
        "pnl_ticks", "pnl_points", "pnl_dollars", "commission", "slippage_cost",
        "net_pnl", "exit_reason", "position_size", "strategy_type",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sym, t in all_trades:
            pnl_ticks = t.pnl_points / 0.25 if t.pnl_points != 0 else 0.0
            writer.writerow({
                "symbol": sym,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "contracts": getattr(t, 'contracts', 1),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "pnl_ticks": round(pnl_ticks, 1),
                "pnl_points": round(t.pnl_points, 4),
                "pnl_dollars": round(t.pnl_dollars, 2),
                "commission": round(t.commission, 2),
                "slippage_cost": round(t.slippage_cost, 2),
                "net_pnl": round(t.net_pnl, 2),
                "exit_reason": t.exit_reason,
                "position_size": t.position_size,
                "strategy_type": t.strategy_type,
            })


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apex ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Live Paper Trading Simulation",
    )

    # Data
    p.add_argument("--data", default="data/mes_4y.csv",
                   help="Path to OHLCV CSV file (single-symbol mode)")
    p.add_argument("--days", type=int, default=2,
                   help="Number of trading days to simulate (from end of data)")
    p.add_argument("--start", default=None,
                   help="Start date filter (YYYY-MM-DD)")
    p.add_argument("--end", default=None,
                   help="End date filter (YYYY-MM-DD)")

    # Multi-symbol
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Symbols to trade (e.g. MES MNQ RTY). Enables multi-symbol mode.")
    p.add_argument("--data-MES", default="data/mes_4y.csv", help="CSV for MES")
    p.add_argument("--data-MNQ", default="data/mnq_4y.csv", help="CSV for MNQ")
    p.add_argument("--data-RTY", default="data/rty_4y.csv", help="CSV for RTY")

    # Portfolio risk (multi-symbol)
    p.add_argument("--portfolio-max-concurrent", type=int, default=3,
                   help="Max total concurrent positions across all symbols")
    p.add_argument("--portfolio-max-same-dir", type=int, default=2,
                   help="Max positions in same direction across all symbols")
    p.add_argument("--portfolio-max-exposure", type=float, default=3.0)
    p.add_argument("--correlation-divisor", type=float, default=2.0,
                   help="Divide position size when correlated symbols are active")

    # Simulation
    p.add_argument("--bar-delay", type=float, default=0.1,
                   help="Delay in seconds between bars (0 for instant)")

    # Strategy
    p.add_argument("--ml-model", default="models/ema_model.pkl",
                   help="Path to ML model")
    p.add_argument("--ml-threshold", type=float, default=0.55,
                   help="ML probability threshold")
    p.add_argument("--ema-periods", nargs="+", type=int, default=[50],
                   help="EMA periods")
    p.add_argument("--entry-types", nargs="+", default=["breakout"],
                   help="Entry types")
    p.add_argument("--selection-strategy", default="global_ml",
                   choices=["global_ml", "priority", "priority_ml_sizing"])
    p.add_argument("--shorts", action="store_true",
                   help="Enable short entries")

    # Risk
    p.add_argument("--max-daily-loss", type=float, default=500.0)
    p.add_argument("--max-trades-per-day", type=int, default=6)
    p.add_argument("--max-concurrent", type=int, default=3)

    # Paper
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--commission", type=float, default=0.62)

    # Prop mode
    p.add_argument("--prop-mode", action="store_true",
                   help="Enable prop firm challenge mode")
    p.add_argument("--prop-target", type=float, default=1500.0)
    p.add_argument("--prop-max-dd", type=float, default=1000.0)
    p.add_argument("--prop-daily-loss", type=float, default=300.0)
    p.add_argument("--prop-daily-lock", type=float, default=400.0)
    p.add_argument("--prop-max-trades", type=int, default=4)
    p.add_argument("--prop-allowed-entries", nargs="+",
                   default=["breakout", "vwap_bounce", "intraday_momentum", "mean_reversion"],
                   help="Allowed entry types in prop mode")
    p.add_argument("--prop-consecutive-losses", type=int, default=5)
    p.add_argument("--prop-min-ml-prob", type=float, default=0.55)

    # Multi-strategy
    p.add_argument("--multi-strategy", action="store_true",
                   help="Enable intraday strategies")
    p.add_argument("--disable-hybrid", action="store_true")
    p.add_argument("--vwap-band-mult", type=float, default=0.3)
    p.add_argument("--momentum-lookback", type=int, default=3)
    p.add_argument("--rsi-oversold", type=float, default=30.0)
    p.add_argument("--rsi-overbought", type=float, default=70.0)
    p.add_argument("--intraday-entry-start", default="10:00")
    p.add_argument("--intraday-entry-end", default="15:30")
    p.add_argument("--intraday-cooldown", type=int, default=6)
    p.add_argument("--min-quality-score", type=float, default=0.5)
    p.add_argument("--max-intraday-per-bar", type=int, default=2)

    # Tradovate SIM
    p.add_argument("--tradovate-sim", action="store_true",
                   help="Route signals to Tradovate SIM (requires .env credentials)")
    p.add_argument("--tradovate-dry-run", action="store_true",
                   help="Validate Tradovate auth + contracts only, no orders")
    p.add_argument("--tradovate-contract-MES", default=None,
                   help="Override Tradovate contract name for MES (e.g. MESM5)")
    p.add_argument("--tradovate-contract-MNQ", default=None,
                   help="Override Tradovate contract name for MNQ (e.g. MNQM5)")

    # Telegram alerts
    p.add_argument("--test-telegram", action="store_true",
                   help="Send a test Telegram message and exit")

    # Logging
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    setup_paper_logging(level=args.log_level)
    paper_logger.info("Apex Paper Trading ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â starting")

    # --test-telegram: send a test message and exit
    if args.test_telegram:
        from strategy.telegram_alerts import create_alerter
        alerter = create_alerter(warn=True)
        if not alerter.enabled:
            paper_logger.error(
                "Telegram credentials missing. Set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env"
            )
            return 1
        ok = alerter.send_test()
        if ok:
            paper_logger.info("Telegram test message sent successfully")
            return 0
        else:
            paper_logger.error("Telegram test message FAILED ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â check token/chat_id")
            return 1

    # Detect multi-symbol mode
    multi_symbol = args.symbols is not None and len(args.symbols) > 1

    if multi_symbol:
        return _main_multi_symbol(args)
    else:
        return _main_single_symbol(args)


def _main_multi_symbol(args: argparse.Namespace) -> int:
    """Multi-symbol path: per-symbol engines, synchronized bars, portfolio risk."""
    symbols = args.symbols
    paper_logger.info("Multi-symbol mode: %s", symbols)

    # Resolve and load data per symbol
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        attr = f"data_{sym}"
        path = getattr(args, attr, None)
        if path is None:
            paper_logger.error(
                "No data file for %s. Use --data-%s <path>", sym, sym,
            )
            return 1
        if not Path(path).exists():
            paper_logger.error("Data file not found: %s", path)
            return 1

        df = load_bars(path, start=args.start, end=args.end)
        paper_logger.info("Loaded %d bars for %s from %s", len(df), sym, path)

        if len(df) == 0:
            paper_logger.error("No bars loaded for %s", sym)
            return 1

        # Select last N trading days
        df.index = pd.to_datetime(df.index)
        unique_days = sorted(set(df.index.date))
        if args.days > 0 and len(unique_days) > args.days:
            target_days = unique_days[-args.days:]
            df = df[df.index.date >= target_days[0]]
            paper_logger.info(
                "  %s: last %d days: %s to %s (%d bars)",
                sym, args.days, target_days[0], target_days[-1], len(df),
            )
        bars_by_symbol[sym] = df

    # Build per-symbol pipelines + shared portfolio risk
    # Tradovate SIM adapter (optional)
    tradovate_adapter = None
    if args.tradovate_sim or args.tradovate_dry_run:
        from execution.tradovate_multi import (
            MultiSymbolTradovateAdapter,
            load_tradovate_config,
        )

        tov_config = load_tradovate_config()
        contract_overrides = {}
        for sym in symbols:
            attr = f"tradovate_contract_{sym}"
            override = getattr(args, attr, None)
            if override:
                contract_overrides[sym] = override

        tradovate_adapter = MultiSymbolTradovateAdapter(
            symbols=symbols,
            config=tov_config,
            contract_overrides=contract_overrides,
        )

        if args.tradovate_dry_run:
            paper_logger.info("Running Tradovate SIM dry-run validation...")
            results = tradovate_adapter.dry_run()
            for sym, info in results.items():
                if sym.startswith("_"):
                    continue
                paper_logger.info(
                    "  %s: %s %s",
                    sym, info.get("status"),
                    info.get("error", ""),
                )
            if results.get("_all_ok"):
                paper_logger.info("DRY RUN PASSED ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â all symbols validated")
            else:
                paper_logger.error("DRY RUN FAILED ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â fix issues above")
            return 0 if results.get("_all_ok") else 1

        # Connect for real execution
        paper_logger.info("Connecting to Tradovate SIM...")
        tradovate_adapter.connect()

    # Telegram alerter (optional ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â degrades gracefully)
    from strategy.telegram_alerts import create_alerter
    telegram_alerter = create_alerter(warn=True)

    router, portfolio_risk, pipelines = build_multi_pipeline(
        args, symbols,
        tradovate_adapter=tradovate_adapter,
        telegram_alerter=telegram_alerter,
    )

    # Log config
    paper_logger.info("--- CONFIRMED CONFIGURATION ---")
    paper_logger.info("  mode:              multi-symbol")
    paper_logger.info("  symbols:           %s", symbols)
    paper_logger.info("  multi-strategy:    %s", args.multi_strategy)
    paper_logger.info("  prop-mode:         %s", args.prop_mode)
    paper_logger.info("  ml-threshold:      %.2f", args.ml_threshold)
    paper_logger.info("  ml-model:          %s", args.ml_model)
    paper_logger.info("  ema-periods:       %s", args.ema_periods)
    paper_logger.info("  entry-types:       %s", args.entry_types)
    paper_logger.info("  max-trades/day:    %d (per symbol)", args.max_trades_per_day)
    paper_logger.info("  portfolio-concurrent: %d", args.portfolio_max_concurrent)
    paper_logger.info("  portfolio-same-dir:   %d", args.portfolio_max_same_dir)
    paper_logger.info("  correlation-divisor:  %.1f", args.correlation_divisor)
    paper_logger.info("  initial-capital:   $%.2f (total)", args.initial_capital)
    paper_logger.info(
        "  per-symbol capital: $%.2f",
        args.initial_capital / len(symbols),
    )
    if args.prop_mode:
        paper_logger.info("  prop-target:       $%.0f (total)", args.prop_target)
        paper_logger.info("  prop-max-dd:       $%.0f (total)", args.prop_max_dd)
    if tradovate_adapter is not None:
        paper_logger.info("  tradovate-sim:     ACTIVE (SIM ONLY)")
    paper_logger.info(
        "  telegram-alerts:   %s",
        "ENABLED" if telegram_alerter.enabled else "DISABLED",
    )
    paper_logger.info("-------------------------------")

    # Startup Telegram alert
    mode = "live-sim" if (tradovate_adapter is not None) else "paper"
    telegram_alerter.send_startup_alert(
        symbols=symbols, mode=mode, telegram_enabled=telegram_alerter.enabled,
    )

    # Run synchronized bar loop
    try:
        summary = run_paper_multi_symbol(
            bars_by_symbol, router, portfolio_risk, pipelines,
            bar_delay=args.bar_delay,
            telegram_alerter=telegram_alerter,
        )
    finally:
        # Always disconnect Tradovate on exit
        if tradovate_adapter is not None and tradovate_adapter.is_connected:
            paper_logger.info("Disconnecting from Tradovate SIM...")
            tradovate_adapter.disconnect()
            tradovate_adapter.export_trade_log()

    # Export trades
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary["all_trades"]:
        export_multi_trades(
            summary["all_trades"],
            str(out_dir / "multi_symbol_trades.csv"),
        )
        paper_logger.info(
            "Trades exported: results/multi_symbol_trades.csv (%d trades)",
            summary["total_trades"],
        )

    # Export per-symbol analytics
    for sym in symbols:
        pipe = pipelines[sym]
        pipe.analytics.export_json(str(out_dir / f"analytics_{sym.lower()}.json"))

    # Final report
    print_multi_report(summary)

    # Shutdown Telegram alert
    telegram_alerter.send_shutdown_alert()

    return 0


def _main_single_symbol(args: argparse.Namespace) -> int:
    """Original single-symbol path ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â unchanged behavior."""
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        paper_logger.error("Data file not found: %s", data_path)
        return 1

    bars = load_bars(str(data_path), start=args.start, end=args.end)
    paper_logger.info("Loaded %d bars from %s", len(bars), data_path.name)

    if len(bars) == 0:
        paper_logger.error("No bars loaded ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â check data file and date filters")
        return 1

    # Select last N trading days
    bars.index = pd.to_datetime(bars.index)
    unique_dates = bars.index.date
    trading_days = sorted(set(unique_dates))

    if args.days > 0 and len(trading_days) > args.days:
        target_days = trading_days[-args.days:]
        start_date = target_days[0]
        bars = bars[bars.index.date >= start_date]
        paper_logger.info(
            "Selected last %d trading days: %s to %s (%d bars)",
            args.days, target_days[0], target_days[-1], len(bars),
        )
    else:
        paper_logger.info("Using all %d trading days", len(trading_days))

    # Build pipeline
    tracker = LivePaperTracker()
    global _prop_trade_cursor
    _prop_trade_cursor = 0

    # Telegram alerter (optional ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â degrades gracefully)
    from strategy.telegram_alerts import create_alerter
    telegram_alerter = create_alerter(warn=True)

    pipeline = build_live_pipeline(args, tracker, telegram_alerter=telegram_alerter)
    paper_logger.info(
        "Pipeline: StrategyEngine -> %sRiskManager -> PaperEngine",
        "PropRiskGate -> " if pipeline.get("prop_gate") else "",
    )

    # Print confirmed configuration
    paper_logger.info("--- CONFIRMED CONFIGURATION ---")
    paper_logger.info("  multi-strategy:    %s", args.multi_strategy)
    paper_logger.info("  prop-mode:         %s", args.prop_mode)
    paper_logger.info("  ml-threshold:      %.2f", args.ml_threshold)
    paper_logger.info("  ml-model:          %s", args.ml_model)
    paper_logger.info("  ema-periods:       %s", args.ema_periods)
    paper_logger.info("  entry-types:       %s", args.entry_types)
    paper_logger.info("  selection-strategy: %s", args.selection_strategy)
    paper_logger.info("  max-trades/day:    %d", args.max_trades_per_day)
    paper_logger.info("  max-concurrent:    %d", args.max_concurrent)
    paper_logger.info("  initial-capital:   $%.2f", args.initial_capital)
    if args.multi_strategy:
        paper_logger.info("  min-quality-score: %.2f", args.min_quality_score)
        paper_logger.info("  intraday-cooldown: %d bars", args.intraday_cooldown)
        paper_logger.info("  max-intraday/bar:  %d", args.max_intraday_per_bar)
    if args.prop_mode:
        paper_logger.info("  prop-max-trades:   %d", args.prop_max_trades)
        paper_logger.info("  prop-allowed:      %s", args.prop_allowed_entries)
        paper_logger.info("  prop-target:       $%.0f", args.prop_target)
        paper_logger.info("  prop-max-dd:       $%.0f", args.prop_max_dd)
        paper_logger.info("  prop-daily-loss:   $%.0f", args.prop_daily_loss)
        paper_logger.info("  prop-min-ml-prob:  %.2f", args.prop_min_ml_prob)
    paper_logger.info(
        "  telegram-alerts:   %s",
        "ENABLED" if telegram_alerter.enabled else "DISABLED",
    )
    paper_logger.info("-------------------------------")

    # Startup Telegram alert
    telegram_alerter.send_startup_alert(
        symbols=["MES"], mode="paper", telegram_enabled=telegram_alerter.enabled,
    )

    # Run
    summary = run_paper_live(
        bars, pipeline, bar_delay=args.bar_delay,
        telegram_alerter=telegram_alerter,
    )

    # Export trades
    paper_engine: PaperEngine = pipeline["paper"]
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if paper_engine.trades:
        export_trades_csv(paper_engine.trades, str(out_dir / "paper_trades.csv"))
        paper_logger.info(
            "Trades exported: results/paper_trades.csv (%d trades)",
            len(paper_engine.trades),
        )

    # Export analytics
    analytics: AnalyticsEngine = pipeline["analytics"]
    analytics.export_json(str(out_dir / "analytics_paper.json"))

    # Final report
    print_final_report(summary)

    # Shutdown Telegram alert
    telegram_alerter.send_shutdown_alert()

    return 0


if __name__ == "__main__":
    sys.exit(main())
