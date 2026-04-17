#!/usr/bin/env python3
"""
Apex — Main System Runner
===========================

Wires the full pipeline:
    data → StrategyEngine → RiskManager → Execution (Paper | Tradovate)

Modes:
    replay  — Feed historical CSV bars through the pipeline.
    paper   — Same as replay but with PaperEngine tracking execution.
    live    — Connect to Tradovate and feed bars from a live source.

Usage:
    python main.py --mode replay --data data/mes_5m.csv
    python main.py --mode paper  --data data/mes_5m.csv
    python main.py --mode live
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import InstrumentConfig, BacktestConfig
from data.loader import load_bars
from strategy.hybrid_ema_ml import HybridEMAMLConfig
from strategy.strategy_engine import StrategyEngine, LiveSignal
from strategy.multi_strategy_engine import MultiStrategyEngine
from strategy.intraday_strategies import IntradayConfig
from strategy.paper_engine import PaperEngine, PaperConfig, PnLUpdate
from strategy.risk_manager import RiskManager, RiskConfig
from strategy.tradovate_client import TradovateClient, TradovateConfig
from dashboard.state import DashboardState
from analytics.engine import AnalyticsEngine
from strategy.prop_challenge import PropConfig, PropRiskGate

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configure root logger with separate handlers for trades, signals, errors."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — user-specified level
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper()))
    console.setFormatter(fmt)
    root.addHandler(console)

    # File — all messages
    all_handler = logging.FileHandler(log_path / "main.log", mode="w")
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(fmt)
    root.addHandler(all_handler)

    # Trades — entries and exits only
    trade_handler = logging.FileHandler(log_path / "trades.log", mode="w")
    trade_handler.setLevel(logging.INFO)
    trade_handler.setFormatter(fmt)
    trade_handler.addFilter(_NameFilter("apex.trades"))
    root.addHandler(trade_handler)

    # Signals — all strategy signals
    signal_handler = logging.FileHandler(log_path / "signals.log", mode="w")
    signal_handler.setLevel(logging.INFO)
    signal_handler.setFormatter(fmt)
    signal_handler.addFilter(_NameFilter("apex.signals"))
    root.addHandler(signal_handler)

    # Errors — WARNING and above
    error_handler = logging.FileHandler(log_path / "errors.log", mode="w")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(fmt)
    root.addHandler(error_handler)


class _NameFilter(logging.Filter):
    """Pass only records whose logger name starts with a prefix."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self._prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefix)


# Dedicated loggers for structured output
trade_logger = logging.getLogger("apex.trades")
signal_logger = logging.getLogger("apex.signals")


# ------------------------------------------------------------------
# Signal / PnL callbacks
# ------------------------------------------------------------------

def _log_signal(sig: LiveSignal) -> None:
    """Log every signal to the signals log."""
    signal_logger.info(
        "%s %s %s @ %.2f  sl=%.2f tp=%.2f size=%.2f [%s] %s",
        sig.timestamp, sig.signal_type.name, sig.direction,
        sig.entry, sig.stop, sig.take_profit, sig.position_size,
        sig.strategy_type, sig.reason,
    )


def _log_trade(sig: LiveSignal) -> None:
    """Log entries and exits to the trades log."""
    if sig.is_entry:
        trade_logger.info(
            "ENTRY %s %s @ %.2f  sl=%.2f tp=%.2f size=%.2f [%s]",
            sig.direction, sig.signal_type.name, sig.entry,
            sig.stop, sig.take_profit, sig.position_size,
            sig.strategy_type,
        )
    elif sig.is_exit:
        trade_logger.info(
            "EXIT %s @ %.2f [%s] %s",
            sig.signal_type.name, sig.entry,
            sig.strategy_type, sig.reason,
        )


def _log_pnl(update: PnLUpdate) -> None:
    """Log PnL updates at debug level."""
    logger.debug(
        "PnL: equity=%.2f realized=%.2f unrealized=%.2f "
        "drawdown=%.2f open=%d",
        update.equity, update.realized_pnl, update.unrealized_pnl,
        update.drawdown, update.open_position_count,
    )


# ------------------------------------------------------------------
# Pipeline builder
# ------------------------------------------------------------------

def build_pipeline(
    mode: str,
    instrument: InstrumentConfig,
    strategy_cfg: HybridEMAMLConfig,
    risk_cfg: RiskConfig,
    paper_cfg: PaperConfig | None = None,
    tradovate_cfg: TradovateConfig | None = None,
    dashboard_state: DashboardState | None = None,
    analytics: AnalyticsEngine | None = None,
    prop_gate: PropRiskGate | None = None,
    **kwargs,
) -> dict:
    """
    Wire the signal pipeline for the requested mode.

    Returns a dict with keys: engine, risk, paper (or None), tradovate (or None),
    dashboard_state (or None), analytics (or None), prop_gate (or None).
    """
    paper: Optional[PaperEngine] = None
    tradovate: Optional[TradovateClient] = None

    # Risk manager is always in the chain
    risk = RiskManager(config=risk_cfg, instrument=instrument)

    if mode in ("replay", "paper"):
        pnl_callback = _log_pnl
        if dashboard_state is not None:
            pnl_callback = _make_pnl_callback(_log_pnl, dashboard_state)
        paper = PaperEngine(
            instrument=instrument,
            config=paper_cfg or PaperConfig(),
            on_update=pnl_callback,
        )
        risk.on_approved = _make_execution_callback(paper.on_signal)

    elif mode == "live":
        if tradovate_cfg is None:
            raise ValueError("--mode live requires Tradovate configuration")
        tradovate = TradovateClient(
            instrument=instrument,
            config=tradovate_cfg,
        )
        risk.on_approved = _make_execution_callback(tradovate.on_signal)

    # Strategy engine emits into risk manager (prop gate sits in front if enabled)
    risk_entry = risk.on_signal
    if prop_gate is not None:
        prop_gate.on_approved = risk.on_signal
        risk_entry = prop_gate.on_signal

    signal_cb = _make_signal_callback(risk_entry)
    if dashboard_state is not None:
        signal_cb = _make_dashboard_signal_callback(signal_cb, dashboard_state)
    if analytics is not None:
        signal_cb = _make_analytics_signal_callback(signal_cb, analytics)

    # Use MultiStrategyEngine when intraday strategies are enabled
    intraday_cfg = kwargs.get("intraday_cfg")
    enable_hybrid = kwargs.get("enable_hybrid", True)
    max_intra_bar = kwargs.get("max_intraday_per_bar", 2)
    if intraday_cfg is not None:
        engine = MultiStrategyEngine(
            strategy_cfg=strategy_cfg,
            intraday_cfg=intraday_cfg,
            on_signal=signal_cb,
            enable_hybrid=enable_hybrid,
            enable_intraday=True,
            max_intraday_entries_per_bar=max_intra_bar,
        )
    else:
        engine = StrategyEngine(
            config=strategy_cfg,
            on_signal=signal_cb,
        )

    return {
        "engine": engine,
        "risk": risk,
        "paper": paper,
        "tradovate": tradovate,
        "dashboard_state": dashboard_state,
        "analytics": analytics,
        "prop_gate": prop_gate,
    }


def _make_signal_callback(downstream):
    """Wrap downstream handler with signal logging."""

    def _cb(sig: LiveSignal) -> None:
        _log_signal(sig)
        downstream(sig)

    return _cb


def _make_execution_callback(downstream):
    """Wrap downstream handler with trade logging."""

    def _cb(sig: LiveSignal) -> None:
        _log_trade(sig)
        downstream(sig)

    return _cb


def _make_pnl_callback(log_fn, dashboard_state: DashboardState):
    """Wrap PnL logging with dashboard state update."""

    def _cb(update: PnLUpdate) -> None:
        log_fn(update)
        dashboard_state.on_pnl(update)

    return _cb


def _make_dashboard_signal_callback(downstream, dashboard_state: DashboardState):
    """Wrap signal callback with dashboard state update."""

    def _cb(sig: LiveSignal) -> None:
        dashboard_state.on_signal(sig)
        downstream(sig)

    return _cb


def _make_analytics_signal_callback(downstream, analytics: AnalyticsEngine):
    """Wrap signal callback with analytics recording."""

    def _cb(sig: LiveSignal) -> None:
        analytics.record_signal(sig)
        downstream(sig)

    return _cb


# ------------------------------------------------------------------
# Bar loop
# ------------------------------------------------------------------

def run_bar_loop(
    bars: pd.DataFrame,
    pipeline: dict,
) -> None:
    """Feed bars through the pipeline (replay / paper modes)."""
    engine: StrategyEngine = pipeline["engine"]
    risk: RiskManager = pipeline["risk"]
    paper: Optional[PaperEngine] = pipeline["paper"]

    total = len(bars)
    logger.info("Starting bar loop: %d bars", total)

    for i, (ts, row) in enumerate(bars.iterrows()):
        bar = {
            "timestamp": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

        # Order matters: prop → risk → paper → engine
        prop: Optional[PropRiskGate] = pipeline.get("prop_gate")
        if prop is not None:
            prop.on_bar(bar)
        risk.on_bar(bar)
        if paper is not None:
            paper.on_bar(bar)
        engine.on_bar(bar)

        # Track trade closes for prop consecutive-loss detection
        if prop is not None and paper is not None:
            _sync_prop_trades(prop, paper)

        # Push completed trades and position state to dashboard
        dash: Optional[DashboardState] = pipeline.get("dashboard_state")
        if dash is not None and paper is not None:
            _sync_dashboard(dash, paper, risk)

        # Push completed trades and risk decisions to analytics
        ana: Optional[AnalyticsEngine] = pipeline.get("analytics")
        if ana is not None and paper is not None:
            _sync_analytics(ana, paper, risk)

        if (i + 1) % 10_000 == 0:
            logger.info("Processed %d / %d bars (%.1f%%)", i + 1, total, (i + 1) / total * 100)

    logger.info("Bar loop complete: %d bars processed", total)


def _sync_dashboard(
    dash: DashboardState,
    paper: PaperEngine,
    risk: RiskManager,
) -> None:
    """Push latest trade/position/risk state to dashboard."""
    # Push newly completed trades (compare counts)
    paper_trades = paper.trades
    dash_count = dash.trade_count()
    for t in paper_trades[dash_count:]:
        dash.on_trade(t)

    # Push risk events
    risk_events = risk.events
    risk_count_in_dash = dash._risk_events_count  # already tracked
    for e in risk_events[risk_count_in_dash:]:
        dash.on_risk_event(e)

    dash.update_open_positions(paper)
    dash.update_risk_state(risk)


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


_prop_trade_cursor: int = 0


def _sync_prop_trades(
    prop: PropRiskGate,
    paper: PaperEngine,
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


def run_live(pipeline: dict) -> None:
    """
    Run the live trading loop with Tradovate.

    Connects to Tradovate, starts state sync, then enters a polling
    loop that feeds bars from the live source.
    """
    tradovate: TradovateClient = pipeline["tradovate"]
    risk: RiskManager = pipeline["risk"]
    engine: StrategyEngine = pipeline["engine"]

    # Graceful shutdown
    shutdown = False

    def _handle_sigint(signum, frame):
        nonlocal shutdown
        logger.info("Shutdown signal received, closing gracefully...")
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        tradovate.connect()
        tradovate.start_sync()
        logger.info("Live mode active — waiting for bars (Ctrl+C to stop)")

        # In a full implementation, this would consume from a live bar feed
        # (WebSocket, polling, or Databento stream). For now, we idle and
        # log that the pipeline is wired and ready.
        while not shutdown:
            time.sleep(1)

    except Exception:
        logger.exception("Fatal error in live loop")
        raise
    finally:
        logger.info("Shutting down live pipeline...")
        tradovate.disconnect()
        logger.info("Live pipeline stopped")


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def print_summary(pipeline: dict, mode: str) -> None:
    """Print end-of-run summary statistics."""
    risk: RiskManager = pipeline["risk"]
    paper: Optional[PaperEngine] = pipeline["paper"]

    print("\n" + "=" * 60)
    print(f"  Apex Run Summary — mode={mode}")
    print("=" * 60)

    # Risk events
    events = risk.events
    blocked = sum(1 for e in events if e.event_type == "blocked")
    capped = sum(1 for e in events if e.event_type == "capped")
    kills = sum(1 for e in events if e.event_type == "kill_switch")
    print(f"  Risk events: {len(events)} total "
          f"({blocked} blocked, {capped} capped, {kills} kill-switch)")

    if paper is not None:
        trades = paper.trades
        equity = paper.equity
        peak = paper.peak_equity
        dd = equity - peak

        wins = sum(1 for t in trades if t.net_pnl > 0)
        total = len(trades)
        win_rate = (wins / total * 100) if total > 0 else 0.0
        total_pnl = sum(t.net_pnl for t in trades)

        print(f"  Trades: {total} ({wins}W / {total - wins}L, {win_rate:.1f}% win rate)")
        print(f"  PnL: ${total_pnl:,.2f}")
        print(f"  Equity: ${equity:,.2f} (peak ${peak:,.2f}, DD ${dd:,.2f})")

    print("=" * 60 + "\n")


def _print_prop_summary(prop: PropRiskGate) -> None:
    """Print prop challenge result to console."""
    t = prop.tracker

    print("\n" + "=" * 60)
    print("  Prop Challenge Result")
    print("=" * 60)

    if t.passed:
        print("  OUTCOME: PASSED")
    elif t.failed:
        print("  OUTCOME: FAILED")
    else:
        print("  OUTCOME: INCOMPLETE")

    print(f"  Equity:           ${t.current_equity:,.2f}")
    print(f"  Peak equity:      ${t.peak_equity:,.2f}")
    print(f"  Gain:             ${t.equity_gain:,.2f}")
    print(f"  Trailing DD lvl:  ${t.trailing_dd_level:,.2f}")
    print(f"  DD buffer:        ${t.dd_buffer_remaining:,.2f}")

    events = prop.events
    blocked = sum(1 for e in events if "blocked" in e.event_type or "filter" in e.event_type)
    stopped = sum(1 for e in events if e.event_type == "prop_stopped_day")
    print(f"  Prop events:      {len(events)} "
          f"({blocked} blocked, {stopped} day-stops)")

    print("=" * 60 + "\n")


def _print_analytics(analytics: AnalyticsEngine) -> None:
    """Print analytics report to console."""
    r = analytics.report()

    print("\n" + "=" * 60)
    print("  Analytics Report")
    print("=" * 60)
    print(f"  Total trades: {r.total_trades} "
          f"({r.win_count}W / {r.loss_count}L, "
          f"{r.win_rate:.1f}% win rate)")
    print(f"  Profit factor: {r.profit_factor:.2f}")
    print(f"  Sharpe ratio:  {r.sharpe_ratio:.2f}")
    print(f"  Total PnL:     ${r.total_pnl:,.2f}")
    print(f"  Max drawdown:  ${r.max_drawdown:,.2f}")
    print(f"  Trading days:  {r.trading_days}")
    print(f"  Signals: {r.total_signals}  Decisions: "
          f"{r.risk_blocked + r.risk_capped + r.risk_kill_switches}")
    if r.total_signals > 0:
        print(f"  Signal->Trade rate: {r.signal_to_trade_rate:.1f}%")

    for s in r.by_strategy:
        print(f"  [{s['strategy_type']}] {s['trade_count']} trades, "
              f"{s['win_rate']:.1f}% WR, PF {s['profit_factor']:.2f}, "
              f"PnL ${s['total_pnl']:,.2f}")

    print("=" * 60 + "\n")


# ------------------------------------------------------------------
# Dashboard server
# ------------------------------------------------------------------

def _start_dashboard_server(state: DashboardState, port: int) -> None:
    """Start the FastAPI dashboard in a background daemon thread."""
    import threading
    import uvicorn
    from dashboard.app import create_app

    app = create_app(state)

    def _run():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True, name="dashboard-server")
    thread.start()
    logger.info("Dashboard started at http://127.0.0.1:%d", port)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apex — MES Futures Trading System Runner",
    )

    # Mode
    p.add_argument(
        "--mode",
        required=True,
        choices=["replay", "paper", "live"],
        help="Operating mode: replay (signals only), paper (simulated execution), live (Tradovate)",
    )

    # Data
    p.add_argument("--data", default="data/mes_5m.csv", help="Path to OHLCV CSV/Parquet file")
    p.add_argument("--start", default=None, help="Start date filter (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date filter (YYYY-MM-DD)")

    # Strategy
    p.add_argument("--ml-model", default="models/ema_model.pkl", help="Path to ML model")
    p.add_argument("--ml-threshold", type=float, default=0.6, help="ML probability threshold")
    p.add_argument("--ema-periods", nargs="+", type=int, default=[50], help="EMA periods")
    p.add_argument("--entry-types", nargs="+", default=["breakout"], help="Entry types")
    p.add_argument("--selection-strategy", default="global_ml",
                   choices=["global_ml", "priority", "priority_ml_sizing"])

    # Risk
    p.add_argument("--max-daily-loss", type=float, default=500.0)
    p.add_argument("--max-trades-per-day", type=int, default=6)
    p.add_argument("--max-concurrent", type=int, default=3)

    # Paper
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--commission", type=float, default=0.62)

    # Logging
    p.add_argument("--log-dir", default="logs", help="Directory for log files")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Dashboard
    p.add_argument("--dashboard", action="store_true", help="Enable web dashboard")
    p.add_argument("--dashboard-port", type=int, default=8501, help="Dashboard port")

    # Analytics
    p.add_argument("--analytics", action="store_true", help="Enable analytics collection")
    p.add_argument("--analytics-output", default="results/analytics.json",
                   help="Path for analytics JSON report")

    # Prop challenge mode
    p.add_argument("--prop-mode", action="store_true",
                   help="Enable prop firm challenge mode")
    p.add_argument("--prop-target", type=float, default=1500.0,
                   help="Profit target in dollars")
    p.add_argument("--prop-max-dd", type=float, default=1000.0,
                   help="Max trailing drawdown in dollars")
    p.add_argument("--prop-daily-loss", type=float, default=300.0,
                   help="Daily loss limit in prop mode")
    p.add_argument("--prop-daily-lock", type=float, default=400.0,
                   help="Daily profit lock in prop mode")
    p.add_argument("--prop-max-trades", type=int, default=4,
                   help="Max trades per day in prop mode")
    p.add_argument("--prop-allowed-entries", nargs="+",
                   default=["breakout", "vwap_bounce", "intraday_momentum", "mean_reversion"],
                   help="Allowed entry types in prop mode (e.g. breakout momentum)")
    p.add_argument("--prop-consecutive-losses", type=int, default=5,
                   help="Max consecutive losses before halting for the day")
    p.add_argument("--prop-min-ml-prob", type=float, default=0.55,
                   help="Minimum ML probability for prop mode entry")

    # Multi-strategy (intraday)
    p.add_argument("--multi-strategy", action="store_true",
                   help="Enable intraday strategies (VWAP Bounce, Momentum, Mean Reversion)")
    p.add_argument("--disable-hybrid", action="store_true",
                   help="Disable the original EMA/ML strategy in multi-strategy mode")
    p.add_argument("--vwap-band-mult", type=float, default=0.3,
                   help="VWAP band width in ATR multiples")
    p.add_argument("--momentum-lookback", type=int, default=3,
                   help="Bars for intraday momentum range")
    p.add_argument("--rsi-oversold", type=float, default=30.0,
                   help="RSI oversold threshold for mean reversion")
    p.add_argument("--rsi-overbought", type=float, default=70.0,
                   help="RSI overbought threshold for mean reversion")
    p.add_argument("--intraday-entry-start", default="10:00",
                   help="Earliest intraday entry time (HH:MM ET)")
    p.add_argument("--intraday-entry-end", default="15:30",
                   help="Latest intraday entry time (HH:MM ET)")
    p.add_argument("--intraday-cooldown", type=int, default=6,
                   help="Min bars between intraday entries per strategy")
    p.add_argument("--min-quality-score", type=float, default=0.5,
                   help="Minimum quality score for intraday entries (0.0-1.0)")
    p.add_argument("--max-intraday-per-bar", type=int, default=2,
                   help="Max intraday entries per bar in multi-strategy mode")

    return p.parse_args(argv)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    setup_logging(log_dir=args.log_dir, level=args.log_level)
    logger.info("Apex starting — mode=%s", args.mode)

    # --- Configs ---
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

    tradovate_cfg = TradovateConfig() if args.mode == "live" else None

    # --- Dashboard ---
    dashboard_state = DashboardState() if args.dashboard else None

    # --- Analytics ---
    analytics = (
        AnalyticsEngine(initial_capital=args.initial_capital)
        if args.analytics else None
    )

    # --- Prop challenge ---
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
        # Override risk config to match prop constraints
        risk_cfg = RiskConfig(
            max_daily_loss=args.prop_daily_loss,
            max_trades_per_day=args.prop_max_trades,
            max_concurrent_positions=args.max_concurrent,
            max_per_direction=2,
        )
        # Force prop capital
        paper_cfg = PaperConfig(
            slippage_ticks=args.slippage_ticks,
            commission_per_side=args.commission,
            initial_capital=args.initial_capital,
        )
        logger.info(
            "PROP MODE: target=+$%.0f dd=-$%.0f daily_loss=$%.0f",
            prop_cfg.profit_target, prop_cfg.max_drawdown,
            prop_cfg.daily_loss_limit,
        )

    # --- Build pipeline ---
    # Reset prop trade cursor
    global _prop_trade_cursor
    _prop_trade_cursor = 0

    # Build intraday config if multi-strategy is enabled
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
        logger.info(
            "MULTI-STRATEGY: hybrid=%s, vwap_band=%.2f, momentum_lb=%d, "
            "rsi=[%.0f,%.0f], entry=%s-%s",
            enable_hybrid, args.vwap_band_mult, args.momentum_lookback,
            args.rsi_oversold, args.rsi_overbought,
            args.intraday_entry_start, args.intraday_entry_end,
        )

    pipeline = build_pipeline(
        mode=args.mode,
        instrument=instrument,
        strategy_cfg=strategy_cfg,
        risk_cfg=risk_cfg,
        paper_cfg=paper_cfg,
        tradovate_cfg=tradovate_cfg,
        dashboard_state=dashboard_state,
        analytics=analytics,
        prop_gate=prop_gate,
        intraday_cfg=intraday_cfg,
        enable_hybrid=enable_hybrid,
        max_intraday_per_bar=getattr(args, 'max_intraday_per_bar', 2),
    )

    # Wire prop gate equity callback now that paper engine exists
    if prop_gate is not None and pipeline.get("paper") is not None:
        paper_ref = pipeline["paper"]
        prop_gate._get_equity = lambda: paper_ref.mark_to_market_equity

    # Start dashboard server in background
    if args.dashboard:
        _start_dashboard_server(dashboard_state, args.dashboard_port)

    logger.info(
        "Pipeline wired: StrategyEngine → RiskManager → %s",
        "TradovateClient" if args.mode == "live" else "PaperEngine",
    )

    # --- Run ---
    if args.mode in ("replay", "paper"):
        data_path = Path(args.data)
        if not data_path.exists():
            logger.error("Data file not found: %s", data_path)
            return 1

        bars = load_bars(str(data_path), start=args.start, end=args.end)
        logger.info("Loaded %d bars from %s", len(bars), data_path.name)

        run_bar_loop(bars, pipeline)
        print_summary(pipeline, args.mode)

        # Export paper trades to CSV
        paper_engine: Optional[PaperEngine] = pipeline.get("paper")
        if paper_engine is not None and paper_engine.trades:
            from backtest.metrics import export_trades_csv
            out_dir = Path("results")
            out_dir.mkdir(parents=True, exist_ok=True)
            export_trades_csv(paper_engine.trades, str(out_dir / "trades.csv"))
            logger.info("Paper trades exported to results/trades.csv (%d trades)",
                        len(paper_engine.trades))

        # Prop challenge summary
        if prop_gate is not None:
            _print_prop_summary(prop_gate)

        # Export analytics report
        if analytics is not None:
            analytics.export_json(args.analytics_output)
            _print_analytics(analytics)

    elif args.mode == "live":
        run_live(pipeline)

    logger.info("Apex finished — mode=%s", args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
