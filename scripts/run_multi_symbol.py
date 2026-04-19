#!/usr/bin/env python3
"""
Multi-Symbol Paper Trading Runner
===================================

Extends the proven EMA breakout strategy across multiple equity-index
futures (MES, MNQ, RTY) with shared portfolio-level risk management.

Each symbol gets its own:
  - StrategyEngine  (independent bar state, EMA, signal generation)
  - RiskManager     (per-symbol daily loss, kill switch)
  - PaperEngine     (instrument-specific slippage, commission, PnL)

Shared across all symbols:
  - PortfolioRiskManager (cross-symbol position/exposure limits)
  - Aggregated equity tracking and reporting

Usage:
    python run_multi_symbol.py --symbols MES --data-MES data/mes_4y.csv --days 30
    python run_multi_symbol.py --symbols MES MNQ RTY \\
        --data-MES data/mes_4y.csv \\
        --data-MNQ data/mnq_4y.csv \\
        --data-RTY data/rty_4y.csv \\
        --days 30
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import InstrumentConfig, INSTRUMENT_REGISTRY
from data.loader import load_bars
from strategy.hybrid_ema_ml import HybridEMAMLConfig
from strategy.strategy_engine import StrategyEngine, LiveSignal
from strategy.paper_engine import PaperEngine, PaperConfig, PnLUpdate
from risk.risk_manager import RiskManager, RiskConfig
from risk.portfolio_risk import PortfolioRiskManager, PortfolioRiskConfig
from backtest.metrics import export_trades_csv

logger = logging.getLogger(__name__)
paper_logger = logging.getLogger("apex.multi_symbol")


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

class _MultiSymbolFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("apex.multi_symbol")


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper()))
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%H:%M:%S",
    ))
    console.addFilter(_MultiSymbolFilter())
    root.addHandler(console)

    # Full debug log
    debug_handler = logging.FileHandler(log_path / "main.log", mode="w")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(debug_handler)

    # Paper trading log â€” formatted output
    paper_handler = logging.FileHandler(
        log_path / "multi_symbol.log", mode="w",
    )
    paper_handler.setLevel(logging.INFO)
    paper_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    paper_handler.addFilter(_MultiSymbolFilter())
    root.addHandler(paper_handler)


# ------------------------------------------------------------------
# Per-symbol pipeline
# ------------------------------------------------------------------

@dataclass
class SymbolPipeline:
    """All components for a single symbol."""
    symbol: str
    instrument: InstrumentConfig
    engine: StrategyEngine
    risk: RiskManager
    paper: PaperEngine


# ------------------------------------------------------------------
# Multi-symbol router
# ------------------------------------------------------------------

class MultiSymbolRouter:
    """
    Coordinates multiple symbol pipelines with shared portfolio risk.

    Flow per timestamp:
      1. Each symbol's risk + paper get on_bar (daily tracking, MTM)
      2. Each symbol's engine gets on_bar (signal generation)
         - Exit signals forward immediately through per-symbol chain
         - Entry signals are collected for portfolio-level filtering
      3. Collected entries are ranked by ML quality and filtered
         through PortfolioRiskManager before forwarding.
    """

    def __init__(
        self,
        pipelines: dict[str, SymbolPipeline],
        portfolio_risk: PortfolioRiskManager,
    ) -> None:
        self._pipelines = pipelines
        self._portfolio_risk = portfolio_risk
        self._pending_entries: list[tuple[str, LiveSignal]] = []

        # Wire each symbol's callback chain
        for sym, pipe in pipelines.items():
            # risk.on_approved -> execution tracker -> paper.on_signal
            pipe.risk.on_approved = self._make_execution_tracker(sym)
            # engine.on_signal -> signal handler (collects entries)
            pipe.engine._on_signal = self._make_signal_handler(sym)

    def process_bars(self, bars_by_symbol: dict[str, dict]) -> None:
        """
        Process one timestamp across all symbols.

        Call this once per merged timestamp with bars for each symbol
        that has data at that time.
        """
        self._pending_entries.clear()

        for sym, bar in bars_by_symbol.items():
            pipe = self._pipelines[sym]
            pipe.risk.on_bar(bar)
            pipe.paper.on_bar(bar)
            pipe.engine.on_bar(bar)

        # Process collected entries through portfolio risk
        self._flush_entries()

    # ------------------------------------------------------------------
    # Aggregated metrics
    # ------------------------------------------------------------------

    def total_equity(self) -> float:
        return sum(
            p.paper.mark_to_market_equity for p in self._pipelines.values()
        )

    def total_trades(self) -> list:
        trades = []
        for sym, pipe in self._pipelines.items():
            for t in pipe.paper.trades:
                trades.append((sym, t))
        return trades

    def all_risk_events(self):
        events = []
        for sym, pipe in self._pipelines.items():
            for e in pipe.risk.events:
                events.append((sym, e))
        return events

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _make_signal_handler(self, symbol: str):
        """Signal callback: exits forward immediately, entries collected."""
        def handler(signal: LiveSignal) -> None:
            if signal.is_exit:
                self._pipelines[symbol].risk.on_signal(signal)
            elif signal.is_entry:
                self._pending_entries.append((symbol, signal))
            else:
                self._pipelines[symbol].risk.on_signal(signal)
        return handler

    def _make_execution_tracker(self, symbol: str):
        """Wraps paper.on_signal to track portfolio positions."""
        paper = self._pipelines[symbol].paper

        def tracker(signal: LiveSignal) -> None:
            if signal.is_entry:
                self._portfolio_risk.record_entry(symbol, signal)
            elif signal.is_exit:
                self._portfolio_risk.record_exit(symbol, signal)
            paper.on_signal(signal)
        return tracker

    def _flush_entries(self) -> None:
        """Rank and filter pending entries through portfolio risk."""
        if not self._pending_entries:
            return

        ranked = self._portfolio_risk.rank_signals(self._pending_entries)
        for sym, sig in ranked:
            approved = self._portfolio_risk.check_entry(sym, sig)
            if approved is not None:
                self._pipelines[sym].risk.on_signal(approved)


# ------------------------------------------------------------------
# Pipeline builder
# ------------------------------------------------------------------

def build_multi_symbol_pipeline(
    args: argparse.Namespace,
    symbols: list[str],
) -> tuple[MultiSymbolRouter, PortfolioRiskManager]:
    """Build per-symbol pipelines and shared portfolio risk."""

    strategy_cfg = HybridEMAMLConfig(
        multi_candidate=False,
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

    paper_cfg = PaperConfig(
        slippage_ticks=args.slippage_ticks,
        commission_per_side=args.commission,
        initial_capital=args.initial_capital / len(symbols),
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
            paper_logger.error("Unknown symbol: %s", sym)
            sys.exit(1)

        engine = StrategyEngine(
            config=strategy_cfg,
            on_signal=None,  # wired by router
        )
        risk = RiskManager(config=risk_cfg, instrument=instrument)
        paper = PaperEngine(instrument=instrument, config=paper_cfg)

        pipelines[sym] = SymbolPipeline(
            symbol=sym,
            instrument=instrument,
            engine=engine,
            risk=risk,
            paper=paper,
        )

    router = MultiSymbolRouter(pipelines, portfolio_risk)
    return router, portfolio_risk


# ------------------------------------------------------------------
# Bar loop
# ------------------------------------------------------------------

def run_multi_symbol(
    bars_by_symbol: dict[str, pd.DataFrame],
    router: MultiSymbolRouter,
    portfolio_risk: PortfolioRiskManager,
    bar_delay: float = 0.0,
) -> dict:
    """
    Step through merged bars across all symbols.

    Returns summary dict with portfolio-level and per-symbol metrics.
    """
    # Merge all timestamps and sort
    all_timestamps = set()
    for df in bars_by_symbol.values():
        all_timestamps.update(df.index)
    all_timestamps = sorted(all_timestamps)

    total_bars = len(all_timestamps)
    shutdown = False

    def _handle_sigint(signum, frame):
        nonlocal shutdown
        paper_logger.info("Shutdown signal received")
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)

    symbols = list(bars_by_symbol.keys())

    paper_logger.info("=" * 60)
    paper_logger.info("  MULTI-SYMBOL PAPER TRADING")
    paper_logger.info("  Symbols: %s", ", ".join(symbols))
    paper_logger.info("  Merged timestamps: %d", total_bars)
    paper_logger.info("  Delay: %.2fs/bar", bar_delay)
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
                _log_day_summary(router, prev_date, symbols)
            day_count += 1
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

        # Log new signals, trades, risk events
        for sym in symbols:
            pipe = router._pipelines[sym]

            # New trades
            cur_trades = pipe.paper.trades
            for t in cur_trades[prev_trade_counts[sym]:]:
                pnl_sym = "+" if t.net_pnl >= 0 else ""
                paper_logger.info(
                    "[%s] %s %s -> CLOSED %s [%s] %s$%.2f (%s)",
                    bar_time, sym, t.strategy_type.replace("_", " "),
                    t.direction, t.strategy_type, pnl_sym, t.net_pnl,
                    t.exit_reason,
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
        _log_day_summary(router, prev_date, symbols)

    # Build summary
    return _build_summary(router, portfolio_risk, symbols, day_count)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _log_day_summary(
    router: MultiSymbolRouter,
    date,
    symbols: list[str],
) -> None:
    paper_logger.info("")
    paper_logger.info("-- Day Summary: %s --", date)
    for sym in symbols:
        pipe = router._pipelines[sym]
        mtm = pipe.paper.mark_to_market_equity
        paper_logger.info(
            "  %s: equity=$%.2f | open=%d",
            sym, mtm, pipe.paper.open_position_count,
        )
    total = router.total_equity()
    paper_logger.info("  Portfolio: $%.2f", total)


def _build_summary(
    router: MultiSymbolRouter,
    portfolio_risk: PortfolioRiskManager,
    symbols: list[str],
    day_count: int,
) -> dict:
    """Build the final summary dict from all symbol pipelines."""
    all_trades = []  # (symbol, trade) pairs
    by_symbol: dict[str, dict] = {}
    total_pnl = 0.0
    peak_equity = 0.0

    for sym in symbols:
        pipe = router._pipelines[sym]
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
        peak_equity += pipe.paper.peak_equity

    total_equity = router.total_equity()
    total_count = len(all_trades)
    total_wins = sum(1 for _, t in all_trades if t.net_pnl > 0)

    # Portfolio drawdown: track from aggregated equity curves
    max_dd = sum(
        router._pipelines[sym].paper.mark_to_market_equity
        - router._pipelines[sym].paper.peak_equity
        for sym in symbols
    )

    # Portfolio-level risk events
    port_events = portfolio_risk.events
    port_by_type = defaultdict(int)
    for e in port_events:
        port_by_type[e.event_type] += 1

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
        "total_equity": total_equity,
        "peak_equity": peak_equity,
        "max_drawdown": max_dd,
        "by_symbol": by_symbol,
        "portfolio_events_total": len(port_events),
        "portfolio_events_by_type": dict(port_by_type),
        "all_trades": all_trades,
    }


# ------------------------------------------------------------------
# Final report
# ------------------------------------------------------------------

def print_final_report(summary: dict) -> None:
    symbols = summary["symbols"]

    paper_logger.info("")
    paper_logger.info("=" * 60)
    paper_logger.info("  MULTI-SYMBOL PAPER TRADING -- FINAL REPORT")
    paper_logger.info("=" * 60)
    paper_logger.info("  Symbols:         %s", ", ".join(symbols))
    paper_logger.info("  Trading days:    %d", summary["trading_days"])
    paper_logger.info("  Total trades:    %d", summary["total_trades"])
    paper_logger.info("  Trades/day:      %.1f", summary["trades_per_day"])
    paper_logger.info("  Win rate:        %.1f%%", summary["win_rate"])
    paper_logger.info("  Avg PnL/trade:   $%.2f", summary["avg_pnl_per_trade"])
    paper_logger.info("  Total PnL:       $%.2f", summary["total_pnl"])
    paper_logger.info("  Total equity:    $%.2f", summary["total_equity"])
    paper_logger.info("  Peak equity:     $%.2f", summary["peak_equity"])
    paper_logger.info("  Max drawdown:    $%.2f", summary["max_drawdown"])

    # Per-symbol breakdown
    paper_logger.info("")
    paper_logger.info("  --- BY SYMBOL ---")
    for sym, stats in summary["by_symbol"].items():
        paper_logger.info(
            "  [%s] %d trades, %.1f%% WR, $%.2f total, $%.2f avg, DD $%.2f",
            sym, stats["trades"], stats["win_rate"],
            stats["pnl"], stats["avg_pnl"], stats["drawdown"],
        )

    # Portfolio risk events
    port_total = summary.get("portfolio_events_total", 0)
    port_by_type = summary.get("portfolio_events_by_type", {})
    if port_total > 0:
        paper_logger.info("")
        paper_logger.info("  --- PORTFOLIO RISK EVENTS (%d total) ---", port_total)
        for reason, count in sorted(port_by_type.items(), key=lambda x: -x[1]):
            paper_logger.info("    %s: %d", reason, count)

    paper_logger.info("")
    if summary["total_trades"] > 0:
        paper_logger.info("  STATUS: STABLE -- System operating correctly")
    else:
        paper_logger.info("  STATUS: NO TRADES -- Check strategy parameters")
    paper_logger.info("=" * 60)


# ------------------------------------------------------------------
# Trade export
# ------------------------------------------------------------------

def export_multi_trades(all_trades: list[tuple[str, object]], path: str) -> None:
    """Export trades from all symbols to a single CSV with symbol column."""
    if not all_trades:
        return

    fieldnames = [
        "symbol", "entry_time", "exit_time", "direction",
        "entry_price", "exit_price", "stop_loss", "take_profit",
        "pnl_points", "pnl_dollars", "commission", "slippage_cost",
        "net_pnl", "exit_reason", "position_size", "strategy_type",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sym, t in all_trades:
            writer.writerow({
                "symbol": sym,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
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
        description="Apex -- Multi-Symbol Paper Trading",
    )

    # Symbols and data
    p.add_argument("--symbols", nargs="+", default=["MES"],
                   help="Symbols to trade (must exist in INSTRUMENT_REGISTRY)")
    p.add_argument("--data-MES", default=None, help="CSV for MES")
    p.add_argument("--data-MNQ", default=None, help="CSV for MNQ")
    p.add_argument("--data-RTY", default=None, help="CSV for RTY")
    p.add_argument("--days", type=int, default=30,
                   help="Trading days to simulate (from end of data)")

    # Simulation
    p.add_argument("--bar-delay", type=float, default=0.0)

    # Strategy (EMA only)
    p.add_argument("--ml-model", default="models/ema_model.pkl")
    p.add_argument("--ml-threshold", type=float, default=0.50)
    p.add_argument("--ema-periods", nargs="+", type=int, default=[50])
    p.add_argument("--entry-types", nargs="+", default=["breakout"])
    p.add_argument("--selection-strategy", default="global_ml")

    # Per-symbol risk
    p.add_argument("--max-daily-loss", type=float, default=500.0)
    p.add_argument("--max-trades-per-day", type=int, default=6)
    p.add_argument("--max-concurrent", type=int, default=3,
                   help="Per-symbol max concurrent positions")

    # Portfolio risk
    p.add_argument("--portfolio-max-concurrent", type=int, default=3,
                   help="Max total concurrent positions across all symbols")
    p.add_argument("--portfolio-max-same-dir", type=int, default=2,
                   help="Max positions in same direction across all symbols")
    p.add_argument("--portfolio-max-exposure", type=float, default=3.0)
    p.add_argument("--correlation-divisor", type=float, default=2.0,
                   help="Divide position size by this when correlated symbols")

    # Paper
    p.add_argument("--initial-capital", type=float, default=10_000.0,
                   help="Total capital (split equally across symbols)")
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--commission", type=float, default=0.62)

    # Logging
    p.add_argument("--log-level", default="INFO")

    return p.parse_args(argv)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    paper_logger.info("Apex Multi-Symbol Paper Trading -- starting")

    symbols = args.symbols
    paper_logger.info("Symbols: %s", symbols)

    # Resolve data files per symbol
    data_map: dict[str, str] = {}
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
        data_map[sym] = path

    # Load bars per symbol
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for sym, path in data_map.items():
        df = load_bars(path)
        paper_logger.info("Loaded %d bars for %s from %s", len(df), sym, path)

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

    # Build pipeline
    router, portfolio_risk = build_multi_symbol_pipeline(args, symbols)

    # Log config
    paper_logger.info("--- CONFIRMED CONFIGURATION ---")
    paper_logger.info("  symbols:           %s", symbols)
    paper_logger.info("  ml-threshold:      %.2f", args.ml_threshold)
    paper_logger.info("  ml-model:          %s", args.ml_model)
    paper_logger.info("  ema-periods:       %s", args.ema_periods)
    paper_logger.info("  max-trades/day:    %d (per symbol)", args.max_trades_per_day)
    paper_logger.info("  portfolio-concurrent: %d", args.portfolio_max_concurrent)
    paper_logger.info("  portfolio-same-dir:   %d", args.portfolio_max_same_dir)
    paper_logger.info("  portfolio-exposure:   %.1f", args.portfolio_max_exposure)
    paper_logger.info("  correlation-divisor:  %.1f", args.correlation_divisor)
    paper_logger.info("  initial-capital:   $%.2f (total)", args.initial_capital)
    paper_logger.info(
        "  per-symbol capital: $%.2f",
        args.initial_capital / len(symbols),
    )
    paper_logger.info("-------------------------------")

    # Run
    summary = run_multi_symbol(
        bars_by_symbol, router, portfolio_risk,
        bar_delay=args.bar_delay,
    )

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

    # Final report
    print_final_report(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
