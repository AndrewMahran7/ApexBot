"""
Paper Trading Execution Engine
===============================

Consumes LiveSignal objects from StrategyEngine and simulates
trade execution with slippage, commissions, and full position
lifecycle tracking.

This is the bridge between the strategy layer (which produces
signals) and a live broker (which this module replaces with a
simulation).  Every field on every trade is computed identically
to backtest/engine.py so results are directly comparable.

Usage (replay mode):
    engine = StrategyEngine(config, on_signal=paper.on_signal)
    for bar in bars:
        paper.on_bar(bar)          # mark-to-market before signals
        engine.on_bar(bar)         # generates signals → paper.on_signal

Usage (live mode):
    engine = StrategyEngine(config, on_signal=paper.on_signal)
    for bar in live_feed:
        paper.on_bar(bar)
        engine.on_bar(bar)
        for update in paper.pending_updates():
            send_to_dashboard(update)
"""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from backtest.engine import Trade, EquityPoint
from config.settings import InstrumentConfig, BacktestConfig, compute_contracts
from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class PaperConfig:
    """Paper trading cost model and settings."""
    slippage_ticks: float = 1.0
    commission_per_side: float = 0.62
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01   # 1% of equity risked per trade
    max_contracts: int = 5         # hard cap on contract count

    @classmethod
    def from_backtest_config(cls, bt: BacktestConfig) -> "PaperConfig":
        """Create PaperConfig mirroring backtest cost model."""
        return cls(
            slippage_ticks=bt.slippage_ticks,
            commission_per_side=bt.commission_per_side,
            initial_capital=bt.initial_capital,
        )


# ------------------------------------------------------------------
# PnL update pushed to subscribers
# ------------------------------------------------------------------

@dataclass
class PnLUpdate:
    """Real-time PnL snapshot emitted after every state change."""
    timestamp: datetime.datetime
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    open_position_count: int
    drawdown: float  # negative dollar amount from peak


# ------------------------------------------------------------------
# Comparison with backtest
# ------------------------------------------------------------------

@dataclass
class PaperValidationResult:
    """Comparison of paper trades against backtest trades."""
    total_paper: int = 0
    total_backtest: int = 0
    matched: int = 0
    pnl_mismatches: list[dict] = field(default_factory=list)
    extra_paper: list[Trade] = field(default_factory=list)
    missing_paper: list[Trade] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            len(self.pnl_mismatches) == 0
            and len(self.extra_paper) == 0
            and len(self.missing_paper) == 0
        )


# ------------------------------------------------------------------
# Paper Engine
# ------------------------------------------------------------------

class PaperEngine:
    """
    Simulates trade execution from LiveSignal objects.

    Mirrors the fill logic of backtest/engine.py (slippage, commission,
    PnL calculation) so paper results are directly comparable.

    Parameters
    ----------
    instrument : InstrumentConfig
        Contract specification (tick_size, point_value, contract_size).
    config : PaperConfig
        Cost model and initial capital.
    on_update : callable, optional
        If provided, called with a PnLUpdate after every state change
        (entry, exit, mark-to-market).
    """

    def __init__(
        self,
        instrument: InstrumentConfig,
        config: PaperConfig | None = None,
        on_update: Optional[Callable[[PnLUpdate], None]] = None,
    ) -> None:
        self._inst = instrument
        self._cfg = config or PaperConfig()
        self._on_update = on_update

        self._equity: float = self._cfg.initial_capital
        self._peak_equity: float = self._equity
        self._realized_pnl: float = 0.0

        # pos_id -> position state dict
        self._open_positions: dict[str, dict] = {}

        self._trades: list[Trade] = []
        self._equity_curve: list[EquityPoint] = []
        self._pending_updates: list[PnLUpdate] = []

        self._last_bar: Optional[dict] = None
        self._bar_count: int = 0

        logger.info(
            "PaperEngine initialised: capital=%.2f, slip_ticks=%.1f, "
            "comm=%.2f, instrument=%s (tick_value=$%.2f), "
            "risk_per_trade=%.1f%%, max_contracts=%d",
            self._cfg.initial_capital,
            self._cfg.slippage_ticks,
            self._cfg.commission_per_side,
            self._inst.symbol,
            self._inst.tick_value,
            self._cfg.risk_per_trade * 100,
            self._cfg.max_contracts,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_bar(self, bar: dict) -> None:
        """
        Process a new bar for mark-to-market equity tracking.

        Call this BEFORE StrategyEngine.on_bar() so the equity curve
        reflects the latest market price before new signals arrive.
        """
        self._last_bar = bar
        self._bar_count += 1

        close = float(bar["close"])
        ts = bar["timestamp"]

        unrealized = self._unrealized_pnl(close)
        mtm_equity = self._equity + unrealized
        self._peak_equity = max(self._peak_equity, mtm_equity)
        dd = mtm_equity - self._peak_equity

        self._equity_curve.append(EquityPoint(
            timestamp=ts,
            equity=mtm_equity,
            drawdown=dd,
        ))

        logger.debug(
            "Bar %d: close=%.2f mtm_equity=%.2f drawdown=%.2f "
            "unrealized=%.2f open_positions=%d",
            self._bar_count, close, mtm_equity, dd, unrealized,
            len(self._open_positions),
        )

        update = PnLUpdate(
            timestamp=ts,
            equity=mtm_equity,
            unrealized_pnl=unrealized,
            realized_pnl=self._realized_pnl,
            open_position_count=len(self._open_positions),
            drawdown=dd,
        )
        self._pending_updates.append(update)

        if self._on_update is not None:
            try:
                self._on_update(update)
            except Exception as e:
                logger.error(
                    "on_update callback error at %s: %s",
                    ts, e, exc_info=True,
                )

    def on_signal(self, signal: LiveSignal) -> None:
        """
        Process a signal from the strategy engine.

        Wire this as the ``on_signal`` callback of StrategyEngine:
            engine = StrategyEngine(cfg, on_signal=paper.on_signal)
        """
        if signal.is_entry:
            self._handle_entry(signal)
        elif signal.is_exit:
            self._handle_exit(signal)
        else:
            logger.warning(
                "Ignoring signal with unknown type %s at %s",
                signal.signal_type, signal.timestamp,
            )

    def pending_updates(self) -> list[PnLUpdate]:
        """Drain and return all pending PnL updates since last call."""
        updates = list(self._pending_updates)
        self._pending_updates.clear()
        return updates

    def reset(self) -> None:
        """Reset all state for a new session."""
        if self._open_positions:
            logger.warning(
                "Reset with %d open positions — discarding",
                len(self._open_positions),
            )
        self._equity = self._cfg.initial_capital
        self._peak_equity = self._equity
        self._realized_pnl = 0.0
        self._open_positions.clear()
        self._trades.clear()
        self._equity_curve.clear()
        self._pending_updates.clear()
        self._last_bar = None
        self._bar_count = 0
        logger.info("PaperEngine reset")

    # ---- Read-only properties -----------------------------------------

    @property
    def equity(self) -> float:
        """Current realised equity (excludes unrealized PnL)."""
        return self._equity

    @property
    def mark_to_market_equity(self) -> float:
        """Equity including unrealized PnL from open positions."""
        if self._last_bar is None:
            return self._equity
        return self._equity + self._unrealized_pnl(float(self._last_bar["close"]))

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def open_positions(self) -> dict[str, dict]:
        """Read-only copy of open positions."""
        return dict(self._open_positions)

    @property
    def trades(self) -> list[Trade]:
        """All completed trades (read-only copy)."""
        return list(self._trades)

    @property
    def equity_curve(self) -> list[EquityPoint]:
        """Full equity curve (read-only copy)."""
        return list(self._equity_curve)

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    # ------------------------------------------------------------------
    # Entry / Exit simulation
    # ------------------------------------------------------------------

    def _handle_entry(self, sig: LiveSignal) -> None:
        """Simulate a trade entry with slippage + commission."""
        pos_id = sig.position_id or "_single"

        if pos_id in self._open_positions:
            logger.warning(
                "Duplicate entry for position %s at %s — ignoring",
                pos_id, sig.timestamp,
            )
            return

        # Risk-based contract sizing
        stop_distance = abs(sig.entry - sig.stop) if sig.stop else 0.0
        stop_ticks = stop_distance / self._inst.tick_size if self._inst.tick_size > 0 else 0.0
        contracts = compute_contracts(
            equity=self._equity + self._unrealized_pnl(
                float(self._last_bar["close"]) if self._last_bar else sig.entry,
            ),
            risk_per_trade=self._cfg.risk_per_trade,
            stop_ticks=stop_ticks,
            tick_value=self._inst.tick_value,
            max_contracts=self._cfg.max_contracts,
        )

        # Per-contract costs
        entry_slip = self._cfg.slippage_ticks * self._inst.tick_value * contracts
        entry_comm = self._cfg.commission_per_side * contracts

        # Deduct entry costs from equity immediately
        self._equity -= (entry_slip + entry_comm)

        self._open_positions[pos_id] = {
            "entry_price": sig.entry,
            "entry_time": sig.timestamp,
            "direction": sig.direction,
            "sl": sig.stop,
            "tp": sig.take_profit,
            "contracts": contracts,
            "position_size": sig.position_size,
            "entry_slip": entry_slip,
            "entry_comm": entry_comm,
            "strategy_type": sig.strategy_type,
        }

        logger.info(
            "PAPER ENTRY %s %s contracts=%d @ %.2f (sl=%.2f tp=%.2f) [%s]",
            pos_id, sig.direction, contracts, sig.entry,
            sig.stop, sig.take_profit,
            sig.strategy_type,
        )

        self._emit_update(sig.timestamp)

    def _handle_exit(self, sig: LiveSignal) -> None:
        """Simulate a trade exit with slippage + commission."""
        pos_id = sig.position_id or None

        if not pos_id:
            # Legacy single-position: close whatever is open
            if self._open_positions:
                pos_id = next(iter(self._open_positions))
            else:
                logger.warning(
                    "Exit signal at %s but no open positions", sig.timestamp,
                )
                return

        if pos_id not in self._open_positions:
            logger.warning(
                "Exit for unknown position %s at %s", pos_id, sig.timestamp,
            )
            return

        pos = self._open_positions.pop(pos_id)
        contracts = pos["contracts"]
        fill_price = sig.entry  # exit price carried in LiveSignal.entry

        exit_slip = self._cfg.slippage_ticks * self._inst.tick_value * contracts
        exit_comm = self._cfg.commission_per_side * contracts

        if pos["direction"] == "short":
            ticks = (pos["entry_price"] - fill_price) / self._inst.tick_size
        else:
            ticks = (fill_price - pos["entry_price"]) / self._inst.tick_size

        pnl_points = ticks * self._inst.tick_size  # price-space move
        pnl_dollars = ticks * self._inst.tick_value * contracts
        total_commission = pos["entry_comm"] + exit_comm
        total_slippage = pos["entry_slip"] + exit_slip
        net_pnl = pnl_dollars - total_slippage - total_commission

        self._equity += pnl_dollars - exit_slip - exit_comm
        self._realized_pnl += net_pnl

        exit_reason = self._exit_type_label(sig.signal_type)

        trade = Trade(
            entry_time=pos["entry_time"],
            exit_time=sig.timestamp,
            entry_price=pos["entry_price"],
            exit_price=fill_price,
            stop_loss=pos["sl"],
            take_profit=pos["tp"],
            direction=pos["direction"],
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            commission=total_commission,
            slippage_cost=total_slippage,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            contracts=contracts,
            position_size=pos["position_size"],
            strategy_type=pos["strategy_type"],
        )
        self._trades.append(trade)

        logger.info(
            "PAPER EXIT %s %s contracts=%d @ %.2f -> %.2f  "
            "ticks=%.0f pnl=$%.2f net=$%.2f [%s]",
            pos_id, exit_reason, contracts,
            pos["entry_price"], fill_price, ticks, pnl_dollars, net_pnl,
            pos["strategy_type"],
        )

        self._emit_update(sig.timestamp)

    # ------------------------------------------------------------------
    # Validation: compare paper trades vs backtest trades
    # ------------------------------------------------------------------

    def compare_with_backtest(
        self,
        backtest_trades: list[Trade],
        *,
        pnl_tolerance: float = 0.01,
    ) -> PaperValidationResult:
        """
        Compare paper trades against backtest trades.

        Matches by (entry_time, direction, strategy_type).  Reports
        mismatches, extra paper trades, and missing trades.

        Parameters
        ----------
        backtest_trades : list[Trade]
            Trades from backtest/engine.py.
        pnl_tolerance : float
            Maximum absolute difference in net_pnl before flagging.

        Returns
        -------
        PaperValidationResult
        """
        result = PaperValidationResult(
            total_paper=len(self._trades),
            total_backtest=len(backtest_trades),
        )

        # Use list-based lookup to avoid key collisions when multiple
        # positions share (entry_time, direction, strategy_type).
        bt_lookup: dict[tuple, list[Trade]] = defaultdict(list)
        for t in backtest_trades:
            key = (t.entry_time, t.direction, t.strategy_type)
            bt_lookup[key].append(t)

        paper_lookup: dict[tuple, list[Trade]] = defaultdict(list)
        for t in self._trades:
            key = (t.entry_time, t.direction, t.strategy_type)
            paper_lookup[key].append(t)

        all_keys = set(bt_lookup.keys()) | set(paper_lookup.keys())

        for key in sorted(all_keys, key=lambda k: k[0]):
            bt_list = bt_lookup.get(key, [])
            pt_list = paper_lookup.get(key, [])

            # Match pairwise by order; extras go to missing/extra
            pairs = max(len(bt_list), len(pt_list))
            for i in range(pairs):
                bt = bt_list[i] if i < len(bt_list) else None
                pt = pt_list[i] if i < len(pt_list) else None

                if pt is None:
                    result.missing_paper.append(bt)
                    continue
                if bt is None:
                    result.extra_paper.append(pt)
                    continue

                result.matched += 1

                for fld in ("entry_price", "exit_price", "pnl_points",
                            "pnl_dollars", "net_pnl", "commission",
                            "slippage_cost", "position_size"):
                    pv = getattr(pt, fld)
                    bv = getattr(bt, fld)
                    if abs(pv - bv) > pnl_tolerance:
                        result.pnl_mismatches.append({
                            "entry_time": key[0],
                            "direction": key[1],
                            "strategy_type": key[2],
                            "field": fld,
                            "paper": pv,
                            "backtest": bv,
                            "diff": pv - bv,
                        })

        if result.passed:
            logger.info(
                "Paper validation PASSED: %d trades matched",
                result.matched,
            )
        else:
            logger.warning(
                "Paper validation FAILED: %d matched, %d mismatches, "
                "%d extra, %d missing",
                result.matched, len(result.pnl_mismatches),
                len(result.extra_paper), len(result.missing_paper),
            )

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unrealized_pnl(self, current_price: float) -> float:
        """Sum unrealized PnL across all open positions."""
        total = 0.0
        for pos in self._open_positions.values():
            if pos["direction"] == "long":
                ticks = (current_price - pos["entry_price"]) / self._inst.tick_size
            else:
                ticks = (pos["entry_price"] - current_price) / self._inst.tick_size
            total += ticks * self._inst.tick_value * pos["contracts"]
        return total

    def _emit_update(self, ts: datetime.datetime) -> None:
        """Push a PnL update after a state change."""
        if self._last_bar is None:
            logger.warning(
                "Cannot emit update before first bar at %s; skipping", ts,
            )
            return

        close = float(self._last_bar["close"])
        unrealized = self._unrealized_pnl(close)
        mtm = self._equity + unrealized
        self._peak_equity = max(self._peak_equity, mtm)

        update = PnLUpdate(
            timestamp=ts,
            equity=mtm,
            unrealized_pnl=unrealized,
            realized_pnl=self._realized_pnl,
            open_position_count=len(self._open_positions),
            drawdown=mtm - self._peak_equity,
        )
        self._pending_updates.append(update)

        if self._on_update is not None:
            try:
                self._on_update(update)
            except Exception as e:
                logger.error(
                    "on_update callback error at %s: %s",
                    ts, e, exc_info=True,
                )

    @staticmethod
    def _exit_type_label(sig_type: SignalType) -> str:
        return {
            SignalType.EXIT_TP: "Take Profit",
            SignalType.EXIT_SL: "Stop Loss",
            SignalType.EXIT_EOD: "End of Day",
        }.get(sig_type, "Unknown")
