"""
Backtest Engine for MES Futures
================================

Iterates bar-by-bar through historical data, feeds each bar to the
strategy, simulates fills with slippage/commissions, and tracks the
full position + equity lifecycle.

Design principles:
  - Strategy sees only current and past bars (no lookahead).
  - Fills are simulated at the signal price +/- slippage.
  - Commission is deducted on both entry and exit.
  - One position at a time (no pyramiding).
  - Results are pure data; plotting/export is done elsewhere.
"""

from __future__ import annotations
import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config.settings import InstrumentConfig, StrategyConfig, BacktestConfig, EvalConfig
from strategy.orb import ORBStrategy, SignalType, Signal

logger = logging.getLogger(__name__)


# Protocol: any strategy passed to the engine must have on_bar(bar) -> Signal and reset()
StrategyProtocol = object  # duck-typed — must have on_bar() and reset()


@dataclass
class EvalResult:
    """Prop-firm evaluation account result."""
    status: str = "INCOMPLETE"          # PASS, FAIL, INCOMPLETE
    fail_timestamp: Optional[str] = None
    pass_timestamp: Optional[str] = None
    peak_equity: float = 0.0
    trailing_threshold: float = 0.0
    distance_to_target: float = 0.0
    distance_to_fail: float = 0.0
    trades_taken: int = 0
    trading_days_used: int = 0


@dataclass
class Trade:
    """A completed round-trip trade."""
    entry_time: datetime.datetime
    exit_time: datetime.datetime
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    direction: str          # 'long' or 'short'
    pnl_points: float
    pnl_dollars: float
    commission: float
    slippage_cost: float
    net_pnl: float          # pnl_dollars - commission - slippage_cost
    exit_reason: str
    contracts: int
    position_size: float = 1.0  # ML-based sizing multiplier (0-1)
    strategy_type: str = ""     # e.g. "ema50_breakout" (multi-candidate mode)


@dataclass
class EquityPoint:
    """Snapshot of equity at a moment in time."""
    timestamp: datetime.datetime
    equity: float
    drawdown: float         # as a negative dollar amount from peak


@dataclass
class BacktestResult:
    """Everything produced by a single backtest run."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    final_equity: float = 0.0
    peak_equity: float = 0.0
    bar_count: int = 0
    eval_result: Optional[EvalResult] = None


class BacktestEngine:
    """
    Bar-by-bar backtest engine for futures strategies.

    Usage:
        engine = BacktestEngine(instrument, strategy_cfg, backtest_cfg)
        result = engine.run(bars_df)
    """

    def __init__(
        self,
        instrument: InstrumentConfig,
        strategy_config,
        backtest_config: BacktestConfig,
        strategy: object = None,
    ):
        self.inst = instrument
        self.strat_cfg = strategy_config
        self.bt_cfg = backtest_config

        if strategy is not None:
            self.strategy = strategy
        else:
            # Default to ORB for backward compatibility
            self.strategy = ORBStrategy(strategy_config)

    def run(self, bars: pd.DataFrame, eval_config: Optional[EvalConfig] = None) -> BacktestResult:
        """
        Run the backtest over the given bar DataFrame.

        Parameters
        ----------
        bars : pd.DataFrame
            Must have a tz-aware DatetimeIndex named 'timestamp' and
            columns: open, high, low, close, volume.
        eval_config : EvalConfig, optional
            If provided and enabled, tracks prop-firm evaluation rules
            and stops the simulation on PASS or FAIL.

        Returns
        -------
        BacktestResult
        """
        self.strategy.reset()

        result = BacktestResult()

        # Initial equity — eval mode can override starting capital
        eval_active = eval_config is not None and eval_config.enabled
        if eval_active:
            equity = eval_config.starting_capital
        else:
            equity = self.bt_cfg.initial_capital
        peak_equity = equity

        slippage_per_side = self.inst.tick_size * self.bt_cfg.slippage_ticks * self.inst.point_value
        commission_per_side = self.bt_cfg.commission_per_side
        contracts = self.inst.contract_size

        # Open positions dict: pos_id -> state dict
        # Supports both single-position (legacy) and multi-position strategies.
        open_positions: dict[str, dict] = {}

        # Eval mode state
        eval_result: Optional[EvalResult] = None
        eval_peak: float = equity
        eval_threshold: float = equity - (eval_config.max_drawdown if eval_active else 0)
        eval_target: float = equity + (eval_config.profit_target if eval_active else float('inf'))
        eval_trading_days: set = set()
        if eval_active:
            eval_result = EvalResult(
                peak_equity=equity,
                trailing_threshold=eval_threshold,
                distance_to_target=eval_config.profit_target,
                distance_to_fail=eval_config.max_drawdown,
            )

        for ts, row in bars.iterrows():
            bar = {
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }

            raw = self.strategy.on_bar(bar)
            if raw is None:
                logger.error(
                    "Strategy.on_bar() returned None at %s. "
                    "Expected Signal or list[Signal].", ts,
                )
                continue
            if isinstance(raw, list):
                signals = [s for s in raw if s.signal_type != SignalType.NONE]
            elif isinstance(raw, Signal):
                signals = [raw] if raw.signal_type != SignalType.NONE else []
            else:
                logger.error(
                    "Strategy.on_bar() returned unexpected type %s at %s",
                    type(raw).__name__, ts,
                )
                continue

            for signal in signals:
                if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
                    pos_id = signal.position_id or "_single"
                    pos_size = signal.position_size
                    e_slip = slippage_per_side * contracts * pos_size
                    e_comm = commission_per_side * contracts * pos_size
                    equity -= (e_slip + e_comm)

                    open_positions[pos_id] = {
                        "entry_price": signal.price,
                        "entry_time": signal.timestamp,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "direction": "long" if signal.signal_type == SignalType.LONG_ENTRY else "short",
                        "position_size": pos_size,
                        "entry_slip": e_slip,
                        "entry_comm": e_comm,
                        "strategy_type": getattr(signal, 'strategy_type', ''),
                    }
                    logger.debug(
                        "ENTRY %s %s @ %.2f (sl=%.2f tp=%.2f size=%.2f) [%s]",
                        pos_id,
                        "long" if signal.signal_type == SignalType.LONG_ENTRY else "short",
                        signal.price, signal.stop_loss or 0.0,
                        signal.take_profit or 0.0, pos_size,
                        getattr(signal, 'strategy_type', ''),
                    )

                elif signal.signal_type in (
                    SignalType.EXIT_TP,
                    SignalType.EXIT_SL,
                    SignalType.EXIT_EOD,
                ):
                    pos_id = signal.position_id if signal.position_id else None
                    if not pos_id:
                        # Legacy single-position: close whatever is open
                        if open_positions:
                            pos_id = next(iter(open_positions))
                        else:
                            continue

                    if pos_id not in open_positions:
                        continue

                    pos = open_positions.pop(pos_id)
                    fill_price = signal.price
                    pos_size = pos["position_size"]
                    exit_slip = slippage_per_side * contracts * pos_size
                    exit_comm = commission_per_side * contracts * pos_size

                    if pos["direction"] == "short":
                        pnl_points = pos["entry_price"] - fill_price
                    else:
                        pnl_points = fill_price - pos["entry_price"]
                    pnl_dollars = pnl_points * self.inst.point_value * contracts * pos_size
                    total_commission = pos["entry_comm"] + exit_comm
                    total_slippage = pos["entry_slip"] + exit_slip

                    equity += pnl_dollars - exit_slip - exit_comm
                    logger.info(
                        "EXIT %s %s @ %.2f -> %.2f  pnl=%.2f [%s]",
                        pos_id, _exit_reason(signal.signal_type),
                        pos["entry_price"], fill_price, pnl_dollars,
                        pos.get("strategy_type", ""),
                    )

                    trade = Trade(
                        entry_time=pos["entry_time"],
                        exit_time=signal.timestamp,
                        entry_price=pos["entry_price"],
                        exit_price=fill_price,
                        stop_loss=pos["sl"] or 0.0,
                        take_profit=pos["tp"] or 0.0,
                        direction=pos["direction"] or "long",
                        pnl_points=pnl_points,
                        pnl_dollars=pnl_dollars,
                        commission=total_commission,
                        slippage_cost=total_slippage,
                        net_pnl=pnl_dollars - total_slippage - total_commission,
                        exit_reason=_exit_reason(signal.signal_type),
                        contracts=contracts,
                        position_size=pos_size,
                        strategy_type=pos.get("strategy_type", ""),
                    )
                    result.trades.append(trade)

            # Track equity
            peak_equity = max(peak_equity, equity)
            dd = equity - peak_equity  # negative or zero

            result.equity_curve.append(EquityPoint(
                timestamp=ts,
                equity=equity,
                drawdown=dd,
            ))

            result.bar_count += 1

            # --- Eval mode: check pass/fail with mark-to-market equity ---
            if eval_active:
                eval_trading_days.add(ts.date())

                # Mark-to-market: include unrealized P&L of ALL open positions
                mtm_equity = equity
                cp = float(row["close"])
                for pos in open_positions.values():
                    if pos["direction"] == "long":
                        mtm_equity += (cp - pos["entry_price"]) * self.inst.point_value * contracts * pos["position_size"]
                    else:
                        mtm_equity += (pos["entry_price"] - cp) * self.inst.point_value * contracts * pos["position_size"]

                eval_peak = max(eval_peak, mtm_equity)
                eval_threshold = eval_peak - eval_config.max_drawdown

                # Check PASS
                if mtm_equity >= eval_target:
                    # Force close ALL open positions
                    for pid in list(open_positions):
                        p = open_positions.pop(pid)
                        equity = self._force_close(
                            result, equity, p["entry_price"], p["entry_time"],
                            p["sl"], p["tp"], p["direction"], cp,
                            ts, slippage_per_side, commission_per_side,
                            contracts, p["entry_slip"], p["entry_comm"], "Eval PASS",
                            self.inst.point_value, p["position_size"],
                            p.get("strategy_type", ""))
                    eval_result.status = "PASS"
                    eval_result.pass_timestamp = ts.isoformat()
                    eval_result.peak_equity = eval_peak
                    eval_result.trailing_threshold = eval_threshold
                    eval_result.distance_to_target = 0
                    eval_result.distance_to_fail = mtm_equity - eval_threshold
                    eval_result.trades_taken = len(result.trades)
                    eval_result.trading_days_used = len(eval_trading_days)
                    break

                # Check FAIL
                if mtm_equity <= eval_threshold:
                    for pid in list(open_positions):
                        p = open_positions.pop(pid)
                        equity = self._force_close(
                            result, equity, p["entry_price"], p["entry_time"],
                            p["sl"], p["tp"], p["direction"], cp,
                            ts, slippage_per_side, commission_per_side,
                            contracts, p["entry_slip"], p["entry_comm"], "Eval FAIL",
                            self.inst.point_value, p["position_size"],
                            p.get("strategy_type", ""))
                    eval_result.status = "FAIL"
                    eval_result.fail_timestamp = ts.isoformat()
                    eval_result.peak_equity = eval_peak
                    eval_result.trailing_threshold = eval_threshold
                    eval_result.distance_to_target = eval_target - mtm_equity
                    eval_result.distance_to_fail = 0
                    eval_result.trades_taken = len(result.trades)
                    eval_result.trading_days_used = len(eval_trading_days)
                    break

        # Finalize
        if eval_active and eval_result is not None and eval_result.status == "INCOMPLETE":
            mtm_equity = equity
            if open_positions:
                cp = float(bars.iloc[-1]["close"])
                for pos in open_positions.values():
                    if pos["direction"] == "long":
                        mtm_equity += (cp - pos["entry_price"]) * self.inst.point_value * contracts * pos["position_size"]
                    else:
                        mtm_equity += (pos["entry_price"] - cp) * self.inst.point_value * contracts * pos["position_size"]
            eval_result.peak_equity = eval_peak
            eval_result.trailing_threshold = eval_threshold
            eval_result.distance_to_target = eval_target - mtm_equity
            eval_result.distance_to_fail = mtm_equity - eval_threshold
            eval_result.trades_taken = len(result.trades)
            eval_result.trading_days_used = len(eval_trading_days)

        result.final_equity = equity
        result.peak_equity = peak_equity
        result.eval_result = eval_result
        logger.info(
            "Backtest complete: %d bars, %d trades, final equity $%.2f",
            result.bar_count, len(result.trades), result.final_equity,
        )
        return result

    @staticmethod
    def _force_close(result, equity, entry_price, entry_time, sl, tp,
                     direction, close_price, ts, slip_per_side, comm_per_side,
                     contracts, entry_slip, entry_comm, reason, point_value,
                     position_size=1.0, strategy_type=""):
        """Force-close an open position (used by eval mode)."""
        exit_slip = slip_per_side * contracts * position_size
        exit_comm = comm_per_side * contracts * position_size
        if direction == "short":
            pnl_points = entry_price - close_price
        else:
            pnl_points = close_price - entry_price
        pnl_dollars = pnl_points * point_value * contracts * position_size
        equity += pnl_dollars - exit_slip - exit_comm
        total_commission = entry_comm + exit_comm
        total_slippage = entry_slip + exit_slip
        trade = Trade(
            entry_time=entry_time,
            exit_time=ts,
            entry_price=entry_price,
            exit_price=close_price,
            stop_loss=sl or 0.0,
            take_profit=tp or 0.0,
            direction=direction or "long",
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            commission=total_commission,
            slippage_cost=total_slippage,
            net_pnl=pnl_dollars - total_slippage - total_commission,
            exit_reason=reason,
            contracts=contracts,
            position_size=position_size,
            strategy_type=strategy_type,
        )
        result.trades.append(trade)
        return equity


def _exit_reason(sig_type: SignalType) -> str:
    return {
        SignalType.EXIT_TP: "Take Profit",
        SignalType.EXIT_SL: "Stop Loss",
        SignalType.EXIT_EOD: "End of Day",
    }.get(sig_type, "Unknown")
