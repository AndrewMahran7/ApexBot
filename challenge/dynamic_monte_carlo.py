"""
Dynamic-Sizing Monte Carlo Simulation
=======================================

Extends the base Monte Carlo engine with account-state-aware sizing.
Instead of applying a flat scale factor to every trade, uses the
PropRiskLayer to dynamically adjust position size based on equity
progress, drawdown proximity, losing streaks, and profit-lock rules.

The base ("fixed-size") simulation lives in challenge/monte_carlo.py.
This module wraps it to add the dynamic layer for comparison.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from strategy.prop_risk_layer import (
    PropRiskConfig,
    PropRiskLayer,
    AccountMode,
)
from challenge.monte_carlo import (
    TradeRecord,
    SimPath,
    TradeSampler,
    MonteCarloResult,
    DrawdownStats,
    SensitivityPoint,
    _extract_pnls,
    _aggregate_paths,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic-sizing path simulation
# ---------------------------------------------------------------------------

@dataclass
class DynamicSimPath(SimPath):
    """Extended SimPath with sizing diagnostics."""
    avg_size: float = 0.0
    trades_blocked: int = 0
    block_reasons: dict = field(default_factory=dict)


def _simulate_dynamic_path(
    sampled_pnls: np.ndarray,
    risk_config: PropRiskConfig,
    dd_buffer: float = 200.0,
    record_path: bool = False,
    symbols: Optional[np.ndarray] = None,
) -> DynamicSimPath:
    """Simulate one equity path with dynamic sizing from PropRiskLayer.

    Each trade PnL is scaled by the risk layer's per-trade decision.
    Trades may also be BLOCKED entirely (size = 0).
    """
    layer = PropRiskLayer(risk_config)

    max_dd_seen = 0.0
    largest_win = 0.0
    largest_loss = 0.0
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    near_boundary_count = 0

    equity = risk_config.starting_capital
    peak_equity = equity
    dd_floor = equity - risk_config.max_drawdown
    target = equity + risk_config.profit_target

    path = [equity] if record_path else []
    outcome = "incomplete"
    trades_taken = 0
    trades_blocked = 0
    total_size = 0.0
    block_reasons: dict[str, int] = {}

    # Simple daily boundary heuristic: reset daily counters every ~3 trades
    # (average ~1-3 trades per session for this strategy)
    trades_per_day_est = 3

    for i in range(len(sampled_pnls)):
        raw_pnl = float(sampled_pnls[i])
        sym = str(symbols[i]) if symbols is not None else ""

        # Simple day boundary: reset daily counters periodically
        if trades_taken > 0 and trades_taken % trades_per_day_est == 0:
            layer.reset_day()

        # Ask risk layer for sizing decision
        decision = layer.evaluate_trade(symbol=sym)

        if decision.blocked:
            trades_blocked += 1
            for r in decision.reasons:
                block_reasons[r] = block_reasons.get(r, 0) + 1
            # Trade is skipped: no PnL, but we still count it as an "attempt"
            # The sampled PnL is consumed but not applied.
            continue

        # Apply dynamic sizing
        size = decision.size_mult
        scaled_pnl = raw_pnl * size
        total_size += size

        # Record trade in layer (updates equity, streak, etc.)
        layer.record_trade(scaled_pnl, symbol=sym)

        equity = layer._equity
        peak_equity = layer._peak_equity
        dd_floor = layer._trailing_dd_limit
        trades_taken += 1

        # Track max drawdown
        dd = peak_equity - equity
        if dd > max_dd_seen:
            max_dd_seen = dd

        # Near-boundary tracking
        if dd_buffer > 0 and (equity - dd_floor) <= dd_buffer:
            near_boundary_count += 1

        # Win/loss tracking (before scaling for streak stat consistency)
        if scaled_pnl > 0:
            largest_win = max(largest_win, scaled_pnl)
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_win_streak = max(max_win_streak, current_streak)
        elif scaled_pnl < 0:
            largest_loss = min(largest_loss, scaled_pnl)
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_loss_streak = max(max_loss_streak, abs(current_streak))

        if record_path:
            path.append(equity)

        # Termination checks
        if layer.passed:
            outcome = "passed"
            break
        if layer.failed:
            outcome = "failed"
            break

    avg_size = total_size / trades_taken if trades_taken > 0 else 0.0

    return DynamicSimPath(
        outcome=outcome,
        trades_taken=trades_taken,
        final_equity=equity,
        peak_equity=peak_equity,
        max_drawdown=max_dd_seen,
        largest_win=largest_win,
        largest_loss=abs(largest_loss),
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        equity_path=path,
        avg_size=avg_size,
        trades_blocked=trades_blocked,
        block_reasons=block_reasons,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dynamic_monte_carlo(
    trades: Sequence[TradeRecord] | Sequence[float] | np.ndarray,
    risk_config: PropRiskConfig,
    *,
    n_simulations: int = 2000,
    max_trades: int = 500,
    dd_buffer: float = 200.0,
    block_size: int = 7,
    n_sample_paths: int = 20,
    seed: Optional[int] = None,
    strategy_filter: Optional[str] = None,
) -> MonteCarloResult:
    """Run Monte Carlo with dynamic sizing from PropRiskLayer.

    Same interface as challenge.monte_carlo.run_monte_carlo() but
    applies account-state-aware sizing per trade instead of flat scaling.
    """
    pnls = _extract_pnls(trades, strategy_filter)
    if len(pnls) == 0:
        raise ValueError("No trades to simulate (after filtering)")

    wins = pnls[pnls > 0]
    source_win_rate = len(wins) / len(pnls) if len(pnls) else 0.0
    source_mean = float(np.mean(pnls))
    source_std = float(np.std(pnls)) if len(pnls) > 1 else 0.0

    rng = np.random.default_rng(seed)
    sampler = TradeSampler(pnls, block_size=block_size, rng=rng)

    logger.info(
        "Running %d dynamic MC sims (max_trades=%d, block=%d) from %d source trades",
        n_simulations, max_trades, block_size, len(pnls),
    )

    paths: list[SimPath] = []
    total_blocked = 0
    total_avg_size = 0.0

    for i in range(n_simulations):
        record = i < n_sample_paths
        sampled = sampler.sample(max_trades, scale=1.0)  # no flat scaling

        path = _simulate_dynamic_path(
            sampled,
            risk_config=risk_config,
            dd_buffer=dd_buffer,
            record_path=record,
        )
        paths.append(path)
        if isinstance(path, DynamicSimPath):
            total_blocked += path.trades_blocked
            total_avg_size += path.avg_size

    result = _aggregate_paths(
        paths,
        n_simulations=n_simulations,
        max_trades=max_trades,
        starting_capital=risk_config.starting_capital,
        profit_target=risk_config.profit_target,
        max_drawdown=risk_config.max_drawdown,
        position_scale=1.0,  # dynamic — no single scale
        block_size=block_size,
        dd_buffer=dd_buffer,
        n_sample_paths=n_sample_paths,
    )

    result.source_trade_count = len(pnls)
    result.source_win_rate = source_win_rate
    result.source_mean_pnl = source_mean
    result.source_std_pnl = source_std

    # Log dynamic sizing stats
    avg_blocked = total_blocked / n_simulations if n_simulations > 0 else 0
    avg_size = total_avg_size / n_simulations if n_simulations > 0 else 0
    logger.info(
        "Dynamic MC complete: pass=%.1f%%, fail=%.1f%%, avg_size=%.3f, avg_blocked=%.1f",
        result.pass_rate * 100, result.fail_rate * 100, avg_size, avg_blocked,
    )

    return result


def run_dynamic_sensitivity(
    trades: Sequence[TradeRecord] | Sequence[float] | np.ndarray,
    base_configs: dict[str, PropRiskConfig],
    *,
    n_simulations: int = 1000,
    max_trades: int = 500,
    block_size: int = 7,
    seed: Optional[int] = None,
    strategy_filter: Optional[str] = None,
) -> dict[str, MonteCarloResult]:
    """Run dynamic MC with multiple named configurations for comparison.

    Parameters
    ----------
    base_configs : dict[str, PropRiskConfig]
        Named configurations to test (e.g. {"conservative": ..., "aggressive": ...}).

    Returns
    -------
    dict[str, MonteCarloResult]
        Results keyed by configuration name.
    """
    results = {}
    for name, config in base_configs.items():
        logger.info("Running dynamic MC for config '%s'...", name)
        results[name] = run_dynamic_monte_carlo(
            trades,
            risk_config=config,
            n_simulations=n_simulations,
            max_trades=max_trades,
            block_size=block_size,
            seed=seed,
            strategy_filter=strategy_filter,
        )
    return results
