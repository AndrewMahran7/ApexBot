"""Monte Carlo simulation engine for prop firm challenge pass probability.

Estimates P(pass before fail) by randomly sampling from historical trades
and simulating many equity paths under prop firm trailing-DD rules.

Usage:
    from challenge.monte_carlo import run_monte_carlo
    result = run_monte_carlo(trades, n_simulations=2000)
    print(f"Pass rate: {result.pass_rate:.1%}")

CLI:
    python -m challenge.monte_carlo --data results/trades.csv --sims 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Minimal trade representation for Monte Carlo sampling."""

    pnl: float
    strategy_type: str = ""


@dataclass
class SimPath:
    """Single simulation equity path result."""

    outcome: str  # "passed", "failed", "incomplete"
    trades_taken: int
    final_equity: float
    peak_equity: float
    max_drawdown: float
    largest_win: float
    largest_loss: float
    max_win_streak: int
    max_loss_streak: int
    equity_path: list[float] = field(default_factory=list)


@dataclass
class DrawdownStats:
    """Aggregate drawdown analysis across all simulations."""

    mean_max_dd: float
    median_max_dd: float
    p95_max_dd: float
    p99_max_dd: float
    mean_largest_loss_streak: float
    median_largest_loss_streak: float
    p95_largest_loss_streak: float
    pct_time_near_boundary: float  # fraction of bars within dd_buffer of limit


@dataclass
class SensitivityPoint:
    """Pass rate at a specific position scale."""

    scale: float
    pass_rate: float
    fail_rate: float
    avg_trades_to_pass: float
    avg_max_dd: float


@dataclass
class MonteCarloResult:
    """Full Monte Carlo simulation output."""

    # Core probabilities
    pass_rate: float
    fail_rate: float
    incomplete_rate: float

    # Timing
    avg_trades_to_pass: float
    avg_trades_to_fail: float
    median_trades_to_pass: float
    median_trades_to_fail: float

    # Equity distribution
    mean_final_equity: float
    median_final_equity: float
    std_final_equity: float

    # Drawdown analysis
    drawdown_stats: DrawdownStats

    # Configuration
    n_simulations: int
    max_trades: int
    starting_capital: float
    profit_target: float
    max_drawdown: float
    position_scale: float
    block_size: int

    # Raw data (optional, for visualization)
    sample_paths: list[SimPath] = field(default_factory=list)
    all_paths: list[SimPath] = field(default_factory=list)

    # Trade source stats
    source_trade_count: int = 0
    source_win_rate: float = 0.0
    source_mean_pnl: float = 0.0
    source_std_pnl: float = 0.0

    # Sensitivity (populated if run_sensitivity called)
    sensitivity: list[SensitivityPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (excludes large equity paths)."""
        return {
            "pass_rate": round(self.pass_rate, 4),
            "fail_rate": round(self.fail_rate, 4),
            "incomplete_rate": round(self.incomplete_rate, 4),
            "avg_trades_to_pass": round(self.avg_trades_to_pass, 1),
            "avg_trades_to_fail": round(self.avg_trades_to_fail, 1),
            "median_trades_to_pass": round(self.median_trades_to_pass, 1),
            "median_trades_to_fail": round(self.median_trades_to_fail, 1),
            "mean_final_equity": round(self.mean_final_equity, 2),
            "median_final_equity": round(self.median_final_equity, 2),
            "std_final_equity": round(self.std_final_equity, 2),
            "drawdown_stats": {
                "mean_max_dd": round(self.drawdown_stats.mean_max_dd, 2),
                "median_max_dd": round(self.drawdown_stats.median_max_dd, 2),
                "p95_max_dd": round(self.drawdown_stats.p95_max_dd, 2),
                "p99_max_dd": round(self.drawdown_stats.p99_max_dd, 2),
                "mean_largest_loss_streak": round(
                    self.drawdown_stats.mean_largest_loss_streak, 1
                ),
                "median_largest_loss_streak": round(
                    self.drawdown_stats.median_largest_loss_streak, 1
                ),
                "p95_largest_loss_streak": round(
                    self.drawdown_stats.p95_largest_loss_streak, 1
                ),
                "pct_time_near_boundary": round(
                    self.drawdown_stats.pct_time_near_boundary, 4
                ),
            },
            "n_simulations": self.n_simulations,
            "max_trades": self.max_trades,
            "starting_capital": self.starting_capital,
            "profit_target": self.profit_target,
            "max_drawdown": self.max_drawdown,
            "position_scale": self.position_scale,
            "block_size": self.block_size,
            "source_trade_count": self.source_trade_count,
            "source_win_rate": round(self.source_win_rate, 4),
            "source_mean_pnl": round(self.source_mean_pnl, 2),
            "source_std_pnl": round(self.source_std_pnl, 2),
            "sensitivity": [
                {
                    "scale": s.scale,
                    "pass_rate": round(s.pass_rate, 4),
                    "fail_rate": round(s.fail_rate, 4),
                    "avg_trades_to_pass": round(s.avg_trades_to_pass, 1),
                    "avg_max_dd": round(s.avg_max_dd, 2),
                }
                for s in self.sensitivity
            ],
        }


# ---------------------------------------------------------------------------
# Trade sampling
# ---------------------------------------------------------------------------

class TradeSampler:
    """Samples trades from historical distribution.

    Supports:
    - iid sampling (with replacement): preserves marginal PnL distribution
    - block sampling: draws contiguous blocks to preserve streak structure
    """

    def __init__(
        self,
        pnls: np.ndarray,
        block_size: int = 1,
        rng: np.random.Generator | None = None,
    ):
        if len(pnls) == 0:
            raise ValueError("Cannot sample from empty trade list")
        self._pnls = pnls.copy()
        self._block_size = max(1, block_size)
        self._rng = rng or np.random.default_rng()

    def sample(self, n: int, scale: float = 1.0) -> np.ndarray:
        """Draw n trade PnLs, optionally scaled."""
        if self._block_size <= 1:
            indices = self._rng.integers(0, len(self._pnls), size=n)
            return self._pnls[indices] * scale

        # Block sampling: draw random starting indices, take contiguous blocks
        blocks_needed = math.ceil(n / self._block_size)
        max_start = max(0, len(self._pnls) - self._block_size)
        starts = self._rng.integers(0, max_start + 1, size=blocks_needed)
        pieces = []
        for s in starts:
            pieces.append(self._pnls[s : s + self._block_size])
        result = np.concatenate(pieces)[:n]
        return result * scale


# ---------------------------------------------------------------------------
# Single path simulation
# ---------------------------------------------------------------------------

def _simulate_path(
    sampled_pnls: np.ndarray,
    starting_capital: float,
    profit_target: float,
    max_drawdown: float,
    dd_buffer: float,
    record_path: bool = False,
) -> SimPath:
    """Simulate one equity path and check pass/fail conditions.

    Uses intraday trailing drawdown: dd_floor = peak_equity - max_drawdown.
    dd_floor only rises (never falls).
    """
    equity = starting_capital
    peak_equity = starting_capital
    dd_floor = starting_capital - max_drawdown
    target = starting_capital + profit_target

    max_dd_seen = 0.0
    largest_win = 0.0
    largest_loss = 0.0
    current_streak = 0  # positive = wins, negative = losses
    max_win_streak = 0
    max_loss_streak = 0
    near_boundary_count = 0

    path = [equity] if record_path else []
    outcome = "incomplete"
    trades_taken = 0

    for i in range(len(sampled_pnls)):
        pnl = float(sampled_pnls[i])
        equity += pnl
        trades_taken += 1

        # Track peak and trailing DD floor
        if equity > peak_equity:
            peak_equity = equity
            dd_floor = peak_equity - max_drawdown

        # Track max drawdown experienced
        dd = peak_equity - equity
        if dd > max_dd_seen:
            max_dd_seen = dd

        # Near-boundary tracking
        if dd_buffer > 0 and (equity - dd_floor) <= dd_buffer:
            near_boundary_count += 1

        # Win/loss tracking
        if pnl > 0:
            largest_win = max(largest_win, pnl)
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_win_streak = max(max_win_streak, current_streak)
        elif pnl < 0:
            largest_loss = min(largest_loss, pnl)
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_loss_streak = max(max_loss_streak, abs(current_streak))

        if record_path:
            path.append(equity)

        # Termination: PASS
        if equity >= target:
            outcome = "passed"
            break

        # Termination: FAIL (trailing DD breach)
        if equity <= dd_floor:
            outcome = "failed"
            break

    return SimPath(
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
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_paths(
    paths: list[SimPath],
    n_simulations: int,
    max_trades: int,
    starting_capital: float,
    profit_target: float,
    max_drawdown: float,
    position_scale: float,
    block_size: int,
    dd_buffer: float,
    n_sample_paths: int,
) -> MonteCarloResult:
    """Aggregate SimPath results into MonteCarloResult."""
    passed = [p for p in paths if p.outcome == "passed"]
    failed = [p for p in paths if p.outcome == "failed"]

    n_total = len(paths)
    pass_rate = len(passed) / n_total if n_total else 0.0
    fail_rate = len(failed) / n_total if n_total else 0.0
    incomplete_rate = 1.0 - pass_rate - fail_rate

    # Timing stats
    pass_trades = [p.trades_taken for p in passed]
    fail_trades = [p.trades_taken for p in failed]

    avg_trades_pass = float(np.mean(pass_trades)) if pass_trades else 0.0
    avg_trades_fail = float(np.mean(fail_trades)) if fail_trades else 0.0
    med_trades_pass = float(np.median(pass_trades)) if pass_trades else 0.0
    med_trades_fail = float(np.median(fail_trades)) if fail_trades else 0.0

    # Equity distribution
    finals = np.array([p.final_equity for p in paths])
    mean_final = float(np.mean(finals))
    med_final = float(np.median(finals))
    std_final = float(np.std(finals)) if len(finals) > 1 else 0.0

    # Drawdown analysis
    max_dds = np.array([p.max_drawdown for p in paths])
    loss_streaks = np.array([p.max_loss_streak for p in paths], dtype=float)

    total_trades_all = sum(p.trades_taken for p in paths)
    near_count = 0
    # Near-boundary is tracked per-path, approximate from max_dd proximity
    for p in paths:
        if p.max_drawdown >= (max_drawdown - dd_buffer):
            near_count += p.trades_taken  # rough: if they hit near, count all
    pct_near = near_count / total_trades_all if total_trades_all > 0 else 0.0

    dd_stats = DrawdownStats(
        mean_max_dd=float(np.mean(max_dds)),
        median_max_dd=float(np.median(max_dds)),
        p95_max_dd=float(np.percentile(max_dds, 95)),
        p99_max_dd=float(np.percentile(max_dds, 99)),
        mean_largest_loss_streak=float(np.mean(loss_streaks)),
        median_largest_loss_streak=float(np.median(loss_streaks)),
        p95_largest_loss_streak=float(np.percentile(loss_streaks, 95)),
        pct_time_near_boundary=pct_near,
    )

    # Select sample paths (prefer a mix of outcomes)
    sample = []
    rng = np.random.default_rng(42)
    if n_sample_paths > 0 and paths:
        # Try to get a balanced sample
        passed_with_path = [p for p in passed if p.equity_path]
        failed_with_path = [p for p in failed if p.equity_path]
        half = n_sample_paths // 2

        if passed_with_path:
            chosen_pass = rng.choice(
                len(passed_with_path),
                size=min(half, len(passed_with_path)),
                replace=False,
            )
            sample.extend(passed_with_path[i] for i in chosen_pass)
        if failed_with_path:
            chosen_fail = rng.choice(
                len(failed_with_path),
                size=min(n_sample_paths - len(sample), len(failed_with_path)),
                replace=False,
            )
            sample.extend(failed_with_path[i] for i in chosen_fail)

        # Fill remainder from any path
        remaining = n_sample_paths - len(sample)
        if remaining > 0:
            all_with_path = [p for p in paths if p.equity_path]
            if all_with_path:
                extra = rng.choice(
                    len(all_with_path),
                    size=min(remaining, len(all_with_path)),
                    replace=False,
                )
                sample.extend(all_with_path[i] for i in extra)

    return MonteCarloResult(
        pass_rate=pass_rate,
        fail_rate=fail_rate,
        incomplete_rate=incomplete_rate,
        avg_trades_to_pass=avg_trades_pass,
        avg_trades_to_fail=avg_trades_fail,
        median_trades_to_pass=med_trades_pass,
        median_trades_to_fail=med_trades_fail,
        mean_final_equity=mean_final,
        median_final_equity=med_final,
        std_final_equity=std_final,
        drawdown_stats=dd_stats,
        n_simulations=n_simulations,
        max_trades=max_trades,
        starting_capital=starting_capital,
        profit_target=profit_target,
        max_drawdown=max_drawdown,
        position_scale=position_scale,
        block_size=block_size,
        sample_paths=sample,
        all_paths=paths,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_monte_carlo(
    trades: Sequence[TradeRecord] | Sequence[float] | np.ndarray,
    *,
    n_simulations: int = 2000,
    max_trades: int = 200,
    starting_capital: float = 25_000.0,
    profit_target: float = 1_500.0,
    max_drawdown: float = 1_000.0,
    dd_buffer: float = 200.0,
    position_scale: float = 1.0,
    block_size: int = 1,
    n_sample_paths: int = 10,
    seed: Optional[int] = None,
    strategy_filter: Optional[str] = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation of prop firm challenge outcomes.

    Parameters
    ----------
    trades : sequence of TradeRecord, floats, or numpy array
        Historical trades to sample from. If TradeRecord, uses .pnl field.
        If float/ndarray, interpreted directly as PnL values.
    n_simulations : int
        Number of equity paths to simulate.
    max_trades : int
        Max trades per simulation before declaring incomplete.
    starting_capital : float
        Account starting balance.
    profit_target : float
        Dollar profit needed to pass (equity >= starting_capital + profit_target).
    max_drawdown : float
        Trailing intraday drawdown limit.
    dd_buffer : float
        Distance from DD limit to count as "near boundary".
    position_scale : float
        Scale factor applied to each trade PnL (e.g. 0.5 = half size).
    block_size : int
        Block sampling size (1 = iid, >1 = contiguous blocks for streak preservation).
    n_sample_paths : int
        Number of full equity paths to record for visualization.
    seed : int or None
        RNG seed for reproducibility.
    strategy_filter : str or None
        If set, only include trades matching this strategy_type.

    Returns
    -------
    MonteCarloResult
        Full simulation results with pass rate, stats, and optional paths.
    """
    # Extract PnL array
    pnls = _extract_pnls(trades, strategy_filter)
    if len(pnls) == 0:
        raise ValueError("No trades to simulate (after filtering)")

    # Source stats
    wins = pnls[pnls > 0]
    source_win_rate = len(wins) / len(pnls) if len(pnls) else 0.0
    source_mean = float(np.mean(pnls))
    source_std = float(np.std(pnls)) if len(pnls) > 1 else 0.0

    rng = np.random.default_rng(seed)
    sampler = TradeSampler(pnls, block_size=block_size, rng=rng)

    logger.info(
        "Running %d Monte Carlo simulations (max_trades=%d, scale=%.2f, "
        "block=%d, seed=%s) from %d source trades",
        n_simulations,
        max_trades,
        position_scale,
        block_size,
        seed,
        len(pnls),
    )

    paths: list[SimPath] = []
    for i in range(n_simulations):
        record = i < n_sample_paths
        sampled = sampler.sample(max_trades, scale=position_scale)
        path = _simulate_path(
            sampled,
            starting_capital=starting_capital,
            profit_target=profit_target,
            max_drawdown=max_drawdown,
            dd_buffer=dd_buffer,
            record_path=record,
        )
        paths.append(path)

    result = _aggregate_paths(
        paths,
        n_simulations=n_simulations,
        max_trades=max_trades,
        starting_capital=starting_capital,
        profit_target=profit_target,
        max_drawdown=max_drawdown,
        position_scale=position_scale,
        block_size=block_size,
        dd_buffer=dd_buffer,
        n_sample_paths=n_sample_paths,
    )

    result.source_trade_count = len(pnls)
    result.source_win_rate = source_win_rate
    result.source_mean_pnl = source_mean
    result.source_std_pnl = source_std

    return result


def run_sensitivity(
    trades: Sequence[TradeRecord] | Sequence[float] | np.ndarray,
    scales: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    *,
    n_simulations: int = 1000,
    max_trades: int = 200,
    starting_capital: float = 25_000.0,
    profit_target: float = 1_500.0,
    max_drawdown: float = 1_000.0,
    block_size: int = 1,
    seed: Optional[int] = None,
    strategy_filter: Optional[str] = None,
) -> MonteCarloResult:
    """Run Monte Carlo at multiple position scales and return combined result.

    The base result uses scale=1.0. The sensitivity field contains
    pass rates at each scale for comparison.
    """
    pnls = _extract_pnls(trades, strategy_filter)
    if len(pnls) == 0:
        raise ValueError("No trades to simulate (after filtering)")

    # Run base at scale 1.0
    base = run_monte_carlo(
        pnls,
        n_simulations=n_simulations,
        max_trades=max_trades,
        starting_capital=starting_capital,
        profit_target=profit_target,
        max_drawdown=max_drawdown,
        position_scale=1.0,
        block_size=block_size,
        seed=seed,
        n_sample_paths=10,
    )

    points: list[SensitivityPoint] = []
    for scale in scales:
        if scale == 1.0:
            # Reuse base result
            points.append(
                SensitivityPoint(
                    scale=1.0,
                    pass_rate=base.pass_rate,
                    fail_rate=base.fail_rate,
                    avg_trades_to_pass=base.avg_trades_to_pass,
                    avg_max_dd=base.drawdown_stats.mean_max_dd,
                )
            )
            continue

        r = run_monte_carlo(
            pnls,
            n_simulations=n_simulations,
            max_trades=max_trades,
            starting_capital=starting_capital,
            profit_target=profit_target,
            max_drawdown=max_drawdown,
            position_scale=scale,
            block_size=block_size,
            seed=seed,
            n_sample_paths=0,
        )
        points.append(
            SensitivityPoint(
                scale=scale,
                pass_rate=r.pass_rate,
                fail_rate=r.fail_rate,
                avg_trades_to_pass=r.avg_trades_to_pass,
                avg_max_dd=r.drawdown_stats.mean_max_dd,
            )
        )

    points.sort(key=lambda p: p.scale)
    base.sensitivity = points
    return base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pnls(
    trades: Sequence[TradeRecord] | Sequence[float] | np.ndarray,
    strategy_filter: Optional[str] = None,
) -> np.ndarray:
    """Convert trade input to numpy PnL array."""
    if isinstance(trades, np.ndarray):
        return trades.astype(float)

    if not trades:
        return np.array([], dtype=float)

    # Check first element type
    first = trades[0]
    if isinstance(first, (int, float)):
        return np.array(trades, dtype=float)

    if isinstance(first, TradeRecord):
        filtered = trades
        if strategy_filter:
            filtered = [t for t in trades if t.strategy_type == strategy_filter]
        return np.array([t.pnl for t in filtered], dtype=float)

    # Duck-type: try .net_pnl (backtest Trade) then .pnl
    result = []
    for t in trades:
        if strategy_filter:
            stype = getattr(t, "strategy_type", "")
            if stype != strategy_filter:
                continue
        if hasattr(t, "net_pnl"):
            result.append(float(t.net_pnl))
        elif hasattr(t, "pnl"):
            result.append(float(t.pnl))
        else:
            raise TypeError(
                f"Cannot extract PnL from {type(t).__name__}. "
                "Expected .net_pnl, .pnl, or numeric value."
            )
    return np.array(result, dtype=float)


def trades_from_csv(path: str | Path) -> list[TradeRecord]:
    """Load trades from a CSV file (e.g. results/trades.csv).

    Looks for columns: net_pnl (or pnl), strategy_type (optional).
    """
    import pandas as pd

    df = pd.read_csv(path)

    # Find PnL column
    pnl_col = None
    for candidate in ("net_pnl", "pnl", "pnl_dollars", "Net PnL", "PnL"):
        if candidate in df.columns:
            pnl_col = candidate
            break
    if pnl_col is None:
        raise ValueError(
            f"Cannot find PnL column in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    strategy_col = None
    for candidate in ("strategy_type", "strategy", "Strategy"):
        if candidate in df.columns:
            strategy_col = candidate
            break

    records = []
    for _, row in df.iterrows():
        pnl = float(row[pnl_col])
        stype = str(row[strategy_col]) if strategy_col else ""
        records.append(TradeRecord(pnl=pnl, strategy_type=stype))

    return records


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(
    result: MonteCarloResult,
    output_dir: str | Path = "results",
    show: bool = False,
) -> dict[str, Path]:
    """Generate visualization plots for Monte Carlo results.

    Returns dict mapping plot name to file path.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # --- 1. Histogram of final outcomes ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    finals = [p.final_equity for p in result.all_paths]
    colors = []
    for p in result.all_paths:
        if p.outcome == "passed":
            colors.append("green")
        elif p.outcome == "failed":
            colors.append("red")
        else:
            colors.append("gray")

    ax = axes[0]
    ax.hist(finals, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(
        result.starting_capital + result.profit_target,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Target (${result.starting_capital + result.profit_target:,.0f})",
    )
    ax.axvline(
        result.starting_capital - result.max_drawdown,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"DD Limit (${result.starting_capital - result.max_drawdown:,.0f})",
    )
    ax.set_xlabel("Final Equity ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Final Equity")
    ax.legend()

    # Outcome pie chart
    ax = axes[1]
    sizes = [result.pass_rate, result.fail_rate, result.incomplete_rate]
    labels_pie = [
        f"Pass ({result.pass_rate:.1%})",
        f"Fail ({result.fail_rate:.1%})",
        f"Incomplete ({result.incomplete_rate:.1%})",
    ]
    pie_colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    # Filter out zero slices
    filtered = [
        (s, l, c) for s, l, c in zip(sizes, labels_pie, pie_colors) if s > 0
    ]
    if filtered:
        ax.pie(
            [f[0] for f in filtered],
            labels=[f[1] for f in filtered],
            colors=[f[2] for f in filtered],
            autopct="%1.1f%%",
            startangle=90,
        )
    ax.set_title("Outcome Distribution")

    plt.tight_layout()
    path = output_dir / "mc_outcome_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["outcome_distribution"] = path

    # --- 2. Sample equity paths ---
    if result.sample_paths:
        fig, ax = plt.subplots(figsize=(12, 6))

        target_line = result.starting_capital + result.profit_target

        for sp in result.sample_paths:
            if not sp.equity_path:
                continue
            x = list(range(len(sp.equity_path)))
            color = "green" if sp.outcome == "passed" else "red"
            alpha = 0.6
            ax.plot(x, sp.equity_path, color=color, alpha=alpha, linewidth=0.8)

        ax.axhline(
            target_line,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Target ${target_line:,.0f}",
        )
        ax.axhline(
            result.starting_capital,
            color="gray",
            linestyle=":",
            linewidth=1,
            label="Starting Capital",
        )
        # DD floor is dynamic, show initial floor
        initial_floor = result.starting_capital - result.max_drawdown
        ax.axhline(
            initial_floor,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Initial DD Floor ${initial_floor:,.0f}",
        )
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity ($)")
        ax.set_title(f"Sample Equity Paths ({len(result.sample_paths)} simulations)")
        ax.legend()

        plt.tight_layout()
        path = output_dir / "mc_equity_paths.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved["equity_paths"] = path

    # --- 3. Cumulative pass probability over trade count ---
    if result.all_paths:
        fig, ax = plt.subplots(figsize=(10, 5))

        max_t = result.max_trades
        pass_by_trade = np.zeros(max_t + 1)
        fail_by_trade = np.zeros(max_t + 1)
        total = len(result.all_paths)

        for p in result.all_paths:
            idx = min(p.trades_taken, max_t)
            if p.outcome == "passed":
                pass_by_trade[idx] += 1
            elif p.outcome == "failed":
                fail_by_trade[idx] += 1

        cum_pass = np.cumsum(pass_by_trade) / total
        cum_fail = np.cumsum(fail_by_trade) / total

        ax.plot(
            range(max_t + 1), cum_pass, color="green", linewidth=2, label="P(pass)"
        )
        ax.plot(
            range(max_t + 1), cum_fail, color="red", linewidth=2, label="P(fail)"
        )
        ax.fill_between(range(max_t + 1), cum_pass, alpha=0.1, color="green")
        ax.fill_between(range(max_t + 1), cum_fail, alpha=0.1, color="red")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Cumulative Pass/Fail Probability Over Time")
        ax.legend()
        ax.set_xlim(0, max_t)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / "mc_cumulative_probability.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved["cumulative_probability"] = path

    # --- 4. Sensitivity chart ---
    if result.sensitivity:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        scales = [s.scale for s in result.sensitivity]
        pass_rates = [s.pass_rate for s in result.sensitivity]
        avg_dds = [s.avg_max_dd for s in result.sensitivity]

        ax1.bar(
            scales,
            pass_rates,
            width=0.08,
            color="steelblue",
            alpha=0.7,
            label="Pass Rate",
        )
        ax1.set_xlabel("Position Scale")
        ax1.set_ylabel("Pass Rate", color="steelblue")
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()
        ax2.plot(
            scales,
            avg_dds,
            color="red",
            marker="o",
            linewidth=2,
            label="Avg Max DD",
        )
        ax2.set_ylabel("Avg Max Drawdown ($)", color="red")

        ax1.set_title("Sensitivity: Pass Rate vs Position Scale")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        path = output_dir / "mc_sensitivity.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved["sensitivity"] = path

    return saved


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_report(result: MonteCarloResult) -> None:
    """Print human-readable Monte Carlo report to stdout."""
    print("\n" + "=" * 60)
    print("  MONTE CARLO SIMULATION REPORT")
    print("=" * 60)

    print(f"\n  Source trades:     {result.source_trade_count}")
    print(f"  Source win rate:   {result.source_win_rate:.1%}")
    print(f"  Source mean PnL:   ${result.source_mean_pnl:,.2f}")
    print(f"  Source std PnL:    ${result.source_std_pnl:,.2f}")

    print(f"\n  Simulations:       {result.n_simulations:,}")
    print(f"  Max trades/sim:    {result.max_trades}")
    print(f"  Position scale:    {result.position_scale:.2f}x")
    print(f"  Block size:        {result.block_size}")

    print(f"\n  Starting capital:  ${result.starting_capital:,.0f}")
    print(f"  Profit target:     +${result.profit_target:,.0f}")
    print(f"  Max drawdown:      -${result.max_drawdown:,.0f}")

    print("\n  --- PROBABILITIES ---")
    print(f"  P(pass before fail):          {result.pass_rate:.1%}")
    print(f"  P(fail before pass):          {result.fail_rate:.1%}")
    print(f"  P(incomplete @ {result.max_trades} trades):  {result.incomplete_rate:.1%}")

    print("\n  --- TIMING ---")
    print(f"  Avg trades to pass:    {result.avg_trades_to_pass:.0f}")
    print(f"  Median trades to pass: {result.median_trades_to_pass:.0f}")
    print(f"  Avg trades to fail:    {result.avg_trades_to_fail:.0f}")
    print(f"  Median trades to fail: {result.median_trades_to_fail:.0f}")

    print("\n  --- EQUITY DISTRIBUTION ---")
    print(f"  Mean final equity:   ${result.mean_final_equity:,.2f}")
    print(f"  Median final equity: ${result.median_final_equity:,.2f}")
    print(f"  Std final equity:    ${result.std_final_equity:,.2f}")

    dd = result.drawdown_stats
    print("\n  --- DRAWDOWN ANALYSIS ---")
    print(f"  Mean max DD:   ${dd.mean_max_dd:,.2f}")
    print(f"  Median max DD: ${dd.median_max_dd:,.2f}")
    print(f"  95th %-ile DD: ${dd.p95_max_dd:,.2f}")
    print(f"  99th %-ile DD: ${dd.p99_max_dd:,.2f}")
    print(f"  Mean loss streak:   {dd.mean_largest_loss_streak:.1f}")
    print(f"  Median loss streak: {dd.median_largest_loss_streak:.1f}")
    print(f"  95th %-ile streak:  {dd.p95_largest_loss_streak:.0f}")

    if result.sensitivity:
        print("\n  --- SENSITIVITY (Position Scale) ---")
        print(f"  {'Scale':<8} {'Pass%':<10} {'Fail%':<10} {'Avg Trades':<12} {'Avg DD'}")
        for s in result.sensitivity:
            print(
                f"  {s.scale:<8.2f} {s.pass_rate:<10.1%} {s.fail_rate:<10.1%} "
                f"{s.avg_trades_to_pass:<12.0f} ${s.avg_max_dd:,.0f}"
            )

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for Monte Carlo simulation."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo prop firm challenge simulator"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to trades CSV (must have net_pnl or pnl column)",
    )
    parser.add_argument("--sims", type=int, default=2000, help="Number of simulations")
    parser.add_argument(
        "--max-trades", type=int, default=200, help="Max trades per sim"
    )
    parser.add_argument(
        "--capital", type=float, default=25_000.0, help="Starting capital"
    )
    parser.add_argument(
        "--target", type=float, default=1_500.0, help="Profit target"
    )
    parser.add_argument(
        "--max-dd", type=float, default=1_000.0, help="Max drawdown"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Position scale factor"
    )
    parser.add_argument(
        "--block-size", type=int, default=1, help="Block sampling size (1=iid)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--strategy-filter", default=None, help="Only include this strategy_type"
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analysis across position scales",
    )
    parser.add_argument(
        "--output", default=None, help="JSON output path"
    )
    parser.add_argument(
        "--plot-dir", default="results", help="Directory for plot output"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plot generation"
    )

    args = parser.parse_args(argv)

    trades = trades_from_csv(args.data)
    logger.info("Loaded %d trades from %s", len(trades), args.data)

    if args.sensitivity:
        result = run_sensitivity(
            trades,
            n_simulations=args.sims,
            max_trades=args.max_trades,
            starting_capital=args.capital,
            profit_target=args.target,
            max_drawdown=args.max_dd,
            block_size=args.block_size,
            seed=args.seed,
            strategy_filter=args.strategy_filter,
        )
    else:
        result = run_monte_carlo(
            trades,
            n_simulations=args.sims,
            max_trades=args.max_trades,
            starting_capital=args.capital,
            profit_target=args.target,
            max_drawdown=args.max_dd,
            position_scale=args.scale,
            block_size=args.block_size,
            seed=args.seed,
            strategy_filter=args.strategy_filter,
        )

    print_report(result)

    if not args.no_plot:
        saved = plot_results(result, output_dir=args.plot_dir)
        if saved:
            print(f"\nPlots saved to {args.plot_dir}/:")
            for name, path in saved.items():
                print(f"  {name}: {path}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
