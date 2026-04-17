"""
Performance metrics and analytics for backtest results.
========================================================

Computes standard trading-system statistics from a BacktestResult.
All metrics work on futures PnL (dollars and points), not stock returns.
"""

from __future__ import annotations
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, Trade
from config.settings import BacktestConfig

logger = logging.getLogger(__name__)


def compute_metrics(result: BacktestResult, config: BacktestConfig) -> dict:
    """
    Compute a full dictionary of performance metrics.

    Returns a JSON-serializable dict.
    """
    trades = result.trades
    n = len(trades)

    if n == 0:
        return _empty_metrics(config)

    net_pnls = np.array([t.net_pnl for t in trades])
    gross_pnls = np.array([t.pnl_dollars for t in trades])
    points = np.array([t.pnl_points for t in trades])

    winners = net_pnls[net_pnls > 0]
    losers = net_pnls[net_pnls <= 0]

    gross_profit = float(winners.sum()) if len(winners) else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) else 0.0

    win_rate = len(winners) / n * 100
    avg_win = float(winners.mean()) if len(winners) else 0.0
    avg_loss = float(losers.mean()) if len(losers) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = float(net_pnls.mean())

    # Equity curve based metrics
    equities = np.array([ep.equity for ep in result.equity_curve])
    # Daily returns — resample equity curve to daily
    daily_returns = _daily_returns_from_equity(result)

    sharpe = _sharpe_ratio(daily_returns)
    max_dd_dollars, max_dd_pct = _max_drawdown(equities, config.initial_capital)

    # Trade durations
    durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades]
    avg_duration_min = float(np.mean(durations)) if durations else 0.0

    # PnL by day of week
    pnl_by_day = _pnl_by_weekday(trades)

    total_net = float(net_pnls.sum())
    total_return_pct = (total_net / config.initial_capital) * 100

    # Direction breakdown
    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]
    long_wins = [t for t in long_trades if t.net_pnl > 0]
    short_wins = [t for t in short_trades if t.net_pnl > 0]

    return {
        "total_trades": n,
        "winning_trades": int(len(winners)),
        "losing_trades": int(len(losers)),
        "win_rate_pct": round(win_rate, 2),
        "total_pnl_dollars": round(total_net, 2),
        "total_pnl_points": round(float(points.sum()), 2),
        "total_return_pct": round(total_return_pct, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "largest_win": round(float(net_pnls.max()), 2),
        "largest_loss": round(float(net_pnls.min()), 2),
        "max_drawdown_dollars": round(max_dd_dollars, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 4),
        "avg_trade_duration_min": round(avg_duration_min, 1),
        "total_commission": round(float(sum(t.commission for t in trades)), 2),
        "total_slippage": round(float(sum(t.slippage_cost for t in trades)), 2),
        "initial_capital": config.initial_capital,
        "final_equity": round(result.final_equity, 2),
        "bars_processed": result.bar_count,
        "pnl_by_weekday": pnl_by_day,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_win_rate_pct": round(len(long_wins) / len(long_trades) * 100, 2) if long_trades else 0.0,
        "short_win_rate_pct": round(len(short_wins) / len(short_trades) * 100, 2) if short_trades else 0.0,
        "long_pnl_dollars": round(sum(t.net_pnl for t in long_trades), 2),
        "short_pnl_dollars": round(sum(t.net_pnl for t in short_trades), 2),
    }


def export_trades_csv(trades: list[Trade], path: str):
    """Write detailed trade log to CSV."""
    if not trades:
        return
    rows = []
    for t in trades:
        # Compute ticks from price move
        pnl_ticks = t.pnl_points / 0.25 if t.pnl_points != 0 else 0.0
        rows.append({
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "direction": t.direction,
            "contracts": t.contracts,
            "position_size": round(t.position_size, 4),
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "pnl_ticks": round(pnl_ticks, 1),
            "pnl_points": round(t.pnl_points, 2),
            "pnl_dollars": round(t.pnl_dollars, 2),
            "commission": round(t.commission, 2),
            "slippage_cost": round(t.slippage_cost, 2),
            "net_pnl": round(t.net_pnl, 2),
            "exit_reason": t.exit_reason,
            "strategy_type": getattr(t, 'strategy_type', ''),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def export_equity_csv(result: BacktestResult, path: str):
    """Write equity curve to CSV."""
    rows = [
        {
            "timestamp": ep.timestamp.isoformat(),
            "equity": round(ep.equity, 2),
            "drawdown": round(ep.drawdown, 2),
        }
        for ep in result.equity_curve
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def export_metrics_json(metrics: dict, path: str):
    """Write metrics dict to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def plot_equity_curve(result: BacktestResult, path: str,
                      benchmark_equity: Optional[list] = None,
                      benchmark_label: str = "Benchmark",
                      benchmarks: Optional[list] = None):
    """
    Save a matplotlib equity curve plot.

    Parameters
    ----------
    benchmark_equity : list[float], optional
        Legacy single benchmark overlay (backward compatible).
    benchmark_label : str
        Label for the legacy benchmark.
    benchmarks : list of (equity_list, label, color) tuples, optional
        Multiple benchmark overlays. If provided, benchmark_equity is ignored.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return

    timestamps = [ep.timestamp for ep in result.equity_curve]
    equities = [ep.equity for ep in result.equity_curve]
    drawdowns = [ep.drawdown for ep in result.equity_curve]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(timestamps, equities, linewidth=1.2, label="Strategy Equity", color="#2196F3")

    # Plot benchmarks
    if benchmarks:
        for bm_equity, bm_label, bm_color in benchmarks:
            if bm_equity and len(bm_equity) == len(timestamps):
                ax1.plot(timestamps, bm_equity, linewidth=1.0, label=bm_label,
                         color=bm_color, alpha=0.8)
    elif benchmark_equity and len(benchmark_equity) == len(timestamps):
        ax1.plot(timestamps, benchmark_equity, linewidth=1.0, label=benchmark_label,
                 color="#FF9800", alpha=0.8)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("MES Backtest — Equity Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(timestamps, drawdowns, 0, color="#EF5350", alpha=0.4)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity curve plot saved to %s", path)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _empty_metrics(config: BacktestConfig) -> dict:
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": 0,
        "total_pnl_dollars": 0,
        "total_pnl_points": 0,
        "total_return_pct": 0,
        "gross_profit": 0,
        "gross_loss": 0,
        "profit_factor": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "expectancy": 0,
        "largest_win": 0,
        "largest_loss": 0,
        "max_drawdown_dollars": 0,
        "max_drawdown_pct": 0,
        "sharpe_ratio": 0,
        "avg_trade_duration_min": 0,
        "total_commission": 0,
        "total_slippage": 0,
        "initial_capital": config.initial_capital,
        "final_equity": config.initial_capital,
        "bars_processed": 0,
        "pnl_by_weekday": {},
        "long_trades": 0,
        "short_trades": 0,
        "long_win_rate_pct": 0,
        "short_win_rate_pct": 0,
        "long_pnl_dollars": 0,
        "short_pnl_dollars": 0,
    }


def _daily_returns_from_equity(result: BacktestResult) -> np.ndarray:
    """Compute daily log returns from the equity curve."""
    if len(result.equity_curve) < 2:
        return np.array([])

    df = pd.DataFrame([
        {"timestamp": ep.timestamp, "equity": ep.equity}
        for ep in result.equity_curve
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    daily = df.set_index("timestamp")["equity"].resample("D").last().dropna()

    if len(daily) < 2:
        return np.array([])

    returns = daily.pct_change().dropna().values
    return returns


def _sharpe_ratio(daily_returns: np.ndarray, risk_free_annual: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_annual / 252
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(252))


def _max_drawdown(equities: np.ndarray, initial: float) -> tuple[float, float]:
    """Return (max_dd_dollars, max_dd_pct) from an equity array."""
    if len(equities) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equities)
    dd = equities - peak
    max_dd = float(dd.min())                   # negative number
    # Percent relative to peak at that point
    peak_at_dd = peak[np.argmin(dd)]
    max_dd_pct = (max_dd / peak_at_dd) * 100 if peak_at_dd > 0 else 0.0
    return abs(max_dd), abs(max_dd_pct)


def _pnl_by_weekday(trades: list[Trade]) -> dict:
    """Aggregate net PnL by day-of-week name."""
    by_day: dict[str, float] = {}
    for t in trades:
        day_name = t.entry_time.strftime("%A")
        by_day[day_name] = by_day.get(day_name, 0.0) + t.net_pnl
    return {k: round(v, 2) for k, v in by_day.items()}


# ------------------------------------------------------------------
# Benchmark comparison utilities
# ------------------------------------------------------------------

def compute_equity_list_metrics(
    equities: list[float],
    timestamps: pd.DatetimeIndex,
    initial_capital: float,
) -> dict:
    """
    Compute summary metrics from a bare equity curve (no trade data).

    Returns a dict compatible with the benchmark comparison table.
    Fields that require trade data (trade_count, win_rate_pct,
    profit_factor) are set to None.
    """
    if not equities:
        return {
            'final_equity': initial_capital, 'net_pnl': 0.0, 'return_pct': 0.0,
            'max_drawdown_dollars': 0.0, 'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0, 'trade_count': None,
            'win_rate_pct': None, 'profit_factor': None,
        }

    final = equities[-1]
    net_pnl = final - initial_capital
    return_pct = (net_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0

    eq_arr = np.array(equities)
    max_dd, max_dd_pct = _max_drawdown(eq_arr, initial_capital)

    # Sharpe from daily returns
    df = pd.DataFrame({'equity': equities}, index=timestamps[:len(equities)])
    daily = df['equity'].resample('D').last().dropna()
    sharpe = 0.0
    if len(daily) >= 2:
        returns = daily.pct_change().dropna().values
        std = float(np.std(returns, ddof=1))
        if std > 0:
            sharpe = float(np.mean(returns) / std * math.sqrt(252))

    return {
        'final_equity': round(final, 2),
        'net_pnl': round(net_pnl, 2),
        'return_pct': round(return_pct, 2),
        'max_drawdown_dollars': round(max_dd, 2),
        'max_drawdown_pct': round(max_dd_pct, 2),
        'sharpe_ratio': round(sharpe, 4),
        'trade_count': None,
        'win_rate_pct': None,
        'profit_factor': None,
    }


def metrics_to_comparison_row(name: str, metrics: dict) -> dict:
    """Extract comparison-relevant fields from a compute_metrics() result."""
    return {
        'name': name,
        'final_equity': metrics.get('final_equity', 0),
        'net_pnl': metrics.get('total_pnl_dollars', 0),
        'return_pct': metrics.get('total_return_pct', 0),
        'max_drawdown_dollars': metrics.get('max_drawdown_dollars', 0),
        'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'trade_count': metrics.get('total_trades'),
        'win_rate_pct': metrics.get('win_rate_pct'),
        'profit_factor': metrics.get('profit_factor'),
    }


def equity_list_to_comparison_row(
    name: str, equities: list[float],
    timestamps: pd.DatetimeIndex, initial_capital: float,
) -> dict:
    """Build a comparison row from a bare equity curve."""
    m = compute_equity_list_metrics(equities, timestamps, initial_capital)
    m['name'] = name
    return m


def print_benchmark_table(rows: list[dict]):
    """Print a formatted benchmark comparison table to console."""
    header = (f"  {'Strategy':<22s} {'Final Eq':>10s} {'Net PnL':>10s} "
              f"{'Ret%':>7s} {'MaxDD':>10s} {'Sharpe':>7s} "
              f"{'Trades':>7s} {'WR%':>6s} {'PF':>6s}")
    sep = "  " + "-" * 90
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK COMPARISON")
    print(f"{'=' * 60}")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        dash = "-"
        if r.get('trade_count') is not None:
            tc = f"{r['trade_count']:>7d}"
        else:
            tc = f"{dash:>7s}"
        if r.get('win_rate_pct') is not None:
            wr = f"{r['win_rate_pct']:>5.1f}%"
        else:
            wr = f"{dash:>6s}"
        if r.get('profit_factor') is not None:
            pf = f"{r['profit_factor']:>6.2f}"
        else:
            pf = f"{dash:>6s}"
        print(f"  {r['name']:<22s} ${r['final_equity']:>8,.0f} ${r['net_pnl']:>8,.0f} "
              f"{r['return_pct']:>6.1f}% ${r['max_drawdown_dollars']:>8,.0f} "
              f"{r['sharpe_ratio']:>7.2f} {tc} {wr} {pf}")
    print(sep)


def print_strategy_comparison(main_row: dict, other_row: dict):
    """Print pairwise comparison between two strategies."""
    mn = main_row['name']
    on = other_row['name']
    pnl_diff = main_row['net_pnl'] - other_row['net_pnl']
    dd_diff = main_row['max_drawdown_dollars'] - other_row['max_drawdown_dollars']
    sh_diff = main_row['sharpe_ratio'] - other_row['sharpe_ratio']

    pnl_better = mn if pnl_diff > 0 else on
    dd_better = mn if dd_diff < 0 else on
    sh_better = mn if sh_diff > 0 else on

    print(f"\n  {mn} vs {on}")
    print(f"  {'-' * 50}")
    print(f"  Net PnL difference   : ${pnl_diff:>+,.2f}  ({pnl_better} better)")
    print(f"  Max DD difference    : ${dd_diff:>+,.2f}  ({dd_better} better)")
    print(f"  Sharpe difference    : {sh_diff:>+.2f}  ({sh_better} better)")
    score_main = sum([pnl_diff > 0, dd_diff < 0, sh_diff > 0])
    if score_main >= 2:
        print(f"  -> {mn} leads on {score_main}/3 key metrics")
    else:
        print(f"  -> {on} leads on {3 - score_main}/3 key metrics")


def export_benchmark_table(rows: list[dict], csv_path: str, json_path: str):
    """Export benchmark comparison to CSV and JSON."""
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info("Benchmark comparison exported to %s", csv_path)


# ------------------------------------------------------------------
# Multi-chart plotting
# ------------------------------------------------------------------

def plot_normalized_comparison(
    series: list[tuple],
    timestamps: list,
    initial_capital: float,
    path: str,
    title_suffix: str = "",
):
    """
    Chart A: All curves normalized to start at 100 (percentage growth).

    Parameters
    ----------
    series : list of (equity_list, name, color)
    timestamps : full list of bar timestamps
    initial_capital : used as fallback for start value
    path : output file path
    title_suffix : optional text appended to chart title
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed - skipping normalized plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    for equities, name, color in series:
        if not equities:
            continue
        start_val = equities[0] if equities[0] != 0 else initial_capital
        normalized = [e / start_val * 100 for e in equities]
        ax.plot(timestamps[:len(normalized)], normalized,
                linewidth=1.2, label=name, color=color)

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_ylabel("Growth (Start = 100)")
    title = "MES Backtest - Normalized Strategy Comparison"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Normalized comparison saved to %s", path)


def plot_drawdown_comparison(
    series: list[tuple],
    timestamps: list,
    path: str,
):
    """
    Chart C: Drawdown curves overlaid for comparison.

    Parameters
    ----------
    series : list of (equity_list, name, color)
    timestamps : full list of bar timestamps
    path : output file path
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed - skipping drawdown plot")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    for equities, name, color in series:
        if not equities:
            continue
        eq_arr = np.array(equities)
        peak = np.maximum.accumulate(eq_arr)
        dd = eq_arr - peak
        ts_range = timestamps[:len(dd)]
        ax.plot(ts_range, dd, linewidth=1.0, label=name, color=color, alpha=0.85)
        ax.fill_between(ts_range, dd, 0, color=color, alpha=0.08)

    ax.set_ylabel("Drawdown ($)")
    ax.set_title("MES Backtest - Drawdown Comparison")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Drawdown comparison saved to %s", path)
