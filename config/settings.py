"""
Configuration for MES Futures ORB Backtesting Framework
========================================================

All configurable parameters live here. Changing strategy behavior
should NOT require editing strategy or engine code.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InstrumentConfig:
    """Futures contract specification."""
    symbol: str = "MES"
    tick_size: float = 0.25        # MES minimum tick
    point_value: float = 5.0       # $5 per point for MES (micro)
    contract_size: int = 1         # number of contracts to trade
    description: str = "Micro E-mini S&P 500"

    @property
    def tick_value(self) -> float:
        """Dollar value of one tick = tick_size * point_value."""
        return self.tick_size * self.point_value


@dataclass
class StrategyConfig:
    """ORB strategy parameters — all times in strategy timezone."""
    # Session / timezone
    timezone: str = "America/New_York"

    # Opening range window (HH:MM in strategy timezone)
    or_start: str = "09:30"         # default: US equity open
    or_end: str = "09:45"           # 15-minute opening range

    # EMA filter
    ema_length: int = 50
    ema_enabled: bool = True

    # Risk management
    reward_risk_ratio: float = 1.5
    one_trade_per_day: bool = True

    # Stop / target logic
    stop_type: str = "or_low"       # 'or_low' = bottom of opening range
    target_type: str = "rr_multiple" # 'rr_multiple' = reward:risk * range

    # End-of-day flat
    eod_exit_time: str = "15:50"    # force flat at this time

    # Direction
    shorts_enabled: bool = False    # allow short entries on breakdown below OR low

    # Filters
    min_range_points: float = 0.0   # minimum opening range size in points (0 = no filter)
    max_entry_time: str = ""        # latest allowed entry time (HH:MM, empty = no limit)


@dataclass
class AdaptiveRegimeConfig:
    """Adaptive Regime Breakout strategy parameters."""
    # Session / timezone
    timezone: str = "America/New_York"

    # Range window (HH:MM in strategy timezone)
    range_start_time: str = "09:30"
    range_end_time: str = "09:45"

    # Entry limits
    max_entry_time: str = "14:00"
    one_trade_per_day: bool = True

    # Direction
    allow_long: bool = True
    allow_short: bool = True

    # EMA trend filter
    ema_length: int = 50
    ema_enabled: bool = True
    ema_slope_enabled: bool = True
    ema_slope_lookback: int = 5

    # Range size filter (points)
    min_range_points: float = 2.0
    max_range_points: float = 20.0

    # Breakout buffer (points added beyond range high/low)
    breakout_buffer_points: float = 1.0

    # Volume filter
    volume_filter_enabled: bool = True
    volume_lookback: int = 20
    volume_threshold_ratio: float = 0.8

    # ATR filter
    atr_filter_enabled: bool = True
    atr_length: int = 14
    atr_min_threshold: float = 1.0

    # Risk management
    reward_risk: float = 2.0
    stop_mode: str = "range"         # 'range' = opposite side of range
    target_mode: str = "rr_multiple"  # 'rr_multiple' = reward_risk * range

    # End-of-day flat
    end_of_day_exit_time: str = "15:50"

    # Regime classification thresholds
    regime_atr_lookback: int = 20
    regime_ema_slope_threshold: float = 0.15   # min EMA slope for trend
    regime_range_ratio_threshold: float = 1.5  # OR/ATR ratio for breakout detection
    regime_volume_ratio_threshold: float = 1.0 # vol/avg for dead detection
    regime_dead_atr_ratio: float = 0.4         # OR/ATR below this = dead

    # Confirmation scoring — shared baseline
    min_confirmation_score: int = 4            # fallback if per-direction scores not set

    # Asymmetric thresholds (long vs short)
    long_min_score: Optional[int] = None       # None = use min_confirmation_score
    short_min_score: Optional[int] = 6         # shorts need stronger confirmation
    short_breakout_buffer_points: Optional[float] = 1.5  # wider buffer for shorts
    short_ema_slope_min: Optional[float] = 0.05  # min abs(slope) for short entries

    # Minimum breakout strength (points beyond buffer the bar must exceed)
    min_breakout_strength: float = 0.5

    # Strict-shorts mode: raise short bar even further
    strict_shorts: bool = False
    strict_short_min_score: int = 8            # override short_min_score when strict
    strict_short_buffer: float = 2.0           # override short buffer when strict
    strict_short_ema_slope_min: float = 0.20   # min abs(EMA slope) when strict


@dataclass
class BacktestConfig:
    """Execution simulation and output settings."""
    # Cost model
    slippage_ticks: float = 1.0     # slippage per side in ticks
    commission_per_side: float = 0.62  # CME + NFA + broker typical for MES

    # Initial account
    initial_capital: float = 10_000.0

    # Output paths
    results_dir: str = "results"
    trades_csv: str = "trades.csv"
    metrics_file: str = "metrics.json"
    equity_csv: str = "equity_curve.csv"
    plot_file: str = "equity_curve.png"

    # Bar data
    bar_interval_minutes: int = 5   # expected bar size (informational)


@dataclass
class EvalConfig:
    """Prop-firm style evaluation account settings."""
    enabled: bool = False
    starting_capital: float = 25_000.0
    profit_target: float = 1_500.0
    max_drawdown: float = 1_000.0
    drawdown_type: str = "trailing_intraday"  # 'trailing_intraday'


# ------------------------------------------------------------------
# Instrument registry for multi-symbol support
# ------------------------------------------------------------------

INSTRUMENT_REGISTRY: dict[str, InstrumentConfig] = {
    "MES": InstrumentConfig(
        symbol="MES",
        tick_size=0.25,
        point_value=5.0,
        contract_size=1,
        description="Micro E-mini S&P 500",
    ),
    "MNQ": InstrumentConfig(
        symbol="MNQ",
        tick_size=0.25,
        point_value=2.0,
        contract_size=1,
        description="Micro E-mini Nasdaq-100",
    ),
    "RTY": InstrumentConfig(
        symbol="RTY",
        tick_size=0.10,
        point_value=5.0,
        contract_size=1,
        description="Micro E-mini Russell 2000",
    ),
}


def compute_contracts(
    equity: float,
    risk_per_trade: float,
    stop_ticks: float,
    tick_value: float,
    max_contracts: int = 5,
) -> int:
    """Risk-based contract sizing.

    contracts = floor(risk_dollars / (stop_ticks * tick_value))
    Clamped to [1, max_contracts].
    """
    if stop_ticks <= 0 or tick_value <= 0:
        return 1
    risk_dollars = equity * risk_per_trade
    raw = risk_dollars / (stop_ticks * tick_value)
    return max(1, min(max_contracts, int(raw)))
