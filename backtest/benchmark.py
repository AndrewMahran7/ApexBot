"""
Benchmark strategies for comparison.
=====================================

Since MES is a futures contract, "buy and hold" is not the same concept
as with stocks. We implement several sensible benchmarks:

1. **Always-Long Benchmark**: Enter long at session open every day, exit
   at EOD. This answers: "What if I just went long every day without any
   filter?" It uses the same cost model as the strategy for fair comparison.

2. **Unfiltered ORB Benchmark**: Take every ORB breakout without the EMA
   filter. This isolates the value of the EMA filter.

3. **Flat Benchmark**: No trades. Equity stays constant. Answers: "Am I
   better off doing nothing?"

4. **EMA Directional Benchmark**: Go long when close > EMA at session open,
   flat otherwise. A simple trend-following baseline.

All benchmarks operate on the same bar data and cost assumptions.
"""

from __future__ import annotations
import datetime
import logging
from typing import Optional

import pandas as pd
import numpy as np

from config.settings import InstrumentConfig, BacktestConfig, StrategyConfig
from strategy.orb import _parse_time

logger = logging.getLogger(__name__)


def always_long_benchmark(
    bars: pd.DataFrame,
    instrument: InstrumentConfig,
    bt_config: BacktestConfig,
    strat_config: StrategyConfig,
) -> list[float]:
    """
    Simple always-long-during-session benchmark.

    Enters at session open bar's open price, exits at EOD exit time bar's close.
    Returns a list of equity values aligned 1:1 with bars index for plotting.

    This is a fair futures benchmark: you pay commissions and slippage
    each day, so it is not free to hold.
    """
    session_start = _parse_time(strat_config.or_start)
    eod_exit = _parse_time(strat_config.eod_exit_time)

    slip = instrument.tick_size * bt_config.slippage_ticks * instrument.point_value
    comm = bt_config.commission_per_side
    contracts = instrument.contract_size

    equity = bt_config.initial_capital
    equities: list[float] = []

    in_position = False
    entry_price: Optional[float] = None
    current_date: Optional[datetime.date] = None

    for ts, row in bars.iterrows():
        bar_date = ts.date()
        bar_time = ts.time()

        # New day
        if current_date is None or bar_date != current_date:
            # Force close any overnight (shouldn't happen but defensive)
            if in_position and entry_price is not None:
                logger.warning("always_long: force-closing overnight position from %s", current_date)
                pnl = (float(row["open"]) - entry_price) * instrument.point_value * contracts
                equity += pnl - slip * contracts - comm * contracts
                in_position = False
                entry_price = None
            current_date = bar_date

        # Enter at first bar of session
        if not in_position and bar_time >= session_start:
            entry_price = float(row["open"])
            equity -= (slip + comm) * contracts  # entry costs
            in_position = True

        # Exit at EOD
        if in_position and bar_time >= eod_exit and entry_price is not None:
            pnl = (float(row["close"]) - entry_price) * instrument.point_value * contracts
            equity += pnl - (slip + comm) * contracts  # exit costs
            in_position = False
            entry_price = None

        equities.append(equity)

    return equities


def unfiltered_orb_benchmark(
    bars: pd.DataFrame,
    instrument: InstrumentConfig,
    bt_config: BacktestConfig,
    strat_config: StrategyConfig,
) -> list[float]:
    """
    ORB strategy without EMA filter — to isolate its contribution.
    Uses the same ORB logic but with ema_enabled=False.
    """
    from config.settings import StrategyConfig as SC
    from backtest.engine import BacktestEngine
    import copy

    no_ema_cfg = copy.copy(strat_config)
    no_ema_cfg.ema_enabled = False

    engine = BacktestEngine(instrument, no_ema_cfg, bt_config)
    result = engine.run(bars)

    # Build equity list aligned to bars
    eq_map = {ep.timestamp: ep.equity for ep in result.equity_curve}
    equities = []
    last_eq = bt_config.initial_capital
    for ts in bars.index:
        if ts in eq_map:
            last_eq = eq_map[ts]
        equities.append(last_eq)

    return equities


def flat_benchmark(
    bars: pd.DataFrame,
    bt_config: BacktestConfig,
) -> list[float]:
    """
    No-trade baseline. Equity stays at initial capital.

    Answers: "Am I better off doing nothing?"
    """
    return [bt_config.initial_capital] * len(bars)


def ema_directional_benchmark(
    bars: pd.DataFrame,
    instrument: InstrumentConfig,
    bt_config: BacktestConfig,
    strat_config: StrategyConfig,
    ema_length: int = 50,
) -> list[float]:
    """
    Simple EMA trend-following benchmark.

    At the first bar of each session:
      - If close > EMA -> enter long at open, exit at EOD.
      - Otherwise -> flat for the day.

    Uses the same cost model as the strategy.
    """
    session_start = _parse_time(strat_config.or_start)
    eod_exit = _parse_time(strat_config.eod_exit_time)

    slip = instrument.tick_size * bt_config.slippage_ticks * instrument.point_value
    comm = bt_config.commission_per_side
    contracts = instrument.contract_size

    equity = bt_config.initial_capital
    equities: list[float] = []

    # EMA state
    ema: Optional[float] = None
    ema_alpha = 2.0 / (ema_length + 1)
    bar_count = 0

    in_position = False
    entry_price: Optional[float] = None
    current_date: Optional[datetime.date] = None
    decided_today = False

    for ts, row in bars.iterrows():
        bar_date = ts.date()
        bar_time = ts.time()
        close = float(row["close"])

        # Update EMA
        bar_count += 1
        if ema is None:
            if bar_count >= 10:
                ema = close
        else:
            ema = close * ema_alpha + ema * (1 - ema_alpha)

        # New day
        if current_date is None or bar_date != current_date:
            # Force close any position from previous day
            if in_position and entry_price is not None:
                logger.warning("ema_directional: force-closing overnight position from %s", current_date)
                pnl = (float(row["open"]) - entry_price) * instrument.point_value * contracts
                equity += pnl - (slip + comm) * contracts
                in_position = False
                entry_price = None
            current_date = bar_date
            decided_today = False

        # Decision at first session bar
        if not decided_today and bar_time >= session_start:
            decided_today = True
            if ema is not None and close > ema:
                entry_price = float(row["open"])
                equity -= (slip + comm) * contracts
                in_position = True

        # Exit at EOD
        if in_position and bar_time >= eod_exit and entry_price is not None:
            pnl = (float(row["close"]) - entry_price) * instrument.point_value * contracts
            equity += pnl - (slip + comm) * contracts
            in_position = False
            entry_price = None

        equities.append(equity)

    return equities


def orb_benchmark(
    bars: pd.DataFrame,
    instrument: InstrumentConfig,
    bt_config: BacktestConfig,
    strat_config: StrategyConfig,
) -> tuple:
    """
    Run the full ORB strategy as a formal benchmark.

    Uses the same data, cost model, and capital as the main strategy.

    Returns
    -------
    tuple of (equity_list, BacktestResult)
        equity_list is aligned 1:1 with bars index for plotting.
    """
    from backtest.engine import BacktestEngine
    from strategy.orb import ORBStrategy

    strategy = ORBStrategy(strat_config)
    engine = BacktestEngine(instrument, strat_config, bt_config, strategy=strategy)
    result = engine.run(bars)

    # Build equity list aligned 1:1 with bars index
    eq_map = {ep.timestamp: ep.equity for ep in result.equity_curve}
    equities: list[float] = []
    last_eq = bt_config.initial_capital
    for ts in bars.index:
        if ts in eq_map:
            last_eq = eq_map[ts]
        equities.append(last_eq)

    return equities, result
