"""Tests for the paper trading execution engine."""

import datetime
import pickle

import numpy as np
import pytest

from backtest.engine import Trade, EquityPoint
from config.settings import InstrumentConfig, BacktestConfig
from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal, StrategyEngine
from strategy.hybrid_ema_ml import HybridEMAMLConfig
from strategy.paper_engine import (
    PaperEngine,
    PaperConfig,
    PnLUpdate,
    PaperValidationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))


def _ts(hour: int, minute: int, day: int = 2) -> datetime.datetime:
    return datetime.datetime(2024, 1, day, hour, minute, tzinfo=_ET)


def _bar(ts, o, h, l, c, v=100):
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _inst() -> InstrumentConfig:
    return InstrumentConfig(symbol="MES", tick_size=0.25, point_value=5.0, contract_size=1)


def _cfg(capital=10_000.0, slip=1.0, comm=0.62) -> PaperConfig:
    return PaperConfig(slippage_ticks=slip, commission_per_side=comm, initial_capital=capital)


def _entry_signal(
    ts=None, direction="long", entry=5000.0, stop=4990.0, tp=5015.0,
    size=1.0, strategy_type="ema50_breakout", pos_id="pos_1",
):
    sig_type = SignalType.LONG_ENTRY if direction == "long" else SignalType.SHORT_ENTRY
    return LiveSignal(
        timestamp=ts or _ts(9, 45),
        direction=direction,
        signal_type=sig_type,
        entry=entry,
        stop=stop,
        take_profit=tp,
        position_size=size,
        strategy_type=strategy_type,
        position_id=pos_id,
    )


def _exit_signal(
    ts=None, exit_type=SignalType.EXIT_TP, price=5015.0,
    pos_id="pos_1", direction="long", stop=4990.0, tp=5015.0,
    strategy_type="ema50_breakout",
):
    return LiveSignal(
        timestamp=ts or _ts(10, 30),
        direction=direction,
        signal_type=exit_type,
        entry=price,   # exit price carried in entry field
        stop=stop,
        take_profit=tp,
        position_size=1.0,
        strategy_type=strategy_type,
        position_id=pos_id,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_config(self):
        eng = PaperEngine(_inst())
        assert eng.equity == 10_000.0
        assert eng.open_position_count == 0
        assert eng.trade_count == 0

    def test_custom_config(self):
        eng = PaperEngine(_inst(), _cfg(capital=50_000.0))
        assert eng.equity == 50_000.0

    def test_from_backtest_config(self):
        bt = BacktestConfig(slippage_ticks=2.0, commission_per_side=1.0, initial_capital=25_000.0)
        cfg = PaperConfig.from_backtest_config(bt)
        assert cfg.slippage_ticks == 2.0
        assert cfg.commission_per_side == 1.0
        assert cfg.initial_capital == 25_000.0


# ---------------------------------------------------------------------------
# Entry simulation
# ---------------------------------------------------------------------------

class TestEntrySimulation:
    def test_long_entry_opens_position(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal())
        assert eng.open_position_count == 1
        assert "pos_1" in eng.open_positions

    def test_short_entry_opens_position(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal(direction="short"))
        assert eng.open_position_count == 1

    def test_entry_deducts_costs(self):
        eng = PaperEngine(_inst(), _cfg(capital=10_000.0, slip=1.0, comm=0.62))
        initial = eng.equity
        eng.on_signal(_entry_signal(size=1.0))
        # slippage = 0.25 * 1.0 * 5.0 * 1 * 1.0 = 1.25
        # commission = 0.62 * 1 * 1.0 = 0.62
        expected_cost = 1.25 + 0.62
        assert eng.equity == pytest.approx(initial - expected_cost, abs=0.01)

    def test_duplicate_entry_ignored(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal(pos_id="dup"))
        eng.on_signal(_entry_signal(pos_id="dup"))
        assert eng.open_position_count == 1

    def test_entry_with_fractional_size(self):
        eng = PaperEngine(_inst(), _cfg(capital=10_000.0))
        eng.on_signal(_entry_signal(size=0.5))
        # Costs should scale with size
        # slippage = 0.25 * 1.0 * 5.0 * 1 * 0.5 = 0.625
        # commission = 0.62 * 1 * 0.5 = 0.31
        expected_cost = 0.625 + 0.31
        assert eng.equity == pytest.approx(10_000.0 - expected_cost, abs=0.01)


# ---------------------------------------------------------------------------
# Exit simulation
# ---------------------------------------------------------------------------

class TestExitSimulation:
    def test_long_tp_exit(self):
        eng = PaperEngine(_inst(), _cfg(capital=10_000.0))
        eng.on_signal(_entry_signal(entry=5000.0, size=1.0))
        eng.on_signal(_exit_signal(price=5015.0, exit_type=SignalType.EXIT_TP))

        assert eng.open_position_count == 0
        assert eng.trade_count == 1

        t = eng.trades[0]
        assert t.direction == "long"
        assert t.entry_price == 5000.0
        assert t.exit_price == 5015.0
        assert t.pnl_points == 15.0
        # pnl_dollars = 15 * 5 * 1 * 1.0 = 75
        assert t.pnl_dollars == pytest.approx(75.0)
        assert t.exit_reason == "Take Profit"

    def test_short_sl_exit(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal(direction="short", entry=5000.0, stop=5010.0, tp=4985.0))
        eng.on_signal(_exit_signal(
            price=5010.0, exit_type=SignalType.EXIT_SL,
            direction="short", pos_id="pos_1",
        ))

        t = eng.trades[0]
        assert t.direction == "short"
        assert t.pnl_points == -10.0  # 5000 - 5010

    def test_eod_exit(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal())
        eng.on_signal(_exit_signal(exit_type=SignalType.EXIT_EOD, price=5005.0))

        t = eng.trades[0]
        assert t.exit_reason == "End of Day"

    def test_exit_unknown_position_ignored(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_exit_signal(pos_id="nonexistent"))
        assert eng.trade_count == 0

    def test_exit_no_open_positions_ignored(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_exit_signal(pos_id=""))
        assert eng.trade_count == 0

    def test_legacy_single_position_exit(self):
        """Exit with empty pos_id should close the only open position."""
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal(pos_id="_single"))
        eng.on_signal(_exit_signal(pos_id="", price=5010.0))
        assert eng.open_position_count == 0
        assert eng.trade_count == 1

    def test_net_pnl_accounts_for_costs(self):
        eng = PaperEngine(_inst(), _cfg(slip=1.0, comm=0.62))
        eng.on_signal(_entry_signal(entry=5000.0, size=1.0))
        eng.on_signal(_exit_signal(price=5015.0))

        t = eng.trades[0]
        # Entry: slip=1.25 comm=0.62; Exit: slip=1.25 comm=0.62
        assert t.slippage_cost == pytest.approx(2.50)
        assert t.commission == pytest.approx(1.24)
        assert t.net_pnl == pytest.approx(t.pnl_dollars - t.slippage_cost - t.commission)


# ---------------------------------------------------------------------------
# Position tracking & equity curve
# ---------------------------------------------------------------------------

class TestPositionTracking:
    def test_multiple_positions(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal(pos_id="a", strategy_type="ema20_breakout"))
        eng.on_signal(_entry_signal(pos_id="b", strategy_type="ema50_breakout"))
        assert eng.open_position_count == 2

        eng.on_signal(_exit_signal(pos_id="a", price=5010.0, strategy_type="ema20_breakout"))
        assert eng.open_position_count == 1
        assert eng.trade_count == 1

    def test_equity_after_winning_trade(self):
        eng = PaperEngine(_inst(), _cfg(capital=10_000.0, slip=0.0, comm=0.0))
        eng.on_signal(_entry_signal(entry=5000.0, size=1.0))
        eng.on_signal(_exit_signal(price=5010.0))  # +10pts = +$50
        assert eng.equity == pytest.approx(10_050.0)

    def test_equity_after_losing_trade(self):
        eng = PaperEngine(_inst(), _cfg(capital=10_000.0, slip=0.0, comm=0.0))
        eng.on_signal(_entry_signal(entry=5000.0, size=1.0))
        eng.on_signal(_exit_signal(price=4990.0, exit_type=SignalType.EXIT_SL))
        assert eng.equity == pytest.approx(9_950.0)

    def test_mark_to_market_equity(self):
        eng = PaperEngine(_inst(), _cfg(capital=10_000.0, slip=0.0, comm=0.0))
        eng.on_signal(_entry_signal(entry=5000.0))
        eng.on_bar(_bar(_ts(10, 0), 5005, 5010, 5003, 5008))
        # unrealized: (5008-5000)*5*1*1 = 40
        assert eng.mark_to_market_equity == pytest.approx(10_040.0)

    def test_equity_curve_populated(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        eng.on_bar(_bar(_ts(9, 35), 5002, 5008, 4998, 5005))
        assert len(eng.equity_curve) == 2
        assert eng.equity_curve[0].timestamp == _ts(9, 30)

    def test_drawdown_tracking(self):
        eng = PaperEngine(_inst(), _cfg(capital=10_000.0, slip=0.0, comm=0.0))
        eng.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        eng.on_signal(_entry_signal(entry=5000.0))
        # After entry: last_bar close=5002, unrealized=(5002-5000)*5=+10, peak→10010
        # Price drops
        eng.on_bar(_bar(_ts(9, 35), 5000, 5000, 4990, 4990))
        # unrealized = (4990-5000)*5 = -50, mtm = 10000-50 = 9950
        # peak was 10010 (from entry update), dd = 9950-10010 = -60
        last = eng.equity_curve[-1]
        assert last.drawdown == pytest.approx(-60.0)


# ---------------------------------------------------------------------------
# PnL updates & callback
# ---------------------------------------------------------------------------

class TestPnLUpdates:
    def test_pending_updates_from_bar(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        updates = eng.pending_updates()
        assert len(updates) == 1
        assert isinstance(updates[0], PnLUpdate)
        assert updates[0].equity == pytest.approx(10_000.0)

    def test_pending_updates_drained(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        eng.pending_updates()
        assert len(eng.pending_updates()) == 0

    def test_update_from_entry(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        eng.pending_updates()  # drain bar update
        eng.on_signal(_entry_signal())
        updates = eng.pending_updates()
        assert len(updates) == 1
        assert updates[0].open_position_count == 1

    def test_update_from_exit(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_bar(_bar(_ts(9, 45), 5000, 5005, 4995, 5002))
        eng.on_signal(_entry_signal())
        eng.pending_updates()
        eng.on_signal(_exit_signal(price=5010.0))
        updates = eng.pending_updates()
        assert len(updates) == 1
        assert updates[0].open_position_count == 0
        assert updates[0].realized_pnl != 0.0

    def test_on_update_callback(self):
        received = []
        eng = PaperEngine(_inst(), _cfg(), on_update=received.append)
        eng.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        assert len(received) == 1
        assert isinstance(received[0], PnLUpdate)

    def test_realized_pnl_accumulates(self):
        eng = PaperEngine(_inst(), _cfg(slip=0.0, comm=0.0))
        eng.on_bar(_bar(_ts(9, 45), 5000, 5010, 4990, 5005))

        # Trade 1: +10pts = +$50
        eng.on_signal(_entry_signal(pos_id="a", entry=5000.0))
        eng.on_signal(_exit_signal(pos_id="a", price=5010.0))
        assert eng.realized_pnl == pytest.approx(50.0)

        # Trade 2: +5pts = +$25
        eng.on_signal(_entry_signal(pos_id="b", entry=5005.0, ts=_ts(10, 0)))
        eng.on_signal(_exit_signal(pos_id="b", price=5010.0, ts=_ts(10, 30)))
        assert eng.realized_pnl == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_state(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        eng.on_signal(_entry_signal())
        eng.on_signal(_exit_signal(price=5010.0))

        eng.reset()

        assert eng.equity == 10_000.0
        assert eng.open_position_count == 0
        assert eng.trade_count == 0
        assert len(eng.equity_curve) == 0
        assert len(eng.pending_updates()) == 0
        assert eng.realized_pnl == 0.0


# ---------------------------------------------------------------------------
# Validation: paper vs backtest
# ---------------------------------------------------------------------------

class TestPaperValidation:
    def test_identical_trades_pass(self):
        eng = PaperEngine(_inst(), _cfg(slip=1.0, comm=0.62))
        eng.on_bar(_bar(_ts(9, 45), 5000, 5010, 4990, 5005))
        eng.on_signal(_entry_signal(entry=5000.0))
        eng.on_signal(_exit_signal(price=5015.0))

        pt = eng.trades[0]
        # Construct a backtest trade with identical values
        bt_trade = Trade(
            entry_time=pt.entry_time,
            exit_time=pt.exit_time,
            entry_price=pt.entry_price,
            exit_price=pt.exit_price,
            stop_loss=pt.stop_loss,
            take_profit=pt.take_profit,
            direction=pt.direction,
            pnl_points=pt.pnl_points,
            pnl_dollars=pt.pnl_dollars,
            commission=pt.commission,
            slippage_cost=pt.slippage_cost,
            net_pnl=pt.net_pnl,
            exit_reason=pt.exit_reason,
            contracts=1,
            position_size=pt.position_size,
            strategy_type=pt.strategy_type,
        )

        result = eng.compare_with_backtest([bt_trade])
        assert result.passed
        assert result.matched == 1

    def test_pnl_mismatch_detected(self):
        eng = PaperEngine(_inst(), _cfg(slip=0.0, comm=0.0))
        eng.on_bar(_bar(_ts(9, 45), 5000, 5010, 4990, 5005))
        eng.on_signal(_entry_signal(entry=5000.0))
        eng.on_signal(_exit_signal(price=5010.0))

        bt_trade = Trade(
            entry_time=_ts(9, 45), exit_time=_ts(10, 30),
            entry_price=5000.0, exit_price=5010.0,
            stop_loss=4990.0, take_profit=5015.0,
            direction="long", pnl_points=10.0,
            pnl_dollars=50.0, commission=0.0,
            slippage_cost=0.0, net_pnl=999.0,  # wrong
            exit_reason="Take Profit", contracts=1,
            strategy_type="ema50_breakout",
        )

        result = eng.compare_with_backtest([bt_trade])
        assert not result.passed
        assert len(result.pnl_mismatches) > 0

    def test_extra_paper_trades_detected(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_bar(_bar(_ts(9, 45), 5000, 5010, 4990, 5005))
        eng.on_signal(_entry_signal())
        eng.on_signal(_exit_signal(price=5010.0))

        result = eng.compare_with_backtest([])
        assert not result.passed
        assert len(result.extra_paper) == 1

    def test_missing_paper_trades_detected(self):
        eng = PaperEngine(_inst(), _cfg())

        bt_trade = Trade(
            entry_time=_ts(9, 45), exit_time=_ts(10, 30),
            entry_price=5000.0, exit_price=5010.0,
            stop_loss=4990.0, take_profit=5015.0,
            direction="long", pnl_points=10.0, pnl_dollars=50.0,
            commission=1.24, slippage_cost=2.50, net_pnl=46.26,
            exit_reason="Take Profit", contracts=1,
            strategy_type="ema50_breakout",
        )

        result = eng.compare_with_backtest([bt_trade])
        assert not result.passed
        assert len(result.missing_paper) == 1


# ---------------------------------------------------------------------------
# Integration: wire StrategyEngine → PaperEngine via replay
# ---------------------------------------------------------------------------

def _make_model_pkl(path: str) -> None:
    """Create a dummy ML model pkl."""
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy="most_frequent")
    X = np.zeros((10, 35))
    y = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    clf.fit(X, y)
    feature_cols = [
        "f_price_ema", "f_price_ema_dist", "f_price_ema_dist_pct",
        "f_price_ema_slope", "f_price_bar_range", "f_price_rolling_range",
        "f_price_ret_12bar", "f_price_gap", "f_price_gap_pct",
        "f_vol_avg", "f_vol_relative", "f_vol_expansion",
        "f_vola_tr", "f_vola_atr", "f_vola_atr_norm", "f_vola_realized",
        "f_time_minutes_since_open", "f_time_minutes_since_range_close",
        "f_time_minutes_to_close", "f_time_weekday",
        "f_range_high", "f_range_low", "f_range_size",
        "f_range_dist_above", "f_range_dist_below", "f_range_size_vs_atr",
        "f_regime_trend_strength", "f_regime_trend_direction",
        "f_regime_compression", "f_regime_breakout_strength",
        "f_regime_vol_trend",
        "f_ema_distance", "f_ema_distance_pct",
        "f_risk_points", "f_range_vs_atr",
    ]
    with open(path, "wb") as f:
        pickle.dump({"model": clf, "feature_columns": feature_cols}, f)


@pytest.fixture
def model_path(tmp_path):
    path = str(tmp_path / "test_model.pkl")
    _make_model_pkl(path)
    return path


def _session_bars(base_price=5000.0, day=2):
    bars = []
    prev_day = day - 1
    for i, m in enumerate(range(0, 75, 5)):
        h_offset = 10 + m // 60
        m_offset = m % 60
        bars.append(_bar(
            _ts(h_offset, m_offset, prev_day),
            base_price + i * 0.5,
            base_price + i * 0.5 + 3,
            base_price + i * 0.5 - 2,
            base_price + i * 0.5 + 1,
            v=150,
        ))
    for m in [30, 35, 40]:
        bars.append(_bar(
            _ts(9, m, day),
            base_price, base_price + 5, base_price - 3, base_price + 2,
            v=200,
        ))
    bars.append(_bar(
        _ts(9, 45, day),
        base_price + 2, base_price + 8, base_price - 1, base_price + 6,
        v=250,
    ))
    for hour in range(10, 16):
        for minute in range(0, 60, 5):
            if hour == 15 and minute > 55:
                break
            bars.append(_bar(
                _ts(hour, minute, day),
                base_price + 6, base_price + 10, base_price + 4, base_price + 7,
                v=150,
            ))
    return bars


class TestIntegration:
    def test_strategy_engine_wired_to_paper(self, model_path):
        """Full integration: strategy → paper engine via on_signal callback."""
        inst = _inst()
        paper = PaperEngine(inst, _cfg(capital=10_000.0, slip=1.0, comm=0.62))

        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        strategy = StrategyEngine(cfg, on_signal=paper.on_signal)

        bars = _session_bars()
        for bar in bars:
            paper.on_bar(bar)
            strategy.on_bar(bar)

        # Should have at least 1 completed trade (entry + EOD exit)
        assert paper.trade_count >= 1
        assert len(paper.equity_curve) == len(bars)

        # All trades are valid
        for t in paper.trades:
            assert t.entry_price > 0
            assert t.exit_price > 0
            assert t.direction in ("long", "short")

    def test_two_runs_identical(self, model_path):
        """Two cold runs produce identical paper trades."""
        inst = _inst()
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        bars = _session_bars()

        def _run():
            paper = PaperEngine(inst, _cfg(slip=1.0, comm=0.62))
            strat = StrategyEngine(cfg, on_signal=paper.on_signal)
            for bar in bars:
                paper.on_bar(bar)
                strat.on_bar(bar)
            return paper.trades

        trades1 = _run()
        trades2 = _run()

        assert len(trades1) == len(trades2)
        for t1, t2 in zip(trades1, trades2):
            assert t1.entry_time == t2.entry_time
            assert t1.exit_time == t2.exit_time
            assert t1.entry_price == t2.entry_price
            assert t1.exit_price == t2.exit_price
            assert t1.net_pnl == pytest.approx(t2.net_pnl)

    def test_paper_equity_moves(self, model_path):
        """Paper equity should change from initial after trades."""
        inst = _inst()
        paper = PaperEngine(inst, _cfg(capital=10_000.0))
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        strat = StrategyEngine(cfg, on_signal=paper.on_signal)

        for bar in _session_bars():
            paper.on_bar(bar)
            strat.on_bar(bar)

        if paper.trade_count > 0:
            # Equity must differ from initial (costs alone guarantee this)
            assert paper.equity != 10_000.0

    def test_multi_candidate_paper_trading(self, model_path):
        """Multi-candidate mode tracks multiple positions correctly."""
        inst = _inst()
        paper = PaperEngine(inst, _cfg())
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=True,
            max_trades_per_day=3,
            ema_periods=(20, 50),
            entry_types=("breakout",),
            selection_strategy="priority",
            ml_threshold=0.0,
        )
        strat = StrategyEngine(cfg, on_signal=paper.on_signal)

        for bar in _session_bars():
            paper.on_bar(bar)
            strat.on_bar(bar)

        # In multi-candidate mode, at session end all positions should be closed
        assert paper.open_position_count == 0
        assert paper.trade_count >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_exit_before_bar(self):
        """on_signal before any on_bar still works (no last_bar)."""
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal())
        assert eng.open_position_count == 1
        # mark_to_market_equity without a bar should fallback to equity
        assert eng.mark_to_market_equity == eng.equity

    def test_zero_size_trade(self):
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal(size=0.0))
        eng.on_signal(_exit_signal(price=5010.0))
        t = eng.trades[0]
        assert t.pnl_dollars == 0.0
        assert t.commission == 0.0

    def test_peak_equity_tracks_unrealized(self):
        eng = PaperEngine(_inst(), _cfg(capital=10_000.0, slip=0.0, comm=0.0))
        eng.on_signal(_entry_signal(entry=5000.0))
        # Price rises
        eng.on_bar(_bar(_ts(10, 0), 5010, 5020, 5008, 5015))
        peak_after_rise = eng.peak_equity
        # Price falls
        eng.on_bar(_bar(_ts(10, 5), 5010, 5012, 4995, 4998))
        # Peak shouldn't decrease
        assert eng.peak_equity == peak_after_rise

    def test_entry_before_bar_no_corrupt_update(self):
        """_emit_update skips when _last_bar is None (no garbage equity)."""
        eng = PaperEngine(_inst(), _cfg())
        eng.on_signal(_entry_signal())
        assert eng.open_position_count == 1
        # No update emitted since no bar yet
        updates = eng.pending_updates()
        assert len(updates) == 0, (
            f"Expected no updates before first bar, got {len(updates)}"
        )

    def test_on_update_callback_exception_does_not_crash_engine(self):
        """Callback exception is caught; engine continues processing."""
        callback_calls = []

        def failing_callback(update):
            callback_calls.append(update)
            raise ValueError("Connection lost")

        eng = PaperEngine(_inst(), _cfg(), on_update=failing_callback)
        eng.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        # Engine should still work despite callback exception
        assert eng._bar_count == 1
        assert len(callback_calls) == 1  # Callback was invoked
        # Next bar should also work
        eng.on_bar(_bar(_ts(9, 35), 5002, 5008, 5000, 5005))
        assert eng._bar_count == 2
        assert len(callback_calls) == 2

    def test_on_update_callback_exception_on_entry(self):
        """Callback exception during entry doesn't halt signal processing."""
        def failing_callback(update):
            raise RuntimeError("Dashboard crash")

        eng = PaperEngine(_inst(), _cfg(), on_update=failing_callback)
        eng.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        eng.on_signal(_entry_signal())
        assert eng.open_position_count == 1

    def test_compare_with_backtest_handles_duplicate_keys(self):
        """Multiple trades with same (entry_time, direction, strategy_type)
        should all be matched, not silently overwritten."""
        eng = PaperEngine(_inst(), _cfg(slip=0.0, comm=0.0))
        eng.on_bar(_bar(_ts(9, 30), 5000, 5010, 4990, 5005))

        # Entry two positions at same time, same direction/strategy
        eng.on_signal(_entry_signal(
            ts=_ts(9, 30), pos_id="a", strategy_type="orb", entry=5000.0,
        ))
        eng.on_signal(_entry_signal(
            ts=_ts(9, 30), pos_id="b", strategy_type="orb", entry=5000.0,
        ))
        # Exit both at different prices
        eng.on_signal(_exit_signal(
            ts=_ts(10, 0), pos_id="a", price=5010.0, strategy_type="orb",
        ))
        eng.on_signal(_exit_signal(
            ts=_ts(10, 5), pos_id="b", price=5015.0, strategy_type="orb",
        ))
        assert eng.trade_count == 2

        # Create backtest trades with same collision-prone key
        bt_trades = [
            Trade(
                entry_time=_ts(9, 30), exit_time=_ts(10, 0),
                entry_price=5000.0, exit_price=5010.0,
                stop_loss=4990.0, take_profit=5015.0,
                direction="long", pnl_points=10.0, pnl_dollars=50.0,
                commission=0.0, slippage_cost=0.0, net_pnl=50.0,
                exit_reason="Take Profit", contracts=1, position_size=1.0,
                strategy_type="orb",
            ),
            Trade(
                entry_time=_ts(9, 30), exit_time=_ts(10, 5),
                entry_price=5000.0, exit_price=5015.0,
                stop_loss=4990.0, take_profit=5015.0,
                direction="long", pnl_points=15.0, pnl_dollars=75.0,
                commission=0.0, slippage_cost=0.0, net_pnl=75.0,
                exit_reason="Take Profit", contracts=1, position_size=1.0,
                strategy_type="orb",
            ),
        ]
        result = eng.compare_with_backtest(bt_trades)
        assert result.matched == 2, (
            f"Expected 2 matched, got {result.matched} "
            f"(extra={len(result.extra_paper)}, missing={len(result.missing_paper)})"
        )

    def test_reset_then_new_session(self):
        """After reset(), engine starts fresh without side effects."""
        eng = PaperEngine(_inst(), _cfg(slip=0.0, comm=0.0))
        eng.on_bar(_bar(_ts(9, 30), 5000, 5010, 4990, 5005))
        eng.on_signal(_entry_signal(entry=5000.0))
        eng.on_signal(_exit_signal(price=5010.0))
        session1_pnl = eng.trades[0].net_pnl
        assert session1_pnl > 0

        eng.reset()
        assert eng.trade_count == 0
        assert eng.open_position_count == 0
        assert eng.equity == eng._cfg.initial_capital

        # New session should work cleanly
        eng.on_bar(_bar(_ts(10, 0, day=3), 5005, 5015, 4995, 5010))
        eng.on_signal(_entry_signal(ts=_ts(10, 0, day=3), entry=5005.0))
        eng.on_signal(_exit_signal(ts=_ts(11, 0, day=3), price=5020.0))
        assert eng.trade_count == 1
        assert eng.equity == eng._cfg.initial_capital + eng.trades[0].pnl_dollars


# ---------------------------------------------------------------------------
# _signal_to_live exit price conversion
# ---------------------------------------------------------------------------

class TestSignalToLiveExitPrice:
    """Verify _signal_to_live correctly passes exit price for exit signals."""

    def test_exit_signal_carries_exit_price_not_entry_price(self):
        """CRITICAL: exit LiveSignal.entry must be exit fill price, not
        the original position entry price."""
        from strategy.orb import Signal
        from strategy.strategy_engine import _signal_to_live

        sig = Signal(
            signal_type=SignalType.EXIT_TP,
            price=5015.0,             # exit fill price
            timestamp=_ts(10, 30),
            reason="TP hit",
            entry_price=5000.0,       # original position entry
            stop_loss=4990.0,
            take_profit=5015.0,
        )
        live = _signal_to_live(sig)
        assert live.entry == 5015.0, (
            f"Exit LiveSignal.entry should be exit price 5015.0, "
            f"got {live.entry} (entry_price leak)"
        )

    def test_exit_sl_carries_sl_price(self):
        from strategy.orb import Signal
        from strategy.strategy_engine import _signal_to_live

        sig = Signal(
            signal_type=SignalType.EXIT_SL,
            price=4990.0,
            timestamp=_ts(10, 30),
            reason="SL hit",
            entry_price=5000.0,
            stop_loss=4990.0,
            take_profit=5015.0,
        )
        live = _signal_to_live(sig)
        assert live.entry == 4990.0

    def test_exit_eod_carries_close_price(self):
        from strategy.orb import Signal
        from strategy.strategy_engine import _signal_to_live

        sig = Signal(
            signal_type=SignalType.EXIT_EOD,
            price=5008.5,
            timestamp=_ts(16, 0),
            reason="EOD",
            entry_price=5000.0,
            stop_loss=4990.0,
            take_profit=5015.0,
        )
        live = _signal_to_live(sig)
        assert live.entry == 5008.5

    def test_entry_signal_still_carries_entry_price(self):
        """Entry signals should still use entry_price field."""
        from strategy.orb import Signal
        from strategy.strategy_engine import _signal_to_live

        sig = Signal(
            signal_type=SignalType.LONG_ENTRY,
            price=5002.0,              # bar price
            timestamp=_ts(9, 45),
            reason="Breakout",
            entry_price=5000.0,        # fill price
            stop_loss=4990.0,
            take_profit=5015.0,
        )
        live = _signal_to_live(sig)
        assert live.entry == 5000.0
