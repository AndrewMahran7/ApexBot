"""Tests for intraday strategies and MultiStrategyEngine."""

import datetime
import pytest

from strategy.orb import SignalType
from strategy.intraday_strategies import (
    IntradayConfig,
    IntradayStrategyRunner,
    VWAPBounce,
    IntradayMomentum,
    MeanReversion,
    _IndicatorState,
    _PositionTracker,
    _OpenPosition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))  # EDT


def _ts(hour: int, minute: int, day: int = 2) -> datetime.datetime:
    return datetime.datetime(2024, 1, day, hour, minute, tzinfo=_ET)


def _bar(ts, o, h, l, c, v=1000):
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _warmup_bars(n=25, base=5000.0, day=2):
    """Generate n quiet bars to warm up indicators."""
    bars = []
    for i in range(n):
        minute = 30 + (i * 5) % 60
        hour = 9 + (30 + i * 5) // 60
        ts = _ts(hour, minute % 60, day)
        price = base + i * 0.25
        bars.append(_bar(ts, price, price + 1, price - 1, price, 1000))
    return bars


# ---------------------------------------------------------------------------
# _IndicatorState
# ---------------------------------------------------------------------------

class TestIndicatorState:

    def test_vwap_calculation(self):
        ind = _IndicatorState(atr_period=5, rsi_period=5)
        bars = [
            _bar(_ts(9, 30), 100, 102, 99, 101, 1000),
            _bar(_ts(9, 35), 101, 103, 100, 102, 1500),
            _bar(_ts(9, 40), 102, 104, 101, 103, 2000),
        ]
        for b in bars:
            ind.update(b)

        assert ind.vwap > 0
        # VWAP should be near the midpoint of these bars
        assert 100 < ind.vwap < 104

    def test_vwap_resets_daily(self):
        ind = _IndicatorState(atr_period=3, rsi_period=3)
        # Day 1
        b1 = _bar(_ts(9, 30, day=2), 100, 102, 99, 101, 1000)
        ind.update(b1)
        vwap_day1 = ind.vwap

        # Day 2 â€” different price level
        b2 = _bar(_ts(9, 30, day=3), 200, 202, 199, 201, 1000)
        ind.update(b2)
        vwap_day2 = ind.vwap

        # VWAP should have reset to day 2 level
        assert vwap_day2 > 190

    def test_atr_ready_after_enough_bars(self):
        ind = _IndicatorState(atr_period=5, rsi_period=14)
        for i in range(10):
            h = 9 + (30 + i * 5) // 60
            m = (30 + i * 5) % 60
            b = _bar(_ts(h, m), 100 + i, 102 + i, 99 + i, 101 + i, 1000)
            ind.update(b)

        assert ind.atr > 0

    def test_rsi_range(self):
        ind = _IndicatorState(atr_period=5, rsi_period=5)
        # Feed enough bars for RSI to initialise
        for i in range(15):
            h = 9 + (30 + i * 5) // 60
            m = (30 + i * 5) % 60
            b = _bar(_ts(h, m), 100 + i, 102 + i, 99 + i, 101 + i, 1000)
            ind.update(b)

        assert 0 <= ind.rsi <= 100

    def test_ready_flag(self):
        ind = _IndicatorState(atr_period=5, rsi_period=5)
        assert not ind.ready

        for i in range(25):
            h = 9 + (30 + i * 5) // 60
            m = (30 + i * 5) % 60
            b = _bar(_ts(h, m), 100 + i, 102 + i, 99 + i, 101 + i, 1000)
            ind.update(b)

        assert ind.ready


# ---------------------------------------------------------------------------
# _PositionTracker
# ---------------------------------------------------------------------------

class TestPositionTracker:

    def test_add_and_count(self):
        tracker = _PositionTracker("15:50")
        pos = _OpenPosition(
            position_id="test_1",
            direction="long",
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=103.0,
            strategy_type="vwap_bounce",
            entry_time=_ts(10, 0),
        )
        tracker.add(pos)
        assert tracker.count == 1

    def test_stop_loss_exit_long(self):
        tracker = _PositionTracker("15:50")
        pos = _OpenPosition(
            position_id="test_sl",
            direction="long",
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=103.0,
            strategy_type="vwap_bounce",
            entry_time=_ts(10, 0),
        )
        tracker.add(pos)

        # Bar that hits the stop
        bar = _bar(_ts(10, 5), 99, 99, 97, 97.5)
        exits = tracker.check_exits(bar)
        assert len(exits) == 1
        assert exits[0].signal_type == SignalType.EXIT_SL
        assert tracker.count == 0

    def test_take_profit_exit_long(self):
        tracker = _PositionTracker("15:50")
        pos = _OpenPosition(
            position_id="test_tp",
            direction="long",
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=103.0,
            strategy_type="vwap_bounce",
            entry_time=_ts(10, 0),
        )
        tracker.add(pos)

        bar = _bar(_ts(10, 5), 101, 104, 100, 103.5)
        exits = tracker.check_exits(bar)
        assert len(exits) == 1
        assert exits[0].signal_type == SignalType.EXIT_TP
        assert tracker.count == 0

    def test_eod_exit(self):
        tracker = _PositionTracker("15:50")
        pos = _OpenPosition(
            position_id="test_eod",
            direction="short",
            entry_price=100.0,
            stop_loss=102.0,
            take_profit=97.0,
            strategy_type="mean_reversion",
            entry_time=_ts(10, 0),
        )
        tracker.add(pos)

        bar = _bar(_ts(15, 50), 100, 101, 99, 100)
        exits = tracker.check_exits(bar)
        assert len(exits) == 1
        assert exits[0].signal_type == SignalType.EXIT_EOD

    def test_stop_loss_exit_short(self):
        tracker = _PositionTracker("15:50")
        pos = _OpenPosition(
            position_id="test_sl_short",
            direction="short",
            entry_price=100.0,
            stop_loss=102.0,
            take_profit=97.0,
            strategy_type="mean_reversion",
            entry_time=_ts(10, 0),
        )
        tracker.add(pos)

        bar = _bar(_ts(10, 5), 101, 103, 100, 102.5)
        exits = tracker.check_exits(bar)
        assert len(exits) == 1
        assert exits[0].signal_type == SignalType.EXIT_SL


# ---------------------------------------------------------------------------
# IntradayStrategyRunner
# ---------------------------------------------------------------------------

class TestIntradayStrategyRunner:

    def test_returns_signals_list(self):
        cfg = IntradayConfig()
        runner = IntradayStrategyRunner(config=cfg)
        bar = _bar(_ts(10, 0), 5000, 5002, 4998, 5001, 1000)
        result = runner.on_bar(bar)
        assert isinstance(result, list)

    def test_no_signals_before_warmup(self):
        """Strategies should produce no entries until indicators are ready."""
        cfg = IntradayConfig()
        runner = IntradayStrategyRunner(config=cfg)
        bar = _bar(_ts(10, 0), 5000, 5002, 4998, 5001, 1000)
        result = runner.on_bar(bar)
        assert len(result) == 0

    def test_signal_has_required_fields(self):
        """Any emitted signal must have the required Signal fields."""
        cfg = IntradayConfig(atr_period=5, rsi_period=5, entry_cooldown_bars=1)
        runner = IntradayStrategyRunner(config=cfg)

        # Warmup
        for i in range(30):
            h = 9 + (30 + i * 5) // 60
            m = (30 + i * 5) % 60
            price = 5000 + i * 0.5
            bar = _bar(_ts(h, m), price, price + 2, price - 2, price + 0.5, 1000)
            sigs = runner.on_bar(bar)

        # All signals should have the required attributes
        all_sigs = []
        for i in range(50):
            h = 10 + (i * 5) // 60
            m = (i * 5) % 60
            price = 5020 + i * 0.1
            vol = 3000 if i % 10 == 0 else 500
            bar = _bar(_ts(h, m), price - 1, price + 3, price - 3, price, vol)
            sigs = runner.on_bar(bar)
            all_sigs.extend(sigs)

        for sig in all_sigs:
            assert sig.signal_type is not None
            assert sig.strategy_type in ("vwap_bounce", "intraday_momentum", "mean_reversion")
            assert sig.position_id != ""

    def test_reset(self):
        cfg = IntradayConfig()
        runner = IntradayStrategyRunner(config=cfg)
        bar = _bar(_ts(10, 0), 5000, 5002, 4998, 5001)
        runner.on_bar(bar)
        runner.reset()
        # Should not crash after reset
        result = runner.on_bar(bar)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# VWAPBounce
# ---------------------------------------------------------------------------

class TestVWAPBounce:

    def test_no_entry_before_entry_start(self):
        cfg = IntradayConfig(entry_start="10:00", atr_period=3, rsi_period=3)
        ind = _IndicatorState(atr_period=3, rsi_period=3)
        strat = VWAPBounce(cfg, ind)

        # Feed bars before 10:00
        for i in range(10):
            b = _bar(_ts(9, 30 + i * 3), 5000 + i, 5002 + i, 4998 + i, 5001 + i, 1000)
            ind.update(b)
            sigs = strat.on_bar(b)
            entries = [s for s in sigs if s.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY)]
            assert len(entries) == 0

    def test_no_entry_after_entry_end(self):
        cfg = IntradayConfig(entry_end="15:30", atr_period=3, rsi_period=3)
        ind = _IndicatorState(atr_period=3, rsi_period=3)
        strat = VWAPBounce(cfg, ind)

        for i in range(10):
            b = _bar(_ts(15, 30 + i), 5000, 5002, 4998, 5001, 1000)
            ind.update(b)
            sigs = strat.on_bar(b)
            entries = [s for s in sigs if s.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY)]
            assert len(entries) == 0


# ---------------------------------------------------------------------------
# IntradayMomentum
# ---------------------------------------------------------------------------

class TestIntradayMomentum:

    def test_no_entry_without_volume(self):
        """Should not enter if volume is below threshold."""
        cfg = IntradayConfig(
            atr_period=3, rsi_period=3,
            momentum_volume_mult=2.0,
            entry_cooldown_bars=1,
        )
        ind = _IndicatorState(atr_period=3, rsi_period=3)
        strat = IntradayMomentum(cfg, ind)

        # Warmup with normal volume
        for i in range(10):
            b = _bar(_ts(10, i * 5), 5000 + i, 5002 + i, 4998 + i, 5001 + i, 1000)
            ind.update(b)
            strat.on_bar(b)

        # Breakout bar with LOW volume
        big_bar = _bar(_ts(10, 55), 5010, 5015, 5010, 5014, 100)
        ind.update(big_bar)
        sigs = strat.on_bar(big_bar)
        entries = [s for s in sigs if s.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY)]
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# MeanReversion
# ---------------------------------------------------------------------------

class TestMeanReversion:

    def test_requires_extreme_rsi_and_deviation(self):
        cfg = IntradayConfig(
            atr_period=3, rsi_period=3,
            rsi_oversold=30, rsi_overbought=70,
            vwap_deviation_threshold=1.5,
            entry_cooldown_bars=1,
        )
        ind = _IndicatorState(atr_period=3, rsi_period=3)
        strat = MeanReversion(cfg, ind)

        # Flat market â€” RSI should be neutral
        for i in range(15):
            h = 10 + (i * 5) // 60
            m = (i * 5) % 60
            b = _bar(_ts(h, m), 5000, 5001, 4999, 5000, 1000)
            ind.update(b)
            sigs = strat.on_bar(b)
            entries = [s for s in sigs if s.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY)]
            assert len(entries) == 0


# ---------------------------------------------------------------------------
# PropRiskGate integration
# ---------------------------------------------------------------------------

class TestPropEntryType:

    def test_intraday_types_pass_through(self):
        from risk.prop_challenge import PropRiskGate
        assert PropRiskGate._extract_entry_type("vwap_bounce") == "vwap_bounce"
        assert PropRiskGate._extract_entry_type("intraday_momentum") == "intraday_momentum"
        assert PropRiskGate._extract_entry_type("mean_reversion") == "mean_reversion"

    def test_legacy_types_still_work(self):
        from risk.prop_challenge import PropRiskGate
        assert PropRiskGate._extract_entry_type("ema50_breakout") == "breakout"
        assert PropRiskGate._extract_entry_type("ema20_momentum") == "momentum"


# ---------------------------------------------------------------------------
# MultiStrategyEngine
# ---------------------------------------------------------------------------

class TestMultiStrategyEngine:

    @pytest.fixture(autouse=True)
    def _check_pandas(self):
        try:
            import pandas  # noqa: F401
        except Exception:
            pytest.skip("pandas/pyarrow unavailable")

    def test_on_bar_returns_list(self):
        from strategy.hybrid_ema_ml import HybridEMAMLConfig
        from strategy.multi_strategy_engine import MultiStrategyEngine
        cfg = HybridEMAMLConfig()
        engine = MultiStrategyEngine(
            strategy_cfg=cfg,
            intraday_cfg=IntradayConfig(),
            enable_hybrid=False,
            enable_intraday=True,
        )
        bar = _bar(_ts(10, 0), 5000, 5002, 4998, 5001)
        result = engine.on_bar(bar)
        assert isinstance(result, list)

    def test_reset_works(self):
        from strategy.hybrid_ema_ml import HybridEMAMLConfig
        from strategy.multi_strategy_engine import MultiStrategyEngine
        cfg = HybridEMAMLConfig()
        engine = MultiStrategyEngine(
            strategy_cfg=cfg,
            intraday_cfg=IntradayConfig(),
            enable_hybrid=False,
            enable_intraday=True,
        )
        bar = _bar(_ts(10, 0), 5000, 5002, 4998, 5001)
        engine.on_bar(bar)
        engine.reset()
        assert engine.bar_count == 0

    def test_callback_fired(self):
        from strategy.hybrid_ema_ml import HybridEMAMLConfig
        from strategy.multi_strategy_engine import MultiStrategyEngine
        from strategy.strategy_engine import LiveSignal
        cfg = HybridEMAMLConfig()
        received = []
        engine = MultiStrategyEngine(
            strategy_cfg=cfg,
            intraday_cfg=IntradayConfig(),
            on_signal=lambda sig: received.append(sig),
            enable_hybrid=False,
            enable_intraday=True,
        )
        # Feed enough bars to potentially generate a signal
        for i in range(50):
            h = 10 + (i * 5) // 60
            m = (i * 5) % 60
            bar = _bar(_ts(h, m), 5000 + i, 5005 + i, 4995 + i, 5002 + i, 3000)
            engine.on_bar(bar)

        # All received signals should be LiveSignal
        for sig in received:
            assert isinstance(sig, LiveSignal)
