"""Tests for strategy.prop_challenge — PropConfig, PropEquityTracker, PropRiskGate, PropPositionSizer."""

import datetime

import pytest

from strategy.prop_challenge import (
    PropConfig,
    PropEquityTracker,
    PropRiskGate,
    PropPositionSizer,
    PropEvent,
)
from strategy.strategy_engine import LiveSignal
from strategy.orb import SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))


def _ts(day: int = 2, hour: int = 10, minute: int = 0) -> datetime.datetime:
    return datetime.datetime(2024, 1, day, hour, minute, tzinfo=_ET)


def _bar(day: int = 2, hour: int = 10, minute: int = 0,
         close: float = 5000.0, volume: float = 100.0) -> dict:
    return {
        "timestamp": _ts(day, hour, minute),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": volume,
    }


def _entry_signal(
    day: int = 2,
    hour: int = 10,
    minute: int = 5,
    direction: str = "long",
    strategy_type: str = "ema50_breakout",
    ml_prob: float = 0.70,
    entry: float = 5000.0,
    stop: float = 4990.0,
    tp: float = 5015.0,
    size: float = 1.0,
) -> LiveSignal:
    sig_type = SignalType.LONG_ENTRY if direction == "long" else SignalType.SHORT_ENTRY
    return LiveSignal(
        timestamp=_ts(day, hour, minute),
        direction=direction,
        signal_type=sig_type,
        entry=entry,
        stop=stop,
        take_profit=tp,
        position_size=size,
        strategy_type=strategy_type,
        ml_prob=ml_prob,
    )


def _exit_signal(day: int = 2, hour: int = 10, minute: int = 30) -> LiveSignal:
    return LiveSignal(
        timestamp=_ts(day, hour, minute),
        direction="",
        signal_type=SignalType.EXIT_TP,
        entry=5015.0,
        stop=0.0,
        take_profit=0.0,
        position_size=0.0,
        strategy_type="ema50_breakout",
    )


def _default_config(**overrides) -> PropConfig:
    defaults = dict(
        starting_capital=25_000.0,
        profit_target=1_500.0,
        max_drawdown=1_000.0,
        daily_loss_limit=300.0,
        daily_profit_lock=400.0,
        max_trades_per_day=4,
        max_consecutive_losses=3,
        allowed_entry_types=("breakout",),
        min_ml_prob=0.60,
    )
    defaults.update(overrides)
    return PropConfig(**defaults)


# ---------------------------------------------------------------------------
# PropEquityTracker
# ---------------------------------------------------------------------------

class TestPropEquityTracker:
    def test_initial_state(self):
        cfg = _default_config()
        t = PropEquityTracker(cfg)
        assert t.current_equity == 25_000.0
        assert t.peak_equity == 25_000.0
        assert t.trailing_dd_level == 24_000.0
        assert t.active is True
        assert t.passed is False
        assert t.failed is False

    def test_equity_gain(self):
        t = PropEquityTracker(_default_config())
        t.update(25_500.0, _ts())
        assert t.equity_gain == 500.0

    def test_peak_tracks_upward(self):
        t = PropEquityTracker(_default_config())
        t.update(25_800.0, _ts())
        t.update(25_500.0, _ts(2, 11))
        assert t.peak_equity == 25_800.0
        assert t.trailing_dd_level == 24_800.0

    def test_pass_on_target(self):
        t = PropEquityTracker(_default_config())
        t.update(26_500.0, _ts())
        assert t.passed is True
        assert t.active is False

    def test_fail_on_dd_breach(self):
        t = PropEquityTracker(_default_config())
        t.update(24_000.0, _ts())
        assert t.failed is True
        assert t.active is False

    def test_trailing_dd_moves_up(self):
        t = PropEquityTracker(_default_config())
        t.update(26_000.0, _ts())  # peak -> 26k, dd -> 25k
        assert t.trailing_dd_level == 25_000.0
        t.update(25_100.0, _ts(2, 11))  # drops but above 25k
        assert t.failed is False
        t.update(25_000.0, _ts(2, 12))
        assert t.failed is True

    def test_daily_pnl_tracking(self):
        t = PropEquityTracker(_default_config())
        t.update(25_000.0, _ts(2, 9, 30))  # day start
        t.update(25_200.0, _ts(2, 10))
        assert t.daily_pnl == 200.0
        assert t.daily_peak_pnl == 200.0
        t.update(25_100.0, _ts(2, 11))
        assert t.daily_pnl == 100.0
        assert t.daily_peak_pnl == 200.0

    def test_day_change_resets_daily(self):
        t = PropEquityTracker(_default_config())
        t.update(25_200.0, _ts(2, 10))
        t.update(25_200.0, _ts(3, 10))  # new day
        assert t.daily_pnl == 0.0

    def test_dd_buffer_remaining(self):
        cfg = _default_config()
        t = PropEquityTracker(cfg)
        # equity 25k, dd_level 24k → buffer = 1000
        assert t.dd_buffer_remaining == 1_000.0
        t.update(24_500.0, _ts())
        assert t.dd_buffer_remaining == 500.0


# ---------------------------------------------------------------------------
# PropPositionSizer
# ---------------------------------------------------------------------------

class TestPropPositionSizer:
    def test_small_tier(self):
        s = PropPositionSizer(_default_config())
        assert s.compute(100.0) == 0.25

    def test_medium_tier(self):
        s = PropPositionSizer(_default_config())
        assert s.compute(700.0) == 0.50

    def test_lock_in_tier(self):
        s = PropPositionSizer(_default_config())
        assert s.compute(1300.0) == 0.35

    def test_negative_gain(self):
        s = PropPositionSizer(_default_config())
        # Below 0 — falls through tiers; last tier returned
        assert s.compute(-100.0) == 0.35  # fallback to last

    def test_at_boundary(self):
        s = PropPositionSizer(_default_config())
        assert s.compute(500.0) == 0.50  # exactly at medium start
        assert s.compute(1200.0) == 0.35  # exactly at lock-in start


# ---------------------------------------------------------------------------
# PropRiskGate — entry filtering
# ---------------------------------------------------------------------------

class TestPropRiskGateFiltering:
    def setup_method(self):
        self.approved = []
        self.gate = PropRiskGate(
            config=_default_config(),
            on_approved=lambda sig: self.approved.append(sig),
        )

    def test_exit_always_passes(self):
        self.gate.on_signal(_exit_signal())
        assert len(self.approved) == 1

    def test_breakout_entry_passes(self):
        sig = _entry_signal(strategy_type="ema50_breakout", ml_prob=0.70)
        self.gate.on_bar(_bar())
        self.gate.on_signal(sig)
        assert len(self.approved) == 1

    def test_pullback_blocked(self):
        sig = _entry_signal(strategy_type="ema50_pullback", ml_prob=0.70)
        self.gate.on_bar(_bar())
        self.gate.on_signal(sig)
        assert len(self.approved) == 0
        assert any("filter" in e.event_type for e in self.gate.events)

    def test_momentum_blocked(self):
        sig = _entry_signal(strategy_type="ema50_momentum", ml_prob=0.70)
        self.gate.on_bar(_bar())
        self.gate.on_signal(sig)
        assert len(self.approved) == 0

    def test_low_ml_prob_blocked(self):
        sig = _entry_signal(ml_prob=0.40)
        self.gate.on_bar(_bar())
        self.gate.on_signal(sig)
        assert len(self.approved) == 0

    def test_ml_prob_exactly_at_threshold(self):
        sig = _entry_signal(ml_prob=0.60)
        self.gate.on_bar(_bar())
        self.gate.on_signal(sig)
        assert len(self.approved) == 1

    def test_max_trades_per_day(self):
        cfg = _default_config(max_trades_per_day=2)
        gate = PropRiskGate(config=cfg, on_approved=lambda s: self.approved.append(s))
        gate.on_bar(_bar())
        gate.on_signal(_entry_signal(minute=5))
        gate.on_signal(_entry_signal(minute=10))
        gate.on_signal(_entry_signal(minute=15))
        assert len(self.approved) == 2


# ---------------------------------------------------------------------------
# PropRiskGate — daily controls
# ---------------------------------------------------------------------------

class TestPropRiskGateDailyControls:
    def setup_method(self):
        self.approved = []

    def test_daily_loss_stops_trading(self):
        cfg = _default_config(daily_loss_limit=300.0)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: self.approved.append(s),
            get_equity=lambda: 24_600.0,  # down 400 from 25k
        )
        gate.on_bar(_bar())  # triggers daily loss check
        gate.on_signal(_entry_signal())
        assert len(self.approved) == 0
        assert gate.day_stopped is True

    def test_daily_profit_lock(self):
        cfg = _default_config(daily_profit_lock=400.0)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: self.approved.append(s),
            get_equity=lambda: 25_500.0,  # up 500
        )
        gate.on_bar(_bar())
        gate.on_signal(_entry_signal())
        assert len(self.approved) == 0
        assert any(e.event_type == "prop_stopped_day" for e in gate.events)

    def test_no_giveback_rule(self):
        cfg = _default_config(giveback_threshold=300.0, giveback_drop=200.0)
        equity_seq = iter([25_400.0, 25_150.0])
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: self.approved.append(s),
            get_equity=lambda: next(equity_seq),
        )
        # First bar: up 400, peak 400
        gate.on_bar(_bar(hour=10))
        # Second bar: up 150, peak was 400, drop = 250 > 200
        gate.on_bar(_bar(hour=11))
        assert gate.day_stopped is True

    def test_day_change_resets_stop(self):
        cfg = _default_config(daily_loss_limit=300.0, daily_profit_lock=9999.0)
        equity_val = [24_600.0]
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: self.approved.append(s),
            get_equity=lambda: equity_val[0],
        )
        gate.on_bar(_bar(day=2))  # day stopped: daily = 24600 - 25000 = -400
        assert gate.day_stopped is True
        # New day — equity stays the same but daily PnL resets to 0
        gate.on_bar(_bar(day=3))  # new day: day_start = 24600, daily = 0
        assert gate.day_stopped is False


# ---------------------------------------------------------------------------
# PropRiskGate — halt conditions
# ---------------------------------------------------------------------------

class TestPropRiskGateHalt:
    def setup_method(self):
        self.approved = []

    def test_target_reached_halts(self):
        cfg = _default_config(profit_target=500.0)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: self.approved.append(s),
            get_equity=lambda: 25_600.0,
        )
        gate.on_bar(_bar())
        gate.on_signal(_entry_signal())
        assert len(self.approved) == 0
        assert gate.halted is True
        assert any(e.event_type == "prop_target_reached" for e in gate.events)

    def test_dd_breach_halts(self):
        cfg = _default_config(max_drawdown=500.0)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: self.approved.append(s),
            get_equity=lambda: 24_400.0,
        )
        gate.on_bar(_bar())
        assert gate.halted is True
        assert any(e.event_type == "prop_drawdown_breach" for e in gate.events)

    def test_consecutive_losses_halt(self):
        cfg = _default_config(max_consecutive_losses=2)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: self.approved.append(s),
        )
        gate.on_bar(_bar())
        gate.on_trade_closed(-10.0)
        gate.on_trade_closed(-15.0)
        gate.on_signal(_entry_signal())
        assert len(self.approved) == 0
        assert gate.day_stopped is True

    def test_win_resets_consecutive_losses(self):
        cfg = _default_config(max_consecutive_losses=3)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: self.approved.append(s),
        )
        gate.on_bar(_bar())
        gate.on_trade_closed(-10.0)
        gate.on_trade_closed(-15.0)
        gate.on_trade_closed(50.0)  # reset
        assert gate.consecutive_losses == 0
        gate.on_signal(_entry_signal())
        assert len(self.approved) == 1


# ---------------------------------------------------------------------------
# PropRiskGate — position sizing
# ---------------------------------------------------------------------------

class TestPropRiskGateSizing:
    def test_small_size_in_early_stage(self):
        approved = []
        gate = PropRiskGate(
            config=_default_config(),
            on_approved=lambda s: approved.append(s),
            get_equity=lambda: 25_200.0,  # gain = 200 → small tier
        )
        gate.on_bar(_bar())
        gate.on_signal(_entry_signal(size=1.0))
        assert len(approved) == 1
        assert approved[0].position_size == pytest.approx(0.25)

    def test_medium_size_in_mid_stage(self):
        approved = []
        cfg = _default_config(daily_profit_lock=9999.0)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: approved.append(s),
            get_equity=lambda: 25_700.0,  # gain = 700 → medium
        )
        gate.on_bar(_bar())
        gate.on_signal(_entry_signal(size=1.0))
        assert len(approved) == 1
        assert approved[0].position_size == pytest.approx(0.50)

    def test_lock_in_size_near_target(self):
        approved = []
        cfg = _default_config(daily_profit_lock=9999.0)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: approved.append(s),
            get_equity=lambda: 26_300.0,  # gain = 1300 → lock-in
        )
        gate.on_bar(_bar())
        gate.on_signal(_entry_signal(size=1.0))
        assert len(approved) == 1
        assert approved[0].position_size == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# PropRiskGate — exit tightening
# ---------------------------------------------------------------------------

class TestPropRiskGateExitTightening:
    def test_stop_tightened(self):
        approved = []
        cfg = _default_config(stop_tightening_pct=0.85, reward_risk_override=1.2)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: approved.append(s),
        )
        gate.on_bar(_bar())
        # orig SL=10pts, TP=15pts from entry
        sig = _entry_signal(entry=5000.0, stop=4990.0, tp=5015.0)
        gate.on_signal(sig)
        assert len(approved) == 1
        adj = approved[0]
        # tightened SL dist = 10 * 0.85 = 8.5; new TP dist = 8.5 * 1.2 = 10.2
        assert adj.stop == pytest.approx(5000.0 - 8.5)  # 4991.5
        assert adj.take_profit == pytest.approx(5000.0 + 10.2)  # 5010.2

    def test_short_stop_tightened(self):
        approved = []
        cfg = _default_config(stop_tightening_pct=0.85, reward_risk_override=1.2)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: approved.append(s),
        )
        gate.on_bar(_bar())
        sig = _entry_signal(direction="short", entry=5000.0, stop=5010.0, tp=4985.0)
        gate.on_signal(sig)
        adj = approved[0]
        # SL dist = 10 * 0.85 = 8.5
        assert adj.stop == pytest.approx(5008.5)
        assert adj.take_profit == pytest.approx(5000.0 - 10.2)


# ---------------------------------------------------------------------------
# PropRiskGate — DD buffer
# ---------------------------------------------------------------------------

class TestPropRiskGateDDBuffer:
    def test_dd_buffer_stops_trading(self):
        approved = []
        cfg = _default_config(dd_buffer=200.0)
        gate = PropRiskGate(
            config=cfg,
            on_approved=lambda s: approved.append(s),
            get_equity=lambda: 24_150.0,  # dd_level=24k, buffer=150 < 200
        )
        gate.on_bar(_bar())
        gate.on_signal(_entry_signal())
        assert len(approved) == 0
        assert gate.day_stopped is True


# ---------------------------------------------------------------------------
# PropRiskGate — entry type extraction
# ---------------------------------------------------------------------------

class TestExtractEntryType:
    def test_standard(self):
        assert PropRiskGate._extract_entry_type("ema50_breakout") == "breakout"

    def test_multi_underscore(self):
        assert PropRiskGate._extract_entry_type("ema50_pullback") == "pullback"

    def test_no_underscore(self):
        assert PropRiskGate._extract_entry_type("breakout") == "breakout"


# ---------------------------------------------------------------------------
# PropEvent dataclass
# ---------------------------------------------------------------------------

class TestPropEvent:
    def test_creation(self):
        e = PropEvent(
            timestamp=_ts(),
            event_type="prop_blocked",
            reason="test",
        )
        assert e.event_type == "prop_blocked"
        assert e.details == {}


# ---------------------------------------------------------------------------
# main.py wiring — CLI and build_pipeline
# ---------------------------------------------------------------------------

class TestMainPropWiring:
    def test_prop_mode_flag(self):
        from main import parse_args
        args = parse_args(["--mode", "replay", "--prop-mode"])
        assert args.prop_mode is True

    def test_prop_mode_defaults(self):
        from main import parse_args
        args = parse_args(["--mode", "replay"])
        assert args.prop_mode is False

    def test_prop_target_arg(self):
        from main import parse_args
        args = parse_args(["--mode", "replay", "--prop-mode",
                           "--prop-target", "2000"])
        assert args.prop_target == 2000.0

    def test_prop_daily_loss_arg(self):
        from main import parse_args
        args = parse_args(["--mode", "replay", "--prop-mode",
                           "--prop-daily-loss", "250"])
        assert args.prop_daily_loss == 250.0

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_build_pipeline_with_prop(self):
        from main import build_pipeline
        from config.settings import InstrumentConfig
        from strategy.hybrid_ema_ml import HybridEMAMLConfig
        from strategy.paper_engine import PaperConfig
        from strategy.risk_manager import RiskConfig

        inst = InstrumentConfig(
            symbol="MES", tick_size=0.25, point_value=5.0, contract_size=1,
        )
        scfg = HybridEMAMLConfig(
            multi_candidate=False, ema_periods=(50,),
            entry_types=("breakout",), model_path="models/ema_model.pkl",
        )
        rcfg = RiskConfig(max_daily_loss=300.0, max_trades_per_day=4)
        pcfg = PaperConfig(initial_capital=25_000.0)
        prop = PropRiskGate(config=_default_config())

        pipeline = build_pipeline(
            mode="replay", instrument=inst, strategy_cfg=scfg,
            risk_cfg=rcfg, paper_cfg=pcfg, prop_gate=prop,
        )
        assert pipeline["prop_gate"] is prop

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_build_pipeline_without_prop(self):
        from main import build_pipeline
        from config.settings import InstrumentConfig
        from strategy.hybrid_ema_ml import HybridEMAMLConfig
        from strategy.paper_engine import PaperConfig
        from strategy.risk_manager import RiskConfig

        inst = InstrumentConfig(
            symbol="MES", tick_size=0.25, point_value=5.0, contract_size=1,
        )
        scfg = HybridEMAMLConfig(
            multi_candidate=False, ema_periods=(50,),
            entry_types=("breakout",), model_path="models/ema_model.pkl",
        )
        rcfg = RiskConfig(max_daily_loss=500.0, max_trades_per_day=6)
        pcfg = PaperConfig(initial_capital=10_000.0)

        pipeline = build_pipeline(
            mode="replay", instrument=inst, strategy_cfg=scfg,
            risk_cfg=rcfg, paper_cfg=pcfg,
        )
        assert pipeline["prop_gate"] is None
