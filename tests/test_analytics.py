"""Tests for analytics.engine — AnalyticsEngine, report computation, helpers."""

import datetime
import json
import math
import threading
from pathlib import Path

import numpy as np
import pytest

from analytics.engine import (
    AnalyticsEngine,
    AnalyticsReport,
    StrategyBreakdown,
    _profit_factor,
    _sharpe_from_trades,
    _max_drawdown_from_pnls,
    _breakdown_by_strategy,
    _daily_pnl,
)
from strategy.strategy_engine import LiveSignal
from strategy.risk_manager import RiskEvent
from strategy.orb import SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))


def _ts(day: int = 2, hour: int = 10, minute: int = 0) -> datetime.datetime:
    return datetime.datetime(2024, 1, day, hour, minute, tzinfo=_ET)


class _FakeTrade:
    """Minimal object matching Trade attribute access for record_trade."""
    def __init__(
        self,
        net_pnl: float,
        direction: str = "long",
        strategy_type: str = "ema50_breakout",
        day: int = 2,
        hour: int = 10,
        minute: int = 0,
    ):
        self.entry_time = _ts(day, hour, minute)
        self.exit_time = _ts(day, hour, minute + 5)
        self.direction = direction
        self.entry_price = 5000.0
        self.exit_price = 5000.0 + net_pnl
        self.net_pnl = net_pnl
        self.pnl_dollars = net_pnl + 2.0  # before costs
        self.commission = 1.0
        self.slippage_cost = 1.0
        self.exit_reason = "take_profit" if net_pnl > 0 else "stop_loss"
        self.strategy_type = strategy_type
        self.position_size = 1.0


def _entry_signal(day: int = 2, hour: int = 10) -> LiveSignal:
    return LiveSignal(
        timestamp=_ts(day, hour),
        direction="long",
        signal_type=SignalType.LONG_ENTRY,
        entry=5000.0,
        stop=4990.0,
        take_profit=5015.0,
        position_size=1.0,
        strategy_type="ema50_breakout",
    )


def _exit_signal(day: int = 2, hour: int = 10) -> LiveSignal:
    return LiveSignal(
        timestamp=_ts(day, hour, 30),
        direction="",
        signal_type=SignalType.EXIT_TP,
        entry=5015.0,
        stop=0.0,
        take_profit=0.0,
        position_size=0.0,
        strategy_type="ema50_breakout",
    )


def _risk_event(event_type: str = "blocked") -> RiskEvent:
    return RiskEvent(
        timestamp=_ts(),
        event_type=event_type,
        reason="Daily loss limit",
    )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

class TestEmptyState:
    def test_empty_report(self):
        ana = AnalyticsEngine()
        r = ana.report()
        assert r.total_trades == 0
        assert r.win_rate == 0.0
        assert r.profit_factor == 0.0
        assert r.sharpe_ratio == 0.0

    def test_empty_counts(self):
        ana = AnalyticsEngine()
        assert ana.trade_count == 0
        assert ana.signal_count == 0
        assert ana.decision_count == 0

    def test_empty_lists(self):
        ana = AnalyticsEngine()
        assert ana.trades == []
        assert ana.signals == []
        assert ana.decisions == []


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_trade(self):
        ana = AnalyticsEngine()
        ana.record_trade(_FakeTrade(50.0))
        assert ana.trade_count == 1
        assert ana.trades[0]["net_pnl"] == 50.0

    def test_record_signal(self):
        ana = AnalyticsEngine()
        ana.record_signal(_entry_signal())
        assert ana.signal_count == 1
        assert ana.signals[0]["is_entry"] is True

    def test_record_decision(self):
        ana = AnalyticsEngine()
        ana.record_decision(_risk_event("blocked"))
        assert ana.decision_count == 1
        assert ana.decisions[0]["event_type"] == "blocked"

    def test_multiple_trades(self):
        ana = AnalyticsEngine()
        for pnl in [50.0, -20.0, 30.0]:
            ana.record_trade(_FakeTrade(pnl))
        assert ana.trade_count == 3

    def test_record_preserves_fields(self):
        ana = AnalyticsEngine()
        t = _FakeTrade(100.0, direction="short", strategy_type="ema21_pullback")
        ana.record_trade(t)
        rec = ana.trades[0]
        assert rec["direction"] == "short"
        assert rec["strategy_type"] == "ema21_pullback"
        assert rec["commission"] == 1.0
        assert rec["slippage_cost"] == 1.0
        assert rec["exit_reason"] == "take_profit"


# ---------------------------------------------------------------------------
# Single trade report
# ---------------------------------------------------------------------------

class TestSingleTradeReport:
    def test_win(self):
        ana = AnalyticsEngine(initial_capital=10_000.0)
        ana.record_trade(_FakeTrade(50.0))
        r = ana.report()
        assert r.total_trades == 1
        assert r.win_count == 1
        assert r.loss_count == 0
        assert r.win_rate == 100.0
        assert r.total_pnl == 50.0
        assert r.profit_factor == float("inf")

    def test_loss(self):
        ana = AnalyticsEngine()
        ana.record_trade(_FakeTrade(-30.0))
        r = ana.report()
        assert r.win_count == 0
        assert r.loss_count == 1
        assert r.win_rate == 0.0
        assert r.profit_factor == 0.0
        assert r.total_pnl == -30.0


# ---------------------------------------------------------------------------
# Multi-trade report
# ---------------------------------------------------------------------------

class TestMultiTradeReport:
    def setup_method(self):
        self.ana = AnalyticsEngine(initial_capital=10_000.0)
        trades = [
            _FakeTrade(50.0, "long", "ema50_breakout", day=2, hour=10),
            _FakeTrade(-20.0, "long", "ema50_breakout", day=2, hour=11),
            _FakeTrade(30.0, "short", "ema21_pullback", day=3, hour=10),
            _FakeTrade(-10.0, "short", "ema21_pullback", day=3, hour=11),
            _FakeTrade(40.0, "long", "ema50_breakout", day=4, hour=10),
        ]
        for t in trades:
            self.ana.record_trade(t)

    def test_counts(self):
        r = self.ana.report()
        assert r.total_trades == 5
        assert r.win_count == 3
        assert r.loss_count == 2

    def test_win_rate(self):
        r = self.ana.report()
        assert r.win_rate == pytest.approx(60.0)

    def test_total_pnl(self):
        r = self.ana.report()
        assert r.total_pnl == pytest.approx(90.0)

    def test_avg_pnl(self):
        r = self.ana.report()
        assert r.avg_pnl == pytest.approx(18.0)

    def test_profit_factor(self):
        r = self.ana.report()
        # gains: 50 + 30 + 40 = 120. losses: 20 + 10 = 30. PF = 4.0
        assert r.profit_factor == pytest.approx(4.0)

    def test_largest_win_loss(self):
        r = self.ana.report()
        assert r.largest_win == pytest.approx(50.0)
        assert r.largest_loss == pytest.approx(-20.0)

    def test_direction(self):
        r = self.ana.report()
        assert r.long_trades == 3
        assert r.short_trades == 2
        assert r.long_pnl == pytest.approx(70.0)
        assert r.short_pnl == pytest.approx(20.0)

    def test_daily_pnl(self):
        r = self.ana.report()
        assert r.trading_days == 3
        # day 2: 50-20=30, day 3: 30-10=20, day 4: 40
        assert r.winning_days == 3
        assert r.losing_days == 0


# ---------------------------------------------------------------------------
# Strategy breakdown
# ---------------------------------------------------------------------------

class TestStrategyBreakdown:
    def test_grouped(self):
        ana = AnalyticsEngine()
        ana.record_trade(_FakeTrade(50.0, strategy_type="ema50_breakout"))
        ana.record_trade(_FakeTrade(-20.0, strategy_type="ema50_breakout"))
        ana.record_trade(_FakeTrade(30.0, strategy_type="ema21_pullback"))
        r = ana.report()
        assert len(r.by_strategy) == 2

        # Sorted alphabetically
        ema21 = r.by_strategy[0]
        ema50 = r.by_strategy[1]
        assert ema21["strategy_type"] == "ema21_pullback"
        assert ema21["trade_count"] == 1
        assert ema50["strategy_type"] == "ema50_breakout"
        assert ema50["trade_count"] == 2

    def test_per_strategy_metrics(self):
        ana = AnalyticsEngine()
        ana.record_trade(_FakeTrade(50.0, strategy_type="ema50_breakout"))
        ana.record_trade(_FakeTrade(-20.0, strategy_type="ema50_breakout"))
        r = ana.report()
        ema50 = r.by_strategy[0]
        assert ema50["win_count"] == 1
        assert ema50["loss_count"] == 1
        assert ema50["win_rate"] == 50.0
        assert ema50["total_pnl"] == 30.0
        assert ema50["profit_factor"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Signals and decisions in report
# ---------------------------------------------------------------------------

class TestSignalDecisionReport:
    def test_signals_counted(self):
        ana = AnalyticsEngine()
        ana.record_signal(_entry_signal())
        ana.record_signal(_exit_signal())
        r = ana.report()
        assert r.total_signals == 2
        assert r.entry_signals == 1
        assert r.exit_signals == 1

    def test_decisions_counted(self):
        ana = AnalyticsEngine()
        ana.record_decision(_risk_event("blocked"))
        ana.record_decision(_risk_event("capped"))
        ana.record_decision(_risk_event("kill_switch"))
        r = ana.report()
        assert r.risk_blocked == 1
        assert r.risk_capped == 1
        assert r.risk_kill_switches == 1

    def test_signal_to_trade_rate(self):
        ana = AnalyticsEngine()
        ana.record_signal(_entry_signal(day=2, hour=10))
        ana.record_signal(_entry_signal(day=2, hour=11))
        ana.record_signal(_exit_signal(day=2, hour=12))
        ana.record_trade(_FakeTrade(50.0))
        r = ana.report()
        # 1 trade / 2 entry signals = 50%
        assert r.signal_to_trade_rate == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_normal(self):
        pnls = np.array([50.0, -20.0, 30.0, -10.0])
        assert _profit_factor(pnls) == pytest.approx(80.0 / 30.0)

    def test_no_losses(self):
        pnls = np.array([50.0, 30.0])
        assert _profit_factor(pnls) == float("inf")

    def test_no_wins(self):
        pnls = np.array([-20.0, -10.0])
        assert _profit_factor(pnls) == 0.0

    def test_all_zero(self):
        pnls = np.array([0.0, 0.0])
        assert _profit_factor(pnls) == 0.0


class TestSharpe:
    def test_positive_returns(self):
        pnls = np.array([10.0, 20.0, 15.0, 25.0, 10.0])
        result = _sharpe_from_trades(pnls, 10_000.0)
        assert result > 0

    def test_single_trade_returns_zero(self):
        pnls = np.array([50.0])
        assert _sharpe_from_trades(pnls, 10_000.0) == 0.0

    def test_zero_std_returns_zero(self):
        pnls = np.array([10.0, 10.0, 10.0])
        assert _sharpe_from_trades(pnls, 10_000.0) == 0.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        pnls = np.array([10.0, 20.0, 30.0])
        dd = _max_drawdown_from_pnls(pnls, 10_000.0)
        assert dd == pytest.approx(0.0)

    def test_drawdown(self):
        pnls = np.array([100.0, -50.0, -80.0, 200.0])
        dd = _max_drawdown_from_pnls(pnls, 10_000.0)
        # Equity: 10100, 10050, 9970, 10170
        # Peak:   10100, 10100, 10100, 10170
        # DD:     0, -50, -130, 0
        assert dd == pytest.approx(-130.0)

    def test_empty_returns_zero(self):
        pnls = np.array([])
        dd = _max_drawdown_from_pnls(pnls, 10_000.0)
        assert dd == 0.0


class TestDailyPnl:
    def test_groups_by_date(self):
        trades = [
            {"exit_time": "2024-01-02 10:05:00", "net_pnl": 50.0},
            {"exit_time": "2024-01-02 11:05:00", "net_pnl": -20.0},
            {"exit_time": "2024-01-03 10:05:00", "net_pnl": 30.0},
        ]
        daily, days, win_days, lose_days = _daily_pnl(trades)
        assert days == 2
        assert daily[0] == {"date": "2024-01-02", "pnl": 30.0}
        assert daily[1] == {"date": "2024-01-03", "pnl": 30.0}
        assert win_days == 2
        assert lose_days == 0

    def test_losing_day(self):
        trades = [
            {"exit_time": "2024-01-02 10:00:00", "net_pnl": -50.0},
        ]
        daily, days, win_days, lose_days = _daily_pnl(trades)
        assert days == 1
        assert win_days == 0
        assert lose_days == 1


class TestBreakdownByStrategy:
    def test_multiple_strategies(self):
        trades = [
            {"strategy_type": "A", "net_pnl": 50.0},
            {"strategy_type": "A", "net_pnl": -20.0},
            {"strategy_type": "B", "net_pnl": 30.0},
        ]
        result = _breakdown_by_strategy(trades)
        assert len(result) == 2
        assert result[0]["strategy_type"] == "A"
        assert result[1]["strategy_type"] == "B"


# ---------------------------------------------------------------------------
# to_dict serialisation
# ---------------------------------------------------------------------------

class TestToDict:
    def test_keys(self):
        r = AnalyticsReport()
        d = r.to_dict()
        assert "overall" in d
        assert "direction" in d
        assert "by_strategy" in d
        assert "signals" in d
        assert "daily" in d

    def test_roundtrip_json(self):
        r = AnalyticsReport(total_trades=5, win_rate=60.0, profit_factor=2.5)
        d = r.to_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded["overall"]["total_trades"] == 5


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_creates_file(self, tmp_path):
        ana = AnalyticsEngine()
        ana.record_trade(_FakeTrade(50.0))
        out = tmp_path / "analytics.json"
        ana.export_json(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["overall"]["total_trades"] == 1

    def test_creates_parent_dirs(self, tmp_path):
        ana = AnalyticsEngine()
        out = tmp_path / "sub" / "dir" / "report.json"
        ana.export_json(str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_writes(self):
        ana = AnalyticsEngine()
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    ana.record_trade(_FakeTrade(float(i)))
                    ana.record_signal(_entry_signal())
                    ana.record_decision(_risk_event())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert ana.trade_count == 400
        assert ana.signal_count == 400
        assert ana.decision_count == 400


# ---------------------------------------------------------------------------
# main.py wiring — CLI and build_pipeline
# ---------------------------------------------------------------------------

class TestMainAnalyticsWiring:
    def test_analytics_cli_flag(self):
        from main import parse_args
        args = parse_args(["--mode", "replay", "--analytics"])
        assert args.analytics is True

    def test_analytics_output_default(self):
        from main import parse_args
        args = parse_args(["--mode", "replay"])
        assert args.analytics_output == "results/analytics.json"

    def test_analytics_output_custom(self):
        from main import parse_args
        args = parse_args(["--mode", "replay", "--analytics-output", "out.json"])
        assert args.analytics_output == "out.json"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_build_pipeline_with_analytics(self):
        from main import build_pipeline
        from config.settings import InstrumentConfig
        from strategy.hybrid_ema_ml import HybridEMAMLConfig
        from strategy.paper_engine import PaperConfig
        from strategy.risk_manager import RiskConfig

        inst = InstrumentConfig(
            symbol="MES", tick_size=0.25, point_value=5.0, contract_size=1,
        )
        scfg = HybridEMAMLConfig(
            multi_candidate=False,
            ema_periods=(50,),
            entry_types=("breakout",),
            model_path="models/ema_model.pkl",
        )
        rcfg = RiskConfig(max_daily_loss=500.0, max_trades_per_day=6)
        pcfg = PaperConfig(initial_capital=10_000.0)

        ana = AnalyticsEngine(initial_capital=10_000.0)
        pipeline = build_pipeline(
            mode="replay",
            instrument=inst,
            strategy_cfg=scfg,
            risk_cfg=rcfg,
            paper_cfg=pcfg,
            analytics=ana,
        )
        assert pipeline["analytics"] is ana

    def test_build_pipeline_without_analytics(self):
        from main import build_pipeline
        from config.settings import InstrumentConfig
        from strategy.hybrid_ema_ml import HybridEMAMLConfig
        from strategy.paper_engine import PaperConfig
        from strategy.risk_manager import RiskConfig

        inst = InstrumentConfig(
            symbol="MES", tick_size=0.25, point_value=5.0, contract_size=1,
        )
        scfg = HybridEMAMLConfig(
            multi_candidate=False,
            ema_periods=(50,),
            entry_types=("breakout",),
            model_path="models/ema_model.pkl",
        )
        rcfg = RiskConfig(max_daily_loss=500.0, max_trades_per_day=6)
        pcfg = PaperConfig(initial_capital=10_000.0)

        pipeline = build_pipeline(
            mode="replay",
            instrument=inst,
            strategy_cfg=scfg,
            risk_cfg=rcfg,
            paper_cfg=pcfg,
        )
        assert pipeline["analytics"] is None
