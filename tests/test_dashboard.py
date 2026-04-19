"""Tests for the dashboard state store and FastAPI app."""

import datetime
import logging
import threading
from unittest.mock import MagicMock

import pytest

from dashboard.state import DashboardState, MAX_EQUITY_POINTS
from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal
from strategy.paper_engine import PnLUpdate
from risk.risk_manager import RiskEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))


def _ts(hour: int = 10, minute: int = 0) -> datetime.datetime:
    return datetime.datetime(2024, 1, 2, hour, minute, tzinfo=_ET)


def _pnl_update(
    equity=10_000.0,
    unrealized=0.0,
    realized=0.0,
    dd=0.0,
    open_count=0,
    ts=None,
) -> PnLUpdate:
    return PnLUpdate(
        timestamp=ts or _ts(),
        equity=equity,
        unrealized_pnl=unrealized,
        realized_pnl=realized,
        open_position_count=open_count,
        drawdown=dd,
    )


def _entry_signal(ts=None) -> LiveSignal:
    return LiveSignal(
        timestamp=ts or _ts(),
        direction="long",
        signal_type=SignalType.LONG_ENTRY,
        entry=5000.0, stop=4990.0, take_profit=5015.0,
        position_size=1.0, strategy_type="ema50_breakout",
        position_id="pos_1",
    )


def _mock_trade(net_pnl=50.0, direction="long"):
    t = MagicMock()
    t.entry_time = _ts(10, 0)
    t.exit_time = _ts(10, 30)
    t.direction = direction
    t.entry_price = 5000.0
    t.exit_price = 5010.0
    t.net_pnl = net_pnl
    t.exit_reason = "tp"
    t.strategy_type = "ema50_breakout"
    t.position_size = 1.0
    return t


def _risk_event(event_type="blocked", reason="max_trades_per_day"):
    return RiskEvent(
        timestamp=_ts(),
        event_type=event_type,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# DashboardState — PnL updates
# ---------------------------------------------------------------------------

class TestDashboardStatePnL:
    def test_initial_snapshot(self):
        state = DashboardState()
        snap = state.snapshot()
        assert snap["pnl"]["equity"] == 0.0
        assert snap["pnl"]["drawdown"] == 0.0
        assert snap["recent_trades"] == []
        assert snap["alerts"] == []
        assert snap["open_positions"] == []

    def test_pnl_update_reflected(self):
        state = DashboardState()
        state.on_pnl(_pnl_update(equity=10_500.0, unrealized=200.0, realized=300.0, dd=-50.0))
        snap = state.snapshot()
        assert snap["pnl"]["equity"] == 10_500.0
        assert snap["pnl"]["unrealized_pnl"] == 200.0
        assert snap["pnl"]["realized_pnl"] == 300.0
        assert snap["pnl"]["drawdown"] == -50.0

    def test_equity_curve_grows(self):
        state = DashboardState()
        state.on_pnl(_pnl_update(equity=10_000.0))
        state.on_pnl(_pnl_update(equity=10_100.0))
        state.on_pnl(_pnl_update(equity=10_050.0))
        curve = state.equity_curve()
        assert len(curve) == 3
        assert curve[0]["equity"] == 10_000.0
        assert curve[2]["equity"] == 10_050.0

    def test_equity_curve_last_n(self):
        state = DashboardState()
        for i in range(10):
            state.on_pnl(_pnl_update(equity=10_000.0 + i))
        curve = state.equity_curve(last_n=3)
        assert len(curve) == 3
        assert curve[0]["equity"] == 10_007.0

    def test_peak_equity_tracked(self):
        state = DashboardState()
        state.on_pnl(_pnl_update(equity=10_500.0))
        state.on_pnl(_pnl_update(equity=10_200.0))
        assert state.snapshot()["pnl"]["peak_equity"] == 10_500.0


# ---------------------------------------------------------------------------
# DashboardState — Drawdown alerts
# ---------------------------------------------------------------------------

class TestDrawdownAlerts:
    def test_warning_alert_on_drawdown(self):
        state = DashboardState(drawdown_warn=-300.0, drawdown_critical=-450.0)
        state.on_pnl(_pnl_update(dd=-310.0))
        snap = state.snapshot()
        assert len(snap["alerts"]) == 1
        assert snap["alerts"][0]["level"] == "warning"
        assert snap["alerts"][0]["category"] == "drawdown"

    def test_critical_alert_on_deep_drawdown(self):
        state = DashboardState(drawdown_warn=-300.0, drawdown_critical=-450.0)
        state.on_pnl(_pnl_update(dd=-500.0))
        snap = state.snapshot()
        assert len(snap["alerts"]) == 1
        assert snap["alerts"][0]["level"] == "critical"

    def test_no_alert_within_tolerance(self):
        state = DashboardState(drawdown_warn=-300.0, drawdown_critical=-450.0)
        state.on_pnl(_pnl_update(dd=-100.0))
        assert state.alert_count() == 0


# ---------------------------------------------------------------------------
# DashboardState — Trade tracking
# ---------------------------------------------------------------------------

class TestTradeTracking:
    def test_trade_recorded(self):
        state = DashboardState()
        state.on_trade(_mock_trade(net_pnl=75.0))
        snap = state.snapshot()
        assert len(snap["recent_trades"]) == 1
        assert snap["recent_trades"][0]["net_pnl"] == 75.0

    def test_trade_count(self):
        state = DashboardState()
        for _ in range(5):
            state.on_trade(_mock_trade())
        assert state.trade_count() == 5

    def test_trade_fields(self):
        state = DashboardState()
        state.on_trade(_mock_trade(net_pnl=-30.0, direction="short"))
        t = state.snapshot()["recent_trades"][0]
        assert t["direction"] == "short"
        assert t["net_pnl"] == -30.0
        assert t["exit_reason"] == "tp"
        assert t["strategy_type"] == "ema50_breakout"


# ---------------------------------------------------------------------------
# DashboardState — Signal tracking
# ---------------------------------------------------------------------------

class TestSignalTracking:
    def test_signal_recorded(self):
        state = DashboardState()
        state.on_signal(_entry_signal())
        snap = state.snapshot()
        # Signals are not in the standard snapshot — they're buffered internally
        # Check via equity/trade state that no crash
        assert snap["pnl"]["equity"] == 0.0


# ---------------------------------------------------------------------------
# DashboardState — Risk events
# ---------------------------------------------------------------------------

class TestRiskEvents:
    def test_blocked_event_creates_warning(self):
        state = DashboardState()
        state.on_risk_event(_risk_event("blocked", "max_trades_per_day"))
        snap = state.snapshot()
        assert len(snap["alerts"]) == 1
        assert snap["alerts"][0]["level"] == "warning"
        assert snap["alerts"][0]["category"] == "risk"

    def test_kill_switch_creates_critical(self):
        state = DashboardState()
        state.on_risk_event(_risk_event("kill_switch", "daily loss breached"))
        snap = state.snapshot()
        assert snap["risk"]["killed"] is True
        assert len(snap["alerts"]) == 1
        assert snap["alerts"][0]["level"] == "critical"
        assert snap["alerts"][0]["category"] == "kill_switch"

    def test_risk_state_update(self):
        state = DashboardState()
        risk_mock = MagicMock()
        risk_mock.killed = False
        risk_mock.daily_entries = 3
        state.update_risk_state(risk_mock)
        snap = state.snapshot()
        assert snap["risk"]["daily_entries"] == 3
        assert snap["risk"]["killed"] is False

    def test_risk_event_count(self):
        state = DashboardState()
        state.on_risk_event(_risk_event("blocked", "a"))
        state.on_risk_event(_risk_event("blocked", "b"))
        snap = state.snapshot()
        assert snap["risk"]["total_events"] == 2


# ---------------------------------------------------------------------------
# DashboardState — Open positions
# ---------------------------------------------------------------------------

class TestOpenPositions:
    def test_positions_from_paper_engine(self):
        state = DashboardState()
        paper_mock = MagicMock()
        paper_mock.open_positions = {
            "pos_1": {
                "direction": "long",
                "entry_price": 5000.0,
                "entry_time": _ts(10, 0),
                "position_size": 1.0,
                "strategy_type": "ema50_breakout",
            },
        }
        state.update_open_positions(paper_mock)
        snap = state.snapshot()
        assert len(snap["open_positions"]) == 1
        assert snap["open_positions"][0]["position_id"] == "pos_1"
        assert snap["open_positions"][0]["direction"] == "long"

    def test_empty_positions(self):
        state = DashboardState()
        paper_mock = MagicMock()
        paper_mock.open_positions = {}
        state.update_open_positions(paper_mock)
        assert state.snapshot()["open_positions"] == []


# ---------------------------------------------------------------------------
# DashboardState — Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_writes(self):
        """Multiple threads writing simultaneously should not corrupt state."""
        state = DashboardState()
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    state.on_pnl(_pnl_update(equity=10_000 + n * 100 + i))
                    state.on_trade(_mock_trade())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert state.trade_count() == 400
        assert len(state.equity_curve()) == 400


# ---------------------------------------------------------------------------
# FastAPI app — endpoint tests
# ---------------------------------------------------------------------------

class TestFastAPIApp:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from dashboard.app import create_app

        state = DashboardState()
        state.on_pnl(_pnl_update(equity=10_200.0, dd=-50.0))
        state.on_trade(_mock_trade(net_pnl=100.0))
        state.on_risk_event(_risk_event("blocked", "test"))

        app = create_app(state)
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["trades"] == 1
        assert data["alerts"] == 1

    def test_snapshot(self, client):
        r = client.get("/api/snapshot")
        assert r.status_code == 200
        data = r.json()
        assert data["pnl"]["equity"] == 10_200.0
        assert len(data["recent_trades"]) == 1
        assert len(data["alerts"]) == 1

    def test_equity_curve(self, client):
        r = client.get("/api/equity")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["equity_curve"][0]["equity"] == 10_200.0

    def test_equity_curve_last_n(self, client):
        r = client.get("/api/equity?last=1")
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_index_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Apex" in r.text

    def test_snapshot_empty_state(self):
        from fastapi.testclient import TestClient
        from dashboard.app import create_app

        app = create_app(DashboardState())
        c = TestClient(app)
        r = c.get("/api/snapshot")
        assert r.status_code == 200
        data = r.json()
        assert data["recent_trades"] == []
        assert data["alerts"] == []


# ---------------------------------------------------------------------------
# main.py integration — dashboard wiring
# ---------------------------------------------------------------------------

class TestMainDashboardWiring:
    def test_build_pipeline_with_dashboard(self):
        from scripts.run_live import build_pipeline
        from config.settings import InstrumentConfig
        from strategy.hybrid_ema_ml import HybridEMAMLConfig
        from risk.risk_manager import RiskConfig
        from strategy.paper_engine import PaperConfig

        state = DashboardState()
        pipeline = build_pipeline(
            mode="paper",
            instrument=InstrumentConfig(),
            strategy_cfg=HybridEMAMLConfig(),
            risk_cfg=RiskConfig(),
            paper_cfg=PaperConfig(),
            dashboard_state=state,
        )
        assert pipeline["dashboard_state"] is state

    def test_build_pipeline_without_dashboard(self):
        from scripts.run_live import build_pipeline
        from config.settings import InstrumentConfig
        from strategy.hybrid_ema_ml import HybridEMAMLConfig
        from risk.risk_manager import RiskConfig

        pipeline = build_pipeline(
            mode="paper",
            instrument=InstrumentConfig(),
            strategy_cfg=HybridEMAMLConfig(),
            risk_cfg=RiskConfig(),
        )
        assert pipeline["dashboard_state"] is None

    def test_dashboard_cli_flag(self):
        from scripts.run_live import parse_args
        args = parse_args(["--mode", "paper", "--dashboard", "--dashboard-port", "9999"])
        assert args.dashboard is True
        assert args.dashboard_port == 9999

    def test_dashboard_disabled_by_default(self):
        from scripts.run_live import parse_args
        args = parse_args(["--mode", "paper"])
        assert args.dashboard is False
        assert args.dashboard_port == 8501
