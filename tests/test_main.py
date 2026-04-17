"""Tests for main.py — system runner pipeline wiring and CLI."""

import datetime
import logging
from unittest.mock import patch, MagicMock

import pytest

from config.settings import InstrumentConfig
from strategy.hybrid_ema_ml import HybridEMAMLConfig
from strategy.paper_engine import PaperConfig
from strategy.risk_manager import RiskConfig
from strategy.strategy_engine import LiveSignal
from strategy.tradovate_client import TradovateConfig
from strategy.orb import SignalType
from main import (
    build_pipeline,
    parse_args,
    setup_logging,
    run_bar_loop,
    print_summary,
    _log_signal,
    _log_trade,
    _log_pnl,
    _NameFilter,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))


def _ts(hour: int = 10, minute: int = 0) -> datetime.datetime:
    return datetime.datetime(2024, 1, 2, hour, minute, tzinfo=_ET)


def _inst() -> InstrumentConfig:
    return InstrumentConfig(
        symbol="MES", tick_size=0.25, point_value=5.0, contract_size=1,
    )


def _strategy_cfg() -> HybridEMAMLConfig:
    return HybridEMAMLConfig(
        multi_candidate=False,
        ema_periods=(50,),
        entry_types=("breakout",),
        model_path="models/ema_model.pkl",
    )


def _risk_cfg() -> RiskConfig:
    return RiskConfig(
        max_daily_loss=500.0,
        max_trades_per_day=6,
        max_concurrent_positions=3,
    )


def _paper_cfg() -> PaperConfig:
    return PaperConfig(initial_capital=10_000.0)


def _entry_signal(ts=None) -> LiveSignal:
    return LiveSignal(
        timestamp=ts or _ts(),
        direction="long",
        signal_type=SignalType.LONG_ENTRY,
        entry=5000.0,
        stop=4990.0,
        take_profit=5015.0,
        position_size=1.0,
        strategy_type="ema50_breakout",
        position_id="pos_1",
    )


def _exit_signal(ts=None) -> LiveSignal:
    return LiveSignal(
        timestamp=ts or _ts(10, 30),
        direction="",
        signal_type=SignalType.EXIT_TP,
        entry=5015.0,
        stop=0.0,
        take_profit=0.0,
        position_size=0.0,
        strategy_type="ema50_breakout",
        position_id="pos_1",
    )


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_mode_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_replay_mode(self):
        args = parse_args(["--mode", "replay", "--data", "data/test.csv"])
        assert args.mode == "replay"
        assert args.data == "data/test.csv"

    def test_paper_mode(self):
        args = parse_args(["--mode", "paper"])
        assert args.mode == "paper"

    def test_live_mode(self):
        args = parse_args(["--mode", "live"])
        assert args.mode == "live"

    def test_defaults(self):
        args = parse_args(["--mode", "replay"])
        assert args.data == "data/mes_5m.csv"
        assert args.ml_threshold == 0.6
        assert args.max_daily_loss == 500.0
        assert args.max_trades_per_day == 6
        assert args.initial_capital == 10_000.0
        assert args.log_level == "INFO"

    def test_custom_risk_params(self):
        args = parse_args([
            "--mode", "paper",
            "--max-daily-loss", "300",
            "--max-trades-per-day", "4",
            "--max-concurrent", "2",
        ])
        assert args.max_daily_loss == 300.0
        assert args.max_trades_per_day == 4
        assert args.max_concurrent == 2

    def test_strategy_params(self):
        args = parse_args([
            "--mode", "replay",
            "--ema-periods", "20", "50", "100",
            "--entry-types", "breakout", "pullback",
            "--selection-strategy", "priority",
        ])
        assert args.ema_periods == [20, 50, 100]
        assert args.entry_types == ["breakout", "pullback"]
        assert args.selection_strategy == "priority"

    def test_date_filters(self):
        args = parse_args([
            "--mode", "replay",
            "--start", "2024-01-01",
            "--end", "2024-06-30",
        ])
        assert args.start == "2024-01-01"
        assert args.end == "2024-06-30"

    def test_invalid_mode_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--mode", "invalid"])

    def test_log_level(self):
        args = parse_args(["--mode", "replay", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_replay_mode_wires_paper(self):
        pipeline = build_pipeline(
            mode="replay",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
            paper_cfg=_paper_cfg(),
        )
        assert pipeline["engine"] is not None
        assert pipeline["risk"] is not None
        assert pipeline["paper"] is not None
        assert pipeline["tradovate"] is None

    def test_paper_mode_wires_paper(self):
        pipeline = build_pipeline(
            mode="paper",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
            paper_cfg=_paper_cfg(),
        )
        assert pipeline["paper"] is not None
        assert pipeline["tradovate"] is None

    def test_live_mode_wires_tradovate(self):
        pipeline = build_pipeline(
            mode="live",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
            tradovate_cfg=TradovateConfig(),
        )
        assert pipeline["tradovate"] is not None
        assert pipeline["paper"] is None

    def test_live_without_config_raises(self):
        with pytest.raises(ValueError, match="Tradovate configuration"):
            build_pipeline(
                mode="live",
                instrument=_inst(),
                strategy_cfg=_strategy_cfg(),
                risk_cfg=_risk_cfg(),
                tradovate_cfg=None,
            )

    def test_risk_on_approved_wired(self):
        pipeline = build_pipeline(
            mode="paper",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
        )
        # on_approved is set (the wrapped callback)
        assert pipeline["risk"].on_approved is not None

    def test_engine_on_signal_wired(self):
        pipeline = build_pipeline(
            mode="paper",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
        )
        # engine has on_signal callback set
        assert pipeline["engine"]._on_signal is not None

    def test_default_paper_config(self):
        """Paper config defaults if None passed."""
        pipeline = build_pipeline(
            mode="paper",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
            paper_cfg=None,
        )
        assert pipeline["paper"] is not None


# ---------------------------------------------------------------------------
# Logging callbacks
# ---------------------------------------------------------------------------

class TestLoggingCallbacks:
    def test_log_signal_no_error(self):
        sig = _entry_signal()
        _log_signal(sig)  # should not raise

    def test_log_trade_entry(self):
        sig = _entry_signal()
        _log_trade(sig)  # should not raise

    def test_log_trade_exit(self):
        sig = _exit_signal()
        _log_trade(sig)  # should not raise

    def test_log_pnl_no_error(self):
        from strategy.paper_engine import PnLUpdate
        update = PnLUpdate(
            timestamp=_ts(),
            equity=10_000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            open_position_count=0,
            drawdown=0.0,
        )
        _log_pnl(update)  # should not raise


class TestNameFilter:
    def test_matching_prefix_passes(self):
        f = _NameFilter("apex.trades")
        record = logging.LogRecord(
            name="apex.trades", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        assert f.filter(record) is True

    def test_non_matching_prefix_blocked(self):
        f = _NameFilter("apex.trades")
        record = logging.LogRecord(
            name="apex.signals", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        assert f.filter(record) is False

    def test_child_logger_passes(self):
        f = _NameFilter("apex.trades")
        record = logging.LogRecord(
            name="apex.trades.detail", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        assert f.filter(record) is True


# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "test_logs"
        setup_logging(log_dir=str(log_dir), level="DEBUG")
        assert log_dir.exists()

        # Cleanup handlers to avoid interference
        root = logging.getLogger()
        for h in root.handlers[:]:
            if hasattr(h, 'baseFilename') and str(tmp_path) in getattr(h, 'baseFilename', ''):
                root.removeHandler(h)
                h.close()

    def test_log_files_created(self, tmp_path):
        log_dir = tmp_path / "test_logs2"
        setup_logging(log_dir=str(log_dir), level="INFO")

        expected = {"main.log", "trades.log", "signals.log", "errors.log"}
        created = {f.name for f in log_dir.iterdir()}
        assert expected.issubset(created)

        root = logging.getLogger()
        for h in root.handlers[:]:
            if hasattr(h, 'baseFilename') and str(tmp_path) in getattr(h, 'baseFilename', ''):
                root.removeHandler(h)
                h.close()


# ---------------------------------------------------------------------------
# Bar loop
# ---------------------------------------------------------------------------

class TestRunBarLoop:
    def test_bar_loop_feeds_components(self):
        """Verify on_bar is called on risk, paper, and engine."""
        import pandas as pd
        import pytz

        et = pytz.timezone("America/New_York")
        index = pd.DatetimeIndex([
            pd.Timestamp("2024-01-02 09:30", tz=et),
            pd.Timestamp("2024-01-02 09:35", tz=et),
            pd.Timestamp("2024-01-02 09:40", tz=et),
        ])
        bars = pd.DataFrame({
            "open": [5000.0, 5001.0, 5002.0],
            "high": [5002.0, 5003.0, 5004.0],
            "low": [4999.0, 5000.0, 5001.0],
            "close": [5001.0, 5002.0, 5003.0],
            "volume": [100, 110, 120],
        }, index=index)
        bars.index.name = "timestamp"

        # Build pipeline and mock on_bar methods
        pipeline = build_pipeline(
            mode="paper",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
            paper_cfg=_paper_cfg(),
        )

        risk_calls = []
        paper_calls = []
        engine_calls = []

        orig_risk_on_bar = pipeline["risk"].on_bar
        orig_paper_on_bar = pipeline["paper"].on_bar
        orig_engine_on_bar = pipeline["engine"].on_bar

        def track_risk(bar):
            risk_calls.append(bar["timestamp"])
            orig_risk_on_bar(bar)

        def track_paper(bar):
            paper_calls.append(bar["timestamp"])
            orig_paper_on_bar(bar)

        def track_engine(bar):
            engine_calls.append(bar["timestamp"])
            orig_engine_on_bar(bar)

        pipeline["risk"].on_bar = track_risk
        pipeline["paper"].on_bar = track_paper
        pipeline["engine"].on_bar = track_engine

        run_bar_loop(bars, pipeline)

        assert len(risk_calls) == 3
        assert len(paper_calls) == 3
        assert len(engine_calls) == 3

    def test_bar_loop_order_risk_before_engine(self):
        """Risk.on_bar must be called before engine.on_bar each bar."""
        import pandas as pd
        import pytz

        et = pytz.timezone("America/New_York")
        index = pd.DatetimeIndex([
            pd.Timestamp("2024-01-02 09:30", tz=et),
        ])
        bars = pd.DataFrame({
            "open": [5000.0],
            "high": [5002.0],
            "low": [4999.0],
            "close": [5001.0],
            "volume": [100],
        }, index=index)
        bars.index.name = "timestamp"

        pipeline = build_pipeline(
            mode="paper",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
            paper_cfg=_paper_cfg(),
        )

        call_order = []

        orig_risk = pipeline["risk"].on_bar
        orig_paper = pipeline["paper"].on_bar
        orig_engine = pipeline["engine"].on_bar

        def risk_bar(bar):
            call_order.append("risk")
            orig_risk(bar)

        def paper_bar(bar):
            call_order.append("paper")
            orig_paper(bar)

        def engine_bar(bar):
            call_order.append("engine")
            orig_engine(bar)

        pipeline["risk"].on_bar = risk_bar
        pipeline["paper"].on_bar = paper_bar
        pipeline["engine"].on_bar = engine_bar

        run_bar_loop(bars, pipeline)

        assert call_order == ["risk", "paper", "engine"]


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_paper_summary(self, capsys):
        pipeline = build_pipeline(
            mode="paper",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
            paper_cfg=_paper_cfg(),
        )
        print_summary(pipeline, "paper")
        output = capsys.readouterr().out
        assert "Apex Run Summary" in output
        assert "paper" in output
        assert "Risk events" in output
        assert "Trades" in output

    def test_replay_summary_without_paper(self, capsys):
        """Replay still prints risk events even without paper trades."""
        pipeline = build_pipeline(
            mode="replay",
            instrument=_inst(),
            strategy_cfg=_strategy_cfg(),
            risk_cfg=_risk_cfg(),
        )
        print_summary(pipeline, "replay")
        output = capsys.readouterr().out
        assert "Risk events" in output


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

class TestMainIntegration:
    def test_replay_with_sample_data(self, tmp_path):
        """Full replay run with a tiny CSV — no errors."""
        csv_path = tmp_path / "test_bars.csv"
        csv_path.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2024-01-02 09:30:00,5000,5002,4999,5001,100\n"
            "2024-01-02 09:35:00,5001,5003,5000,5002,110\n"
            "2024-01-02 09:40:00,5002,5004,5001,5003,120\n"
            "2024-01-02 09:45:00,5003,5005,5002,5004,130\n"
            "2024-01-02 09:50:00,5004,5006,5003,5005,140\n"
        )
        log_dir = tmp_path / "logs"

        result = main([
            "--mode", "paper",
            "--data", str(csv_path),
            "--log-dir", str(log_dir),
            "--log-level", "DEBUG",
        ])

        assert result == 0
        assert log_dir.exists()
        assert (log_dir / "main.log").exists()
        assert (log_dir / "trades.log").exists()
        assert (log_dir / "signals.log").exists()
        assert (log_dir / "errors.log").exists()

    def test_missing_data_returns_error(self, tmp_path):
        result = main([
            "--mode", "replay",
            "--data", str(tmp_path / "nonexistent.csv"),
            "--log-dir", str(tmp_path / "logs"),
        ])
        assert result == 1
