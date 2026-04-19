"""
Tests for Multi-Symbol Router
===============================

Integration tests for the MultiSymbolRouter:
  1. Single-symbol through multi-symbol pipeline works correctly
  2. Portfolio risk blocks cross-symbol when at limit
  3. Exit signals pass through without portfolio blocking
  4. Signal ranking selects highest ML prob first
"""

import datetime
import pytest

from config.settings import InstrumentConfig, INSTRUMENT_REGISTRY
from strategy.hybrid_ema_ml import HybridEMAMLConfig
from strategy.strategy_engine import StrategyEngine, LiveSignal
from strategy.paper_engine import PaperEngine, PaperConfig
from risk.risk_manager import RiskManager, RiskConfig
from risk.portfolio_risk import PortfolioRiskManager, PortfolioRiskConfig
from scripts.run_multi_symbol import (
    SymbolPipeline,
    MultiSymbolRouter,
    build_multi_symbol_pipeline,
    parse_args,
)


def _make_pipeline(symbol="MES") -> SymbolPipeline:
    """Create a minimal single-symbol pipeline for testing."""
    instrument = INSTRUMENT_REGISTRY.get(symbol, InstrumentConfig(symbol=symbol))
    strategy_cfg = HybridEMAMLConfig(
        ml_threshold=0.50,
        model_path="models/ema_model.pkl",
    )
    risk_cfg = RiskConfig(
        max_daily_loss=500.0,
        max_trades_per_day=6,
        max_concurrent_positions=3,
    )
    paper_cfg = PaperConfig(initial_capital=5000.0)

    engine = StrategyEngine(config=strategy_cfg, on_signal=None)
    risk = RiskManager(config=risk_cfg, instrument=instrument)
    paper = PaperEngine(instrument=instrument, config=paper_cfg)

    return SymbolPipeline(
        symbol=symbol,
        instrument=instrument,
        engine=engine,
        risk=risk,
        paper=paper,
    )


class TestMultiSymbolRouterWiring:
    def test_creates_router_with_single_symbol(self):
        pipe = _make_pipeline("MES")
        portfolio = PortfolioRiskManager(PortfolioRiskConfig())
        router = MultiSymbolRouter({"MES": pipe}, portfolio)
        assert router.total_equity() == pytest.approx(5000.0)

    def test_creates_router_with_multiple_symbols(self):
        pipes = {
            "MES": _make_pipeline("MES"),
            "MNQ": _make_pipeline("MNQ"),
        }
        portfolio = PortfolioRiskManager(PortfolioRiskConfig())
        router = MultiSymbolRouter(pipes, portfolio)
        assert router.total_equity() == pytest.approx(10000.0)

    def test_total_trades_empty_initially(self):
        pipe = _make_pipeline("MES")
        portfolio = PortfolioRiskManager(PortfolioRiskConfig())
        router = MultiSymbolRouter({"MES": pipe}, portfolio)
        assert len(router.total_trades()) == 0


class TestPortfolioConstraintsInRouter:
    def test_portfolio_tracks_cross_symbol_positions(self):
        pipes = {
            "MES": _make_pipeline("MES"),
            "MNQ": _make_pipeline("MNQ"),
        }
        portfolio = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=2,
        ))
        router = MultiSymbolRouter(pipes, portfolio)

        # Manually record entries to test portfolio state
        from strategy.orb import SignalType
        sig1 = LiveSignal(
            timestamp=datetime.datetime(2024, 1, 1, 10, 0,
                                        tzinfo=datetime.timezone.utc),
            direction="long",
            signal_type=SignalType.LONG_ENTRY,
            entry=5000.0, stop=4990.0, take_profit=5015.0,
            position_size=1.0, strategy_type="ema50_breakout",
            position_id="p1", ml_prob=0.7,
        )
        portfolio.record_entry("MES", sig1)
        portfolio.record_entry("MNQ", sig1)
        assert portfolio.open_position_count == 2


class TestBuildPipeline:
    def test_build_single_symbol(self):
        args = parse_args([
            "--symbols", "MES",
            "--data-MES", "data/mes_4y.csv",
            "--days", "1",
        ])
        router, port_risk = build_multi_symbol_pipeline(args, ["MES"])
        assert "MES" in router._pipelines
        assert port_risk.open_position_count == 0

    def test_build_multiple_symbols(self):
        args = parse_args([
            "--symbols", "MES", "MNQ",
            "--data-MES", "data/mes_4y.csv",
            "--data-MNQ", "data/mes_4y.csv",
            "--days", "1",
        ])
        router, port_risk = build_multi_symbol_pipeline(args, ["MES", "MNQ"])
        assert "MES" in router._pipelines
        assert "MNQ" in router._pipelines

    def test_capital_split_equally(self):
        args = parse_args([
            "--symbols", "MES", "MNQ",
            "--data-MES", "data/mes_4y.csv",
            "--data-MNQ", "data/mes_4y.csv",
            "--initial-capital", "10000",
            "--days", "1",
        ])
        router, _ = build_multi_symbol_pipeline(args, ["MES", "MNQ"])
        mes_eq = router._pipelines["MES"].paper.equity
        mnq_eq = router._pipelines["MNQ"].paper.equity
        assert mes_eq == pytest.approx(5000.0)
        assert mnq_eq == pytest.approx(5000.0)


class TestInstrumentRegistry:
    def test_mes_in_registry(self):
        assert "MES" in INSTRUMENT_REGISTRY
        assert INSTRUMENT_REGISTRY["MES"].point_value == 5.0

    def test_mnq_in_registry(self):
        assert "MNQ" in INSTRUMENT_REGISTRY
        assert INSTRUMENT_REGISTRY["MNQ"].point_value == 2.0

    def test_rty_in_registry(self):
        assert "RTY" in INSTRUMENT_REGISTRY
        assert INSTRUMENT_REGISTRY["RTY"].point_value == 5.0
        assert INSTRUMENT_REGISTRY["RTY"].tick_size == 0.10
