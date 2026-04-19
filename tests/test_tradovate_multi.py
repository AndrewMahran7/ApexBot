"""Tests for the multi-symbol Tradovate SIM adapter."""

import datetime
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from config.settings import InstrumentConfig, INSTRUMENT_REGISTRY
from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal
from execution.tradovate_client import TradovateConfig, TradovateClient
from execution.tradovate_multi import (
    MultiSymbolTradovateAdapter,
    load_tradovate_config,
    SYMBOL_TO_TRADOVATE,
    _setup_sim_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))


def _ts(hour=10, minute=0, day=2):
    return datetime.datetime(2024, 1, day, hour, minute, tzinfo=_ET)


def _entry_signal(direction="long", entry=5000.0, stop=4990.0, tp=5015.0):
    sig_type = SignalType.LONG_ENTRY if direction == "long" else SignalType.SHORT_ENTRY
    return LiveSignal(
        signal_type=sig_type,
        timestamp=_ts(),
        entry=entry,
        stop=stop,
        take_profit=tp,
        direction=direction,
        position_size=1.0,
        strategy_type="ema50_breakout",
        position_id="pos_1",
    )


def _make_config():
    return TradovateConfig(
        username="testuser",
        password="testpass",
        cid="testcid",
        secret="testsecret",
        environment="demo",
        sim_only=True,
    )


# ---------------------------------------------------------------------------
# TradovateConfig safety
# ---------------------------------------------------------------------------


class TestConfigSafety:
    def test_config_defaults_to_demo(self):
        cfg = TradovateConfig()
        assert cfg.environment == "demo"
        assert cfg.sim_only is True

    def test_base_url_demo(self):
        cfg = TradovateConfig(environment="demo")
        assert "demo" in cfg.base_url

    def test_base_url_live_blocked_by_sim_only(self):
        cfg = TradovateConfig(environment="live", sim_only=True)
        # The client should refuse to connect, but config itself is just data
        assert cfg.base_url  # returns live URL, but connect() will block it


# ---------------------------------------------------------------------------
# MultiSymbolTradovateAdapter
# ---------------------------------------------------------------------------


class TestAdapterConstruction:
    def test_forces_demo_sim_only(self):
        """Adapter must force demo + sim_only regardless of input."""
        cfg = _make_config()
        cfg.environment = "live"
        cfg.sim_only = False

        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES", "MNQ"],
            config=cfg,
        )
        # After init, config should be forced to demo + sim_only
        assert adapter._config.environment == "demo"
        assert adapter._config.sim_only is True

    def test_not_connected_initially(self):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES"], config=_make_config(),
        )
        assert adapter.is_connected is False

    def test_contract_overrides(self):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES"],
            config=_make_config(),
            contract_overrides={"MES": "MESZ4"},
        )
        assert adapter._contract_overrides["MES"] == "MESZ4"


class TestAdapterRouteSignal:
    def test_route_signal_not_connected_logs_error(self, caplog):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES"], config=_make_config(),
        )
        sig = _entry_signal()
        adapter.route_signal("MES", sig)
        # Should not crash, just log error
        assert not adapter.is_connected

    @patch.object(TradovateClient, "connect")
    @patch.object(TradovateClient, "start_sync")
    @patch.object(TradovateClient, "on_signal")
    def test_route_signal_calls_client(self, mock_on_signal, mock_sync, mock_connect):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES"], config=_make_config(),
        )
        # Simulate the connect setting internal state
        mock_connect.return_value = None
        mock_sync.return_value = None

        adapter.connect()
        sig = _entry_signal()
        adapter.route_signal("MES", sig)

        mock_on_signal.assert_called_once_with(sig)

    @patch.object(TradovateClient, "connect")
    @patch.object(TradovateClient, "start_sync")
    def test_route_signal_records_trade_log(self, mock_sync, mock_connect):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES"], config=_make_config(),
        )
        # Patch on_signal to avoid real API calls
        with patch.object(TradovateClient, "on_signal"):
            adapter.connect()
            sig = _entry_signal()
            adapter.route_signal("MES", sig)

            assert len(adapter._trade_log) == 1
            entry = adapter._trade_log[0]
            assert entry["symbol"] == "MES"
            assert entry["action"] == "entry"
            assert entry["direction"] == "long"

    @patch.object(TradovateClient, "connect")
    @patch.object(TradovateClient, "start_sync")
    def test_route_unknown_symbol_logs_error(self, mock_sync, mock_connect):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES"], config=_make_config(),
        )
        adapter.connect()
        sig = _entry_signal()
        # Should not crash
        adapter.route_signal("UNKNOWN", sig)
        assert len(adapter._trade_log) == 0


class TestAdapterDisconnect:
    @patch.object(TradovateClient, "connect")
    @patch.object(TradovateClient, "start_sync")
    @patch.object(TradovateClient, "disconnect")
    def test_disconnect_all(self, mock_disconnect, mock_sync, mock_connect):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES", "MNQ"], config=_make_config(),
        )
        adapter.connect()
        adapter.disconnect()

        assert mock_disconnect.call_count == 2
        assert adapter.is_connected is False


class TestAdapterLiquidateAll:
    @patch.object(TradovateClient, "connect")
    @patch.object(TradovateClient, "start_sync")
    @patch.object(TradovateClient, "liquidate_all")
    def test_liquidate_all_symbols(self, mock_liquidate, mock_sync, mock_connect):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES", "MNQ"], config=_make_config(),
        )
        adapter.connect()
        adapter.liquidate_all()

        assert mock_liquidate.call_count == 2


class TestAdapterDryRun:
    @patch.object(TradovateClient, "connect")
    @patch.object(TradovateClient, "disconnect")
    def test_dry_run_all_ok(self, mock_disconnect, mock_connect):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES", "MNQ"], config=_make_config(),
        )
        results = adapter.dry_run()

        assert results["_all_ok"] is True
        assert results["MES"]["status"] == "OK"
        assert results["MNQ"]["status"] == "OK"

    @patch.object(TradovateClient, "connect", side_effect=RuntimeError("auth failed"))
    def test_dry_run_auth_failure(self, mock_connect):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES"], config=_make_config(),
        )
        results = adapter.dry_run()

        assert results["_all_ok"] is False
        assert results["MES"]["status"] == "FAIL"
        assert "auth failed" in results["MES"]["error"]


class TestAdapterTradeExport:
    @patch.object(TradovateClient, "connect")
    @patch.object(TradovateClient, "start_sync")
    def test_export_empty(self, mock_sync, mock_connect, tmp_path):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES"], config=_make_config(),
        )
        path = str(tmp_path / "trades.csv")
        adapter.export_trade_log(path)
        # No file created when no trades
        assert not (tmp_path / "trades.csv").exists()

    @patch.object(TradovateClient, "connect")
    @patch.object(TradovateClient, "start_sync")
    def test_export_with_trades(self, mock_sync, mock_connect, tmp_path):
        adapter = MultiSymbolTradovateAdapter(
            symbols=["MES"], config=_make_config(),
        )
        with patch.object(TradovateClient, "on_signal"):
            adapter.connect()
            adapter.route_signal("MES", _entry_signal())

        path = str(tmp_path / "trades.csv")
        adapter.export_trade_log(path)

        assert (tmp_path / "trades.csv").exists()
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["symbol"] == "MES"


# ---------------------------------------------------------------------------
# load_tradovate_config from environment
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_missing_credentials_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="Missing Tradovate credentials"):
                load_tradovate_config()

    def test_valid_env_loads(self):
        env = {
            "TRADOVATE_USERNAME": "user",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "cid",
            "TRADOVATE_SECRET": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            # Patch load_dotenv to avoid file IO
            with patch("execution.tradovate_multi.load_dotenv"):
                cfg = load_tradovate_config()
                assert cfg.username == "user"
                assert cfg.environment == "demo"
                assert cfg.sim_only is True

    def test_custom_app_id(self):
        env = {
            "TRADOVATE_USERNAME": "user",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "cid",
            "TRADOVATE_SECRET": "secret",
            "TRADOVATE_APP_ID": "MyApp",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("execution.tradovate_multi.load_dotenv"):
                cfg = load_tradovate_config()
                assert cfg.app_id == "MyApp"


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------


class TestSymbolMapping:
    def test_default_mapping_exists(self):
        assert "MES" in SYMBOL_TO_TRADOVATE
        assert "MNQ" in SYMBOL_TO_TRADOVATE

    def test_mes_and_mnq_in_instrument_registry(self):
        assert "MES" in INSTRUMENT_REGISTRY
        assert "MNQ" in INSTRUMENT_REGISTRY


# ---------------------------------------------------------------------------
# CLI flag integration
# ---------------------------------------------------------------------------


class TestCLIFlags:
    def test_tradovate_sim_flag(self):
        from scripts.run_paper import parse_args
        args = parse_args(["--symbols", "MES", "MNQ", "--tradovate-sim"])
        assert args.tradovate_sim is True
        assert args.tradovate_dry_run is False

    def test_tradovate_dry_run_flag(self):
        from scripts.run_paper import parse_args
        args = parse_args(["--symbols", "MES", "MNQ", "--tradovate-dry-run"])
        assert args.tradovate_dry_run is True

    def test_tradovate_contract_overrides(self):
        from scripts.run_paper import parse_args
        args = parse_args([
            "--symbols", "MES", "MNQ",
            "--tradovate-contract-MES", "MESZ4",
            "--tradovate-contract-MNQ", "MNQZ4",
        ])
        assert args.tradovate_contract_MES == "MESZ4"
        assert args.tradovate_contract_MNQ == "MNQZ4"

    def test_default_no_tradovate(self):
        from scripts.run_paper import parse_args
        args = parse_args(["--symbols", "MES", "MNQ"])
        assert args.tradovate_sim is False
        assert args.tradovate_dry_run is False
