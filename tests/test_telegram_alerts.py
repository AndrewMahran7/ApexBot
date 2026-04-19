"""Tests for the Telegram alert module."""

import datetime
import json
import os
from unittest.mock import MagicMock, patch, Mock

import pytest

from strategy.telegram_alerts import (
    TelegramAlerter,
    create_alerter,
    _format_ts,
    _one_line,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))


def _ts(hour=10, minute=35) -> datetime.datetime:
    return datetime.datetime(2026, 4, 17, hour, minute, tzinfo=_ET)


# ---------------------------------------------------------------------------
# TelegramAlerter â€” disabled mode
# ---------------------------------------------------------------------------


class TestDisabledAlerter:
    def test_disabled_does_not_send(self):
        alerter = TelegramAlerter(bot_token="", chat_id="", enabled=False)
        assert alerter.enabled is False
        result = alerter.send_test()
        assert result is False
        assert alerter.send_count == 0

    def test_disabled_entry_alert_noop(self):
        alerter = TelegramAlerter(bot_token="", chat_id="", enabled=False)
        result = alerter.send_entry_alert(
            symbol="MES", direction="long", entry=5000.0,
            stop=4990.0, target=5015.0, size=1.0,
            strategy_type="ema50_breakout",
        )
        assert result is False

    def test_disabled_exit_alert_noop(self):
        alerter = TelegramAlerter(bot_token="", chat_id="", enabled=False)
        result = alerter.send_exit_alert(
            symbol="MNQ", direction="long", exit_price=18278.0,
            exit_reason="TP", net_pnl=43.75,
        )
        assert result is False

    def test_disabled_startup_alert_noop(self):
        alerter = TelegramAlerter(bot_token="", chat_id="", enabled=False)
        result = alerter.send_startup_alert(symbols=["MES", "MNQ"])
        assert result is False

    def test_disabled_shutdown_alert_noop(self):
        alerter = TelegramAlerter(bot_token="", chat_id="", enabled=False)
        result = alerter.send_shutdown_alert()
        assert result is False


# ---------------------------------------------------------------------------
# TelegramAlerter â€” enabled mode (mocked HTTP)
# ---------------------------------------------------------------------------


class TestEnabledAlerter:
    def _make_alerter(self):
        return TelegramAlerter(bot_token="123:ABC", chat_id="999", enabled=True)

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_send_test_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        alerter = self._make_alerter()
        result = alerter.send_test()

        assert result is True
        assert alerter.send_count == 1
        mock_urlopen.assert_called_once()

        # Verify the request was made to the correct URL
        req = mock_urlopen.call_args[0][0]
        assert "123:ABC" in req.full_url
        assert "sendMessage" in req.full_url

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_send_test_http_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=401, msg="Unauthorized",
            hdrs=None, fp=MagicMock(read=lambda: b"bad token"),
        )

        alerter = self._make_alerter()
        result = alerter.send_test()

        assert result is False
        assert alerter.send_count == 0

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_send_entry_alert_format(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        alerter = self._make_alerter()
        result = alerter.send_entry_alert(
            symbol="MNQ",
            direction="long",
            entry=18234.25,
            stop=18212.50,
            target=18278.00,
            size=0.75,
            strategy_type="ema50_breakout",
            ml_prob=0.61,
            quality_score=0.58,
            timestamp=_ts(),
            rr_ratio=2.0,
            open_positions=1,
        )

        assert result is True
        # Verify message content
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        text = body["text"]
        assert "NEW TRADE" in text
        assert "MNQ" in text
        assert "LONG" in text
        assert "18234.25" in text
        assert "18212.50" in text
        assert "18278.00" in text
        assert "0.75" in text
        assert "ema50_breakout" in text
        assert "0.61" in text
        assert "0.58" in text
        assert "2.0" in text

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_send_exit_alert_format(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        alerter = self._make_alerter()
        result = alerter.send_exit_alert(
            symbol="MNQ",
            direction="long",
            exit_price=18278.00,
            exit_reason="TP",
            net_pnl=43.75,
            timestamp=_ts(11, 10),
        )

        assert result is True
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        text = body["text"]
        assert "TRADE CLOSED" in text
        assert "MNQ" in text
        assert "LONG" in text
        assert "18278.00" in text
        assert "TP" in text
        assert "+43.75" in text

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_send_exit_alert_negative_pnl(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        alerter = self._make_alerter()
        alerter.send_exit_alert(
            symbol="MES", direction="long", exit_price=4990.0,
            exit_reason="SL", net_pnl=-25.50,
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        text = body["text"]
        assert "-25.50" in text

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_send_startup_alert_format(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        alerter = self._make_alerter()
        result = alerter.send_startup_alert(
            symbols=["MES", "MNQ"], mode="paper", telegram_enabled=True,
        )

        assert result is True
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        text = body["text"]
        assert "BOT STARTED" in text
        assert "MES, MNQ" in text
        assert "paper" in text
        assert "enabled" in text

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_send_shutdown_alert_format(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        alerter = self._make_alerter()
        # Simulate some sends
        alerter._send_count = 5
        result = alerter.send_shutdown_alert()

        assert result is True
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        text = body["text"]
        assert "BOT STOPPED" in text
        assert "5" in text

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_send_count_increments(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        alerter = self._make_alerter()
        assert alerter.send_count == 0
        alerter.send_test()
        assert alerter.send_count == 1
        alerter.send_test()
        assert alerter.send_count == 2

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_connection_error_returns_false(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionError("no network")

        alerter = self._make_alerter()
        result = alerter.send_test()

        assert result is False
        assert alerter.send_count == 0

    @patch("strategy.telegram_alerts.urllib.request.urlopen")
    def test_size_reduced_flag(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_resp

        alerter = self._make_alerter()
        alerter.send_entry_alert(
            symbol="MES", direction="long", entry=5000.0,
            stop=4990.0, target=5015.0, size=0.5,
            strategy_type="ema50_breakout", size_reduced=True,
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        text = body["text"]
        assert "Size reduced" in text


# ---------------------------------------------------------------------------
# create_alerter factory
# ---------------------------------------------------------------------------


class TestCreateAlerter:
    def test_missing_credentials_returns_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("strategy.telegram_alerts.load_dotenv"):
                alerter = create_alerter(warn=False)
                assert alerter.enabled is False

    def test_partial_credentials_returns_disabled(self):
        env = {"TELEGRAM_BOT_TOKEN": "123:ABC"}
        with patch.dict(os.environ, env, clear=True):
            with patch("strategy.telegram_alerts.load_dotenv"):
                alerter = create_alerter(warn=False)
                assert alerter.enabled is False

    def test_valid_credentials_returns_enabled(self):
        env = {"TELEGRAM_BOT_TOKEN": "123:ABC", "TELEGRAM_CHAT_ID": "999"}
        with patch.dict(os.environ, env, clear=True):
            with patch("strategy.telegram_alerts.load_dotenv"):
                alerter = create_alerter(warn=False)
                assert alerter.enabled is True

    def test_missing_credentials_logs_warning(self, caplog):
        import logging
        with patch.dict(os.environ, {}, clear=True):
            with patch("strategy.telegram_alerts.load_dotenv"):
                with caplog.at_level(logging.WARNING):
                    create_alerter(warn=True)
                assert "DISABLED" in caplog.text or "missing" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_format_ts_with_datetime(self):
        ts = _ts(10, 35)
        result = _format_ts(ts)
        assert "2026-04-17" in result
        assert "10:35" in result
        assert "ET" in result

    def test_format_ts_none_uses_now(self):
        result = _format_ts(None)
        assert len(result) > 10

    def test_one_line_collapses(self):
        text = "line1\nline2\nline3"
        result = _one_line(text)
        assert "\n" not in result
        assert "line1" in result
        assert "line2" in result

    def test_one_line_truncates(self):
        text = "x" * 500
        result = _one_line(text)
        assert len(result) <= 200


# ---------------------------------------------------------------------------
# CLI flag integration
# ---------------------------------------------------------------------------


class TestCLIFlags:
    def test_test_telegram_flag(self):
        from scripts.run_paper import parse_args
        args = parse_args(["--test-telegram"])
        assert args.test_telegram is True

    def test_default_no_test_telegram(self):
        from scripts.run_paper import parse_args
        args = parse_args([])
        assert args.test_telegram is False


# ---------------------------------------------------------------------------
# Entry alert in execution tracker â€” integration
# ---------------------------------------------------------------------------


class TestMultiSymbolRouterTelegram:
    """Verify that the MultiSymbolRouter sends entry alerts via telegram."""

    def test_execution_tracker_sends_entry_alert(self):
        """When telegram_alerter is attached, entry signals produce alerts."""
        from unittest.mock import MagicMock, patch as _patch
        from strategy.orb import SignalType
        from strategy.strategy_engine import LiveSignal
        from risk.portfolio_risk import PortfolioRiskManager, PortfolioRiskConfig
        from scripts.run_paper import MultiSymbolRouter, SymbolPipeline
        from config.settings import InstrumentConfig

        # Minimal mock pipeline
        mock_paper = MagicMock()
        mock_risk = MagicMock()
        mock_risk.on_approved = None
        mock_engine = MagicMock()
        mock_engine._on_signal = None
        mock_analytics = MagicMock()
        mock_tracker = MagicMock()

        pipe = SymbolPipeline(
            symbol="MES",
            instrument=InstrumentConfig(),
            engine=mock_engine,
            risk=mock_risk,
            paper=mock_paper,
            analytics=mock_analytics,
            tracker=mock_tracker,
        )

        portfolio_risk = PortfolioRiskManager(PortfolioRiskConfig())
        mock_alerter = MagicMock()

        router = MultiSymbolRouter(
            pipelines={"MES": pipe},
            portfolio_risk=portfolio_risk,
            telegram_alerter=mock_alerter,
        )

        # Simulate an approved entry signal reaching the execution tracker
        sig = LiveSignal(
            signal_type=SignalType.LONG_ENTRY,
            timestamp=_ts(),
            entry=5000.0,
            stop=4990.0,
            take_profit=5015.0,
            direction="long",
            position_size=1.0,
            strategy_type="ema50_breakout",
            position_id="test_1",
            ml_prob=0.61,
            quality_score=0.58,
        )

        # Call the execution tracker directly
        exec_tracker = router._make_execution_tracker("MES")
        exec_tracker(sig)

        # Verify Telegram was called
        mock_alerter.send_entry_alert.assert_called_once()
        call_kwargs = mock_alerter.send_entry_alert.call_args
        assert call_kwargs[1]["symbol"] == "MES" or call_kwargs.kwargs["symbol"] == "MES"

    def test_execution_tracker_no_alert_on_exit(self):
        """Exit signals should NOT trigger entry alerts."""
        from strategy.orb import SignalType
        from strategy.strategy_engine import LiveSignal
        from risk.portfolio_risk import PortfolioRiskManager, PortfolioRiskConfig
        from scripts.run_paper import MultiSymbolRouter, SymbolPipeline
        from config.settings import InstrumentConfig

        mock_paper = MagicMock()
        mock_risk = MagicMock()
        mock_risk.on_approved = None
        mock_engine = MagicMock()
        mock_engine._on_signal = None

        pipe = SymbolPipeline(
            symbol="MES",
            instrument=InstrumentConfig(),
            engine=mock_engine,
            risk=mock_risk,
            paper=mock_paper,
            analytics=MagicMock(),
            tracker=MagicMock(),
        )

        portfolio_risk = PortfolioRiskManager(PortfolioRiskConfig())
        mock_alerter = MagicMock()

        router = MultiSymbolRouter(
            pipelines={"MES": pipe},
            portfolio_risk=portfolio_risk,
            telegram_alerter=mock_alerter,
        )

        exit_sig = LiveSignal(
            signal_type=SignalType.EXIT_TP,
            timestamp=_ts(),
            entry=5015.0,
            stop=0.0,
            take_profit=0.0,
            direction="long",
            position_size=1.0,
            strategy_type="ema50_breakout",
            position_id="test_1",
        )

        exec_tracker = router._make_execution_tracker("MES")
        exec_tracker(exit_sig)

        # Should NOT call send_entry_alert
        mock_alerter.send_entry_alert.assert_not_called()
