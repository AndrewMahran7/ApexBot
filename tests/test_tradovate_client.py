"""Tests for the Tradovate execution client."""

import datetime
import json
import uuid
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from config.settings import InstrumentConfig
from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal
from strategy.tradovate_client import (
    TradovateClient,
    TradovateConfig,
    TrackedOrder,
    TrackedPosition,
    OrderStatus,
    DEMO_URL,
    LIVE_URL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))


def _ts(hour: int = 10, minute: int = 0, day: int = 2) -> datetime.datetime:
    return datetime.datetime(2024, 1, day, hour, minute, tzinfo=_ET)


def _entry_signal(
    direction: str = "long",
    entry: float = 5000.0,
    stop: float = 4990.0,
    tp: float = 5015.0,
    pos_id: str = "pos_001",
) -> LiveSignal:
    return LiveSignal(
        timestamp=_ts(),
        direction=direction,
        signal_type=SignalType.LONG_ENTRY if direction == "long" else SignalType.SHORT_ENTRY,
        entry=entry,
        stop=stop,
        take_profit=tp,
        position_size=1.0,
        strategy_type="ema50_breakout",
        reason="test",
        position_id=pos_id,
    )


def _exit_signal(
    signal_type: SignalType = SignalType.EXIT_TP,
    pos_id: str = "pos_001",
    exit_price: float = 5015.0,
) -> LiveSignal:
    return LiveSignal(
        timestamp=_ts(10, 30),
        direction="",
        signal_type=signal_type,
        entry=exit_price,
        stop=0.0,
        take_profit=0.0,
        position_size=1.0,
        strategy_type="ema50_breakout",
        reason="tp_hit",
        position_id=pos_id,
    )


def _make_config(**overrides) -> TradovateConfig:
    defaults = dict(
        username="testuser",
        password="testpass",
        app_id="TestApp",
        app_version="1.0",
        cid="8",
        secret="test-secret-key",
        device_id="test-device-123",
        environment="demo",
        sim_only=True,
        account_id=12345,
        account_spec="DEMO12345",
    )
    defaults.update(overrides)
    return TradovateConfig(**defaults)


def _make_instrument() -> InstrumentConfig:
    return InstrumentConfig(symbol="MESM4", tick_size=0.25, point_value=5.0, contract_size=1)


def _mock_http_client():
    """Create a mock httpx.Client with chainable post/get."""
    mock = MagicMock()
    mock.headers = {}
    return mock


def _auth_response():
    """Standard auth success response."""
    return {
        "accessToken": "test-token-abc123",
        "mdAccessToken": "test-md-token",
        "expirationTime": "2030-01-01T00:00:00Z",
        "userId": 100,
        "name": "testuser",
        "userStatus": "Active",
    }


def _contract_response():
    """Contract find response."""
    return {"id": 999, "name": "MESM4"}


def _account_list_response():
    """Account list response."""
    return [{"id": 12345, "name": "DEMO12345", "active": True}]


# ---------------------------------------------------------------------------
# SIM-Only Guard
# ---------------------------------------------------------------------------


class TestSimOnlyGuard:
    """Test that the safety guard prevents live connections."""

    def test_sim_only_blocks_live(self):
        cfg = _make_config(environment="live", sim_only=True)
        client = TradovateClient(_make_instrument(), cfg)
        with pytest.raises(RuntimeError, match="sim_only"):
            client.connect()

    def test_sim_only_allows_demo(self):
        cfg = _make_config(environment="demo", sim_only=True)
        client = TradovateClient(_make_instrument(), cfg)
        with patch("strategy.tradovate_client.httpx.Client") as MockHTTP:
            mock_http = _mock_http_client()
            MockHTTP.return_value = mock_http

            # Auth
            auth_resp = MagicMock()
            auth_resp.json.return_value = _auth_response()
            # Contract
            contract_resp = MagicMock()
            contract_resp.json.return_value = _contract_response()

            mock_http.post.return_value = auth_resp
            mock_http.get.return_value = contract_resp

            client.connect()
            assert client.is_connected

    def test_live_allowed_when_sim_only_false(self):
        cfg = _make_config(environment="live", sim_only=False)
        client = TradovateClient(_make_instrument(), cfg)
        with patch("strategy.tradovate_client.httpx.Client") as MockHTTP:
            mock_http = _mock_http_client()
            MockHTTP.return_value = mock_http

            auth_resp = MagicMock()
            auth_resp.json.return_value = _auth_response()
            contract_resp = MagicMock()
            contract_resp.json.return_value = _contract_response()

            mock_http.post.return_value = auth_resp
            mock_http.get.return_value = contract_resp

            client.connect()
            assert client.is_connected

    def test_config_urls(self):
        demo_cfg = _make_config(environment="demo")
        assert demo_cfg.base_url == DEMO_URL

        live_cfg = _make_config(environment="live", sim_only=False)
        assert live_cfg.base_url == LIVE_URL


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Test Tradovate API authentication flow."""

    def test_auth_sets_token(self):
        cfg = _make_config()
        client = TradovateClient(_make_instrument(), cfg)

        with patch("strategy.tradovate_client.httpx.Client") as MockHTTP:
            mock_http = _mock_http_client()
            MockHTTP.return_value = mock_http

            auth_resp = MagicMock()
            auth_resp.json.return_value = _auth_response()
            contract_resp = MagicMock()
            contract_resp.json.return_value = _contract_response()

            mock_http.post.return_value = auth_resp
            mock_http.get.return_value = contract_resp

            client.connect()

            assert client._access_token == "test-token-abc123"
            assert "Authorization" in mock_http.headers
            assert mock_http.headers["Authorization"] == "Bearer test-token-abc123"

    def test_auth_failure_raises(self):
        cfg = _make_config()
        client = TradovateClient(_make_instrument(), cfg)

        with patch("strategy.tradovate_client.httpx.Client") as MockHTTP:
            mock_http = _mock_http_client()
            MockHTTP.return_value = mock_http

            auth_resp = MagicMock()
            auth_resp.json.return_value = {
                "errorText": "Invalid credentials",
            }
            mock_http.post.return_value = auth_resp

            with pytest.raises(RuntimeError, match="auth failed"):
                client.connect()

    def test_auth_sends_correct_body(self):
        cfg = _make_config()
        client = TradovateClient(_make_instrument(), cfg)

        with patch("strategy.tradovate_client.httpx.Client") as MockHTTP:
            mock_http = _mock_http_client()
            MockHTTP.return_value = mock_http

            auth_resp = MagicMock()
            auth_resp.json.return_value = _auth_response()
            contract_resp = MagicMock()
            contract_resp.json.return_value = _contract_response()

            mock_http.post.return_value = auth_resp
            mock_http.get.return_value = contract_resp

            client.connect()

            call_args = mock_http.post.call_args_list[0]
            body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert body["name"] == "testuser"
            assert body["password"] == "testpass"
            assert body["appId"] == "TestApp"
            assert body["cid"] == "8"
            assert body["sec"] == "test-secret-key"
            assert body["deviceId"] == "test-device-123"


# ---------------------------------------------------------------------------
# Contract Resolution
# ---------------------------------------------------------------------------


class TestContractResolution:
    """Test contract ID lookup."""

    def test_resolves_contract_id(self):
        cfg = _make_config()
        client = TradovateClient(_make_instrument(), cfg)

        with patch("strategy.tradovate_client.httpx.Client") as MockHTTP:
            mock_http = _mock_http_client()
            MockHTTP.return_value = mock_http

            auth_resp = MagicMock()
            auth_resp.json.return_value = _auth_response()
            contract_resp = MagicMock()
            contract_resp.json.return_value = _contract_response()

            mock_http.post.return_value = auth_resp
            mock_http.get.return_value = contract_resp

            client.connect()
            assert client._contract_id == 999

    def test_contract_not_found_raises(self):
        cfg = _make_config()
        client = TradovateClient(_make_instrument(), cfg)

        with patch("strategy.tradovate_client.httpx.Client") as MockHTTP:
            mock_http = _mock_http_client()
            MockHTTP.return_value = mock_http

            auth_resp = MagicMock()
            auth_resp.json.return_value = _auth_response()
            bad_resp = MagicMock()
            bad_resp.json.return_value = []  # no match

            mock_http.post.return_value = auth_resp
            mock_http.get.return_value = bad_resp

            with pytest.raises(RuntimeError, match="Could not resolve"):
                client.connect()


# ---------------------------------------------------------------------------
# Entry Signal → Bracket Order
# ---------------------------------------------------------------------------


def _connected_client(on_fill=None, on_reject=None):
    """Create a client that is already 'connected' with mocked HTTP."""
    cfg = _make_config()
    inst = _make_instrument()
    client = TradovateClient(inst, cfg, on_fill=on_fill, on_reject=on_reject)

    mock_http = _mock_http_client()
    client._http = mock_http
    client._connected = True
    client._access_token = "test-token"
    client._contract_id = 999
    client._token_expiry = datetime.datetime(2030, 1, 1, tzinfo=datetime.timezone.utc)

    return client, mock_http


class TestEntrySignal:
    """Test that entry signals place correct bracket orders."""

    def test_long_entry_places_oso(self):
        client, mock_http = _connected_client()

        resp = MagicMock()
        resp.json.return_value = {"orderId": 100, "oso1Id": 101, "oso2Id": 102}
        mock_http.post.return_value = resp

        sig = _entry_signal(direction="long", entry=5000, stop=4990, tp=5015)
        client.on_signal(sig)

        # Verify placeoso was called
        call = mock_http.post.call_args
        assert "/order/placeoso" in call.args[0]

        body = call.kwargs["json"]
        assert body["action"] == "Buy"
        assert body["symbol"] == "MESM4"
        assert body["orderQty"] == 1
        assert body["orderType"] == "Market"
        assert body["isAutomated"] is True
        assert body["bracket1"]["action"] == "Sell"
        assert body["bracket1"]["orderType"] == "Stop"
        assert body["bracket1"]["stopPrice"] == 4990.0
        assert body["bracket2"]["action"] == "Sell"
        assert body["bracket2"]["orderType"] == "Limit"
        assert body["bracket2"]["price"] == 5015.0

    def test_short_entry_places_oso(self):
        client, mock_http = _connected_client()

        resp = MagicMock()
        resp.json.return_value = {"orderId": 200, "oso1Id": 201, "oso2Id": 202}
        mock_http.post.return_value = resp

        sig = _entry_signal(direction="short", entry=5000, stop=5010, tp=4985)
        client.on_signal(sig)

        call = mock_http.post.call_args
        body = call.kwargs["json"]
        assert body["action"] == "Sell"
        assert body["bracket1"]["action"] == "Buy"
        assert body["bracket1"]["stopPrice"] == 5010.0
        assert body["bracket2"]["action"] == "Buy"
        assert body["bracket2"]["price"] == 4985.0

    def test_entry_tracks_orders(self):
        client, mock_http = _connected_client()

        resp = MagicMock()
        resp.json.return_value = {"orderId": 100, "oso1Id": 101, "oso2Id": 102}
        mock_http.post.return_value = resp

        sig = _entry_signal(pos_id="pos_abc")
        client.on_signal(sig)

        assert 100 in client._orders
        assert 101 in client._orders
        assert 102 in client._orders
        assert client._orders[100].order_type == "Market"
        assert client._orders[101].order_type == "Stop"
        assert client._orders[102].order_type == "Limit"

    def test_entry_tracks_position(self):
        client, mock_http = _connected_client()

        resp = MagicMock()
        resp.json.return_value = {"orderId": 100, "oso1Id": 101, "oso2Id": 102}
        mock_http.post.return_value = resp

        sig = _entry_signal(pos_id="pos_xyz")
        client.on_signal(sig)

        assert "pos_xyz" in client._positions
        pos = client._positions["pos_xyz"]
        assert pos.entry_order_id == 100
        assert pos.sl_order_id == 101
        assert pos.tp_order_id == 102

    def test_duplicate_entry_ignored(self):
        client, mock_http = _connected_client()

        resp = MagicMock()
        resp.json.return_value = {"orderId": 100, "oso1Id": 101, "oso2Id": 102}
        mock_http.post.return_value = resp

        sig1 = _entry_signal(pos_id="dup")
        sig2 = _entry_signal(pos_id="dup")

        client.on_signal(sig1)
        client.on_signal(sig2)  # should be ignored

        assert mock_http.post.call_count == 1

    def test_entry_rejection_calls_callback(self):
        reject_calls = []
        client, mock_http = _connected_client(
            on_reject=lambda pid, reason: reject_calls.append((pid, reason))
        )

        resp = MagicMock()
        resp.json.return_value = {
            "failureReason": "AccountClosed",
            "failureText": "Account is inactive",
        }
        mock_http.post.return_value = resp

        sig = _entry_signal(pos_id="rejected_pos")
        client.on_signal(sig)

        assert len(reject_calls) == 1
        assert reject_calls[0][0] == "rejected_pos"
        assert "Account is inactive" in reject_calls[0][1]

    def test_not_connected_rejects_signal(self):
        cfg = _make_config()
        client = TradovateClient(_make_instrument(), cfg)
        assert not client.is_connected

        sig = _entry_signal()
        client.on_signal(sig)  # should not raise, just log error


# ---------------------------------------------------------------------------
# Exit Signal → Cancel + Flatten
# ---------------------------------------------------------------------------


class TestExitSignal:
    """Test that exit signals cancel brackets and flatten."""

    def test_exit_cancels_sl_tp_and_flattens(self):
        client, mock_http = _connected_client()

        # Setup: create a tracked position
        client._positions["pos_001"] = TrackedPosition(
            position_id="pos_001",
            contract_id=999,
            entry_order_id=100,
            sl_order_id=101,
            tp_order_id=102,
            net_pos=1,
        )
        client._orders[101] = TrackedOrder(
            order_id=101, position_id="pos_001", action="Sell",
            order_type="Stop", symbol="MESM4", qty=1,
            status=OrderStatus.WORKING,
        )
        client._orders[102] = TrackedOrder(
            order_id=102, position_id="pos_001", action="Sell",
            order_type="Limit", symbol="MESM4", qty=1,
            status=OrderStatus.WORKING,
        )

        # Mock responses: cancel + placeorder
        cancel_resp = MagicMock()
        cancel_resp.json.return_value = {"commandId": 1}
        flatten_resp = MagicMock()
        flatten_resp.json.return_value = {"orderId": 300}

        mock_http.post.return_value = cancel_resp  # default for cancels

        def side_effect(url, **kwargs):
            r = MagicMock()
            if "cancelorder" in url:
                r.json.return_value = {"commandId": 1}
            elif "placeorder" in url:
                r.json.return_value = {"orderId": 300}
            return r

        mock_http.post.side_effect = side_effect

        sig = _exit_signal(pos_id="pos_001")
        client.on_signal(sig)

        # Should have called cancel for SL and TP, then placeorder
        post_calls = [c.args[0] for c in mock_http.post.call_args_list]
        assert any("cancelorder" in url for url in post_calls)
        assert any("placeorder" in url for url in post_calls)

        # Position should be removed
        assert "pos_001" not in client._positions

    def test_exit_for_unknown_position_warns(self):
        client, mock_http = _connected_client()

        sig = _exit_signal(pos_id="nonexistent")
        # Should not raise, just log warning
        client.on_signal(sig)
        assert mock_http.post.call_count == 0

    def test_exit_already_filled_skips_cancel(self):
        client, mock_http = _connected_client()

        client._positions["pos_001"] = TrackedPosition(
            position_id="pos_001",
            contract_id=999,
            sl_order_id=101,
            tp_order_id=102,
            net_pos=1,
        )
        # SL already filled, TP still working
        client._orders[101] = TrackedOrder(
            order_id=101, position_id="pos_001", action="Sell",
            order_type="Stop", symbol="MESM4", qty=1,
            status=OrderStatus.FILLED,
        )
        client._orders[102] = TrackedOrder(
            order_id=102, position_id="pos_001", action="Sell",
            order_type="Limit", symbol="MESM4", qty=1,
            status=OrderStatus.WORKING,
        )

        def side_effect(url, **kwargs):
            r = MagicMock()
            if "cancelorder" in url:
                r.json.return_value = {"commandId": 1}
            elif "placeorder" in url:
                r.json.return_value = {"orderId": 300}
            return r

        mock_http.post.side_effect = side_effect

        sig = _exit_signal(pos_id="pos_001")
        client.on_signal(sig)

        # Should cancel only TP (102), not SL (101 already filled)
        cancel_calls = [
            c for c in mock_http.post.call_args_list
            if "cancelorder" in c.args[0]
        ]
        assert len(cancel_calls) == 1
        cancel_body = cancel_calls[0].kwargs.get("json", {})
        assert cancel_body.get("orderId") == 102


# ---------------------------------------------------------------------------
# Cancel Order
# ---------------------------------------------------------------------------


class TestCancelOrder:
    """Test order cancellation."""

    def test_cancel_success(self):
        client, mock_http = _connected_client()

        client._orders[500] = TrackedOrder(
            order_id=500, position_id="pos", action="Sell",
            order_type="Stop", symbol="MESM4", qty=1,
            status=OrderStatus.WORKING,
        )

        resp = MagicMock()
        resp.json.return_value = {"commandId": 10}
        mock_http.post.return_value = resp

        result = client._cancel_order(500)
        assert result is True
        assert client._orders[500].status == OrderStatus.CANCELLED

    def test_cancel_failure(self):
        client, mock_http = _connected_client()

        client._orders[500] = TrackedOrder(
            order_id=500, position_id="pos", action="Sell",
            order_type="Stop", symbol="MESM4", qty=1,
            status=OrderStatus.WORKING,
        )

        resp = MagicMock()
        resp.json.return_value = {
            "failureReason": "OrderNotFound",
            "failureText": "Order does not exist",
        }
        mock_http.post.return_value = resp

        result = client._cancel_order(500)
        assert result is False


# ---------------------------------------------------------------------------
# State Sync
# ---------------------------------------------------------------------------


class TestStateSync:
    """Test order/position state synchronization."""

    def test_sync_orders_updates_status(self):
        client, mock_http = _connected_client()

        client._orders[100] = TrackedOrder(
            order_id=100, position_id="pos", action="Buy",
            order_type="Market", symbol="MESM4", qty=1,
            status=OrderStatus.PENDING_NEW,
        )

        resp = MagicMock()
        resp.json.return_value = [
            {"id": 100, "ordStatus": "Filled", "action": "Buy"},
        ]
        mock_http.get.return_value = resp

        client._sync_orders()
        assert client._orders[100].status == "Filled"

    def test_sync_orders_fires_reject_callback(self):
        rejects = []
        client, mock_http = _connected_client(
            on_reject=lambda pid, reason: rejects.append((pid, reason))
        )

        client._orders[100] = TrackedOrder(
            order_id=100, position_id="pos_r", action="Buy",
            order_type="Market", symbol="MESM4", qty=1,
            status=OrderStatus.PENDING_NEW,
        )

        resp = MagicMock()
        resp.json.return_value = [
            {"id": 100, "ordStatus": "Rejected", "text": "Insufficient margin"},
        ]
        mock_http.get.return_value = resp

        client._sync_orders()
        assert len(rejects) == 1
        assert rejects[0][0] == "pos_r"
        assert "Insufficient margin" in rejects[0][1]

    def test_sync_positions_updates_net_pos(self):
        client, mock_http = _connected_client()

        client._positions["pos_001"] = TrackedPosition(
            position_id="pos_001",
            contract_id=999,
            net_pos=0,
        )

        resp = MagicMock()
        resp.json.return_value = [
            {"id": 50, "contractId": 999, "netPos": 1, "netPrice": 5005.0},
        ]
        mock_http.get.return_value = resp

        client._sync_positions()
        assert client._positions["pos_001"].net_pos == 1
        assert client._positions["pos_001"].net_price == 5005.0
        assert client._positions["pos_001"].tradovate_pos_id == 50


# ---------------------------------------------------------------------------
# Fill Tracking
# ---------------------------------------------------------------------------


class TestFillTracking:
    """Test execution report / fill detection."""

    def test_sync_fills_detects_fill(self):
        fills_received = []
        client, mock_http = _connected_client(
            on_fill=lambda pid, px, qty: fills_received.append((pid, px, qty))
        )

        client._orders[100] = TrackedOrder(
            order_id=100, position_id="pos_f", action="Buy",
            order_type="Market", symbol="MESM4", qty=1,
            status=OrderStatus.WORKING,
        )

        resp = MagicMock()
        resp.json.return_value = [
            {
                "orderId": 100,
                "execType": "Fill",
                "lastQty": 1,
                "lastPx": 5001.50,
                "avgPx": 5001.50,
                "cumQty": 1,
            },
        ]
        mock_http.get.return_value = resp

        fills = client.sync_fills()

        assert len(fills) == 1
        assert fills[0]["position_id"] == "pos_f"
        assert fills[0]["fill_price"] == 5001.50
        assert fills[0]["is_complete"] is True
        assert len(fills_received) == 1
        assert client._orders[100].status == OrderStatus.FILLED

    def test_partial_fill_tracked(self):
        client, mock_http = _connected_client()

        client._orders[200] = TrackedOrder(
            order_id=200, position_id="pos_pf", action="Buy",
            order_type="Market", symbol="MESM4", qty=3,
            status=OrderStatus.WORKING,
        )

        resp = MagicMock()
        resp.json.return_value = [
            {
                "orderId": 200,
                "execType": "Trade",
                "lastQty": 1,
                "lastPx": 5000.0,
                "avgPx": 5000.0,
                "cumQty": 1,
            },
        ]
        mock_http.get.return_value = resp

        fills = client.sync_fills()
        assert len(fills) == 1
        assert fills[0]["is_complete"] is False
        assert fills[0]["cum_qty"] == 1
        assert client._orders[200].fill_qty == 1
        assert client._orders[200].status != OrderStatus.FILLED  # not complete yet


# ---------------------------------------------------------------------------
# Liquidate All
# ---------------------------------------------------------------------------


class TestLiquidateAll:
    """Test emergency liquidation."""

    def test_liquidate_cancels_and_flattens(self):
        client, mock_http = _connected_client()

        client._orders[100] = TrackedOrder(
            order_id=100, position_id="pos", action="Sell",
            order_type="Stop", symbol="MESM4", qty=1,
            status=OrderStatus.WORKING,
        )
        client._positions["pos"] = TrackedPosition(
            position_id="pos", contract_id=999, net_pos=1,
        )

        def side_effect(url, **kwargs):
            r = MagicMock()
            if "cancelorder" in url:
                r.json.return_value = {"commandId": 1}
            elif "liquidateposition" in url:
                r.json.return_value = {"orderId": 999}
            return r

        mock_http.post.side_effect = side_effect

        client.liquidate_all()

        post_urls = [c.args[0] for c in mock_http.post.call_args_list]
        assert any("cancelorder" in u for u in post_urls)
        assert any("liquidateposition" in u for u in post_urls)
        assert len(client._positions) == 0


# ---------------------------------------------------------------------------
# Query Helpers
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    """Test working_orders, filled_orders, rejected_orders."""

    def test_working_orders(self):
        client, _ = _connected_client()

        client._orders[1] = TrackedOrder(
            order_id=1, position_id="p", action="Buy",
            order_type="Market", symbol="MESM4", qty=1,
            status=OrderStatus.WORKING,
        )
        client._orders[2] = TrackedOrder(
            order_id=2, position_id="p", action="Sell",
            order_type="Stop", symbol="MESM4", qty=1,
            status=OrderStatus.FILLED,
        )
        client._orders[3] = TrackedOrder(
            order_id=3, position_id="p", action="Sell",
            order_type="Limit", symbol="MESM4", qty=1,
            status=OrderStatus.PENDING_NEW,
        )

        working = client.working_orders()
        assert len(working) == 2  # WORKING + PENDING_NEW

    def test_filled_orders(self):
        client, _ = _connected_client()

        client._orders[1] = TrackedOrder(
            order_id=1, position_id="p", action="Buy",
            order_type="Market", symbol="MESM4", qty=1,
            status=OrderStatus.FILLED,
        )
        client._orders[2] = TrackedOrder(
            order_id=2, position_id="p", action="Sell",
            order_type="Stop", symbol="MESM4", qty=1,
            status=OrderStatus.WORKING,
        )

        filled = client.filled_orders()
        assert len(filled) == 1
        assert filled[0].order_id == 1

    def test_rejected_orders(self):
        client, _ = _connected_client()

        client._orders[1] = TrackedOrder(
            order_id=1, position_id="p", action="Buy",
            order_type="Market", symbol="MESM4", qty=1,
            status=OrderStatus.REJECTED,
            error_text="Insufficient margin",
        )

        rejected = client.rejected_orders()
        assert len(rejected) == 1
        assert rejected[0].error_text == "Insufficient margin"


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


class TestDisconnect:
    """Test clean disconnection."""

    def test_disconnect_cleans_up(self):
        client, mock_http = _connected_client()

        client.disconnect()
        assert not client.is_connected
        mock_http.close.assert_called_once()
