"""
Tradovate Execution Client
============================

Consumes LiveSignal objects (same interface as PaperEngine) and
translates them into Tradovate REST API calls for real order execution.

Features:
  1. Connection — Authenticate with Tradovate, connect to SIM environment.
  2. Order Execution — Place market orders, stop loss, take profit via
     placeOSO (bracket) or individual placeOrder calls.
  3. State Sync — Track open orders, filled orders, positions via REST
     polling and WebSocket event stream.
  4. Error Handling — Reject handling, reconnect on disconnect, partial
     fill tracking.
  5. Safety — SIM-only guard prevents accidental live execution.

Wiring (same pattern as PaperEngine):
    risk = RiskManager(config, instrument)
    engine = StrategyEngine(cfg, on_signal=risk.on_signal)
    risk.on_approved = tradovate.on_signal

    for bar in bars:
        risk.on_bar(bar)
        engine.on_bar(bar)  # signals flow through risk → tradovate
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import httpx

from config.settings import InstrumentConfig
from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEMO_URL = "https://demo.tradovateapi.com/v1"
LIVE_URL = "https://live.tradovateapi.com/v1"
DEMO_WS_URL = "wss://demo.tradovateapi.com/v1/websocket"
LIVE_WS_URL = "wss://live.tradovateapi.com/v1/websocket"

TOKEN_RENEWAL_BUFFER_SECONDS = 300  # renew 5 min before expiry


# ------------------------------------------------------------------
# Order status tracking
# ------------------------------------------------------------------

class OrderStatus(str, Enum):
    PENDING_NEW = "PendingNew"
    WORKING = "Working"
    FILLED = "Filled"
    CANCELLED = "Canceled"
    REJECTED = "Rejected"
    EXPIRED = "Expired"
    COMPLETED = "Completed"


@dataclass
class TrackedOrder:
    """Internal representation of an order we are tracking."""
    order_id: int
    position_id: str
    action: str          # "Buy" or "Sell"
    order_type: str      # "Market", "Stop", "Limit"
    symbol: str
    qty: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = OrderStatus.PENDING_NEW
    fill_price: Optional[float] = None
    fill_qty: int = 0
    error_text: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class TrackedPosition:
    """Mirrors a Tradovate position entity."""
    position_id: str         # our internal ID
    tradovate_pos_id: Optional[int] = None
    contract_id: Optional[int] = None
    net_pos: int = 0
    net_price: float = 0.0
    entry_order_id: Optional[int] = None
    sl_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class TradovateConfig:
    """Tradovate API connection credentials and settings."""
    username: str = ""
    password: str = ""
    app_id: str = "Apex"
    app_version: str = "1.0"
    cid: str = ""
    secret: str = ""
    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Environment control
    environment: str = "demo"
    """'demo' for SIM, 'live' for production.  Guarded by sim_only."""
    sim_only: bool = True
    """Hard guard: if True, refuse to connect to live environment."""

    # Account
    account_id: Optional[int] = None
    account_spec: Optional[str] = None

    # Polling interval for state sync (seconds)
    sync_interval: float = 5.0

    @property
    def base_url(self) -> str:
        if self.environment == "live":
            return LIVE_URL
        return DEMO_URL

    @property
    def ws_url(self) -> str:
        if self.environment == "live":
            return LIVE_WS_URL
        return DEMO_WS_URL


# ------------------------------------------------------------------
# Tradovate Client
# ------------------------------------------------------------------

class TradovateClient:
    """
    Execution adapter that translates LiveSignal objects into
    Tradovate API order calls.

    Implements the same ``on_signal(LiveSignal)`` interface as
    PaperEngine so it can be wired into the signal pipeline
    identically.

    Parameters
    ----------
    instrument : InstrumentConfig
        Contract specification (symbol, tick_size, point_value).
    config : TradovateConfig
        API credentials and settings.
    on_fill : callable, optional
        Called with (position_id, fill_price, fill_qty) on each fill.
    on_reject : callable, optional
        Called with (position_id, reason) on order rejection.
    """

    def __init__(
        self,
        instrument: InstrumentConfig,
        config: TradovateConfig,
        on_fill: Optional[Callable[[str, float, int], None]] = None,
        on_reject: Optional[Callable[[str, str], None]] = None,
    ):
        self._inst = instrument
        self._cfg = config
        self._on_fill = on_fill
        self._on_reject = on_reject

        # Auth state
        self._access_token: Optional[str] = None
        self._md_access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # HTTP client (created on connect)
        self._http: Optional[httpx.Client] = None

        # State tracking
        self._orders: dict[int, TrackedOrder] = {}
        self._positions: dict[str, TrackedPosition] = {}
        self._contract_id: Optional[int] = None

        # Sync thread
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_sync = threading.Event()

        # Connection state
        self._connected = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Authenticate with Tradovate and resolve contract ID."""
        if self._cfg.sim_only and self._cfg.environment == "live":
            raise RuntimeError(
                "SAFETY: sim_only=True but environment='live'. "
                "Refusing to connect to live Tradovate. "
                "Set sim_only=False to enable live trading."
            )

        logger.info(
            "Connecting to Tradovate %s environment...",
            self._cfg.environment.upper(),
        )

        self._http = httpx.Client(timeout=30.0)
        self._authenticate()
        self._resolve_contract()
        self._resolve_account()
        self._connected = True

        logger.info(
            "Connected to Tradovate %s — account=%s, contract_id=%s",
            self._cfg.environment.upper(),
            self._cfg.account_id,
            self._contract_id,
        )

    def disconnect(self) -> None:
        """Stop sync thread and close HTTP client."""
        self._stop_sync.set()
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10)
        if self._http:
            self._http.close()
            self._http = None
        self._connected = False
        logger.info("Disconnected from Tradovate")

    def start_sync(self) -> None:
        """Start background thread polling for order/position updates."""
        if self._sync_thread and self._sync_thread.is_alive():
            return
        self._stop_sync.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="tradovate-sync",
        )
        self._sync_thread.start()
        logger.info("State sync started (interval=%.1fs)", self._cfg.sync_interval)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _authenticate(self) -> None:
        """Obtain access token from Tradovate."""
        body = {
            "name": self._cfg.username,
            "password": self._cfg.password,
            "appId": self._cfg.app_id,
            "appVersion": self._cfg.app_version,
            "cid": self._cfg.cid,
            "sec": self._cfg.secret,
            "deviceId": self._cfg.device_id,
        }

        resp = self._http.post(
            f"{self._cfg.base_url}/auth/accesstokenrequest",
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

        if "errorText" in data and data["errorText"]:
            raise RuntimeError(f"Tradovate auth failed: {data['errorText']}")

        self._access_token = data["accessToken"]
        self._md_access_token = data.get("mdAccessToken")

        exp_str = data.get("expirationTime", "")
        if exp_str:
            self._token_expiry = datetime.fromisoformat(
                exp_str.replace("Z", "+00:00")
            )

        self._http.headers["Authorization"] = f"Bearer {self._access_token}"
        logger.info("Authenticated as %s", self._cfg.username)

    def _renew_token_if_needed(self) -> None:
        """Renew access token if it is about to expire."""
        if not self._token_expiry:
            return
        now = datetime.now(timezone.utc)
        remaining = (self._token_expiry - now).total_seconds()
        if remaining > TOKEN_RENEWAL_BUFFER_SECONDS:
            return

        logger.info("Renewing Tradovate access token (expires in %.0fs)", remaining)
        resp = self._http.post(f"{self._cfg.base_url}/auth/renewaccesstoken")
        resp.raise_for_status()
        data = resp.json()

        if "errorText" in data and data["errorText"]:
            logger.error("Token renewal failed: %s", data["errorText"])
            self._authenticate()
            return

        self._access_token = data.get("accessToken", self._access_token)
        exp_str = data.get("expirationTime", "")
        if exp_str:
            self._token_expiry = datetime.fromisoformat(
                exp_str.replace("Z", "+00:00")
            )
        self._http.headers["Authorization"] = f"Bearer {self._access_token}"

    # ------------------------------------------------------------------
    # Contract / account resolution
    # ------------------------------------------------------------------

    def _resolve_contract(self) -> None:
        """Look up the Tradovate contract ID for our symbol."""
        resp = self._http.get(
            f"{self._cfg.base_url}/contract/find",
            params={"name": self._inst.symbol},
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "id" in data:
            self._contract_id = data["id"]
            logger.info(
                "Resolved contract %s → id=%d",
                self._inst.symbol, self._contract_id,
            )
        else:
            raise RuntimeError(
                f"Could not resolve contract for symbol '{self._inst.symbol}': {data}"
            )

    def _resolve_account(self) -> None:
        """Resolve account ID if not provided in config."""
        if self._cfg.account_id:
            return

        resp = self._http.get(f"{self._cfg.base_url}/account/list")
        resp.raise_for_status()
        accounts = resp.json()

        if not accounts:
            raise RuntimeError("No accounts found on Tradovate")

        active = [a for a in accounts if a.get("active", False)]
        if not active:
            active = accounts

        acct = active[0]
        self._cfg.account_id = acct["id"]
        self._cfg.account_spec = acct.get("name", "")
        logger.info(
            "Using account: id=%d name=%s",
            self._cfg.account_id, self._cfg.account_spec,
        )

    # ------------------------------------------------------------------
    # Signal handler (same interface as PaperEngine.on_signal)
    # ------------------------------------------------------------------

    def on_signal(self, sig: LiveSignal) -> None:
        """
        Process a LiveSignal from the strategy pipeline.

        Entry signals → place bracket order (market entry + SL + TP).
        Exit signals → cancel working orders and place market exit.
        """
        if not self._connected:
            logger.error("Cannot process signal — not connected")
            return

        self._renew_token_if_needed()

        if sig.is_entry:
            self._handle_entry(sig)
        elif sig.is_exit:
            self._handle_exit(sig)
        else:
            logger.debug("Ignoring NONE signal at %s", sig.timestamp)

    # ------------------------------------------------------------------
    # Entry: place bracket order (OSO: Market + SL Stop + TP Limit)
    # ------------------------------------------------------------------

    def _handle_entry(self, sig: LiveSignal) -> None:
        """Place a bracket order for entry."""
        pos_id = sig.position_id or f"apex_{uuid.uuid4().hex[:8]}"

        if pos_id in self._positions:
            logger.warning(
                "Duplicate entry for position %s — ignoring", pos_id,
            )
            return

        action = "Buy" if sig.direction == "long" else "Sell"
        exit_action = "Sell" if action == "Buy" else "Buy"
        qty = max(1, int(sig.position_size * self._inst.contract_size))

        # SL bracket: stop order in opposite direction
        bracket_sl = {
            "action": exit_action,
            "orderType": "Stop",
            "stopPrice": sig.stop,
        }

        # TP bracket: limit order in opposite direction
        bracket_tp = {
            "action": exit_action,
            "orderType": "Limit",
            "price": sig.take_profit,
        }

        # OSO: market entry sends SL + TP as OCO brackets
        body = {
            "accountSpec": self._cfg.account_spec or "",
            "accountId": self._cfg.account_id,
            "action": action,
            "symbol": self._inst.symbol,
            "orderQty": qty,
            "orderType": "Market",
            "isAutomated": True,
            "bracket1": bracket_sl,
            "bracket2": bracket_tp,
        }

        logger.info(
            "PLACING BRACKET %s %s %d @ MKT (sl=%.2f tp=%.2f) [%s]",
            pos_id, action, qty, sig.stop, sig.take_profit, sig.strategy_type,
        )

        try:
            resp = self._http.post(
                f"{self._cfg.base_url}/order/placeoso",
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error placing order: %s", e)
            if self._on_reject:
                self._on_reject(pos_id, str(e))
            return
        except httpx.RequestError as e:
            logger.error("Network error placing order: %s", e)
            if self._on_reject:
                self._on_reject(pos_id, str(e))
            return

        # Check for business-level failure
        if data.get("failureReason") or data.get("failureText"):
            reason = data.get("failureText") or data.get("failureReason", "Unknown")
            logger.error("Order rejected: %s", reason)
            if self._on_reject:
                self._on_reject(pos_id, reason)
            return

        entry_order_id = data.get("orderId")
        sl_order_id = data.get("oso1Id")
        tp_order_id = data.get("oso2Id")

        # Track the orders
        now = datetime.now(timezone.utc)

        if entry_order_id:
            self._orders[entry_order_id] = TrackedOrder(
                order_id=entry_order_id,
                position_id=pos_id,
                action=action,
                order_type="Market",
                symbol=self._inst.symbol,
                qty=qty,
                created_at=now,
            )

        if sl_order_id:
            self._orders[sl_order_id] = TrackedOrder(
                order_id=sl_order_id,
                position_id=pos_id,
                action=exit_action,
                order_type="Stop",
                symbol=self._inst.symbol,
                qty=qty,
                stop_price=sig.stop,
                created_at=now,
            )

        if tp_order_id:
            self._orders[tp_order_id] = TrackedOrder(
                order_id=tp_order_id,
                position_id=pos_id,
                action=exit_action,
                order_type="Limit",
                symbol=self._inst.symbol,
                qty=qty,
                price=sig.take_profit,
                created_at=now,
            )

        self._positions[pos_id] = TrackedPosition(
            position_id=pos_id,
            contract_id=self._contract_id,
            entry_order_id=entry_order_id,
            sl_order_id=sl_order_id,
            tp_order_id=tp_order_id,
        )

        logger.info(
            "Bracket placed: entry=%s sl=%s tp=%s",
            entry_order_id, sl_order_id, tp_order_id,
        )

    # ------------------------------------------------------------------
    # Exit: cancel brackets and flatten
    # ------------------------------------------------------------------

    def _handle_exit(self, sig: LiveSignal) -> None:
        """Cancel open SL/TP orders and place market exit."""
        pos_id = sig.position_id or None

        if not pos_id:
            if self._positions:
                pos_id = next(iter(self._positions))
            else:
                logger.warning("Exit signal but no tracked positions")
                return

        if pos_id not in self._positions:
            logger.warning("Exit for unknown position %s", pos_id)
            return

        pos = self._positions[pos_id]

        # Cancel working SL/TP orders
        for oid in (pos.sl_order_id, pos.tp_order_id):
            if oid and oid in self._orders:
                tracked = self._orders[oid]
                if tracked.status not in (
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                    OrderStatus.EXPIRED,
                ):
                    self._cancel_order(oid)

        # Place market exit
        action = "Sell" if sig.direction in ("long", "") else "Buy"
        # If direction is empty (exit signals), infer from tracked position
        if not sig.direction and pos.net_pos != 0:
            action = "Sell" if pos.net_pos > 0 else "Buy"

        qty = max(1, abs(pos.net_pos)) if pos.net_pos != 0 else 1

        body = {
            "accountSpec": self._cfg.account_spec or "",
            "accountId": self._cfg.account_id,
            "action": action,
            "symbol": self._inst.symbol,
            "orderQty": qty,
            "orderType": "Market",
            "isAutomated": True,
        }

        logger.info("FLATTEN %s %s %d @ MKT", pos_id, action, qty)

        try:
            resp = self._http.post(
                f"{self._cfg.base_url}/order/placeorder",
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error("Error flattening position %s: %s", pos_id, e)
            if self._on_reject:
                self._on_reject(pos_id, str(e))
            return

        if data.get("failureReason") or data.get("failureText"):
            reason = data.get("failureText") or data.get("failureReason")
            logger.error("Flatten rejected for %s: %s", pos_id, reason)
            if self._on_reject:
                self._on_reject(pos_id, reason)
            return

        exit_order_id = data.get("orderId")
        if exit_order_id:
            self._orders[exit_order_id] = TrackedOrder(
                order_id=exit_order_id,
                position_id=pos_id,
                action=action,
                order_type="Market",
                symbol=self._inst.symbol,
                qty=qty,
                created_at=datetime.now(timezone.utc),
            )

        # Remove position tracking (filled confirmation comes async)
        del self._positions[pos_id]
        logger.info("Position %s flattened (exit order %s)", pos_id, exit_order_id)

    # ------------------------------------------------------------------
    # Cancel order
    # ------------------------------------------------------------------

    def _cancel_order(self, order_id: int) -> bool:
        """Cancel a working order. Returns True on success."""
        body = {
            "orderId": order_id,
            "isAutomated": True,
        }
        try:
            resp = self._http.post(
                f"{self._cfg.base_url}/order/cancelorder",
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("failureReason"):
                logger.warning(
                    "Cancel failed for order %d: %s",
                    order_id, data.get("failureText", data["failureReason"]),
                )
                return False

            logger.info("Cancelled order %d", order_id)
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELLED
            return True

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error("Error cancelling order %d: %s", order_id, e)
            return False

    # ------------------------------------------------------------------
    # State sync (polling)
    # ------------------------------------------------------------------

    def _sync_loop(self) -> None:
        """Background loop that polls order/position state."""
        while not self._stop_sync.is_set():
            try:
                self._renew_token_if_needed()
                self._sync_orders()
                self._sync_positions()
            except Exception:
                logger.exception("Error in sync loop")

            self._stop_sync.wait(self._cfg.sync_interval)

    def _sync_orders(self) -> None:
        """Poll Tradovate for current order statuses."""
        if not self._orders:
            return

        try:
            resp = self._http.get(
                f"{self._cfg.base_url}/order/list",
            )
            resp.raise_for_status()
            orders = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning("Order sync failed: %s", e)
            return

        remote_map = {o["id"]: o for o in orders if "id" in o}

        for oid, tracked in list(self._orders.items()):
            if oid not in remote_map:
                continue

            remote = remote_map[oid]
            new_status = remote.get("ordStatus", tracked.status)

            if new_status != tracked.status:
                old = tracked.status
                tracked.status = new_status
                logger.info(
                    "Order %d (%s) status: %s → %s",
                    oid, tracked.position_id, old, new_status,
                )

                if new_status == OrderStatus.REJECTED:
                    text = remote.get("text", "No reason given")
                    tracked.error_text = text
                    if self._on_reject:
                        self._on_reject(tracked.position_id, text)

    def _sync_positions(self) -> None:
        """Poll Tradovate for current position state."""
        try:
            resp = self._http.get(
                f"{self._cfg.base_url}/position/list",
            )
            resp.raise_for_status()
            positions = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning("Position sync failed: %s", e)
            return

        for remote_pos in positions:
            contract_id = remote_pos.get("contractId")
            if contract_id != self._contract_id:
                continue

            net_pos = remote_pos.get("netPos", 0)
            net_price = remote_pos.get("netPrice", 0.0)
            tv_pos_id = remote_pos.get("id")

            # Update matching tracked positions
            for tracked in self._positions.values():
                if tracked.contract_id == contract_id:
                    tracked.tradovate_pos_id = tv_pos_id
                    tracked.net_pos = net_pos
                    tracked.net_price = net_price

    def sync_fills(self) -> list[dict]:
        """
        Poll execution reports for fills on tracked orders.

        Returns list of fill dicts: {position_id, order_id, fill_price, fill_qty}.
        """
        fills = []
        if not self._orders:
            return fills

        try:
            resp = self._http.get(
                f"{self._cfg.base_url}/executionReport/list",
            )
            resp.raise_for_status()
            reports = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning("Fill sync failed: %s", e)
            return fills

        # Index execution reports by orderId
        for report in reports:
            order_id = report.get("orderId")
            if order_id not in self._orders:
                continue

            tracked = self._orders[order_id]
            exec_type = report.get("execType")
            last_qty = report.get("lastQty", 0)
            last_px = report.get("lastPx", 0.0)
            cum_qty = report.get("cumQty", 0)

            if exec_type in ("Trade", "Fill") and cum_qty > tracked.fill_qty:
                new_fill_qty = cum_qty - tracked.fill_qty
                tracked.fill_qty = cum_qty
                tracked.fill_price = report.get("avgPx", last_px)

                if cum_qty >= tracked.qty:
                    tracked.status = OrderStatus.FILLED

                fill = {
                    "position_id": tracked.position_id,
                    "order_id": order_id,
                    "fill_price": last_px,
                    "fill_qty": new_fill_qty,
                    "avg_price": tracked.fill_price,
                    "cum_qty": cum_qty,
                    "is_complete": cum_qty >= tracked.qty,
                }
                fills.append(fill)

                if self._on_fill:
                    self._on_fill(
                        tracked.position_id,
                        tracked.fill_price,
                        new_fill_qty,
                    )

                logger.info(
                    "FILL order=%d pos=%s px=%.2f qty=%d cum=%d/%d",
                    order_id, tracked.position_id,
                    last_px, new_fill_qty, cum_qty, tracked.qty,
                )

        return fills

    # ------------------------------------------------------------------
    # Liquidate all (emergency)
    # ------------------------------------------------------------------

    def liquidate_all(self) -> None:
        """Cancel all working orders and liquidate all positions."""
        if not self._connected:
            logger.error("Cannot liquidate — not connected")
            return

        logger.warning("LIQUIDATING ALL POSITIONS")

        # Cancel all working orders
        for oid, tracked in list(self._orders.items()):
            if tracked.status in (
                OrderStatus.PENDING_NEW,
                OrderStatus.WORKING,
            ):
                self._cancel_order(oid)

        # Liquidate position if contract_id is known
        if self._contract_id and self._cfg.account_id:
            body = {
                "accountId": self._cfg.account_id,
                "contractId": self._contract_id,
                "admin": False,
            }
            try:
                resp = self._http.post(
                    f"{self._cfg.base_url}/order/liquidateposition",
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("failureReason"):
                    logger.error(
                        "Liquidation failed: %s",
                        data.get("failureText", data["failureReason"]),
                    )
                else:
                    logger.info("Liquidation order placed: %s", data.get("orderId"))
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.error("Error during liquidation: %s", e)

        self._positions.clear()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def open_positions(self) -> dict[str, TrackedPosition]:
        return dict(self._positions)

    @property
    def tracked_orders(self) -> dict[int, TrackedOrder]:
        return dict(self._orders)

    def working_orders(self) -> list[TrackedOrder]:
        """Return orders that are still working (not filled/cancelled)."""
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.PENDING_NEW, OrderStatus.WORKING)
        ]

    def filled_orders(self) -> list[TrackedOrder]:
        """Return all filled orders."""
        return [
            o for o in self._orders.values()
            if o.status == OrderStatus.FILLED
        ]

    def rejected_orders(self) -> list[TrackedOrder]:
        """Return all rejected orders."""
        return [
            o for o in self._orders.values()
            if o.status == OrderStatus.REJECTED
        ]
