"""
OpenClaw driver for Tradovate UI automation.

This module provides the low-level interface between the execution layer
and the Tradovate desktop application via OpenClaw.

Design principles:
- Read before write — always verify UI state before clicking
- ATM templates handle SL/TP — we only click Buy or Sell
- Every click is logged
- All methods return structured results, never raise on UI errors

OpenClaw must be installed separately:
    pip install openclaw

NOTE: This driver is designed around Tradovate's DOM/chart trader layout.
Button labels and UI element paths may need adjustment if the Tradovate
layout changes.  All element identifiers are collected in _ELEMENTS dict
for easy maintenance.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from execution.validators import PostTradeState, UIState

logger = logging.getLogger(__name__)

# ── UI Element identifiers ───────────────────────────────────────────────────
# These are OpenClaw search patterns for Tradovate UI elements.
# Adjust if Tradovate changes UI layout.

_ELEMENTS = {
    # Window identification
    "window_title_pattern": "Tradovate",

    # Account / instrument
    "account_label_role": "text",
    "account_label_pattern": "Account",
    "symbol_tab_role": "tab",
    "symbol_input_role": "edit",

    # Order entry
    "quantity_field_role": "spinbutton",
    "quantity_field_name": "Qty",
    "buy_button_name": "Buy",
    "sell_button_name": "Sell",
    "buy_button_role": "button",
    "sell_button_role": "button",

    # ATM template
    "atm_dropdown_role": "combobox",
    "atm_dropdown_pattern": "ATM",

    # Position display
    "position_panel_role": "group",
    "position_panel_pattern": "Position",

    # Orders
    "orders_panel_role": "group",
    "orders_panel_pattern": "Order",
}


@dataclass
class ClickResult:
    """Outcome of a UI click action."""
    success: bool
    action: str  # "buy_click", "sell_click", "qty_set", etc.
    detail: str = ""
    elapsed_ms: int = 0


class OpenClawDriver:
    """
    Low-level Tradovate UI driver using OpenClaw.

    Usage:
        driver = OpenClawDriver()
        ui_state = driver.read_ui_state()
        # ... validate ...
        result = driver.click_buy()
        post_state = driver.read_post_trade_state()

    In dry-run mode, read methods work but click methods return
    simulated results without actually clicking.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self._dry_run = dry_run
        self._claw = None  # Lazy-loaded OpenClaw instance
        self._window = None
        logger.info("OpenClawDriver initialized (dry_run=%s)", dry_run)

    def _ensure_claw(self) -> bool:
        """
        Lazy-load OpenClaw and find the Tradovate window.

        Returns True if the window is found, False otherwise.
        """
        if self._claw is not None and self._window is not None:
            return True

        try:
            import openclaw  # type: ignore
            self._claw = openclaw
        except ImportError:
            logger.error(
                "OpenClaw is not installed. Install with: pip install openclaw"
            )
            return False

        try:
            windows = self._claw.find_windows(
                title_re=_ELEMENTS["window_title_pattern"]
            )
            if not windows:
                logger.warning("No Tradovate window found")
                self._window = None
                return False
            self._window = windows[0]
            logger.debug("Tradovate window found: %s", self._window.title)
            return True
        except Exception as e:
            logger.error("Failed to find Tradovate window: %s", e)
            self._window = None
            return False

    # ── Read methods ─────────────────────────────────────────────────────────

    def read_ui_state(self) -> UIState:
        """
        Read the current Tradovate UI state.

        Returns a UIState snapshot.  Works in both dry-run and live mode.
        """
        state = UIState()

        if not self._ensure_claw():
            state.read_errors.append("OpenClaw not available or Tradovate window not found")
            return state

        state.window_found = True
        state.window_title = getattr(self._window, "title", "Tradovate")

        # Read account label
        try:
            account_el = self._window.find_first(
                role=_ELEMENTS["account_label_role"],
                name_re=_ELEMENTS["account_label_pattern"],
            )
            if account_el:
                state.account_label = account_el.text
        except Exception as e:
            state.read_errors.append(f"account_label: {e}")

        # Read active symbol
        try:
            symbol_tabs = self._window.find_all(role=_ELEMENTS["symbol_tab_role"])
            for tab in symbol_tabs:
                if getattr(tab, "is_selected", False):
                    state.active_symbol = tab.text
                    break
            if state.active_symbol is None and symbol_tabs:
                # Fallback: try the window title for symbol info
                title = state.window_title
                for sym in ("MNQ", "MES", "NQ", "ES", "RTY"):
                    if sym in title.upper():
                        state.active_symbol = sym
                        break
        except Exception as e:
            state.read_errors.append(f"active_symbol: {e}")

        # Read quantity
        try:
            qty_el = self._window.find_first(
                role=_ELEMENTS["quantity_field_role"],
                name_re=_ELEMENTS["quantity_field_name"],
            )
            if qty_el:
                text = qty_el.text or qty_el.value
                state.quantity_value = int(text) if text else None
        except Exception as e:
            state.read_errors.append(f"quantity: {e}")

        # Read ATM template
        try:
            atm_el = self._window.find_first(
                role=_ELEMENTS["atm_dropdown_role"],
                name_re=_ELEMENTS["atm_dropdown_pattern"],
            )
            if atm_el:
                state.atm_template_name = atm_el.text or atm_el.value
        except Exception as e:
            state.read_errors.append(f"atm_template: {e}")

        # Read position info
        try:
            pos_el = self._window.find_first(
                role=_ELEMENTS["position_panel_role"],
                name_re=_ELEMENTS["position_panel_pattern"],
            )
            if pos_el and pos_el.text:
                text = pos_el.text.strip()
                if text and text != "0" and text.lower() != "flat":
                    state.has_open_position = True
                    if "long" in text.lower() or "+" in text:
                        state.open_position_side = "long"
                    elif "short" in text.lower() or "-" in text:
                        state.open_position_side = "short"
                    # Try to parse size
                    for part in text.split():
                        try:
                            val = abs(int(part))
                            if val > 0:
                                state.open_position_size = val
                                break
                        except ValueError:
                            continue
        except Exception as e:
            state.read_errors.append(f"position: {e}")

        # Read pending orders count
        try:
            order_el = self._window.find_first(
                role=_ELEMENTS["orders_panel_role"],
                name_re=_ELEMENTS["orders_panel_pattern"],
            )
            if order_el:
                children = order_el.children if hasattr(order_el, "children") else []
                state.pending_orders = len(children)
        except Exception as e:
            state.read_errors.append(f"pending_orders: {e}")

        # Check button visibility
        try:
            buy_btn = self._window.find_first(
                role=_ELEMENTS["buy_button_role"],
                name_re=_ELEMENTS["buy_button_name"],
            )
            state.buy_button_visible = buy_btn is not None
        except Exception as e:
            state.read_errors.append(f"buy_button: {e}")

        try:
            sell_btn = self._window.find_first(
                role=_ELEMENTS["sell_button_role"],
                name_re=_ELEMENTS["sell_button_name"],
            )
            state.sell_button_visible = sell_btn is not None
        except Exception as e:
            state.read_errors.append(f"sell_button: {e}")

        logger.debug(
            "UI state read: window=%s, symbol=%s, qty=%s, atm=%s, pos=%s, errors=%d",
            state.window_found,
            state.active_symbol,
            state.quantity_value,
            state.atm_template_name,
            state.has_open_position,
            len(state.read_errors),
        )
        return state

    def read_post_trade_state(self, timeout_ms: int = 5000) -> PostTradeState:
        """
        Read position state after an execution click.

        Waits up to timeout_ms for a position to appear.
        """
        post = PostTradeState()
        start = time.monotonic()
        deadline = start + (timeout_ms / 1000.0)

        while time.monotonic() < deadline:
            ui = self.read_ui_state()
            if ui.has_open_position:
                post.position_detected = True
                post.position_side = ui.open_position_side
                post.position_size = ui.open_position_size
                post.order_status = "filled"
                post.time_elapsed_ms = int((time.monotonic() - start) * 1000)
                post.read_errors = ui.read_errors
                return post
            time.sleep(0.5)

        post.time_elapsed_ms = int((time.monotonic() - start) * 1000)
        post.order_status = "timeout"
        logger.warning("Post-trade read timed out after %dms", post.time_elapsed_ms)
        return post

    # ── Click methods ────────────────────────────────────────────────────────

    def set_quantity(self, qty: int) -> ClickResult:
        """Set the quantity in the order entry."""
        if self._dry_run:
            logger.info("[DRY RUN] Would set quantity to %d", qty)
            return ClickResult(success=True, action="qty_set", detail=f"dry_run qty={qty}")

        if not self._ensure_claw():
            return ClickResult(success=False, action="qty_set", detail="window not found")

        try:
            start = time.monotonic()
            qty_el = self._window.find_first(
                role=_ELEMENTS["quantity_field_role"],
                name_re=_ELEMENTS["quantity_field_name"],
            )
            if not qty_el:
                return ClickResult(success=False, action="qty_set", detail="quantity field not found")

            qty_el.click()
            qty_el.select_all()
            qty_el.type_text(str(qty))
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info("Quantity set to %d (%dms)", qty, elapsed)
            return ClickResult(success=True, action="qty_set", detail=f"qty={qty}", elapsed_ms=elapsed)
        except Exception as e:
            logger.error("Failed to set quantity: %s", e)
            return ClickResult(success=False, action="qty_set", detail=str(e))

    def select_symbol(self, symbol: str) -> ClickResult:
        """Switch to the correct instrument tab or input."""
        if self._dry_run:
            logger.info("[DRY RUN] Would select symbol %s", symbol)
            return ClickResult(success=True, action="symbol_select", detail=f"dry_run symbol={symbol}")

        if not self._ensure_claw():
            return ClickResult(success=False, action="symbol_select", detail="window not found")

        try:
            start = time.monotonic()
            tabs = self._window.find_all(role=_ELEMENTS["symbol_tab_role"])
            for tab in tabs:
                if symbol.upper() in (tab.text or "").upper():
                    tab.click()
                    elapsed = int((time.monotonic() - start) * 1000)
                    logger.info("Symbol tab selected: %s (%dms)", symbol, elapsed)
                    return ClickResult(success=True, action="symbol_select", detail=f"tab={symbol}", elapsed_ms=elapsed)

            return ClickResult(success=False, action="symbol_select", detail=f"tab not found for {symbol}")
        except Exception as e:
            logger.error("Failed to select symbol: %s", e)
            return ClickResult(success=False, action="symbol_select", detail=str(e))

    def click_buy(self) -> ClickResult:
        """Click the BUY button."""
        return self._click_order_button("BUY")

    def click_sell(self) -> ClickResult:
        """Click the SELL button."""
        return self._click_order_button("SELL")

    def _click_order_button(self, side: str) -> ClickResult:
        """Click Buy or Sell button."""
        action = f"{side.lower()}_click"

        if self._dry_run:
            logger.info("[DRY RUN] Would click %s", side)
            return ClickResult(success=True, action=action, detail=f"dry_run {side}")

        if not self._ensure_claw():
            return ClickResult(success=False, action=action, detail="window not found")

        try:
            start = time.monotonic()
            btn_name = _ELEMENTS["buy_button_name"] if side == "BUY" else _ELEMENTS["sell_button_name"]
            btn_role = _ELEMENTS["buy_button_role"] if side == "BUY" else _ELEMENTS["sell_button_role"]

            btn = self._window.find_first(role=btn_role, name_re=btn_name)
            if not btn:
                return ClickResult(success=False, action=action, detail=f"{side} button not found")

            btn.click()
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info("%s button clicked (%dms)", side, elapsed)
            return ClickResult(success=True, action=action, detail=side, elapsed_ms=elapsed)
        except Exception as e:
            logger.error("Failed to click %s: %s", side, e)
            return ClickResult(success=False, action=action, detail=str(e))

    def take_screenshot(self, filepath: str) -> bool:
        """
        Capture a screenshot of the Tradovate window.

        Returns True if saved successfully.
        """
        if not self._ensure_claw():
            logger.warning("Cannot take screenshot — window not found")
            return False

        try:
            if hasattr(self._window, "screenshot"):
                self._window.screenshot(filepath)
            elif hasattr(self._claw, "screenshot"):
                self._claw.screenshot(filepath)
            else:
                logger.warning("OpenClaw does not expose a screenshot method")
                return False
            logger.info("Screenshot saved: %s", filepath)
            return True
        except Exception as e:
            logger.error("Screenshot failed: %s", e)
            return False

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        self._dry_run = value
        logger.info("OpenClawDriver dry_run set to %s", value)
