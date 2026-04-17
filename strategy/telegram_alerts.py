"""
Telegram Alert Module
======================

Sends formatted trade alerts to a Telegram chat via the Bot API.

Credentials are loaded from environment variables:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

If either is missing the module disables itself gracefully —
no crash, just a warning.

All send attempts are logged to logs/telegram_alerts.log.
"""

from __future__ import annotations

import datetime
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Dedicated file logger for Telegram alerts
_tg_logger = logging.getLogger("apex.telegram")


def _setup_tg_logging(log_dir: str = "logs") -> None:
    """Add file handler for Telegram alert logs (idempotent)."""
    if _tg_logger.handlers:
        return
    p = Path(log_dir)
    p.mkdir(exist_ok=True)

    fh = logging.FileHandler(p / "telegram_alerts.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _tg_logger.addHandler(fh)
    _tg_logger.setLevel(logging.DEBUG)
    # Prevent propagation to root logger (avoids cp1252 encoding errors on Windows)
    _tg_logger.propagate = False


class TelegramAlerter:
    """
    Sends trade alerts to Telegram.

    Parameters
    ----------
    bot_token : str
        Telegram Bot API token.
    chat_id : str
        Telegram chat ID to send messages to.
    enabled : bool
        If False, all send calls are silently skipped.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._enabled = enabled
        self._send_count = 0

        _setup_tg_logging()

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------

    def _send_message(self, text: str) -> bool:
        """Send a message via the Telegram Bot API. Returns True on success."""
        if not self._enabled:
            return False

        url = (
            f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        )
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
            if status == 200:
                self._send_count += 1
                _tg_logger.info("SENT [%d] %s", self._send_count, _one_line(text))
                return True
            else:
                _tg_logger.error("FAIL status=%d text=%s", status, _one_line(text))
                return False
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            _tg_logger.error(
                "FAIL HTTPError %d: %s — msg=%s",
                e.code, body, _one_line(text),
            )
            return False
        except Exception as e:
            _tg_logger.error("FAIL %s: %s — msg=%s", type(e).__name__, e, _one_line(text))
            return False

    # ------------------------------------------------------------------
    # Formatted alert methods
    # ------------------------------------------------------------------

    def send_entry_alert(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop: float,
        target: float,
        size: float,
        strategy_type: str,
        ml_prob: float = 0.0,
        quality_score: float = 0.0,
        timestamp: Optional[datetime.datetime] = None,
        rr_ratio: float = 0.0,
        open_positions: int = 0,
        size_reduced: bool = False,
    ) -> bool:
        """Send a NEW TRADE entry alert."""
        ts_str = _format_ts(timestamp)
        rr_str = f"{rr_ratio:.1f}" if rr_ratio > 0 else "—"

        lines = [
            "<b>📈 NEW TRADE</b>",
            f"Symbol: <b>{symbol}</b>",
            f"Side: <b>{direction.upper()}</b>",
            f"Entry: {entry:.2f}",
            f"Stop: {stop:.2f}",
            f"Target: {target:.2f}",
            f"Size: {size:.2f}",
            f"Strategy: {strategy_type}",
            f"ML Prob: {ml_prob:.2f}",
            f"Quality: {quality_score:.2f}",
            f"R:R: {rr_str}",
            f"Open positions: {open_positions}",
        ]
        if size_reduced:
            lines.append("⚠️ Size reduced by portfolio risk")
        lines.append(f"Time: {ts_str}")

        return self._send_message("\n".join(lines))

    def send_exit_alert(
        self,
        symbol: str,
        direction: str,
        exit_price: float,
        exit_reason: str,
        net_pnl: float,
        timestamp: Optional[datetime.datetime] = None,
    ) -> bool:
        """Send a TRADE CLOSED exit alert."""
        ts_str = _format_ts(timestamp)
        pnl_sign = "+" if net_pnl >= 0 else ""
        emoji = "✅" if net_pnl >= 0 else "❌"

        lines = [
            f"<b>{emoji} TRADE CLOSED</b>",
            f"Symbol: <b>{symbol}</b>",
            f"Side: <b>{direction.upper()}</b>",
            f"Exit: {exit_price:.2f}",
            f"Reason: {exit_reason}",
            f"PnL: <b>{pnl_sign}{net_pnl:.2f}</b>",
            f"Time: {ts_str}",
        ]
        return self._send_message("\n".join(lines))

    def send_startup_alert(
        self,
        symbols: list[str],
        mode: str = "paper",
        telegram_enabled: bool = True,
    ) -> bool:
        """Send a BOT STARTED alert."""
        lines = [
            "<b>🟢 BOT STARTED</b>",
            f"Symbols: {', '.join(symbols)}",
            f"Mode: {mode}",
            f"Telegram alerts: {'enabled' if telegram_enabled else 'disabled'}",
        ]
        return self._send_message("\n".join(lines))

    def send_shutdown_alert(
        self,
        trades_sent: Optional[int] = None,
    ) -> bool:
        """Send a BOT STOPPED alert."""
        count = trades_sent if trades_sent is not None else self._send_count
        lines = [
            "<b>🔴 BOT STOPPED</b>",
            "Session complete",
            f"Alerts sent: {count}",
        ]
        return self._send_message("\n".join(lines))

    def send_test(self) -> bool:
        """Send a test message to verify credentials."""
        return self._send_message(
            "✅ Telegram alert test from Apex trading bot."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def send_count(self) -> int:
        return self._send_count


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_alerter(warn: bool = True) -> TelegramAlerter:
    """
    Create a TelegramAlerter from environment variables.

    Loads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env / env vars.
    If either is missing, returns a disabled alerter (no crash).

    Parameters
    ----------
    warn : bool
        If True, log a warning when credentials are missing.
    """
    load_dotenv()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not bot_token or not chat_id:
        missing = []
        if not bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        if warn:
            logger.warning(
                "Telegram alerts DISABLED — missing env vars: %s. "
                "Set them in .env to enable alerts.",
                ", ".join(missing),
            )
        return TelegramAlerter(bot_token="", chat_id="", enabled=False)

    logger.info("Telegram alerts enabled (chat_id=%s...)", chat_id[:6])
    return TelegramAlerter(bot_token=bot_token, chat_id=chat_id, enabled=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _format_ts(ts: Optional[datetime.datetime]) -> str:
    """Format a timestamp as ET string."""
    if ts is None:
        ts = datetime.datetime.now()
    try:
        return ts.strftime("%Y-%m-%d %H:%M ET")
    except Exception:
        return str(ts)


def _one_line(text: str) -> str:
    """Collapse multi-line text for log readability."""
    return text.replace("\n", " | ")[:200]
