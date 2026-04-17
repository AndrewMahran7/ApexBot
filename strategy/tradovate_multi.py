"""
Multi-Symbol Tradovate SIM Adapter
====================================

Wraps one TradovateClient per symbol, sharing a single auth session,
so MES + MNQ can execute independently on Tradovate SIM.

The adapter exposes a ``route_signal(symbol, signal)`` method that
matches the MultiSymbolRouter execution flow, replacing PaperEngine.

Safety:
  - sim_only=True is ALWAYS enforced.
  - Refuses to proceed if environment != "demo".
  - Dry-run mode validates auth + contract resolution without placing orders.
"""

from __future__ import annotations

import csv
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from config.settings import INSTRUMENT_REGISTRY, InstrumentConfig
from strategy.tradovate_client import (
    TradovateClient,
    TradovateConfig,
    TrackedOrder,
    TrackedPosition,
)
from strategy.strategy_engine import LiveSignal

logger = logging.getLogger(__name__)

# Dedicated sim logger — writes to logs/tradovate_sim.log
sim_logger = logging.getLogger("apex.tradovate_sim")

# ------------------------------------------------------------------
# Symbol → Tradovate contract mapping
# ------------------------------------------------------------------

# Tradovate resolves front-month via ".c.0" suffix
SYMBOL_TO_TRADOVATE = {
    "MES": "MESM5",   # update to current front-month contract
    "MNQ": "MNQM5",
    "RTY": "M2KM5",
}


def _setup_sim_logging(log_dir: str = "logs") -> None:
    """Add file handler for Tradovate SIM logs if not already present."""
    if sim_logger.handlers:
        return
    p = Path(log_dir)
    p.mkdir(exist_ok=True)

    fh = logging.FileHandler(p / "tradovate_sim.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    sim_logger.addHandler(fh)
    sim_logger.setLevel(logging.DEBUG)


def load_tradovate_config() -> TradovateConfig:
    """
    Load Tradovate SIM credentials from environment / .env file.

    Required env vars:
        TRADOVATE_USERNAME, TRADOVATE_PASSWORD, TRADOVATE_CID, TRADOVATE_SECRET

    Optional:
        TRADOVATE_APP_ID  (default "Apex")
        TRADOVATE_DEVICE_ID  (auto-generated if missing)
    """
    load_dotenv()

    username = os.getenv("TRADOVATE_USERNAME", "")
    password = os.getenv("TRADOVATE_PASSWORD", "")
    cid = os.getenv("TRADOVATE_CID", "")
    secret = os.getenv("TRADOVATE_SECRET", "")
    app_id = os.getenv("TRADOVATE_APP_ID", "Apex")
    device_id = os.getenv("TRADOVATE_DEVICE_ID", str(uuid.uuid4()))

    missing = []
    if not username:
        missing.append("TRADOVATE_USERNAME")
    if not password:
        missing.append("TRADOVATE_PASSWORD")
    if not cid:
        missing.append("TRADOVATE_CID")
    if not secret:
        missing.append("TRADOVATE_SECRET")

    if missing:
        raise RuntimeError(
            f"Missing Tradovate credentials in .env: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your values."
        )

    return TradovateConfig(
        username=username,
        password=password,
        cid=cid,
        secret=secret,
        app_id=app_id,
        device_id=device_id,
        environment="demo",
        sim_only=True,
    )


class MultiSymbolTradovateAdapter:
    """
    Manages one TradovateClient per symbol, sharing auth credentials.

    Parameters
    ----------
    symbols : list[str]
        Symbols to connect (e.g. ["MES", "MNQ"]).
    config : TradovateConfig
        Shared credentials — environment is forced to "demo".
    contract_overrides : dict[str, str], optional
        Override the Tradovate contract name per symbol.
    """

    def __init__(
        self,
        symbols: list[str],
        config: TradovateConfig,
        contract_overrides: Optional[dict[str, str]] = None,
    ) -> None:
        # HARD SAFETY: force SIM
        config.environment = "demo"
        config.sim_only = True

        self._symbols = symbols
        self._config = config
        self._contract_overrides = contract_overrides or {}
        self._clients: dict[str, TradovateClient] = {}
        self._connected = False
        self._trade_log: list[dict] = []

        _setup_sim_logging()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Authenticate and resolve contracts for all symbols."""
        sim_logger.info("=" * 60)
        sim_logger.info("  TRADOVATE SIM — CONNECTING")
        sim_logger.info("  Symbols: %s", self._symbols)
        sim_logger.info("  Environment: %s (sim_only=True)", self._config.environment)
        sim_logger.info("=" * 60)

        for sym in self._symbols:
            instrument = INSTRUMENT_REGISTRY.get(sym)
            if instrument is None:
                raise RuntimeError(f"Unknown symbol '{sym}' — not in INSTRUMENT_REGISTRY")

            # Create a per-symbol instrument with the Tradovate contract name
            tov_name = self._contract_overrides.get(
                sym, SYMBOL_TO_TRADOVATE.get(sym, sym),
            )
            sym_instrument = InstrumentConfig(
                symbol=tov_name,
                tick_size=instrument.tick_size,
                point_value=instrument.point_value,
            )

            def _on_fill(pos_id, fill_px, fill_qty, _sym=sym):
                sim_logger.info(
                    "FILL %s pos=%s px=%.2f qty=%d",
                    _sym, pos_id, fill_px, fill_qty,
                )

            def _on_reject(pos_id, reason, _sym=sym):
                sim_logger.warning(
                    "REJECT %s pos=%s reason=%s",
                    _sym, pos_id, reason,
                )

            client = TradovateClient(
                instrument=sym_instrument,
                config=self._config,
                on_fill=_on_fill,
                on_reject=_on_reject,
            )

            sim_logger.info("Connecting %s (contract=%s)...", sym, tov_name)
            client.connect()
            client.start_sync()
            self._clients[sym] = client
            sim_logger.info(
                "Connected %s — contract_id=%s, account=%s",
                sym, client._contract_id, self._config.account_id,
            )

        self._connected = True
        sim_logger.info("All symbols connected to Tradovate SIM")

    def disconnect(self) -> None:
        """Disconnect all symbol clients."""
        for sym, client in self._clients.items():
            sim_logger.info("Disconnecting %s...", sym)
            client.disconnect()
        self._connected = False
        sim_logger.info("All symbols disconnected")

    # ------------------------------------------------------------------
    # Dry-run validation (no orders placed)
    # ------------------------------------------------------------------

    def dry_run(self) -> dict:
        """
        Validate connectivity without placing orders.

        Returns a dict with {symbol: {contract_id, account_id, status}}.
        """
        sim_logger.info("=" * 60)
        sim_logger.info("  DRY RUN — validating Tradovate SIM connectivity")
        sim_logger.info("=" * 60)

        results: dict[str, dict] = {}
        all_ok = True

        for sym in self._symbols:
            instrument = INSTRUMENT_REGISTRY.get(sym)
            if instrument is None:
                results[sym] = {"status": "FAIL", "error": "Unknown symbol"}
                all_ok = False
                continue

            tov_name = self._contract_overrides.get(
                sym, SYMBOL_TO_TRADOVATE.get(sym, sym),
            )
            sym_instrument = InstrumentConfig(
                symbol=tov_name,
                tick_size=instrument.tick_size,
                point_value=instrument.point_value,
            )

            client = TradovateClient(
                instrument=sym_instrument,
                config=self._config,
            )

            try:
                client.connect()
                results[sym] = {
                    "status": "OK",
                    "contract_name": tov_name,
                    "contract_id": client._contract_id,
                    "account_id": self._config.account_id,
                }
                sim_logger.info(
                    "DRY RUN %s: OK (contract=%s, id=%s, account=%s)",
                    sym, tov_name, client._contract_id, self._config.account_id,
                )
                client.disconnect()
            except Exception as e:
                results[sym] = {"status": "FAIL", "error": str(e)}
                all_ok = False
                sim_logger.error("DRY RUN %s: FAIL — %s", sym, e)

        results["_all_ok"] = all_ok
        sim_logger.info("Dry run result: %s", "ALL OK" if all_ok else "FAILURES DETECTED")
        return results

    # ------------------------------------------------------------------
    # Signal routing (replaces paper.on_signal in the pipeline)
    # ------------------------------------------------------------------

    def route_signal(self, symbol: str, signal: LiveSignal) -> None:
        """
        Route a signal to the correct symbol's TradovateClient.

        This is the replacement for paper.on_signal() in the
        MultiSymbolRouter execution tracker.
        """
        if not self._connected:
            sim_logger.error("Cannot route signal — not connected")
            return

        client = self._clients.get(symbol)
        if client is None:
            sim_logger.error("No client for symbol '%s'", symbol)
            return

        action = "entry" if signal.is_entry else "exit"
        sim_logger.info(
            "SIGNAL %s %s %s dir=%s px=%.2f sl=%.2f tp=%.2f prob=%.3f",
            symbol, action, signal.strategy_type,
            signal.direction, signal.entry,
            signal.stop, signal.take_profit,
            getattr(signal, "ml_prob", 0.0),
        )

        # Log to trade record
        self._trade_log.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "direction": signal.direction,
            "strategy_type": signal.strategy_type,
            "entry_price": signal.entry,
            "stop_loss": signal.stop,
            "take_profit": signal.take_profit,
            "ml_prob": getattr(signal, "ml_prob", 0.0),
        })

        client.on_signal(signal)

    # ------------------------------------------------------------------
    # Liquidation
    # ------------------------------------------------------------------

    def liquidate_all(self) -> None:
        """Emergency liquidation across all symbols."""
        sim_logger.warning("EMERGENCY LIQUIDATION — all symbols")
        for sym, client in self._clients.items():
            sim_logger.warning("Liquidating %s...", sym)
            client.liquidate_all()
        sim_logger.warning("Liquidation complete")

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def open_positions(self) -> dict[str, dict[str, TrackedPosition]]:
        """Return {symbol: {pos_id: TrackedPosition}}."""
        return {sym: c.open_positions for sym, c in self._clients.items()}

    def working_orders(self) -> dict[str, list[TrackedOrder]]:
        """Return {symbol: [working orders]}."""
        return {sym: c.working_orders() for sym, c in self._clients.items()}

    def sync_fills(self) -> dict[str, list]:
        """Sync fills for all symbols, return {symbol: [fills]}."""
        return {sym: c.sync_fills() for sym, c in self._clients.items()}

    # ------------------------------------------------------------------
    # Trade log export
    # ------------------------------------------------------------------

    def export_trade_log(self, path: str = "results/tradovate_sim_trades.csv") -> None:
        """Write the signal/trade log to CSV."""
        if not self._trade_log:
            sim_logger.info("No trades to export")
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self._trade_log[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._trade_log)

        sim_logger.info("Exported %d trade records to %s", len(self._trade_log), path)
