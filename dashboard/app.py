"""
Dashboard FastAPI Application
===============================

Minimal web app that serves real-time trading data from DashboardState.

Endpoints:
    GET /                — HTML dashboard page
    GET /api/snapshot    — Full state snapshot (PnL, positions, trades, alerts)
    GET /api/equity      — Equity curve data (optional ?last=N)
    GET /api/health      — Liveness check

The app does NOT own any trading state.  It reads from a shared
DashboardState instance that is populated by pipeline callbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dashboard.state import DashboardState

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(state: DashboardState) -> FastAPI:
    """
    Factory that builds a FastAPI app wired to the given state.

    Parameters
    ----------
    state : DashboardState
        Shared state store populated by pipeline callbacks.

    Returns
    -------
    FastAPI
        Ready-to-run application.
    """
    app = FastAPI(title="Apex Dashboard", version="1.0.0")

    # Serve static files (CSS, JS if needed)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Store state reference for endpoint access
    app.state.dashboard = state

    logger.info("Dashboard app created")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the dashboard HTML page."""
        html_path = STATIC_DIR / "index.html"
        if not html_path.exists():
            logger.error("Dashboard HTML not found at %s", html_path)
            return HTMLResponse("<h1>Dashboard HTML not found</h1>", status_code=500)
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/snapshot")
    async def snapshot():
        """Full dashboard snapshot."""
        data = app.state.dashboard.snapshot()
        logger.debug("Snapshot requested: %d trades, %d alerts",
                      len(data["recent_trades"]), len(data["alerts"]))
        return JSONResponse(data)

    @app.get("/api/equity")
    async def equity_curve(last: Optional[int] = Query(None, ge=1)):
        """Equity curve data, optionally limited to last N points."""
        curve = app.state.dashboard.equity_curve(last_n=last)
        return JSONResponse({"equity_curve": curve, "count": len(curve)})

    @app.get("/api/health")
    async def health():
        """Liveness check."""
        return JSONResponse({
            "status": "ok",
            "trades": app.state.dashboard.trade_count(),
            "alerts": app.state.dashboard.alert_count(),
        })

    return app
