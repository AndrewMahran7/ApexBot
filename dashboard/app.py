"""
Dashboard FastAPI Application
===============================

Minimal web app that serves real-time trading data from DashboardState.

Endpoints:
    GET /                   — HTML dashboard page (strategy PnL)
    GET /exec               — Execution monitor page (operator panel)
    GET /api/snapshot       — Full strategy state snapshot
    GET /api/equity         — Equity curve data (optional ?last=N)
    GET /api/health         — Liveness check
    GET /api/exec/snapshot  — Execution layer state snapshot
    GET /api/exec/alerts    — Filtered execution alerts
    POST /api/exec/kill     — Activate kill switch
    POST /api/exec/unkill   — Reset kill switch

The app does NOT own any trading state.  It reads from a shared
DashboardState instance that is populated by pipeline callbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from dashboard.state import DashboardState
from execution.monitor import ExecutionMonitorState

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    state: DashboardState,
    exec_monitor: Optional[ExecutionMonitorState] = None,
) -> FastAPI:
    """
    Factory that builds a FastAPI app wired to the given state.

    Parameters
    ----------
    state : DashboardState
        Shared state store populated by pipeline callbacks.
    exec_monitor : ExecutionMonitorState, optional
        Execution-layer monitor state.  If provided, /exec endpoints
        are enabled.

    Returns
    -------
    FastAPI
        Ready-to-run application.
    """
    app = FastAPI(title="Apex Dashboard", version="1.0.0")

    # Serve static files (CSS, JS if needed)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Store state references for endpoint access
    app.state.dashboard = state
    app.state.exec_monitor = exec_monitor

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

    # ------------------------------------------------------------------
    # Execution monitor endpoints
    # ------------------------------------------------------------------

    @app.get("/exec", response_class=HTMLResponse)
    async def exec_monitor_page():
        """Serve the execution monitor HTML page."""
        html_path = STATIC_DIR / "execution_monitor.html"
        if not html_path.exists():
            return HTMLResponse("<h1>Execution monitor HTML not found</h1>", status_code=500)
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/exec/snapshot")
    async def exec_snapshot():
        """Full execution layer state snapshot."""
        monitor = app.state.exec_monitor
        if monitor is None:
            return JSONResponse({"error": "Execution monitor not configured"}, status_code=503)
        return JSONResponse(monitor.snapshot())

    @app.get("/api/exec/alerts")
    async def exec_alerts(level: Optional[str] = Query(None)):
        """Execution alerts, optionally filtered by level."""
        monitor = app.state.exec_monitor
        if monitor is None:
            return JSONResponse({"error": "Execution monitor not configured"}, status_code=503)
        if level:
            return JSONResponse({"alerts": monitor.alerts_by_level(level)})
        snap = monitor.snapshot()
        return JSONResponse({"alerts": snap.get("alerts", [])})

    @app.post("/api/exec/kill")
    async def exec_kill_switch(request: Request):
        """Activate the kill switch."""
        monitor = app.state.exec_monitor
        if monitor is None or not monitor.is_attached:
            return JSONResponse({"error": "Not attached to controller"}, status_code=503)
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        reason = body.get("reason", "Operator kill via dashboard")
        monitor._controller.activate_kill_switch(reason)
        logger.warning("Kill switch activated via dashboard: %s", reason)
        return JSONResponse({"status": "kill_switch_activated", "reason": reason})

    @app.post("/api/exec/unkill")
    async def exec_unkill():
        """Reset the kill switch."""
        monitor = app.state.exec_monitor
        if monitor is None or not monitor.is_attached:
            return JSONResponse({"error": "Not attached to controller"}, status_code=503)
        monitor._controller.fail_safe.reset_kill_switch()
        logger.info("Kill switch reset via dashboard")
        return JSONResponse({"status": "kill_switch_reset"})

    return app
