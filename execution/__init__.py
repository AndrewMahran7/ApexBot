"""
Execution layer for automated trade placement.

Modules:
    signal_schema   — Machine-readable signal contract
    validators      — Pre-trade and post-trade validation
    adapter         — Abstract execution adapter interface
    openclaw_adapter— OpenClaw/Tradovate UI adapter implementation
    openclaw_driver — Low-level Tradovate UI interaction via OpenClaw
    execution_controller — Orchestrates signal → validation → execution → confirmation
    risk_bridge     — Bridges PropRiskLayer sizing into execution signals
    fail_safes      — Kill switch, cooldown, duplicate suppression, dry-run
    audit_logger    — Structured JSON logging and screenshot capture
    prop_sizing     — Mode-aware trade gating and contract sizing via PropRiskLayer
    reconciliation  — Continuous position reconciliation loop (read-only monitoring)
"""
