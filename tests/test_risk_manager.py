"""Tests for the risk management layer."""

import datetime

import pytest

from config.settings import InstrumentConfig
from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal
from risk.risk_manager import RiskManager, RiskConfig, RiskEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))


def _ts(hour: int, minute: int, day: int = 2) -> datetime.datetime:
    return datetime.datetime(2024, 1, day, hour, minute, tzinfo=_ET)


def _bar(ts, o, h, l, c, v=100):
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _inst() -> InstrumentConfig:
    return InstrumentConfig(
        symbol="MES", tick_size=0.25, point_value=5.0, contract_size=1,
    )


def _risk_cfg(**overrides) -> RiskConfig:
    defaults = dict(
        max_daily_loss=500.0,
        max_trades_per_day=3,
        max_concurrent_positions=2,
        max_position_size=1.0,
        max_total_exposure=2.0,
        kill_switch_close_positions=True,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _entry(
    ts=None, direction="long", entry=5000.0, stop=4990.0, tp=5015.0,
    size=1.0, strategy_type="ema50_breakout", pos_id="pos_1",
):
    sig_type = (
        SignalType.LONG_ENTRY if direction == "long" else SignalType.SHORT_ENTRY
    )
    return LiveSignal(
        timestamp=ts or _ts(9, 45),
        direction=direction,
        signal_type=sig_type,
        entry=entry,
        stop=stop,
        take_profit=tp,
        position_size=size,
        strategy_type=strategy_type,
        position_id=pos_id,
    )


def _exit(
    ts=None, exit_type=SignalType.EXIT_TP, price=5015.0,
    pos_id="pos_1", direction="long", strategy_type="ema50_breakout",
):
    return LiveSignal(
        timestamp=ts or _ts(10, 30),
        direction=direction,
        signal_type=exit_type,
        entry=price,
        stop=0.0,
        take_profit=0.0,
        position_size=1.0,
        strategy_type=strategy_type,
        position_id=pos_id,
    )


def _make_risk(on_approved=None, **cfg_overrides):
    """Create a RiskManager with optional config overrides."""
    cfg = _risk_cfg(**cfg_overrides)
    return RiskManager(cfg, _inst(), on_approved=on_approved)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_state(self):
        rm = _make_risk()
        assert rm.killed is False
        assert rm.daily_entries == 0
        assert rm.daily_realized_pnl == 0.0
        assert rm.open_position_count == 0
        assert rm.total_exposure == 0.0
        assert rm.events == []

    def test_custom_config(self):
        rm = _make_risk(max_daily_loss=200.0, max_trades_per_day=5)
        assert rm._cfg.max_daily_loss == 200.0
        assert rm._cfg.max_trades_per_day == 5


# ---------------------------------------------------------------------------
# Max trades per day
# ---------------------------------------------------------------------------

class TestMaxTradesPerDay:
    def test_allows_up_to_limit(self):
        approved = []
        rm = _make_risk(on_approved=approved.append, max_trades_per_day=2)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(pos_id="a"))
        rm.on_signal(_entry(pos_id="b"))
        assert len(approved) == 2
        assert rm.daily_entries == 2

    def test_blocks_over_limit(self):
        approved = []
        rm = _make_risk(on_approved=approved.append, max_trades_per_day=2)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(pos_id="a"))
        rm.on_signal(_entry(pos_id="b"))
        rm.on_signal(_entry(pos_id="c"))  # should be blocked
        assert len(approved) == 2
        assert rm.daily_entries == 2

        blocked = [e for e in rm.events if e.event_type == "blocked"]
        assert len(blocked) == 1
        assert blocked[0].reason == "max_trades_per_day"

    def test_resets_on_new_day(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_trades_per_day=1, max_concurrent_positions=10,
        )
        rm.on_bar(_bar(_ts(9, 30, day=2), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(ts=_ts(9, 45, day=2), pos_id="a"))
        assert rm.daily_entries == 1

        # Next day resets the counter
        rm.on_bar(_bar(_ts(9, 30, day=3), 5000, 5005, 4995, 5002))
        assert rm.daily_entries == 0
        rm.on_signal(_entry(ts=_ts(9, 45, day=3), pos_id="b"))
        assert rm.daily_entries == 1
        assert len(approved) == 2


# ---------------------------------------------------------------------------
# Max concurrent positions
# ---------------------------------------------------------------------------

class TestMaxConcurrentPositions:
    def test_allows_up_to_limit(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_concurrent_positions=2, max_trades_per_day=10,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(pos_id="a"))
        rm.on_signal(_entry(pos_id="b"))
        assert rm.open_position_count == 2

    def test_blocks_over_limit(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_concurrent_positions=2, max_trades_per_day=10,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(pos_id="a"))
        rm.on_signal(_entry(pos_id="b"))
        rm.on_signal(_entry(pos_id="c"))  # blocked
        assert rm.open_position_count == 2
        assert len(approved) == 2

        blocked = [e for e in rm.events if e.event_type == "blocked"]
        assert blocked[0].reason == "max_concurrent_positions"

    def test_exit_frees_slot(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_concurrent_positions=1, max_trades_per_day=10,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(pos_id="a"))
        assert rm.open_position_count == 1

        rm.on_signal(_exit(pos_id="a", price=5010.0))
        assert rm.open_position_count == 0

        rm.on_signal(_entry(pos_id="b"))
        assert rm.open_position_count == 1
        assert len(approved) == 3  # entry_a + exit_a + entry_b


# ---------------------------------------------------------------------------
# Position size cap
# ---------------------------------------------------------------------------

class TestPositionSizeCap:
    def test_passes_under_limit(self):
        approved = []
        rm = _make_risk(on_approved=approved.append, max_position_size=1.0)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(size=0.8))
        assert approved[0].position_size == 0.8

    def test_caps_over_limit(self):
        approved = []
        rm = _make_risk(on_approved=approved.append, max_position_size=0.5)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(size=1.0))
        assert approved[0].position_size == 0.5

        capped = [e for e in rm.events if e.event_type == "capped"]
        assert len(capped) == 1
        assert capped[0].details["original_size"] == 1.0
        assert capped[0].details["capped_size"] == 0.5


# ---------------------------------------------------------------------------
# Total exposure cap
# ---------------------------------------------------------------------------

class TestExposureCap:
    def test_blocks_when_fully_exposed(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_total_exposure=1.0,
            max_concurrent_positions=5,
            max_trades_per_day=10,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(pos_id="a", size=1.0))
        rm.on_signal(_entry(pos_id="b", size=0.5))  # blocked
        assert len(approved) == 1
        assert rm.total_exposure == 1.0

        blocked = [e for e in rm.events if e.event_type == "blocked"]
        assert blocked[0].reason == "max_total_exposure"

    def test_caps_to_remaining_exposure(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_total_exposure=1.5,
            max_position_size=2.0,
            max_concurrent_positions=5,
            max_trades_per_day=10,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(pos_id="a", size=1.0))
        rm.on_signal(_entry(pos_id="b", size=1.0))  # capped to 0.5
        assert len(approved) == 2
        assert approved[1].position_size == 0.5


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------

class TestDailyLossLimit:
    def test_blocks_after_realized_loss(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_daily_loss=100.0,
            max_trades_per_day=10,
            max_concurrent_positions=5,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        # Entry at 5000, exit at 4980 → loss = 20 pts × $5 = -$100
        rm.on_signal(_entry(pos_id="a", entry=5000.0))
        rm.on_signal(_exit(pos_id="a", price=4980.0))
        assert rm.daily_realized_pnl == -100.0

        # Next entry should be blocked
        rm.on_signal(_entry(pos_id="b", entry=5000.0))
        assert len(approved) == 2  # entry_a + exit_a, not entry_b

        blocked = [e for e in rm.events if e.event_type == "blocked"]
        assert blocked[0].reason == "daily_loss_limit"

    def test_unrealized_loss_triggers_kill_switch(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_daily_loss=100.0,
            max_trades_per_day=10,
            max_concurrent_positions=5,
            kill_switch_close_positions=False,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(pos_id="a", entry=5000.0))

        # Price drops: unrealized = (4975 - 5000) * 5 = -125
        rm.on_bar(_bar(_ts(9, 35), 4978, 4980, 4970, 4975))
        assert rm.killed is True

        ks_events = [e for e in rm.events if e.event_type == "kill_switch"]
        assert len(ks_events) == 1


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_blocks_new_entries(self):
        approved = []
        rm = _make_risk(on_approved=approved.append)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.activate_kill_switch(_ts(9, 30), "test trigger")
        assert rm.killed is True

        rm.on_signal(_entry(pos_id="a"))
        assert len(approved) == 0

        blocked = [e for e in rm.events if e.event_type == "blocked"]
        assert blocked[0].reason == "kill_switch_active"

    def test_still_forwards_exits(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append, kill_switch_close_positions=False,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(pos_id="a"))

        rm.activate_kill_switch(_ts(9, 35), "test")
        rm.on_signal(_exit(pos_id="a", price=5005.0))
        # entry + exit both forwarded
        assert len(approved) == 2
        assert approved[1].signal_type == SignalType.EXIT_TP

    def test_force_closes_positions(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_concurrent_positions=5,
            max_trades_per_day=10,
            kill_switch_close_positions=True,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(pos_id="a"))
        rm.on_signal(_entry(pos_id="b"))
        assert rm.open_position_count == 2

        rm.activate_kill_switch(_ts(9, 35), "test")
        assert rm.open_position_count == 0

        # Should have: entry_a, entry_b, exit_a, exit_b
        exit_signals = [
            s for s in approved if s.signal_type == SignalType.EXIT_EOD
        ]
        assert len(exit_signals) == 2
        for s in exit_signals:
            assert s.reason == "kill_switch_force_close"

    def test_reset_kill_switch(self):
        rm = _make_risk()
        rm.activate_kill_switch(_ts(9, 30), "test")
        assert rm.killed is True

        rm.reset_kill_switch()
        assert rm.killed is False

        reset_events = [e for e in rm.events if e.event_type == "reset"]
        assert len(reset_events) == 1

    def test_kill_switch_persists_across_days(self):
        approved = []
        rm = _make_risk(
            on_approved=approved.append, kill_switch_close_positions=False,
        )
        rm.on_bar(_bar(_ts(9, 30, day=2), 5000, 5005, 4995, 5002))
        rm.activate_kill_switch(_ts(9, 30, day=2), "test")

        # Next day
        rm.on_bar(_bar(_ts(9, 30, day=3), 5000, 5005, 4995, 5002))
        assert rm.killed is True

        rm.on_signal(_entry(ts=_ts(9, 45, day=3)))
        assert len(approved) == 0


# ---------------------------------------------------------------------------
# Exit tracking / P&L
# ---------------------------------------------------------------------------

class TestExitTracking:
    def test_long_pnl_tracked(self):
        rm = _make_risk(on_approved=lambda _: None)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(pos_id="a", entry=5000.0, direction="long"))
        rm.on_signal(_exit(pos_id="a", price=5010.0, direction="long"))
        # 10 pts * $5 = $50
        assert rm.daily_realized_pnl == 50.0

    def test_short_pnl_tracked(self):
        rm = _make_risk(on_approved=lambda _: None)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(
            pos_id="a", entry=5000.0, direction="short",
        ))
        rm.on_signal(_exit(
            pos_id="a", price=4990.0, direction="short",
        ))
        # (5000 - 4990) * 5 = $50
        assert rm.daily_realized_pnl == 50.0

    def test_exit_unknown_position_passes_through(self):
        approved = []
        rm = _make_risk(on_approved=approved.append)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        # Exit for a position risk manager doesn't track
        rm.on_signal(_exit(pos_id="unknown", price=5010.0))
        assert len(approved) == 1  # still forwarded


# ---------------------------------------------------------------------------
# Callback safety
# ---------------------------------------------------------------------------

class TestCallbackSafety:
    def test_on_approved_exception_does_not_crash(self):
        def failing(sig):
            raise RuntimeError("Downstream crash")

        rm = _make_risk(on_approved=failing)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        # Should not raise
        rm.on_signal(_entry(pos_id="a"))
        assert rm.open_position_count == 1


# ---------------------------------------------------------------------------
# Full reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_full_reset(self):
        rm = _make_risk(on_approved=lambda _: None)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(pos_id="a"))
        rm.activate_kill_switch(_ts(9, 35), "test")
        assert rm.open_position_count > 0 or rm.killed

        rm.reset()
        assert rm.killed is False
        assert rm.daily_entries == 0
        assert rm.daily_realized_pnl == 0.0
        assert rm.open_position_count == 0
        assert rm.events == []


# ---------------------------------------------------------------------------
# Risk event audit trail
# ---------------------------------------------------------------------------

class TestRiskEvents:
    def test_events_recorded_for_blocks(self):
        rm = _make_risk(on_approved=lambda _: None, max_trades_per_day=1)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(pos_id="a"))
        rm.on_signal(_entry(pos_id="b"))  # blocked

        assert len(rm.events) == 1
        ev = rm.events[0]
        assert ev.event_type == "blocked"
        assert ev.reason == "max_trades_per_day"
        assert ev.signal is not None
        assert ev.signal.position_id == "b"

    def test_events_recorded_for_caps(self):
        rm = _make_risk(on_approved=lambda _: None, max_position_size=0.3)
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(size=0.8))

        capped = [e for e in rm.events if e.event_type == "capped"]
        assert len(capped) == 1

    def test_events_recorded_for_kill_switch(self):
        rm = _make_risk(kill_switch_close_positions=False)
        rm.activate_kill_switch(_ts(9, 30), "operator decision")

        ks = [e for e in rm.events if e.event_type == "kill_switch"]
        assert len(ks) == 1
        assert "operator decision" in ks[0].reason

    def test_events_are_read_only_copy(self):
        rm = _make_risk()
        events = rm.events
        events.append(RiskEvent(
            timestamp=_ts(9, 30), event_type="fake", reason="hack",
        ))
        assert len(rm.events) == 0


# ---------------------------------------------------------------------------
# Integration: risk → paper wiring
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_risk_wired_to_paper(self):
        """Full signal flow: entry → exit through risk manager."""
        from strategy.paper_engine import PaperEngine, PaperConfig

        paper = PaperEngine(
            _inst(),
            PaperConfig(slippage_ticks=0.0, commission_per_side=0.0, max_contracts=1),
        )
        rm = _make_risk(
            on_approved=paper.on_signal,
            max_trades_per_day=10,
            max_concurrent_positions=5,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        paper.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(pos_id="a", entry=5000.0))
        assert paper.open_position_count == 1

        rm.on_signal(_exit(pos_id="a", price=5010.0))
        assert paper.open_position_count == 0
        assert paper.trade_count == 1
        assert paper.trades[0].pnl_dollars == 50.0  # 10 pts * $5

    def test_risk_blocks_do_not_reach_paper(self):
        from strategy.paper_engine import PaperEngine, PaperConfig

        paper = PaperEngine(
            _inst(),
            PaperConfig(slippage_ticks=0.0, commission_per_side=0.0, max_contracts=1),
        )
        rm = _make_risk(
            on_approved=paper.on_signal,
            max_trades_per_day=1,
            max_concurrent_positions=5,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        paper.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(pos_id="a"))
        rm.on_signal(_entry(pos_id="b"))  # blocked by risk
        assert paper.open_position_count == 1  # only first reached paper

    def test_kill_switch_force_close_reaches_paper(self):
        from strategy.paper_engine import PaperEngine, PaperConfig

        paper = PaperEngine(
            _inst(),
            PaperConfig(slippage_ticks=0.0, commission_per_side=0.0, max_contracts=1),
        )
        rm = _make_risk(
            on_approved=paper.on_signal,
            max_concurrent_positions=5,
            max_trades_per_day=10,
            kill_switch_close_positions=True,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        paper.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))

        rm.on_signal(_entry(pos_id="a", entry=5000.0))
        rm.on_signal(_entry(pos_id="b", entry=5005.0))
        assert paper.open_position_count == 2

        rm.activate_kill_switch(_ts(9, 35), "abort")
        assert paper.open_position_count == 0
        assert paper.trade_count == 2


# ---------------------------------------------------------------------------
# Force-close edge cases
# ---------------------------------------------------------------------------

class TestForceCloseEdgeCases:
    def test_force_close_before_any_bar_skips_gracefully(self):
        """Kill switch before any on_bar should NOT produce 0.0 exits."""
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_concurrent_positions=5,
            max_trades_per_day=10,
            kill_switch_close_positions=True,
        )
        # Manually inject a position without going through on_bar
        rm._open_positions["a"] = {
            "direction": "long",
            "entry_price": 5000.0,
            "position_size": 1.0,
            "entry_time": _ts(9, 30),
            "strategy_type": "ema50_breakout",
        }
        rm.activate_kill_switch(_ts(9, 31), "no bar yet")
        # Positions should still be open (force_close skipped)
        assert rm.open_position_count == 1
        # No exit signals emitted
        exits = [s for s in approved if s.signal_type == SignalType.EXIT_EOD]
        assert len(exits) == 0

    def test_force_close_updates_realized_pnl(self):
        """Force-closed positions must update daily_realized_pnl."""
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_concurrent_positions=5,
            max_trades_per_day=10,
            kill_switch_close_positions=True,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        # Enter long at 5000
        rm.on_signal(_entry(pos_id="a", entry=5000.0, direction="long"))
        assert rm.daily_realized_pnl == 0.0

        # Close price is 5002 → PnL = (5002-5000)*5 = $10
        rm.activate_kill_switch(_ts(9, 35), "test pnl tracking")
        assert rm.daily_realized_pnl == 10.0

    def test_force_close_short_pnl_correct(self):
        """Force-close short position PnL is computed correctly."""
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_concurrent_positions=5,
            max_trades_per_day=10,
            kill_switch_close_positions=True,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(
            pos_id="a", entry=5010.0, direction="short",
        ))
        # Close = 5002 → PnL = (5010-5002)*5 = $40
        rm.activate_kill_switch(_ts(9, 35), "test short")
        assert rm.daily_realized_pnl == 40.0

    def test_force_close_losing_position_updates_daily_loss(self):
        """Force-closed losing positions contribute to daily realized loss."""
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_daily_loss=200.0,
            max_concurrent_positions=5,
            max_trades_per_day=10,
            kill_switch_close_positions=True,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4990, 4995))
        rm.on_signal(_entry(pos_id="a", entry=5010.0, direction="long"))
        # Close = 4995 → PnL = (4995-5010)*5 = -$75
        rm.activate_kill_switch(_ts(9, 35), "loss test")
        assert rm.daily_realized_pnl == -75.0


# ---------------------------------------------------------------------------
# Day-change with overnight position
# ---------------------------------------------------------------------------

class TestDayChangeWithPositions:
    def test_overnight_position_exit_on_day2(self):
        """Position entered on day 1, exited on day 2 — day 2 realized
        PnL reflects the full exit PnL (not carried from day 1)."""
        rm = _make_risk(
            on_approved=lambda _: None,
            max_trades_per_day=10,
            max_concurrent_positions=5,
        )
        rm.on_bar(_bar(_ts(9, 30, day=2), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(
            ts=_ts(9, 45, day=2), pos_id="a", entry=5000.0,
        ))
        assert rm.daily_realized_pnl == 0.0

        # Day 2 ends, day 3 starts
        rm.on_bar(_bar(_ts(9, 30, day=3), 5020, 5025, 5015, 5022))
        assert rm.daily_realized_pnl == 0.0  # reset on day change
        assert rm.open_position_count == 1  # position carried over

        # Exit on day 3
        rm.on_signal(_exit(
            ts=_ts(10, 0, day=3), pos_id="a", price=5020.0,
        ))
        # Full exit PnL: (5020-5000)*5 = $100
        assert rm.daily_realized_pnl == 100.0

    def test_day_change_resets_entries_but_not_positions(self):
        """Verify daily_entries resets but open positions persist."""
        rm = _make_risk(
            on_approved=lambda _: None,
            max_trades_per_day=2,
            max_concurrent_positions=5,
        )
        rm.on_bar(_bar(_ts(9, 30, day=2), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(ts=_ts(9, 45, day=2), pos_id="a"))
        rm.on_signal(_entry(ts=_ts(9, 50, day=2), pos_id="b"))
        assert rm.daily_entries == 2
        assert rm.open_position_count == 2

        rm.on_bar(_bar(_ts(9, 30, day=3), 5010, 5015, 5005, 5012))
        assert rm.daily_entries == 0  # reset
        assert rm.open_position_count == 2  # positions persist


# ---------------------------------------------------------------------------
# Duplicate entry prevention
# ---------------------------------------------------------------------------

class TestDuplicateEntry:
    def test_duplicate_pos_id_overwrites_tracking(self):
        """If same pos_id is sent twice, second overwrites risk tracking.
        Risk manager doesn't own position lifecycle — it trusts upstream."""
        approved = []
        rm = _make_risk(
            on_approved=approved.append,
            max_trades_per_day=10,
            max_concurrent_positions=5,
        )
        rm.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        rm.on_signal(_entry(pos_id="a", entry=5000.0))
        rm.on_signal(_entry(pos_id="a", entry=5010.0))  # duplicate
        # Both forwarded (risk doesn't block duplicates — that's PaperEngine's job)
        assert len(approved) == 2
        # But internal tracking shows last entry price
        assert rm._open_positions["a"]["entry_price"] == 5010.0
