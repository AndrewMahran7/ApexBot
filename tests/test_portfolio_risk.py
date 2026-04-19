"""
Tests for PortfolioRiskManager
===============================

Focused tests for cross-symbol risk constraints:
  1. Max total concurrent positions
  2. Max same-direction positions
  3. Exposure cap
  4. Correlation reduction
  5. Signal ranking
  6. Position tracking lifecycle
"""

import datetime
import pytest

from strategy.orb import SignalType
from strategy.strategy_engine import LiveSignal
from risk.portfolio_risk import (
    PortfolioRiskManager,
    PortfolioRiskConfig,
)


def _make_signal(
    direction="long",
    position_size=1.0,
    ml_prob=0.6,
    strategy_type="ema50_breakout",
    position_id=None,
):
    """Create a test entry signal."""
    sig_type = (
        SignalType.LONG_ENTRY if direction == "long"
        else SignalType.SHORT_ENTRY
    )
    return LiveSignal(
        timestamp=datetime.datetime(2024, 1, 1, 10, 0, tzinfo=datetime.timezone.utc),
        direction=direction,
        signal_type=sig_type,
        entry=5000.0,
        stop=4990.0,
        take_profit=5015.0,
        position_size=position_size,
        strategy_type=strategy_type,
        position_id=position_id or f"{strategy_type}_2024-01-01",
        ml_prob=ml_prob,
    )


def _make_exit(position_id="ema50_breakout_2024-01-01"):
    """Create a test exit signal."""
    return LiveSignal(
        timestamp=datetime.datetime(2024, 1, 1, 11, 0, tzinfo=datetime.timezone.utc),
        direction="",
        signal_type=SignalType.EXIT_TP,
        entry=5015.0,
        stop=0.0,
        take_profit=0.0,
        position_size=1.0,
        strategy_type="ema50_breakout",
        position_id=position_id,
    )


class TestMaxConcurrent:
    def test_allows_within_limit(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(max_total_concurrent=3))
        sig = _make_signal(position_id="pos1")
        result = mgr.check_entry("MES", sig)
        assert result is not None

    def test_blocks_at_limit(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(max_total_concurrent=2))
        # Fill up positions
        s1 = _make_signal(position_id="pos1")
        s2 = _make_signal(direction="short", position_id="pos2")
        mgr.record_entry("MES", s1)
        mgr.record_entry("MNQ", s2)

        # Third should be blocked
        s3 = _make_signal(position_id="pos3")
        result = mgr.check_entry("RTY", s3)
        assert result is None

    def test_allows_after_exit(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(max_total_concurrent=1))
        s1 = _make_signal(position_id="pos1")
        mgr.record_entry("MES", s1)

        # Blocked
        s2 = _make_signal(position_id="pos2")
        assert mgr.check_entry("MNQ", s2) is None

        # Exit first position
        mgr.record_exit("MES", _make_exit("pos1"))

        # Now allowed
        assert mgr.check_entry("MNQ", s2) is not None


class TestMaxSameDirection:
    def test_blocks_third_long(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=2,
        ))
        mgr.record_entry("MES", _make_signal(direction="long", position_id="p1"))
        mgr.record_entry("MNQ", _make_signal(direction="long", position_id="p2"))

        result = mgr.check_entry("RTY", _make_signal(direction="long", position_id="p3"))
        assert result is None

    def test_allows_opposite_direction(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=2,
        ))
        mgr.record_entry("MES", _make_signal(direction="long", position_id="p1"))
        mgr.record_entry("MNQ", _make_signal(direction="long", position_id="p2"))

        result = mgr.check_entry("RTY", _make_signal(direction="short", position_id="p3"))
        assert result is not None


class TestExposureCap:
    def test_blocks_when_full(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=2.0,
        ))
        mgr.record_entry("MES", _make_signal(position_size=1.0, position_id="p1"))
        mgr.record_entry("MNQ", _make_signal(
            direction="short", position_size=1.0, position_id="p2",
        ))

        result = mgr.check_entry("RTY", _make_signal(position_size=1.0, position_id="p3"))
        assert result is None

    def test_caps_to_remaining(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=1.5,
        ))
        mgr.record_entry("MES", _make_signal(position_size=1.0, position_id="p1"))

        sig = _make_signal(direction="short", position_size=1.0, position_id="p2")
        result = mgr.check_entry("MNQ", sig)
        assert result is not None
        assert result.position_size == pytest.approx(0.5)


class TestCorrelationReduction:
    def test_reduces_size_when_correlated(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=5.0,
            correlation_divisor=2.0,
        ))
        mgr.record_entry("MES", _make_signal(direction="long", position_id="p1"))

        sig = _make_signal(direction="long", position_size=1.0, position_id="p2")
        result = mgr.check_entry("MNQ", sig)
        assert result is not None
        assert result.position_size == pytest.approx(0.5)

    def test_no_reduction_for_same_symbol(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=5.0,
            correlation_divisor=2.0,
        ))
        mgr.record_entry("MES", _make_signal(direction="long", position_id="p1"))

        sig = _make_signal(direction="long", position_size=1.0, position_id="p2")
        result = mgr.check_entry("MES", sig)
        # Same symbol's existing position doesn't trigger correlation reduction
        assert result is not None
        assert result.position_size == pytest.approx(1.0)

    def test_no_reduction_opposite_direction(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=5.0,
            correlation_divisor=2.0,
        ))
        mgr.record_entry("MES", _make_signal(direction="long", position_id="p1"))

        sig = _make_signal(direction="short", position_size=1.0, position_id="p2")
        result = mgr.check_entry("MNQ", sig)
        assert result is not None
        assert result.position_size == pytest.approx(1.0)


class TestSignalRanking:
    def test_ranks_by_ml_prob_descending(self):
        """With equal quality_score=0, ranking falls back to ml_prob order."""
        mgr = PortfolioRiskManager(PortfolioRiskConfig())
        signals = [
            ("MES", _make_signal(ml_prob=0.5, position_id="p1")),
            ("MNQ", _make_signal(ml_prob=0.8, position_id="p2")),
            ("RTY", _make_signal(ml_prob=0.65, position_id="p3")),
        ]
        ranked = mgr.rank_signals(signals)
        probs = [s[1].ml_prob for s in ranked]
        assert probs == [0.8, 0.65, 0.5]

    def test_composite_score_blends_ml_and_quality(self):
        """quality_score can lift a lower-ml signal above a higher-ml one."""
        mgr = PortfolioRiskManager(PortfolioRiskConfig())
        # sig_a: ml=0.6, qs=0.9 -> composite = 0.7*0.6 + 0.3*0.9 = 0.69
        sig_a = _make_signal(ml_prob=0.6, position_id="p1")
        object.__setattr__(sig_a, "quality_score", 0.9)
        # sig_b: ml=0.7, qs=0.0 -> composite = 0.7*0.7 + 0.3*0.0 = 0.49
        sig_b = _make_signal(ml_prob=0.7, position_id="p2")
        object.__setattr__(sig_b, "quality_score", 0.0)
        ranked = mgr.rank_signals([("MNQ", sig_b), ("MES", sig_a)])
        assert ranked[0][0] == "MES"  # sig_a wins with higher composite

    def test_conflict_takes_best_only(self):
        """When more signals than allowed, only the best-ranked pass."""
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=1,
            max_same_direction=1,
            max_total_exposure=5.0,
            max_exposure_per_direction=5.0,
        ))
        signals = [
            ("MES", _make_signal(ml_prob=0.5, position_id="p1")),
            ("MNQ", _make_signal(ml_prob=0.8, position_id="p2")),
            ("RTY", _make_signal(ml_prob=0.65, position_id="p3")),
        ]
        ranked = mgr.rank_signals(signals)
        approved = []
        for sym, sig in ranked:
            result = mgr.check_entry(sym, sig)
            if result is not None:
                mgr.record_entry(sym, result)
                approved.append(sym)
        # Only NQ (best ml_prob) should get through
        assert approved == ["MNQ"]
        assert mgr.open_position_count == 1


class TestPositionTracking:
    def test_entry_exit_lifecycle(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig())
        sig = _make_signal(position_id="pos1")
        mgr.record_entry("MES", sig)
        assert mgr.open_position_count == 1

        mgr.record_exit("MES", _make_exit("pos1"))
        assert mgr.open_position_count == 0

    def test_multi_symbol_tracking(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig())
        mgr.record_entry("MES", _make_signal(position_id="p1"))
        mgr.record_entry("MNQ", _make_signal(position_id="p2"))
        assert mgr.open_position_count == 2
        assert mgr.positions_for_symbol("MES") == 1
        assert mgr.positions_for_symbol("MNQ") == 1

    def test_reset_clears_all(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig())
        mgr.record_entry("MES", _make_signal(position_id="p1"))
        mgr.reset()
        assert mgr.open_position_count == 0
        assert len(mgr.events) == 0


class TestEventLogging:
    def test_block_creates_event(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(max_total_concurrent=0))
        sig = _make_signal()
        mgr.check_entry("MES", sig)
        assert len(mgr.events) == 1
        assert mgr.events[0].event_type == "portfolio_max_concurrent"

    def test_correlation_creates_event(self):
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=5.0,
            max_exposure_per_direction=5.0,
            correlation_divisor=2.0,
        ))
        mgr.record_entry("MES", _make_signal(direction="long", position_id="p1"))
        mgr.check_entry("MNQ", _make_signal(direction="long", position_id="p2"))
        corr_events = [e for e in mgr.events if e.event_type == "portfolio_corr_reduced"]
        assert len(corr_events) == 1


class TestDirectionExposureCap:
    def test_blocks_when_direction_full(self):
        """Per-direction exposure cap blocks new signals in that direction."""
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=5.0,
            max_exposure_per_direction=1.0,
            correlation_divisor=1.0,  # no correlation reduction
        ))
        mgr.record_entry("MES", _make_signal(
            direction="long", position_size=1.0, position_id="p1",
        ))
        result = mgr.check_entry("MNQ", _make_signal(
            direction="long", position_size=0.5, position_id="p2",
        ))
        assert result is None
        block_events = [e for e in mgr.events
                        if e.event_type == "portfolio_max_direction_exposure"]
        assert len(block_events) == 1

    def test_allows_opposite_direction_when_one_full(self):
        """Opposite direction still allowed even if one direction is capped."""
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=5.0,
            max_exposure_per_direction=1.0,
            correlation_divisor=1.0,
        ))
        mgr.record_entry("MES", _make_signal(
            direction="long", position_size=1.0, position_id="p1",
        ))
        result = mgr.check_entry("MNQ", _make_signal(
            direction="short", position_size=0.5, position_id="p2",
        ))
        assert result is not None

    def test_caps_size_to_direction_remaining(self):
        """Position size is reduced to fit within direction exposure budget."""
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=5.0,
            max_exposure_per_direction=1.5,
            correlation_divisor=1.0,
        ))
        mgr.record_entry("MES", _make_signal(
            direction="long", position_size=1.0, position_id="p1",
        ))
        sig = _make_signal(direction="long", position_size=1.0, position_id="p2")
        result = mgr.check_entry("MNQ", sig)
        assert result is not None
        assert result.position_size == pytest.approx(0.5)


class TestEndToEndConflict:
    def test_three_symbols_same_bar_two_pass(self):
        """Simulate 3 signals at the same bar; only 2 pass (max_same_dir=2)."""
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=3,
            max_same_direction=2,
            max_total_exposure=5.0,
            max_exposure_per_direction=5.0,
        ))
        signals = [
            ("MES", _make_signal(ml_prob=0.7, direction="long", position_id="p1")),
            ("MNQ", _make_signal(ml_prob=0.9, direction="long", position_id="p2")),
            ("RTY", _make_signal(ml_prob=0.5, direction="long", position_id="p3")),
        ]
        ranked = mgr.rank_signals(signals)
        approved = []
        for sym, sig in ranked:
            result = mgr.check_entry(sym, sig)
            if result is not None:
                mgr.record_entry(sym, result)
                approved.append(sym)
        # NQ (0.9) and MES (0.7) pass; RTY (0.5) blocked by max_same_direction
        assert len(approved) == 2
        assert "MNQ" in approved
        assert "MES" in approved
        assert "RTY" not in approved

    def test_position_size_preserved_no_increase(self):
        """Portfolio risk never increases position size beyond EMA sizing."""
        mgr = PortfolioRiskManager(PortfolioRiskConfig(
            max_total_concurrent=5,
            max_same_direction=5,
            max_total_exposure=10.0,
            max_exposure_per_direction=10.0,
            correlation_divisor=1.0,
        ))
        sig = _make_signal(position_size=1.0, position_id="p1")
        result = mgr.check_entry("MES", sig)
        assert result is not None
        assert result.position_size <= sig.position_size
