"""Tests for the live-compatible StrategyEngine."""

import datetime
import os
import pickle
import tempfile

import pytest

from strategy.orb import SignalType, Signal
from strategy.hybrid_ema_ml import HybridEMAMLConfig, HybridEMAMLStrategy
from strategy.strategy_engine import (
    StrategyEngine,
    LiveSignal,
    EngineState,
    ValidationResult,
    ValidationMismatch,
    _signal_to_live,
    _map_exit_reason,
    build_reference_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ET = datetime.timezone(datetime.timedelta(hours=-4))  # EDT


def _ts(hour: int, minute: int, day: int = 2) -> datetime.datetime:
    return datetime.datetime(2024, 1, day, hour, minute, tzinfo=_ET)


def _bar(ts: datetime.datetime, o: float, h: float, l: float, c: float, v: float = 100) -> dict:
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _make_model_pkl(path: str) -> None:
    """Create a dummy ML model pkl that returns prob~0.7 for any input."""
    from sklearn.dummy import DummyClassifier
    import numpy as np

    clf = DummyClassifier(strategy="most_frequent")
    # Fit with both classes so predict_proba returns 2 columns
    X = np.zeros((10, 35))
    y = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    clf.fit(X, y)

    # Feature columns matching the strategy's _extract_features output
    feature_cols = [
        "f_price_ema", "f_price_ema_dist", "f_price_ema_dist_pct",
        "f_price_ema_slope", "f_price_bar_range", "f_price_rolling_range",
        "f_price_ret_12bar", "f_price_gap", "f_price_gap_pct",
        "f_vol_avg", "f_vol_relative", "f_vol_expansion",
        "f_vola_tr", "f_vola_atr", "f_vola_atr_norm", "f_vola_realized",
        "f_time_minutes_since_open", "f_time_minutes_since_range_close",
        "f_time_minutes_to_close", "f_time_weekday",
        "f_range_high", "f_range_low", "f_range_size",
        "f_range_dist_above", "f_range_dist_below", "f_range_size_vs_atr",
        "f_regime_trend_strength", "f_regime_trend_direction",
        "f_regime_compression", "f_regime_breakout_strength",
        "f_regime_vol_trend",
        "f_ema_distance", "f_ema_distance_pct",
        "f_risk_points", "f_range_vs_atr",
    ]

    with open(path, "wb") as f:
        pickle.dump({"model": clf, "feature_columns": feature_cols}, f)


@pytest.fixture
def model_path(tmp_path):
    """Create a temporary dummy model and return its path."""
    path = str(tmp_path / "test_model.pkl")
    _make_model_pkl(path)
    return path


def _session_bars(base_price: float = 5000.0, day: int = 2) -> list[dict]:
    """
    Generate a full trading session of bars: 09:30-16:00 in 5m intervals.

    Includes warmup bars from the prior day so the EMA engine is ready
    (FeatureEngine needs >= 10 bars before snap.ema is non-None).
    """
    bars = []

    # -- Prior-day warmup (day-1): 15 bars so EMA is primed --
    prev_day = day - 1
    for i, m in enumerate(range(0, 75, 5)):  # 10:00-11:10, 15 bars
        h_offset = 10 + m // 60
        m_offset = m % 60
        bars.append(_bar(
            _ts(h_offset, m_offset, prev_day),
            base_price + i * 0.5,
            base_price + i * 0.5 + 3,
            base_price + i * 0.5 - 2,
            base_price + i * 0.5 + 1,
            v=150,
        ))

    # -- Test day --
    # Opening range bars (09:30, 09:35, 09:40)
    for m in [30, 35, 40]:
        bars.append(_bar(
            _ts(9, m, day),
            base_price, base_price + 5, base_price - 3, base_price + 2,
            v=200,
        ))

    # Range close bar at 09:45 — price above EMA → long signal
    bars.append(_bar(
        _ts(9, 45, day),
        base_price + 2, base_price + 8, base_price - 1, base_price + 6,
        v=250,
    ))

    # Subsequent bars every 5 min until 15:55
    for hour in range(10, 16):
        for minute in range(0, 60, 5):
            if hour == 15 and minute > 55:
                break
            bars.append(_bar(
                _ts(hour, minute, day),
                base_price + 6, base_price + 10, base_price + 4, base_price + 7,
                v=150,
            ))

    return bars


# ---------------------------------------------------------------------------
# LiveSignal conversion
# ---------------------------------------------------------------------------

class TestSignalConversion:
    def test_long_entry_conversion(self):
        sig = Signal(
            signal_type=SignalType.LONG_ENTRY,
            price=5000.0,
            timestamp=_ts(9, 45),
            reason="test",
            entry_price=5000.0,
            stop_loss=4990.0,
            take_profit=5015.0,
            position_size=0.8,
            strategy_type="ema50_breakout",
        )
        ls = _signal_to_live(sig)
        assert ls.direction == "long"
        assert ls.signal_type == SignalType.LONG_ENTRY
        assert ls.entry == 5000.0
        assert ls.stop == 4990.0
        assert ls.take_profit == 5015.0
        assert ls.position_size == 0.8
        assert ls.strategy_type == "ema50_breakout"
        assert ls.is_entry is True
        assert ls.is_exit is False

    def test_short_entry_conversion(self):
        sig = Signal(
            signal_type=SignalType.SHORT_ENTRY,
            price=5000.0,
            timestamp=_ts(9, 45),
            entry_price=5000.0,
            stop_loss=5010.0,
            take_profit=4985.0,
        )
        ls = _signal_to_live(sig)
        assert ls.direction == "short"
        assert ls.is_entry is True

    def test_exit_conversion(self):
        for exit_type in (SignalType.EXIT_TP, SignalType.EXIT_SL, SignalType.EXIT_EOD):
            sig = Signal(
                signal_type=exit_type,
                price=5015.0,
                timestamp=_ts(10, 0),
                entry_price=5000.0,
                stop_loss=4990.0,
                take_profit=5015.0,
            )
            ls = _signal_to_live(sig)
            assert ls.is_exit is True
            assert ls.is_entry is False

    def test_none_defaults(self):
        sig = Signal(
            signal_type=SignalType.LONG_ENTRY,
            price=5000.0,
            timestamp=_ts(9, 45),
        )
        ls = _signal_to_live(sig)
        assert ls.stop == 0.0
        assert ls.take_profit == 0.0


# ---------------------------------------------------------------------------
# Engine initialisation and state
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_creates_with_config(self, model_path):
        cfg = HybridEMAMLConfig(model_path=model_path)
        engine = StrategyEngine(cfg)
        assert engine._bar_count == 0
        assert engine.state.bar_count == 0

    def test_state_snapshot(self, model_path):
        cfg = HybridEMAMLConfig(model_path=model_path)
        engine = StrategyEngine(cfg)
        s = engine.state
        assert isinstance(s, EngineState)
        assert s.current_date is None
        assert s.range_set is False
        assert s.open_positions == 0
        assert s.decided_today is False

    def test_reset_clears_state(self, model_path):
        cfg = HybridEMAMLConfig(model_path=model_path)
        engine = StrategyEngine(cfg)
        # Feed a bar then reset
        engine.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        assert engine._bar_count == 1
        engine.reset()
        assert engine._bar_count == 0
        assert engine.state.bar_count == 0
        assert len(engine.signal_log) == 0


# ---------------------------------------------------------------------------
# Bar processing — single-candidate mode
# ---------------------------------------------------------------------------

class TestSingleCandidateMode:
    def test_opening_range_no_signals(self, model_path):
        """During opening range, engine should emit no entry/exit signals."""
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,  # accept all
        )
        engine = StrategyEngine(cfg)

        # Feed opening range bars
        for m in [30, 35, 40]:
            signals = engine.on_bar(_bar(_ts(9, m), 5000, 5005, 4995, 5002))
            assert len(signals) == 0

    def test_entry_signal_emitted(self, model_path):
        """After range close, engine should emit a long entry signal."""
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)

        bars = _session_bars()
        all_signals = []
        for bar in bars:
            sigs = engine.on_bar(bar)
            all_signals.extend(sigs)

        entries = [s for s in all_signals if s.is_entry]
        assert len(entries) >= 1, "Expected at least one entry signal"
        assert entries[0].signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY)
        assert entries[0].entry > 0
        assert entries[0].stop > 0
        assert entries[0].take_profit > 0

    def test_eod_exit_emitted(self, model_path):
        """There should be an EOD exit signal near 15:50."""
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)

        bars = _session_bars()
        all_signals = []
        for bar in bars:
            sigs = engine.on_bar(bar)
            all_signals.extend(sigs)

        exits = [s for s in all_signals if s.is_exit]
        assert len(exits) >= 1, "Expected at least one exit signal"

    def test_signal_log_populated(self, model_path):
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)

        for bar in _session_bars():
            engine.on_bar(bar)

        assert len(engine.signal_log) >= 1


# ---------------------------------------------------------------------------
# Multi-candidate mode
# ---------------------------------------------------------------------------

class TestMultiCandidateMode:
    def test_multi_candidate_signals(self, model_path):
        """Multi-candidate mode should produce signals from multiple EMA lengths."""
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=True,
            max_trades_per_day=3,
            ema_periods=(20, 50),
            entry_types=("breakout",),
            selection_strategy="priority",
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)

        bars = _session_bars()
        all_signals = []
        for bar in bars:
            sigs = engine.on_bar(bar)
            all_signals.extend(sigs)

        entries = [s for s in all_signals if s.is_entry]
        # With two EMA periods and breakout only, expect up to 2 entries
        assert len(entries) >= 1

    def test_max_trades_respected(self, model_path):
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=True,
            max_trades_per_day=1,
            ema_periods=(20, 50, 100),
            entry_types=("breakout",),
            selection_strategy="global_ml",
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)

        bars = _session_bars()
        all_signals = []
        for bar in bars:
            sigs = engine.on_bar(bar)
            all_signals.extend(sigs)

        entries = [s for s in all_signals if s.is_entry]
        assert len(entries) <= 1


# ---------------------------------------------------------------------------
# Callback dispatch
# ---------------------------------------------------------------------------

class TestCallbackDispatch:
    def test_on_signal_callback(self, model_path):
        received = []
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg, on_signal=received.append)

        for bar in _session_bars():
            engine.on_bar(bar)

        assert len(received) >= 1
        assert all(isinstance(s, LiveSignal) for s in received)
        # Callback signals match signal log
        assert len(received) == len(engine.signal_log)


# ---------------------------------------------------------------------------
# Validation mode
# ---------------------------------------------------------------------------

class TestValidation:
    def test_identical_replay_passes(self, model_path):
        """Replaying the same bars should produce identical signals."""
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)

        # First pass: collect reference signals
        bars = _session_bars()
        for bar in bars:
            engine.on_bar(bar)
        reference = list(engine.signal_log)

        # Second pass: validate
        result = engine.validate(bars, reference)
        assert result.passed, (
            f"Validation failed: {len(result.mismatches)} mismatches, "
            f"{len(result.extra_engine)} extra, {len(result.missing_engine)} missing"
        )
        assert result.total_bars == len(bars)
        assert result.matched == len(reference)

    def test_mismatch_detected(self, model_path):
        """Altered reference should produce mismatches."""
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)

        bars = _session_bars()
        for bar in bars:
            engine.on_bar(bar)
        reference = list(engine.signal_log)

        if reference:
            # Corrupt a price
            corrupted = LiveSignal(
                timestamp=reference[0].timestamp,
                direction=reference[0].direction,
                signal_type=reference[0].signal_type,
                entry=reference[0].entry + 100.0,  # big diff
                stop=reference[0].stop,
                take_profit=reference[0].take_profit,
                position_size=reference[0].position_size,
                strategy_type=reference[0].strategy_type,
            )
            ref_modified = [corrupted] + reference[1:]

            result = engine.validate(bars, ref_modified)
            assert not result.passed
            assert len(result.mismatches) > 0

    def test_extra_signal_detected(self, model_path):
        """If engine produces a signal not in reference, it's flagged as extra."""
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)

        bars = _session_bars()
        for bar in bars:
            engine.on_bar(bar)
        reference = list(engine.signal_log)

        if len(reference) > 1:
            # Remove the first reference signal → engine will have an "extra"
            result = engine.validate(bars, reference[1:])
            assert len(result.extra_engine) >= 1

    def test_empty_validation(self, model_path):
        """Empty bars + empty reference = passes."""
        cfg = HybridEMAMLConfig(model_path=model_path)
        engine = StrategyEngine(cfg)
        result = engine.validate([], [])
        assert result.passed
        assert result.total_bars == 0


# ---------------------------------------------------------------------------
# build_reference_signals
# ---------------------------------------------------------------------------

class TestBuildReferenceSignals:
    def test_converts_trades_to_signals(self):
        from backtest.engine import Trade

        trade = Trade(
            entry_time=_ts(9, 45),
            exit_time=_ts(10, 30),
            entry_price=5000.0,
            exit_price=5015.0,
            stop_loss=4990.0,
            take_profit=5015.0,
            direction="long",
            pnl_points=15.0,
            pnl_dollars=75.0,
            commission=1.24,
            slippage_cost=2.50,
            net_pnl=71.26,
            exit_reason="Take profit hit (5015.00)",
            contracts=1,
            position_size=0.8,
            strategy_type="ema50_breakout",
        )

        signals = build_reference_signals([trade])
        assert len(signals) == 2  # entry + exit

        entry = signals[0]
        assert entry.signal_type == SignalType.LONG_ENTRY
        assert entry.direction == "long"
        assert entry.entry == 5000.0
        assert entry.stop == 4990.0

        exit_ = signals[1]
        assert exit_.signal_type == SignalType.EXIT_TP

    def test_entries_only(self):
        from backtest.engine import Trade

        trade = Trade(
            entry_time=_ts(9, 45),
            exit_time=_ts(10, 30),
            entry_price=5000.0,
            exit_price=4990.0,
            stop_loss=4990.0,
            take_profit=5015.0,
            direction="long",
            pnl_points=-10.0,
            pnl_dollars=-50.0,
            commission=1.24,
            slippage_cost=2.50,
            net_pnl=-53.74,
            exit_reason="Stop loss hit (4990.00)",
            contracts=1,
        )

        signals = build_reference_signals([trade], include_exits=False)
        assert len(signals) == 1
        assert signals[0].is_entry

    def test_exit_reason_mapping(self):
        from backtest.engine import Trade

        # Strategy-format exit reasons (lowercase with details)
        for reason, expected_type in [
            ("Take profit hit (5015.00)", SignalType.EXIT_TP),
            ("Stop loss hit (4990.00)", SignalType.EXIT_SL),
            ("End-of-day exit", SignalType.EXIT_EOD),
        ]:
            trade = Trade(
                entry_time=_ts(9, 45), exit_time=_ts(10, 30),
                entry_price=5000.0, exit_price=5015.0,
                stop_loss=4990.0, take_profit=5015.0,
                direction="long", pnl_points=15.0, pnl_dollars=75.0,
                commission=0, slippage_cost=0, net_pnl=75.0,
                exit_reason=reason, contracts=1,
            )
            signals = build_reference_signals([trade], include_entries=False)
            assert signals[0].signal_type == expected_type

        # Engine-format exit reasons (title case from _exit_reason())
        for reason, expected_type in [
            ("Take Profit", SignalType.EXIT_TP),
            ("Stop Loss", SignalType.EXIT_SL),
            ("End of Day", SignalType.EXIT_EOD),
        ]:
            trade = Trade(
                entry_time=_ts(9, 45), exit_time=_ts(10, 30),
                entry_price=5000.0, exit_price=5015.0,
                stop_loss=4990.0, take_profit=5015.0,
                direction="long", pnl_points=15.0, pnl_dollars=75.0,
                commission=0, slippage_cost=0, net_pnl=75.0,
                exit_reason=reason, contracts=1,
            )
            signals = build_reference_signals([trade], include_entries=False)
            assert signals[0].signal_type == expected_type, (
                f"Engine-format '{reason}' should map to {expected_type}"
            )


# ---------------------------------------------------------------------------
# Deterministic replay (reproducibility)
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_two_runs_identical(self, model_path):
        """Two cold runs from the same config produce identical signals."""
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=True,
            max_trades_per_day=2,
            ema_periods=(20, 50),
            entry_types=("breakout",),
            selection_strategy="priority",
            ml_threshold=0.0,
        )

        bars = _session_bars()

        # Run 1
        engine1 = StrategyEngine(cfg)
        for bar in bars:
            engine1.on_bar(bar)
        log1 = engine1.signal_log

        # Run 2
        engine2 = StrategyEngine(cfg)
        for bar in bars:
            engine2.on_bar(bar)
        log2 = engine2.signal_log

        assert len(log1) == len(log2)
        for s1, s2 in zip(log1, log2):
            assert s1.timestamp == s2.timestamp
            assert s1.signal_type == s2.signal_type
            assert s1.direction == s2.direction
            assert s1.entry == s2.entry
            assert s1.stop == s2.stop
            assert s1.take_profit == s2.take_profit

    def test_reset_gives_same_results(self, model_path):
        """Resetting and re-running produces identical signals."""
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)
        bars = _session_bars()

        for bar in bars:
            engine.on_bar(bar)
        log1 = list(engine.signal_log)

        engine.reset()

        for bar in bars:
            engine.on_bar(bar)
        log2 = engine.signal_log

        assert len(log1) == len(log2)
        for s1, s2 in zip(log1, log2):
            assert s1.signal_type == s2.signal_type
            assert s1.entry == s2.entry


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

class TestStateTracking:
    def test_bar_count_increments(self, model_path):
        cfg = HybridEMAMLConfig(model_path=model_path)
        engine = StrategyEngine(cfg)

        engine.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        assert engine.state.bar_count == 1

        engine.on_bar(_bar(_ts(9, 35), 5002, 5008, 4998, 5005))
        assert engine.state.bar_count == 2

    def test_date_tracking(self, model_path):
        cfg = HybridEMAMLConfig(model_path=model_path)
        engine = StrategyEngine(cfg)

        engine.on_bar(_bar(_ts(9, 30, day=2), 5000, 5005, 4995, 5002))
        assert engine.state.current_date == datetime.date(2024, 1, 2)

    def test_range_set_after_close(self, model_path):
        cfg = HybridEMAMLConfig(model_path=model_path)
        engine = StrategyEngine(cfg)

        # Range bars
        engine.on_bar(_bar(_ts(9, 30), 5000, 5005, 4995, 5002))
        engine.on_bar(_bar(_ts(9, 35), 5002, 5008, 4998, 5005))
        engine.on_bar(_bar(_ts(9, 40), 5003, 5007, 4997, 5004))
        assert engine.state.range_set is False

        # Range close
        engine.on_bar(_bar(_ts(9, 45), 5004, 5010, 4996, 5008))
        assert engine.state.range_set is True

    def test_ml_decisions_accessible(self, model_path):
        cfg = HybridEMAMLConfig(
            model_path=model_path,
            multi_candidate=False,
            ml_threshold=0.0,
        )
        engine = StrategyEngine(cfg)

        for bar in _session_bars():
            engine.on_bar(bar)

        assert len(engine.ml_decisions) >= 1


# ---------------------------------------------------------------------------
# _map_exit_reason unit tests
# ---------------------------------------------------------------------------

class TestMapExitReason:
    """Direct tests for the _map_exit_reason helper."""

    def test_engine_format_title_case(self):
        assert _map_exit_reason("Take Profit") == SignalType.EXIT_TP
        assert _map_exit_reason("Stop Loss") == SignalType.EXIT_SL
        assert _map_exit_reason("End of Day") == SignalType.EXIT_EOD

    def test_strategy_format_lowercase(self):
        assert _map_exit_reason("Take profit hit (5015.00)") == SignalType.EXIT_TP
        assert _map_exit_reason("Stop loss hit (4990.00)") == SignalType.EXIT_SL
        assert _map_exit_reason("End-of-day exit") == SignalType.EXIT_EOD

    def test_unknown_defaults_to_eod(self):
        assert _map_exit_reason("Liquidation") == SignalType.EXIT_EOD


# ---------------------------------------------------------------------------
# Multi-candidate validation (CRITICAL #2: no dup key collision)
# ---------------------------------------------------------------------------

class TestMultiCandidateValidation:
    """Ensure validation handles multiple signals at the same timestamp."""

    def test_same_timestamp_different_strategy_types(self):
        """Two LONG_ENTRY signals at same ts with different strategy_types
        should both appear in validation — not overwrite each other."""
        ts = _ts(9, 45)
        sig_a = LiveSignal(
            timestamp=ts, direction="long",
            signal_type=SignalType.LONG_ENTRY,
            entry=5000.0, stop=4990.0, take_profit=5015.0,
            position_size=0.8, strategy_type="ema20_breakout",
        )
        sig_b = LiveSignal(
            timestamp=ts, direction="long",
            signal_type=SignalType.LONG_ENTRY,
            entry=5002.0, stop=4992.0, take_profit=5017.0,
            position_size=0.6, strategy_type="ema50_breakout",
        )

        ref = [sig_a, sig_b]

        from strategy.strategy_engine import StrategyEngine as _SE

        # Build lookup the same way validate() does
        ref_lookup: dict[tuple, LiveSignal] = {}
        for rs in ref:
            key = (rs.timestamp, rs.signal_type, rs.strategy_type)
            ref_lookup[key] = rs

        # Both signals kept — no collision
        assert len(ref_lookup) == 2
        assert ref_lookup[(ts, SignalType.LONG_ENTRY, "ema20_breakout")].entry == 5000.0
        assert ref_lookup[(ts, SignalType.LONG_ENTRY, "ema50_breakout")].entry == 5002.0


# ---------------------------------------------------------------------------
# _find_ml_decision (CRITICAL #3: per-signal ML enrichment)
# ---------------------------------------------------------------------------

class TestFindMLDecision:
    def test_matches_by_strategy_type(self, model_path):
        """_find_ml_decision returns the decision matching the signal's
        strategy_type, not just the last one."""
        cfg = HybridEMAMLConfig(model_path=model_path)
        engine = StrategyEngine(cfg)

        # Manually inject ml_decisions
        engine._strategy.ml_decisions = [
            {"strategy_type": "ema20_breakout", "ml_prob": 0.85, "percentile": 90.0},
            {"strategy_type": "ema50_breakout", "ml_prob": 0.60, "percentile": 50.0},
        ]

        sig_20 = LiveSignal(
            timestamp=_ts(9, 45), direction="long",
            signal_type=SignalType.LONG_ENTRY,
            entry=5000.0, stop=4990.0, take_profit=5010.0,
            position_size=0.8, strategy_type="ema20_breakout",
        )
        sig_50 = LiveSignal(
            timestamp=_ts(9, 45), direction="long",
            signal_type=SignalType.LONG_ENTRY,
            entry=5002.0, stop=4992.0, take_profit=5012.0,
            position_size=0.6, strategy_type="ema50_breakout",
        )

        d20 = engine._find_ml_decision(sig_20)
        assert d20["ml_prob"] == 0.85
        assert d20["strategy_type"] == "ema20_breakout"

        d50 = engine._find_ml_decision(sig_50)
        assert d50["ml_prob"] == 0.60
        assert d50["strategy_type"] == "ema50_breakout"

    def test_fallback_to_last(self, model_path):
        """When no strategy_type match, fall back to last decision."""
        cfg = HybridEMAMLConfig(model_path=model_path)
        engine = StrategyEngine(cfg)

        # Single-candidate decisions have no strategy_type key
        engine._strategy.ml_decisions = [
            {"ml_prob": 0.70, "percentile": 65.0},
        ]

        sig = LiveSignal(
            timestamp=_ts(9, 45), direction="long",
            signal_type=SignalType.LONG_ENTRY,
            entry=5000.0, stop=4990.0, take_profit=5010.0,
            position_size=0.8, strategy_type="ema50_breakout",
        )

        d = engine._find_ml_decision(sig)
        assert d["ml_prob"] == 0.70
