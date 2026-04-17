"""Tests for HybridEMAMLStrategy selection logic and configuration."""

import datetime
import os
import pickle
import tempfile

import pytest

from strategy.hybrid_ema_ml import (
    HybridEMAMLConfig,
    HybridEMAMLStrategy,
    TradeCandidate,
    STRATEGY_PRIORITY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    ema_length: int = 50,
    entry_type: str = "breakout",
    direction: str = "long",
    ml_prob: float = 0.6,
    strategy_type: str = "",
) -> TradeCandidate:
    """Create a minimal TradeCandidate for testing."""
    if not strategy_type:
        strategy_type = f"ema{ema_length}_{entry_type}"
    return TradeCandidate(
        ema_length=ema_length,
        entry_type=entry_type,
        direction=direction,
        entry_price=5000.0,
        stop_loss=4990.0,
        take_profit=5015.0,
        features={},
        ml_prob=ml_prob,
        timestamp=datetime.datetime(2024, 1, 2, 10, 0),
        strategy_type=strategy_type,
    )


def _base_config(**overrides) -> HybridEMAMLConfig:
    """Create a valid config with multi-candidate defaults."""
    defaults = dict(
        multi_candidate=True,
        max_trades_per_day=3,
        ema_periods=(20, 50, 100),
        entry_types=("breakout", "pullback", "momentum"),
        ml_selection_mode="threshold",
        ml_threshold=0.0,
        position_sizing_mode="none",
        selection_strategy="global_ml",
        model_path="__nonexistent__",  # won't load during tests
    )
    defaults.update(overrides)
    return HybridEMAMLConfig(**defaults)


def _make_dummy_model(path: str):
    """Write a minimal valid model pkl for testing."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.random.RandomState(42).randn(20, 3)
    y = (X[:, 0] > 0).astype(int)
    cols = ["f_a", "f_b", "f_c"]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    with open(path, "wb") as f:
        pickle.dump({"model": pipe, "feature_columns": cols}, f)


# ===========================================================================
# Config validation
# ===========================================================================

class TestConfigValidation:
    def test_valid_config(self):
        """A correct config does not raise."""
        cfg = _base_config()
        HybridEMAMLStrategy(cfg)

    def test_invalid_selection_strategy(self):
        with pytest.raises(ValueError, match="selection_strategy"):
            HybridEMAMLStrategy(_base_config(selection_strategy="magic"))

    def test_invalid_position_sizing_mode(self):
        with pytest.raises(ValueError, match="position_sizing_mode"):
            HybridEMAMLStrategy(_base_config(position_sizing_mode="quadratic"))

    def test_invalid_ml_selection_mode(self):
        with pytest.raises(ValueError, match="ml_selection_mode"):
            HybridEMAMLStrategy(_base_config(ml_selection_mode="neural"))

    def test_invalid_max_trades(self):
        with pytest.raises(ValueError, match="max_trades_per_day"):
            HybridEMAMLStrategy(_base_config(max_trades_per_day=0))

    def test_invalid_ml_threshold(self):
        with pytest.raises(ValueError, match="ml_threshold"):
            HybridEMAMLStrategy(_base_config(ml_threshold=1.5))

    def test_invalid_ema_period(self):
        with pytest.raises(ValueError, match="EMA period"):
            HybridEMAMLStrategy(_base_config(ema_periods=(0, 50)))

    def test_invalid_entry_type(self):
        with pytest.raises(ValueError, match="entry_type"):
            HybridEMAMLStrategy(_base_config(entry_types=("breakout", "fakeout")))


# ===========================================================================
# Model loading
# ===========================================================================

class TestModelLoading:
    def test_missing_model_file(self):
        cfg = _base_config(model_path="/nonexistent/model.pkl")
        strat = HybridEMAMLStrategy(cfg)
        with pytest.raises(FileNotFoundError, match="not found"):
            strat._ensure_model_loaded()

    def test_corrupted_model_file(self, tmp_path):
        bad_pkl = str(tmp_path / "bad.pkl")
        with open(bad_pkl, "wb") as f:
            f.write(b"not a pickle")
        cfg = _base_config(model_path=bad_pkl)
        strat = HybridEMAMLStrategy(cfg)
        with pytest.raises(RuntimeError, match="Failed to load"):
            strat._ensure_model_loaded()

    def test_missing_keys_in_model(self, tmp_path):
        bad_pkl = str(tmp_path / "no_keys.pkl")
        with open(bad_pkl, "wb") as f:
            pickle.dump({"wrong_key": 42}, f)
        cfg = _base_config(model_path=bad_pkl)
        strat = HybridEMAMLStrategy(cfg)
        with pytest.raises(ValueError, match="missing required keys"):
            strat._ensure_model_loaded()

    def test_valid_model_loads(self, tmp_path):
        model_path = str(tmp_path / "model.pkl")
        _make_dummy_model(model_path)
        cfg = _base_config(model_path=model_path)
        strat = HybridEMAMLStrategy(cfg)
        strat._ensure_model_loaded()
        assert strat._model is not None
        assert strat._feature_columns == ["f_a", "f_b", "f_c"]

    def test_idempotent_load(self, tmp_path):
        model_path = str(tmp_path / "model.pkl")
        _make_dummy_model(model_path)
        cfg = _base_config(model_path=model_path)
        strat = HybridEMAMLStrategy(cfg)
        strat._ensure_model_loaded()
        strat._ensure_model_loaded()  # second call should not re-load
        assert strat._model_loaded


# ===========================================================================
# Selection strategies
# ===========================================================================

class TestGlobalMLSelection:
    """Tests for the original global_ml selection strategy."""

    def test_empty_candidates(self):
        strat = HybridEMAMLStrategy(_base_config())
        result = strat._rank_and_select([])
        assert result == []

    def test_sorts_by_ml_prob_descending(self):
        strat = HybridEMAMLStrategy(_base_config(max_trades_per_day=5))
        candidates = [
            _make_candidate(ml_prob=0.3),
            _make_candidate(ml_prob=0.9),
            _make_candidate(ml_prob=0.6),
        ]
        selected = strat._select_global_ml(candidates)
        assert len(selected) == 3
        probs = [c.ml_prob for c in selected]
        assert probs == sorted(probs, reverse=True)

    def test_respects_max_trades(self):
        strat = HybridEMAMLStrategy(_base_config(max_trades_per_day=2))
        candidates = [
            _make_candidate(ml_prob=0.9),
            _make_candidate(ml_prob=0.8),
            _make_candidate(ml_prob=0.7),
        ]
        selected = strat._select_global_ml(candidates)
        assert len(selected) == 2

    def test_threshold_filters(self):
        strat = HybridEMAMLStrategy(_base_config(ml_threshold=0.5))
        candidates = [
            _make_candidate(ml_prob=0.6),
            _make_candidate(ml_prob=0.4),
        ]
        selected = strat._select_global_ml(candidates)
        assert len(selected) == 1
        assert selected[0].ml_prob == 0.6

    def test_logs_all_decisions(self):
        strat = HybridEMAMLStrategy(_base_config(max_trades_per_day=1))
        candidates = [
            _make_candidate(ml_prob=0.9),
            _make_candidate(ml_prob=0.5),
        ]
        strat._select_global_ml(candidates)
        assert len(strat.ml_decisions) == 2
        assert strat.ml_decisions[0]["accepted"] is True
        assert strat.ml_decisions[1]["accepted"] is False


class TestPrioritySelection:
    """Tests for the priority-based selection strategy."""

    def test_breakout_before_momentum(self):
        strat = HybridEMAMLStrategy(
            _base_config(selection_strategy="priority", max_trades_per_day=1)
        )
        # Momentum has higher ml_prob but lower priority
        candidates = [
            _make_candidate(entry_type="momentum", ml_prob=0.95),
            _make_candidate(entry_type="breakout", ml_prob=0.5),
        ]
        selected = strat._select_priority(candidates)
        assert len(selected) == 1
        assert selected[0].entry_type == "breakout"

    def test_momentum_before_pullback(self):
        strat = HybridEMAMLStrategy(
            _base_config(selection_strategy="priority", max_trades_per_day=1)
        )
        candidates = [
            _make_candidate(entry_type="pullback", ml_prob=0.95),
            _make_candidate(entry_type="momentum", ml_prob=0.5),
        ]
        selected = strat._select_priority(candidates)
        assert len(selected) == 1
        assert selected[0].entry_type == "momentum"

    def test_ml_ranks_within_group(self):
        strat = HybridEMAMLStrategy(
            _base_config(selection_strategy="priority", max_trades_per_day=1)
        )
        candidates = [
            _make_candidate(ema_length=20, entry_type="breakout", ml_prob=0.4),
            _make_candidate(ema_length=100, entry_type="breakout", ml_prob=0.9),
        ]
        selected = strat._select_priority(candidates)
        assert len(selected) == 1
        # Should pick the breakout with higher ml_prob
        assert selected[0].ema_length == 100

    def test_within_group_threshold(self):
        strat = HybridEMAMLStrategy(
            _base_config(
                selection_strategy="priority",
                ml_within_group_threshold=0.6,
                max_trades_per_day=5,
            )
        )
        candidates = [
            _make_candidate(entry_type="breakout", ml_prob=0.3),
            _make_candidate(entry_type="breakout", ml_prob=0.7),
            _make_candidate(entry_type="momentum", ml_prob=0.5),
        ]
        selected = strat._select_priority(candidates)
        # Only the breakout with 0.7 passes threshold
        assert len(selected) == 1
        assert selected[0].ml_prob == 0.7

    def test_full_priority_ordering(self):
        strat = HybridEMAMLStrategy(
            _base_config(selection_strategy="priority", max_trades_per_day=6)
        )
        candidates = [
            _make_candidate(entry_type="pullback", ml_prob=0.9),
            _make_candidate(entry_type="breakout", ml_prob=0.5),
            _make_candidate(entry_type="momentum", ml_prob=0.7),
            _make_candidate(entry_type="breakout", ml_prob=0.8),
            _make_candidate(entry_type="pullback", ml_prob=0.6),
            _make_candidate(entry_type="momentum", ml_prob=0.3),
        ]
        selected = strat._select_priority(candidates)
        types = [c.entry_type for c in selected]
        # All breakouts first, then all momentums, then all pullbacks
        assert types == ["breakout", "breakout", "momentum", "momentum",
                         "pullback", "pullback"]
        # Within breakouts: 0.8 before 0.5
        breakouts = [c for c in selected if c.entry_type == "breakout"]
        assert breakouts[0].ml_prob > breakouts[1].ml_prob


class TestPriorityMLSizing:
    """Tests for priority_ml_sizing selection."""

    def test_always_accepts(self):
        strat = HybridEMAMLStrategy(
            _base_config(selection_strategy="priority_ml_sizing", max_trades_per_day=5)
        )
        candidates = [
            _make_candidate(entry_type="pullback", ml_prob=0.01),
            _make_candidate(entry_type="breakout", ml_prob=0.99),
        ]
        selected = strat._select_priority_ml_sizing(candidates)
        # Both accepted regardless of ml_prob
        assert len(selected) == 2

    def test_respects_priority_order(self):
        strat = HybridEMAMLStrategy(
            _base_config(selection_strategy="priority_ml_sizing", max_trades_per_day=1)
        )
        candidates = [
            _make_candidate(entry_type="pullback", ml_prob=0.99),
            _make_candidate(entry_type="breakout", ml_prob=0.01),
        ]
        selected = strat._select_priority_ml_sizing(candidates)
        assert len(selected) == 1
        assert selected[0].entry_type == "breakout"

    def test_size_varies_by_percentile(self):
        strat = HybridEMAMLStrategy(
            _base_config(selection_strategy="priority_ml_sizing", max_trades_per_day=5)
        )
        # Seed rolling window so percentiles differ
        for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
            strat._prob_window_all.append(p)
        candidates = [
            _make_candidate(entry_type="breakout", ml_prob=0.2),
            _make_candidate(entry_type="breakout", ml_prob=0.8),
        ]
        selected = strat._select_priority_ml_sizing(candidates)
        assert len(selected) == 2
        # Higher ml_prob should get larger position_size
        sizes = {c.ml_prob: c.position_size for c in selected}
        assert sizes[0.8] > sizes[0.2]

    def test_minimum_size_floor(self):
        strat = HybridEMAMLStrategy(
            _base_config(
                selection_strategy="priority_ml_sizing",
                max_trades_per_day=5,
                base_size=1.0,
            )
        )
        # Even worst percentile should get at least 0.1 * base_size
        for p in [0.9, 0.95, 0.99]:
            strat._prob_window_all.append(p)
        candidates = [_make_candidate(entry_type="breakout", ml_prob=0.01)]
        selected = strat._select_priority_ml_sizing(candidates)
        assert selected[0].position_size >= 0.1


# ===========================================================================
# Priority ordering helper
# ===========================================================================

class TestOrderByPriority:
    def test_priority_constant_ordering(self):
        assert STRATEGY_PRIORITY["breakout"] < STRATEGY_PRIORITY["momentum"]
        assert STRATEGY_PRIORITY["momentum"] < STRATEGY_PRIORITY["pullback"]

    def test_order_by_priority(self):
        strat = HybridEMAMLStrategy(_base_config(selection_strategy="priority"))
        candidates = [
            _make_candidate(entry_type="pullback", ml_prob=0.9),
            _make_candidate(entry_type="breakout", ml_prob=0.3),
            _make_candidate(entry_type="momentum", ml_prob=0.6),
        ]
        ordered = strat._order_by_priority(candidates)
        types = [c.entry_type for c in ordered]
        assert types == ["breakout", "momentum", "pullback"]


# ===========================================================================
# Position sizing
# ===========================================================================

class TestPositionSizing:
    def test_none_mode(self):
        strat = HybridEMAMLStrategy(_base_config(position_sizing_mode="none"))
        assert strat._compute_position_size(0.0) == 1.0
        assert strat._compute_position_size(1.0) == 1.0

    def test_linear_mode(self):
        strat = HybridEMAMLStrategy(
            _base_config(position_sizing_mode="linear", base_size=2.0)
        )
        assert strat._compute_position_size(0.0) == 0.0
        assert strat._compute_position_size(0.5) == pytest.approx(1.0)
        assert strat._compute_position_size(1.0) == pytest.approx(2.0)

    def test_convex_mode(self):
        strat = HybridEMAMLStrategy(
            _base_config(position_sizing_mode="convex", base_size=1.0)
        )
        assert strat._compute_position_size(0.5) == pytest.approx(0.25)

    def test_hybrid_mode(self):
        strat = HybridEMAMLStrategy(
            _base_config(position_sizing_mode="hybrid", base_size=1.0)
        )
        assert strat._compute_position_size(0.3) == 0.0
        assert strat._compute_position_size(0.75) == pytest.approx(0.5)
        assert strat._compute_position_size(1.0) == pytest.approx(1.0)

    def test_unknown_mode_raises(self):
        # Validation should catch this at config time
        with pytest.raises(ValueError, match="position_sizing_mode"):
            HybridEMAMLStrategy(_base_config(position_sizing_mode="quartic"))


# ===========================================================================
# Reset and reproducibility
# ===========================================================================

class TestResetReproducibility:
    def test_reset_clears_all_state(self):
        strat = HybridEMAMLStrategy(_base_config())
        # Simulate some state accumulation
        strat._prob_window_all.append(0.5)
        strat._prob_window_long.append(0.6)
        strat.ml_decisions.append({"test": True})
        strat._bar_count = 10
        strat._close_history.append(5000.0)
        strat._open_positions["test"] = {"direction": "long"}
        strat.in_position = True
        strat._current_date = datetime.date(2024, 1, 1)

        strat.reset()

        assert len(strat._prob_window_all) == 0
        assert len(strat._prob_window_long) == 0
        assert len(strat._prob_window_short) == 0
        assert len(strat.ml_decisions) == 0
        assert strat._bar_count == 0
        assert len(strat._close_history) == 0
        assert len(strat._open_positions) == 0
        assert strat.in_position is False
        assert strat._current_date is None

    def test_two_runs_identical(self, tmp_path):
        """Running the same config twice produces identical decisions."""
        model_path = str(tmp_path / "model.pkl")
        _make_dummy_model(model_path)
        cfg = _base_config(model_path=model_path, max_trades_per_day=2)
        strat = HybridEMAMLStrategy(cfg)

        # Build minimal candidates
        cands = [
            _make_candidate(ml_prob=0.7),
            _make_candidate(ml_prob=0.3),
        ]

        # Run 1
        strat._ensure_model_loaded()
        r1 = strat._rank_and_select(list(cands))
        d1 = list(strat.ml_decisions)

        # Reset and run 2
        strat.reset()
        strat._ensure_model_loaded()
        r2 = strat._rank_and_select(list(cands))
        d2 = list(strat.ml_decisions)

        assert len(r1) == len(r2)
        assert len(d1) == len(d2)
        for a, b in zip(d1, d2):
            assert a["accepted"] == b["accepted"]
            assert a["ml_prob"] == b["ml_prob"]
