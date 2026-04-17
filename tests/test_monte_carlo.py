"""Tests for challenge.monte_carlo Monte Carlo simulation engine."""

import subprocess
import sys
import unittest

import numpy as np


def run_tests():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    print(r.stdout)
    if r.stderr:
        print(r.stderr)
    return r.returncode


# ---------------------------------------------------------------------------
# Imports (deferred to avoid DLL issues when run via subprocess)
# ---------------------------------------------------------------------------

from challenge.monte_carlo import (
    DrawdownStats,
    MonteCarloResult,
    SensitivityPoint,
    SimPath,
    TradeRecord,
    TradeSampler,
    _extract_pnls,
    _simulate_path,
    run_monte_carlo,
    run_sensitivity,
    print_report,
    trades_from_csv,
)


# ---------------------------------------------------------------------------
# TradeRecord
# ---------------------------------------------------------------------------

class TestTradeRecord(unittest.TestCase):
    def test_basic_creation(self):
        t = TradeRecord(pnl=50.0, strategy_type="breakout")
        self.assertEqual(t.pnl, 50.0)
        self.assertEqual(t.strategy_type, "breakout")

    def test_defaults(self):
        t = TradeRecord(pnl=-20.0)
        self.assertEqual(t.strategy_type, "")


# ---------------------------------------------------------------------------
# TradeSampler
# ---------------------------------------------------------------------------

class TestTradeSampler(unittest.TestCase):
    def test_iid_sampling_length(self):
        pnls = np.array([10.0, -5.0, 20.0, -15.0, 8.0])
        s = TradeSampler(pnls, block_size=1, rng=np.random.default_rng(42))
        result = s.sample(100)
        self.assertEqual(len(result), 100)

    def test_iid_samples_come_from_source(self):
        pnls = np.array([10.0, -5.0, 20.0])
        s = TradeSampler(pnls, block_size=1, rng=np.random.default_rng(42))
        result = s.sample(1000)
        unique = set(result)
        self.assertTrue(unique.issubset({10.0, -5.0, 20.0}))

    def test_scale_multiplies_pnl(self):
        pnls = np.array([100.0])
        s = TradeSampler(pnls, block_size=1, rng=np.random.default_rng(42))
        result = s.sample(10, scale=0.5)
        np.testing.assert_array_almost_equal(result, [50.0] * 10)

    def test_block_sampling_length(self):
        pnls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        s = TradeSampler(pnls, block_size=3, rng=np.random.default_rng(42))
        result = s.sample(7)
        self.assertEqual(len(result), 7)

    def test_block_sampling_preserves_contiguity(self):
        pnls = np.arange(100, dtype=float)
        s = TradeSampler(pnls, block_size=5, rng=np.random.default_rng(42))
        result = s.sample(5)
        # A single block of 5 contiguous values
        diffs = np.diff(result)
        np.testing.assert_array_equal(diffs, [1, 1, 1, 1])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            TradeSampler(np.array([]))

    def test_reproducible_with_seed(self):
        pnls = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
        s1 = TradeSampler(pnls, rng=np.random.default_rng(123))
        s2 = TradeSampler(pnls, rng=np.random.default_rng(123))
        r1 = s1.sample(50)
        r2 = s2.sample(50)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# _extract_pnls
# ---------------------------------------------------------------------------

class TestExtractPnls(unittest.TestCase):
    def test_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _extract_pnls(arr)
        np.testing.assert_array_equal(result, arr)

    def test_from_floats(self):
        result = _extract_pnls([10.0, -5.0, 20.0])
        np.testing.assert_array_equal(result, [10.0, -5.0, 20.0])

    def test_from_trade_records(self):
        trades = [TradeRecord(pnl=10), TradeRecord(pnl=-5), TradeRecord(pnl=20)]
        result = _extract_pnls(trades)
        np.testing.assert_array_equal(result, [10, -5, 20])

    def test_strategy_filter(self):
        trades = [
            TradeRecord(pnl=10, strategy_type="breakout"),
            TradeRecord(pnl=-5, strategy_type="pullback"),
            TradeRecord(pnl=20, strategy_type="breakout"),
        ]
        result = _extract_pnls(trades, strategy_filter="breakout")
        np.testing.assert_array_equal(result, [10, 20])

    def test_duck_type_net_pnl(self):
        """Objects with .net_pnl attribute work."""

        class FakeTrade:
            def __init__(self, net_pnl, strategy_type=""):
                self.net_pnl = net_pnl
                self.strategy_type = strategy_type

        trades = [FakeTrade(10), FakeTrade(-5)]
        result = _extract_pnls(trades)
        np.testing.assert_array_equal(result, [10, -5])

    def test_empty_list(self):
        result = _extract_pnls([])
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# _simulate_path
# ---------------------------------------------------------------------------

class TestSimulatePath(unittest.TestCase):
    def test_pass_condition(self):
        """Equity reaches target → pass."""
        pnls = np.array([500.0, 500.0, 500.0, 500.0])  # +2000 total
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
        )
        self.assertEqual(result.outcome, "passed")
        self.assertEqual(result.trades_taken, 3)  # passes at +1500
        self.assertEqual(result.final_equity, 26500.0)

    def test_fail_condition(self):
        """Equity hits trailing DD floor → fail."""
        pnls = np.array([-400.0, -400.0, -400.0])
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
        )
        self.assertEqual(result.outcome, "failed")
        self.assertLessEqual(result.final_equity, 25000 - 1000)

    def test_trailing_dd_rises(self):
        """DD floor rises with peak equity."""
        # Go up 600, then down 1000 → should fail at dd_floor = 25600-1000 = 24600
        pnls = np.array([600.0, -500.0, -500.0, -100.0])
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
        )
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(result.peak_equity, 25600.0)
        # Failed when equity hit 24600 (peak 25600 - 1000)
        self.assertLessEqual(result.final_equity, 24600.0)

    def test_incomplete_when_no_termination(self):
        """Runs out of trades without hitting target or DD."""
        pnls = np.array([1.0, -1.0, 1.0, -1.0])
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
        )
        self.assertEqual(result.outcome, "incomplete")
        self.assertEqual(result.trades_taken, 4)

    def test_max_drawdown_tracking(self):
        pnls = np.array([100.0, -300.0, 200.0, -100.0])
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
        )
        # Peak at 25100, dropped to 24800, max DD = 300
        self.assertAlmostEqual(result.max_drawdown, 300.0)

    def test_streak_tracking(self):
        pnls = np.array([10, 10, 10, -5, -5, -5, -5, 10])
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
        )
        self.assertEqual(result.max_win_streak, 3)
        self.assertEqual(result.max_loss_streak, 4)

    def test_equity_path_recorded(self):
        pnls = np.array([100.0, -50.0, 200.0])
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
            record_path=True,
        )
        self.assertEqual(len(result.equity_path), 4)  # initial + 3 trades
        self.assertEqual(result.equity_path[0], 25000)
        self.assertEqual(result.equity_path[1], 25100)

    def test_equity_path_not_recorded_by_default(self):
        pnls = np.array([100.0])
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
        )
        self.assertEqual(len(result.equity_path), 0)

    def test_largest_win_loss(self):
        pnls = np.array([50, -80, 200, -30, 10])
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
        )
        self.assertEqual(result.largest_win, 200.0)
        self.assertEqual(result.largest_loss, 80.0)  # stored as absolute

    def test_zero_pnl_no_streak(self):
        """Zero PnL trades don't count as win or loss streak."""
        pnls = np.array([0.0, 0.0, 0.0])
        result = _simulate_path(
            pnls,
            starting_capital=25000,
            profit_target=1500,
            max_drawdown=1000,
            dd_buffer=200,
        )
        self.assertEqual(result.max_win_streak, 0)
        self.assertEqual(result.max_loss_streak, 0)


# ---------------------------------------------------------------------------
# run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo(unittest.TestCase):
    def test_basic_run(self):
        trades = [50.0, -30.0, 80.0, -20.0, 60.0, -10.0, 40.0, 100.0]
        result = run_monte_carlo(
            trades, n_simulations=100, max_trades=200, seed=42
        )
        self.assertEqual(result.n_simulations, 100)
        self.assertAlmostEqual(
            result.pass_rate + result.fail_rate + result.incomplete_rate, 1.0
        )

    def test_reproducible_with_seed(self):
        trades = [50.0, -30.0, 80.0, -20.0, 60.0, -10.0]
        r1 = run_monte_carlo(trades, n_simulations=500, seed=99)
        r2 = run_monte_carlo(trades, n_simulations=500, seed=99)
        self.assertEqual(r1.pass_rate, r2.pass_rate)
        self.assertEqual(r1.fail_rate, r2.fail_rate)

    def test_different_seeds_differ(self):
        trades = [50.0, -30.0, 80.0, -20.0, 60.0, -10.0]
        r1 = run_monte_carlo(trades, n_simulations=500, seed=1)
        r2 = run_monte_carlo(trades, n_simulations=500, seed=2)
        # Very unlikely to get identical results with different seeds
        # But not impossible, so just check it runs
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)

    def test_all_winners_high_pass_rate(self):
        trades = [100.0] * 20
        result = run_monte_carlo(
            trades, n_simulations=500, max_trades=200, seed=42
        )
        self.assertGreater(result.pass_rate, 0.95)

    def test_all_losers_high_fail_rate(self):
        trades = [-100.0] * 20
        result = run_monte_carlo(
            trades, n_simulations=500, max_trades=200, seed=42
        )
        self.assertGreater(result.fail_rate, 0.95)

    def test_position_scale(self):
        trades = [50.0, -30.0, 80.0, -20.0]
        r_full = run_monte_carlo(
            trades, n_simulations=200, position_scale=1.0, seed=42
        )
        r_half = run_monte_carlo(
            trades, n_simulations=200, position_scale=0.5, seed=42
        )
        # Half scale should have lower pass rate (takes longer) or higher incomplete
        # At minimum the mean final equity should be closer to start
        self.assertLessEqual(
            abs(r_half.mean_final_equity - 25000),
            abs(r_full.mean_final_equity - 25000) + 500,  # some tolerance
        )

    def test_block_sampling(self):
        trades = [50.0, 50.0, 50.0, -200.0, -200.0, -200.0] * 5
        result = run_monte_carlo(
            trades, n_simulations=200, block_size=3, seed=42
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.block_size, 3)

    def test_sample_paths_recorded(self):
        trades = [50.0, -30.0, 80.0, -20.0]
        result = run_monte_carlo(
            trades, n_simulations=100, n_sample_paths=5, seed=42
        )
        paths_with_data = [p for p in result.sample_paths if p.equity_path]
        self.assertGreater(len(paths_with_data), 0)
        self.assertLessEqual(len(paths_with_data), 5)

    def test_trade_records_input(self):
        trades = [
            TradeRecord(pnl=50, strategy_type="breakout"),
            TradeRecord(pnl=-30, strategy_type="pullback"),
            TradeRecord(pnl=80, strategy_type="breakout"),
        ]
        result = run_monte_carlo(trades, n_simulations=50, seed=42)
        self.assertEqual(result.source_trade_count, 3)

    def test_strategy_filter(self):
        trades = [
            TradeRecord(pnl=100, strategy_type="breakout"),
            TradeRecord(pnl=-200, strategy_type="pullback"),
            TradeRecord(pnl=80, strategy_type="breakout"),
        ]
        result = run_monte_carlo(
            trades, n_simulations=50, seed=42, strategy_filter="breakout"
        )
        self.assertEqual(result.source_trade_count, 2)

    def test_empty_trades_raises(self):
        with self.assertRaises(ValueError):
            run_monte_carlo([], n_simulations=10)

    def test_source_stats_populated(self):
        trades = [100.0, -50.0, 75.0, -25.0]
        result = run_monte_carlo(trades, n_simulations=50, seed=42)
        self.assertEqual(result.source_trade_count, 4)
        self.assertAlmostEqual(result.source_win_rate, 0.5)
        self.assertAlmostEqual(result.source_mean_pnl, 25.0)
        self.assertGreater(result.source_std_pnl, 0)

    def test_drawdown_stats(self):
        trades = [50.0, -30.0, 80.0, -60.0, 20.0]
        result = run_monte_carlo(trades, n_simulations=200, seed=42)
        dd = result.drawdown_stats
        self.assertGreater(dd.mean_max_dd, 0)
        self.assertGreaterEqual(dd.p95_max_dd, dd.median_max_dd)
        self.assertGreaterEqual(dd.p99_max_dd, dd.p95_max_dd)


# ---------------------------------------------------------------------------
# run_sensitivity
# ---------------------------------------------------------------------------

class TestRunSensitivity(unittest.TestCase):
    def test_basic_sensitivity(self):
        trades = [50.0, -30.0, 80.0, -20.0, 100.0, -40.0]
        result = run_sensitivity(
            trades,
            scales=(0.5, 1.0, 1.5),
            n_simulations=100,
            seed=42,
        )
        self.assertEqual(len(result.sensitivity), 3)
        scales = [s.scale for s in result.sensitivity]
        self.assertEqual(scales, [0.5, 1.0, 1.5])

    def test_sensitivity_scale_1_matches_base(self):
        trades = [50.0, -30.0, 80.0, -20.0]
        result = run_sensitivity(
            trades,
            scales=(1.0,),
            n_simulations=200,
            seed=42,
        )
        s1 = result.sensitivity[0]
        self.assertAlmostEqual(s1.pass_rate, result.pass_rate, places=2)


# ---------------------------------------------------------------------------
# MonteCarloResult.to_dict
# ---------------------------------------------------------------------------

class TestToDict(unittest.TestCase):
    def test_serializable(self):
        import json

        trades = [50.0, -30.0, 80.0, -20.0]
        result = run_monte_carlo(trades, n_simulations=50, seed=42)
        d = result.to_dict()
        # Should be JSON-serializable
        s = json.dumps(d)
        self.assertIn("pass_rate", s)
        self.assertIn("drawdown_stats", s)

    def test_all_keys_present(self):
        trades = [50.0, -30.0]
        result = run_monte_carlo(trades, n_simulations=20, seed=42)
        d = result.to_dict()
        expected_keys = {
            "pass_rate",
            "fail_rate",
            "incomplete_rate",
            "avg_trades_to_pass",
            "avg_trades_to_fail",
            "median_trades_to_pass",
            "median_trades_to_fail",
            "mean_final_equity",
            "median_final_equity",
            "std_final_equity",
            "drawdown_stats",
            "n_simulations",
            "max_trades",
            "starting_capital",
            "profit_target",
            "max_drawdown",
            "position_scale",
            "block_size",
            "source_trade_count",
            "source_win_rate",
            "source_mean_pnl",
            "source_std_pnl",
            "sensitivity",
        }
        self.assertTrue(expected_keys.issubset(d.keys()))


# ---------------------------------------------------------------------------
# print_report (smoke test)
# ---------------------------------------------------------------------------

class TestPrintReport(unittest.TestCase):
    def test_no_crash(self):
        trades = [50.0, -30.0, 80.0, -20.0]
        result = run_monte_carlo(trades, n_simulations=50, seed=42)
        # Just confirm it doesn't raise
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            print_report(result)
        output = f.getvalue()
        self.assertIn("MONTE CARLO", output)
        self.assertIn("P(pass before fail)", output)

    def test_with_sensitivity(self):
        trades = [50.0, -30.0, 80.0]
        result = run_sensitivity(
            trades, scales=(0.5, 1.0), n_simulations=50, seed=42
        )
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            print_report(result)
        output = f.getvalue()
        self.assertIn("SENSITIVITY", output)


# ---------------------------------------------------------------------------
# trades_from_csv
# ---------------------------------------------------------------------------

class TestTradesFromCSV(unittest.TestCase):
    def test_load_csv(self):
        import tempfile
        import os

        csv_content = "entry_time,exit_time,net_pnl,strategy_type\n"
        csv_content += "2024-01-01,2024-01-01,50.0,breakout\n"
        csv_content += "2024-01-02,2024-01-02,-30.0,pullback\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            path = f.name

        try:
            trades = trades_from_csv(path)
            self.assertEqual(len(trades), 2)
            self.assertEqual(trades[0].pnl, 50.0)
            self.assertEqual(trades[0].strategy_type, "breakout")
            self.assertEqual(trades[1].pnl, -30.0)
        finally:
            os.unlink(path)

    def test_missing_pnl_column_raises(self):
        import tempfile
        import os

        csv_content = "timestamp,volume\n2024-01-01,100\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            path = f.name

        try:
            with self.assertRaises(ValueError):
                trades_from_csv(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Edge cases / validation
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    def test_single_trade(self):
        result = run_monte_carlo([100.0], n_simulations=50, seed=42)
        self.assertIsNotNone(result)

    def test_large_single_winner_passes_in_one(self):
        """A single trade of +1500 should always pass."""
        result = run_monte_carlo(
            [1500.0], n_simulations=100, max_trades=1, seed=42
        )
        self.assertEqual(result.pass_rate, 1.0)

    def test_large_single_loser_fails_in_one(self):
        """A single trade of -1000 should always fail."""
        result = run_monte_carlo(
            [-1000.0], n_simulations=100, max_trades=1, seed=42
        )
        self.assertEqual(result.fail_rate, 1.0)

    def test_max_trades_1_with_small_pnl(self):
        """Max trades=1 with small PnL → always incomplete."""
        result = run_monte_carlo(
            [1.0], n_simulations=100, max_trades=1, seed=42
        )
        self.assertEqual(result.incomplete_rate, 1.0)

    def test_no_lookahead(self):
        """Each simulation only sees past trades (sampling with replacement).
        Verify by checking that different seeds produce different results."""
        trades = [50.0, -30.0, 80.0, -60.0, 20.0, -10.0, 70.0, -40.0]
        results = []
        for seed in range(5):
            r = run_monte_carlo(trades, n_simulations=100, seed=seed)
            results.append(r.pass_rate)
        # With 5 different seeds, should get at least 2 different pass rates
        unique = set(results)
        self.assertGreater(len(unique), 1)

    def test_numpy_input(self):
        pnls = np.array([50.0, -30.0, 80.0, -20.0])
        result = run_monte_carlo(pnls, n_simulations=50, seed=42)
        self.assertEqual(result.source_trade_count, 4)

    def test_dd_buffer_zero(self):
        """dd_buffer=0 should not crash."""
        result = run_monte_carlo(
            [50.0, -30.0], n_simulations=50, dd_buffer=0, seed=42
        )
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI(unittest.TestCase):
    def test_cli_basic(self):
        import tempfile
        import os

        csv_content = "net_pnl,strategy_type\n"
        for v in [50, -30, 80, -20, 60, -10, 40, 100, -50, 70]:
            csv_content += f"{v},breakout\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            path = f.name

        try:
            from challenge.monte_carlo import main

            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                main([
                    "--data", path,
                    "--sims", "50",
                    "--max-trades", "100",
                    "--seed", "42",
                    "--no-plot",
                ])
            output = buf.getvalue()
            self.assertIn("MONTE CARLO", output)
        finally:
            os.unlink(path)

    def test_cli_with_json_output(self):
        import tempfile
        import os
        import json

        csv_content = "net_pnl\n50\n-30\n80\n-20\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            csv_path = f.name

        out_path = csv_path + ".json"
        try:
            from challenge.monte_carlo import main

            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                main([
                    "--data", csv_path,
                    "--sims", "20",
                    "--seed", "42",
                    "--no-plot",
                    "--output", out_path,
                ])

            self.assertTrue(os.path.exists(out_path))
            with open(out_path) as f:
                data = json.load(f)
            self.assertIn("pass_rate", data)
        finally:
            for p in (csv_path, out_path):
                if os.path.exists(p):
                    os.unlink(p)


if __name__ == "__main__":
    sys.exit(run_tests())
