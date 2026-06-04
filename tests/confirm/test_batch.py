"""Tests for crabquant.confirm.batch module."""

from unittest.mock import patch, MagicMock, call

import pytest

from crabquant.confirm import ConfirmationResult
from crabquant.confirm.batch import (
    _aggregate_results,
    batch_confirm,
    PERIODS,
    SLIPPAGE_LEVELS,
)


def _make_result(passed=True, sharpe=1.5, trades=20, notes=None):
    return ConfirmationResult(
        sharpe=sharpe, total_return=0.1, max_dd=-0.05,
        trades=trades, win_rate=0.55, profit_factor=1.3,
        expectancy=0.01, passed=passed,
        notes=notes or [],
    )


class TestAggregateResults:
    def test_robust_pass(self):
        """All pass including 0.2% slippage → robust."""
        # 3 slippage levels × 3 periods = 9 results
        # All pass
        results = [_make_result(True) for _ in range(9)]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is True
        assert fragile is False

    def test_fragile_pass(self):
        """Passes base but fails at 0.2% slippage → fragile."""
        results = []
        # Period 1: pass at 0%, pass at 0.1%, FAIL at 0.2%
        results.append(_make_result(True))
        results.append(_make_result(True))
        results.append(_make_result(False))
        # Period 2: pass at 0%, pass at 0.1%, FAIL at 0.2%
        results.append(_make_result(True))
        results.append(_make_result(True))
        results.append(_make_result(False))
        # Period 3: pass at 0%, pass at 0.1%, FAIL at 0.2%
        results.append(_make_result(True))
        results.append(_make_result(True))
        results.append(_make_result(False))

        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is False
        assert fragile is True

    def test_failed(self):
        """Fails at base → failed."""
        results = []
        # Period 1: fail at 0%
        results.append(_make_result(False))
        results.append(_make_result(False))
        results.append(_make_result(False))
        # Period 2: pass at 0%
        results.append(_make_result(True))
        results.append(_make_result(True))
        results.append(_make_result(True))
        # Period 3: pass at 0%
        results.append(_make_result(True))
        results.append(_make_result(True))
        results.append(_make_result(True))

        base, robust, fragile = _aggregate_results(results)
        assert base is False
        assert robust is False
        assert fragile is False

    def test_empty_results(self):
        base, robust, fragile = _aggregate_results([])
        assert base is False
        assert robust is False
        assert fragile is False

    def test_robust_needs_multi_period(self):
        """Robust requires passing at 0.2% slip AND ≥2 periods at 0% slip."""
        results = []
        # Period 1 (2y): pass at all slippage levels
        results.append(_make_result(True))
        results.append(_make_result(True))
        results.append(_make_result(True))
        # Period 2 (1y): FAIL at 0% slip (and presumably others)
        results.append(_make_result(False))
        results.append(_make_result(False))
        results.append(_make_result(False))
        # Period 3 (6mo): pass at 0% slip
        results.append(_make_result(True))
        results.append(_make_result(True))
        results.append(_make_result(True))

        base, robust, fragile = _aggregate_results(results)
        # Passes at 0.2% slip (primary period) but only 2/3 periods at 0% → robust
        assert base is True
        assert robust is True  # 2 out of 3 periods pass at 0% slip
        assert fragile is False

    def test_single_period_only(self):
        """Only 3 results (1 period × 3 slippage levels)."""
        results = [
            _make_result(True),   # 0% slip
            _make_result(True),   # 0.1% slip
            _make_result(True),   # 0.2% slip
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is False  # Only 1 period passes at 0% slip, need ≥2
        assert fragile is False

    def test_two_periods_robust(self):
        """Two periods, both pass at 0% slip and primary passes at 0.2%."""
        results = [
            _make_result(True), _make_result(True), _make_result(True),  # 2y
            _make_result(True), _make_result(True), _make_result(True),  # 1y
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is True  # 2/2 periods pass at 0% slip
        assert fragile is False

    # ── New tests ──────────────────────────────────────────────────────

    def test_single_result(self):
        """Single result only — not enough for robust, but fragile since base passes."""
        results = [_make_result(True)]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is False  # Need ≥2 periods for multi-period check
        # fragile = base_pass and not overall_robust → True and not False → True
        assert fragile is True

    def test_two_results(self):
        """Two results (0% and 0.1% slippage for one period)."""
        results = [
            _make_result(True),   # 0% slip
            _make_result(True),   # 0.1% slip
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is False  # No 0.2% result

    def test_base_fail_robust_pass_impossible(self):
        """If base fails but other periods pass and primary 0.2% passes → robust can be True."""
        results = [
            _make_result(False), _make_result(False), _make_result(True),  # 2y: fail base
            _make_result(True),  _make_result(True),  _make_result(True),  # 1y
            _make_result(True),  _make_result(True),  _make_result(True),  # 6mo
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is False
        # primary_results[2].passed = True → robust_pass = True
        # period_pass_count = 2 (1y + 6mo at 0% slip) → multi_period_pass = True
        # overall_robust = True and True = True
        assert robust is True
        assert fragile is False  # fragile = base and not robust = False

    def test_all_fail(self):
        """All results fail → all False."""
        results = [_make_result(False) for _ in range(9)]
        base, robust, fragile = _aggregate_results(results)
        assert base is False
        assert robust is False
        assert fragile is False

    def test_four_results(self):
        """Four results: 1 full period + 1 partial (only 0% slip)."""
        results = [
            _make_result(True), _make_result(True), _make_result(True),   # 2y full
            _make_result(True),                                           # 1y 0% only
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is True  # 2 periods pass at 0% slip (2y and 1y)
        assert fragile is False

    def test_robust_with_all_three_periods(self):
        """All 3 periods pass at 0% and primary passes at 0.2%."""
        results = [
            _make_result(True), _make_result(True), _make_result(True),  # 2y
            _make_result(True), _make_result(True), _make_result(True),  # 1y
            _make_result(True), _make_result(True), _make_result(True),  # 6mo
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is True
        assert fragile is False

    def test_primary_pass_robust_fail_only_one_period_pass(self):
        """Primary passes at 0.2% but only 1/3 periods pass at 0%."""
        results = [
            _make_result(True),  _make_result(True),  _make_result(True),  # 2y: all pass
            _make_result(False), _make_result(False), _make_result(False), # 1y: all fail
            _make_result(False), _make_result(False), _make_result(False), # 6mo: all fail
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is False  # Only 1/3 periods pass at 0% slip
        assert fragile is False  # Passes 0.2% but robust is False, fragile = base and not robust

    def test_fragile_not_robust(self):
        """Fragile and robust are mutually exclusive."""
        results = [
            _make_result(True),  _make_result(True),  _make_result(False), # 2y
            _make_result(True),  _make_result(True),  _make_result(False), # 1y
            _make_result(True),  _make_result(True),  _make_result(False), # 6mo
        ]
        base, robust, fragile = _aggregate_results(results)
        # fragile = base and not overall_robust
        # overall_robust = robust_pass and multi_period_pass
        # robust_pass = primary_results[2].passed = False
        # So overall_robust = False
        # fragile = True and not False = True
        assert base is True
        assert robust is False
        assert fragile is True
        assert not (robust and fragile)

    def test_five_results(self):
        """5 results: 1 full period + 1 full period + 1 partial."""
        results = [
            _make_result(True), _make_result(True), _make_result(True),  # 2y
            _make_result(True), _make_result(True), _make_result(True),  # 1y
            _make_result(True),                                           # 6mo 0% only
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is True  # 3 periods at 0% slip pass

    def test_primary_period_fails_robust_but_multi_period_pass(self):
        """Primary fails at 0.2% but 2+ periods pass at 0% → not robust."""
        results = [
            _make_result(True),  _make_result(True),  _make_result(False), # 2y: fail 0.2%
            _make_result(True),  _make_result(True),  _make_result(True),  # 1y
            _make_result(True),  _make_result(True),  _make_result(True),  # 6mo
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is False  # robust_pass = primary_results[2].passed = False
        assert fragile is True


class TestBatchConfirm:
    def test_robust_verdict(self):
        """Strategy that passes everywhere → ROBUST."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result):
            result = batch_confirm(winner)

        assert result.verdict == "ROBUST"
        assert result.passed is True
        assert result.sharpe == 1.5

    def test_failed_verdict(self):
        """Strategy that fails at base → FAILED."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=0.2, total_return=0.01, max_dd=-0.15,
            trades=5, win_rate=0.4, profit_factor=0.8,
            expectancy=-0.01, passed=False,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result):
            result = batch_confirm(winner)

        assert result.verdict == "FAILED"
        assert result.passed is False

    def test_fragile_verdict(self):
        """Strategy that passes at 0% but fails at 0.2% → FRAGILE."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        def mock_confirm(*args, **kwargs):
            slip = kwargs.get("slippage_pct", 0.0)
            if slip >= 0.002:
                return ConfirmationResult(
                    sharpe=0.3, total_return=0.01, max_dd=-0.12,
                    trades=10, win_rate=0.48, profit_factor=0.9,
                    expectancy=-0.005, passed=False,
                )
            return ConfirmationResult(
                sharpe=1.5, total_return=0.1, max_dd=-0.05,
                trades=20, win_rate=0.55, profit_factor=1.3,
                expectancy=0.01, passed=True,
            )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", side_effect=mock_confirm):
            result = batch_confirm(winner)

        assert result.verdict == "FRAGILE"
        assert result.passed is True  # FRAGILE still "passes" for promotion consideration

    def test_insufficient_data(self):
        """Short data → padded with failures → FAILED."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=50)  # < 100 threshold

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df):
            result = batch_confirm(winner)

        assert result.verdict == "FAILED"
        assert result.passed is False

    def test_data_load_error(self):
        """Data load failure → FAILED."""
        winner = {"strategy": "test", "ticker": "BAD", "params": {}}

        with patch("crabquant.confirm.batch.load_data", side_effect=Exception("No data")):
            result = batch_confirm(winner)

        assert result.verdict == "FAILED"
        assert result.passed is False

    def test_n_periods_param(self):
        """n_periods should limit the number of periods tested."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df) as mock_load, \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result) as mock_confirm:
            result = batch_confirm(winner, n_periods=1)

        # Should only load data for 1 period
        assert mock_load.call_count == 1
        # Should run 3 slippage levels for that period
        assert mock_confirm.call_count == 3

    def test_notes_contain_period_and_slippage(self):
        """Notes should indicate which period and slippage level."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        def make_mock_result():
            return ConfirmationResult(
                sharpe=1.5, total_return=0.1, max_dd=-0.05,
                trades=20, win_rate=0.55, profit_factor=1.3,
                expectancy=0.01, passed=True,
            )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", side_effect=lambda *a, **k: make_mock_result()):
            result = batch_confirm(winner)

        # Check notes contain verdict
        assert any("Verdict: ROBUST" in n for n in result.notes)
        # Check notes contain period info (first period is 2y)
        assert any("2y" in n for n in result.notes)

    # ── New tests ──────────────────────────────────────────────────────

    def test_slippage_levels_constant(self):
        """SLIPPAGE_LEVELS should contain expected values."""
        assert SLIPPAGE_LEVELS == [0.0, 0.001, 0.002]

    def test_periods_constant(self):
        """PERIODS should contain expected values."""
        assert PERIODS == ["2y", "1y", "6mo"]

    def test_robust_result_stats_from_primary(self):
        """Result stats should come from primary period, 0% slippage."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        primary = ConfirmationResult(
            sharpe=2.0, total_return=0.25, max_dd=-0.03,
            trades=30, win_rate=0.65, profit_factor=1.8,
            expectancy=0.02, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=primary):
            result = batch_confirm(winner)

        assert result.sharpe == 2.0
        assert result.total_return == 0.25
        assert result.max_dd == -0.03
        assert result.trades == 30
        assert result.win_rate == 0.65
        assert result.profit_factor == 1.8
        assert result.expectancy == 0.02

    def test_notes_contain_pass_fail_status(self):
        """Each result's note should show PASS or FAIL."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result):
            result = batch_confirm(winner)

        # Should have PASS notes
        assert any("PASS" in n for n in result.notes)

    def test_notes_contain_fail_for_failed_results(self):
        """Failed results should show FAIL in notes."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=0.2, total_return=0.01, max_dd=-0.15,
            trades=5, win_rate=0.4, profit_factor=0.8,
            expectancy=-0.01, passed=False,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result):
            result = batch_confirm(winner)

        assert any("FAIL" in n for n in result.notes)

    def test_all_data_load_failures(self):
        """All periods fail to load → FAILED verdict."""
        winner = {"strategy": "test", "ticker": "BAD", "params": {}}

        with patch("crabquant.confirm.batch.load_data", side_effect=Exception("No data")):
            result = batch_confirm(winner)

        assert result.verdict == "FAILED"
        assert result.passed is False
        # Should have notes about missing data
        assert any("No data" in n for n in result.notes)

    def test_exact_100_bars_passes_threshold(self):
        """DataFrame with exactly 100 bars should be accepted."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=100)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result):
            result = batch_confirm(winner)

        assert result.verdict == "ROBUST"

    def test_99_bars_fails_threshold(self):
        """DataFrame with 99 bars should be rejected (< 100)."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=99)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df):
            result = batch_confirm(winner)

        assert result.verdict == "FAILED"
        assert any("Insufficient data" in n for n in result.notes)

    def test_mixed_data_load_some_fail(self):
        """Some periods load OK, others fail → should still produce a verdict."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        good_df = MagicMock()
        good_df.__len__ = MagicMock(return_value=500)
        good_df.copy = MagicMock(return_value=good_df)

        call_count = [0]
        def alternating_data(ticker, period):
            call_count[0] += 1
            if call_count[0] == 1:
                return good_df
            raise Exception("No data")

        with patch("crabquant.confirm.batch.load_data", side_effect=alternating_data), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result):
            result = batch_confirm(winner)

        # First period passes, rest fail → not enough for robust (need ≥2 periods)
        assert result.verdict == "FAILED"

    def test_df_copy_called_for_each_slippage_level(self):
        """df.copy() should be called for each slippage level to avoid mutation."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df) as mock_load, \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result) as mock_confirm:
            batch_confirm(winner, n_periods=1)

        # copy() should be called 3 times (once per slippage level)
        assert mock_df.copy.call_count == 3

    def test_run_confirmation_receives_correct_slippage(self):
        """run_confirmation should be called with correct slippage_pct values."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result) as mock_confirm:
            batch_confirm(winner, n_periods=1)

        slippage_values = [c.kwargs["slippage_pct"] for c in mock_confirm.call_args_list]
        assert slippage_values == [0.0, 0.001, 0.002]

    def test_run_confirmation_receives_strategy_and_ticker(self):
        """run_confirmation should receive strategy name and ticker from winner."""
        winner = {"strategy": "my_strategy", "ticker": "AAPL", "params": {"fast": 10}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result) as mock_confirm:
            batch_confirm(winner, n_periods=1)

        first_call = mock_confirm.call_args_list[0]
        assert first_call.args[0] == "my_strategy"
        assert first_call.args[1] == "AAPL"
        assert first_call.args[2] == {"fast": 10}

    def test_n_periods_two(self):
        """n_periods=2 should load data for exactly 2 periods."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df) as mock_load, \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result) as mock_confirm:
            batch_confirm(winner, n_periods=2)

        assert mock_load.call_count == 2
        assert mock_confirm.call_count == 6  # 2 periods × 3 slippage levels

    def test_load_data_called_with_correct_period(self):
        """load_data should be called with correct period strings."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=20, win_rate=0.55, profit_factor=1.3,
            expectancy=0.01, passed=True,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df) as mock_load, \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result):
            batch_confirm(winner)

        periods_called = [c.kwargs["period"] for c in mock_load.call_args_list]
        assert periods_called == ["2y", "1y", "6mo"]

    def test_no_trades_in_any_result(self):
        """If no result has trades, any_trades should be False (doesn't affect verdict)."""
        winner = {"strategy": "test", "ticker": "SPY", "params": {}}

        mock_result = ConfirmationResult(
            sharpe=0.0, total_return=0.0, max_dd=0.0,
            trades=0, win_rate=0.0, profit_factor=0.0,
            expectancy=0.0, passed=False,
        )

        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=500)
        mock_df.copy = MagicMock(return_value=mock_df)

        with patch("crabquant.confirm.batch.load_data", return_value=mock_df), \
             patch("crabquant.confirm.batch.run_confirmation", return_value=mock_result):
            result = batch_confirm(winner)

        assert result.trades == 0
        assert result.verdict == "FAILED"

    def test_default_confirmation_result_when_empty(self):
        """Default ConfirmationResult should have expected defaults."""
        r = ConfirmationResult()
        assert r.sharpe == 0.0
        assert r.total_return == 0.0
        assert r.max_dd == 0.0
        assert r.trades == 0
        assert r.passed is False
        assert r.verdict == "FAILED"
        assert r.notes == []
