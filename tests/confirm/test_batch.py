"""Tests for crabquant.confirm.batch module."""

from unittest.mock import patch, MagicMock

import pytest

from crabquant.confirm import ConfirmationResult
from crabquant.confirm.batch import (
    _aggregate_results,
    batch_confirm,
    PERIODS,
    SLIPPAGE_LEVELS,
)


def _make_result(passed=True, sharpe=1.5, trades=20):
    return ConfirmationResult(
        sharpe=sharpe, total_return=0.1, max_dd=-0.05,
        trades=trades, win_rate=0.55, profit_factor=1.3,
        expectancy=0.01, passed=passed,
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
