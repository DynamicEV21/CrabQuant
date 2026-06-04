"""Tests for crabquant.refinement.regime_sharpe — Phase 3."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from crabquant.refinement.regime_sharpe import (
    compute_regime_sharpe,
    is_regime_dependent,
    RegimeSharpeReport,
    _extract_contiguous_segments,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_portfolio_mock(n: int = 504, seed: int = 42):
    """Mock portfolio with returns spanning 2 years."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-03", periods=n, freq="B")
    daily_returns = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    pf = MagicMock()
    pf.returns.return_value = daily_returns
    pf.equity.return_value = (1 + daily_returns).cumprod() * 10000
    return pf, idx


def make_regime_label_series(idx, pattern: str = "alternating") -> pd.Series:
    """Create a series of regime labels for each bar.

    Patterns:
        - 'alternating': bull/bear/bull/bear (contiguous blocks)
        - 'single': all one regime
        - 'three': bull/bear/sideways (contiguous blocks)
    """
    n = len(idx)
    if pattern == "single":
        return pd.Series(["bull"] * n, index=idx)
    elif pattern == "alternating":
        half = n // 2
        labels = ["bull"] * half + ["bear"] * (n - half)
        return pd.Series(labels, index=idx)
    elif pattern == "three":
        third = n // 3
        remainder = n - 2 * third
        labels = ["bull"] * third + ["bear"] * third + ["sideways"] * remainder
        return pd.Series(labels, index=idx)
    return pd.Series(["bull"] * n, index=idx)


# ── compute_regime_sharpe ─────────────────────────────────────────────────────

class TestComputeRegimeSharpe:

    def test_returns_regime_sharpe_report(self):
        pf, idx = make_portfolio_mock()
        regime_labels = make_regime_label_series(idx, "three")

        report = compute_regime_sharpe(pf, regime_labels)

        assert isinstance(report, RegimeSharpeReport)
        assert isinstance(report.sharpe_by_regime, dict)
        assert isinstance(report.regime_segments, list)

    def test_sharpe_by_regime_has_expected_keys(self):
        pf, idx = make_portfolio_mock()
        regime_labels = make_regime_label_series(idx, "three")

        report = compute_regime_sharpe(pf, regime_labels)

        for regime in ["bull", "bear", "sideways"]:
            assert regime in report.sharpe_by_regime

    def test_sharpe_by_regime_values_are_floats(self):
        pf, idx = make_portfolio_mock()
        regime_labels = make_regime_label_series(idx, "alternating")

        report = compute_regime_sharpe(pf, regime_labels)

        for v in report.sharpe_by_regime.values():
            assert isinstance(v, float)

    def test_regime_segments_list_with_details(self):
        pf, idx = make_portfolio_mock()
        regime_labels = make_regime_label_series(idx, "alternating")

        report = compute_regime_sharpe(pf, regime_labels)

        for seg in report.regime_segments:
            assert "regime" in seg
            assert "start" in seg
            assert "end" in seg
            assert "sharpe" in seg
            assert "n_bars" in seg

    def test_single_regime_returns_one_segment(self):
        pf, idx = make_portfolio_mock()
        regime_labels = make_regime_label_series(idx, "single")

        report = compute_regime_sharpe(pf, regime_labels)

        assert len(report.regime_segments) == 1
        assert report.regime_segments[0]["regime"] == "bull"

    def test_none_portfolio_returns_empty_report(self):
        report = compute_regime_sharpe(None, pd.Series())
        assert isinstance(report, RegimeSharpeReport)
        assert report.sharpe_by_regime == {}
        assert report.regime_segments == []

    def test_segment_sharpe_uses_annualized_formula(self):
        """Verify Sharpe = mean/std * sqrt(252) within tolerance."""
        pf, idx = make_portfolio_mock(n=252, seed=123)
        regime_labels = make_regime_label_series(idx, "single")

        report = compute_regime_sharpe(pf, regime_labels)

        returns = pf.returns.return_value
        expected = returns.mean() / returns.std() * np.sqrt(252)
        actual = report.sharpe_by_regime["bull"]
        assert abs(actual - round(expected, 4)) < 0.01


# ── is_regime_dependent ───────────────────────────────────────────────────────

class TestIsRegimeDependent:

    def test_uniform_sharpe_not_dependent(self):
        """If all regimes have similar Sharpe, not dependent."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 1.5, "bear": 1.4, "sideways": 1.3},
            regime_segments=[],
        )
        assert is_regime_dependent(report) is False

    def test_widely_varying_sharpe_is_dependent(self):
        """If one regime has much higher Sharpe, strategy is regime-dependent."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 3.0, "bear": -1.0, "sideways": 0.2},
            regime_segments=[],
        )
        assert is_regime_dependent(report) is True

    def test_empty_report_not_dependent(self):
        report = RegimeSharpeReport(sharpe_by_regime={}, regime_segments=[])
        assert is_regime_dependent(report) is False

    def test_negative_in_any_regime_is_dependent(self):
        """If any regime has negative Sharpe, that's a dependency signal."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 1.8, "bear": -0.5},
            regime_segments=[],
        )
        assert is_regime_dependent(report) is True

    def test_threshold_param_respected(self):
        """Custom threshold should be respected."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 1.5, "bear": 0.5},
            regime_segments=[],
        )
        # Default threshold (2.0) → not dependent
        assert is_regime_dependent(report, threshold=2.0) is False
        # Lower threshold (0.8) → dependent
        assert is_regime_dependent(report, threshold=0.8) is True


# ── New tests: edge cases and uncovered paths ──────────────────────────────────


class TestComputeRegimeSharpeEdgeCases:
    """Additional edge case tests for compute_regime_sharpe."""

    def test_none_regime_labels_returns_empty(self):
        """None regime_labels should return empty report."""
        pf, _ = make_portfolio_mock()
        report = compute_regime_sharpe(pf, None)
        assert isinstance(report, RegimeSharpeReport)
        assert report.sharpe_by_regime == {}
        assert report.regime_segments == []

    def test_empty_regime_labels_returns_empty(self):
        """Empty regime_labels Series should return empty report."""
        pf, _ = make_portfolio_mock()
        report = compute_regime_sharpe(pf, pd.Series(dtype=str))
        assert report.sharpe_by_regime == {}
        assert report.regime_segments == []

    def test_portfolio_returns_raises_exception(self):
        """If portfolio.returns() raises, should return empty report."""
        pf = MagicMock()
        pf.returns.side_effect = RuntimeError("portfolio error")
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        regime_labels = pd.Series(["bull"] * 252, index=idx)

        report = compute_regime_sharpe(pf, regime_labels)
        assert report.sharpe_by_regime == {}

    def test_portfolio_returns_none(self):
        """If portfolio.returns() returns None, should return empty report."""
        pf = MagicMock()
        pf.returns.return_value = None
        report = compute_regime_sharpe(pf, pd.Series(["bull"] * 50, index=pd.date_range("2023-01-03", periods=50, freq="B")))
        assert report.sharpe_by_regime == {}

    def test_portfolio_returns_empty_series(self):
        """If portfolio.returns() returns empty Series, should return empty report."""
        pf = MagicMock()
        pf.returns.return_value = pd.Series(dtype=float)
        report = compute_regime_sharpe(pf, pd.Series(["bull"] * 50, index=pd.date_range("2023-01-03", periods=50, freq="B")))
        assert report.sharpe_by_regime == {}

    def test_misaligned_indices_returns_empty(self):
        """If regime_labels index doesn't overlap with returns index, return empty."""
        pf, _ = make_portfolio_mock(n=252)
        # Different date range
        regime_labels = pd.Series(
            ["bull"] * 252,
            index=pd.date_range("2020-01-03", periods=252, freq="B"),
        )
        report = compute_regime_sharpe(pf, regime_labels)
        assert report.sharpe_by_regime == {}

    def test_short_segment_skipped(self):
        """Segments with fewer than 10 bars should be skipped."""
        pf, idx = make_portfolio_mock(n=504, seed=42)
        # Make first regime very short (5 bars), second regime long
        labels = ["short"] * 5 + ["long"] * (len(idx) - 5)
        regime_labels = pd.Series(labels, index=idx)

        report = compute_regime_sharpe(pf, regime_labels)
        # "short" regime should not appear (only 5 bars < 10 minimum)
        assert "short" not in report.sharpe_by_regime
        assert "long" in report.sharpe_by_regime

    def test_zero_std_returns_zero_sharpe(self):
        """If returns are constant (zero std), Sharpe should be 0.0."""
        pf = MagicMock()
        idx = pd.date_range("2023-01-03", periods=100, freq="B")
        constant_returns = pd.Series([0.001] * 100, index=idx)
        pf.returns.return_value = constant_returns
        regime_labels = pd.Series(["bull"] * 100, index=idx)

        report = compute_regime_sharpe(pf, regime_labels)
        assert report.sharpe_by_regime["bull"] == 0.0

    def test_multiple_segments_same_regime(self):
        """Multiple segments of same regime should be averaged."""
        pf, idx = make_portfolio_mock(n=504, seed=99)
        # Create 4 segments: bull, bear, bull, bear
        quarter = len(idx) // 4
        remainder = len(idx) - 3 * quarter
        labels = (
            ["bull"] * quarter +
            ["bear"] * quarter +
            ["bull"] * quarter +
            ["bear"] * remainder
        )
        regime_labels = pd.Series(labels, index=idx)

        report = compute_regime_sharpe(pf, regime_labels)
        # Should have exactly 2 regimes (not 4 segments as separate regimes)
        assert set(report.sharpe_by_regime.keys()) == {"bull", "bear"}
        # Should have 4 detailed segments
        assert len(report.regime_segments) == 4

    def test_very_small_portfolio(self):
        """Small portfolio (20 bars) should work with single regime."""
        pf, idx = make_portfolio_mock(n=20, seed=7)
        regime_labels = pd.Series(["bull"] * 20, index=idx)
        report = compute_regime_sharpe(pf, regime_labels)
        # 20 bars >= 10 minimum, should get a result
        assert "bull" in report.sharpe_by_regime

    def test_exactly_10_bars_segment(self):
        """Segment of exactly 10 bars should be included (>=10, not >10)."""
        pf, idx = make_portfolio_mock(n=20, seed=7)
        regime_labels = pd.Series(
            ["bull"] * 10 + ["bear"] * 10,
            index=idx,
        )
        report = compute_regime_sharpe(pf, regime_labels)
        assert "bull" in report.sharpe_by_regime
        assert "bear" in report.sharpe_by_regime

    def test_nine_bars_segment_excluded(self):
        """Segment of exactly 9 bars should be excluded (<10 minimum)."""
        pf, idx = make_portfolio_mock(n=20, seed=7)
        regime_labels = pd.Series(
            ["short"] * 9 + ["long"] * 11,
            index=idx,
        )
        report = compute_regime_sharpe(pf, regime_labels)
        assert "short" not in report.sharpe_by_regime
        assert "long" in report.sharpe_by_regime


class TestIsRegimeDependentEdgeCases:
    """Additional edge case tests for is_regime_dependent."""

    def test_single_regime_not_dependent(self):
        """Only one regime → not enough data to determine dependency."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 2.5},
            regime_segments=[],
        )
        assert is_regime_dependent(report) is False

    def test_all_positive_close_sharpes_not_dependent(self):
        """All positive, close Sharpe values → not dependent."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 1.0, "bear": 1.1, "sideways": 0.9},
            regime_segments=[],
        )
        assert is_regime_dependent(report) is False

    def test_all_negative_sharpes_is_dependent(self):
        """All negative Sharpe → any(s < 0) → dependent."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": -0.5, "bear": -1.2},
            regime_segments=[],
        )
        assert is_regime_dependent(report) is True

    def test_one_negative_others_positive(self):
        """One negative regime among positives → dependent."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 2.0, "bear": -0.1, "sideways": 1.5},
            regime_segments=[],
        )
        assert is_regime_dependent(report) is True

    def test_threshold_exactly_at_boundary(self):
        """When range exactly equals threshold, NOT dependent (> not >=)."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 2.0, "bear": 0.0},
            regime_segments=[],
        )
        # Range = 2.0 - 0.0 = 2.0, threshold = 2.0, NOT > → not dependent
        assert is_regime_dependent(report, threshold=2.0) is False
        # Range = 2.0 - 0.0 = 2.0, threshold = 1.99, > → dependent
        assert is_regime_dependent(report, threshold=1.99) is True

    def test_zero_threshold_all_positive(self):
        """Threshold=0 means any range > 0 → dependent (unless all equal)."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 1.0, "bear": 0.99},
            regime_segments=[],
        )
        assert is_regime_dependent(report, threshold=0.0) is True

    def test_zero_threshold_all_equal(self):
        """Threshold=0 with equal Sharpes → range=0, not > 0 → not dependent."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 1.0, "bear": 1.0},
            regime_segments=[],
        )
        assert is_regime_dependent(report, threshold=0.0) is False

    def test_large_threshold_never_range_dependent(self):
        """Very large threshold → range never exceeds it."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 100.0, "bear": 0.0},
            regime_segments=[],
        )
        # Range = 100, but threshold = 1000 → not dependent via range
        # But bear Sharpe is 0.0 (not negative), so still not dependent
        assert is_regime_dependent(report, threshold=1000.0) is False

    def test_very_large_threshold_with_negative(self):
        """Even with huge threshold, negative Sharpe → dependent."""
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 100.0, "bear": -0.01},
            regime_segments=[],
        )
        assert is_regime_dependent(report, threshold=10000.0) is True


class TestExtractContiguousSegments:
    """Tests for the _extract_contiguous_segments helper."""

    def test_empty_series(self):
        result = _extract_contiguous_segments(pd.Series(dtype=str))
        assert result == []

    def test_single_element(self):
        s = pd.Series(["bull"])
        result = _extract_contiguous_segments(s)
        assert len(result) == 1
        assert result[0] == {"regime": "bull", "start": 0, "end": 1}

    def test_all_same_regime(self):
        s = pd.Series(["bear"] * 20)
        result = _extract_contiguous_segments(s)
        assert len(result) == 1
        assert result[0]["regime"] == "bear"
        assert result[0]["start"] == 0
        assert result[0]["end"] == 20

    def test_alternating_every_bar(self):
        """Every bar changes regime → each bar is its own segment."""
        s = pd.Series(["bull", "bear", "bull", "bear", "bull"])
        result = _extract_contiguous_segments(s)
        assert len(result) == 5

    def test_two_segments(self):
        s = pd.Series(["bull"] * 10 + ["bear"] * 10)
        result = _extract_contiguous_segments(s)
        assert len(result) == 2
        assert result[0] == {"regime": "bull", "start": 0, "end": 10}
        assert result[1] == {"regime": "bear", "start": 10, "end": 20}

    def test_three_segments(self):
        s = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 7)
        result = _extract_contiguous_segments(s)
        assert len(result) == 3
        assert result[0]["regime"] == "a"
        assert result[1]["regime"] == "b"
        assert result[2]["regime"] == "c"

    def test_segment_end_is_exclusive(self):
        """End index should be exclusive (Python slicing convention)."""
        s = pd.Series(["bull"] * 3 + ["bear"] * 2)
        result = _extract_contiguous_segments(s)
        assert result[0]["end"] == 3
        assert result[1]["start"] == 3


class TestRegimeSharpeReportDataclass:
    """Tests for RegimeSharpeReport dataclass."""

    def test_default_values(self):
        report = RegimeSharpeReport()
        assert report.sharpe_by_regime == {}
        assert report.regime_segments == []

    def test_to_dict(self):
        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 1.5},
            regime_segments=[{"regime": "bull", "start": 0, "end": 100}],
        )
        d = report.to_dict()
        assert d["sharpe_by_regime"] == {"bull": 1.5}
        assert len(d["regime_segments"]) == 1

    def test_sharpe_values_rounded(self):
        """Sharpe values in report should be rounded to 4 decimal places."""
        pf, idx = make_portfolio_mock(n=504, seed=42)
        regime_labels = make_regime_label_series(idx, "single")
        report = compute_regime_sharpe(pf, regime_labels)
        for sharpe in report.sharpe_by_regime.values():
            # Check that the value is rounded to 4 decimal places
            assert sharpe == round(sharpe, 4)
