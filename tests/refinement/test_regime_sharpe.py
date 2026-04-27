"""Tests for crabquant.refinement.regime_sharpe — Phase 3."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from crabquant.refinement.regime_sharpe import (
    compute_regime_sharpe,
    is_regime_dependent,
    RegimeSharpeReport,
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
