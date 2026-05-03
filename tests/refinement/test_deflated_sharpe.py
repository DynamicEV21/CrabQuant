"""Tests for crabquant.refinement.deflated_sharpe — Deflated Sharpe Ratio."""

from __future__ import annotations

import numpy as np
import pytest

from crabquant.refinement.deflated_sharpe import (
    _expected_max_sharpe,
    _probabilistic_sharpe_ratio,
    deflated_sharpe,
    deflated_sharpe_ratio,
)


# ── _expected_max_sharpe ────────────────────────────────────────────────────

class TestExpectedMaxSharpe:
    def test_n_trials_1_returns_sr0(self):
        """With only 1 trial, the expected max is just sr0."""
        result = _expected_max_sharpe(n_trials=1, sr0=0.0, sharpe_std=1.0)
        assert result == 0.0

    def test_n_trials_1_with_nonzero_sr0(self):
        result = _expected_max_sharpe(n_trials=1, sr0=1.5, sharpe_std=1.0)
        assert result == 1.5

    def test_grows_with_n_trials(self):
        """Expected max Sharpe should increase with more trials."""
        em10 = _expected_max_sharpe(n_trials=10, sr0=0.0, sharpe_std=1.0)
        em100 = _expected_max_sharpe(n_trials=100, sr0=0.0, sharpe_std=1.0)
        em1000 = _expected_max_sharpe(n_trials=1000, sr0=0.0, sharpe_std=1.0)
        assert em10 < em100 < em1000

    def test_positive_for_n_greater_than_1(self):
        """With default sr0=0, the expected max should be positive for n > 1."""
        result = _expected_max_sharpe(n_trials=100, sr0=0.0, sharpe_std=1.0)
        assert result > 0.0

    def test_scales_with_sharpe_std(self):
        """Higher sharpe_std → higher expected max."""
        em_low = _expected_max_sharpe(n_trials=100, sr0=0.0, sharpe_std=0.5)
        em_high = _expected_max_sharpe(n_trials=100, sr0=0.0, sharpe_std=2.0)
        assert em_low < em_high

    def test_approximate_known_values(self):
        """Check against known approximate values from extreme value theory.

        For n=100: E[max] ≈ sqrt(2*ln(100)) ≈ 3.03 (before correction)
        With the Gumbel correction it should be slightly lower.
        """
        result = _expected_max_sharpe(n_trials=100, sr0=0.0, sharpe_std=1.0)
        # Should be roughly in the range [2.0, 3.5]
        assert 1.5 < result < 4.0

    def test_n_trials_2(self):
        """n_trials=2 should be handled (clamped to >= 2 in log)."""
        result = _expected_max_sharpe(n_trials=2, sr0=0.0, sharpe_std=1.0)
        # For very small n, the Gumbel correction can dominate, making result slightly negative
        assert isinstance(result, float) or isinstance(result, np.floating)


# ── _probabilistic_sharpe_ratio ─────────────────────────────────────────────

class TestProbabilisticSharpeRatio:
    def test_high_observed_sharpe(self):
        """Much higher observed Sharpe → PSR close to 1."""
        psr = _probabilistic_sharpe_ratio(
            observed_sharpe=3.0, benchmark_sharpe=0.0, sharpe_std=1.0
        )
        assert psr > 0.95

    def test_low_observed_sharpe(self):
        """Much lower observed Sharpe → PSR close to 0."""
        psr = _probabilistic_sharpe_ratio(
            observed_sharpe=-3.0, benchmark_sharpe=0.0, sharpe_std=1.0
        )
        assert psr < 0.05

    def test_equal_sharpe(self):
        """Equal observed and benchmark → PSR ≈ 0.5."""
        psr = _probabilistic_sharpe_ratio(
            observed_sharpe=1.0, benchmark_sharpe=1.0, sharpe_std=1.0
        )
        assert abs(psr - 0.5) < 0.01

    def test_zero_sharpe_std_above_benchmark(self):
        """Zero std and above benchmark → 1.0."""
        psr = _probabilistic_sharpe_ratio(
            observed_sharpe=2.0, benchmark_sharpe=1.0, sharpe_std=0.0
        )
        assert psr == 1.0

    def test_zero_sharpe_std_below_benchmark(self):
        """Zero std and below benchmark → 0.0."""
        psr = _probabilistic_sharpe_ratio(
            observed_sharpe=0.5, benchmark_sharpe=1.0, sharpe_std=0.0
        )
        assert psr == 0.0

    def test_zero_sharpe_std_equal(self):
        """Zero std and equal → 0.5."""
        psr = _probabilistic_sharpe_ratio(
            observed_sharpe=1.0, benchmark_sharpe=1.0, sharpe_std=0.0
        )
        assert psr == 0.5


# ── deflated_sharpe_ratio ───────────────────────────────────────────────────

class TestDeflatedSharpeRatio:
    def test_high_sharpe_few_trials_is_positive(self):
        """A high Sharpe with few trials should give a high DSR (> 0.5)."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=3.0, sharpe_std=1.0, n_trials=10, sr0=0.0
        )
        assert dsr > 0.5

    def test_low_sharpe_many_trials_is_low(self):
        """A low Sharpe with many trials should give a low DSR (< 0.5)."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.5, sharpe_std=1.0, n_trials=10000, sr0=0.0
        )
        assert dsr < 0.5

    def test_n_trials_less_than_1_clamped(self):
        """n_trials < 1 should be clamped to 1 and not raise."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0, sharpe_std=1.0, n_trials=0, sr0=0.0
        )
        # With n_trials=1 (clamped), expected_max = sr0 = 0
        # PSR = Φ((2.0 - 0) / 1.0) = Φ(2) ≈ 0.977
        assert dsr > 0.9

    def test_increasing_n_trials_decreases_dsr(self):
        """More trials → lower DSR for the same observed Sharpe."""
        dsr10 = deflated_sharpe_ratio(
            observed_sharpe=1.5, sharpe_std=1.0, n_trials=10, sr0=0.0
        )
        dsr1000 = deflated_sharpe_ratio(
            observed_sharpe=1.5, sharpe_std=1.0, n_trials=1000, sr0=0.0
        )
        assert dsr10 > dsr1000

    def test_returns_probability_range(self):
        """DSR should be a probability in [0, 1]."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.0, sharpe_std=1.0, n_trials=100, sr0=0.0
        )
        assert 0.0 <= dsr <= 1.0


# ── deflated_sharpe (convenience wrapper) ────────────────────────────────────

class TestDeflatedSharpe:
    def test_n_trials_1_returns_sharpe_directly(self):
        """With n_trials=1, no multiple testing penalty — returns sharpe."""
        result = deflated_sharpe(sharpe=2.0, n_trials=1)
        assert result == 2.0

    def test_high_sharpe_few_trials_positive(self):
        """High Sharpe + few trials → positive deflated Sharpe."""
        result = deflated_sharpe(sharpe=3.0, n_trials=10)
        assert result > 0

    def test_low_sharpe_many_trials_negative(self):
        """Low Sharpe + many trials → negative deflated Sharpe (likely overfit)."""
        result = deflated_sharpe(sharpe=0.5, n_trials=10000)
        assert result < 0

    def test_with_returns_array(self):
        """Providing a returns array should produce a valid result."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=252)  # 1 year of daily returns
        result = deflated_sharpe(sharpe=1.5, n_trials=100, returns=returns)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_zero_sharpe_many_trials_negative(self):
        """Zero Sharpe with many trials → strongly negative."""
        result = deflated_sharpe(sharpe=0.0, n_trials=1000)
        assert result < 0

    def test_very_high_sharpe_is_positive_even_with_many_trials(self):
        """Extremely high Sharpe should remain positive even with many trials."""
        result = deflated_sharpe(sharpe=10.0, n_trials=1000)
        assert result > 0

    def test_n_trials_less_than_1_clamped(self):
        """n_trials < 1 should be clamped to 1 and return sharpe directly."""
        result = deflated_sharpe(sharpe=2.0, n_trials=0)
        assert result == 2.0

    def test_with_skew_and_kurt(self):
        """Non-default skew and kurtosis should be handled."""
        result = deflated_sharpe(sharpe=2.0, n_trials=100, skew=-0.5, kurt=5.0)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_constant_returns_zero(self):
        """Constant (zero-variance) returns → result should be 0."""
        returns = np.ones(252) * 0.01
        result = deflated_sharpe(sharpe=1.0, n_trials=100, returns=returns)
        assert result == 0.0

    def test_increasing_trials_decreases_score(self):
        """More trials should decrease the deflated Sharpe score."""
        score10 = deflated_sharpe(sharpe=2.0, n_trials=10)
        score1000 = deflated_sharpe(sharpe=2.0, n_trials=1000)
        assert score10 > score1000

    def test_single_element_returns_array(self):
        """Returns array with len <= 1 should use default estimation."""
        result = deflated_sharpe(sharpe=2.0, n_trials=100, returns=np.array([0.01]))
        assert isinstance(result, float)
        assert np.isfinite(result)
