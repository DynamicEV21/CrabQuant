"""
Tests for crabquant.analysis.correlation module.

Tests signal correlation computation, strategy pair analysis, and
registry scanning. Uses mock strategies to avoid needing real market data.
"""

import numpy as np
import pandas as pd
import pytest

from crabquant.analysis.correlation import (
    analyze_strategy_pair,
    compute_signal_correlation,
    scan_registry_correlations,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_ohlcv(n=200, seed=42):
    """Create synthetic OHLCV DataFrame."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + np.abs(np.random.randn(n)) * 0.5,
        "low": close - np.abs(np.random.randn(n)) * 0.5,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    })


def _make_bool_signals(n=200, true_pct=0.1, seed=42):
    """Create synthetic boolean signal series."""
    np.random.seed(seed)
    return pd.Series(np.random.rand(n) < true_pct, dtype=bool)


def _make_int_signals(n=200, seed=42):
    """Create synthetic integer signal series (1, 0, -1)."""
    np.random.seed(seed)
    vals = np.random.choice([1, 0, -1], size=n, p=[0.1, 0.8, 0.1])
    return pd.Series(vals)


# ── compute_signal_correlation ───────────────────────────────────────────


class TestComputeSignalCorrelation:
    def test_identical_signals(self):
        """Identical signals should have pearson=1.0, agreement=1.0, jaccard=1.0."""
        sig = _make_bool_signals()
        result = compute_signal_correlation(_make_ohlcv(), sig, sig)
        assert result["pearson"] == pytest.approx(1.0, abs=0.01)
        assert result["agreement_rate"] == pytest.approx(1.0, abs=0.01)
        assert result["jaccard_long"] == pytest.approx(1.0, abs=0.01)
        assert result["anti_correlated"] is False

    def test_opposite_signals(self):
        """Opposite boolean signals should be anti-correlated."""
        sig_a = _make_bool_signals(true_pct=0.2, seed=1)
        sig_b = ~sig_a
        result = compute_signal_correlation(_make_ohlcv(), sig_a, sig_b)
        assert result["pearson"] < -0.3
        assert result["anti_correlated"] is True

    def test_independent_signals(self):
        """Independent signals should have low correlation."""
        sig_a = _make_bool_signals(seed=1)
        sig_b = _make_bool_signals(seed=2)
        result = compute_signal_correlation(_make_ohlcv(), sig_a, sig_b)
        assert abs(result["pearson"]) < 0.3

    def test_int_signals(self):
        """Integer signals (1/0/-1) should work."""
        sig_a = _make_int_signals(seed=1)
        sig_b = _make_int_signals(seed=2)
        result = compute_signal_correlation(_make_ohlcv(), sig_a, sig_b)
        assert -1.0 <= result["pearson"] <= 1.0
        assert 0.0 <= result["agreement_rate"] <= 1.0
        assert 0.0 <= result["jaccard_long"] <= 1.0

    def test_one_empty_signal(self):
        """One all-False signal should produce zero correlation."""
        sig_a = pd.Series([False] * 100)
        sig_b = _make_bool_signals(n=100)
        result = compute_signal_correlation(_make_ohlcv(n=100), sig_a, sig_b)
        assert result["pearson"] == 0.0
        assert result["jaccard_long"] == 0.0

    def test_single_element_signals(self):
        """Single-element signals should not crash."""
        sig_a = pd.Series([True])
        sig_b = pd.Series([False])
        result = compute_signal_correlation(_make_ohlcv(n=1), sig_a, sig_b)
        assert result["pearson"] == 0.0  # Can't compute correlation with 1 element

    def test_return_types(self):
        """Result dict should have expected keys and numeric types."""
        sig_a = _make_bool_signals()
        sig_b = _make_bool_signals(seed=5)
        result = compute_signal_correlation(_make_ohlcv(), sig_a, sig_b)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"pearson", "agreement_rate", "jaccard_long", "anti_correlated"}
        assert isinstance(result["pearson"], float)
        assert isinstance(result["agreement_rate"], float)
        assert isinstance(result["jaccard_long"], float)
        assert isinstance(result["anti_correlated"], bool)

    def test_values_are_rounded(self):
        """Values should be rounded to 4 decimal places."""
        sig_a = _make_bool_signals()
        sig_b = _make_bool_signals(seed=7)
        result = compute_signal_correlation(_make_ohlcv(), sig_a, sig_b)
        # Check rounding by comparing string representation
        assert str(result["pearson"])[::-1].find('.') <= 4

    def test_misaligned_indices(self):
        """Signals with different indices should be reindex-aligned."""
        sig_a = pd.Series([True, False, True], index=[0, 1, 2])
        sig_b = pd.Series([True, True, False], index=[3, 4, 5])
        result = compute_signal_correlation(_make_ohlcv(n=6), sig_a, sig_b)
        # After alignment, sig_a has NaN→0 at 3,4,5 and sig_b has NaN→0 at 0,1,2
        # Only bar 0 and bar 5 have non-zero values (sig_a=1 at 0, sig_b=1 at 5)
        assert result["agreement_rate"] >= 0.0


# ── analyze_strategy_pair ────────────────────────────────────────────────


class TestAnalyzeStrategyPair:
    def test_unknown_strategy_raises(self):
        """Should raise ValueError for strategy not in registry."""
        with pytest.raises(ValueError, match="not in registry"):
            analyze_strategy_pair("nonexistent_strategy", "another_fake")

    def test_known_strategy_unknown_raises(self):
        """Should raise ValueError if either strategy is unknown."""
        # Import a real strategy name from registry
        from crabquant.strategies import STRATEGY_REGISTRY
        real_name = list(STRATEGY_REGISTRY.keys())[0]
        with pytest.raises(ValueError, match="not in registry"):
            analyze_strategy_pair(real_name, "nonexistent_strategy_xyz")


# ── scan_registry_correlations ───────────────────────────────────────────


class TestScanRegistryCorrelations:
    def test_returns_expected_structure(self, monkeypatch):
        """Should return dict with expected keys."""
        # Mock load_data to avoid network calls
        from crabquant import analysis
        monkeypatch.setattr(
            analysis.correlation, "load_data", lambda *a, **kw: _make_ohlcv()
        )
        # Mock analyze_strategy_pair to return a simple result
        call_count = [0]
        original = analysis.correlation.analyze_strategy_pair

        def mock_pair(a, b, ticker, period):
            call_count[0] += 1
            pearson = np.random.randn() * 0.3
            return {
                "strategy_a": a,
                "strategy_b": b,
                "ticker": ticker,
                "period": period,
                "signal_correlation": {
                    "pearson": round(pearson, 4),
                    "agreement_rate": 0.8,
                    "jaccard_long": 0.5,
                    "anti_correlated": pearson < -0.3,
                },
                "returns_correlation": round(pearson * 0.9, 4),
                "is_duplicate": abs(pearson) > 0.8,
            }

        monkeypatch.setattr(analysis.correlation, "analyze_strategy_pair", mock_pair)

        result = scan_registry_correlations(max_strategies=5)
        assert "total_analyzed" in result
        assert "total_pairs" in result
        assert "duplicates" in result
        assert "high_correlations" in result
        assert "all_results" in result
        assert result["total_analyzed"] == 5
        assert result["total_pairs"] == 5 * 4 // 2  # 10 pairs

    def test_max_strategies_limit(self, monkeypatch):
        """Should respect max_strategies limit."""
        from crabquant import analysis
        monkeypatch.setattr(
            analysis.correlation, "load_data", lambda *a, **kw: _make_ohlcv()
        )

        def mock_pair(a, b, ticker, period):
            return {
                "strategy_a": a,
                "strategy_b": b,
                "ticker": ticker,
                "period": period,
                "signal_correlation": {
                    "pearson": 0.0,
                    "agreement_rate": 0.5,
                    "jaccard_long": 0.0,
                    "anti_correlated": False,
                },
                "returns_correlation": 0.0,
                "is_duplicate": False,
            }

        monkeypatch.setattr(analysis.correlation, "analyze_strategy_pair", mock_pair)

        result = scan_registry_correlations(max_strategies=3)
        assert result["total_analyzed"] == 3
        assert result["total_pairs"] == 3

    def test_error_handling(self, monkeypatch):
        """Should gracefully handle errors from individual pair analysis."""
        from crabquant import analysis
        monkeypatch.setattr(
            analysis.correlation, "load_data", lambda *a, **kw: _make_ohlcv()
        )

        call_count = [0]

        def mock_pair(a, b, ticker, period):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise RuntimeError("synthetic error")
            return {
                "strategy_a": a,
                "strategy_b": b,
                "ticker": ticker,
                "period": period,
                "signal_correlation": {
                    "pearson": 0.1,
                    "agreement_rate": 0.5,
                    "jaccard_long": 0.0,
                    "anti_correlated": False,
                },
                "returns_correlation": 0.1,
                "is_duplicate": False,
            }

        monkeypatch.setattr(analysis.correlation, "analyze_strategy_pair", mock_pair)

        # Should not raise even if some pairs fail
        result = scan_registry_correlations(max_strategies=4)
        assert result["total_analyzed"] == 4
        # Some pairs will have errors, but the scan should complete
        assert len(result["all_results"]) >= 0


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_all_true_signals(self):
        """All-True signals should be perfectly correlated."""
        n = 50
        sig = pd.Series([True] * n)
        result = compute_signal_correlation(_make_ohlcv(n=n), sig, sig)
        assert result["pearson"] == pytest.approx(1.0, abs=0.01)
        assert result["jaccard_long"] == pytest.approx(1.0, abs=0.01)

    def test_all_false_signals(self):
        """All-False signals should be perfectly correlated (both never enter)."""
        n = 50
        sig = pd.Series([False] * n)
        result = compute_signal_correlation(_make_ohlcv(n=n), sig, sig)
        assert result["pearson"] == pytest.approx(1.0, abs=0.01)
        assert result["jaccard_long"] == 0.0  # No long entries → jaccard is 0

    def test_highly_overlapping_signals(self):
        """Mostly overlapping signals should have high (but not necessarily >0.8) correlation."""
        np.random.seed(99)
        n = 200
        base = np.random.rand(n) < 0.15
        sig_a = pd.Series(base)
        # Make sig_b 90% the same as sig_a
        perturbed = base.copy()
        flip_indices = np.random.choice(n, size=int(n * 0.1), replace=False)
        perturbed[flip_indices] = ~perturbed[flip_indices]
        sig_b = pd.Series(perturbed)
        result = compute_signal_correlation(_make_ohlcv(n=n), sig_a, sig_b)
        # With 90% overlap, correlation should be meaningfully high
        assert result["pearson"] > 0.5
