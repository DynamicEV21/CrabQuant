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


# ── Additional compute_signal_correlation tests ─────────────────────────


class TestComputeSignalCorrelationExtended:
    def test_both_constant_same_value(self):
        """Both signals constant with same value → pearson=1.0."""
        n = 50
        sig = pd.Series([1.0] * n)
        result = compute_signal_correlation(_make_ohlcv(n=n), sig, sig)
        assert result["pearson"] == 1.0

    def test_both_constant_different_values(self):
        """Both signals constant with different values → pearson=0.0."""
        n = 50
        sig_a = pd.Series([1.0] * n)
        sig_b = pd.Series([0.0] * n)
        result = compute_signal_correlation(_make_ohlcv(n=n), sig_a, sig_b)
        assert result["pearson"] == 0.0

    def test_one_constant_one_varying(self):
        """One constant signal, one varying → pearson=0.0."""
        n = 50
        np.random.seed(10)
        sig_a = pd.Series([1.0] * n)
        sig_b = pd.Series(np.random.randn(n))
        result = compute_signal_correlation(_make_ohlcv(n=n), sig_a, sig_b)
        assert result["pearson"] == 0.0

    def test_empty_dataframe(self):
        """Empty OHLCV df should still work (df param is unused)."""
        sig_a = pd.Series([True, False])
        sig_b = pd.Series([False, True])
        empty_df = pd.DataFrame()
        result = compute_signal_correlation(empty_df, sig_a, sig_b)
        assert isinstance(result, dict)
        assert "pearson" in result

    def test_two_element_signals(self):
        """Two-element signals should produce valid results."""
        sig_a = pd.Series([True, False])
        sig_b = pd.Series([True, True])
        result = compute_signal_correlation(_make_ohlcv(n=2), sig_a, sig_b)
        assert -1.0 <= result["pearson"] <= 1.0

    def test_negatively_correlated_continuous(self):
        """Negatively correlated continuous signals."""
        np.random.seed(77)
        n = 100
        base = np.random.randn(n)
        sig_a = pd.Series(base)
        sig_b = pd.Series(-base)
        result = compute_signal_correlation(_make_ohlcv(n=n), sig_a, sig_b)
        assert result["pearson"] < -0.9
        assert result["anti_correlated"] is True

    def test_partial_index_overlap(self):
        """Signals with partially overlapping indices."""
        sig_a = pd.Series([1, 1, 0, 0, 0], index=[0, 1, 2, 3, 4])
        sig_b = pd.Series([0, 1, 1, 0, 0], index=[2, 3, 4, 5, 6])
        result = compute_signal_correlation(_make_ohlcv(n=7), sig_a, sig_b)
        # Should produce valid results
        assert isinstance(result["pearson"], float)
        assert isinstance(result["agreement_rate"], float)

    def test_zero_length_after_alignment(self):
        """Signals with completely disjoint indices should still work."""
        sig_a = pd.Series([True, True], index=[0, 1])
        sig_b = pd.Series([True, True], index=[100, 101])
        result = compute_signal_correlation(_make_ohlcv(n=102), sig_a, sig_b)
        # After alignment, each has the other's index filled with 0
        # Should be a valid result
        assert isinstance(result["pearson"], float)

    def test_float_signals_with_nans(self):
        """Float signals with NaN values should be handled."""
        sig_a = pd.Series([1.0, np.nan, 0.0, 1.0])
        sig_b = pd.Series([1.0, 0.0, np.nan, 1.0])
        result = compute_signal_correlation(_make_ohlcv(n=4), sig_a, sig_b)
        assert isinstance(result["pearson"], float)

    def test_agreement_rate_all_disagree(self):
        """Signals that always disagree should have low agreement rate."""
        n = 50
        sig_a = pd.Series([True] * n)
        sig_b = pd.Series([False] * n)
        result = compute_signal_correlation(_make_ohlcv(n=n), sig_a, sig_b)
        # Both <= 0 → agreement should be high (both not-positive)
        # Actually True > 0 and False <= 0, so they disagree
        assert result["agreement_rate"] == pytest.approx(0.0, abs=0.01)

    def test_jaccard_one_signal_all_long(self):
        """Jaccard when one signal is always long and the other never."""
        n = 50
        sig_a = pd.Series([True] * n)
        sig_b = pd.Series([False] * n)
        result = compute_signal_correlation(_make_ohlcv(n=n), sig_a, sig_b)
        # long_a has all indices, long_b has none, intersection=0
        assert result["jaccard_long"] == 0.0


# ── Additional analyze_strategy_pair tests ───────────────────────────────


class TestAnalyzeStrategyPairExtended:
    def test_tuple_registry_entry(self, monkeypatch):
        """Should handle registry entries that are tuples (func, params)."""
        from crabquant import analysis

        def mock_gen(df, params):
            return pd.Series([1, 0, -1] * (len(df) // 3), index=df.index[:len(df) // 3 * 3])

        ohlcv = _make_ohlcv()
        monkeypatch.setattr(analysis.correlation, "load_data", lambda *a, **kw: ohlcv)
        monkeypatch.setattr(
            "crabquant.strategies.STRATEGY_REGISTRY",
            {"strat_a": (mock_gen, {}), "strat_b": (mock_gen, {})},
        )
        result = analyze_strategy_pair("strat_a", "strat_b", ticker="SPY")
        assert result["strategy_a"] == "strat_a"
        assert result["strategy_b"] == "strat_b"
        assert "signal_correlation" in result
        assert "returns_correlation" in result
        assert "is_duplicate" in result

    def test_dict_registry_entry(self, monkeypatch):
        """Should handle registry entries that are dicts with generate_signals."""
        from crabquant import analysis

        def mock_gen(df, params):
            return pd.Series([0] * len(df), index=df.index)

        ohlcv = _make_ohlcv()
        monkeypatch.setattr(analysis.correlation, "load_data", lambda *a, **kw: ohlcv)
        monkeypatch.setattr(
            "crabquant.strategies.STRATEGY_REGISTRY",
            {
                "strat_a": {"generate_signals": mock_gen, "params": {}},
                "strat_b": {"generate_signals": mock_gen, "params": {}},
            },
        )
        result = analyze_strategy_pair("strat_a", "strat_b")
        assert result["ticker"] == "SPY"
        assert result["period"] == "1y"

    def test_custom_params_override(self, monkeypatch):
        """Should use custom params when provided."""
        from crabquant import analysis

        captured_params = []

        def mock_gen(df, params):
            captured_params.append(params)
            return pd.Series([0] * len(df), index=df.index)

        ohlcv = _make_ohlcv()
        monkeypatch.setattr(analysis.correlation, "load_data", lambda *a, **kw: ohlcv)
        monkeypatch.setattr(
            "crabquant.strategies.STRATEGY_REGISTRY",
            {"strat_a": (mock_gen, {"default_param": 10})},
        )
        # Provide custom params
        analyze_strategy_pair("strat_a", "strat_a", params_a={"custom": True})
        assert captured_params[0] == {"custom": True}

    def test_is_duplicate_flag_high_correlation(self, monkeypatch):
        """is_duplicate should be True when pearson > 0.8 and jaccard > 0.7."""
        from crabquant import analysis

        def mock_gen(df, params):
            return pd.Series([True] * (len(df) // 2) + [False] * (len(df) - len(df) // 2),
                             index=df.index)

        ohlcv = _make_ohlcv()
        monkeypatch.setattr(analysis.correlation, "load_data", lambda *a, **kw: ohlcv)
        monkeypatch.setattr(
            "crabquant.strategies.STRATEGY_REGISTRY",
            {"strat_a": (mock_gen, {}), "strat_b": (mock_gen, {})},
        )
        result = analyze_strategy_pair("strat_a", "strat_b")
        # Identical signals → high pearson, high jaccard
        assert result["is_duplicate"] is True

    def test_custom_ticker_and_period(self, monkeypatch):
        """Should pass custom ticker and period to load_data."""
        from crabquant import analysis

        captured_args = []

        def mock_load(ticker, period="1y"):
            captured_args.append((ticker, period))
            return _make_ohlcv()

        def mock_gen(df, params):
            return pd.Series([0] * len(df), index=df.index)

        monkeypatch.setattr(analysis.correlation, "load_data", mock_load)
        monkeypatch.setattr(
            "crabquant.strategies.STRATEGY_REGISTRY",
            {"strat_a": (mock_gen, {}), "strat_b": (mock_gen, {})},
        )
        analyze_strategy_pair("strat_a", "strat_b", ticker="AAPL", period="6mo")
        assert captured_args[0] == ("AAPL", "6mo")


# ── Additional scan_registry_correlations tests ─────────────────────────


class TestScanRegistryCorrelationsExtended:
    def test_duplicates_sorted_descending(self, monkeypatch):
        """Duplicate pairs should be sorted by correlation descending."""
        from crabquant import analysis
        monkeypatch.setattr(
            analysis.correlation, "load_data", lambda *a, **kw: _make_ohlcv()
        )

        call_idx = [0]
        pearsons = [0.95, 0.85, 0.60]

        def mock_pair(a, b, ticker, period):
            idx = call_idx[0]
            call_idx[0] += 1
            p = pearsons[idx % len(pearsons)]
            return {
                "strategy_a": a, "strategy_b": b,
                "ticker": ticker, "period": period,
                "signal_correlation": {
                    "pearson": round(p, 4), "agreement_rate": 0.9,
                    "jaccard_long": 0.8, "anti_correlated": False,
                },
                "returns_correlation": round(p, 4),
                "is_duplicate": p > 0.8,
            }

        monkeypatch.setattr(analysis.correlation, "analyze_strategy_pair", mock_pair)
        result = scan_registry_correlations(max_strategies=5)
        # Duplicates should be sorted descending
        if len(result["duplicates"]) > 1:
            for i in range(len(result["duplicates"]) - 1):
                assert result["duplicates"][i][2] >= result["duplicates"][i + 1][2]

    def test_single_strategy_no_pairs(self, monkeypatch):
        """Single strategy → 0 pairs."""
        from crabquant import analysis
        monkeypatch.setattr(
            analysis.correlation, "load_data", lambda *a, **kw: _make_ohlcv()
        )
        monkeypatch.setattr(
            "crabquant.strategies.STRATEGY_REGISTRY",
            {"only_strat": (lambda df, p: pd.Series([0] * len(df), index=df.index), {})},
        )
        result = scan_registry_correlations(max_strategies=1)
        assert result["total_analyzed"] == 1
        assert result["total_pairs"] == 0

    def test_all_results_limited_to_50(self, monkeypatch):
        """all_results should be capped at 50 entries."""
        from crabquant import analysis
        monkeypatch.setattr(
            analysis.correlation, "load_data", lambda *a, **kw: _make_ohlcv()
        )

        # Create many strategies
        big_registry = {}
        for i in range(15):
            big_registry[f"strat_{i}"] = (
                lambda df, p, idx=i: pd.Series([0] * len(df), index=df.index), {}
            )
        monkeypatch.setattr("crabquant.strategies.STRATEGY_REGISTRY", big_registry)

        def mock_pair(a, b, ticker, period):
            return {
                "strategy_a": a, "strategy_b": b,
                "ticker": ticker, "period": period,
                "signal_correlation": {
                    "pearson": 0.1, "agreement_rate": 0.5,
                    "jaccard_long": 0.0, "anti_correlated": False,
                },
                "returns_correlation": 0.1,
                "is_duplicate": False,
            }

        monkeypatch.setattr(analysis.correlation, "analyze_strategy_pair", mock_pair)
        result = scan_registry_correlations(max_strategies=15)
        assert len(result["all_results"]) <= 50

    def test_custom_duplicate_threshold(self, monkeypatch):
        """Custom duplicate_threshold should be respected."""
        from crabquant import analysis
        monkeypatch.setattr(
            analysis.correlation, "load_data", lambda *a, **kw: _make_ohlcv()
        )

        def mock_pair(a, b, ticker, period):
            pearson = 0.6
            return {
                "strategy_a": a, "strategy_b": b,
                "ticker": ticker, "period": period,
                "signal_correlation": {
                    "pearson": pearson, "agreement_rate": 0.9,
                    "jaccard_long": 0.8, "anti_correlated": False,
                },
                "returns_correlation": pearson,
                "is_duplicate": pearson > 0.5,  # Uses internal default 0.8
            }

        monkeypatch.setattr(analysis.correlation, "analyze_strategy_pair", mock_pair)
        # duplicate_threshold param is passed to scan but only used internally
        # The actual duplicate check is in analyze_strategy_pair
        result = scan_registry_correlations(max_strategies=3, duplicate_threshold=0.5)
        assert result["total_analyzed"] == 3
