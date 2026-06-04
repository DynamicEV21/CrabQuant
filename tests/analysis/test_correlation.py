"""
Comprehensive tests for crabquant.analysis.correlation module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from crabquant.analysis.correlation import (
    analyze_strategy_pair,
    compute_signal_correlation,
    scan_registry_correlations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signals_a(length=100, seed=42):
    """Return a deterministic boolean-ish signal series."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.choice([0, 1], size=length), index=pd.date_range("2024-01-01", periods=length, freq="D"))


def _make_signals_b(length=100, seed=99):
    rng = np.random.RandomState(seed)
    return pd.Series(rng.choice([-1, 0, 1], size=length), index=pd.date_range("2024-01-01", periods=length, freq="D"))


def _make_df(length=100):
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {"close": 100 + rng.randn(length).cumsum()},
        index=pd.date_range("2024-01-01", periods=length, freq="D"),
    )


# ---------------------------------------------------------------------------
# compute_signal_correlation
# ---------------------------------------------------------------------------

class TestComputeSignalCorrelation:
    """Tests for the compute_signal_correlation() function."""

    def test_returns_expected_keys(self):
        df = _make_df()
        sa = _make_signals_a()
        sb = _make_signals_b()
        result = compute_signal_correlation(df, sa, sb)
        assert set(result.keys()) == {"pearson", "agreement_rate", "jaccard_long", "anti_correlated"}

    def test_pearson_range(self):
        df = _make_df()
        sa = _make_signals_a()
        sb = _make_signals_b()
        result = compute_signal_correlation(df, sa, sb)
        assert -1.0 <= result["pearson"] <= 1.0

    def test_agreement_rate_range(self):
        df = _make_df()
        sa = _make_signals_a()
        sb = _make_signals_b()
        result = compute_signal_correlation(df, sa, sb)
        assert 0.0 <= result["agreement_rate"] <= 1.0

    def test_jaccard_long_range(self):
        df = _make_df()
        sa = _make_signals_a()
        sb = _make_signals_b()
        result = compute_signal_correlation(df, sa, sb)
        assert 0.0 <= result["jaccard_long"] <= 1.0

    def test_identical_signals_perfect_correlation(self):
        df = _make_df()
        sa = _make_signals_a()
        result = compute_signal_correlation(df, sa, sa)
        assert result["pearson"] == 1.0
        assert result["agreement_rate"] == 1.0
        assert result["jaccard_long"] == 1.0

    def test_constant_signals_both_zero(self):
        df = _make_df()
        sa = pd.Series(0, index=pd.date_range("2024-01-01", periods=50, freq="D"))
        sb = pd.Series(0, index=pd.date_range("2024-01-01", periods=50, freq="D"))
        result = compute_signal_correlation(df, sa, sb)
        assert result["pearson"] == 1.0  # Both constant same value → 1.0
        assert result["agreement_rate"] == 1.0

    def test_constant_signals_different_values(self):
        df = _make_df()
        sa = pd.Series(1, index=pd.date_range("2024-01-01", periods=50, freq="D"))
        sb = pd.Series(0, index=pd.date_range("2024-01-01", periods=50, freq="D"))
        result = compute_signal_correlation(df, sa, sb)
        assert result["pearson"] == 0.0  # Both constant but different → 0.0

    def test_constant_one_signal(self):
        df = _make_df()
        sa = pd.Series(0, index=pd.date_range("2024-01-01", periods=50, freq="D"))
        sb = _make_signals_b(50)
        result = compute_signal_correlation(df, sa, sb)
        assert result["pearson"] == 0.0  # One constant, one varying → 0.0

    def test_single_element_signals(self):
        df = _make_df(1)
        sa = pd.Series([1], index=pd.date_range("2024-01-01", periods=1, freq="D"))
        sb = pd.Series([1], index=pd.date_range("2024-01-01", periods=1, freq="D"))
        result = compute_signal_correlation(df, sa, sb)
        assert result["pearson"] == 0.0  # len <= 1 → 0.0

    def test_empty_signals(self):
        df = _make_df()
        sa = pd.Series([], dtype=float)
        sb = pd.Series([], dtype=float)
        result = compute_signal_correlation(df, sa, sb)
        assert result["pearson"] == 0.0
        assert result["agreement_rate"] == 0.0
        assert result["jaccard_long"] == 0.0

    def test_nan_handling_in_signals(self):
        df = _make_df()
        vals = [1.0, 0.0, np.nan, 1.0, 0.0]
        sa = pd.Series(vals, index=pd.date_range("2024-01-01", periods=5, freq="D"))
        sb = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0], index=pd.date_range("2024-01-01", periods=5, freq="D"))
        result = compute_signal_correlation(df, sa, sb)
        # Should not raise, NaN filled with 0
        assert isinstance(result["pearson"], float)

    def test_anti_correlated_signals(self):
        df = _make_df()
        sa = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float,
                        index=pd.date_range("2024-01-01", periods=10, freq="D"))
        sb = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float,
                        index=pd.date_range("2024-01-01", periods=10, freq="D"))
        result = compute_signal_correlation(df, sa, sb)
        assert result["anti_correlated"] is True
        assert result["pearson"] < -0.3

    def test_not_anti_correlated(self):
        df = _make_df()
        sa = _make_signals_a(50)
        sb = _make_signals_a(50)  # identical
        result = compute_signal_correlation(df, sa, sb)
        assert result["anti_correlated"] is False

    def test_misaligned_indices_are_aligned(self):
        df = _make_df()
        sa = pd.Series([1.0, 0.0], index=pd.date_range("2024-01-01", periods=2, freq="D"))
        sb = pd.Series([0.0, 1.0], index=pd.date_range("2024-01-03", periods=2, freq="D"))
        result = compute_signal_correlation(df, sa, sb)
        # Union index: 4 days; first two have s2=0, last two have s1=0
        assert result["agreement_rate"] > 0  # At least some agreement on the union
        assert len(result) == 4  # 4 keys

    def test_negative_signals_agreement(self):
        df = _make_df()
        sa = pd.Series([-1, -1, 0, 0], dtype=float,
                        index=pd.date_range("2024-01-01", periods=4, freq="D"))
        sb = pd.Series([-1, 0, -1, 0], dtype=float,
                        index=pd.date_range("2024-01-01", periods=4, freq="D"))
        result = compute_signal_correlation(df, sa, sb)
        # Agreement: both <=0 everywhere, so agreement_rate should be 1.0
        assert result["agreement_rate"] == 1.0
        # Neither has "long" entries (s > 0), so jaccard_long = 0
        assert result["jaccard_long"] == 0.0

    def test_jaccard_long_with_disjoint_long_entries(self):
        df = _make_df()
        sa = pd.Series([1, 0, 0, 0], dtype=float,
                        index=pd.date_range("2024-01-01", periods=4, freq="D"))
        sb = pd.Series([0, 0, 1, 0], dtype=float,
                        index=pd.date_range("2024-01-01", periods=4, freq="D"))
        result = compute_signal_correlation(df, sa, sb)
        assert result["jaccard_long"] == 0.0  # No overlap

    def test_rounding(self):
        df = _make_df()
        sa = _make_signals_a()
        sb = _make_signals_b()
        result = compute_signal_correlation(df, sa, sb)
        # Check that values are rounded to 4 decimal places
        for key in ("pearson", "agreement_rate", "jaccard_long"):
            val = result[key]
            assert val == round(val, 4), f"{key} not rounded to 4 decimals"


# ---------------------------------------------------------------------------
# analyze_strategy_pair
# ---------------------------------------------------------------------------

class TestAnalyzeStrategyPair:
    """Tests for analyze_strategy_pair — patches the strategies module where it's imported."""

    @patch("crabquant.analysis.correlation.load_data")
    def test_returns_expected_keys(self, mock_load):
        df = _make_df()
        mock_load.return_value = df

        def gen_signals(data, params):
            return pd.Series(0, index=data.index, dtype=float)

        registry = {
            "strat_a": (gen_signals, {}),
            "strat_b": (gen_signals, {}),
        }

        with patch("crabquant.strategies.STRATEGY_REGISTRY", registry, create=True):
            result = analyze_strategy_pair("strat_a", "strat_b")

        assert "strategy_a" in result
        assert "strategy_b" in result
        assert "signal_correlation" in result
        assert "returns_correlation" in result
        assert "is_duplicate" in result

    @patch("crabquant.analysis.correlation.load_data")
    def test_constant_signals_zero_correlation(self, mock_load):
        df = _make_df()
        mock_load.return_value = df

        def gen_zero(data, params):
            return pd.Series(0, index=data.index, dtype=float)

        registry = {"strat_a": (gen_zero, {}), "strat_b": (gen_zero, {})}

        with patch("crabquant.strategies.STRATEGY_REGISTRY", registry, create=True):
            result = analyze_strategy_pair("strat_a", "strat_b")

        assert result["signal_correlation"]["pearson"] == 1.0  # Both constant 0 → same

    @patch("crabquant.analysis.correlation.load_data")
    def test_strategy_not_found_raises(self, mock_load):
        df = _make_df()
        mock_load.return_value = df

        registry = {"strat_a": (lambda d, p: pd.Series(0, index=d.index, dtype=float), {})}

        with patch("crabquant.strategies.STRATEGY_REGISTRY", registry, create=True):
            with pytest.raises(ValueError, match="not in registry"):
                analyze_strategy_pair("strat_a", "nonexistent")

    @patch("crabquant.analysis.correlation.load_data")
    def test_ticker_and_period_forwarded(self, mock_load):
        df = _make_df()
        mock_load.return_value = df

        def gen(data, params):
            return pd.Series(0, index=data.index, dtype=float)

        registry = {"s1": (gen, {}), "s2": (gen, {})}

        with patch("crabquant.strategies.STRATEGY_REGISTRY", registry, create=True):
            result = analyze_strategy_pair("s1", "s2", ticker="AAPL", period="3mo")

        assert result["ticker"] == "AAPL"
        assert result["period"] == "3mo"
        mock_load.assert_called_once_with("AAPL", period="3mo")

    @patch("crabquant.analysis.correlation.load_data")
    def test_is_duplicate_flag(self, mock_load):
        df = _make_df(100)
        mock_load.return_value = df

        # Use deterministic identical signals
        signal_vals = np.array([1, 0, 1, 0, 1] * 20, dtype=float)[:100]

        def gen_identical(data, params):
            return pd.Series(signal_vals, index=data.index)

        registry = {"s1": (gen_identical, {}), "s2": (gen_identical, {})}

        with patch("crabquant.strategies.STRATEGY_REGISTRY", registry, create=True):
            result = analyze_strategy_pair("s1", "s2")

        # Identical signals → pearson=1.0, jaccard=1.0
        assert result["signal_correlation"]["pearson"] > 0.8
        assert result["signal_correlation"]["jaccard_long"] > 0.7
        assert result["is_duplicate"] is True

    @patch("crabquant.analysis.correlation.load_data")
    def test_params_override(self, mock_load):
        df = _make_df()
        mock_load.return_value = df

        captured_params = {}

        def gen(data, params):
            captured_params["a"] = params
            return pd.Series(0, index=data.index, dtype=float)

        registry = {"s1": (gen, {"default": True}), "s2": (gen, {"default": True})}

        with patch("crabquant.strategies.STRATEGY_REGISTRY", registry, create=True):
            analyze_strategy_pair("s1", "s2", params_a={"override": True}, params_b={"override": True})

        assert captured_params["a"] == {"override": True}

    @patch("crabquant.analysis.correlation.load_data")
    def test_dict_style_registry_entry(self, mock_load):
        """Test that dict-style registry entries (with generate_signals key) work."""
        df = _make_df()
        mock_load.return_value = df

        def gen(data, params):
            return pd.Series(0, index=data.index, dtype=float)

        registry = {
            "s1": {"generate_signals": gen, "params": {"x": 1}},
            "s2": {"generate_signals": gen, "params": {"x": 2}},
        }

        with patch("crabquant.strategies.STRATEGY_REGISTRY", registry, create=True):
            result = analyze_strategy_pair("s1", "s2")

        assert result["strategy_a"] == "s1"
        assert result["strategy_b"] == "s2"


# ---------------------------------------------------------------------------
# scan_registry_correlations
# ---------------------------------------------------------------------------

class TestScanRegistryCorrelations:

    @patch("crabquant.analysis.correlation.analyze_strategy_pair")
    @patch("crabquant.analysis.correlation.load_data")
    def test_returns_expected_keys(self, mock_load, mock_pair):
        registry = {"s1": None, "s2": None, "s3": None}

        with patch("crabquant.analysis.correlation.STRATEGY_REGISTRY", registry, create=True):
            mock_pair.return_value = {
                "strategy_a": "s1", "strategy_b": "s2",
                "signal_correlation": {"pearson": 0.5, "jaccard_long": 0.3},
                "is_duplicate": False,
            }
            result = scan_registry_correlations(ticker="SPY", period="1mo", max_strategies=3)

        assert "total_analyzed" in result
        assert "total_pairs" in result
        assert "duplicates" in result
        assert "high_correlations" in result
        assert "all_results" in result

    @patch("crabquant.analysis.correlation.analyze_strategy_pair")
    @patch("crabquant.analysis.correlation.load_data")
    def test_pair_count(self, mock_load, mock_pair):
        registry = {"s1": None, "s2": None, "s3": None, "s4": None}

        with patch("crabquant.analysis.correlation.STRATEGY_REGISTRY", registry, create=True):
            mock_pair.return_value = {
                "strategy_a": "s1", "strategy_b": "s2",
                "signal_correlation": {"pearson": 0.1, "jaccard_long": 0.1},
                "is_duplicate": False,
            }
            result = scan_registry_correlations(max_strategies=4)

        # 4 choose 2 = 6
        assert result["total_pairs"] == 6
        assert result["total_analyzed"] == 4

    @patch("crabquant.analysis.correlation.analyze_strategy_pair")
    @patch("crabquant.analysis.correlation.load_data")
    def test_duplicates_detected(self, mock_load, mock_pair):
        registry = {"s1": None, "s2": None}

        with patch("crabquant.analysis.correlation.STRATEGY_REGISTRY", registry, create=True):
            mock_pair.return_value = {
                "strategy_a": "s1", "strategy_b": "s2",
                "signal_correlation": {"pearson": 0.9, "jaccard_long": 0.8},
                "is_duplicate": True,
            }
            result = scan_registry_correlations(max_strategies=2)

        assert len(result["duplicates"]) == 1
        assert result["duplicates"][0][2] == 0.9

    @patch("crabquant.analysis.correlation.analyze_strategy_pair")
    @patch("crabquant.analysis.correlation.load_data")
    def test_error_handling_skips_pair(self, mock_load, mock_pair):
        registry = {"s1": None, "s2": None}

        with patch("crabquant.analysis.correlation.STRATEGY_REGISTRY", registry, create=True):
            mock_pair.side_effect = RuntimeError("boom")
            result = scan_registry_correlations(max_strategies=2)

        # Should not crash, just skip
        assert result["total_pairs"] == 1
        assert len(result["all_results"]) == 0

    @patch("crabquant.analysis.correlation.analyze_strategy_pair")
    @patch("crabquant.analysis.correlation.load_data")
    def test_high_correlations_bucket(self, mock_load, mock_pair):
        registry = {"s1": None, "s2": None}

        with patch("crabquant.analysis.correlation.STRATEGY_REGISTRY", registry, create=True):
            mock_pair.return_value = {
                "strategy_a": "s1", "strategy_b": "s2",
                "signal_correlation": {"pearson": 0.65, "jaccard_long": 0.4},
                "is_duplicate": False,
            }
            result = scan_registry_correlations(max_strategies=2)

        assert len(result["high_correlations"]) == 1
        assert len(result["duplicates"]) == 0

    @patch("crabquant.analysis.correlation.analyze_strategy_pair")
    @patch("crabquant.analysis.correlation.load_data")
    def test_max_strategies_limits_analysis(self, mock_load, mock_pair):
        registry = {"s1": None, "s2": None, "s3": None, "s4": None, "s5": None}

        with patch("crabquant.analysis.correlation.STRATEGY_REGISTRY", registry, create=True):
            mock_pair.return_value = {
                "strategy_a": "s1", "strategy_b": "s2",
                "signal_correlation": {"pearson": 0.1, "jaccard_long": 0.1},
                "is_duplicate": False,
            }
            result = scan_registry_correlations(max_strategies=3)

        assert result["total_analyzed"] == 3
        assert result["total_pairs"] == 3  # 3 choose 2

    @patch("crabquant.analysis.correlation.analyze_strategy_pair")
    @patch("crabquant.analysis.correlation.load_data")
    def test_all_results_capped_at_50(self, mock_load, mock_pair):
        # Create 20 strategies → 190 pairs, but only top 50 returned
        registry = {f"s{i}": None for i in range(20)}

        with patch("crabquant.analysis.correlation.STRATEGY_REGISTRY", registry, create=True):
            mock_pair.return_value = {
                "strategy_a": "s0", "strategy_b": "s1",
                "signal_correlation": {"pearson": 0.5, "jaccard_long": 0.3},
                "is_duplicate": False,
            }
            result = scan_registry_correlations(max_strategies=20)

        assert len(result["all_results"]) <= 50
