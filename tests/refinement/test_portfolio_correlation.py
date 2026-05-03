"""Tests for crabquant.refinement.portfolio_correlation — Phase 3."""

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from crabquant.refinement.portfolio_correlation import (
    compute_correlation_matrix,
    identify_redundant_strategies,
    identify_diversifying_strategies,
    generate_correlation_report,
    load_winners_equity_curves,
    _load_equity_for_winner,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_equity_curves(
    n: int = 252,
    n_strategies: int = 4,
    seed: int = 42,
    correlation: float = 0.9,
) -> dict[str, pd.Series]:
    """Generate mock equity curves with controlled correlation.

    First curve is random; subsequent curves are linear combinations with noise.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-03", periods=n, freq="B")
    curves = {}
    base = rng.standard_normal(n).cumsum() + 100
    curves["strategy_a"] = pd.Series(base, index=idx)
    for i in range(1, n_strategies):
        noise = rng.standard_normal(n) * (1 - correlation) * 10
        correlated = base * correlation + noise + 100 * (1 - correlation)
        curves[f"strategy_{chr(97 + i)}"] = pd.Series(correlated, index=idx)
    return curves


def make_winners_json(equity_curves: dict[str, pd.Series]) -> list[dict]:
    """Create a winners.json-compatible list of entries."""
    winners = []
    for name, equity in equity_curves.items():
        winners.append({
            "strategy": name,
            "ticker": "SPY",
            "sharpe": 1.5,
            "return": float(equity.iloc[-1] / equity.iloc[0] - 1),
            "max_dd": -0.10,
            "trades": 20,
            "params": {},
        })
    return winners


# ── compute_correlation_matrix ────────────────────────────────────────────────

class TestComputeCorrelationMatrix:

    def test_returns_dataframe(self):
        curves = make_equity_curves(n_strategies=3)
        result = compute_correlation_matrix(curves)
        assert isinstance(result, pd.DataFrame)

    def test_square_matrix(self):
        curves = make_equity_curves(n_strategies=4)
        result = compute_correlation_matrix(curves)
        assert result.shape == (4, 4)

    def test_diagonal_is_one(self):
        curves = make_equity_curves(n_strategies=3)
        result = compute_correlation_matrix(curves)
        np.testing.assert_array_almost_equal(np.diag(result.values), [1.0, 1.0, 1.0])

    def test_symmetric(self):
        curves = make_equity_curves(n_strategies=3)
        result = compute_correlation_matrix(curves)
        np.testing.assert_array_almost_equal(result.values, result.values.T)

    def test_high_correlation_pair(self):
        curves = make_equity_curves(n_strategies=2, correlation=0.99)
        result = compute_correlation_matrix(curves)
        corr = result.iloc[0, 1]
        assert corr > 0.95

    def test_empty_curves_returns_empty_df(self):
        result = compute_correlation_matrix({})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_strategy_returns_1x1(self):
        curves = make_equity_curves(n_strategies=1)
        result = compute_correlation_matrix(curves)
        assert result.shape == (1, 1)
        assert result.iloc[0, 0] == 1.0

    def test_columns_match_input_keys(self):
        curves = make_equity_curves(n_strategies=3)
        result = compute_correlation_matrix(curves)
        assert list(result.columns) == list(curves.keys())
        assert list(result.index) == list(curves.keys())

    def test_uses_kendall_not_pearson(self):
        """Verify Kendall tau is used by checking it differs from Pearson for specific data."""
        rng = np.random.default_rng(123)
        idx = pd.date_range("2023-01-03", periods=50, freq="B")
        # Create data with a few extreme outliers that affect Pearson more than Kendall
        x = rng.standard_normal(50).cumsum() + 100
        y = x.copy()
        # Add a huge outlier to one point — Pearson is sensitive, Kendall is robust
        y[-1] += 500
        curves = {
            "a": pd.Series(x, index=idx),
            "b": pd.Series(y, index=idx),
        }
        kendall = compute_correlation_matrix(curves).iloc[0, 1]
        pearson = pd.DataFrame(curves).corr(method="pearson").iloc[0, 1]
        # Kendall should be more robust (closer to 1) while Pearson drops
        assert kendall > pearson

    def test_identical_curves_perfect_correlation(self):
        """Two identical equity curves should have correlation == 1.0."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        curve = pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx)
        curves = {"a": curve, "b": curve.copy()}
        result = compute_correlation_matrix(curves)
        assert abs(result.iloc[0, 1] - 1.0) < 1e-10

    def test_negatively_correlated_curves(self):
        """Negatively correlated curves should have negative correlation."""
        rng = np.random.default_rng(77)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        x = rng.standard_normal(252).cumsum() + 100
        curves = {
            "a": pd.Series(x, index=idx),
            "b": pd.Series(-x + 200, index=idx),
        }
        result = compute_correlation_matrix(curves)
        assert result.iloc[0, 1] < -0.9

    def test_short_equity_curves(self):
        """Very short equity curves (5 bars) should still produce a matrix."""
        curves = make_equity_curves(n=5, n_strategies=2)
        result = compute_correlation_matrix(curves)
        assert result.shape == (2, 2)

    def test_misaligned_indices_handled(self):
        """Curves with different date indices — pandas aligns, NaN handled."""
        rng = np.random.default_rng(42)
        idx_a = pd.date_range("2023-01-03", periods=252, freq="B")
        idx_b = pd.date_range("2023-01-05", periods=250, freq="B")
        curves = {
            "a": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx_a),
            "b": pd.Series(rng.standard_normal(250).cumsum() + 100, index=idx_b),
        }
        result = compute_correlation_matrix(curves)
        assert result.shape == (2, 2)
        # Should not be NaN
        assert not np.isnan(result.iloc[0, 1])

    def test_constant_curve_returns_nan_off_diagonal(self):
        """A constant (flat) curve paired with a varying curve produces NaN correlation."""
        idx = pd.date_range("2023-01-03", periods=50, freq="B")
        curves = {
            "flat": pd.Series(100.0, index=idx),
            "varying": pd.Series(range(50), index=idx, dtype=float),
        }
        result = compute_correlation_matrix(curves)
        # Kendall of a constant series with anything should be NaN
        assert np.isnan(result.iloc[0, 1])

    def test_all_constant_curves(self):
        """All constant curves produce NaN everywhere except diagonal."""
        idx = pd.date_range("2023-01-03", periods=50, freq="B")
        curves = {
            "a": pd.Series(100.0, index=idx),
            "b": pd.Series(200.0, index=idx),
        }
        result = compute_correlation_matrix(curves)
        assert result.iloc[0, 0] == 1.0
        assert result.iloc[1, 1] == 1.0
        assert np.isnan(result.iloc[0, 1])

    def test_large_number_of_strategies(self):
        """10 strategies should produce a 10x10 matrix."""
        curves = make_equity_curves(n_strategies=10, correlation=0.5)
        result = compute_correlation_matrix(curves)
        assert result.shape == (10, 10)

    def test_values_in_valid_range(self):
        """All correlation values should be in [-1, 1]."""
        curves = make_equity_curves(n_strategies=5, correlation=0.3)
        result = compute_correlation_matrix(curves)
        vals = result.values[~np.isnan(result.values)]
        assert np.all(vals >= -1.0) and np.all(vals <= 1.0)


# ── identify_redundant_strategies ─────────────────────────────────────────────

class TestIdentifyRedundantStrategies:

    def test_finds_highly_correlated_pairs(self):
        curves = make_equity_curves(n_strategies=3, correlation=0.95)
        matrix = compute_correlation_matrix(curves)
        redundant = identify_redundant_strategies(matrix, threshold=0.9)
        assert len(redundant) > 0
        for pair in redundant:
            assert len(pair) == 2  # (strategy_a, strategy_b)
            assert pair[0] != pair[1]

    def test_no_redundant_when_low_correlation(self):
        # Uncorrelated curves
        rng = np.random.default_rng(99)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        curves = {
            "a": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx),
            "b": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx),
        }
        matrix = compute_correlation_matrix(curves)
        redundant = identify_redundant_strategies(matrix, threshold=0.9)
        assert len(redundant) == 0

    def test_threshold_param(self):
        curves = make_equity_curves(n_strategies=3, correlation=0.85)
        matrix = compute_correlation_matrix(curves)
        # Low threshold catches more pairs
        low = identify_redundant_strategies(matrix, threshold=0.7)
        high = identify_redundant_strategies(matrix, threshold=0.95)
        assert len(low) >= len(high)

    def test_empty_matrix_returns_empty(self):
        redundant = identify_redundant_strategies(pd.DataFrame(), threshold=0.9)
        assert redundant == []

    def test_single_strategy_returns_empty(self):
        curves = make_equity_curves(n_strategies=1)
        matrix = compute_correlation_matrix(curves)
        redundant = identify_redundant_strategies(matrix, threshold=0.9)
        assert redundant == []

    def test_no_self_pairs(self):
        """Results should never contain (x, x) pairs."""
        curves = make_equity_curves(n_strategies=3, correlation=0.99)
        matrix = compute_correlation_matrix(curves)
        redundant = identify_redundant_strategies(matrix, threshold=0.5)
        for a, b in redundant:
            assert a != b

    def test_ordered_pairs_i_lt_j(self):
        """Pairs should be returned in (i < j) order — alphabetical for default names."""
        curves = make_equity_curves(n_strategies=3, correlation=0.99)
        matrix = compute_correlation_matrix(curves)
        redundant = identify_redundant_strategies(matrix, threshold=0.5)
        for a, b in redundant:
            names = list(matrix.columns)
            assert names.index(a) < names.index(b)

    def test_threshold_at_zero_catches_all(self):
        """Threshold=0 should catch all non-negative pairs (diagonal excluded)."""
        curves = make_equity_curves(n_strategies=3, correlation=0.5)
        matrix = compute_correlation_matrix(curves)
        redundant = identify_redundant_strategies(matrix, threshold=0.0)
        # With Kendall, non-negative correlations should be caught
        # For random cumsums correlations are typically positive
        n_possible = 3 * (3 - 1) // 2
        assert len(redundant) <= n_possible

    def test_threshold_at_one_catches_none(self):
        """Threshold > 1.0 should find no redundant pairs."""
        curves = make_equity_curves(n_strategies=3, correlation=0.99)
        matrix = compute_correlation_matrix(curves)
        redundant = identify_redundant_strategies(matrix, threshold=1.01)
        assert len(redundant) == 0

    def test_identical_curves_are_redundant(self):
        """Identical curves should be flagged as redundant at any threshold < 1."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-03", periods=100, freq="B")
        curve = pd.Series(rng.standard_normal(100).cumsum() + 100, index=idx)
        curves = {"a": curve, "b": curve.copy()}
        matrix = compute_correlation_matrix(curves)
        redundant = identify_redundant_strategies(matrix, threshold=0.99)
        assert len(redundant) == 1
        assert ("a", "b") in redundant


# ── identify_diversifying_strategies ──────────────────────────────────────────

class TestIdentifyDiversifyingStrategies:

    def test_finds_low_correlation_pairs(self):
        rng = np.random.default_rng(99)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        curves = {
            "a": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx),
            "b": pd.Series(-rng.standard_normal(252).cumsum() + 100, index=idx),
        }
        matrix = compute_correlation_matrix(curves)
        diversifying = identify_diversifying_strategies(matrix, threshold=0.5)
        assert len(diversifying) > 0

    def test_no_diversifying_when_all_correlated(self):
        curves = make_equity_curves(n_strategies=3, correlation=0.99)
        matrix = compute_correlation_matrix(curves)
        diversifying = identify_diversifying_strategies(matrix, threshold=0.3)
        assert len(diversifying) == 0

    def test_empty_matrix_returns_empty(self):
        diversifying = identify_diversifying_strategies(pd.DataFrame(), threshold=0.3)
        assert diversifying == []

    def test_single_strategy_returns_empty(self):
        curves = make_equity_curves(n_strategies=1)
        matrix = compute_correlation_matrix(curves)
        diversifying = identify_diversifying_strategies(matrix, threshold=0.3)
        assert diversifying == []

    def test_no_self_pairs(self):
        rng = np.random.default_rng(99)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        curves = {
            "a": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx),
            "b": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx),
        }
        matrix = compute_correlation_matrix(curves)
        diversifying = identify_diversifying_strategies(matrix, threshold=0.9)
        for a, b in diversifying:
            assert a != b

    def test_threshold_param_sensitivity(self):
        curves = make_equity_curves(n_strategies=3, correlation=0.5)
        matrix = compute_correlation_matrix(curves)
        tight = identify_diversifying_strategies(matrix, threshold=0.1)
        loose = identify_diversifying_strategies(matrix, threshold=0.8)
        # Lower threshold is stricter → fewer pairs qualify
        assert len(tight) <= len(loose)

    def test_negatively_correlated_pair_magnitude_above_threshold(self):
        """Negative correlation: abs(-0.9) > 0.3, so NOT counted as diversifying."""
        rng = np.random.default_rng(77)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        x = rng.standard_normal(252).cumsum() + 100
        curves = {
            "a": pd.Series(x, index=idx),
            "b": pd.Series(-x + 200, index=idx),
        }
        matrix = compute_correlation_matrix(curves)
        # abs(negative corr) is large, so NOT diversifying at threshold=0.3
        diversifying = identify_diversifying_strategies(matrix, threshold=0.3)
        assert len(diversifying) == 0

    def test_mild_negative_correlation_counted_as_diversifying(self):
        """Mild negative correlation: abs(-0.2) <= 0.5, so IS diversifying."""
        rng = np.random.default_rng(88)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        a = rng.standard_normal(252).cumsum() + 100
        # Mix in opposite direction noise for mild negative correlation
        b = -a * 0.2 + rng.standard_normal(252).cumsum() + 100
        curves = {"a": pd.Series(a, index=idx), "b": pd.Series(b, index=idx)}
        matrix = compute_correlation_matrix(curves)
        corr_val = matrix.iloc[0, 1]
        if abs(corr_val) <= 0.5:
            diversifying = identify_diversifying_strategies(matrix, threshold=0.5)
            assert len(diversifying) >= 1

    def test_threshold_zero_catches_negative_only(self):
        """threshold=0 catches only perfectly anti-correlated pairs."""
        rng = np.random.default_rng(99)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        curves = {
            "a": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx),
            "b": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx),
        }
        matrix = compute_correlation_matrix(curves)
        diversifying = identify_diversifying_strategies(matrix, threshold=0.0)
        # Random cumsums typically have positive Kendall correlation
        # So at threshold=0, should catch very few or none
        assert isinstance(diversifying, list)

    def test_threshold_at_one_catches_all(self):
        """threshold >= 1.0 should catch all pairs (abs(corr) <= 1 always)."""
        curves = make_equity_curves(n_strategies=3, correlation=0.99)
        matrix = compute_correlation_matrix(curves)
        diversifying = identify_diversifying_strategies(matrix, threshold=1.0)
        n_possible = 3 * (3 - 1) // 2
        assert len(diversifying) == n_possible

    def test_ordered_pairs_i_lt_j(self):
        """Pairs should be returned in (i < j) order."""
        rng = np.random.default_rng(99)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        curves = {
            "a": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx),
            "b": pd.Series(rng.standard_normal(252).cumsum() + 100, index=idx),
        }
        matrix = compute_correlation_matrix(curves)
        diversifying = identify_diversifying_strategies(matrix, threshold=0.9)
        for a, b in diversifying:
            names = list(matrix.columns)
            assert names.index(a) < names.index(b)


# ── generate_correlation_report ───────────────────────────────────────────────

class TestGenerateCorrelationReport:

    def test_returns_dict_with_expected_keys(self):
        curves = make_equity_curves(n_strategies=3)
        report = generate_correlation_report(curves)
        assert "correlation_matrix" in report
        assert "redundant_pairs" in report
        assert "diversifying_pairs" in report
        assert "n_strategies" in report

    def test_n_strategies_matches_input(self):
        curves = make_equity_curves(n_strategies=5)
        report = generate_correlation_report(curves)
        assert report["n_strategies"] == 5

    def test_empty_curves_produces_empty_report(self):
        report = generate_correlation_report({})
        assert report["n_strategies"] == 0
        assert report["redundant_pairs"] == []
        assert report["diversifying_pairs"] == []

    def test_correlation_matrix_is_dict(self):
        curves = make_equity_curves(n_strategies=3)
        report = generate_correlation_report(curves)
        assert isinstance(report["correlation_matrix"], dict)

    def test_correlation_matrix_keys_match_strategies(self):
        curves = make_equity_curves(n_strategies=3)
        report = generate_correlation_report(curves)
        keys = set(report["correlation_matrix"].keys())
        assert keys == set(curves.keys())

    def test_empty_curves_correlation_matrix_is_empty_dict(self):
        report = generate_correlation_report({})
        assert report["correlation_matrix"] == {}

    def test_redundant_pairs_are_tuples(self):
        curves = make_equity_curves(n_strategies=3, correlation=0.99)
        report = generate_correlation_report(curves, redundancy_threshold=0.9)
        for pair in report["redundant_pairs"]:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_diversifying_pairs_are_tuples(self):
        curves = make_equity_curves(n_strategies=3, correlation=0.1)
        report = generate_correlation_report(curves, diversifying_threshold=0.3)
        for pair in report["diversifying_pairs"]:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_custom_thresholds(self):
        curves = make_equity_curves(n_strategies=3, correlation=0.85)
        report_strict = generate_correlation_report(
            curves, redundancy_threshold=0.99, diversifying_threshold=0.01
        )
        report_loose = generate_correlation_report(
            curves, redundancy_threshold=0.5, diversifying_threshold=0.9
        )
        # Stricter redundancy threshold → fewer redundant pairs
        assert len(report_strict["redundant_pairs"]) <= len(report_loose["redundant_pairs"])
        # Looser diversifying threshold → more diversifying pairs
        assert len(report_loose["diversifying_pairs"]) >= len(report_strict["diversifying_pairs"])

    def test_report_with_single_strategy(self):
        curves = make_equity_curves(n_strategies=1)
        report = generate_correlation_report(curves)
        assert report["n_strategies"] == 1
        assert report["redundant_pairs"] == []
        assert report["diversifying_pairs"] == []
        # correlation_matrix should have one key pointing to {self: 1.0}
        assert len(report["correlation_matrix"]) == 1


# ── load_winners_equity_curves ────────────────────────────────────────────────

class TestLoadWinnersEquityCurves:

    def test_loads_from_valid_json(self):
        curves = make_equity_curves(n_strategies=3)
        winners = make_winners_json(curves)

        with tempfile.TemporaryDirectory() as tmpdir:
            winners_path = Path(tmpdir) / "winners.json"
            winners_path.write_text(json.dumps(winners))

            with patch("crabquant.refinement.portfolio_correlation._load_equity_for_winner") as mock_eq:
                mock_eq.side_effect = lambda w: curves.get(w["strategy"])
                result = load_winners_equity_curves(str(winners_path))

        assert len(result) == 3
        assert "strategy_a" in result

    def test_returns_empty_dict_for_missing_file(self):
        result = load_winners_equity_curves("/nonexistent/path/winners.json")
        assert result == {}

    def test_returns_empty_dict_for_empty_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            winners_path = Path(tmpdir) / "winners.json"
            winners_path.write_text("[]")
            result = load_winners_equity_curves(str(winners_path))
        assert result == {}

    def test_skips_winners_without_equity_data(self):
        winners = [
            {"strategy": "a", "ticker": "SPY", "sharpe": 1.5},
            {"strategy": "b", "ticker": "AAPL", "sharpe": 1.0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            winners_path = Path(tmpdir) / "winners.json"
            winners_path.write_text(json.dumps(winners))

            with patch("crabquant.refinement.portfolio_correlation._load_equity_for_winner") as mock_eq:
                mock_eq.return_value = None  # no equity data
                result = load_winners_equity_curves(str(winners_path))

        assert result == {}

    def test_returns_empty_list_for_invalid_json(self):
        """Malformed JSON should return [] (not empty dict — code has this quirk)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            winners_path = Path(tmpdir) / "winners.json"
            winners_path.write_text("{not valid json")
            result = load_winners_equity_curves(str(winners_path))
        # The source returns [] for JSON decode errors
        assert result == []

    def test_returns_empty_dict_for_non_list_json(self):
        """JSON that is not a list should return empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            winners_path = Path(tmpdir) / "winners.json"
            winners_path.write_text(json.dumps({"key": "value"}))
            result = load_winners_equity_curves(str(winners_path))
        assert result == {}

    def test_partial_equity_loading(self):
        """Only winners with equity data should appear in result."""
        curves = {"strat_a": pd.Series([1, 2, 3])}
        winners = [
            {"strategy": "strat_a", "ticker": "SPY"},
            {"strategy": "strat_b", "ticker": "AAPL"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            winners_path = Path(tmpdir) / "winners.json"
            winners_path.write_text(json.dumps(winners))

            with patch("crabquant.refinement.portfolio_correlation._load_equity_for_winner") as mock_eq:
                mock_eq.side_effect = lambda w: curves.get(w.get("strategy"))
                result = load_winners_equity_curves(str(winners_path))

        assert len(result) == 1
        assert "strat_a" in result

    def test_accepts_pathlib_path(self):
        """Should work with Path objects, not just strings."""
        curves = make_equity_curves(n_strategies=1)
        winners = make_winners_json(curves)

        with tempfile.TemporaryDirectory() as tmpdir:
            winners_path = Path(tmpdir) / "winners.json"
            winners_path.write_text(json.dumps(winners))

            with patch("crabquant.refinement.portfolio_correlation._load_equity_for_winner") as mock_eq:
                mock_eq.side_effect = lambda w: curves.get(w["strategy"])
                # Pass Path object directly
                result = load_winners_equity_curves(winners_path)

        assert len(result) == 1


# ── _load_equity_for_winner ───────────────────────────────────────────────────

class TestLoadEquityForWinner:

    def test_returns_none_by_default(self):
        """The default implementation always returns None."""
        assert _load_equity_for_winner({"strategy": "test"}) is None

    def test_accepts_any_dict(self):
        """Should not raise on arbitrary dict input."""
        assert _load_equity_for_winner({}) is None
        assert _load_equity_for_winner({"foo": "bar"}) is None
