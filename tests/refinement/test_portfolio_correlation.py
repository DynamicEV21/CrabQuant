"""Tests for crabquant.refinement.portfolio_correlation — Phase 3."""

import json
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


# ── load_winners_equity_curves ────────────────────────────────────────────────

class TestLoadWinnersEquityCurves:

    def test_loads_from_valid_json(self):
        curves = make_equity_curves(n_strategies=3)
        winners = make_winners_json(curves)

        with tempfile.TemporaryDirectory() as tmpdir:
            winners_path = Path(tmpdir) / "winners.json"
            winners_path.write_text(json.dumps(winners))

            # Since we can't generate real equity curves from winners.json
            # (that would require backtesting), we test that it handles
            # the case where equity data is not embedded
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
