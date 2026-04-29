"""Tests for regime_tagger.py — compute preferred_regimes for strategies."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestRegimeTagger:
    """Test strategy regime tagging."""

    def test_compute_strategy_regime_tags_returns_expected_keys(self):
        """Result has all expected keys."""
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.num_trades = 10
        mock_result.sharpe = 1.5
        mock_engine.run.return_value = mock_result

        mock_df = pd.DataFrame({
            "close": [100 + i * 0.5 for i in range(300)],
            "high": [101 + i * 0.5 for i in range(300)],
            "low": [99 + i * 0.5 for i in range(300)],
            "open": [100 + i * 0.5 for i in range(300)],
            "volume": [1000000] * 300,
        }, index=pd.date_range("2023-01-01", periods=300))

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df), \
             patch("crabquant.refinement.regime_tagger.BacktestEngine", return_value=mock_engine):
            result = compute_strategy_regime_tags(mock_fn, {}, ticker="SPY")

        assert "preferred_regimes" in result
        assert "acceptable_regimes" in result
        assert "weak_regimes" in result
        assert "regime_sharpes" in result
        assert "is_regime_specific" in result

    def test_compute_strategy_regime_tags_empty_on_insufficient_data(self):
        """Returns empty result when data is too short."""
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")

        mock_df = pd.DataFrame({
            "close": [100] * 50,
            "high": [101] * 50,
            "low": [99] * 50,
            "open": [100] * 50,
            "volume": [1000000] * 50,
        }, index=pd.date_range("2023-01-01", periods=50))

        with patch("crabquant.refinement.regime_tagger.load_data", return_value=mock_df):
            result = compute_strategy_regime_tags(mock_fn, {}, ticker="SPY")

        assert result["preferred_regimes"] == []
        assert result["is_regime_specific"] is False

    def test_compute_strategy_regime_tags_handles_load_error(self):
        """Returns empty result when data loading fails."""
        from crabquant.refinement.regime_tagger import compute_strategy_regime_tags

        mock_fn = MagicMock(__name__="test_strategy")

        with patch("crabquant.refinement.regime_tagger.load_data", side_effect=Exception("API down")):
            result = compute_strategy_regime_tags(mock_fn, {}, ticker="BAD")

        assert result["preferred_regimes"] == []
        assert result["regime_sharpes"] == {}

    def test_get_regime_strategies_filters_by_regime(self):
        """Filters strategies by regime Sharpe threshold."""
        from crabquant.refinement.regime_tagger import get_regime_strategies

        registry = {
            "trend_strategy": {
                "fn": MagicMock(),
                "description": "Trend following",
                "regime_sharpes": {"trending_up": 1.5, "high_volatility": -0.3},
            },
            "range_strategy": {
                "fn": MagicMock(),
                "description": "Range bound",
                "regime_sharpes": {"range_bound": 1.2, "trending_up": 0.1},
            },
            "universal_strategy": {
                "fn": MagicMock(),
                "description": "Works everywhere",
                "regime_sharpes": {"trending_up": 2.0, "range_bound": 1.8, "high_volatility": 1.0},
            },
        }

        # Get strategies that work in trending_up
        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)

        names = [r["name"] for r in results]
        assert "trend_strategy" in names
        assert "universal_strategy" in names
        assert "range_strategy" not in names  # only 0.1 in trending_up

    def test_get_regime_strategies_handles_legacy_tuples(self):
        """Legacy tuple entries are included without regime filtering."""
        from crabquant.refinement.regime_tagger import get_regime_strategies

        registry = {
            "new_strategy": {
                "fn": MagicMock(),
                "description": "New format",
                "regime_sharpes": {"trending_up": 1.5},
            },
            "old_strategy": (
                MagicMock(),  # fn
                {},  # defaults
                {},  # grid
                "Old format strategy",  # description
                None,  # matrix_fn
            ),
        }

        results = get_regime_strategies("trending_up", registry, min_sharpe=0.5)

        # New strategy with known regime Sharpe comes first
        assert results[0]["name"] == "new_strategy"
        assert results[0]["regime_sharpe"] == 1.5
        # Old strategy has no regime data, included but sorted last
        assert results[1]["name"] == "old_strategy"
        assert results[1]["regime_sharpe"] is None
