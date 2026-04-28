"""Tests for regime-aware scanner (5.5.6)."""

import pytest
from unittest.mock import patch, MagicMock

from crabquant.production.regime_scanner import (
    scan_regime_strategies,
    scan_production_by_regime,
)
from crabquant.regime import MarketRegime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dict_entry(name, regime_sharpes, preferred=None, weak=None, is_specific=True):
    return {
        "generate_signals": MagicMock(),
        "params": {},
        "regime_sharpes": regime_sharpes,
        "preferred_regimes": preferred or [],
        "weak_regimes": weak or [],
        "is_regime_specific": is_specific,
        "description": f"{name} strategy",
    }


def _make_tuple_entry():
    return (MagicMock(), {}, None, "legacy strategy")


# ---------------------------------------------------------------------------
# scan_regime_strategies
# ---------------------------------------------------------------------------

class TestScanRegimeStrategies:

    def test_empty_registry(self):
        result = scan_regime_strategies({}, current_regime="trending_up")
        assert result["regime"] == "trending_up"
        assert result["strategies"] == []
        assert result["matched_count"] == 0
        assert result["total_registry"] == 0

    def test_skips_weak_regime_strategies(self):
        registry = {
            "bear_killer": _make_dict_entry(
                "bear_killer",
                {"trending_up": -0.5, "trending_down": 2.0},
                preferred=["trending_down"],
                weak=["trending_up"],
            ),
        }
        result = scan_regime_strategies(registry, current_regime="trending_up")
        assert result["matched_count"] == 0
        assert "bear_killer" in result["skipped_regime_specific"]

    def test_includes_good_regime_strategies(self):
        registry = {
            "trend_follower": _make_dict_entry(
                "trend_follower",
                {"trending_up": 1.8, "trending_down": -0.3},
                preferred=["trending_up"],
                weak=["trending_down"],
            ),
        }
        result = scan_regime_strategies(registry, current_regime="trending_up")
        assert result["matched_count"] == 1
        assert result["strategies"][0]["name"] == "trend_follower"
        assert result["strategies"][0]["regime_sharpe"] == 1.8

    def test_generalist_included_without_high_sharpe(self):
        """Non-regime-specific strategies (generalists) included even with low Sharpe."""
        registry = {
            "generalist": _make_dict_entry(
                "generalist",
                {"trending_up": 0.2, "trending_down": 0.1},
                is_specific=False,
            ),
        }
        result = scan_regime_strategies(registry, current_regime="trending_up")
        assert result["matched_count"] == 1

    def test_legacy_tuple_skipped(self):
        registry = {
            "legacy_strat": _make_tuple_entry(),
        }
        result = scan_regime_strategies(registry, current_regime="trending_up")
        assert result["matched_count"] == 0
        assert "legacy_strat" in result["skipped_no_data"]

    def test_max_strategies_cap(self):
        registry = {
            f"strat_{i}": _make_dict_entry(
                f"strat_{i}",
                {"trending_up": float(i + 1)},
                is_specific=True,
            )
            for i in range(10)
        }
        result = scan_regime_strategies(
            registry, current_regime="trending_up", max_strategies=3
        )
        assert result["matched_count"] == 3
        # Should be the top 3 by Sharpe
        assert result["strategies"][0]["name"] == "strat_9"

    def test_sorted_by_regime_sharpe_desc(self):
        registry = {
            "low": _make_dict_entry("low", {"trending_up": 0.5}),
            "high": _make_dict_entry("high", {"trending_up": 2.0}),
            "mid": _make_dict_entry("mid", {"trending_up": 1.0}),
        }
        result = scan_regime_strategies(registry, current_regime="trending_up")
        names = [s["name"] for s in result["strategies"]]
        assert names == ["high", "mid", "low"]

    def test_min_regime_sharpe_filter(self):
        registry = {
            "below": _make_dict_entry("below", {"trending_up": 0.2}),
            "above": _make_dict_entry("above", {"trending_up": 0.8}),
        }
        result = scan_regime_strategies(
            registry, current_regime="trending_up", min_regime_sharpe=0.5
        )
        assert result["matched_count"] == 1
        assert result["strategies"][0]["name"] == "above"

    def test_override_regime_with_enum(self):
        result = scan_regime_strategies(
            {}, current_regime=MarketRegime.HIGH_VOLATILITY
        )
        assert result["regime"] == "high_volatility"
        assert result["regime_confidence"] == 1.0


class TestScanProductionByRegime:

    @patch("crabquant.production.get_production_strategies")
    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_empty_production(self, mock_detect, mock_get_prod):
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8})
        mock_get_prod.return_value = []

        result = scan_production_by_regime()
        assert result["matched_count"] == 0
        assert result["regime"] == "trending_up"

    @patch("crabquant.production.get_production_strategies")
    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_production_strategy_without_regime_data(self, mock_detect, mock_get_prod):
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8})
        mock_get_prod.return_value = [
            {
                "key": "some_strategy|SPY|abc123",
                "strategy_name": "some_strategy",
                "ticker": "SPY",
                "promoted_at": "2025-01-01",
            }
        ]

        result = scan_production_by_regime()
        # No regime data available — included by default
        assert result["matched_count"] == 1
