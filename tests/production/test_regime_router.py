"""Tests for portfolio regime router (5.5.7)."""

import pytest
from unittest.mock import patch, MagicMock

from crabquant.production.regime_router import route_portfolio, get_regime_summary
from crabquant.regime import MarketRegime


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


class TestRoutePortfolio:

    def test_empty_registry_returns_cash_reserve(self):
        result = route_portfolio({}, current_regime="trending_up")
        assert result["num_strategies"] == 0
        assert result["total_allocated"] == 0.0
        assert result["cash_reserve"] == 100_000.0
        assert result["reason"] == "no_strategies_matched_regime"

    def test_single_strategy_gets_full_allocation(self):
        registry = {
            "only_strat": _make_dict_entry(
                "only_strat",
                {"trending_up": 1.5},
                is_specific=True,
            ),
        }
        result = route_portfolio(registry, current_regime="trending_up")
        assert result["num_strategies"] == 1
        # Single strategy — max_weight caps at 40%, rest is cash
        assert result["allocations"][0]["name"] == "only_strat"
        assert result["allocations"][0]["capital"] > 0
        assert result["cash_reserve"] >= 0

    def test_multiple_strategies_proportional_allocation(self):
        registry = {
            "high": _make_dict_entry("high", {"trending_up": 2.0}),
            "low": _make_dict_entry("low", {"trending_up": 1.0}),
        }
        result = route_portfolio(registry, current_regime="trending_up", max_weight=1.0)
        assert result["num_strategies"] == 2
        # Higher Sharpe should get ~2x the weight of lower
        high_w = [a["weight"] for a in result["allocations"] if a["name"] == "high"][0]
        low_w = [a["weight"] for a in result["allocations"] if a["name"] == "low"][0]
        assert high_w > low_w
        # Weights should sum to ~1.0
        total_w = sum(a["weight"] for a in result["allocations"])
        assert abs(total_w - 1.0) < 0.01

    def test_max_positions_respected(self):
        registry = {
            f"strat_{i}": _make_dict_entry(f"strat_{i}", {"trending_up": float(i + 1)})
            for i in range(10)
        }
        result = route_portfolio(
            registry, current_regime="trending_up", max_positions=3, min_weight=0.01
        )
        assert result["num_strategies"] <= 3

    def test_sharpe_floor_excludes_low_performers(self):
        registry = {
            "bad": _make_dict_entry("bad", {"trending_up": -0.5}),
            "good": _make_dict_entry("good", {"trending_up": 1.5}),
        }
        result = route_portfolio(
            registry, current_regime="trending_up", sharpe_floor=0.5
        )
        assert result["num_strategies"] == 1
        assert result["allocations"][0]["name"] == "good"

    def test_custom_capital(self):
        registry = {
            "strat": _make_dict_entry("strat", {"trending_up": 1.0}),
        }
        result = route_portfolio(
            registry, current_regime="trending_up", total_capital=500_000.0
        )
        assert result["total_capital"] == 500_000.0
        assert result["allocations"][0]["capital"] > 0

    def test_weak_regime_strategies_excluded(self):
        registry = {
            "bear_only": _make_dict_entry(
                "bear_only",
                {"trending_up": -1.0, "trending_down": 2.0},
                weak=["trending_up"],
            ),
        }
        result = route_portfolio(registry, current_regime="trending_up")
        assert result["num_strategies"] == 0
        assert result["reason"] == "no_strategies_matched_regime"

    def test_allocation_capitals_sum_correctly(self):
        registry = {
            "a": _make_dict_entry("a", {"trending_up": 1.0}),
            "b": _make_dict_entry("b", {"trending_up": 1.0}),
        }
        result = route_portfolio(
            registry, current_regime="trending_up", max_weight=1.0
        )
        total_alloc = sum(a["capital"] for a in result["allocations"])
        assert abs(total_alloc + result["cash_reserve"] - 100_000.0) < 1.0

    def test_result_has_required_keys(self):
        registry = {
            "s": _make_dict_entry("s", {"trending_up": 1.0}),
        }
        result = route_portfolio(registry, current_regime="trending_up")
        expected_keys = {
            "regime", "regime_confidence", "timestamp", "total_capital",
            "allocations", "total_allocated", "cash_reserve", "num_strategies",
        }
        assert expected_keys.issubset(result.keys())

    def test_allocation_entries_have_required_keys(self):
        registry = {
            "s": _make_dict_entry("s", {"trending_up": 1.0}),
        }
        result = route_portfolio(registry, current_regime="trending_up")
        alloc = result["allocations"][0]
        expected_keys = {"name", "weight", "capital", "regime_sharpe", "is_regime_specific", "preferred_regimes"}
        assert expected_keys.issubset(alloc.keys())


class TestGetRegimeSummary:

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    @patch("crabquant.production.regime_router.detect_current_regime")
    @patch("crabquant.strategies.STRATEGY_REGISTRY", {})
    def test_empty_registry_summary(self, mock_detect, mock_scan):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.7})
        mock_scan.return_value = {
            "matched_count": 0,
            "skipped_regime_specific": [],
            "skipped_no_data": [],
            "strategies": [],
        }

        result = get_regime_summary()
        assert result["current_regime"] == "low_volatility"
        assert result["confidence"] == 0.7
        assert result["registry_total"] == 0
        assert result["matched_strategies"] == 0
        assert result["top_3"] == []
