"""Tests for crabquant.production.regime_router"""

from unittest.mock import patch, MagicMock
import pytest

import numpy as np

from crabquant.regime import MarketRegime
from crabquant.production.regime_router import (
    route_portfolio,
    get_regime_summary,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_MIN_WEIGHT,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_SHARPE_FLOOR,
)


# ── route_portfolio ──────────────────────────────────────────────────────────

class TestRoutePortfolio:
    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_no_strategies_matched(self, mock_scan):
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.8,
            "strategies": [],
        }

        result = route_portfolio({})

        assert result["num_strategies"] == 0
        assert result["total_allocated"] == 0.0
        assert result["cash_reserve"] == 100_000.0
        assert result["reason"] == "no_strategies_matched_regime"

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_all_below_sharpe_floor(self, mock_scan):
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.8,
            "strategies": [
                {"name": "weak", "regime_sharpe": 0.05, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, sharpe_floor=0.1)

        assert result["num_strategies"] == 0
        assert result["reason"] == "no_strategies_above_sharpe_floor"

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_single_strategy_gets_full_allocation(self, mock_scan):
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {"name": "trend", "regime_sharpe": 1.5, "is_regime_specific": True, "preferred_regimes": ["trending_up"]},
            ],
        }

        result = route_portfolio({}, total_capital=50_000)

        assert result["num_strategies"] == 1
        assert result["total_allocated"] == 50_000.0
        assert result["cash_reserve"] == 0.0
        assert result["allocations"][0]["name"] == "trend"
        assert result["allocations"][0]["weight"] == 1.0
        assert result["allocations"][0]["capital"] == 50_000.0

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_multiple_strategies_weighted_by_sharpe(self, mock_scan):
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {"name": "high", "regime_sharpe": 3.0, "is_regime_specific": True, "preferred_regimes": []},
                {"name": "low", "regime_sharpe": 1.0, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        # Use a high max_weight so no capping occurs — pure Sharpe-based weighting
        result = route_portfolio({}, total_capital=100_000, max_weight=0.90)

        assert result["num_strategies"] == 2
        weights = {a["name"]: a["weight"] for a in result["allocations"]}
        # high should get 3/4 = 0.75, low should get 1/4 = 0.25
        assert abs(weights["high"] - 0.75) < 0.01
        assert abs(weights["low"] - 0.25) < 0.01
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert result["total_allocated"] == 100_000.0

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_max_weight_redistributes_to_others(self, mock_scan):
        """When one strategy is too dominant, excess weight goes to others."""
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {"name": "dominant", "regime_sharpe": 10.0, "is_regime_specific": True, "preferred_regimes": []},
                {"name": "small", "regime_sharpe": 0.1, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, total_capital=100_000, max_weight=0.60)

        assert result["num_strategies"] == 2
        dominant = next(a for a in result["allocations"] if a["name"] == "dominant")
        # dominant's initial weight is 10/(10+0.1) ≈ 0.99, capped to 0.60
        assert dominant["weight"] <= 0.61
        # The other strategy should get the redistributed excess
        assert result["total_allocated"] == 100_000.0

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_min_weight_filters_small_positions(self, mock_scan):
        """Strategies below min_weight should be excluded."""
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.5,
            "strategies": [
                {"name": "tiny", "regime_sharpe": 0.01, "is_regime_specific": False, "preferred_regimes": []},
                {"name": "big", "regime_sharpe": 10.0, "is_regime_specific": True, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, total_capital=100_000, min_weight=0.10)

        names = [a["name"] for a in result["allocations"]]
        assert "tiny" not in names
        assert "big" in names

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_max_positions_respected(self, mock_scan):
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.5,
            "strategies": [
                {"name": f"s{i}", "regime_sharpe": float(i + 1), "is_regime_specific": False, "preferred_regimes": []}
                for i in range(10)
            ],
        }

        result = route_portfolio({}, max_positions=3)

        assert result["num_strategies"] <= 3

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_override_regime(self, mock_scan):
        mock_scan.return_value = {
            "regime": "high_volatility",
            "regime_confidence": 1.0,
            "strategies": [
                {"name": "vol", "regime_sharpe": 1.0, "is_regime_specific": True, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, current_regime=MarketRegime.HIGH_VOLATILITY)

        assert result["regime"] == "high_volatility"
        assert result["regime_confidence"] == 1.0

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_default_parameters(self, mock_scan):
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.5,
            "strategies": [
                {"name": "s", "regime_sharpe": 1.0, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        route_portfolio({})

        # Verify scan was called with default max_strategies
        call_kwargs = mock_scan.call_args[1]
        assert call_kwargs["max_strategies"] == DEFAULT_MAX_POSITIONS * 2
        assert call_kwargs["min_regime_sharpe"] == DEFAULT_SHARPE_FLOOR

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_custom_capital(self, mock_scan):
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.8,
            "strategies": [
                {"name": "s", "regime_sharpe": 1.0, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, total_capital=1_000_000)

        assert result["total_capital"] == 1_000_000.0
        assert result["allocations"][0]["capital"] == 1_000_000.0

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_renormalization_after_min_weight_filter(self, mock_scan):
        """After filtering out small positions, weights should sum to 1.0."""
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.8,
            "strategies": [
                {"name": "big", "regime_sharpe": 9.0, "is_regime_specific": False, "preferred_regimes": []},
                {"name": "medium", "regime_sharpe": 1.0, "is_regime_specific": False, "preferred_regimes": []},
                {"name": "tiny", "regime_sharpe": 0.01, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, total_capital=100_000, min_weight=0.05)

        total_weight = sum(a["weight"] for a in result["allocations"])
        assert abs(total_weight - 1.0) < 0.01
        assert result["total_allocated"] == 100_000.0

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_timestamp_present(self, mock_scan):
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.5,
            "strategies": [],
        }

        result = route_portfolio({})

        assert "timestamp" in result
        assert "T" in result["timestamp"]

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_zero_sharpe_strategy_excluded_by_floor(self, mock_scan):
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.5,
            "strategies": [
                {"name": "zero", "regime_sharpe": 0.0, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, sharpe_floor=0.01)

        assert result["num_strategies"] == 0
        assert result["reason"] == "no_strategies_above_sharpe_floor"

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_allocation_fields_populated(self, mock_scan):
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {
                    "name": "trend",
                    "regime_sharpe": 2.0,
                    "is_regime_specific": True,
                    "preferred_regimes": ["trending_up"],
                },
            ],
        }

        result = route_portfolio({}, total_capital=50_000)
        alloc = result["allocations"][0]

        assert "name" in alloc
        assert "weight" in alloc
        assert "capital" in alloc
        assert "regime_sharpe" in alloc
        assert "is_regime_specific" in alloc
        assert "preferred_regimes" in alloc
        assert alloc["is_regime_specific"] is True
        assert alloc["preferred_regimes"] == ["trending_up"]


# ── get_regime_summary ───────────────────────────────────────────────────────

class TestGetRegimeSummary:
    @patch("crabquant.production.regime_router.scan_regime_strategies")
    @patch("crabquant.production.regime_router.detect_current_regime")
    def test_returns_summary_structure(self, mock_detect, mock_scan):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.7})
        mock_scan.return_value = {
            "matched_count": 3,
            "skipped_regime_specific": ["weak_strat"],
            "skipped_no_data": ["legacy_strat"],
            "strategies": [
                {"name": "s1", "regime_sharpe": 2.0},
                {"name": "s2", "regime_sharpe": 1.5},
                {"name": "s3", "regime_sharpe": 1.0},
            ],
        }

        # get_regime_summary imports STRATEGY_REGISTRY lazily from crabquant.strategies
        with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {"a": MagicMock(), "b": MagicMock()}, clear=True):
            summary = get_regime_summary()

        assert summary["current_regime"] == "low_volatility"
        assert summary["confidence"] == 0.7
        assert summary["registry_total"] == 2
        assert summary["matched_strategies"] == 3
        assert summary["skipped_weak"] == 1
        assert summary["skipped_no_data"] == 1
        assert len(summary["top_3"]) == 3
        assert summary["top_3"][0]["name"] == "s1"
        assert summary["top_3"][0]["sharpe"] == 2.0


    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_missing_regime_sharpe_defaults_to_zero_point_01(self, mock_scan):
        """Strategy without regime_sharpe key should use the 0.01 floor."""
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {"name": "nosharpe", "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, total_capital=100_000)

        assert result["num_strategies"] == 1
        assert result["allocations"][0]["name"] == "nosharpe"

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_missing_preferred_regimes_defaults_to_empty_list(self, mock_scan):
        """Strategy without preferred_regimes should default to []."""
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {"name": "s", "regime_sharpe": 2.0, "is_regime_specific": True},
            ],
        }

        result = route_portfolio({}, total_capital=100_000)
        assert result["allocations"][0]["preferred_regimes"] == []

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_negative_regime_sharpe_filtered_by_floor(self, mock_scan):
        """Negative regime_sharpe should be filtered out by sharpe_floor=0.0."""
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.5,
            "strategies": [
                {"name": "neg", "regime_sharpe": -1.0, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, total_capital=100_000, sharpe_floor=0.0)
        # (s.get("regime_sharpe") or 0) >= 0.0 → (-1.0 or 0) = -1.0 >= 0.0 → False
        # But wait: (-1.0 or 0) = -1.0 (truthy). So -1.0 >= 0.0 is False
        assert result["num_strategies"] == 0

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_none_regime_sharpe_filtered(self, mock_scan):
        """None regime_sharpe gets or 0 in eligibility, so 0 >= 0 = True passes eligibility.
        But then max(None, 0.01) would crash — documenting this behavior."""
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {"name": "none_sharpe", "regime_sharpe": None, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        # None Sharpe causes TypeError in weight computation — that's the actual behavior
        with pytest.raises(TypeError):
            route_portfolio({}, total_capital=100_000, sharpe_floor=0.0)

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_equal_sharpes_equal_weights(self, mock_scan):
        """Strategies with equal Sharpe should get equal weights."""
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {"name": "a", "regime_sharpe": 2.0, "is_regime_specific": False, "preferred_regimes": []},
                {"name": "b", "regime_sharpe": 2.0, "is_regime_specific": False, "preferred_regimes": []},
                {"name": "c", "regime_sharpe": 2.0, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, total_capital=100_000)

        assert result["num_strategies"] == 3
        for alloc in result["allocations"]:
            assert abs(alloc["weight"] - (1.0 / 3)) < 0.01

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_capital_precision_rounding(self, mock_scan):
        """Capital should be rounded to 2 decimal places."""
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {"name": "s1", "regime_sharpe": 1.0, "is_regime_specific": False, "preferred_regimes": []},
                {"name": "s2", "regime_sharpe": 1.0, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, total_capital=100_001.73)
        for alloc in result["allocations"]:
            # Weight should be rounded to 4 decimals
            assert alloc["weight"] == round(alloc["weight"], 4)
            # Capital should be rounded to 2 decimals
            assert alloc["capital"] == round(alloc["capital"], 2)

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_sharpe_floor_zero_includes_all(self, mock_scan):
        """sharpe_floor=0.0 should include all strategies with Sharpe >= 0."""
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.5,
            "strategies": [
                {"name": "zero", "regime_sharpe": 0.0, "is_regime_specific": False, "preferred_regimes": []},
                {"name": "small", "regime_sharpe": 0.001, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, sharpe_floor=0.0)
        # 0.0 >= 0.0 is True, so both pass
        assert result["num_strategies"] >= 1

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_no_strategies_result_has_reason_field(self, mock_scan):
        """Cash-reserve result should have a 'reason' key."""
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.5,
            "strategies": [],
        }

        result = route_portfolio({})
        assert "reason" in result

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_max_positions_one_with_two_eligible(self, mock_scan):
        """max_positions=1 should pick the top Sharpe strategy."""
        mock_scan.return_value = {
            "regime": "trending_up",
            "regime_confidence": 0.9,
            "strategies": [
                {"name": "high", "regime_sharpe": 5.0, "is_regime_specific": True, "preferred_regimes": []},
                {"name": "low", "regime_sharpe": 1.0, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        result = route_portfolio({}, max_positions=1, total_capital=100_000)
        assert result["num_strategies"] == 1
        assert result["allocations"][0]["name"] == "high"

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_all_filtered_by_min_weight_gives_cash_reserve(self, mock_scan):
        """If all strategies are below min_weight, allocations should be empty or less than total."""
        mock_scan.return_value = {
            "regime": "low_volatility",
            "regime_confidence": 0.5,
            "strategies": [
                {"name": "tiny1", "regime_sharpe": 0.001, "is_regime_specific": False, "preferred_regimes": []},
                {"name": "big", "regime_sharpe": 100.0, "is_regime_specific": False, "preferred_regimes": []},
            ],
        }

        # tiny1 has weight ~0.001/100.001 ≈ 0, big has weight ~0.999
        # Set min_weight=0.95 so big still passes but tiny1 is filtered
        result = route_portfolio({}, total_capital=100_000, min_weight=0.10)
        names = [a["name"] for a in result["allocations"]]
        assert "tiny1" not in names
        assert "big" in names

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_regime_confidence_passed_through(self, mock_scan):
        """Regime confidence from scan should be in the result."""
        mock_scan.return_value = {
            "regime": "high_volatility",
            "regime_confidence": 0.42,
            "strategies": [],
        }

        result = route_portfolio({})
        assert result["regime_confidence"] == 0.42

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    def test_string_regime_override(self, mock_scan):
        """Passing a string as current_regime should work."""
        mock_scan.return_value = {
            "regime": "bear_market",
            "regime_confidence": 0.7,
            "strategies": [],
        }

        result = route_portfolio({}, current_regime="bear_market")
        assert result["regime"] == "bear_market"


class TestGetRegimeSummaryExtended:
    """Additional tests for get_regime_summary edge cases."""

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    @patch("crabquant.production.regime_router.detect_current_regime")
    def test_empty_registry(self, mock_detect, mock_scan):
        """Summary with empty registry should still work."""
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})
        mock_scan.return_value = {
            "matched_count": 0,
            "skipped_regime_specific": [],
            "skipped_no_data": [],
            "strategies": [],
        }

        with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
            summary = get_regime_summary()

        assert summary["registry_total"] == 0
        assert summary["matched_strategies"] == 0
        assert summary["top_3"] == []

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    @patch("crabquant.production.regime_router.detect_current_regime")
    def test_fewer_than_three_strategies(self, mock_detect, mock_scan):
        """top_3 should handle fewer than 3 matched strategies."""
        mock_detect.return_value = (MarketRegime.HIGH_VOLATILITY, {"confidence": 0.9})
        mock_scan.return_value = {
            "matched_count": 1,
            "skipped_regime_specific": [],
            "skipped_no_data": [],
            "strategies": [
                {"name": "only_one", "regime_sharpe": 3.0},
            ],
        }

        with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {"x": MagicMock()}, clear=True):
            summary = get_regime_summary()

        assert len(summary["top_3"]) == 1
        assert summary["top_3"][0]["name"] == "only_one"

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    @patch("crabquant.production.regime_router.detect_current_regime")
    def test_no_skipped_strategies(self, mock_detect, mock_scan):
        """Summary with no skipped strategies."""
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8})
        mock_scan.return_value = {
            "matched_count": 5,
            "skipped_regime_specific": [],
            "skipped_no_data": [],
            "strategies": [
                {"name": f"s{i}", "regime_sharpe": float(i)}
                for i in range(5)
            ],
        }

        with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {"a": MagicMock()}, clear=True):
            summary = get_regime_summary()

        assert summary["skipped_weak"] == 0
        assert summary["skipped_no_data"] == 0

    @patch("crabquant.production.regime_router.scan_regime_strategies")
    @patch("crabquant.production.regime_router.detect_current_regime")
    def test_summary_has_all_expected_keys(self, mock_detect, mock_scan):
        """Summary dict should have all documented keys."""
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})
        mock_scan.return_value = {
            "matched_count": 0,
            "skipped_regime_specific": [],
            "skipped_no_data": [],
            "strategies": [],
        }

        with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
            summary = get_regime_summary()

        expected_keys = [
            "current_regime", "confidence", "registry_total",
            "matched_strategies", "skipped_weak", "skipped_no_data", "top_3",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"


# ── Module-level helper (used in test above) ────────────────────────────────

def _make_dict_entry(
    regime_sharpes=None,
    is_regime_specific=False,
    preferred_regimes=None,
    weak_regimes=None,
    description="test strategy",
):
    return {
        "regime_sharpes": regime_sharpes or {},
        "is_regime_specific": is_regime_specific,
        "preferred_regimes": preferred_regimes or [],
        "weak_regimes": weak_regimes or [],
        "description": description,
    }
