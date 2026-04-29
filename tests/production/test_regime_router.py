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
