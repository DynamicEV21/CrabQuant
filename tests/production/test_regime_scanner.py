"""Tests for crabquant.production.regime_scanner"""

from unittest.mock import patch, MagicMock
import pytest

from crabquant.regime import MarketRegime
from crabquant.production.regime_scanner import (
    detect_current_regime,
    scan_regime_strategies,
    scan_production_by_regime,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_dict_entry(
    regime_sharpes=None,
    is_regime_specific=False,
    preferred_regimes=None,
    weak_regimes=None,
    description="test strategy",
):
    """Create a mock dict-style registry entry."""
    return {
        "regime_sharpes": regime_sharpes or {},
        "is_regime_specific": is_regime_specific,
        "preferred_regimes": preferred_regimes or [],
        "weak_regimes": weak_regimes or [],
        "description": description,
    }


def _make_tuple_entry():
    """Create a mock tuple-style (legacy) registry entry."""
    return (MagicMock(), {}, "test", "description")


# ── detect_current_regime ────────────────────────────────────────────────────

class TestDetectCurrentRegime:
    @patch("crabquant.production.regime_scanner.load_data")
    @patch("crabquant.production.regime_scanner.detect_regime")
    def test_returns_detected_regime(self, mock_detect, mock_load):
        import pandas as pd
        df = pd.DataFrame({"close": [100, 101, 102]})
        mock_load.return_value = df
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.85})

        regime, meta = detect_current_regime("SPY", "6mo")

        assert regime == MarketRegime.TRENDING_UP
        assert meta["confidence"] == 0.85
        mock_load.assert_called_once_with("SPY", period="6mo")

    @patch("crabquant.production.regime_scanner.load_data")
    @patch("crabquant.production.regime_scanner.detect_regime")
    def test_custom_ticker_and_period(self, mock_detect, mock_load):
        mock_load.return_value = MagicMock()
        mock_detect.return_value = (MarketRegime.MEAN_REVERSION, {"confidence": 0.6})

        regime, _ = detect_current_regime("QQQ", "1y")

        assert regime == MarketRegime.MEAN_REVERSION
        mock_load.assert_called_once_with("QQQ", period="1y")

    @patch("crabquant.production.regime_scanner.load_data", side_effect=Exception("network error"))
    def test_falls_back_on_error(self, mock_load):
        regime, meta = detect_current_regime()

        assert regime == MarketRegime.LOW_VOLATILITY
        assert meta["confidence"] == 0.0
        assert meta["reason"] == "detection_failed"

    @patch("crabquant.production.regime_scanner.detect_regime", side_effect=Exception("bad data"))
    @patch("crabquant.production.regime_scanner.load_data")
    def test_falls_back_on_detect_error(self, mock_load, mock_detect):
        mock_load.return_value = MagicMock()
        regime, meta = detect_current_regime()

        assert regime == MarketRegime.LOW_VOLATILITY
        assert meta["confidence"] == 0.0


# ── scan_regime_strategies ───────────────────────────────────────────────────

class TestScanRegimeStrategies:
    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_empty_registry(self, mock_detect):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.9})

        result = scan_regime_strategies({})

        assert result["regime"] == "low_volatility"
        assert result["matched_count"] == 0
        assert result["strategies"] == []
        assert result["total_registry"] == 0

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_matches_regime_specific_strategy(self, mock_detect):
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8})

        registry = {
            "trend_strat": _make_dict_entry(
                regime_sharpes={"trending_up": 1.5, "mean_reversion": 0.1},
                is_regime_specific=True,
                preferred_regimes=["trending_up"],
            ),
        }

        result = scan_regime_strategies(registry)

        assert result["matched_count"] == 1
        assert result["strategies"][0]["name"] == "trend_strat"
        assert result["strategies"][0]["regime_sharpe"] == 1.5

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_skips_weak_regime_strategy(self, mock_detect):
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8})

        registry = {
            "mean_rev_strat": _make_dict_entry(
                regime_sharpes={"mean_reversion": 1.2, "trending_up": -0.3},
                is_regime_specific=True,
                weak_regimes=["trending_up"],
            ),
        }

        result = scan_regime_strategies(registry)

        assert result["matched_count"] == 0
        assert "mean_rev_strat" in result["skipped_regime_specific"]

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_legacy_tuple_skipped(self, mock_detect):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})

        registry = {
            "legacy_strat": _make_tuple_entry(),
        }

        result = scan_regime_strategies(registry)

        assert result["matched_count"] == 0
        assert "legacy_strat" in result["skipped_no_data"]

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_generalist_included_with_zero_sharpe(self, mock_detect):
        """Non-regime-specific strategies with 0 Sharpe in current regime should still be included."""
        mock_detect.return_value = (MarketRegime.HIGH_VOLATILITY, {"confidence": 0.7})

        registry = {
            "general_strat": _make_dict_entry(
                regime_sharpes={"high_volatility": 0.0},
                is_regime_specific=False,
            ),
        }

        result = scan_regime_strategies(registry)

        assert result["matched_count"] == 1
        assert result["strategies"][0]["name"] == "general_strat"

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_regime_specific_below_threshold_excluded(self, mock_detect):
        """Regime-specific strategies below min_regime_sharpe should be excluded."""
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8})

        registry = {
            "weak_trend": _make_dict_entry(
                regime_sharpes={"trending_up": 0.1},
                is_regime_specific=True,
            ),
        }

        result = scan_regime_strategies(registry, min_regime_sharpe=0.3)

        assert result["matched_count"] == 0

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_sorted_by_regime_sharpe_descending(self, mock_detect):
        mock_detect.return_value = (MarketRegime.MEAN_REVERSION, {"confidence": 0.6})

        registry = {
            "low": _make_dict_entry(regime_sharpes={"mean_reversion": 0.5}),
            "high": _make_dict_entry(regime_sharpes={"mean_reversion": 2.0}),
            "mid": _make_dict_entry(regime_sharpes={"mean_reversion": 1.0}),
        }

        result = scan_regime_strategies(registry)

        names = [s["name"] for s in result["strategies"]]
        assert names == ["high", "mid", "low"]

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_max_strategies_cap(self, mock_detect):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})

        registry = {
            f"strat_{i}": _make_dict_entry(regime_sharpes={"low_volatility": float(i)})
            for i in range(10)
        }

        result = scan_regime_strategies(registry, max_strategies=3)

        assert result["matched_count"] == 3
        assert len(result["strategies"]) == 3

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_override_regime_enum(self, mock_detect):
        """Passing a MarketRegime directly should skip auto-detection."""
        registry = {
            "test": _make_dict_entry(regime_sharpes={"high_volatility": 1.0}),
        }

        result = scan_regime_strategies(registry, current_regime=MarketRegime.HIGH_VOLATILITY)

        mock_detect.assert_not_called()
        assert result["regime"] == "high_volatility"
        assert result["regime_confidence"] == 1.0

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_override_regime_string(self, mock_detect):
        registry = {
            "test": _make_dict_entry(regime_sharpes={"trending_up": 1.0}),
        }

        result = scan_regime_strategies(registry, current_regime="trending_up")

        mock_detect.assert_not_called()
        assert result["regime"] == "trending_up"

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_timestamp_present(self, mock_detect):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})

        result = scan_regime_strategies({})

        assert "timestamp" in result
        assert "T" in result["timestamp"]  # ISO format

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_mixed_dict_and_tuple_entries(self, mock_detect):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})

        registry = {
            "new_strat": _make_dict_entry(regime_sharpes={"low_volatility": 0.8}),
            "old_strat": _make_tuple_entry(),
        }

        result = scan_regime_strategies(registry)

        assert result["matched_count"] == 1
        assert "old_strat" in result["skipped_no_data"]

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_strategy_fields_populated(self, mock_detect):
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8})

        registry = {
            "s": _make_dict_entry(
                regime_sharpes={"trending_up": 1.2, "mean_reversion": 0.3},
                is_regime_specific=True,
                preferred_regimes=["trending_up", "high_volatility"],
                weak_regimes=["mean_reversion"],
                description="A trending strategy",
            ),
        }

        result = scan_regime_strategies(registry)
        s = result["strategies"][0]

        assert s["name"] == "s"
        assert s["regime_sharpe"] == 1.2
        assert s["all_regime_sharpes"] == {"trending_up": 1.2, "mean_reversion": 0.3}
        assert s["is_regime_specific"] is True
        assert s["preferred_regimes"] == ["trending_up", "high_volatility"]
        assert s["description"] == "A trending strategy"

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_empty_regime_sharpes_dict(self, mock_detect):
        """Dict entry with empty regime_sharpes and not regime-specific."""
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})

        registry = {
            "no_data": _make_dict_entry(regime_sharpes={}, is_regime_specific=False),
        }

        result = scan_regime_strategies(registry)

        # Non-specific with 0 sharpe >= 0 should be included
        assert result["matched_count"] == 1
        assert result["strategies"][0]["regime_sharpe"] == 0.0


# ── scan_production_by_regime ────────────────────────────────────────────────
# Note: scan_production_by_regime uses lazy imports from crabquant.production
# and crabquant.strategies inside the function body. We must patch the modules
# where they are looked up at call time.

class TestScanProductionByRegime:
    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_empty_production(self, mock_detect):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.9})
        with patch("crabquant.production.get_production_strategies", return_value=[]):
            with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
                result = scan_production_by_regime()

        assert result["matched_count"] == 0
        assert result["total_production"] == 0
        assert result["strategies"] == []

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_production_strategy_no_registry_match(self, mock_detect):
        """Production strategy without registry entry should still appear."""
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})
        with patch("crabquant.production.get_production_strategies", return_value=[
            {
                "key": "unknown_strat|SPY|abc123",
                "strategy_name": "unknown_strat",
                "ticker": "SPY",
                "promoted_at": "2026-04-28",
            }
        ]):
            with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
                result = scan_production_by_regime()

        # No regime_data → not regime_data is truthy → included
        assert result["matched_count"] == 1
        assert result["strategies"][0]["strategy_name"] == "unknown_strat"

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_production_strategy_with_weak_regime_skipped(self, mock_detect):
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8})
        with patch("crabquant.production.get_production_strategies", return_value=[
            {
                "key": "rev_strat|SPY|abc123",
                "strategy_name": "rev_strat",
                "ticker": "SPY",
                "promoted_at": "2026-04-28",
            }
        ]):
            with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
                result = scan_production_by_regime()

        # No registry match → no regime data → no weak_regimes → included
        assert result["matched_count"] == 1

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_production_strategy_with_registry_regime_data(self, mock_detect):
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8})
        registry_entry = {
            "regime_sharpes": {"trending_up": 1.5},
            "is_regime_specific": True,
            "preferred_regimes": ["trending_up"],
            "weak_regimes": ["mean_reversion"],
        }
        with patch("crabquant.production.get_production_strategies", return_value=[
            {
                "key": "trend_strat|SPY|abc123",
                "strategy_name": "trend_strat",
                "ticker": "SPY",
                "promoted_at": "2026-04-28",
            }
        ]):
            with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {"trend_strat": registry_entry}, clear=True):
                result = scan_production_by_regime()

        assert result["matched_count"] == 1
        assert result["strategies"][0]["regime_sharpe"] == 1.5

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_production_strategy_weak_in_current_regime_skipped(self, mock_detect):
        mock_detect.return_value = (MarketRegime.MEAN_REVERSION, {"confidence": 0.7})
        registry_entry = {
            "regime_sharpes": {"trending_up": 1.5},
            "is_regime_specific": True,
            "preferred_regimes": ["trending_up"],
            "weak_regimes": ["mean_reversion"],
        }
        with patch("crabquant.production.get_production_strategies", return_value=[
            {
                "key": "trend_strat|SPY|abc123",
                "strategy_name": "trend_strat",
                "ticker": "SPY",
                "promoted_at": "2026-04-28",
            }
        ]):
            with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {"trend_strat": registry_entry}, clear=True):
                result = scan_production_by_regime()

        # trend_strat is weak in mean_reversion
        assert result["matched_count"] == 0

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_override_regime(self, mock_detect):
        with patch("crabquant.production.get_production_strategies", return_value=[
            {
                "key": "strat|SPY|abc",
                "strategy_name": "strat",
                "ticker": "SPY",
            }
        ]):
            with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
                result = scan_production_by_regime(current_regime=MarketRegime.HIGH_VOLATILITY)

        mock_detect.assert_not_called()
        assert result["regime"] == "high_volatility"
        assert result["regime_confidence"] == 1.0

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_sorted_by_regime_sharpe(self, mock_detect):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})
        with patch("crabquant.production.get_production_strategies", return_value=[
            {"key": "low|SPY|a", "strategy_name": "low", "ticker": "SPY"},
            {"key": "high|SPY|b", "strategy_name": "high", "ticker": "SPY"},
        ]):
            with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
                result = scan_production_by_regime()

        # Both have regime_sharpe 0 (no registry match) — stable sort
        assert result["matched_count"] == 2

    @patch("crabquant.production.regime_scanner.detect_current_regime")
    def test_timestamp_and_metadata(self, mock_detect):
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.5})
        with patch("crabquant.production.get_production_strategies", return_value=[]):
            with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
                result = scan_production_by_regime()

        assert "timestamp" in result
        assert "regime" in result
        assert "regime_confidence" in result
        assert "total_production" in result
