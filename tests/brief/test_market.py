"""
Tests for crabquant.brief.market module.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crabquant.brief.market import (
    _get_results_dir,
    get_best_strategies_for_regime,
    get_market_regime,
)


def _make_spy_df(length=60):
    """Create a minimal SPY-like DataFrame."""
    rng = __import__("numpy").random.RandomState(0)
    return pd.DataFrame(
        {"close": 450 + rng.randn(length).cumsum()},
        index=pd.date_range("2025-01-01", periods=length, freq="D"),
    )


class TestGetMarketRegime:

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_returns_expected_keys(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        from crabquant.regime import MarketRegime
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {
            "confidence": 0.85, "realized_vol": 0.12, "scores": {"trend": 0.9}
        })

        result = get_market_regime()

        assert "regime" in result
        assert "confidence" in result
        assert "spy_20d_return" in result
        assert "realized_vol" in result
        assert "scores" in result

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_regime_value(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        from crabquant.regime import MarketRegime
        mock_detect.return_value = (MarketRegime.TRENDING_DOWN, {
            "confidence": 0.7, "realized_vol": 0.25, "scores": {}
        })

        result = get_market_regime()
        assert result["regime"] == "trending_down"

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_spy_20d_return_with_enough_data(self, mock_detect, mock_load):
        df = _make_spy_df(60)  # More than 21 bars
        mock_load.return_value = df
        from crabquant.regime import MarketRegime
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8, "realized_vol": 0.1})

        result = get_market_regime()
        assert result["spy_20d_return"] is not None
        # Verify the calculation manually
        expected = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100
        assert result["spy_20d_return"] == round(expected, 1)

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_spy_20d_return_insufficient_data(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df(15)  # Less than 21 bars
        from crabquant.regime import MarketRegime
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.8, "realized_vol": 0.1})

        result = get_market_regime()
        assert result["spy_20d_return"] is None

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_confidence_forwarded(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        from crabquant.regime import MarketRegime
        mock_detect.return_value = (MarketRegime.MEAN_REVERSION, {
            "confidence": 0.92, "realized_vol": 0.08
        })

        result = get_market_regime()
        assert result["confidence"] == 0.92

    @patch("crabquant.brief.market.load_data")
    def test_load_data_called_with_spy(self, mock_load):
        mock_load.return_value = _make_spy_df()
        from crabquant.regime import MarketRegime

        with patch("crabquant.brief.market.detect_regime", return_value=(MarketRegime.TRENDING_UP, {})):
            get_market_regime()

        mock_load.assert_called_once_with("SPY", period="2mo")

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_realized_vol_forwarded(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        from crabquant.regime import MarketRegime
        mock_detect.return_value = (MarketRegime.HIGH_VOLATILITY, {
            "confidence": 0.6, "realized_vol": 0.35
        })

        result = get_market_regime()
        assert result["realized_vol"] == 0.35


class TestGetBestStrategiesForRegime:

    @patch("crabquant.brief.market._get_results_dir")
    def test_returns_empty_when_no_confirmed_file(self, mock_dir, tmp_path):
        mock_dir.return_value = str(tmp_path)
        # No confirmed.json exists
        result = get_best_strategies_for_regime("trending_up")
        assert result == []

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_returns_strategies_from_confirmed(self, mock_dir, mock_ranking, tmp_path):
        # Set up directory structure
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        # Write confirmed.json
        confirmed_data = [
            {"ticker": "SPY", "strategy": "sma_cross", "confirm_sharpe": 1.8,
             "confirm_return": 0.15, "verdict": "pass", "discovery_regime": "trending_up"},
            {"ticker": "QQQ", "strategy": "rsi_mean", "confirm_sharpe": 1.2,
             "confirm_return": 0.10, "verdict": "pass", "discovery_regime": "mean_reversion"},
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma_cross", 0.9), ("rsi_mean", 0.5)]

        result = get_best_strategies_for_regime("trending_up")

        assert len(result) == 2
        assert result[0]["ticker"] == "SPY"
        assert result[0]["strategy_name"] == "sma_cross"
        assert result[0]["sharpe"] == 1.8
        assert result[0]["total_return"] == 15.0  # 0.15 * 100

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_top_n_limits_results(self, mock_dir, mock_ranking, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": f"T{i}", "strategy": f"strat_{i}", "confirm_sharpe": float(i),
             "confirm_return": 0.01 * i, "verdict": "pass", "discovery_regime": ""}
            for i in range(10)
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [(f"strat_{i}", float(i)) for i in range(10)]

        result = get_best_strategies_for_regime("trending_up", top_n=3)
        assert len(result) == 3

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_unknown_regime_falls_back_to_all(self, mock_dir, mock_ranking, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "SPY", "strategy": "sma_cross", "confirm_sharpe": 1.0,
             "confirm_return": 0.05, "verdict": "pass", "discovery_regime": ""}
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)

        # unknown regime → should not call get_strategy_ranking
        result = get_best_strategies_for_regime("unknown_regime_xyz")
        mock_ranking.assert_not_called()
        assert len(result) == 1

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_missing_fields_use_defaults(self, mock_dir, mock_ranking, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        # Entry missing many optional fields
        confirmed_data = [{"strategy": "minimal"}]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("minimal", 0.5)]

        result = get_best_strategies_for_regime("trending_up")

        assert len(result) == 1
        assert result[0]["ticker"] == "?"
        assert result[0]["strategy_name"] == "minimal"
        assert result[0]["sharpe"] == 0.0
        assert result[0]["total_return"] == 0.0
        assert result[0]["verdict"] == ""

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_sorted_by_regime_affinity(self, mock_dir, mock_ranking, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "A", "strategy": "low_score", "confirm_sharpe": 3.0,
             "confirm_return": 0.3, "verdict": "pass"},
            {"ticker": "B", "strategy": "high_score", "confirm_sharpe": 1.0,
             "confirm_return": 0.1, "verdict": "pass"},
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("high_score", 10.0), ("low_score", 1.0)]

        result = get_best_strategies_for_regime("trending_up")
        # high_score should be first despite lower sharpe, due to regime affinity
        assert result[0]["strategy_name"] == "high_score"
