"""
Tests for crabquant.brief.market module.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from crabquant.brief.market import (
    RESULTS_DIR,
    _get_results_dir,
    get_best_strategies_for_regime,
    get_market_regime,
)
from crabquant.regime import MarketRegime


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_spy_df(length=60):
    """Create a minimal SPY-like DataFrame."""
    rng = __import__("numpy").random.RandomState(0)
    return pd.DataFrame(
        {"close": 450 + rng.randn(length).cumsum()},
        index=pd.date_range("2025-01-01", periods=length, freq="D"),
    )


def _mock_detect(regime=MarketRegime.TRENDING_UP, confidence=0.85, realized_vol=0.12, scores=None):
    """Return a (regime, metadata) tuple for detect_regime mock."""
    return (regime, {
        "confidence": confidence,
        "realized_vol": realized_vol,
        "scores": scores or {},
    })


# ── _get_results_dir ─────────────────────────────────────────────────────


class TestGetResultsDir:

    def test_returns_string(self):
        result = _get_results_dir()
        assert isinstance(result, str)

    def test_contains_results(self):
        result = _get_results_dir()
        assert "results" in result

    def test_resolves_to_absolute_path(self):
        result = _get_results_dir()
        assert Path(result).is_absolute()

    def test_caching_returns_same_value(self):
        r1 = _get_results_dir()
        r2 = _get_results_dir()
        assert r1 == r2

    def test_global_cache_variable_exists(self):
        from crabquant.brief.market import RESULTS_DIR
        # Before first call, may be None; after, is a string
        _get_results_dir()
        assert RESULTS_DIR is not None


# ── get_market_regime ─────────────────────────────────────────────────────


class TestGetMarketRegime:

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_returns_expected_keys(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        mock_detect.return_value = _mock_detect(scores={"trend": 0.9})

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
        mock_detect.return_value = _mock_detect(regime=MarketRegime.TRENDING_DOWN, confidence=0.7, realized_vol=0.25)

        result = get_market_regime()
        assert result["regime"] == "trending_down"

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_spy_20d_return_with_enough_data(self, mock_detect, mock_load):
        df = _make_spy_df(60)
        mock_load.return_value = df
        mock_detect.return_value = _mock_detect()

        result = get_market_regime()
        assert result["spy_20d_return"] is not None
        expected = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100
        assert result["spy_20d_return"] == round(expected, 1)

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_spy_20d_return_insufficient_data(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df(15)
        mock_detect.return_value = _mock_detect()

        result = get_market_regime()
        assert result["spy_20d_return"] is None

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_confidence_forwarded(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        mock_detect.return_value = _mock_detect(regime=MarketRegime.MEAN_REVERSION, confidence=0.92, realized_vol=0.08)

        result = get_market_regime()
        assert result["confidence"] == 0.92

    @patch("crabquant.brief.market.load_data")
    def test_load_data_called_with_spy(self, mock_load):
        mock_load.return_value = _make_spy_df()

        with patch("crabquant.brief.market.detect_regime", return_value=(MarketRegime.TRENDING_UP, {})):
            get_market_regime()

        mock_load.assert_called_once_with("SPY", period="2mo")

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_realized_vol_forwarded(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        mock_detect.return_value = _mock_detect(regime=MarketRegime.HIGH_VOLATILITY, confidence=0.6, realized_vol=0.35)

        result = get_market_regime()
        assert result["realized_vol"] == 0.35

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_all_regime_types(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df(60)
        for regime in MarketRegime:
            mock_detect.return_value = _mock_detect(regime=regime)
            result = get_market_regime()
            assert result["regime"] == regime.value

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_confidence_default_when_missing(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {})
        result = get_market_regime()
        assert result["confidence"] == 0.0

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_scores_default_empty_when_missing(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        mock_detect.return_value = (MarketRegime.TRENDING_UP, {"confidence": 0.5})
        result = get_market_regime()
        assert result["scores"] == {}

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_scores_forwarded(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        scores = {"trend": 0.9, "volatility": 0.3, "mean_rev": 0.1}
        mock_detect.return_value = _mock_detect(scores=scores)
        result = get_market_regime()
        assert result["scores"] == scores

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_spy_20d_return_positive(self, mock_detect, mock_load):
        """When SPY is up over 20 days, return should be positive."""
        rng = __import__("numpy").random.RandomState(42)
        # Create a steadily rising price series
        df = pd.DataFrame(
            {"close": [400 + i * 0.5 for i in range(60)]},
            index=pd.date_range("2025-01-01", periods=60, freq="D"),
        )
        mock_load.return_value = df
        mock_detect.return_value = _mock_detect()

        result = get_market_regime()
        assert result["spy_20d_return"] is not None
        assert result["spy_20d_return"] > 0

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_spy_20d_return_negative(self, mock_detect, mock_load):
        """When SPY is down over 20 days, return should be negative."""
        df = pd.DataFrame(
            {"close": [500 - i * 0.5 for i in range(60)]},
            index=pd.date_range("2025-01-01", periods=60, freq="D"),
        )
        mock_load.return_value = df
        mock_detect.return_value = _mock_detect()

        result = get_market_regime()
        assert result["spy_20d_return"] is not None
        assert result["spy_20d_return"] < 0

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_spy_20d_return_exactly_21_bars(self, mock_detect, mock_load):
        """Boundary: exactly 21 bars should be enough for 20d return."""
        df = _make_spy_df(21)
        mock_load.return_value = df
        mock_detect.return_value = _mock_detect()

        result = get_market_regime()
        assert result["spy_20d_return"] is not None

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_spy_20d_return_20_bars_not_enough(self, mock_detect, mock_load):
        """Boundary: 20 bars should not be enough (need 21 for 20d lookback)."""
        df = _make_spy_df(20)
        mock_load.return_value = df
        mock_detect.return_value = _mock_detect()

        result = get_market_regime()
        assert result["spy_20d_return"] is None

    @patch("crabquant.brief.market.load_data")
    @patch("crabquant.brief.market.detect_regime")
    def test_returns_dict_type(self, mock_detect, mock_load):
        mock_load.return_value = _make_spy_df()
        mock_detect.return_value = _mock_detect()

        result = get_market_regime()
        assert isinstance(result, dict)


# ── get_best_strategies_for_regime ───────────────────────────────────────


class TestGetBestStrategiesForRegime:

    @patch("crabquant.brief.market._get_results_dir")
    def test_returns_empty_when_no_confirmed_file(self, mock_dir, tmp_path):
        mock_dir.return_value = str(tmp_path)
        result = get_best_strategies_for_regime("trending_up")
        assert result == []

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_returns_strategies_from_confirmed(self, mock_dir, mock_ranking, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

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
        assert result[0]["total_return"] == 15.0

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
        assert result[0]["strategy_name"] == "high_score"

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_result_dict_keys(self, mock_dir, mock_ranking, tmp_path):
        """Each result dict should have all expected keys."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "SPY", "strategy": "sma", "confirm_sharpe": 1.0,
             "confirm_return": 0.05, "verdict": "pass", "discovery_regime": "trending_up"}
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        result = get_best_strategies_for_regime("trending_up")
        entry = result[0]
        assert "ticker" in entry
        assert "strategy_name" in entry
        assert "sharpe" in entry
        assert "total_return" in entry
        assert "verdict" in entry
        assert "discovery_regime" in entry

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_sharpe_rounding(self, mock_dir, mock_ranking, tmp_path):
        """Sharpe should be rounded to 2 decimal places."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "SPY", "strategy": "sma", "confirm_sharpe": 1.23456,
             "confirm_return": 0.05, "verdict": "pass", "discovery_regime": ""}
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        result = get_best_strategies_for_regime("trending_up")
        assert result[0]["sharpe"] == 1.23

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_total_return_is_percentage(self, mock_dir, mock_ranking, tmp_path):
        """confirm_return (decimal) should be converted to percentage."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "SPY", "strategy": "sma", "confirm_sharpe": 1.0,
             "confirm_return": 0.12345, "verdict": "pass", "discovery_regime": ""}
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        result = get_best_strategies_for_regime("trending_up")
        assert result[0]["total_return"] == round(0.12345 * 100, 1)

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_empty_confirmed_list(self, mock_dir, mock_ranking, tmp_path):
        """Empty confirmed.json should return empty list."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        (confirmed_dir / "confirmed.json").write_text(json.dumps([]))

        mock_dir.return_value = str(results_dir)

        result = get_best_strategies_for_regime("trending_up")
        assert result == []

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_top_n_greater_than_available(self, mock_dir, mock_ranking, tmp_path):
        """top_n larger than available should return all."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "SPY", "strategy": "sma", "confirm_sharpe": 1.0,
             "confirm_return": 0.05, "verdict": "pass", "discovery_regime": ""}
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        result = get_best_strategies_for_regime("trending_up", top_n=100)
        assert len(result) == 1

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_top_n_default_is_five(self, mock_dir, mock_ranking, tmp_path):
        """Default top_n should be 5."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": f"T{i}", "strategy": f"s{i}", "confirm_sharpe": float(i),
             "confirm_return": 0.01 * i, "verdict": "pass", "discovery_regime": ""}
            for i in range(10)
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [(f"s{i}", float(i)) for i in range(10)]

        result = get_best_strategies_for_regime("trending_up")
        assert len(result) == 5

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_regime_name_case_insensitive(self, mock_dir, mock_ranking, tmp_path):
        """Regime names should be matched case-insensitively."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "SPY", "strategy": "sma", "confirm_sharpe": 1.0,
             "confirm_return": 0.05, "verdict": "pass", "discovery_regime": ""}
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        result = get_best_strategies_for_regime("TRENDING_UP")
        mock_ranking.assert_called_once()

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_all_known_regimes_map_correctly(self, mock_dir, mock_ranking, tmp_path):
        """Each known regime name should trigger get_strategy_ranking."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [{"ticker": "SPY", "strategy": "sma", "confirm_sharpe": 1.0,
                          "confirm_return": 0.05, "verdict": "pass", "discovery_regime": ""}]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        for regime_name in ["trending_up", "trending_down", "mean_reversion",
                            "high_volatility", "low_volatility"]:
            mock_ranking.reset_mock()
            get_best_strategies_for_regime(regime_name)
            mock_ranking.assert_called_once()

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_discovery_regime_forwarded(self, mock_dir, mock_ranking, tmp_path):
        """discovery_regime field should be forwarded to result."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "SPY", "strategy": "sma", "confirm_sharpe": 1.0,
             "confirm_return": 0.05, "verdict": "pass", "discovery_regime": "mean_reversion"}
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        result = get_best_strategies_for_regime("trending_up")
        assert result[0]["discovery_regime"] == "mean_reversion"

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_strategy_with_duplicate_names_sorted_by_sharpe(self, mock_dir, mock_ranking, tmp_path):
        """When strategies have the same regime affinity, sort by sharpe."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "A", "strategy": "sma", "confirm_sharpe": 1.0,
             "confirm_return": 0.05, "verdict": "pass", "discovery_regime": ""},
            {"ticker": "B", "strategy": "sma", "confirm_sharpe": 3.0,
             "confirm_return": 0.15, "verdict": "pass", "discovery_regime": ""},
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        result = get_best_strategies_for_regime("trending_up")
        # Both have same regime score, so higher sharpe comes first
        assert result[0]["ticker"] == "B"
        assert result[1]["ticker"] == "A"

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_returns_list_type(self, mock_dir, mock_ranking, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [{"ticker": "SPY", "strategy": "sma", "confirm_sharpe": 1.0,
                          "confirm_return": 0.05, "verdict": "pass", "discovery_regime": ""}]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        result = get_best_strategies_for_regime("trending_up")
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    @patch("crabquant.brief.market.get_strategy_ranking")
    @patch("crabquant.brief.market._get_results_dir")
    def test_negative_return_formatted_correctly(self, mock_dir, mock_ranking, tmp_path):
        """Negative confirm_return should produce negative total_return."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        confirmed_data = [
            {"ticker": "SPY", "strategy": "sma", "confirm_sharpe": -0.5,
             "confirm_return": -0.10, "verdict": "fail", "discovery_regime": ""}
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed_data))

        mock_dir.return_value = str(results_dir)
        mock_ranking.return_value = [("sma", 0.9)]

        result = get_best_strategies_for_regime("trending_up")
        assert result[0]["total_return"] == -10.0
        assert result[0]["sharpe"] == -0.5
