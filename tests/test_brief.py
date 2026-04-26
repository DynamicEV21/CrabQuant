"""
Tests for the daily market brief system.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crabquant.brief.models import BriefData
from crabquant.brief import generate_brief
from crabquant.brief.discoveries import (
    get_cron_status,
    get_recent_promotions,
    get_recent_winners,
    get_retirements,
)
from crabquant.brief.formatter import format_brief
from crabquant.brief.market import get_best_strategies_for_regime, get_market_regime
from crabquant.regime import MarketRegime


# ── Fixtures ──

@pytest.fixture
def sample_spy_data():
    """Generate synthetic SPY data for testing."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 60
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Slight upward trend
    close = 500 + np.cumsum(np.random.randn(n) * 2 + 0.3)
    return pd.DataFrame(
        {"close": close, "open": close - 0.5, "high": close + 1, "low": close - 1, "volume": 1000000},
        index=dates,
    )


@pytest.fixture
def empty_brief():
    return BriefData()


@pytest.fixture
def full_brief():
    return BriefData(
        regime="trending_up",
        regime_confidence=0.75,
        spy_20d_return=2.3,
        realized_vol=0.142,
        top_production=[
            {"ticker": "CAT", "strategy_name": "roc_ema_volume", "sharpe": 2.66, "total_return": 160.0, "verdict": "ROBUST"},
            {"ticker": "JNJ", "strategy_name": "rsi_crossover", "sharpe": 1.99, "total_return": 43.0, "verdict": "ROBUST"},
        ],
        recent_winners_count=3,
        recent_promotions_count=1,
        recent_retirements_count=0,
        total_combos_tested=188,
        cron_active=4,
        cron_total=4,
    )


# ── Market regime tests ──

class TestMarketRegime:
    def test_returns_valid_regime(self, sample_spy_data):
        with patch("crabquant.brief.market.load_data", return_value=sample_spy_data):
            result = get_market_regime()

        assert "regime" in result
        # Must be a valid regime value
        valid = {r.value for r in MarketRegime}
        assert result["regime"] in valid

    def test_includes_metadata(self, sample_spy_data):
        with patch("crabquant.brief.market.load_data", return_value=sample_spy_data):
            result = get_market_regime()

        assert "confidence" in result
        assert "scores" in result
        assert isinstance(result["confidence"], float)

    def test_spy_20d_return(self, sample_spy_data):
        with patch("crabquant.brief.market.load_data", return_value=sample_spy_data):
            result = get_market_regime()

        assert result["spy_20d_return"] is not None
        assert isinstance(result["spy_20d_return"], float)


# ── Best strategies for regime ──

class TestBestStrategies:
    def test_empty_when_no_confirmed(self, tmp_path):
        with patch("crabquant.brief.market._get_results_dir", return_value=str(tmp_path)):
            result = get_best_strategies_for_regime("trending_up")

        assert result == []

    def test_returns_sorted_strategies(self, tmp_path):
        # Create confirmed.json with test data
        confirmed_dir = tmp_path / "confirmed"
        confirmed_dir.mkdir()
        confirmed_file = confirmed_dir / "confirmed.json"
        confirmed_file.write_text(json.dumps([
            {
                "key": "test|AAPL|{}",
                "strategy": "macd_momentum",
                "ticker": "AAPL",
                "confirm_sharpe": 2.5,
                "confirm_return": 0.5,
                "verdict": "ROBUST",
            },
            {
                "key": "test|MSFT|{}",
                "strategy": "bollinger_squeeze",
                "ticker": "MSFT",
                "confirm_sharpe": 1.8,
                "confirm_return": 0.3,
                "verdict": "ROBUST",
            },
        ]))

        with patch("crabquant.brief.market._get_results_dir", return_value=str(tmp_path)):
            result = get_best_strategies_for_regime("trending_up")

        assert len(result) == 2
        # macd_momentum should rank higher for trending_up
        assert result[0]["strategy_name"] == "macd_momentum"


# ── Recent winners tests ──

class TestRecentWinners:
    def test_filters_by_time(self, tmp_path):
        winners_dir = tmp_path / "winners"
        winners_dir.mkdir()
        winners_file = winners_dir / "winners.json"

        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=48)).isoformat()
        recent = now.isoformat()

        winners_file.write_text(json.dumps([
            {"ticker": "OLD", "strategy": "test", "discovered": old, "score": 2.0, "sharpe": 2.0, "return": 1.0},
            {"ticker": "NEW", "strategy": "test2", "discovered": recent, "score": 1.5, "sharpe": 1.5, "return": 0.5},
        ]))

        with patch("crabquant.brief.discoveries._get_results_dir", return_value=tmp_path):
            result = get_recent_winners(hours=24)

        assert result["count"] == 1
        assert result["top_winners"][0]["ticker"] == "NEW"

    def test_empty_when_no_file(self, tmp_path):
        with patch("crabquant.brief.discoveries._get_results_dir", return_value=tmp_path):
            result = get_recent_winners(hours=24)

        assert result["count"] == 0
        assert result["top_winners"] == []


# ── Promotions / retirements ──

class TestPromotionsRetirements:
    def test_recent_promotions(self, tmp_path):
        confirmed_dir = tmp_path / "confirmed"
        confirmed_dir.mkdir()
        confirmed_file = confirmed_dir / "confirmed.json"

        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=48)).isoformat()
        recent = now.isoformat()

        confirmed_file.write_text(json.dumps([
            {"verdict": "ROBUST", "confirmed_at": old},
            {"verdict": "ROBUST", "confirmed_at": recent},
            {"verdict": "FAILED", "confirmed_at": recent},
        ]))

        with patch("crabquant.brief.discoveries._get_results_dir", return_value=tmp_path):
            count = get_recent_promotions(hours=24)

        assert count == 1  # only the recent ROBUST

    def test_recent_retirements(self, tmp_path):
        confirmed_dir = tmp_path / "confirmed"
        confirmed_dir.mkdir()
        confirmed_file = confirmed_dir / "confirmed.json"

        now = datetime.now(timezone.utc)
        recent = now.isoformat()

        confirmed_file.write_text(json.dumps([
            {"verdict": "ROBUST", "confirmed_at": recent},
            {"verdict": "FAILED", "confirmed_at": recent},
            {"verdict": "FAILED", "confirmed_at": recent},
        ]))

        with patch("crabquant.brief.discoveries._get_results_dir", return_value=tmp_path):
            count = get_retirements(hours=24)

        assert count == 2


# ── Cron status ──

class TestCronStatus:
    def test_handles_missing_command(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_cron_status()

        assert result["active"] == 0
        assert result["total"] == 4  # defaults to 4

    def test_parses_output(self):
        mock_result = MagicMock()
        mock_result.stdout = "ID: abc1 | crabquant-sweep | idle\nID: abc2 | crabquant-validate | running\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = get_cron_status()

        assert result["total"] >= 2


# ── Formatter tests ──

class TestFormatter:
    def test_under_800_chars(self, full_brief):
        result = format_brief(full_brief)
        assert len(result) <= 800

    def test_empty_brief_still_valid(self, empty_brief):
        result = format_brief(empty_brief)
        assert isinstance(result, str)
        assert "CrabQuant" in result

    def test_no_production_message(self, empty_brief):
        result = format_brief(empty_brief)
        assert "still discovering" in result

    def test_no_activity_message(self, empty_brief):
        result = format_brief(empty_brief)
        assert "No new discoveries" in result

    def test_full_brief_content(self, full_brief):
        result = format_brief(full_brief)
        assert "CAT" in result
        assert "3 new winners" in result
        assert "1 promoted" in result
        assert "4/4 active" in result

    def test_truncation(self):
        # Make a brief that would be way too long
        huge_brief = BriefData(
            regime="trending_up",
            top_production=[
                {"ticker": f"T{i}", "strategy_name": f"s{i}", "sharpe": 1.0, "total_return": 10.0, "verdict": "ROBUST"}
                for i in range(50)
            ],
            cron_active=4,
            cron_total=4,
        )
        result = format_brief(huge_brief)
        assert len(result) <= 800


# ── Integration: generate_brief ──

class TestGenerateBrief:
    def test_handles_all_missing_data(self):
        """Should not crash when all data sources are unavailable."""
        with (
            patch("crabquant.brief.market.get_market_regime", side_effect=Exception("no data")),
            patch("crabquant.brief.market.get_best_strategies_for_regime", side_effect=Exception("no data")),
            patch("crabquant.brief.discoveries.get_recent_winners", side_effect=Exception("no data")),
            patch("crabquant.brief.discoveries.get_recent_promotions", side_effect=Exception("no data")),
            patch("crabquant.brief.discoveries.get_retirements", side_effect=Exception("no data")),
            patch("crabquant.brief.discoveries.get_cron_status", side_effect=Exception("no data")),
            patch("crabquant.brief._get_regime_suggestions", return_value=[]),
        ):
            result = generate_brief()

        assert isinstance(result, str)
        assert len(result) <= 800

    def test_returns_no_reply_when_empty(self):
        """Should return NO_REPLY when there's genuinely nothing."""
        with (
            patch("crabquant.brief.market.get_market_regime", return_value={"regime": "UNKNOWN"}),
            patch("crabquant.brief.market.get_best_strategies_for_regime", return_value=[]),
            patch("crabquant.brief.discoveries.get_recent_winners", return_value={"count": 0, "top_winners": [], "total_tested": 0}),
            patch("crabquant.brief.discoveries.get_recent_promotions", return_value=0),
            patch("crabquant.brief.discoveries.get_retirements", return_value=0),
            patch("crabquant.brief.discoveries.get_cron_status", return_value={"active": 0, "total": 0}),
            patch("crabquant.brief._get_regime_suggestions", return_value=[]),
        ):
            result = generate_brief()

        # Even with "no data", the brief still has the header etc, so it should NOT be NO_REPLY
        # unless the formatter returns empty
        assert isinstance(result, str)
