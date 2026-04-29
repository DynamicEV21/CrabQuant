"""Tests for the brief module — discoveries, models, market."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from crabquant.brief.models import BriefData
from crabquant.brief.discoveries import (
    get_recent_winners,
    get_recent_promotions,
    get_retirements,
    get_cron_status,
    get_promotion_metrics,
)
from crabquant.brief.market import _get_results_dir


# ── BriefData ─────────────────────────────────────────────────────────────


class TestBriefData:

    def test_default_values(self):
        b = BriefData()
        assert b.regime == "UNKNOWN"
        assert b.regime_confidence == 0.0
        assert b.spy_20d_return is None
        assert b.realized_vol is None
        assert b.top_production == []
        assert b.recent_winners_count == 0
        assert b.promotion_metrics == {}

    def test_custom_values(self):
        b = BriefData(
            regime="TRENDING_UP",
            regime_confidence=0.85,
            spy_20d_return=2.3,
            recent_winners_count=5,
        )
        assert b.regime == "TRENDING_UP"
        assert b.regime_confidence == 0.85
        assert b.spy_20d_return == 2.3
        assert b.recent_winners_count == 5

    def test_regime_strategy_suggestions(self):
        b = BriefData(regime_strategy_suggestions=[("momentum", 0.9), ("mean_reversion", 0.3)])
        assert len(b.regime_strategy_suggestions) == 2
        assert b.regime_strategy_suggestions[0] == ("momentum", 0.9)


# ── get_recent_winners ────────────────────────────────────────────────────


class TestGetRecentWinners:

    def _make_winners(self, hours_ago=2):
        now = datetime.now(timezone.utc)
        discovered = (now - timedelta(hours=hours_ago)).isoformat()
        return [
            {"ticker": "SPY", "strategy": "ema_crossover", "sharpe": 1.5,
             "return": 0.12, "score": 3.0, "discovered": discovered},
            {"ticker": "AAPL", "strategy": "rsi_reversion", "sharpe": 1.2,
             "return": 0.08, "score": 2.5, "discovered": discovered},
            {"ticker": "QQQ", "strategy": "momentum", "sharpe": 2.0,
             "return": 0.15, "score": 4.0, "discovered": discovered},
        ]

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_returns_recent_winners(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        winners_dir = results_dir / "winners"
        winners_dir.mkdir()
        (winners_dir / "winners.json").write_text(
            json.dumps(self._make_winners(hours_ago=2))
        )
        mock_dir.return_value = results_dir

        result = get_recent_winners(hours=24)
        assert result["count"] == 3
        assert len(result["top_winners"]) == 3
        # Top winner should be QQQ (highest score)
        assert result["top_winners"][0]["ticker"] == "QQQ"

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_ignores_old_winners(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        winners_dir = results_dir / "winners"
        winners_dir.mkdir()
        # Winner from 48 hours ago
        old_winners = self._make_winners(hours_ago=48)
        (winners_dir / "winners.json").write_text(json.dumps(old_winners))
        mock_dir.return_value = results_dir

        result = get_recent_winners(hours=24)
        assert result["count"] == 0
        assert result["top_winners"] == []

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_no_winners_file(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        mock_dir.return_value = results_dir

        result = get_recent_winners(hours=24)
        assert result["count"] == 0
        assert result["top_winners"] == []

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_total_tested_from_cron_state(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        winners_dir = results_dir / "winners"
        winners_dir.mkdir()
        (winners_dir / "winners.json").write_text(json.dumps([]))
        (results_dir / "cron_state.json").write_text(
            json.dumps({"completed_combos": ["combo1", "combo2", "combo3"]})
        )
        mock_dir.return_value = results_dir

        result = get_recent_winners(hours=24)
        assert result["total_tested"] == 3

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_winners_without_discovered_timestamp(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        winners_dir = results_dir / "winners"
        winners_dir.mkdir()
        winners = [
            {"ticker": "SPY", "strategy": "ema_crossover", "sharpe": 1.5,
             "return": 0.12, "score": 3.0, "discovered": ""},
            {"ticker": "AAPL", "strategy": "rsi_reversion", "sharpe": 1.2,
             "return": 0.08, "score": 2.5},  # No discovered key
        ]
        (winners_dir / "winners.json").write_text(json.dumps(winners))
        mock_dir.return_value = results_dir

        result = get_recent_winners(hours=24)
        assert result["count"] == 0  # No valid timestamps

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_invalid_timestamp_ignored(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        winners_dir = results_dir / "winners"
        winners_dir.mkdir()
        winners = [
            {"ticker": "SPY", "strategy": "ema", "sharpe": 1.5,
             "return": 0.12, "score": 3.0, "discovered": "not-a-date"},
        ]
        (winners_dir / "winners.json").write_text(json.dumps(winners))
        mock_dir.return_value = results_dir

        result = get_recent_winners(hours=24)
        assert result["count"] == 0

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_top_3_only(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        winners_dir = results_dir / "winners"
        winners_dir.mkdir()
        winners = self._make_winners(hours_ago=1) + self._make_winners(hours_ago=1)
        (winners_dir / "winners.json").write_text(json.dumps(winners))
        mock_dir.return_value = results_dir

        result = get_recent_winners(hours=24)
        assert result["count"] == 6
        assert len(result["top_winners"]) == 3


# ── get_recent_promotions ─────────────────────────────────────────────────


class TestGetRecentPromotions:

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_counts_robust_confirmations(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=2)).isoformat()
        old = (now - timedelta(hours=48)).isoformat()

        confirmed = [
            {"confirmed_at": recent, "verdict": "ROBUST", "strategy": "ema"},
            {"confirmed_at": recent, "verdict": "ROBUST", "strategy": "rsi"},
            {"confirmed_at": recent, "verdict": "FAILED", "strategy": "bad"},
            {"confirmed_at": old, "verdict": "ROBUST", "strategy": "old_robust"},
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed))
        mock_dir.return_value = results_dir

        count = get_recent_promotions(hours=24)
        assert count == 2  # Only recent ROBUST ones

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_no_confirmed_file(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        mock_dir.return_value = results_dir

        count = get_recent_promotions(hours=24)
        assert count == 0

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_registry_promotions(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()
        (confirmed_dir / "confirmed.json").write_text(json.dumps([]))

        strategies_dir = tmp_path / "strategies" / "production"
        strategies_dir.mkdir(parents=True)

        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=2)).isoformat()

        registry = [
            {"promoted_at": recent, "strategy": "ema", "ticker": "SPY"},
        ]
        (strategies_dir / "registry.json").write_text(json.dumps(registry))
        mock_dir.return_value = results_dir

        count = get_recent_promotions(hours=24)
        assert count == 1


# ── get_retirements ───────────────────────────────────────────────────────


class TestGetRetirements:

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_counts_failed_verdicts(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        confirmed_dir = results_dir / "confirmed"
        confirmed_dir.mkdir()

        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=2)).isoformat()

        confirmed = [
            {"confirmed_at": recent, "verdict": "FAILED", "strategy": "bad1"},
            {"confirmed_at": recent, "verdict": "FAILED", "strategy": "bad2"},
            {"confirmed_at": recent, "verdict": "ROBUST", "strategy": "good"},
        ]
        (confirmed_dir / "confirmed.json").write_text(json.dumps(confirmed))
        mock_dir.return_value = results_dir

        count = get_retirements(hours=24)
        assert count == 2

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_no_confirmed_file(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        mock_dir.return_value = results_dir

        count = get_retirements(hours=24)
        assert count == 0


# ── get_cron_status ───────────────────────────────────────────────────────


class TestGetCronStatus:

    @patch("subprocess.run")
    def test_command_unavailable(self, mock_run):
        mock_run.side_effect = FileNotFoundError("openclaw not found")
        result = get_cron_status()
        assert result["active"] == 0
        assert result["total"] == 4

    @patch("subprocess.run")
    def test_command_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("openclaw", 10)
        result = get_cron_status()
        assert result["active"] == 0
        assert result["total"] == 4

    @patch("subprocess.run")
    def test_parses_crabquant_lines(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="ID: abc1 crabquant sweep\nID: abc2 validate\nactive\n",
            stderr="",
        )
        result = get_cron_status()
        assert result["total"] >= 1
        assert len(result["details"]) > 0


# ── get_promotion_metrics ─────────────────────────────────────────────────


class TestGetPromotionMetrics:

    @patch("crabquant.brief.discoveries._get_results_dir")
    def test_delegates_to_promoter(self, mock_dir, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        winners_dir = results_dir / "winners"
        winners_dir.mkdir()
        (winners_dir / "winners.json").write_text(json.dumps([]))
        mock_dir.return_value = results_dir

        metrics = get_promotion_metrics()
        assert isinstance(metrics, dict)
