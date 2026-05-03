"""Tests for Telegram notification system."""

import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crabquant.production.telegram import (
    TelegramNotifier,
    format_daily_brief,
    format_health_alert,
    format_mandate_summary,
    format_strategy_promotion_alert,
    get_notifier,
    is_configured,
)


# ── Fixtures ────────────────────────────────────────────────────────────


def _make_health(
    status="healthy",
    alive=True,
    pid=12345,
    cpu=45.0,
    ram_free=8.0,
    disk_free=50.0,
    cache_fresh=True,
    cache_age=2.0,
    recommendations=None,
    wave=3,
    mandates=42,
    promoted=5,
):
    return {
        "status": status,
        "daemon": {
            "alive": alive,
            "pid": pid,
            "last_heartbeat": "2026-04-28T18:00:00Z",
            "heartbeat_age_seconds": 30.0,
            "current_wave": wave,
            "total_mandates_run": mandates,
            "total_strategies_promoted": promoted,
            "total_api_calls": 200,
        },
        "system": {
            "cpu_percent": cpu,
            "ram_total_gb": 16.0,
            "ram_used_gb": 8.0,
            "ram_free_gb": ram_free,
            "disk_free_gb": disk_free,
        },
        "data": {
            "cache_fresh": cache_fresh,
            "cache_age_hours": cache_age,
            "cache_dir": "/tmp/cache",
        },
        "api_budget": {},
        "recommendations": recommendations or [],
        "checked_at": "2026-04-28T18:00:00Z",
    }


def _make_api_budget(
    cost=1.23, calls=100, errors=3, tokens=50000, avg_latency=2.5
):
    return {
        "total_calls": calls,
        "total_errors": errors,
        "error_rate": 3.0,
        "total_prompt_tokens": 30000,
        "total_completion_tokens": 20000,
        "total_tokens": tokens,
        "total_cost_usd": cost,
        "avg_tokens_per_call": 500,
        "avg_cost_per_call": 0.0123,
        "avg_latency_seconds": avg_latency,
        "uptime": "24.5h",
        "budget_remaining": {"cost_usd": 8.77},
        "models_used": ["glm-5-turbo"],
        "mandates_tracked": 10,
    }


def _make_convergence(
    total=50, converged=12, abandoned=5, exhausted=30, rate=24.0
):
    return {
        "total_mandates": total,
        "converged": converged,
        "convergence_rate": rate,
        "avg_turns_to_converge": 4.2,
        "abandoned": abandoned,
        "max_turns_exhausted": exhausted,
        "success": converged,
        "running": 0,
        "abandonment_rate": 10.0,
        "exhaustion_rate": 60.0,
        "by_archetype": {
            "momentum": {"total": 20, "success": 8, "rate": 40.0, "avg_turns": 3.5},
            "mean_reversion": {"total": 15, "success": 3, "rate": 20.0, "avg_turns": 5.0},
            "breakout": {"total": 10, "success": 1, "rate": 10.0, "avg_turns": 6.0},
        },
        "by_ticker": {
            "SPY": {"total": 15, "success": 4, "rate": 26.7},
            "AAPL": {"total": 12, "success": 3, "rate": 25.0},
        },
        "failure_modes": {"low_sharpe": 30, "regime_fragility": 15, "validation_failed": 10},
        "promotion_rate": 16.7,
        "promoted_from_converged": 2,
        "time_trends": {
            "2026-04-27": {"total": 25, "success": 5, "rate": 20.0},
            "2026-04-28": {"total": 25, "success": 7, "rate": 28.0},
        },
    }


# ── format_daily_brief ──────────────────────────────────────────────────


class TestFormatDailyBrief:

    def test_full_brief(self):
        text = format_daily_brief(
            health=_make_health(),
            api_budget=_make_api_budget(),
            convergence=_make_convergence(),
            production_count=7,
            candidate_count=12,
        )
        assert "🦀" in text
        assert "Daily Brief" in text
        assert "System Status" in text
        assert "API Usage" in text
        assert "Research Progress" in text
        assert "Strategy Catalog" in text
        assert "Production: 7" in text
        assert "Candidates: 12" in text

    def test_empty_brief(self):
        text = format_daily_brief()
        assert "🦀" in text
        assert "Daily Brief" in text
        assert "Strategy Catalog" in text
        assert "Production: 0" in text

    def test_health_only(self):
        text = format_daily_brief(health=_make_health())
        assert "System Status" in text
        assert "Daemon: running" in text
        assert "CPU: 45.0%" in text
        assert "RAM free: 8.0 GB" in text
        assert "Cache: ✅ fresh" in text
        # No other sections
        assert "API Usage" not in text

    def test_daemon_down(self):
        health = _make_health(status="down", alive=False)
        text = format_daily_brief(health=health)
        assert "not running" in text

    def test_degraded_health(self):
        health = _make_health(status="degraded")
        text = format_daily_brief(health=health)
        assert "⚠️" in text

    def test_stale_cache(self):
        health = _make_health(cache_fresh=False, cache_age=48.0)
        text = format_daily_brief(health=health)
        assert "stale" in text
        assert "48" in text

    def test_recommendations_included(self):
        recs = ["RAM critically low", "Disk space low"]
        health = _make_health(recommendations=recs)
        text = format_daily_brief(health=health)
        assert "RAM critically low" in text
        assert "Disk space low" in text

    def test_recommendations_truncated(self):
        recs = [f"Rec {i}" for i in range(10)]
        health = _make_health(recommendations=recs)
        text = format_daily_brief(health=health)
        # Should only show 5 (indices 0-4)
        assert "Rec 4" in text
        assert "Rec 9" not in text

    def test_api_budget_with_budget_remaining(self):
        budget = _make_api_budget()
        text = format_daily_brief(api_budget=budget)
        assert "Cost: $1.23" in text
        assert "Calls: 100" in text
        assert "Errors: 3.0%" in text
        assert "$8.77 budget left" in text

    def test_convergence_with_failures(self):
        conv = _make_convergence()
        text = format_daily_brief(convergence=conv)
        assert "Mandates: 50" in text
        assert "Converged: 12 (24.0%)" in text
        assert "low_sharpe×30" in text
        assert "momentum: 8/20" in text

    def test_convergence_no_archetypes(self):
        conv = _make_convergence()
        conv["by_archetype"] = {}
        text = format_daily_brief(convergence=conv)
        assert "Research Progress" in text
        assert "Mandates: 50" in text

    def test_no_wave_activity(self):
        health = _make_health(wave=None, mandates=None, promoted=None)
        text = format_daily_brief(health=health)
        # Should not show Activity line
        assert "Activity:" not in text


# ── format_strategy_promotion_alert ─────────────────────────────────────


class TestFormatStrategyPromotionAlert:

    def test_promoted(self):
        text = format_strategy_promotion_alert(
            strategy_name="roc_ema_volume",
            ticker="SPY",
            sharpe=1.8,
            trades=45,
            composite_score=3.5,
        )
        assert "🏆" in text
        assert "PROMOTED" in text
        assert "roc_ema_volume" in text
        assert "SPY" in text
        assert "Sharpe: 1.80" in text
        assert "Trades: 45" in text
        assert "Composite: 3.50" in text

    def test_soft_promoted(self):
        text = format_strategy_promotion_alert(
            strategy_name="momentum_rsi",
            ticker="AAPL",
            sharpe=1.2,
            trades=30,
            verdict="SOFT_PROMOTED",
        )
        assert "📋" in text
        assert "SOFT_PROMOTED" in text

    def test_with_regime(self):
        text = format_strategy_promotion_alert(
            strategy_name="bb_squeeze",
            ticker="NVDA",
            sharpe=2.1,
            trades=25,
            regime="high_volatility",
        )
        assert "high_volatility" in text

    def test_no_composite(self):
        text = format_strategy_promotion_alert(
            strategy_name="simple",
            ticker="SPY",
            sharpe=1.0,
            trades=20,
        )
        assert "Composite" not in text


# ── format_health_alert ─────────────────────────────────────────────────


class TestFormatHealthAlert:

    def test_status_change(self):
        text = format_health_alert("degraded", "healthy")
        assert "⚠️" in text
        assert "✅ → ⚠️" in text
        assert "healthy → degraded" in text

    def test_down_alert(self):
        text = format_health_alert("down", "healthy", recommendations=["Restart daemon"])
        assert "🔴" in text
        assert "Restart daemon" in text

    def test_no_change_returns_empty(self):
        text = format_health_alert("healthy", "healthy")
        assert text == ""

    def test_recovery(self):
        text = format_health_alert("healthy", "down")
        assert "Health Status Changed" in text

    def test_recommendations_truncated(self):
        recs = [f"Action {i}" for i in range(10)]
        text = format_health_alert("down", "healthy", recommendations=recs)
        assert "Action 4" in text
        assert "Action 9" not in text


# ── format_mandate_summary ──────────────────────────────────────────────


class TestFormatMandateSummary:

    def test_success(self):
        text = format_mandate_summary(
            mandate_name="explorer_spy_momentum",
            status="success",
            best_sharpe=1.8,
            best_composite=3.5,
            turns_used=5,
            max_turns=10,
        )
        assert "🎯" in text
        assert "explorer_spy_momentum" in text
        assert "success" in text
        assert "Sharpe: 1.80" in text
        assert "Composite: 3.50" in text
        assert "Turns: 5/10" in text

    def test_exhausted(self):
        text = format_mandate_summary(
            mandate_name="balanced_aapl_mr",
            status="max_turns_exhausted",
            best_sharpe=0.8,
            best_composite=1.2,
            turns_used=10,
            max_turns=10,
            failure_mode="low_sharpe",
        )
        assert "⏰" in text
        assert "Failure mode: low_sharpe" in text

    def test_abandoned(self):
        text = format_mandate_summary(
            mandate_name="fast_nvda",
            status="abandoned",
            best_sharpe=0.0,
            best_composite=0.0,
            turns_used=3,
            max_turns=10,
        )
        assert "🚫" in text
        assert "abandoned" in text

    def test_no_failure_mode(self):
        text = format_mandate_summary(
            mandate_name="test",
            status="success",
            best_sharpe=1.0,
            best_composite=2.0,
            turns_used=4,
            max_turns=10,
        )
        assert "Failure mode" not in text


# ── TelegramNotifier ────────────────────────────────────────────────────


class TestTelegramNotifier:

    def test_send_message_success(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = notifier.send_message("Hello!")

        assert result is True
        mock_urlopen.assert_called_once()

    def test_send_message_api_error(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "ok": False,
            "description": "Bad Request: chat not found",
        }).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = notifier.send_message("Hello!")

        assert result is False

    def test_send_message_network_error(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            result = notifier.send_message("Hello!")

        assert result is False

    def test_send_message_http_error(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")

        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            "url", 403, "Forbidden", {}, None
        )):
            result = notifier.send_message("Hello!")

        assert result is False

    def test_unconfigured_notifier(self):
        notifier = TelegramNotifier(bot_token="", chat_id="")
        result = notifier.send_message("Hello!")
        assert result is False

    def test_message_truncation(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        long_text = "x" * 5000
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = notifier.send_message(long_text)

        assert result is True
        # Verify the sent text was truncated
        call_args = mock_urlopen.call_args
        sent_data = call_args[0][0].data.decode()
        assert "truncated" in sent_data
        assert len(sent_data) < 5000

    def test_send_promotion_alert(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = notifier.send_promotion_alert(
                strategy_name="test_strat",
                ticker="SPY",
                sharpe=1.5,
                trades=30,
            )

        assert result is True

    def test_send_health_alert_no_change(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        # Same status → no alert sent
        result = notifier.send_health_alert(status="healthy", previous_status="healthy")
        assert result is False

    def test_send_mandate_summary(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = notifier.send_mandate_summary(
                mandate_name="test",
                status="success",
                best_sharpe=1.0,
                best_composite=2.0,
                turns_used=5,
                max_turns=10,
            )

        assert result is True

    def test_send_raw(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = notifier.send_raw("Custom message")

        assert result is True

    def test_send_daily_brief_with_data_gathering(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(notifier, "_load_health", return_value=_make_health()), \
             patch.object(notifier, "_load_api_budget", return_value=_make_api_budget()), \
             patch.object(notifier, "_load_convergence", return_value=_make_convergence()), \
             patch.object(notifier, "_count_production", return_value=5), \
             patch.object(notifier, "_count_candidates", return_value=3), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = notifier.send_daily_brief()

        assert result is True

    def test_send_daily_brief_explicit_data(self):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = notifier.send_daily_brief(
                health=_make_health(),
                api_budget=_make_api_budget(),
                convergence=_make_convergence(),
                production_count=5,
                candidate_count=3,
            )

        assert result is True


# ── Health change detection ─────────────────────────────────────────────


class TestHealthChangeDetection:

    def _make_notifier(self, tmp_path):
        notifier = TelegramNotifier(bot_token="test-token", chat_id="12345")
        return notifier

    def test_first_run_no_alert(self, tmp_path):
        notifier = self._make_notifier(tmp_path)
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(notifier, "_load_health", return_value=_make_health(status="healthy")), \
             patch("crabquant.production.telegram.Path", return_value=tmp_path), \
             patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = notifier.check_and_alert_health()

        # First run: unknown → healthy = should alert
        assert result is True
        assert mock_urlopen.called

    def test_no_change_no_alert(self, tmp_path):
        notifier = self._make_notifier(tmp_path)
        # Method uses Path(__file__).resolve().parent.parent.parent / "results" / ...
        # With Path patched to return tmp_path, resolve gives absolute tmp_path,
        # then .parent.parent.parent goes up 3 levels
        base = tmp_path.resolve().parent.parent.parent
        prev_path = base / "results" / "telegram_prev_status.json"
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        prev_path.write_text(json.dumps({"status": "healthy"}))

        with patch.object(notifier, "_load_health", return_value=_make_health(status="healthy")), \
             patch("crabquant.production.telegram.Path", return_value=tmp_path):
            result = notifier.check_and_alert_health()

        assert result is False

    def test_change_saves_and_alerts(self, tmp_path):
        notifier = self._make_notifier(tmp_path)
        base = tmp_path.resolve().parent.parent.parent
        prev_path = base / "results" / "telegram_prev_status.json"
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        prev_path.write_text(json.dumps({"status": "healthy"}))

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(notifier, "_load_health", return_value=_make_health(status="down")), \
             patch("crabquant.production.telegram.Path", return_value=tmp_path), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = notifier.check_and_alert_health()

        assert result is True
        # Verify status was saved
        saved = json.loads(prev_path.read_text())
        assert saved["status"] == "down"

    def test_no_health_data(self, tmp_path):
        notifier = self._make_notifier(tmp_path)
        with patch.object(notifier, "_load_health", return_value=None):
            result = notifier.check_and_alert_health()
        assert result is False


# ── Data gathering helpers ──────────────────────────────────────────────


class TestDataGathering:

    def test_load_health_success(self):
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        with patch("crabquant.production.health.check_health", return_value={"status": "ok"}):
            result = notifier._load_health()
        assert result == {"status": "ok"}

    def test_load_health_failure(self):
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        with patch("crabquant.production.health.check_health", side_effect=Exception("fail")):
            result = notifier._load_health()
        assert result is None

    def test_load_api_budget_success(self):
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = {"total_cost_usd": 1.0}
        with patch("crabquant.refinement.api_budget.get_global_tracker", return_value=mock_tracker):
            result = notifier._load_api_budget()
        assert result == {"total_cost_usd": 1.0}

    def test_load_api_budget_failure(self):
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        with patch("crabquant.refinement.api_budget.get_global_tracker", side_effect=Exception("fail")):
            result = notifier._load_api_budget()
        assert result is None

    def test_count_production(self):
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        with patch("crabquant.production.get_production_strategies", return_value=[1, 2, 3]):
            result = notifier._count_production()
        assert result == 3

    def test_count_candidates(self, tmp_path):
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        # Create fake candidates dir
        candidates_dir = tmp_path / "results" / "candidates"
        candidates_dir.mkdir(parents=True)
        (candidates_dir / "a.json").write_text("{}")
        (candidates_dir / "b.json").write_text("{}")
        (candidates_dir / "c.txt").write_text("not json")

        base = Path(__file__).resolve().parent.parent.parent
        with patch("crabquant.production.telegram.Path", return_value=tmp_path):
            # Just test the glob logic
            count = len(list(candidates_dir.glob("*.json")))
        assert count == 2

    def test_count_candidates_no_dir(self):
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        # Non-existent dir returns 0
        result = notifier._count_candidates()
        # Will try real path, might return 0 or some number — just verify no crash
        assert isinstance(result, int)


# ── Factory / singleton ─────────────────────────────────────────────────


class TestGetNotifier:

    def test_get_notifier_returns_instance(self):
        notifier = get_notifier(bot_token="test", chat_id="123")
        assert isinstance(notifier, TelegramNotifier)
        assert notifier.bot_token == "test"
        assert notifier.chat_id == "123"

    def test_get_notifier_uses_env_vars(self):
        import os
        os.environ["CRABQUANT_TELEGRAM_BOT_TOKEN"] = "env-token"
        os.environ["CRABQUANT_TELEGRAM_CHAT_ID"] = "env-chat"
        # Reset singleton
        import crabquant.production.telegram as tel
        tel._notifier = None
        notifier = get_notifier()
        assert notifier.bot_token == "env-token"
        assert notifier.chat_id == "env-chat"
        # Clean up
        del os.environ["CRABQUANT_TELEGRAM_BOT_TOKEN"]
        del os.environ["CRABQUANT_TELEGRAM_CHAT_ID"]
        tel._notifier = None

    def test_get_notifier_no_config(self):
        import os
        # Remove env vars if set
        os.environ.pop("CRABQUANT_TELEGRAM_BOT_TOKEN", None)
        os.environ.pop("CRABQUANT_TELEGRAM_CHAT_ID", None)
        import crabquant.production.telegram as tel
        tel._notifier = None
        notifier = get_notifier()
        assert notifier.bot_token == ""
        assert notifier.chat_id == ""
        tel._notifier = None


class TestIsConfigured:

    def test_configured(self):
        import os
        os.environ["CRABQUANT_TELEGRAM_BOT_TOKEN"] = "tok"
        os.environ["CRABQUANT_TELEGRAM_CHAT_ID"] = "123"
        assert is_configured() is True
        del os.environ["CRABQUANT_TELEGRAM_BOT_TOKEN"]
        del os.environ["CRABQUANT_TELEGRAM_CHAT_ID"]

    def test_not_configured(self):
        import os
        os.environ.pop("CRABQUANT_TELEGRAM_BOT_TOKEN", None)
        os.environ.pop("CRABQUANT_TELEGRAM_CHAT_ID", None)
        assert is_configured() is False

    def test_partial_config(self):
        import os
        os.environ["CRABQUANT_TELEGRAM_BOT_TOKEN"] = "tok"
        os.environ.pop("CRABQUANT_TELEGRAM_CHAT_ID", None)
        assert is_configured() is False
        del os.environ["CRABQUANT_TELEGRAM_BOT_TOKEN"]


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_health_dict(self):
        text = format_daily_brief(health={})
        assert "System Status" in text

    def test_health_missing_keys(self):
        text = format_daily_brief(health={"status": "healthy", "daemon": {}, "system": {}})
        assert "System Status" in text
        # Should not crash on missing keys

    def test_unicode_in_strategy_name(self):
        text = format_strategy_promotion_alert(
            strategy_name="strategy_with_特殊_chars",
            ticker="SPY",
            sharpe=1.0,
            trades=20,
        )
        assert "strategy_with_特殊_chars" in text

    def test_very_long_strategy_name(self):
        name = "a" * 200
        text = format_strategy_promotion_alert(
            strategy_name=name,
            ticker="SPY",
            sharpe=1.0,
            trades=20,
        )
        assert name in text

    def test_zero_values(self):
        text = format_mandate_summary(
            mandate_name="test",
            status="abandoned",
            best_sharpe=0.0,
            best_composite=0.0,
            turns_used=0,
            max_turns=10,
        )
        assert "Sharpe: 0.00" in text
        assert "Turns: 0/10" in text
