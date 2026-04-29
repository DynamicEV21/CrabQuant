"""Tests for crabquant.refinement.status_reporter — 8+ unit + smoke tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from crabquant.refinement.status_reporter import StatusReporter


# ── helpers ───────────────────────────────────────────────────────────────


def _make_reporter(tmp_path: Path, with_files: str = "") -> StatusReporter:
    """Create a StatusReporter pointed at a temp results directory.

    *with_files* can be a space-separated list:
      ``daemon``  — write a minimal daemon_state.json
      ``history`` — write a 3-entry run_history.jsonl
      ``winners`` — write 5 winning strategy JSONs
      ``budget``  — write a minimal api_budget.json
    """
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)

    wanted = set(with_files.split()) if with_files else set()

    if "daemon" in wanted:
        state = {
            "daemon_id": "test-uuid",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "last_wave_completed": "",
            "current_wave": 11,
            "total_mandates_run": 23,
            "total_strategies_promoted": 3,
            "total_api_calls": 347,
            "pending_mandates": [],
            "completed_mandates": [],
            "failed_mandates": [],
            "last_error": None,
            "shutdown_requested": False,
            "api_budget_used_today": 347,
            "api_budget_throttled": False,
        }
        (results / "daemon_state.json").write_text(json.dumps(state))

    if "history" in wanted:
        entries = [
            {"mandate": "m1", "success": True, "sharpe": 1.2, "timestamp": "2026-04-29T00:00:00Z"},
            {"mandate": "m2", "success": True, "sharpe": 0.8, "timestamp": "2026-04-29T01:00:00Z"},
            {"mandate": "m3", "success": False, "sharpe": 0.0, "failure_mode": "crash", "timestamp": "2026-04-29T02:00:00Z"},
        ]
        with open(results / "run_history.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    if "winners" in wanted:
        winners_dir = results / "winning_strategies"
        winners_dir.mkdir()
        for i, (name, ticker, sharpe) in enumerate([
            ("roc_ema_volume", "AAPL", 1.23),
            ("momentum_breakout", "MSFT", 1.15),
            ("mean_reversion_bb", "GOOG", 1.05),
            ("trend_follow_adx", "AMZN", 0.98),
            ("rsi_divergence", "TSLA", 0.87),
        ]):
            (winners_dir / f"strat_{i}.json").write_text(
                json.dumps({"strategy_name": name, "ticker": ticker, "sharpe": sharpe})
            )

    if "budget" in wanted:
        (results / "api_budget.json").write_text(
            json.dumps({
                "last_reset_date": "2026-04-29",
                "daily_count": 347,
                "weekly_count": 1823,
                "history": {},
            })
        )

    return StatusReporter(results_dir=str(results))


# ── tests ─────────────────────────────────────────────────────────────────


class TestReportGeneration:
    """Report generation with various data configurations."""

    def test_full_report_with_all_data(self, tmp_path):
        """Generate report when all data sources are available."""
        reporter = _make_reporter(tmp_path, with_files="daemon history winners budget")

        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget"
        ) as mock_budget_import, mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter"
        ) as mock_resource_import, mock.patch(
            "crabquant.refinement.api_budget.ApiBudgetTracker"
        ) as MockBudget, mock.patch(
            "crabquant.refinement.resource_limiter.ResourceLimiter"
        ) as MockResource:
            mock_budget_import.return_value = MockBudget
            mock_resource_import.return_value = MockResource
            MockBudget.return_value.get_usage_summary.return_value = {
                "daily_count": 347,
                "daily_limit": 500,
                "daily_pct": 0.694,
                "weekly_count": 1823,
                "weekly_limit": 2000,
                "weekly_pct": 0.9115,
                "throttled": False,
                "recommended_model": "zai/glm-5-turbo",
                "alert_active": False,
                "last_reset_date": "2026-04-29",
                "today": "2026-04-29",
                "history": {},
            }
            MockResource.return_value.get_status_summary.return_value = {
                "cpu_percent": 45.0,
                "ram_free_gb": 8.2,
                "disk_free_gb": 120.5,
                "recommended_parallel": 3,
                "should_pause": False,
                "max_parallel": 4,
                "min_ram_gb": 2.0,
                "min_disk_gb": 10.0,
                "cpu_threshold": 85.0,
            }

            report = reporter.generate_report()

        assert "🦀 CrabQuant Daily Report" in report
        assert "healthy ✅" in report
        assert "Wave: 11" in report
        assert "Mandates: 23" in report
        assert "Promoted: 3" in report
        assert "roc_ema_volume" in report
        assert "AAPL" in report
        assert "Sharpe 1.23" in report

    def test_report_with_minimal_data(self, tmp_path):
        """Report still generates when no data files exist."""
        reporter = _make_reporter(tmp_path, with_files="")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget",
            return_value=None,
        ), mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter",
            return_value=None,
        ):
            report = reporter.generate_report()

        assert "🦀 CrabQuant Daily Report" in report
        assert "No daemon state found" in report


class TestTelegramFormat:
    """Telegram markdown formatting and char limit."""

    def test_under_char_limit(self, tmp_path):
        """Report must be under 4096 characters."""
        reporter = _make_reporter(tmp_path, with_files="daemon history winners budget")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget",
            return_value=None,
        ), mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter",
            return_value=None,
        ):
            report = reporter.generate_report()

        assert len(report) <= 4096

    def test_has_markdown_bold(self, tmp_path):
        """Report uses Telegram markdown bold markers."""
        reporter = _make_reporter(tmp_path, with_files="daemon")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget",
            return_value=None,
        ), mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter",
            return_value=None,
        ):
            report = reporter.generate_report()

        assert "**Daemon**" in report
        assert "**API Budget**" in report
        assert "**Resources**" in report

    def test_truncation_with_many_strategies(self, tmp_path):
        """When there are too many strategies, report still fits in limit."""
        reporter = _make_reporter(tmp_path, with_files="daemon")
        # Create 200 winning strategies to stress-test truncation
        winners_dir = Path(reporter._results_dir) / "winning_strategies"
        winners_dir.mkdir(exist_ok=True)
        for i in range(200):
            (winners_dir / f"s_{i}.json").write_text(
                json.dumps({
                    "strategy_name": f"very_long_strategy_name_number_{i}",
                    "ticker": "TICK",
                    "sharpe": 2.0 - i * 0.01,
                })
            )

        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget",
            return_value=None,
        ), mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter",
            return_value=None,
        ):
            report = reporter.generate_report()

        assert len(report) <= 4096
        assert "truncated" in report or "more" in report


class TestMissingDataSources:
    """Graceful handling of missing modules and files."""

    def test_budget_na_when_import_fails(self, tmp_path):
        """API budget shows N/A when ApiBudgetTracker can't be imported."""
        reporter = _make_reporter(tmp_path, with_files="daemon")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget",
            return_value=None,
        ), mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter",
            return_value=None,
        ):
            report = reporter.generate_report()

        assert "N/A" in report
        # Budget section should exist but show N/A
        assert "💰" in report

    def test_resources_na_when_import_fails(self, tmp_path):
        """Resources show N/A when ResourceLimiter can't be imported."""
        reporter = _make_reporter(tmp_path, with_files="daemon")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget",
            return_value=None,
        ), mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter",
            return_value=None,
        ):
            report = reporter.generate_report()

        assert "N/A" in report
        assert "💻" in report


class TestDaemonHealthParsing:
    """Daemon health section reads state correctly."""

    def test_healthy_daemon(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="daemon")
        health = reporter._get_daemon_health()

        assert health["available"] is True
        assert "healthy" in health["status"]
        assert health["wave"] == 11
        assert health["total_mandates"] == 23
        assert health["promoted"] == 3

    def test_missing_daemon_state(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="")
        health = reporter._get_daemon_health()

        assert health["available"] is False
        assert health["status"] == "N/A"

    def test_degraded_daemon_with_error(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="")
        # Write state with error
        state = {
            "daemon_id": "x",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "last_wave_completed": "",
            "current_wave": 5,
            "total_mandates_run": 10,
            "total_strategies_promoted": 1,
            "total_api_calls": 200,
            "pending_mandates": [],
            "completed_mandates": [],
            "failed_mandates": [],
            "last_error": "Connection timeout to API",
            "shutdown_requested": False,
            "api_budget_used_today": 200,
            "api_budget_throttled": False,
        }
        (Path(reporter._results_dir) / "daemon_state.json").write_text(json.dumps(state))
        health = reporter._get_daemon_health()

        assert health["available"] is True
        assert "degraded" in health["status"]
        assert health["last_error"] == "Connection timeout to API"


class TestBudgetStatusParsing:
    """Budget status section reads tracker correctly."""

    def test_budget_summary_parsed(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="budget")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget"
        ) as mock_import, mock.patch(
            "crabquant.refinement.api_budget.ApiBudgetTracker"
        ) as MockBudget:
            mock_import.return_value = MockBudget
            MockBudget.return_value.get_usage_summary.return_value = {
                "daily_count": 400,
                "daily_limit": 500,
                "daily_pct": 0.80,
                "weekly_count": 1900,
                "weekly_limit": 2000,
                "weekly_pct": 0.95,
                "throttled": True,
                "recommended_model": "zai/glm-4.7",
                "alert_active": True,
                "last_reset_date": "2026-04-29",
                "today": "2026-04-29",
                "history": {},
            }

            budget = reporter._get_budget_status()

        assert budget["available"] is True
        assert budget["daily_count"] == 400
        assert budget["weekly_pct"] == 0.95
        assert budget["throttled"] is True

    def test_budget_unavailable_without_file(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget",
            return_value=None,
        ):
            budget = reporter._get_budget_status()

        assert budget["available"] is False


class TestResourceStatusParsing:
    """Resource status section reads limiter correctly."""

    def test_resource_summary_parsed(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter"
        ) as mock_import, mock.patch(
            "crabquant.refinement.resource_limiter.ResourceLimiter"
        ) as MockResource:
            mock_import.return_value = MockResource
            MockResource.return_value.get_status_summary.return_value = {
                "cpu_percent": 72.5,
                "ram_free_gb": 6.1,
                "disk_free_gb": 200.0,
                "recommended_parallel": 2,
                "should_pause": False,
                "max_parallel": 4,
                "min_ram_gb": 2.0,
                "min_disk_gb": 10.0,
                "cpu_threshold": 85.0,
            }

            resources = reporter._get_resource_status()

        assert resources["available"] is True
        assert resources["cpu_percent"] == 72.5
        assert resources["recommended_parallel"] == 2

    def test_resource_unavailable_without_module(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter",
            return_value=None,
        ):
            resources = reporter._get_resource_status()

        assert resources["available"] is False


class TestEmptyResultsDirectory:
    """Entirely empty results directory doesn't crash."""

    def test_empty_dir_report(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget",
            return_value=None,
        ), mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter",
            return_value=None,
        ):
            report = reporter.generate_report()

        assert isinstance(report, str)
        assert len(report) > 0
        assert len(report) <= 4096
        assert "🦀" in report

    def test_nonexistent_results_dir(self):
        """StatusReporter handles a completely nonexistent directory."""
        reporter = StatusReporter(results_dir="/tmp/nonexistent_crabquant_results_xyz")
        with mock.patch(
            "crabquant.refinement.status_reporter._try_import_api_budget",
            return_value=None,
        ), mock.patch(
            "crabquant.refinement.status_reporter._try_import_resource_limiter",
            return_value=None,
        ):
            report = reporter.generate_report()

        assert isinstance(report, str)
        assert len(report) <= 4096


class TestIssuesCollection:
    """Issues section collects warnings correctly."""

    def test_high_budget_triggers_warning(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="daemon")
        budget = {"available": True, "weekly_pct": 0.93, "daily_pct": 0.50, "throttled": False}
        resources = {"available": True, "should_pause": False, "cpu_percent": 30.0}
        daemon = {"available": True, "status": "healthy ✅", "last_error": None}

        issues = reporter._collect_issues(budget, resources, daemon)
        assert any("91%" in i or "93%" in i for i in issues)

    def test_no_issues_when_all_ok(self, tmp_path):
        reporter = _make_reporter(tmp_path, with_files="daemon")
        budget = {"available": True, "weekly_pct": 0.50, "daily_pct": 0.30, "throttled": False}
        resources = {"available": True, "should_pause": False, "cpu_percent": 30.0}
        daemon = {"available": True, "status": "healthy ✅", "last_error": None}

        issues = reporter._collect_issues(budget, resources, daemon)
        assert len(issues) == 0
