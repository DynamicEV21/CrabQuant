"""Tests for crabquant.refinement.api_budget — ApiBudgetTracker."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from crabquant.refinement.api_budget import (
    THROTTLE_THRESHOLD,
    ApiBudgetTracker,
    MODEL_PREMIUM,
    MODEL_THROTTLED,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_budget(tmp_path: Path) -> Path:
    return tmp_path / "api_budget.json"


@pytest.fixture()
def tracker(tmp_budget: Path) -> ApiBudgetTracker:
    return ApiBudgetTracker(budget_file=str(tmp_budget))


@pytest.fixture()
def zero_budget_tracker(tmp_budget: Path) -> ApiBudgetTracker:
    return ApiBudgetTracker(budget_file=str(tmp_budget), daily_limit=0, weekly_limit=0)


# ── Helpers ───────────────────────────────────────────────────────────────


def _set_date(target: date) -> None:
    """Patch date.today() to return *target*."""
    patch("crabquant.refinement.api_budget.date", target).start()


# ── Tests ─────────────────────────────────────────────────────────────────


class TestInit:
    """Initialization behaviour."""

    def test_fresh_state(self, tracker: ApiBudgetTracker) -> None:
        assert tracker.daily_count == 0
        assert tracker.weekly_count == 0
        assert tracker.daily_limit == 500
        assert tracker.weekly_limit == 2000

    def test_custom_limits(self, tmp_budget: Path) -> None:
        t = ApiBudgetTracker(budget_file=str(tmp_budget), daily_limit=100, weekly_limit=400)
        assert t.daily_limit == 100
        assert t.weekly_limit == 400

    def test_zero_budget_clamped(self, tmp_budget: Path) -> None:
        t = ApiBudgetTracker(budget_file=str(tmp_budget), daily_limit=-5, weekly_limit=-10)
        assert t.daily_limit == 0
        assert t.weekly_limit == 0


class TestRecordPrompt:
    """record_prompt() increments counters."""

    def test_single_record(self, tracker: ApiBudgetTracker) -> None:
        tracker.record_prompt()
        assert tracker.daily_count == 1
        assert tracker.weekly_count == 1

    def test_multiple_records(self, tracker: ApiBudgetTracker) -> None:
        for _ in range(10):
            tracker.record_prompt()
        assert tracker.daily_count == 10
        assert tracker.weekly_count == 10

    def test_persists_after_record(self, tracker: ApiBudgetTracker, tmp_budget: Path) -> None:
        tracker.record_prompt()
        data = json.loads(tmp_budget.read_text())
        assert data["daily_count"] == 1
        assert data["weekly_count"] == 1

    def test_negative_tokens_treated_as_zero(self, tracker: ApiBudgetTracker) -> None:
        tracker.record_prompt(tokens=-10)
        assert tracker.daily_count == 1

    def test_none_model_accepted(self, tracker: ApiBudgetTracker) -> None:
        tracker.record_prompt(model=None)
        assert tracker.daily_count == 1


class TestShouldThrottle:
    """should_throttle() at 80 % threshold."""

    def test_not_throttled_below_threshold(self, tracker: ApiBudgetTracker) -> None:
        # 80 % of 500 = 400
        for _ in range(399):
            tracker.record_prompt()
        assert not tracker.should_throttle()

    def test_throttled_at_threshold(self, tracker: ApiBudgetTracker) -> None:
        for _ in range(400):
            tracker.record_prompt()
        assert tracker.should_throttle()

    def test_throttled_above_threshold(self, tracker: ApiBudgetTracker) -> None:
        for _ in range(450):
            tracker.record_prompt()
        assert tracker.should_throttle()

    def test_zero_budget_always_throttled(self, zero_budget_tracker: ApiBudgetTracker) -> None:
        assert zero_budget_tracker.should_throttle()


class TestRecommendedModel:
    """get_recommended_model() returns correct model."""

    def test_returns_premium_when_not_throttled(self, tracker: ApiBudgetTracker) -> None:
        tracker.record_prompt()
        assert tracker.get_recommended_model() == MODEL_PREMIUM

    def test_returns_throttled_model_when_throttled(self, tracker: ApiBudgetTracker) -> None:
        # Force throttle by setting daily count directly
        tracker.daily_count = int(500 * THROTTLE_THRESHOLD)
        assert tracker.get_recommended_model() == MODEL_THROTTLED


class TestPersistence:
    """Save / load round-trip."""

    def test_round_trip(self, tmp_budget: Path) -> None:
        t1 = ApiBudgetTracker(budget_file=str(tmp_budget))
        for _ in range(42):
            t1.record_prompt()

        t2 = ApiBudgetTracker(budget_file=str(tmp_budget))
        assert t2.daily_count == 42
        assert t2.weekly_count == 42

    def test_corrupt_file_graceful(self, tmp_budget: Path) -> None:
        tmp_budget.write_text("not json{{{")
        t = ApiBudgetTracker(budget_file=str(tmp_budget))
        assert t.daily_count == 0


class TestResetDaily:
    """reset_daily() clears daily counter."""

    def test_clears_daily(self, tracker: ApiBudgetTracker) -> None:
        for _ in range(50):
            tracker.record_prompt()
        tracker.reset_daily()
        assert tracker.daily_count == 0

    def test_prunes_old_history(self, tracker: ApiBudgetTracker) -> None:
        today = date.today().isoformat()
        old = (date.today() - timedelta(days=10)).isoformat()
        tracker._history = {old: 100, today: 5}
        tracker.reset_daily()
        assert old not in tracker._history

    def test_recomputes_weekly_from_history(self, tracker: ApiBudgetTracker) -> None:
        today = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        tracker._history = {yesterday: 30, today: 20}
        tracker.daily_count = 20
        tracker.reset_daily()
        # After reset: daily_count=0, weekly should include history within 7 days
        assert tracker.daily_count == 0
        assert tracker.weekly_count == 50  # yesterday 30 + today 20


class TestAlertThreshold:
    """Alert threshold (90 %) behaviour."""

    def test_alert_not_active_below_threshold(self, tracker: ApiBudgetTracker) -> None:
        # 90 % of 500 = 450
        for _ in range(449):
            tracker.record_prompt()
        summary = tracker.get_usage_summary()
        assert not summary["alert_active"]

    def test_alert_active_at_threshold(self, tracker: ApiBudgetTracker) -> None:
        for _ in range(450):
            tracker.record_prompt()
        summary = tracker.get_usage_summary()
        assert summary["alert_active"]


class TestWeeklyCounter:
    """Weekly counter accumulation."""

    def test_accumulates_across_records(self, tracker: ApiBudgetTracker) -> None:
        for _ in range(100):
            tracker.record_prompt()
        assert tracker.weekly_count == 100

    def test_weekly_reflects_in_summary(self, tracker: ApiBudgetTracker) -> None:
        for _ in range(100):
            tracker.record_prompt()
        summary = tracker.get_usage_summary()
        assert summary["weekly_count"] == 100
        assert summary["weekly_pct"] == 100 / 2000


class TestUsageSummary:
    """get_usage_summary() completeness."""

    def test_keys_present(self, tracker: ApiBudgetTracker) -> None:
        summary = tracker.get_usage_summary()
        expected_keys = {
            "daily_count", "daily_limit", "daily_pct",
            "weekly_count", "weekly_limit", "weekly_pct",
            "throttled", "recommended_model", "alert_active",
            "last_reset_date", "today", "history",
        }
        assert set(summary.keys()) == expected_keys

    def test_history_in_summary(self, tracker: ApiBudgetTracker) -> None:
        tracker.record_prompt()
        summary = tracker.get_usage_summary()
        today = date.today().isoformat()
        assert today in summary["history"]


class TestEdgeCases:
    """Edge cases: zero budget, negative inputs."""

    def test_zero_budget_throttle_and_alert(self, zero_budget_tracker: ApiBudgetTracker) -> None:
        assert zero_budget_tracker.should_throttle() is True
        summary = zero_budget_tracker.get_usage_summary()
        assert summary["throttled"] is True
        assert summary["alert_active"] is True
        assert summary["recommended_model"] == MODEL_THROTTLED

    def test_negative_token_count_no_crash(self, tracker: ApiBudgetTracker) -> None:
        tracker.record_prompt(tokens=-999)
        assert tracker.daily_count == 1
