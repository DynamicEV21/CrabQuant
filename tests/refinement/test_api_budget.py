"""
Tests for crabquant.refinement.api_budget module.

Tests API call recording, budget enforcement, persistence, and summaries.
"""

import json
import time
from pathlib import Path

import pytest

from crabquant.refinement.api_budget import (
    ApiBudgetState,
    ApiBudgetTracker,
    BudgetExceededError,
)


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture
def tracker(tmp_path):
    """Create a tracker with a temp state file."""
    state_path = tmp_path / "api_budget.json"
    return ApiBudgetTracker(state_path=str(state_path))


def _record_success(tracker, **kwargs):
    """Record a successful API call with defaults."""
    return tracker.record_call(
        model=kwargs.get("model", "glm-5-turbo"),
        prompt_tokens=kwargs.get("prompt_tokens", 1000),
        completion_tokens=kwargs.get("completion_tokens", 500),
        latency_seconds=kwargs.get("latency_seconds", 2.0),
        success=True,
        mandate_id=kwargs.get("mandate_id", ""),
        turn=kwargs.get("turn", 0),
    )


def _record_error(tracker, **kwargs):
    """Record a failed API call."""
    return tracker.record_call(
        model=kwargs.get("model", "glm-5-turbo"),
        prompt_tokens=kwargs.get("prompt_tokens", 1000),
        completion_tokens=kwargs.get("completion_tokens", 0),
        latency_seconds=kwargs.get("latency_seconds", 30.0),
        success=False,
        error=kwargs.get("error", "timeout"),
        mandate_id=kwargs.get("mandate_id", ""),
        turn=kwargs.get("turn", 0),
    )


# ── Basic recording ──────────────────────────────────────────────────────


class TestRecordCall:
    def test_single_call(self, tracker):
        record = _record_success(tracker)
        assert record.model == "glm-5-turbo"
        assert record.prompt_tokens == 1000
        assert record.completion_tokens == 500
        assert record.total_tokens == 1500
        assert record.success is True
        assert record.cost_usd > 0

    def test_multiple_calls_accumulate(self, tracker):
        _record_success(tracker, prompt_tokens=100, completion_tokens=50)
        _record_success(tracker, prompt_tokens=200, completion_tokens=100)
        s = tracker.state
        assert s.total_calls == 2
        assert s.total_prompt_tokens == 300
        assert s.total_completion_tokens == 150
        assert s.total_tokens == 450

    def test_error_call_tracked(self, tracker):
        _record_error(tracker, error="timeout")
        s = tracker.state
        assert s.total_calls == 1
        assert s.total_errors == 1

    def test_error_rate(self, tracker):
        _record_success(tracker)
        _record_error(tracker)
        _record_error(tracker)
        summary = tracker.get_summary()
        assert summary["error_rate"] == pytest.approx(66.7, abs=0.1)

    def test_cost_calculation(self, tracker):
        """Cost should be prompt * rate_prompt + completion * rate_completion."""
        _record_success(tracker, prompt_tokens=1_000_000, completion_tokens=0)
        s = tracker.state
        # Default: $2/MTok prompt, $8/MTok completion
        assert s.total_cost_usd == pytest.approx(2.0, abs=0.01)

    def test_record_has_timestamp(self, tracker):
        record = _record_success(tracker)
        assert record.timestamp is not None
        assert len(record.timestamp) > 10  # ISO format


# ── Budget enforcement ───────────────────────────────────────────────────


class TestBudgetEnforcement:
    def test_cost_limit_enforced(self, tracker):
        tracker.set_budget(max_cost_usd=0.01)
        # First call should succeed (small cost)
        _record_success(tracker, prompt_tokens=100, completion_tokens=50)
        # Second call should eventually exceed
        with pytest.raises(BudgetExceededError, match="Cost limit"):
            tracker.record_call(
                model="glm-5-turbo",
                prompt_tokens=10_000_000,
                completion_tokens=1_000_000,
                latency_seconds=1.0,
                success=True,
            )

    def test_call_limit_enforced(self, tracker):
        tracker.set_budget(max_calls=2)
        _record_success(tracker)
        _record_success(tracker)
        with pytest.raises(BudgetExceededError, match="Call limit"):
            _record_success(tracker)

    def test_no_limit_by_default(self, tracker):
        """Should allow unlimited calls when no budget set."""
        for _ in range(100):
            _record_success(tracker, prompt_tokens=1000, completion_tokens=500)
        assert tracker.state.total_calls == 100

    def test_budget_remaining(self, tracker):
        tracker.set_budget(max_cost_usd=1.0, max_calls=100)
        _record_success(tracker, prompt_tokens=100, completion_tokens=50)
        remaining = tracker.get_summary()["budget_remaining"]
        assert "cost_usd" in remaining
        assert "calls" in remaining
        assert remaining["cost_usd"] < 1.0
        assert remaining["calls"] == 99

    def test_budget_remaining_no_limit(self, tracker):
        _record_success(tracker)
        remaining = tracker.get_summary()["budget_remaining"]
        assert remaining == {}


# ── Per-model tracking ───────────────────────────────────────────────────


class TestModelTracking:
    def test_single_model(self, tracker):
        _record_success(tracker, model="glm-5-turbo")
        assert "glm-5-turbo" in tracker.state.model_usage
        mu = tracker.state.model_usage["glm-5-turbo"]
        assert mu["calls"] == 1
        assert mu["prompt_tokens"] == 1000

    def test_multiple_models(self, tracker):
        _record_success(tracker, model="glm-5-turbo", prompt_tokens=100)
        _record_success(tracker, model="glm-5-large", prompt_tokens=200)
        assert len(tracker.state.model_usage) == 2
        assert tracker.state.model_usage["glm-5-turbo"]["calls"] == 1
        assert tracker.state.model_usage["glm-5-large"]["calls"] == 1

    def test_models_in_summary(self, tracker):
        _record_success(tracker, model="model-a")
        _record_success(tracker, model="model-b")
        summary = tracker.get_summary()
        assert "model-a" in summary["models_used"]
        assert "model-b" in summary["models_used"]


# ── Per-mandate tracking ─────────────────────────────────────────────────


class TestMandateTracking:
    def test_single_mandate(self, tracker):
        _record_success(tracker, mandate_id="mandate-1", turn=1)
        _record_success(tracker, mandate_id="mandate-1", turn=2)
        mm = tracker.get_mandate_summary("mandate-1")
        assert mm is not None
        assert mm["calls"] == 2
        assert mm["turns"] == 2

    def test_multiple_mandates(self, tracker):
        _record_success(tracker, mandate_id="mandate-1", turn=1)
        _record_success(tracker, mandate_id="mandate-2", turn=1)
        assert tracker.state.mandate_usage["mandate-1"]["calls"] == 1
        assert tracker.state.mandate_usage["mandate-2"]["calls"] == 1
        assert tracker.get_summary()["mandates_tracked"] == 2

    def test_unknown_mandate_returns_none(self, tracker):
        assert tracker.get_mandate_summary("nonexistent") is None


# ── Persistence ──────────────────────────────────────────────────────────


class TestPersistence:
    def test_state_persists_across_reloads(self, tracker):
        _record_success(tracker, prompt_tokens=500, completion_tokens=250)
        assert tracker.state.total_tokens == 750

        # Reload from disk
        tracker2 = ApiBudgetTracker(state_path=str(tracker.state_path))
        assert tracker2.state.total_tokens == 750
        assert tracker2.state.total_calls == 1

    def test_state_file_created(self, tracker):
        _record_success(tracker)
        assert tracker.state_path.exists()
        data = json.loads(tracker.state_path.read_text())
        assert "total_calls" in data
        assert data["total_calls"] == 1

    def test_corrupt_state_starts_fresh(self, tmp_path):
        state_path = tmp_path / "bad.json"
        state_path.write_text("not valid json {{{")
        tracker = ApiBudgetTracker(state_path=str(state_path))
        assert tracker.state.total_calls == 0

    def test_missing_state_starts_fresh(self, tmp_path):
        state_path = tmp_path / "missing.json"
        tracker = ApiBudgetTracker(state_path=str(state_path))
        assert tracker.state.total_calls == 0
        assert tracker.state.started_at != ""


# ── Recent calls ─────────────────────────────────────────────────────────


class TestRecentCalls:
    def test_recent_calls_bounded(self, tracker):
        """Should keep only the last N calls."""
        tracker.max_history = 5
        for i in range(10):
            _record_success(tracker, mandate_id=f"m{i}")
        assert len(tracker.state.recent_calls) == 5
        # Last 5 mandates should be m5-m9
        last_mandates = [c["mandate_id"] for c in tracker.state.recent_calls]
        assert last_mandates == ["m5", "m6", "m7", "m8", "m9"]


# ── Summary ──────────────────────────────────────────────────────────────


class TestSummary:
    def test_empty_summary(self, tracker):
        summary = tracker.get_summary()
        assert summary["total_calls"] == 0
        assert summary["total_cost_usd"] == 0.0
        assert summary["error_rate"] == 0.0
        assert summary["avg_latency_seconds"] == 0.0

    def test_summary_with_data(self, tracker):
        _record_success(tracker, latency_seconds=2.0)
        _record_success(tracker, latency_seconds=4.0)
        summary = tracker.get_summary()
        assert summary["total_calls"] == 2
        assert summary["avg_latency_seconds"] == 3.0
        assert summary["avg_tokens_per_call"] == 1500
        assert summary["avg_cost_per_call"] > 0

    def test_uptime_in_summary(self, tracker):
        _record_success(tracker)
        summary = tracker.get_summary()
        assert summary["uptime"] != ""
        assert "h" in summary["uptime"]  # hours


# ── Reset ────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self, tracker):
        _record_success(tracker)
        _record_error(tracker)
        assert tracker.state.total_calls == 2

        tracker.reset()
        assert tracker.state.total_calls == 0
        assert tracker.state.total_tokens == 0
        assert tracker.state.total_cost_usd == 0.0
        assert tracker.state.total_errors == 0

    def test_reset_preserves_started_at(self, tracker):
        old_started = tracker.state.started_at
        _record_success(tracker)
        tracker.reset()
        # New started_at should be set
        assert tracker.state.started_at != ""


# ── ApiBudgetState serialization ────────────────────────────────────────


class TestApiBudgetState:
    def test_to_dict_roundtrip(self):
        state = ApiBudgetState(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_cost_usd=0.01,
            model_usage={"glm-5-turbo": {"calls": 1, "prompt_tokens": 1000,
                                           "completion_tokens": 500, "total_tokens": 1500, "cost_usd": 0.01}},
        )
        d = state.to_dict()
        restored = ApiBudgetState.from_dict(d)
        assert restored.total_prompt_tokens == 1000
        assert restored.total_completion_tokens == 500
        assert restored.total_cost_usd == 0.01
        assert "glm-5-turbo" in restored.model_usage

    def test_from_dict_ignores_unknown_fields(self):
        data = {
            "total_prompt_tokens": 100,
            "total_completion_tokens": 50,
            "total_tokens": 150,
            "total_cost_usd": 0.001,
            "total_calls": 0,
            "total_errors": 0,
            "started_at": "",
            "last_call_at": "",
            "max_cost_usd": 0.0,
            "max_calls": 0,
            "max_prompt_tokens": 0,
            "model_usage": {},
            "mandate_usage": {},
            "recent_calls": [],
            "future_field": "should be ignored",
        }
        state = ApiBudgetState.from_dict(data)
        assert state.total_prompt_tokens == 100
        assert not hasattr(state, "future_field")


# ── Global singleton ─────────────────────────────────────────────────────


class TestGlobalTracker:
    def test_get_global_tracker_returns_instance(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CRABQUANT_API_BUDGET_PATH", str(tmp_path / "global.json"))
        from crabquant.refinement.api_budget import get_global_tracker, _global_tracker
        # Reset global state
        import crabquant.refinement.api_budget as mod
        mod._global_tracker = None
        tracker = get_global_tracker()
        assert isinstance(tracker, ApiBudgetTracker)
        assert tracker.state_path == tmp_path / "global.json"

    def test_get_global_tracker_is_singleton(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CRABQUANT_API_BUDGET_PATH", str(tmp_path / "singleton.json"))
        import crabquant.refinement.api_budget as mod
        mod._global_tracker = None
        t1 = mod.get_global_tracker()
        t2 = mod.get_global_tracker()
        assert t1 is t2
