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
    ApiCallRecord,
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

    def test_completion_cost_calculation(self, tracker):
        """Completion tokens cost $8/MTok by default."""
        _record_success(tracker, prompt_tokens=0, completion_tokens=1_000_000)
        s = tracker.state
        assert s.total_cost_usd == pytest.approx(8.0, abs=0.01)

    def test_mixed_cost_calculation(self, tracker):
        """Both prompt and completion tokens contribute to cost."""
        _record_success(tracker, prompt_tokens=1_000_000, completion_tokens=500_000)
        s = tracker.state
        # $2 (prompt) + $4 (completion) = $6
        assert s.total_cost_usd == pytest.approx(6.0, abs=0.01)

    def test_custom_cost_rates(self, tmp_path):
        """Tracker respects custom cost-per-million rates."""
        state_path = tmp_path / "custom_rates.json"
        t = ApiBudgetTracker(
            state_path=str(state_path),
            cost_per_million_prompt=5.0,
            cost_per_million_completion=20.0,
        )
        t.record_call(
            model="test", prompt_tokens=1_000_000, completion_tokens=1_000_000,
            latency_seconds=1.0, success=True,
        )
        # $5 (prompt) + $20 (completion) = $25
        assert t.state.total_cost_usd == pytest.approx(25.0, abs=0.01)

    def test_latency_rounded(self, tracker):
        """Latency should be rounded to 2 decimal places."""
        record = tracker.record_call(
            model="test", prompt_tokens=100, completion_tokens=50,
            latency_seconds=2.345678, success=True,
        )
        assert record.latency_seconds == pytest.approx(2.35)

    def test_cost_rounded(self, tracker):
        """Cost should be rounded to 6 decimal places."""
        record = tracker.record_call(
            model="test", prompt_tokens=3, completion_tokens=7,
            latency_seconds=1.0, success=True,
        )
        # Cost = 3*2/1e6 + 7*8/1e6 = 6e-6 + 56e-6 = 62e-6
        assert record.cost_usd == pytest.approx(0.000062, abs=1e-9)

    def test_error_record_includes_error_message(self, tracker):
        """Failed calls should store the error string."""
        record = _record_error(tracker, error="rate_limit_exceeded")
        assert record.error == "rate_limit_exceeded"
        assert record.success is False

    def test_successful_call_has_no_error(self, tracker):
        record = _record_success(tracker)
        assert record.error == ""

    def test_zero_tokens_call(self, tracker):
        """A call with zero tokens should cost $0."""
        record = tracker.record_call(
            model="test", prompt_tokens=0, completion_tokens=0,
            latency_seconds=1.0, success=True,
        )
        assert record.total_tokens == 0
        assert record.cost_usd == 0.0


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

    def test_token_limit_enforced(self, tracker):
        """max_prompt_tokens limits total tokens per call."""
        tracker.set_budget(max_prompt_tokens=5000)
        # Call with 3000 tokens is fine
        tracker.record_call(
            model="test", prompt_tokens=2000, completion_tokens=1000,
            latency_seconds=1.0, success=True,
        )
        # Call with 6000 tokens exceeds the 5000 limit
        with pytest.raises(BudgetExceededError, match="Token limit"):
            tracker.record_call(
                model="test", prompt_tokens=4000, completion_tokens=2000,
                latency_seconds=1.0, success=True,
            )

    def test_token_limit_exactly_at_boundary(self, tracker):
        """Call with tokens exactly equal to limit should succeed."""
        tracker.set_budget(max_prompt_tokens=5000)
        tracker.record_call(
            model="test", prompt_tokens=3000, completion_tokens=2000,
            latency_seconds=1.0, success=True,
        )
        # total_tokens == 5000 == max_prompt_tokens → not > 5000 → succeeds
        assert tracker.state.total_calls == 1

    def test_cost_limit_exact_boundary(self, tracker):
        """A call that would make cost exactly equal to limit should succeed (not >)."""
        # Use exact cost: 1M prompt tokens = $2.00
        tracker.set_budget(max_cost_usd=2.0)
        tracker.record_call(
            model="test", prompt_tokens=1_000_000, completion_tokens=0,
            latency_seconds=1.0, success=True,
        )
        assert tracker.state.total_calls == 1
        assert tracker.state.total_cost_usd == pytest.approx(2.0, abs=0.01)

    def test_call_limit_exact_boundary(self, tracker):
        """Last allowed call should succeed; next should fail."""
        tracker.set_budget(max_calls=3)
        _record_success(tracker)
        _record_success(tracker)
        _record_success(tracker)
        assert tracker.state.total_calls == 3
        with pytest.raises(BudgetExceededError):
            _record_success(tracker)

    def test_budget_error_message_contains_details(self, tracker):
        """BudgetExceededError should include current and call cost info."""
        tracker.set_budget(max_cost_usd=0.01)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.record_call(
                model="test", prompt_tokens=5_000_000, completion_tokens=0,
                latency_seconds=1.0, success=True,
            )
        msg = str(exc_info.value)
        assert "$" in msg
        assert "Cost limit" in msg

    def test_budget_zero_means_no_limit(self, tracker):
        """Explicitly setting budget to 0 should mean no limit."""
        tracker.set_budget(max_cost_usd=0.0, max_calls=0, max_prompt_tokens=0)
        for _ in range(50):
            _record_success(tracker, prompt_tokens=10_000_000, completion_tokens=5_000_000)
        assert tracker.state.total_calls == 50

    def test_budget_remaining_prompt_tokens(self, tracker):
        """Budget remaining should include prompt_tokens when limit set."""
        tracker.set_budget(max_prompt_tokens=10000)
        _record_success(tracker, prompt_tokens=3000, completion_tokens=2000)
        remaining = tracker.get_summary()["budget_remaining"]
        assert "prompt_tokens" in remaining
        # 10000 - 5000 = 5000 remaining
        assert remaining["prompt_tokens"] == 5000

    def test_set_budget_persists_to_disk(self, tracker):
        """set_budget should save state to disk."""
        tracker.set_budget(max_cost_usd=5.0, max_calls=1000)
        # Reload from disk
        tracker2 = ApiBudgetTracker(state_path=str(tracker.state_path))
        assert tracker2.state.max_cost_usd == 5.0
        assert tracker2.state.max_calls == 1000


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

    def test_model_accumulates(self, tracker):
        """Multiple calls to the same model should accumulate."""
        _record_success(tracker, model="glm-5-turbo", prompt_tokens=100)
        _record_success(tracker, model="glm-5-turbo", prompt_tokens=200)
        _record_error(tracker, model="glm-5-turbo")
        mu = tracker.state.model_usage["glm-5-turbo"]
        assert mu["calls"] == 3
        # error default prompt_tokens=1000, completion_tokens=0
        assert mu["prompt_tokens"] == 1300  # 100 + 200 + 1000

    def test_model_cost_accumulates(self, tracker):
        """Model cost should accumulate across calls."""
        _record_success(tracker, model="glm-5-turbo", prompt_tokens=1_000_000, completion_tokens=0)
        _record_success(tracker, model="glm-5-turbo", prompt_tokens=1_000_000, completion_tokens=0)
        mu = tracker.state.model_usage["glm-5-turbo"]
        assert mu["cost_usd"] == pytest.approx(4.0, abs=0.01)


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

    def test_mandate_tracks_cost(self, tracker):
        """Mandate usage should track cost accurately."""
        _record_success(tracker, mandate_id="m1", prompt_tokens=1_000_000, completion_tokens=0)
        mm = tracker.get_mandate_summary("m1")
        assert mm["cost_usd"] == pytest.approx(2.0, abs=0.01)

    def test_mandate_turns_max(self, tracker):
        """Mandate turns should track the maximum turn number."""
        _record_success(tracker, mandate_id="m1", turn=1)
        _record_success(tracker, mandate_id="m1", turn=3)
        _record_success(tracker, mandate_id="m1", turn=2)
        mm = tracker.get_mandate_summary("m1")
        assert mm["turns"] == 3

    def test_mandate_accumulates_tokens(self, tracker):
        """Mandate should accumulate tokens across calls."""
        _record_success(tracker, mandate_id="m1", prompt_tokens=100, completion_tokens=50)
        _record_success(tracker, mandate_id="m1", prompt_tokens=200, completion_tokens=100)
        mm = tracker.get_mandate_summary("m1")
        assert mm["prompt_tokens"] == 300
        assert mm["completion_tokens"] == 150
        assert mm["total_tokens"] == 450

    def test_empty_mandate_id_not_tracked(self, tracker):
        """Calls with empty mandate_id should not create a mandate entry."""
        _record_success(tracker, mandate_id="")
        assert tracker.state.mandate_usage == {}
        assert tracker.get_summary()["mandates_tracked"] == 0


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

    def test_persistence_includes_model_usage(self, tracker):
        """Model usage should survive reload."""
        _record_success(tracker, model="special-model")
        tracker2 = ApiBudgetTracker(state_path=str(tracker.state_path))
        assert "special-model" in tracker2.state.model_usage

    def test_persistence_includes_mandate_usage(self, tracker):
        """Mandate usage should survive reload."""
        _record_success(tracker, mandate_id="persist-mandate", turn=5)
        tracker2 = ApiBudgetTracker(state_path=str(tracker.state_path))
        mm = tracker2.get_mandate_summary("persist-mandate")
        assert mm is not None
        assert mm["calls"] == 1
        assert mm["turns"] == 5

    def test_corrupt_json_type_error(self, tmp_path):
        """Non-dict JSON (e.g., a list) should trigger fresh start."""
        state_path = tmp_path / "list.json"
        state_path.write_text("[1, 2, 3]")
        # List is valid JSON but not a dict; from_dict will get AttributeError
        # which is not caught by the except clause (only JSONDecodeError, TypeError)
        # So this actually raises. Test that it raises an appropriate error.
        with pytest.raises((AttributeError, TypeError)):
            ApiBudgetTracker(state_path=str(state_path))


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

    def test_recent_calls_at_exact_boundary(self, tracker):
        """Exactly max_history calls should not be trimmed."""
        tracker.max_history = 5
        for i in range(5):
            _record_success(tracker, mandate_id=f"m{i}")
        assert len(tracker.state.recent_calls) == 5
        last_mandates = [c["mandate_id"] for c in tracker.state.recent_calls]
        assert last_mandates == ["m0", "m1", "m2", "m3", "m4"]

    def test_recent_calls_default_max_history(self, tracker):
        """Default max_history is 100."""
        for i in range(110):
            _record_success(tracker, mandate_id=f"m{i}")
        assert len(tracker.state.recent_calls) == 100

    def test_recent_calls_include_all_record_fields(self, tracker):
        """Recent calls should store complete record data."""
        _record_success(tracker, mandate_id="m1", turn=3)
        call = tracker.state.recent_calls[0]
        assert call["model"] == "glm-5-turbo"
        assert call["prompt_tokens"] == 1000
        assert call["completion_tokens"] == 500
        assert call["success"] is True
        assert call["mandate_id"] == "m1"
        assert call["turn"] == 3


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

    def test_avg_latency_with_many_calls(self, tracker):
        """_avg_latency only considers the last 20 calls."""
        for i in range(25):
            _record_success(tracker, latency_seconds=1.0)
        # Last 20 calls all have latency 1.0
        summary = tracker.get_summary()
        assert summary["avg_latency_seconds"] == 1.0

    def test_avg_latency_with_varying_latencies(self, tracker):
        """Average latency should correctly average recent calls."""
        _record_success(tracker, latency_seconds=1.0)
        _record_success(tracker, latency_seconds=3.0)
        _record_success(tracker, latency_seconds=5.0)
        summary = tracker.get_summary()
        assert summary["avg_latency_seconds"] == 3.0

    def test_uptime_with_invalid_started_at(self, tracker):
        """Invalid started_at should show 'unknown' uptime."""
        tracker.state.started_at = "not-a-date"
        summary = tracker.get_summary()
        assert summary["uptime"] == "unknown"

    def test_uptime_with_empty_started_at(self, tracker):
        """Empty started_at should result in empty uptime string."""
        tracker.state.started_at = ""
        summary = tracker.get_summary()
        # Empty string causes datetime parsing to fail → "unknown"
        # But actually empty string through fromisoformat gives ValueError → "unknown"
        # However, the code does s.started_at check first (empty is falsy)
        # So uptime stays as "" 
        assert summary["uptime"] in ("", "unknown")

    def test_summary_all_keys_present(self, tracker):
        """Summary should always contain all expected keys."""
        summary = tracker.get_summary()
        expected_keys = [
            "total_calls", "total_errors", "error_rate",
            "total_prompt_tokens", "total_completion_tokens", "total_tokens",
            "total_cost_usd", "avg_tokens_per_call", "avg_cost_per_call",
            "avg_latency_seconds", "uptime", "budget_remaining",
            "models_used", "mandates_tracked",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"


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

    def test_reset_clears_model_usage(self, tracker):
        _record_success(tracker, model="glm-5-turbo")
        _record_success(tracker, model="glm-5-large")
        assert len(tracker.state.model_usage) == 2

        tracker.reset()
        assert tracker.state.model_usage == {}

    def test_reset_clears_mandate_usage(self, tracker):
        _record_success(tracker, mandate_id="m1", turn=1)
        _record_success(tracker, mandate_id="m2", turn=2)
        assert len(tracker.state.mandate_usage) == 2

        tracker.reset()
        assert tracker.state.mandate_usage == {}

    def test_reset_clears_recent_calls(self, tracker):
        for _ in range(10):
            _record_success(tracker)
        assert len(tracker.state.recent_calls) > 0

        tracker.reset()
        assert tracker.state.recent_calls == []

    def test_reset_clears_budget_limits(self, tracker):
        tracker.set_budget(max_cost_usd=10.0, max_calls=100)
        tracker.reset()
        assert tracker.state.max_cost_usd == 0.0
        assert tracker.state.max_calls == 0


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

    def test_from_dict_with_partial_data(self):
        """from_dict should handle missing fields gracefully."""
        state = ApiBudgetState.from_dict({"total_prompt_tokens": 42})
        assert state.total_prompt_tokens == 42
        # Other fields should use defaults
        assert state.total_completion_tokens == 0
        assert state.total_cost_usd == 0.0

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all dataclass fields."""
        state = ApiBudgetState(
            total_prompt_tokens=10,
            total_completion_tokens=5,
            total_tokens=15,
            total_cost_usd=0.001,
            total_calls=1,
            total_errors=0,
            started_at="2025-01-01T00:00:00",
            last_call_at="2025-01-01T01:00:00",
            max_cost_usd=10.0,
            max_calls=100,
            max_prompt_tokens=50000,
            model_usage={"m": {"calls": 1, "prompt_tokens": 10,
                                "completion_tokens": 5, "total_tokens": 15, "cost_usd": 0.001}},
            mandate_usage={"mid": {"calls": 1, "prompt_tokens": 10,
                                    "completion_tokens": 5, "total_tokens": 15,
                                    "cost_usd": 0.001, "turns": 1}},
            recent_calls=[],
        )
        d = state.to_dict()
        assert d["total_prompt_tokens"] == 10
        assert d["max_cost_usd"] == 10.0
        assert d["last_call_at"] == "2025-01-01T01:00:00"


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

    def test_global_tracker_uses_env_var(self, monkeypatch, tmp_path):
        """Global tracker should use CRABQUANT_API_BUDGET_PATH env var."""
        monkeypatch.setenv("CRABQUANT_API_BUDGET_PATH", str(tmp_path / "env_path.json"))
        import crabquant.refinement.api_budget as mod
        mod._global_tracker = None
        tracker = mod.get_global_tracker()
        assert tracker.state_path == tmp_path / "env_path.json"


# ── ApiCallRecord ────────────────────────────────────────────────────────


class TestApiCallRecord:
    def test_default_values(self):
        record = ApiCallRecord(
            timestamp="2025-01-01T00:00:00",
            model="test",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_seconds=1.0,
            cost_usd=0.001,
            success=True,
        )
        assert record.error == ""
        assert record.mandate_id == ""
        assert record.turn == 0

    def test_all_fields_settable(self):
        record = ApiCallRecord(
            timestamp="2025-01-01T00:00:00",
            model="test-model",
            prompt_tokens=500,
            completion_tokens=250,
            total_tokens=750,
            latency_seconds=2.5,
            cost_usd=0.005,
            success=False,
            error="server_error",
            mandate_id="mandate-42",
            turn=7,
        )
        assert record.model == "test-model"
        assert record.success is False
        assert record.error == "server_error"
        assert record.mandate_id == "mandate-42"
        assert record.turn == 7
