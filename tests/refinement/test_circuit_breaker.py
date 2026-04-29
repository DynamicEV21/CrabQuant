"""Tests for crabquant.refinement.circuit_breaker — LLM validation pass rate tracking."""

import pytest
from crabquant.refinement.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerStatus,
)


class TestCircuitBreakerState:
    """Test CircuitBreakerState dataclass."""

    def test_defaults(self):
        state = CircuitBreakerState()
        assert state.total_attempts == 0
        assert state.passes == 0
        assert state.failures == 0
        assert state.window == 20
        assert state.min_pass_rate == 0.2
        assert state.min_attempts == 5
        assert state.grace_turns == 2
        assert state.status == CircuitBreakerStatus.CLOSED
        assert state.history == []

    def test_custom_config(self):
        state = CircuitBreakerState(window=10, min_pass_rate=0.5, min_attempts=3, grace_turns=1)
        assert state.window == 10
        assert state.min_pass_rate == 0.5
        assert state.min_attempts == 3
        assert state.grace_turns == 1

    def test_to_dict_roundtrip(self):
        state = CircuitBreakerState(window=15, min_pass_rate=0.4)
        state.total_attempts = 5
        state.passes = 3
        state.failures = 2
        d = state.to_dict()
        assert d["total_attempts"] == 5
        assert d["passes"] == 3
        assert d["window"] == 15

        restored = CircuitBreakerState.from_dict(d)
        assert restored.total_attempts == 5
        assert restored.passes == 3
        assert restored.window == 15
        assert restored.min_pass_rate == 0.4

    def test_to_json_roundtrip(self):
        state = CircuitBreakerState()
        json_str = state.to_json()
        restored = CircuitBreakerState.from_json(json_str)
        assert restored.total_attempts == 0
        assert restored.window == 20
        assert restored.min_attempts == 5
        assert restored.grace_turns == 2

    def test_to_json_with_indent(self):
        """to_json should support kwargs like indent."""
        state = CircuitBreakerState(window=10)
        json_str = state.to_json(indent=2)
        assert "\n" in json_str  # Pretty-printed

    def test_from_dict_ignores_unknown_fields(self):
        """from_dict should silently drop unknown keys."""
        data = {
            "total_attempts": 5,
            "passes": 3,
            "failures": 2,
            "window": 20,
            "min_pass_rate": 0.2,
            "min_attempts": 5,
            "grace_turns": 2,
            "status": "CLOSED",
            "history": [],
            "unknown_future_field": "ignored",
        }
        state = CircuitBreakerState.from_dict(data)
        assert state.total_attempts == 5
        assert not hasattr(state, "unknown_future_field")

    def test_from_dict_with_partial_data(self):
        """from_dict should handle missing fields using defaults."""
        state = CircuitBreakerState.from_dict({"window": 50})
        assert state.window == 50
        assert state.min_pass_rate == 0.2  # default
        assert state.history == []

    def test_status_enum_values(self):
        """CircuitBreakerStatus should have CLOSED and OPEN values."""
        assert CircuitBreakerStatus.CLOSED == "CLOSED"
        assert CircuitBreakerStatus.OPEN == "OPEN"
        assert len(CircuitBreakerStatus) == 2


class TestCircuitBreaker:
    """Test CircuitBreaker pass/fail tracking and threshold logic."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.is_open() is False
        assert cb.status == CircuitBreakerStatus.CLOSED

    def test_single_pass_does_not_open(self):
        cb = CircuitBreaker()
        cb.record(True)
        assert cb.is_open() is False

    def test_single_failure_does_not_open(self):
        cb = CircuitBreaker(min_attempts=5)
        cb.record(False)
        # Below min_attempts → never opens
        assert cb.is_open() is False

    def test_one_pass_one_failure_does_not_open(self):
        cb = CircuitBreaker()
        cb.record(True)
        cb.record(False)
        # 1/2 = 50% > 30% → closed
        assert cb.is_open() is False

    def test_pass_rate_computation(self):
        cb = CircuitBreaker(window=20, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        for _ in range(14):
            cb.record(False)
        for _ in range(6):
            cb.record(True)
        # 6 passes / 20 attempts = 0.3, above threshold (0.2)
        assert cb.pass_rate == pytest.approx(0.3)
        assert cb.is_open() is False  # 0.3 > 0.2 → closed

    def test_opens_below_threshold(self):
        cb = CircuitBreaker(window=20, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        # 3 passes out of 20 = 0.15 < 0.2 → open
        for _ in range(17):
            cb.record(False)
        for _ in range(3):
            cb.record(True)
        assert cb.pass_rate == pytest.approx(0.15)
        assert cb.is_open() is True

    def test_stays_closed_with_high_pass_rate(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        for _ in range(8):
            cb.record(True)
        for _ in range(2):
            cb.record(False)
        assert cb.pass_rate == pytest.approx(0.8)
        assert cb.is_open() is False

    def test_window_sliding(self):
        """When total exceeds window, only the most recent window is considered."""
        cb = CircuitBreaker(window=5, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        # Fill window with all failures (0/5 = 0%)
        for _ in range(5):
            cb.record(False)
        assert cb.is_open() is True

        # Add passes that push out the old failures
        # After 5 more passes: window = [True x5] → 100%
        for _ in range(5):
            cb.record(True)
        assert cb.pass_rate == pytest.approx(1.0)
        assert cb.is_open() is False

    def test_respects_custom_min_pass_rate(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.5, min_attempts=5, grace_turns=0)
        # 4/10 = 0.4 < 0.5 → open
        for _ in range(6):
            cb.record(False)
        for _ in range(4):
            cb.record(True)
        assert cb.pass_rate == pytest.approx(0.4)
        assert cb.is_open() is True

    def test_status_transitions(self):
        cb = CircuitBreaker(window=5, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        assert cb.status == CircuitBreakerStatus.CLOSED

        # Trigger open
        for _ in range(5):
            cb.record(False)
        assert cb.status == CircuitBreakerStatus.OPEN

        # Recover
        for _ in range(5):
            cb.record(True)
        assert cb.status == CircuitBreakerStatus.CLOSED

    def test_history_tracks_all_events(self):
        cb = CircuitBreaker(window=5, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        cb.record(True)
        cb.record(False)
        cb.record(True)
        assert len(cb._history) == 3
        assert cb._history[0]["passed"] is True
        assert cb._history[1]["passed"] is False

    def test_get_state_returns_snapshot(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.5, min_attempts=5, grace_turns=2)
        cb.record(True)
        cb.record(False)
        state = cb.get_state()
        assert state.total_attempts == 2
        assert state.passes == 1
        assert state.failures == 1
        assert state.min_attempts == 5
        assert state.grace_turns == 2

    def test_restore_state(self):
        state = CircuitBreakerState(
            total_attempts=10,
            passes=3,
            failures=7,
            window=20,
            min_pass_rate=0.2,
            min_attempts=5,
            grace_turns=2,
            history=[{"passed": True}, {"passed": False}],
        )
        cb = CircuitBreaker.restore(state)
        assert cb.window == 20
        assert cb.min_attempts == 5
        assert cb.grace_turns == 2
        # History is restored as-is: 2 entries, 1 pass → 50%
        assert cb.total_attempts == 2
        assert cb.passes == 1
        assert cb.failures == 1
        assert cb.pass_rate == pytest.approx(0.5)

    def test_record_batch(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        cb.record_batch([True, False, True, True, False])
        assert cb.total_attempts == 5
        assert cb.passes == 3
        assert cb.failures == 2

    def test_empty_history_no_crash(self):
        cb = CircuitBreaker(window=5, min_pass_rate=0.2)
        assert cb.pass_rate == 0.0
        assert cb.is_open() is False

    def test_summary_string(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        cb.record(True)
        cb.record(False)
        summary = cb.summary()
        assert "1/2" in summary
        assert "50.0%" in summary
        assert "CLOSED" in summary

    def test_reset(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        cb.record(True)
        cb.record(False)
        cb.reset()
        assert cb.total_attempts == 0
        assert cb.passes == 0
        assert cb.failures == 0
        assert cb._history == []
        assert cb.is_open() is False

    # ── New tests for grace period and min_attempts ──────────────────────

    def test_grace_period_prevents_firing(self):
        """During grace turns, the breaker never fires regardless of pass rate."""
        cb = CircuitBreaker(window=5, min_pass_rate=0.5, min_attempts=1, grace_turns=2)
        # All failures on turn 1 (within grace)
        for _ in range(5):
            cb.record(False, turn=1)
        assert cb.is_open() is False

        # All failures on turn 2 (still within grace)
        for _ in range(5):
            cb.record(False, turn=2)
        assert cb.is_open() is False

        # First failure on turn 3 (past grace) → should open
        for _ in range(5):
            cb.record(False, turn=3)
        assert cb.is_open() is True

    def test_min_attempts_prevents_early_firing(self):
        """Below min_attempts, the breaker never fires."""
        cb = CircuitBreaker(window=20, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        # 4 failures, all on turn 5 (past grace) — but only 4 attempts
        for _ in range(4):
            cb.record(False, turn=5)
        assert cb.is_open() is False

        # 5th failure → min_attempts reached, rate is 0% < 20%
        cb.record(False, turn=5)
        assert cb.is_open() is True

    def test_no_turn_info_no_grace(self):
        """When no turn info is recorded, grace period doesn't apply."""
        cb = CircuitBreaker(window=5, min_pass_rate=0.5, min_attempts=1, grace_turns=2)
        # Record failures without turn info — grace can't apply
        for _ in range(5):
            cb.record(False)
        # No turns seen, so grace check is skipped (turns_seen is empty)
        # min_attempts=1 is met, rate=0% < 50% → opens
        assert cb.is_open() is True

    def test_typical_abandonment_scenario_is_prevented(self):
        """Simulate the exact scenario that killed mandates: 3 failures on turn 1.

        Previously: 0/3 = 0% < 30% → breaker opens on turn 2.
        Now: min_attempts=5 not reached AND grace_turns=2 not exceeded.
        """
        cb = CircuitBreaker(window=20, min_pass_rate=0.2, min_attempts=5, grace_turns=2)
        # Turn 1: 3 failed validation attempts (the typical scenario)
        cb.record(False, turn=1)
        cb.record(False, turn=1)
        cb.record(False, turn=1)
        # Turn 2: 3 more failed attempts
        cb.record(False, turn=2)
        cb.record(False, turn=2)
        cb.record(False, turn=2)
        # 6 total attempts, 0% pass rate — but still in grace period (turn 2 ≤ 2)
        assert cb.is_open() is False

        # Turn 3: still failing
        cb.record(False, turn=3)
        cb.record(False, turn=3)
        cb.record(False, turn=3)
        # Now past grace, min_attempts met, 0% < 20% → opens
        assert cb.is_open() is True

    # ── Additional coverage tests ────────────────────────────────────────

    def test_record_with_turn_and_mandate(self):
        """record() should store turn and mandate metadata."""
        cb = CircuitBreaker()
        cb.record(True, turn=5, mandate="alpha")
        cb.record(False, turn=5, mandate="beta")
        assert cb._history[0]["turn"] == 5
        assert cb._history[0]["mandate"] == "alpha"
        assert cb._history[1]["turn"] == 5
        assert cb._history[1]["mandate"] == "beta"

    def test_record_with_none_turn(self):
        """turn=None should be stored as None."""
        cb = CircuitBreaker()
        cb.record(True, turn=None)
        assert cb._history[0]["turn"] is None

    def test_record_with_none_mandate(self):
        """mandate=None should be stored as None."""
        cb = CircuitBreaker()
        cb.record(True, mandate=None)
        assert cb._history[0]["mandate"] is None

    def test_exact_boundary_pass_rate(self):
        """Pass rate exactly at threshold should stay closed (not < threshold)."""
        cb = CircuitBreaker(window=10, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        # 2 passes out of 10 = 0.2 = exactly at threshold → NOT open (need < threshold)
        for _ in range(8):
            cb.record(False)
        for _ in range(2):
            cb.record(True)
        assert cb.pass_rate == pytest.approx(0.2)
        assert cb.is_open() is False  # 0.2 is NOT < 0.2

    def test_window_of_one(self):
        """Window=1 should only consider the very last result."""
        cb = CircuitBreaker(window=1, min_pass_rate=0.5, min_attempts=1, grace_turns=0)
        # First failure → open (0/1 = 0% < 50%)
        cb.record(False, turn=5)
        assert cb.is_open() is True
        # Single pass → closed (1/1 = 100% >= 50%)
        cb.record(True, turn=5)
        assert cb.is_open() is False
        # Single failure → open again
        cb.record(False, turn=5)
        assert cb.is_open() is True

    def test_min_attempts_of_one(self):
        """min_attempts=1 should allow opening after a single attempt."""
        cb = CircuitBreaker(window=5, min_pass_rate=0.5, min_attempts=1, grace_turns=0)
        cb.record(False, turn=5)
        # 1 attempt (meets min), 0% < 50% → opens
        assert cb.is_open() is True

    def test_grace_turns_zero(self):
        """grace_turns=0 means grace period is effectively off."""
        cb = CircuitBreaker(window=5, min_pass_rate=0.5, min_attempts=1, grace_turns=0)
        cb.record(False, turn=1)
        # turn 1 > 0 (grace_turns), so grace doesn't apply
        assert cb.is_open() is True

    def test_recovery_after_open(self):
        """Circuit breaker should recover when pass rate improves."""
        cb = CircuitBreaker(window=10, min_pass_rate=0.2, min_attempts=5, grace_turns=0)
        # Drive to open
        for _ in range(10):
            cb.record(False, turn=5)
        assert cb.is_open() is True

        # Recover with passes
        for _ in range(10):
            cb.record(True, turn=5)
        assert cb.pass_rate == pytest.approx(1.0)
        assert cb.is_open() is False

    def test_summary_with_open_status(self):
        """Summary string should show OPEN when breaker is open."""
        cb = CircuitBreaker(window=5, min_pass_rate=0.5, min_attempts=1, grace_turns=0)
        for _ in range(5):
            cb.record(False, turn=5)
        summary = cb.summary()
        assert "OPEN" in summary
        assert "0/5" in summary

    def test_summary_with_empty_history(self):
        """Summary should handle empty history gracefully."""
        cb = CircuitBreaker()
        summary = cb.summary()
        assert "CLOSED" in summary
        assert "0/0" in summary

    def test_pass_rate_with_window_larger_than_history(self):
        """When history < window, pass rate uses all available history."""
        cb = CircuitBreaker(window=100, min_pass_rate=0.2, min_attempts=3, grace_turns=0)
        cb.record(True, turn=5)
        cb.record(True, turn=5)
        cb.record(False, turn=5)
        # 2/3 ≈ 0.667
        assert cb.pass_rate == pytest.approx(2 / 3)

    def test_total_attempts_property(self):
        """total_attempts should equal len(history)."""
        cb = CircuitBreaker()
        assert cb.total_attempts == 0
        cb.record(True)
        cb.record(False)
        assert cb.total_attempts == 2

    def test_passes_and_failures_properties(self):
        """passes + failures should equal total_attempts."""
        cb = CircuitBreaker()
        cb.record_batch([True, True, False, False, False, True])
        assert cb.passes == 3
        assert cb.failures == 3
        assert cb.passes + cb.failures == cb.total_attempts

    def test_record_batch_empty(self):
        """record_batch with empty list should be a no-op."""
        cb = CircuitBreaker()
        cb.record_batch([])
        assert cb.total_attempts == 0

    def test_multiple_record_batches(self):
        """Multiple record_batch calls should accumulate."""
        cb = CircuitBreaker()
        cb.record_batch([True, False])
        cb.record_batch([True, True, False])
        assert cb.total_attempts == 5
        assert cb.passes == 3
        assert cb.failures == 2

    def test_reset_after_open(self):
        """Reset should close the breaker even if it was open."""
        cb = CircuitBreaker(window=5, min_pass_rate=0.5, min_attempts=1, grace_turns=0)
        for _ in range(5):
            cb.record(False, turn=5)
        assert cb.is_open() is True

        cb.reset()
        assert cb.is_open() is False
        assert cb.status == CircuitBreakerStatus.CLOSED

    def test_restore_preserves_history_metadata(self):
        """Restored breaker should preserve turn/mandate in history."""
        state = CircuitBreakerState(
            total_attempts=2,
            passes=1,
            failures=1,
            window=20,
            min_pass_rate=0.2,
            min_attempts=5,
            grace_turns=2,
            history=[
                {"passed": True, "turn": 3, "mandate": "alpha"},
                {"passed": False, "turn": 4, "mandate": "beta"},
            ],
        )
        cb = CircuitBreaker.restore(state)
        assert cb._history[0]["turn"] == 3
        assert cb._history[0]["mandate"] == "alpha"
        assert cb._history[1]["turn"] == 4
        assert cb._history[1]["mandate"] == "beta"

    def test_status_vs_is_open_consistency(self):
        """status property and is_open() should always agree."""
        cb = CircuitBreaker(window=5, min_pass_rate=0.5, min_attempts=1, grace_turns=0)
        # Empty
        assert cb.is_open() == (cb.status == CircuitBreakerStatus.OPEN)

        # Some passes
        cb.record(True, turn=5)
        assert cb.is_open() == (cb.status == CircuitBreakerStatus.OPEN)

        # All failures → open
        for _ in range(5):
            cb.record(False, turn=5)
        assert cb.is_open() == (cb.status == CircuitBreakerStatus.OPEN)
        assert cb.is_open() is True

        # Recover
        for _ in range(5):
            cb.record(True, turn=5)
        assert cb.is_open() == (cb.status == CircuitBreakerStatus.OPEN)
        assert cb.is_open() is False

    def test_get_state_includes_history(self):
        """get_state() should include the full history."""
        cb = CircuitBreaker()
        cb.record(True, turn=1, mandate="m1")
        cb.record(False, turn=2, mandate="m2")
        state = cb.get_state()
        assert len(state.history) == 2
        assert state.history[0]["mandate"] == "m1"
        assert state.history[1]["mandate"] == "m2"

    def test_all_failures_open_immediately_when_min_attempts_met(self):
        """100% failure rate should open when min_attempts is met."""
        cb = CircuitBreaker(window=20, min_pass_rate=0.01, min_attempts=3, grace_turns=0)
        # Even with a very low threshold, 0% should still open
        for _ in range(3):
            cb.record(False, turn=5)
        assert cb.pass_rate == pytest.approx(0.0)
        assert cb.is_open() is True

    def test_mixed_turns_grace_uses_max_turn(self):
        """Grace period should use the maximum turn seen."""
        cb = CircuitBreaker(window=10, min_pass_rate=0.5, min_attempts=1, grace_turns=3)
        # Record on turn 1, 2, 3 — all within grace
        for t in [1, 2, 3]:
            cb.record(False, turn=t)
        assert cb.is_open() is False  # max turn = 3 ≤ grace_turns 3

        # Turn 4 exceeds grace
        cb.record(False, turn=4)
        assert cb.is_open() is True  # max turn = 4 > grace_turns 3
