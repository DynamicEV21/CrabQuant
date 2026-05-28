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
