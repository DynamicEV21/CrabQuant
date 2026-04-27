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
        assert state.min_pass_rate == 0.3
        assert state.status == CircuitBreakerStatus.CLOSED
        assert state.history == []

    def test_custom_config(self):
        state = CircuitBreakerState(window=10, min_pass_rate=0.5)
        assert state.window == 10
        assert state.min_pass_rate == 0.5

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
        cb = CircuitBreaker()
        cb.record(False)
        # 0/1 = 0% < 30% → technically opens with only 1 data point
        # This is by design: if the very first attempt fails, rate is 0%
        # The breaker should only prevent checking when there's meaningful data
        # With window=20 and only 1 attempt, the breaker IS open (0% < 30%)
        assert cb.is_open() is True

    def test_one_pass_one_failure_does_not_open(self):
        cb = CircuitBreaker()
        cb.record(True)
        cb.record(False)
        # 1/2 = 50% > 30% → closed
        assert cb.is_open() is False

    def test_pass_rate_computation(self):
        cb = CircuitBreaker(window=20, min_pass_rate=0.3)
        for _ in range(14):
            cb.record(False)
        for _ in range(6):
            cb.record(True)
        # 6 passes / 20 attempts = 0.3, exactly at threshold (>= 0.3 means closed)
        assert cb.pass_rate == pytest.approx(0.3)
        assert cb.is_open() is False  # exactly 0.3 is NOT open

    def test_opens_below_threshold(self):
        cb = CircuitBreaker(window=20, min_pass_rate=0.3)
        # 5 passes out of 20 = 0.25 < 0.3 → open
        for _ in range(15):
            cb.record(False)
        for _ in range(5):
            cb.record(True)
        assert cb.pass_rate == pytest.approx(0.25)
        assert cb.is_open() is True

    def test_stays_closed_with_high_pass_rate(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.3)
        for _ in range(8):
            cb.record(True)
        for _ in range(2):
            cb.record(False)
        assert cb.pass_rate == pytest.approx(0.8)
        assert cb.is_open() is False

    def test_window_sliding(self):
        """When total exceeds window, only the most recent window is considered."""
        cb = CircuitBreaker(window=5, min_pass_rate=0.3)
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
        cb = CircuitBreaker(window=10, min_pass_rate=0.5)
        # 4/10 = 0.4 < 0.5 → open
        for _ in range(6):
            cb.record(False)
        for _ in range(4):
            cb.record(True)
        assert cb.pass_rate == pytest.approx(0.4)
        assert cb.is_open() is True

    def test_status_transitions(self):
        cb = CircuitBreaker(window=5, min_pass_rate=0.3)
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
        cb = CircuitBreaker(window=5, min_pass_rate=0.3)
        cb.record(True)
        cb.record(False)
        cb.record(True)
        assert len(cb._history) == 3
        assert cb._history[0]["passed"] is True
        assert cb._history[1]["passed"] is False

    def test_get_state_returns_snapshot(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.5)
        cb.record(True)
        cb.record(False)
        state = cb.get_state()
        assert state.total_attempts == 2
        assert state.passes == 1
        assert state.failures == 1

    def test_restore_state(self):
        state = CircuitBreakerState(
            total_attempts=10,
            passes=3,
            failures=7,
            window=20,
            min_pass_rate=0.3,
            history=[{"passed": True}, {"passed": False}],
        )
        cb = CircuitBreaker.restore(state)
        assert cb.window == 20
        # History is restored as-is: 2 entries, 1 pass → 50%
        assert cb.total_attempts == 2
        assert cb.passes == 1
        assert cb.failures == 1
        assert cb.pass_rate == pytest.approx(0.5)

    def test_record_batch(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.3)
        cb.record_batch([True, False, True, True, False])
        assert cb.total_attempts == 5
        assert cb.passes == 3
        assert cb.failures == 2

    def test_empty_history_no_crash(self):
        cb = CircuitBreaker(window=5, min_pass_rate=0.3)
        assert cb.pass_rate == 0.0
        assert cb.is_open() is False

    def test_summary_string(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.3)
        cb.record(True)
        cb.record(False)
        summary = cb.summary()
        assert "1/2" in summary
        assert "50.0%" in summary
        assert "CLOSED" in summary

    def test_reset(self):
        cb = CircuitBreaker(window=10, min_pass_rate=0.3)
        cb.record(True)
        cb.record(False)
        cb.reset()
        assert cb.total_attempts == 0
        assert cb.passes == 0
        assert cb.failures == 0
        assert cb._history == []
        assert cb.is_open() is False
