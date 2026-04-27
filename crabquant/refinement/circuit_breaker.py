"""
Circuit Breaker — Track LLM validation pass rate and halt if below threshold.

When the LLM consistently produces invalid code (syntax errors, missing functions,
bad signals), the circuit breaker opens to prevent wasted API calls and compute.

Grace Period & Minimum Attempts:
    The circuit breaker includes a **grace period** (default: first 2 turns) during
    which it will never fire, giving the LLM time to calibrate.  Additionally, a
    **minimum attempts** threshold (default: 5) must be reached before the breaker
    starts evaluating the pass rate at all.  This prevents premature abandonment
    after just a handful of early failures (e.g. 0/3 on turn 1).

Usage:
    cb = CircuitBreaker(window=20, min_pass_rate=0.2, min_attempts=5, grace_turns=2)
    for attempt in range(3):
        passed, errors = run_validation_gates(strategy_code)
        cb.record(passed, turn=current_turn)
        if cb.is_open():
            logger.warning("Circuit breaker open! Pass rate: %.1f%%", cb.pass_rate * 100)
            break
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CircuitBreakerStatus(str, Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"    # Normal operation — LLM quality is acceptable
    OPEN = "OPEN"        # Halted — LLM quality too low, stop making calls


@dataclass
class CircuitBreakerState:
    """Serializable snapshot of circuit breaker state for persistence."""

    total_attempts: int = 0
    passes: int = 0
    failures: int = 0
    window: int = 20
    min_pass_rate: float = 0.2
    min_attempts: int = 5
    grace_turns: int = 2
    status: str = CircuitBreakerStatus.CLOSED
    history: list = field(default_factory=list)
    # Each entry: {"passed": bool, "turn": int | None, "mandate": str | None}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CircuitBreakerState":
        known = {
            "total_attempts", "passes", "failures", "window",
            "min_pass_rate", "min_attempts", "grace_turns", "status", "history",
        }
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self, **kwargs) -> str:
        import json
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, blob: str) -> "CircuitBreakerState":
        import json
        return cls.from_dict(json.loads(blob))


class CircuitBreaker:
    """Track LLM validation pass rate over a sliding window.

    Opens (halts) when the recent pass rate drops below *min_pass_rate*,
    but only after at least *min_attempts* have been recorded AND the
    current turn exceeds *grace_turns*.

    Args:
        window: Number of recent attempts to consider.
        min_pass_rate: Minimum pass rate (0.0-1.0) to stay closed.
            Default 0.2 = 20% of recent validations must pass.
        min_attempts: Don't evaluate the pass rate until at least this
            many attempts have been recorded.  Prevents premature
            firing on tiny samples (e.g. 0/3).  Default 5.
        grace_turns: Number of initial turns during which the breaker
            will never fire, regardless of pass rate.  Gives the LLM
            time to calibrate.  Default 2.
    """

    def __init__(
        self,
        window: int = 20,
        min_pass_rate: float = 0.2,
        min_attempts: int = 5,
        grace_turns: int = 2,
    ):
        self.window = window
        self.min_pass_rate = min_pass_rate
        self.min_attempts = min_attempts
        self.grace_turns = grace_turns
        self._history: list[dict] = []

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def total_attempts(self) -> int:
        return len(self._history)

    @property
    def passes(self) -> int:
        return sum(1 for e in self._history if e["passed"])

    @property
    def failures(self) -> int:
        return sum(1 for e in self._history if not e["passed"])

    @property
    def status(self) -> CircuitBreakerStatus:
        if self._should_open():
            return CircuitBreakerStatus.OPEN
        return CircuitBreakerStatus.CLOSED

    @property
    def pass_rate(self) -> float:
        """Pass rate over the sliding window."""
        window_entries = self._history[-self.window:]
        if not window_entries:
            return 0.0
        passed = sum(1 for e in window_entries if e["passed"])
        return passed / len(window_entries)

    # ── Core Methods ────────────────────────────────────────────────────────

    def record(self, passed: bool, turn: int | None = None,
               mandate: str | None = None) -> None:
        """Record a single validation result.

        Args:
            passed: Whether the LLM output passed validation.
            turn: Optional iteration turn number for logging.
            mandate: Optional mandate name for logging.
        """
        self._history.append({
            "passed": bool(passed),
            "turn": turn,
            "mandate": mandate,
        })

    def record_batch(self, results: list[bool]) -> None:
        """Record multiple validation results at once."""
        for passed in results:
            self.record(passed)

    def is_open(self) -> bool:
        """Check if the circuit breaker is open (should halt)."""
        if not self._history:
            return False
        return self._should_open()

    def reset(self) -> None:
        """Clear all history and reset to closed state."""
        self._history.clear()

    # ── Persistence ─────────────────────────────────────────────────────────

    def get_state(self) -> CircuitBreakerState:
        """Return a serializable snapshot of current state."""
        return CircuitBreakerState(
            total_attempts=self.total_attempts,
            passes=self.passes,
            failures=self.failures,
            window=self.window,
            min_pass_rate=self.min_pass_rate,
            status=self.status,
            history=list(self._history),
        )

    @classmethod
    def restore(cls, state: CircuitBreakerState) -> "CircuitBreaker":
        """Restore a circuit breaker from a saved state."""
        cb = cls(window=state.window, min_pass_rate=state.min_pass_rate,
                min_attempts=state.min_attempts, grace_turns=state.grace_turns)
        cb._history = list(state.history)
        return cb

    # ── Reporting ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of circuit breaker state."""
        window_entries = self._history[-self.window:]
        window_passes = sum(1 for e in window_entries if e["passed"])
        window_total = len(window_entries)
        rate = (window_passes / window_total * 100) if window_total else 0.0
        return (
            f"CircuitBreaker [{self.status}] "
            f"Window: {window_passes}/{window_total} ({rate:.1f}%) "
            f"Overall: {self.passes}/{self.total_attempts} "
            f"Threshold: {self.min_pass_rate:.0%}"
        )

    # ── Private ─────────────────────────────────────────────────────────────

    def _should_open(self) -> bool:
        """Determine if the breaker should be open based on windowed pass rate.

        Returns False during the grace period or when fewer than
        *min_attempts* results have been recorded.
        """
        # Grace period: don't fire during the first N turns
        turns_seen = {e.get("turn") for e in self._history if e.get("turn") is not None}
        if turns_seen and max(turns_seen) <= self.grace_turns:
            return False

        # Minimum attempts: need enough data before evaluating
        if len(self._history) < self.min_attempts:
            return False

        window_entries = self._history[-self.window:]
        if not window_entries:
            return False
        passed = sum(1 for e in window_entries if e["passed"])
        rate = passed / len(window_entries)
        return rate < self.min_pass_rate
