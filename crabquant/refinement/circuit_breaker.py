"""
Circuit Breaker — Track LLM validation pass rate and halt if below threshold.

When the LLM consistently produces invalid code (syntax errors, missing functions,
bad signals), the circuit breaker opens to prevent wasted API calls and compute.

Usage:
    cb = CircuitBreaker(window=20, min_pass_rate=0.3)
    for attempt in range(3):
        passed, errors = run_validation_gates(strategy_code)
        cb.record(passed)
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
    min_pass_rate: float = 0.3
    status: str = CircuitBreakerStatus.CLOSED
    history: list = field(default_factory=list)
    # Each entry: {"passed": bool, "turn": int | None, "mandate": str | None}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CircuitBreakerState":
        known = {
            "total_attempts", "passes", "failures", "window",
            "min_pass_rate", "status", "history",
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

    Opens (halts) when the recent pass rate drops below min_pass_rate.
    Closes (resumes) when the pass rate recovers above the threshold.

    Args:
        window: Number of recent attempts to consider.
        min_pass_rate: Minimum pass rate (0.0-1.0) to stay closed.
            Default 0.3 = 30% of recent validations must pass.
    """

    def __init__(self, window: int = 20, min_pass_rate: float = 0.3):
        self.window = window
        self.min_pass_rate = min_pass_rate
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
        cb = cls(window=state.window, min_pass_rate=state.min_pass_rate)
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
        """Determine if the breaker should be open based on windowed pass rate."""
        window_entries = self._history[-self.window:]
        if not window_entries:
            return False
        passed = sum(1 for e in window_entries if e["passed"])
        rate = passed / len(window_entries)
        return rate < self.min_pass_rate
