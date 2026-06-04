"""
API Budget Tracker — monitors LLM API token usage and cost.

Tracks prompt_tokens, completion_tokens, and total cost across all API calls.
Persists state to disk (JSON) so it survives restarts.
Provides budget enforcement to prevent runaway API spending.

Phase 6 prep item — wired into llm_api.py for automatic tracking.
"""

import os

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default cost per 1M tokens (z.ai / GLM-5 pricing — approximate)
DEFAULT_COST_PER_MILLION_PROMPT = 2.0    # $2/MTok input
DEFAULT_COST_PER_MILLION_COMPLETION = 8.0  # $8/MTok output


@dataclass
class ApiCallRecord:
    """Single API call record."""
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    cost_usd: float
    success: bool
    error: str = ""
    mandate_id: str = ""
    turn: int = 0


@dataclass
class ApiBudgetState:
    """Persistent API budget state."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_errors: int = 0
    started_at: str = ""
    last_call_at: str = ""

    # Budget limits (0 = no limit)
    max_cost_usd: float = 0.0
    max_calls: int = 0
    max_prompt_tokens: int = 0

    # Per-model breakdown
    model_usage: dict = field(default_factory=dict)

    # Per-mandate breakdown (mandate_id → {tokens, cost, calls})
    mandate_usage: dict = field(default_factory=dict)

    # Recent call history (last 100)
    recent_calls: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ApiBudgetState":
        from dataclasses import fields
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ── Tracker class ─────────────────────────────────────────────────────────


class ApiBudgetTracker:
    """Tracks API usage across all LLM calls with budget enforcement."""

    def __init__(
        self,
        state_path: str = "results/api_budget.json",
        cost_per_million_prompt: float = DEFAULT_COST_PER_MILLION_PROMPT,
        cost_per_million_completion: float = DEFAULT_COST_PER_MILLION_COMPLETION,
        max_history: int = 100,
    ):
        self.state_path = Path(state_path)
        self.cost_per_million_prompt = cost_per_million_prompt
        self.cost_per_million_completion = cost_per_million_completion
        self.max_history = max_history
        self.state = self._load()

    def _load(self) -> ApiBudgetState:
        """Load state from disk."""
        try:
            if self.state_path.exists():
                data = json.loads(self.state_path.read_text())
                return ApiBudgetState.from_dict(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Corrupt API budget state, starting fresh: %s", e)
        return ApiBudgetState(
            started_at=datetime.now(timezone.utc).isoformat()
        )

    def _save(self) -> None:
        """Persist state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state.to_dict(), indent=2)
        )

    def record_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        success: bool = True,
        error: str = "",
        mandate_id: str = "",
        turn: int = 0,
    ) -> ApiCallRecord:
        """Record a single API call and update budget state.

        Args:
            model: Model name used.
            prompt_tokens: Tokens in the prompt.
            completion_tokens: Tokens in the completion.
            latency_seconds: Wall-clock time for the call.
            success: Whether the call succeeded.
            error: Error message if failed.
            mandate_id: Optional mandate identifier for per-mandate tracking.
            turn: Refinement loop turn number.

        Returns:
            ApiCallRecord for the call.

        Raises:
            BudgetExceededError: If budget limit would be exceeded.
        """
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = (
            prompt_tokens * self.cost_per_million_prompt / 1_000_000
            + completion_tokens * self.cost_per_million_completion / 1_000_000
        )

        record = ApiCallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=round(latency_seconds, 2),
            cost_usd=round(cost_usd, 6),
            success=success,
            error=error,
            mandate_id=mandate_id,
            turn=turn,
        )

        # Check budget before updating
        self._check_budget(cost_usd, total_tokens)

        # Update aggregate state
        self.state.total_prompt_tokens += prompt_tokens
        self.state.total_completion_tokens += completion_tokens
        self.state.total_tokens += total_tokens
        self.state.total_cost_usd = round(self.state.total_cost_usd + cost_usd, 6)
        self.state.total_calls += 1
        self.state.last_call_at = record.timestamp
        if not success:
            self.state.total_errors += 1

        # Per-model breakdown
        if model not in self.state.model_usage:
            self.state.model_usage[model] = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        mu = self.state.model_usage[model]
        mu["calls"] += 1
        mu["prompt_tokens"] += prompt_tokens
        mu["completion_tokens"] += completion_tokens
        mu["total_tokens"] += total_tokens
        mu["cost_usd"] = round(mu["cost_usd"] + cost_usd, 6)

        # Per-mandate breakdown
        if mandate_id:
            if mandate_id not in self.state.mandate_usage:
                self.state.mandate_usage[mandate_id] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "turns": 0,
                }
            mm = self.state.mandate_usage[mandate_id]
            mm["calls"] += 1
            mm["prompt_tokens"] += prompt_tokens
            mm["completion_tokens"] += completion_tokens
            mm["total_tokens"] += total_tokens
            mm["cost_usd"] = round(mm["cost_usd"] + cost_usd, 6)
            mm["turns"] = max(mm["turns"], turn)

        # Recent calls (bounded)
        self.state.recent_calls.append(asdict(record))
        if len(self.state.recent_calls) > self.max_history:
            self.state.recent_calls = self.state.recent_calls[-self.max_history:]

        self._save()
        return record

    def _check_budget(self, cost_usd: float, total_tokens: int) -> None:
        """Check if budget limits would be exceeded. Raise if so."""
        if self.state.max_cost_usd > 0:
            if self.state.total_cost_usd + cost_usd > self.state.max_cost_usd:
                raise BudgetExceededError(
                    f"Cost limit ${self.state.max_cost_usd:.2f} would be exceeded "
                    f"(current: ${self.state.total_cost_usd:.2f}, "
                    f"this call: ${cost_usd:.4f})"
                )
        if self.state.max_calls > 0:
            if self.state.total_calls + 1 > self.state.max_calls:
                raise BudgetExceededError(
                    f"Call limit {self.state.max_calls} would be exceeded "
                    f"(current: {self.state.total_calls})"
                )
        if self.state.max_prompt_tokens > 0:
            if total_tokens > self.state.max_prompt_tokens:
                raise BudgetExceededError(
                    f"Token limit {self.state.max_prompt_tokens} would be exceeded "
                    f"(current: {self.state.total_tokens}, "
                    f"this call: {total_tokens})"
                )

    def set_budget(
        self,
        max_cost_usd: float = 0.0,
        max_calls: int = 0,
        max_prompt_tokens: int = 0,
    ) -> None:
        """Set budget limits. 0 means no limit."""
        self.state.max_cost_usd = max_cost_usd
        self.state.max_calls = max_calls
        self.state.max_prompt_tokens = max_prompt_tokens
        self._save()

    def get_summary(self) -> dict:
        """Get a human-readable summary of API usage."""
        s = self.state
        uptime = ""
        if s.started_at:
            try:
                started = datetime.fromisoformat(s.started_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                hours = (now - started).total_seconds() / 3600
                uptime = f"{hours:.1f}h"
            except (ValueError, TypeError):
                uptime = "unknown"

        return {
            "total_calls": s.total_calls,
            "total_errors": s.total_errors,
            "error_rate": round(s.total_errors / max(s.total_calls, 1) * 100, 1),
            "total_prompt_tokens": s.total_prompt_tokens,
            "total_completion_tokens": s.total_completion_tokens,
            "total_tokens": s.total_tokens,
            "total_cost_usd": round(s.total_cost_usd, 4),
            "avg_tokens_per_call": round(s.total_tokens / max(s.total_calls, 1)),
            "avg_cost_per_call": round(s.total_cost_usd / max(s.total_calls, 1), 4),
            "avg_latency_seconds": self._avg_latency(),
            "uptime": uptime,
            "budget_remaining": self._budget_remaining(),
            "models_used": list(s.model_usage.keys()),
            "mandates_tracked": len(s.mandate_usage),
        }

    def get_mandate_summary(self, mandate_id: str) -> Optional[dict]:
        """Get usage summary for a specific mandate."""
        return self.state.mandate_usage.get(mandate_id)

    def reset(self) -> None:
        """Reset all tracking state."""
        self.state = ApiBudgetState(
            started_at=datetime.now(timezone.utc).isoformat()
        )
        self._save()

    def _avg_latency(self) -> float:
        """Average latency of recent calls."""
        calls = self.state.recent_calls
        if not calls:
            return 0.0
        latencies = [c.get("latency_seconds", 0) for c in calls[-20:]]
        return round(sum(latencies) / len(latencies), 2)

    def _budget_remaining(self) -> dict:
        """How much budget remains."""
        s = self.state
        remaining = {}
        if s.max_cost_usd > 0:
            remaining["cost_usd"] = round(max(0, s.max_cost_usd - s.total_cost_usd), 4)
        if s.max_calls > 0:
            remaining["calls"] = max(0, s.max_calls - s.total_calls)
        if s.max_prompt_tokens > 0:
            remaining["prompt_tokens"] = max(0, s.max_prompt_tokens - s.total_tokens)
        return remaining


class BudgetExceededError(Exception):
    """Raised when an API call would exceed budget limits."""
    pass


# ── Global singleton ─────────────────────────────────────────────────────

_global_tracker: Optional[ApiBudgetTracker] = None


def get_global_tracker() -> ApiBudgetTracker:
    """Get or create the global API budget tracker singleton.

    State file defaults to results/api_budget.json.
    Override with CRABQUANT_API_BUDGET_PATH env var.
    """
    global _global_tracker
    if _global_tracker is None:
        state_path = os.environ.get(
            "CRABQUANT_API_BUDGET_PATH", "results/api_budget.json"
        )
        _global_tracker = ApiBudgetTracker(state_path=state_path)
    return _global_tracker
