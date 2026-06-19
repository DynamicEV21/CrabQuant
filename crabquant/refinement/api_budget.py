"""
ApiBudgetTracker — tracks LLM API prompt usage with daily/weekly budgets,
throttling, and persistence.

Budget defaults
---------------
- Daily prompt limit: 500
- Weekly prompt limit: 2 000
- Throttle threshold: 80 % of daily budget → recommend glm-4.7
- Alert threshold:   90 % of daily budget
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_DAILY_LIMIT: int = 500
DEFAULT_WEEKLY_LIMIT: int = 2_000
THROTTLE_THRESHOLD: float = 0.80  # 80 %
ALERT_THRESHOLD: float = 0.90  # 90 %

MODEL_PREMIUM: str = "zai/glm-5-turbo"
MODEL_THROTTLED: str = "zai/glm-4.7"

_HISTORY_DAYS: int = 7  # how many past days to keep in history


# ── Tracker ───────────────────────────────────────────────────────────────


class ApiBudgetTracker:
    """Track LLM prompt counts with daily/weekly budgets and persistence."""

    def __init__(
        self,
        budget_file: str = "results/api_budget.json",
        daily_limit: int = DEFAULT_DAILY_LIMIT,
        weekly_limit: int = DEFAULT_WEEKLY_LIMIT,
    ) -> None:
        self._budget_file = Path(budget_file)
        self.daily_limit = max(daily_limit, 0)
        self.weekly_limit = max(weekly_limit, 0)

        today = date.today().isoformat()

        # Try loading persisted state
        if self._budget_file.exists():
            data = self._load()
            self._last_reset_date: str = data.get("last_reset_date", today)
            self.daily_count: int = data.get("daily_count", 0)
            self.weekly_count: int = data.get("weekly_count", 0)
            self._history: dict[str, int] = data.get("history", {})
        else:
            self._last_reset_date = today
            self.daily_count = 0
            self.weekly_count = 0
            self._history = {}

        # Auto-reset if the day has rolled over
        if self._last_reset_date != today:
            self.reset_daily()

    # ── Public API ──────────────────────────────────────────────────────

    def record_prompt(
        self,
        model: str = MODEL_PREMIUM,
        tokens: int | None = None,
    ) -> None:
        """Record a single LLM API call and persist."""
        if model is None:
            model = MODEL_PREMIUM
        if tokens is not None and tokens < 0:
            logger.warning("record_prompt received negative token count %s — treating as 0", tokens)
            tokens = 0

        self.daily_count += 1
        self.weekly_count += 1

        # Update history for today
        today = date.today().isoformat()
        self._history[today] = self._history.get(today, 0) + 1

        # Check alert threshold
        if self.daily_limit > 0:
            usage_pct = self.daily_count / self.daily_limit
            if usage_pct >= ALERT_THRESHOLD:
                logger.warning(
                    "API budget alert: %.0f%% of daily budget consumed (%d/%d)",
                    usage_pct * 100,
                    self.daily_count,
                    self.daily_limit,
                )

        self._save()

    def should_throttle(self) -> bool:
        """Return True if daily usage has reached the throttle threshold (80 %)."""
        if self.daily_limit == 0:
            return True  # zero budget → always throttle
        return (self.daily_count / self.daily_limit) >= THROTTLE_THRESHOLD

    def get_recommended_model(self) -> str:
        """Return the model to use based on current budget state."""
        return MODEL_THROTTLED if self.should_throttle() else MODEL_PREMIUM

    def get_usage_summary(self) -> dict:
        """Return a snapshot of current usage statistics."""
        today = date.today().isoformat()
        daily_pct = (
            self.daily_count / self.daily_limit if self.daily_limit > 0 else float("inf")
        )
        weekly_pct = (
            self.weekly_count / self.weekly_limit if self.weekly_limit > 0 else float("inf")
        )
        return {
            "daily_count": self.daily_count,
            "daily_limit": self.daily_limit,
            "daily_pct": round(daily_pct, 4),
            "weekly_count": self.weekly_count,
            "weekly_limit": self.weekly_limit,
            "weekly_pct": round(weekly_pct, 4),
            "throttled": self.should_throttle(),
            "recommended_model": self.get_recommended_model(),
            "alert_active": daily_pct >= ALERT_THRESHOLD if self.daily_limit > 0 else True,
            "last_reset_date": self._last_reset_date,
            "today": today,
            "history": dict(self._history),
        }

    def reset_daily(self) -> None:
        """Reset the daily counter and prune old history entries."""
        today = date.today().isoformat()

        # Carry forward today's count into history before resetting
        if self._last_reset_date and self._last_reset_date != today:
            self._history[self._last_reset_date] = self.daily_count

        self.daily_count = 0
        self._last_reset_date = today

        # Prune history older than _HISTORY_DAYS
        cutoff = _date_n_days_ago(_HISTORY_DAYS)
        self._history = {
            d: c for d, c in self._history.items() if d >= cutoff
        }

        # Recompute weekly count from history
        week_cutoff = _date_n_days_ago(7)
        self.weekly_count = sum(
            c for d, c in self._history.items() if d >= week_cutoff
        )

        self._save()

    # ── Persistence helpers ────────────────────────────────────────────

    def _save(self) -> None:
        """Write current state to the budget file."""
        self._budget_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "daily_count": self.daily_count,
            "weekly_count": self.weekly_count,
            "daily_limit": self.daily_limit,
            "weekly_limit": self.weekly_limit,
            "last_reset_date": self._last_reset_date,
            "history": self._history,
        }
        self._budget_file.write_text(json.dumps(payload, indent=2))

    def _load(self) -> dict:
        """Load persisted state from the budget file."""
        try:
            return json.loads(self._budget_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load budget file %s: %s", self._budget_file, exc)
            return {}


# ── Utility ───────────────────────────────────────────────────────────────


def _date_n_days_ago(n: int) -> str:
    """Return ISO date string for *n* days ago."""
    from datetime import timedelta

    return (date.today() - timedelta(days=n)).isoformat()
