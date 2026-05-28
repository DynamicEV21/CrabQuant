"""
Cosmetic guard — track action history and force structural intervention after
consecutive modify_params actions.

Detects when the LLM is stuck making cosmetic parameter tweaks instead of
structural changes, and forces a different action type.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ── Action types that count as "structural" (non-cosmetic) ──────────────────

_STRUCTURAL_ACTIONS: list[str] = [
    "full_rewrite",
    "change_entry_logic",
    "change_exit_logic",
    "add_regime_filter",
    "novel",
]

_DEFAULT_THRESHOLD = 3


@dataclass
class CosmeticGuardState:
    """Persistent state for the cosmetic guard across turns."""

    consecutive_modify_params: int = 0
    total_modify_params: int = 0
    threshold: int = _DEFAULT_THRESHOLD
    action_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "consecutive_modify_params": self.consecutive_modify_params,
            "total_modify_params": self.total_modify_params,
            "threshold": self.threshold,
            "action_history": self.action_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CosmeticGuardState":
        return cls(
            consecutive_modify_params=d.get("consecutive_modify_params", 0),
            total_modify_params=d.get("total_modify_params", 0),
            threshold=d.get("threshold", _DEFAULT_THRESHOLD),
            action_history=d.get("action_history", []),
        )


@dataclass
class CosmeticGuardResult:
    """Result of a cosmetic guard check."""

    forced: bool          # True when intervention is required
    warning: str          # Human-readable explanation (empty when not forced)
    forced_action: str    # Suggested structural action type (empty when not forced)


def check_cosmetic_guard(
    history: list[dict],
    state: CosmeticGuardState | None = None,
) -> tuple[CosmeticGuardState, CosmeticGuardResult]:
    """Check if the LLM is stuck making consecutive cosmetic modify_params actions.

    Scans the action history for consecutive modify_params. When the count
    reaches the threshold, forces a structural intervention.

    Args:
        history: List of turn dicts with an "action" key.
        state: Optional pre-existing state (for persistence across turns).

    Returns:
        (updated_state, result) tuple.
    """
    if state is None:
        state = CosmeticGuardState()

    # Count consecutive modify_params from the end of history
    consecutive = 0
    for entry in reversed(history):
        action = entry.get("action", "")
        if action == "modify_params":
            consecutive += 1
        else:
            break

    # Update state
    state.consecutive_modify_params = consecutive

    # Count total modify_params in current history
    new_modify_count = sum(1 for e in history if e.get("action") == "modify_params")
    # Track previously-seen modify_params from state's action_history
    prev_modify_count = sum(1 for e in state.action_history if e == "modify_params")
    # Increment total by only the *new* modify_params entries
    state.total_modify_params += max(new_modify_count - prev_modify_count, 0)
    # Ensure total is at least the current history's count
    state.total_modify_params = max(state.total_modify_params, new_modify_count)

    # Track action history
    state.action_history = [e.get("action", "") for e in history]

    # Check threshold
    if consecutive >= state.threshold:
        forced_action = random.choice(_STRUCTURAL_ACTIONS)
        return (
            state,
            CosmeticGuardResult(
                forced=True,
                warning=(
                    f"{consecutive} consecutive modify_params actions detected. "
                    f"Forcing structural intervention: {forced_action}. "
                    f"Parameter tweaks alone cannot fix this failure mode."
                ),
                forced_action=forced_action,
            ),
        )

    return (
        state,
        CosmeticGuardResult(forced=False, warning="", forced_action=""),
    )
