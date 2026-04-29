"""
Cosmetic guard — track action history and force structural intervention after
consecutive modify_params actions or repeated action/failure_mode failures.

Detects when the LLM is stuck making cosmetic parameter tweaks instead of
structural changes, and forces a different action type.  Also tracks
(failure_mode, action) cooldowns so that repeatedly failing with the same
action on the same failure mode triggers warnings and forced overrides.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from crabquant.refinement.prompts import RECOMMENDED_ACTIONS

# ── Action types that count as "structural" (non-cosmetic) ──────────────────

_STRUCTURAL_ACTIONS: list[str] = [
    "full_rewrite",
    "change_entry_logic",
    "change_exit_logic",
    "add_regime_filter",
    "novel",
]

_DEFAULT_THRESHOLD = 3
_COOLDOWN_WARN_THRESHOLD = 2
_COOLDOWN_FORCE_THRESHOLD = 3


# ── Failure-mode → recommended actions lookup ──────────────────────────────
# RECOMMENDED_ACTIONS in prompts.py maps failure_mode → (action, reason).
# We build a fallback list of structural actions per failure mode for cases
# where the recommended action itself has been tried and failed.

def _get_recommended_action(failure_mode: str) -> str | None:
    """Return the recommended action for a failure mode, or None."""
    entry = RECOMMENDED_ACTIONS.get(failure_mode)
    if entry is not None:
        return entry[0]
    return None


def _get_fallback_actions(failure_mode: str) -> list[str]:
    """Return ordered list of fallback actions for a failure mode.

    Starts with the recommended action, then appends structural actions
    that are not the recommended one.
    """
    recommended = _get_recommended_action(failure_mode)
    fallbacks: list[str] = []
    if recommended:
        fallbacks.append(recommended)
    for a in _STRUCTURAL_ACTIONS:
        if a not in fallbacks:
            fallbacks.append(a)
    # Add non-structural actions as last resort
    for a in ("replace_indicator", "add_filter", "modify_params"):
        if a not in fallbacks:
            fallbacks.append(a)
    return fallbacks


def _pick_forced_action(failure_mode: str, cooldowns: dict[str, dict[str, int]]) -> str:
    """Pick the best forced action for a failure mode, avoiding cooldown-hot actions.

    Strategy:
    1. Get ordered fallback actions for the failure mode.
    2. Skip any action that already has >= _COOLDOWN_FORCE_THRESHOLD consecutive
       failures for this failure mode.
    3. Among the remaining, prefer the first (recommended) one.
    4. If all are exhausted, fall back to random structural action.
    """
    fallbacks = _get_fallback_actions(failure_mode)
    mode_cooldowns = cooldowns.get(failure_mode, {})

    for action in fallbacks:
        consecutive = mode_cooldowns.get(action, 0)
        if consecutive < _COOLDOWN_FORCE_THRESHOLD:
            return action

    # All actions exhausted — random structural action
    return random.choice(_STRUCTURAL_ACTIONS)


@dataclass
class CosmeticGuardState:
    """Persistent state for the cosmetic guard across turns."""

    consecutive_modify_params: int = 0
    total_modify_params: int = 0
    threshold: int = _DEFAULT_THRESHOLD
    action_history: list[str] = field(default_factory=list)

    # ── Within-run action cooldown ─────────────────────────────────────
    # Maps failure_mode → {action → consecutive_fail_count}
    # Tracked across turns within a single run to detect when the same
    # action keeps failing for the same failure mode.
    cooldowns: dict[str, dict[str, int]] = field(default_factory=dict)

    # Cooldown thresholds (configurable for testing)
    cooldown_warn_threshold: int = _COOLDOWN_WARN_THRESHOLD
    cooldown_force_threshold: int = _COOLDOWN_FORCE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "consecutive_modify_params": self.consecutive_modify_params,
            "total_modify_params": self.total_modify_params,
            "threshold": self.threshold,
            "action_history": self.action_history,
            "cooldowns": self.cooldowns,
            "cooldown_warn_threshold": self.cooldown_warn_threshold,
            "cooldown_force_threshold": self.cooldown_force_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CosmeticGuardState":
        return cls(
            consecutive_modify_params=d.get("consecutive_modify_params", 0),
            total_modify_params=d.get("total_modify_params", 0),
            threshold=d.get("threshold", _DEFAULT_THRESHOLD),
            action_history=d.get("action_history", []),
            cooldowns=d.get("cooldowns", {}),
            cooldown_warn_threshold=d.get("cooldown_warn_threshold", _COOLDOWN_WARN_THRESHOLD),
            cooldown_force_threshold=d.get("cooldown_force_threshold", _COOLDOWN_FORCE_THRESHOLD),
        )


@dataclass
class CosmeticGuardResult:
    """Result of a cosmetic guard check."""

    forced: bool          # True when intervention is required
    warning: str          # Human-readable explanation (empty when not forced)
    forced_action: str    # Suggested structural action type (empty when not forced)
    cooldown_warning: str # Warning about repeated (failure_mode, action) failures


def update_cooldowns(
    state: CosmeticGuardState,
    failure_mode: str,
    action: str,
    success: bool,
) -> None:
    """Update the cooldown tracker based on the outcome of a turn.

    Args:
        state: The cosmetic guard state to update.
        failure_mode: The failure mode for this turn (empty string if success).
        action: The action taken this turn.
        success: Whether the turn succeeded (Sharpe >= target).
    """
    if not failure_mode:
        # Turn succeeded — reset cooldowns for the previous failure mode
        # associated with this action (if any)
        for mode_actions in state.cooldowns.values():
            if action in mode_actions:
                mode_actions[action] = 0
        return

    if failure_mode not in state.cooldowns:
        state.cooldowns[failure_mode] = {}

    mode_cooldowns = state.cooldowns[failure_mode]

    if action in mode_cooldowns:
        if success:
            mode_cooldowns[action] = 0
        else:
            mode_cooldowns[action] += 1
    else:
        # First failure with this (failure_mode, action) pair
        mode_cooldowns[action] = 1


def get_cooldown_warning(
    state: CosmeticGuardState,
    failure_mode: str,
    action: str,
) -> str:
    """Check if there's a cooldown warning for the given (failure_mode, action).

    Returns a warning string if consecutive failures >= warn_threshold, else "".
    """
    if not failure_mode:
        return ""

    mode_cooldowns = state.cooldowns.get(failure_mode, {})
    consecutive = mode_cooldowns.get(action, 0)

    if consecutive >= state.cooldown_force_threshold:
        return (
            f"Action cooldown CRITICAL: '{action}' has failed {consecutive} "
            f"consecutive times on failure_mode '{failure_mode}'. "
            f"Forcing a different action."
        )
    elif consecutive >= state.cooldown_warn_threshold:
        return (
            f"Action cooldown warning: '{action}' has failed {consecutive} "
            f"consecutive times on failure_mode '{failure_mode}'. "
            f"Consider switching to a different action type."
        )
    return ""


def check_cosmetic_guard(
    history: list[dict],
    state: CosmeticGuardState | None = None,
    failure_mode: str = "",
) -> tuple[CosmeticGuardState, CosmeticGuardResult]:
    """Check if the LLM is stuck making consecutive cosmetic modify_params actions
    or if a (failure_mode, action) pair has exceeded the cooldown threshold.

    Scans the action history for consecutive modify_params. When the count
    reaches the threshold, forces a structural intervention using the
    recommended action for the current failure mode.

    Also checks the cooldown tracker: if the last action used for the current
    failure mode has failed >= cooldown_force_threshold consecutive times,
    forces a different action.

    Args:
        history: List of turn dicts with an "action" key.
        state: Optional pre-existing state (for persistence across turns).
        failure_mode: The current failure mode (from the previous turn's
            classification, or from the most recent history entry).

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

    # Determine the most recent action and failure mode from history if not provided
    last_action = ""
    effective_failure_mode = failure_mode
    if history:
        last_action = history[-1].get("action", "")
        if not effective_failure_mode:
            effective_failure_mode = history[-1].get("failure_mode", "")

    # ── Check modify_params threshold ─────────────────────────────────
    if consecutive >= state.threshold:
        forced_action = _pick_forced_action(
            effective_failure_mode, state.cooldowns
        )
        return (
            state,
            CosmeticGuardResult(
                forced=True,
                warning=(
                    f"{consecutive} consecutive modify_params actions detected. "
                    f"Forcing structural intervention: {forced_action}. "
                    f"Failure mode: {effective_failure_mode or 'unknown'}. "
                    f"Parameter tweaks alone cannot fix this."
                ),
                forced_action=forced_action,
                cooldown_warning="",
            ),
        )

    # ── Check (failure_mode, action) cooldown ─────────────────────────
    cooldown_warning = ""
    cooldown_forced_action = ""

    if effective_failure_mode and last_action:
        mode_cooldowns = state.cooldowns.get(effective_failure_mode, {})
        consecutive_fails = mode_cooldowns.get(last_action, 0)

        if consecutive_fails >= state.cooldown_force_threshold:
            cooldown_forced_action = _pick_forced_action(
                effective_failure_mode, state.cooldowns
            )
            cooldown_warning = (
                f"Action '{last_action}' has failed {consecutive_fails} consecutive "
                f"times on failure_mode '{effective_failure_mode}'. "
                f"Forcing override to: {cooldown_forced_action}."
            )
        elif consecutive_fails >= state.cooldown_warn_threshold:
            cooldown_warning = (
                f"Action '{last_action}' has failed {consecutive_fails} consecutive "
                f"times on failure_mode '{effective_failure_mode}'. "
                f"Consider switching to a different action type."
            )

    if cooldown_forced_action:
        return (
            state,
            CosmeticGuardResult(
                forced=True,
                warning=cooldown_warning,
                forced_action=cooldown_forced_action,
                cooldown_warning=cooldown_warning,
            ),
        )

    return (
        state,
        CosmeticGuardResult(
            forced=False,
            warning="",
            forced_action="",
            cooldown_warning=cooldown_warning,
        ),
    )
