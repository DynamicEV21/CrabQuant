"""Tests for the cosmetic guard — failure-mode-aware action selection and cooldown tracking."""

from __future__ import annotations

import pytest

from crabquant.refinement.cosmetic_guard import (
    CosmeticGuardState,
    CosmeticGuardResult,
    check_cosmetic_guard,
    get_cooldown_warning,
    update_cooldowns,
    _get_recommended_action,
    _get_fallback_actions,
    _pick_forced_action,
    _STRUCTURAL_ACTIONS,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_history(*actions_and_modes: tuple[str, str]) -> list[dict]:
    """Build a history list from (action, failure_mode) pairs.

    failure_mode may be "" for successful turns.
    """
    history = []
    for i, (action, fmode) in enumerate(actions_and_modes, start=1):
        entry: dict = {"turn": i, "action": action}
        if fmode:
            entry["failure_mode"] = fmode
        history.append(entry)
    return history


# ── Recommended action lookups ─────────────────────────────────────────────

class TestRecommendedActionLookup:
    """Tests for _get_recommended_action and _get_fallback_actions."""

    def test_known_failure_mode_returns_action(self):
        assert _get_recommended_action("low_sharpe") == "replace_indicator"

    def test_unknown_failure_mode_returns_none(self):
        assert _get_recommended_action("totally_unknown_mode") is None

    def test_fallback_includes_recommended_first(self):
        fallbacks = _get_fallback_actions("low_sharpe")
        assert fallbacks[0] == "replace_indicator"

    def test_fallback_includes_structural_actions(self):
        fallbacks = _get_fallback_actions("excessive_drawdown")
        for a in _STRUCTURAL_ACTIONS:
            assert a in fallbacks

    def test_fallback_unknown_mode_still_has_structural(self):
        fallbacks = _get_fallback_actions("unknown_mode")
        assert len(fallbacks) >= len(_STRUCTURAL_ACTIONS)

    def test_fallback_no_duplicates(self):
        fallbacks = _get_fallback_actions("low_sharpe")
        assert len(fallbacks) == len(set(fallbacks))


# ── _pick_forced_action ────────────────────────────────────────────────────

class TestPickForcedAction:
    """Tests for _pick_forced_action cooldown-aware selection."""

    def test_prefers_recommended_action(self):
        # No cooldowns — should pick the recommended action first
        action = _pick_forced_action("flat_signal", {})
        assert action == "change_entry_logic"

    def test_skips_cooldown_exhausted_action(self):
        # "change_entry_logic" exhausted for "flat_signal"
        cooldowns = {"flat_signal": {"change_entry_logic": 3}}
        action = _pick_forced_action("flat_signal", cooldowns)
        assert action != "change_entry_logic"
        # Should fall to next in fallback list
        assert action in _STRUCTURAL_ACTIONS or action == "novel"

    def test_all_exhausted_falls_to_random(self):
        cooldowns = {"backtest_crash": {a: 3 for a in _STRUCTURAL_ACTIONS + ["replace_indicator", "add_filter", "modify_params"]}}
        action = _pick_forced_action("backtest_crash", cooldowns)
        assert action in _STRUCTURAL_ACTIONS  # random structural

    def test_unknown_failure_mode_uses_structural(self):
        action = _pick_forced_action("unknown_xyz", {})
        # Should return first structural action since no recommendation
        assert action in _STRUCTURAL_ACTIONS or action in ("replace_indicator", "add_filter")


# ── update_cooldowns ───────────────────────────────────────────────────────

class TestUpdateCooldowns:
    """Tests for update_cooldowns."""

    def test_first_failure_sets_count_to_one(self):
        state = CosmeticGuardState()
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        assert state.cooldowns["low_sharpe"]["modify_params"] == 1

    def test_consecutive_failures_increment(self):
        state = CosmeticGuardState()
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        assert state.cooldowns["low_sharpe"]["modify_params"] == 3

    def test_success_resets_cooldown_for_action(self):
        state = CosmeticGuardState()
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        update_cooldowns(state, "low_sharpe", "modify_params", success=True)
        assert state.cooldowns["low_sharpe"]["modify_params"] == 0

    def test_empty_failure_mode_resets_across_modes(self):
        """When failure_mode is empty (success), reset that action across all modes."""
        state = CosmeticGuardState()
        update_cooldowns(state, "low_sharpe", "replace_indicator", success=False)
        update_cooldowns(state, "flat_signal", "replace_indicator", success=False)
        # Simulate success
        update_cooldowns(state, "", "replace_indicator", success=True)
        assert state.cooldowns["low_sharpe"]["replace_indicator"] == 0
        assert state.cooldowns["flat_signal"]["replace_indicator"] == 0

    def test_different_actions_tracked_independently(self):
        state = CosmeticGuardState()
        update_cooldowns(state, "low_sharpe", "replace_indicator", success=False)
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        update_cooldowns(state, "low_sharpe", "replace_indicator", success=False)
        assert state.cooldowns["low_sharpe"]["replace_indicator"] == 2
        assert state.cooldowns["low_sharpe"]["modify_params"] == 1

    def test_different_failure_modes_tracked_independently(self):
        state = CosmeticGuardState()
        update_cooldowns(state, "low_sharpe", "replace_indicator", success=False)
        update_cooldowns(state, "flat_signal", "replace_indicator", success=False)
        assert state.cooldowns["low_sharpe"]["replace_indicator"] == 1
        assert state.cooldowns["flat_signal"]["replace_indicator"] == 1


# ── get_cooldown_warning ───────────────────────────────────────────────────

class TestGetCooldownWarning:
    """Tests for get_cooldown_warning."""

    def test_no_warning_when_below_threshold(self):
        state = CosmeticGuardState()
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        warning = get_cooldown_warning(state, "low_sharpe", "modify_params")
        assert warning == ""

    def test_warning_at_threshold(self):
        state = CosmeticGuardState(cooldown_warn_threshold=2)
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        warning = get_cooldown_warning(state, "low_sharpe", "modify_params")
        assert "cooldown warning" in warning.lower()
        assert "2" in warning

    def test_critical_warning_at_force_threshold(self):
        state = CosmeticGuardState(cooldown_warn_threshold=2, cooldown_force_threshold=3)
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        update_cooldowns(state, "low_sharpe", "modify_params", success=False)
        warning = get_cooldown_warning(state, "low_sharpe", "modify_params")
        assert "CRITICAL" in warning
        assert "3" in warning

    def test_no_warning_for_empty_failure_mode(self):
        state = CosmeticGuardState()
        warning = get_cooldown_warning(state, "", "modify_params")
        assert warning == ""

    def test_no_warning_for_unknown_pair(self):
        state = CosmeticGuardState()
        warning = get_cooldown_warning(state, "unknown", "never_tried")
        assert warning == ""


# ── check_cosmetic_guard: modify_params threshold ──────────────────────────

class TestCosmeticGuardModifyParams:
    """Tests for the existing modify_params consecutive threshold behavior."""

    def test_no_force_below_threshold(self):
        history = _make_history(
            ("replace_indicator", "low_sharpe"),
            ("modify_params", "low_sharpe"),
        )
        state, result = check_cosmetic_guard(history)
        assert result.forced is False
        assert result.forced_action == ""

    def test_force_at_threshold(self):
        history = _make_history(
            ("modify_params", "low_sharpe"),
            ("modify_params", "low_sharpe"),
            ("modify_params", "low_sharpe"),
        )
        state, result = check_cosmetic_guard(history)
        assert result.forced is True
        assert result.forced_action != ""

    def test_forced_action_uses_failure_mode_recommendation(self):
        """When modify_params threshold triggers, the forced action should
        be informed by the failure mode, not purely random."""
        history = _make_history(
            ("modify_params", "flat_signal"),
            ("modify_params", "flat_signal"),
            ("modify_params", "flat_signal"),
        )
        state, result = check_cosmetic_guard(history, failure_mode="flat_signal")
        assert result.forced is True
        # For flat_signal, recommended is change_entry_logic — should be picked
        # (unless cooldowns block it, which they don't here)
        assert result.forced_action == "change_entry_logic"

    def test_forced_action_falls_back_when_recommended_exhausted(self):
        """If the recommended action is cooldown-exhausted, fall to next."""
        history = _make_history(
            ("modify_params", "low_sharpe"),
            ("modify_params", "low_sharpe"),
            ("modify_params", "low_sharpe"),
        )
        state = CosmeticGuardState()
        # Exhaust the recommended action for low_sharpe (replace_indicator)
        state.cooldowns = {"low_sharpe": {"replace_indicator": 3}}
        state, result = check_cosmetic_guard(history, state, failure_mode="low_sharpe")
        assert result.forced is True
        assert result.forced_action != "replace_indicator"

    def test_consecutive_count_resets_on_different_action(self):
        history = _make_history(
            ("modify_params", "low_sharpe"),
            ("modify_params", "low_sharpe"),
            ("replace_indicator", "low_sharpe"),
            ("modify_params", "low_sharpe"),
        )
        state, result = check_cosmetic_guard(history)
        assert state.consecutive_modify_params == 1
        assert result.forced is False


# ── check_cosmetic_guard: cooldown-based forcing ───────────────────────────

class TestCosmeticGuardCooldownForcing:
    """Tests for the (failure_mode, action) cooldown forcing."""

    def test_no_cooldown_warning_fresh_state(self):
        history = _make_history(
            ("replace_indicator", "low_sharpe"),
            ("modify_params", "low_sharpe"),
        )
        state, result = check_cosmetic_guard(history, failure_mode="low_sharpe")
        assert result.forced is False
        assert result.cooldown_warning == ""

    def test_cooldown_warning_at_2_consecutive(self):
        """After 2 consecutive (failure_mode, action) failures, inject warning."""
        history = _make_history(
            ("replace_indicator", "low_sharpe"),
            ("replace_indicator", "low_sharpe"),
        )
        state = CosmeticGuardState(cooldown_warn_threshold=2, cooldown_force_threshold=3)
        # Simulate 2 failures with replace_indicator on low_sharpe
        state.cooldowns = {"low_sharpe": {"replace_indicator": 2}}
        state, result = check_cosmetic_guard(history, state, failure_mode="low_sharpe")
        assert result.forced is False
        assert "consecutive" in result.cooldown_warning.lower()
        assert "2" in result.cooldown_warning

    def test_cooldown_force_at_3_consecutive(self):
        """After 3 consecutive (failure_mode, action) failures, force override."""
        history = _make_history(
            ("replace_indicator", "low_sharpe"),
            ("replace_indicator", "low_sharpe"),
            ("replace_indicator", "low_sharpe"),
        )
        state = CosmeticGuardState(cooldown_warn_threshold=2, cooldown_force_threshold=3)
        state.cooldowns = {"low_sharpe": {"replace_indicator": 3}}
        state, result = check_cosmetic_guard(history, state, failure_mode="low_sharpe")
        assert result.forced is True
        assert result.forced_action != "replace_indicator"
        assert "Forcing override" in result.warning

    def test_cooldown_force_picks_failure_mode_aware_action(self):
        """Forced action from cooldown should use failure mode recommendations."""
        history = _make_history(
            ("add_filter", "flat_signal"),
            ("add_filter", "flat_signal"),
            ("add_filter", "flat_signal"),
        )
        state = CosmeticGuardState(cooldown_warn_threshold=2, cooldown_force_threshold=3)
        state.cooldowns = {"flat_signal": {"add_filter": 3}}
        state, result = check_cosmetic_guard(history, state, failure_mode="flat_signal")
        assert result.forced is True
        # Should pick recommended action for flat_signal (change_entry_logic)
        assert result.forced_action == "change_entry_logic"

    def test_cooldown_infers_failure_mode_from_history(self):
        """When failure_mode param is empty, infer from last history entry."""
        history = _make_history(
            ("replace_indicator", "low_sharpe"),
            ("replace_indicator", "low_sharpe"),
            ("replace_indicator", "low_sharpe"),
        )
        state = CosmeticGuardState(cooldown_warn_threshold=2, cooldown_force_threshold=3)
        state.cooldowns = {"low_sharpe": {"replace_indicator": 3}}
        # Don't pass failure_mode — should infer from history
        state, result = check_cosmetic_guard(history, state, failure_mode="")
        assert result.forced is True
        assert result.forced_action != "replace_indicator"


# ── CosmeticGuardState serialization ───────────────────────────────────────

class TestCosmeticGuardStateSerialization:
    """Tests for CosmeticGuardState.to_dict/from_dict roundtrip."""

    def test_roundtrip_preserves_cooldowns(self):
        state = CosmeticGuardState(threshold=5)
        state.cooldowns = {"low_sharpe": {"modify_params": 2, "replace_indicator": 1}}
        d = state.to_dict()
        restored = CosmeticGuardState.from_dict(d)
        assert restored.cooldowns == {"low_sharpe": {"modify_params": 2, "replace_indicator": 1}}
        assert restored.threshold == 5

    def test_from_dict_defaults_missing_keys(self):
        d = {"threshold": 3}
        restored = CosmeticGuardState.from_dict(d)
        assert restored.cooldowns == {}
        assert restored.consecutive_modify_params == 0

    def test_roundtrip_preserves_thresholds(self):
        state = CosmeticGuardState(
            cooldown_warn_threshold=4,
            cooldown_force_threshold=5,
        )
        d = state.to_dict()
        restored = CosmeticGuardState.from_dict(d)
        assert restored.cooldown_warn_threshold == 4
        assert restored.cooldown_force_threshold == 5
