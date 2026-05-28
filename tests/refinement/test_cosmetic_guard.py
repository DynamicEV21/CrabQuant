"""Tests for crabquant.refinement.cosmetic_guard — track action history, force structural intervention."""

import pytest

from crabquant.refinement.cosmetic_guard import (
    CosmeticGuardState,
    CosmeticGuardResult,
    check_cosmetic_guard,
)


class TestCosmeticGuardState:
    """Test the state dataclass."""

    def test_defaults(self):
        state = CosmeticGuardState()
        assert state.consecutive_modify_params == 0
        assert state.threshold == 3
        assert state.total_modify_params == 0
        assert state.action_history == []

    def test_custom_threshold(self):
        state = CosmeticGuardState(threshold=5)
        assert state.threshold == 5

    def test_to_dict(self):
        state = CosmeticGuardState(
            consecutive_modify_params=2,
            threshold=3,
            total_modify_params=5,
            action_history=["modify_params", "modify_params"],
        )
        d = state.to_dict()
        assert d["consecutive_modify_params"] == 2
        assert d["action_history"] == ["modify_params", "modify_params"]

    def test_from_dict_roundtrip(self):
        original = CosmeticGuardState(
            consecutive_modify_params=1,
            total_modify_params=4,
            action_history=["add_filter", "modify_params"],
        )
        restored = CosmeticGuardState.from_dict(original.to_dict())
        assert restored == original


class TestCheckCosmeticGuard:
    """Test the cosmetic guard check function."""

    def test_no_history_no_warning(self):
        state, result = check_cosmetic_guard([], CosmeticGuardState())
        assert result.forced is False
        assert result.warning == ""
        assert state.consecutive_modify_params == 0

    def test_single_modify_params_no_warning(self):
        history = [{"turn": 1, "action": "modify_params"}]
        state, result = check_cosmetic_guard(history, CosmeticGuardState())
        assert result.forced is False
        assert state.consecutive_modify_params == 1

    def test_two_consecutive_modify_params_no_warning(self):
        history = [
            {"turn": 1, "action": "modify_params"},
            {"turn": 2, "action": "modify_params"},
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced is False
        assert state.consecutive_modify_params == 2

    def test_three_consecutive_modify_params_triggers_warning(self):
        history = [
            {"turn": 1, "action": "modify_params"},
            {"turn": 2, "action": "modify_params"},
            {"turn": 3, "action": "modify_params"},
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced is True
        assert "modify_params" in result.warning.lower() or "cosmetic" in result.warning.lower() or "structural" in result.warning.lower()
        assert result.forced_action in ("full_rewrite", "change_entry_logic", "change_exit_logic", "add_regime_filter", "novel")

    def test_non_modify_params_resets_counter(self):
        history = [
            {"turn": 1, "action": "modify_params"},
            {"turn": 2, "action": "modify_params"},
            {"turn": 3, "action": "add_filter"},
            {"turn": 4, "action": "modify_params"},
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced is False
        assert state.consecutive_modify_params == 1

    def test_custom_threshold(self):
        history = [
            {"turn": i, "action": "modify_params"}
            for i in range(1, 5)
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=4))
        assert result.forced is True

    def test_custom_threshold_not_triggered_at_three(self):
        history = [
            {"turn": i, "action": "modify_params"}
            for i in range(1, 4)
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=5))
        assert result.forced is False

    def test_result_is_dataclass(self):
        state, result = check_cosmetic_guard([], CosmeticGuardState())
        assert isinstance(result, CosmeticGuardResult)
        assert isinstance(result.forced, bool)
        assert isinstance(result.warning, str)
        assert isinstance(result.forced_action, str)

    def test_persistent_state_tracks_total(self):
        state = CosmeticGuardState(total_modify_params=3)
        history = [{"turn": 1, "action": "modify_params"}]
        state, result = check_cosmetic_guard(history, state)
        assert state.total_modify_params == 4

    def test_empty_history_dict_no_action_key(self):
        history = [{"turn": 1}]
        state, result = check_cosmetic_guard(history, CosmeticGuardState())
        assert result.forced is False
        assert state.consecutive_modify_params == 0

    def test_forced_action_is_string(self):
        history = [
            {"turn": i, "action": "modify_params"}
            for i in range(1, 4)
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState())
        assert isinstance(result.forced_action, str)
        assert len(result.forced_action) > 0
