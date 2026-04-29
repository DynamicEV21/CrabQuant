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


class TestCosmeticGuardStateExpanded:
    """Additional tests for CosmeticGuardState serialization and edge cases."""

    def test_from_dict_missing_keys_use_defaults(self):
        state = CosmeticGuardState.from_dict({})
        assert state.consecutive_modify_params == 0
        assert state.total_modify_params == 0
        assert state.threshold == 3
        assert state.action_history == []

    def test_from_dict_partial_keys(self):
        state = CosmeticGuardState.from_dict({"threshold": 7})
        assert state.threshold == 7
        assert state.consecutive_modify_params == 0
        assert state.total_modify_params == 0

    def test_from_dict_with_all_keys(self):
        state = CosmeticGuardState.from_dict({
            "consecutive_modify_params": 5,
            "total_modify_params": 10,
            "threshold": 2,
            "action_history": ["modify_params", "full_rewrite"],
        })
        assert state.consecutive_modify_params == 5
        assert state.total_modify_params == 10
        assert state.threshold == 2
        assert state.action_history == ["modify_params", "full_rewrite"]

    def test_from_dict_ignores_extra_keys(self):
        state = CosmeticGuardState.from_dict({
            "consecutive_modify_params": 1,
            "total_modify_params": 2,
            "threshold": 3,
            "action_history": [],
            "unexpected_key": "ignored",
        })
        assert state.consecutive_modify_params == 1
        assert not hasattr(state, "unexpected_key")

    def test_to_dict_keys_match_expected(self):
        state = CosmeticGuardState()
        d = state.to_dict()
        expected_keys = {"consecutive_modify_params", "total_modify_params", "threshold", "action_history"}
        assert set(d.keys()) == expected_keys

    def test_roundtrip_preserves_all_fields(self):
        original = CosmeticGuardState(
            consecutive_modify_params=99,
            total_modify_params=200,
            threshold=10,
            action_history=["a", "b", "c"],
        )
        restored = CosmeticGuardState.from_dict(original.to_dict())
        assert restored.consecutive_modify_params == 99
        assert restored.total_modify_params == 200
        assert restored.threshold == 10
        assert restored.action_history == ["a", "b", "c"]

    def test_action_history_default_is_new_list(self):
        s1 = CosmeticGuardState()
        s2 = CosmeticGuardState()
        s1.action_history.append("x")
        assert s2.action_history == []

    def test_threshold_default_is_three(self):
        from crabquant.refinement.cosmetic_guard import _DEFAULT_THRESHOLD
        state = CosmeticGuardState()
        assert state.threshold == _DEFAULT_THRESHOLD


class TestCheckCosmeticGuardExpanded:
    """Additional tests for check_cosmetic_guard edge cases and branches."""

    def test_none_state_creates_fresh_state(self):
        history = []
        state, result = check_cosmetic_guard(history)
        assert state.consecutive_modify_params == 0
        assert result.forced is False

    def test_none_state_with_modify_params(self):
        history = [{"action": "modify_params"}]
        state, result = check_cosmetic_guard(history)
        assert state.consecutive_modify_params == 1
        assert result.forced is False

    def test_threshold_of_one_triggers_immediately(self):
        history = [{"action": "modify_params"}]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=1))
        assert result.forced is True
        assert len(result.forced_action) > 0

    def test_threshold_of_zero_always_triggers(self):
        history = [{"action": "full_rewrite"}]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=0))
        assert result.forced is True

    def test_threshold_of_zero_with_empty_history(self):
        history = []
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=0))
        assert result.forced is True

    def test_exactly_at_threshold_triggers(self):
        history = [
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "modify_params"},
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced is True

    def test_one_below_threshold_does_not_trigger(self):
        history = [
            {"action": "modify_params"},
            {"action": "modify_params"},
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced is False

    def test_far_above_threshold_triggers(self):
        history = [{"action": "modify_params"} for _ in range(20)]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced is True
        assert state.consecutive_modify_params == 20

    def test_single_structural_action_between_modify_params(self):
        history = [
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "full_rewrite"},
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "modify_params"},
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced is True
        assert state.consecutive_modify_params == 3

    def test_forced_warning_contains_consecutive_count(self):
        history = [{"action": "modify_params"} for _ in range(5)]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert "5" in result.warning
        assert "modify_params" in result.warning

    def test_forced_warning_contains_forced_action(self):
        history = [{"action": "modify_params"} for _ in range(3)]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced_action in result.warning

    def test_forced_warning_mentions_structural_intervention(self):
        history = [{"action": "modify_params"} for _ in range(3)]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert "structural" in result.warning.lower()

    def test_non_forced_result_has_empty_strings(self):
        history = [{"action": "full_rewrite"}]
        state, result = check_cosmetic_guard(history, CosmeticGuardState())
        assert result.forced is False
        assert result.warning == ""
        assert result.forced_action == ""

    def test_persistence_total_modify_params_across_calls(self):
        state = CosmeticGuardState()
        # First call: 2 modify_params
        history1 = [{"action": "modify_params"}, {"action": "modify_params"}]
        state, _ = check_cosmetic_guard(history1, state)
        assert state.total_modify_params == 2

        # Second call: same history again (no new entries)
        state, _ = check_cosmetic_guard(history1, state)
        assert state.total_modify_params == 2

    def test_persistence_total_increments_with_new_entries(self):
        state = CosmeticGuardState()
        history1 = [{"action": "modify_params"}]
        state, _ = check_cosmetic_guard(history1, state)
        assert state.total_modify_params == 1

        # history2 has 2 modify_params, state already saw 1 → max(1 + max(2-1,0), 2) = 2
        history2 = [
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "full_rewrite"},
        ]
        state, _ = check_cosmetic_guard(history2, state)
        assert state.total_modify_params == 2

    def test_action_history_updated_in_state(self):
        history = [
            {"action": "modify_params"},
            {"action": "full_rewrite"},
            {"action": "change_entry_logic"},
        ]
        state, _ = check_cosmetic_guard(history, CosmeticGuardState())
        assert state.action_history == ["modify_params", "full_rewrite", "change_entry_logic"]

    def test_action_history_strips_non_action_keys(self):
        history = [
            {"turn": 1, "action": "modify_params", "score": 0.5},
            {"turn": 2, "action": "full_rewrite", "score": 0.8},
        ]
        state, _ = check_cosmetic_guard(history, CosmeticGuardState())
        assert state.action_history == ["modify_params", "full_rewrite"]

    def test_entries_without_action_key_count_as_empty(self):
        history = [
            {"turn": 1},
            {"turn": 2},
            {"turn": 3},
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert state.consecutive_modify_params == 0
        assert result.forced is False

    def test_mixed_empty_action_and_modify_params(self):
        history = [
            {"action": "modify_params"},
            {"turn": 2},  # no action key → treated as ""
            {"action": "modify_params"},
            {"action": "modify_params"},
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert state.consecutive_modify_params == 2
        assert result.forced is False

    def test_consecutive_count_only_from_tail(self):
        history = [
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "full_rewrite"},
            {"action": "modify_params"},
            {"action": "modify_params"},
        ]
        state, _ = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert state.consecutive_modify_params == 2

    def test_all_structural_actions_no_force(self):
        history = [
            {"action": "full_rewrite"},
            {"action": "change_entry_logic"},
            {"action": "change_exit_logic"},
        ]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced is False
        assert state.consecutive_modify_params == 0

    def test_forced_action_is_always_structural(self):
        from crabquant.refinement.cosmetic_guard import _STRUCTURAL_ACTIONS
        history = [{"action": "modify_params"} for _ in range(3)]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        assert result.forced_action in _STRUCTURAL_ACTIONS

    def test_total_modify_params_never_decreases(self):
        state = CosmeticGuardState(total_modify_params=10, action_history=[])
        # History with 1 modify_params — prev_modify_count=0, new=1, total=max(10+1, 1)=11
        history = [{"action": "modify_params"}]
        state, _ = check_cosmetic_guard(history, state)
        # Since state.action_history was empty, prev=0, new=1, total becomes max(10+1, 1)=11
        assert state.total_modify_params == 11

    def test_large_history_performance(self):
        """Ensure large history doesn't blow up."""
        history = [{"action": "modify_params" if i % 5 < 3 else "full_rewrite"} for i in range(1000)]
        state, result = check_cosmetic_guard(history, CosmeticGuardState(threshold=3))
        # Last 3 entries: 996(modify), 997(modify), 998(modify) — but 999 is full_rewrite at i=999, i%5=4
        # i=995 → 995%5=0 < 3 → modify
        # i=996 → 996%5=1 < 3 → modify
        # i=997 → 997%5=2 < 3 → modify
        # i=998 → 998%5=3 ≥ 3 → full_rewrite
        # i=999 → 999%5=4 ≥ 3 → full_rewrite
        assert state.consecutive_modify_params == 0
        assert result.forced is False
