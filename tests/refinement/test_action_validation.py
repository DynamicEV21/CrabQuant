"""Tests for action type validation and mapping in the refinement loop.

Verifies that:
- Valid actions pass through unchanged
- Known aliases are mapped correctly
- Unknown actions default to "novel" with a warning
- Edge cases (None, empty string, whitespace, hyphens, mixed case) are handled
"""

import logging
import pytest

from scripts.refinement_loop import (
    validate_action,
    ACTION_ALIASES,
    DEFAULT_FALLBACK_ACTION,
)
from crabquant.refinement.prompts import VALID_ACTIONS


# ── Valid actions pass through unchanged ────────────────────────────────────

class TestValidActionsPassThrough:
    """Valid actions from VALID_ACTIONS should be returned as-is."""

    @pytest.mark.parametrize("action", VALID_ACTIONS)
    def test_valid_action_returns_same(self, action):
        validated, was_remapped = validate_action(action)
        assert validated == action
        assert was_remapped is False

    def test_returns_tuple(self):
        result = validate_action("novel")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ── Known aliases are mapped correctly ──────────────────────────────────────

class TestAliasMapping:
    """Common invalid LLM actions should be mapped to valid ones."""

    def test_new_strategy_to_novel(self):
        validated, was_remapped = validate_action("new_strategy")
        assert validated == "novel"
        assert was_remapped is True

    def test_create_to_novel(self):
        validated, was_remapped = validate_action("create")
        assert validated == "novel"
        assert was_remapped is True

    def test_propose_strategy_to_novel(self):
        validated, was_remapped = validate_action("propose_strategy")
        assert validated == "novel"
        assert was_remapped is True

    def test_propose_to_novel(self):
        validated, was_remapped = validate_action("propose")
        assert validated == "novel"
        assert was_remapped is True

    def test_iterate_to_modify_params(self):
        validated, was_remapped = validate_action("iterate")
        assert validated == "modify_params"
        assert was_remapped is True

    def test_simplify_to_replace_indicator(self):
        validated, was_remapped = validate_action("simplify")
        assert validated == "replace_indicator"
        assert was_remapped is True

    def test_modify_to_modify_params(self):
        validated, was_remapped = validate_action("modify")
        assert validated == "modify_params"
        assert was_remapped is True

    def test_refine_to_replace_indicator(self):
        validated, was_remapped = validate_action("refine")
        assert validated == "replace_indicator"
        assert was_remapped is True

    def test_refine_params_to_modify_params(self):
        validated, was_remapped = validate_action("refine_params")
        assert validated == "modify_params"
        assert was_remapped is True

    def test_all_aliases_map_to_valid_actions(self):
        """Every alias target must be a valid action."""
        for alias, target in ACTION_ALIASES.items():
            assert target in VALID_ACTIONS, (
                f"Alias '{alias}' maps to '{target}' which is not in VALID_ACTIONS"
            )

    def test_all_aliases_are_lowercase_underscore(self):
        """Aliases should be lowercase with underscores for consistent lookup."""
        for alias in ACTION_ALIASES:
            assert alias == alias.lower(), f"Alias '{alias}' is not lowercase"
            assert "-" not in alias, f"Alias '{alias}' contains hyphen"


# ── Unknown actions default to "novel" ─────────────────────────────────────

class TestUnknownActionsFallback:
    """Completely unknown actions should default to "novel"."""

    def test_unknown_action_defaults_to_novel(self):
        validated, was_remapped = validate_action("completely_unknown_action")
        assert validated == DEFAULT_FALLBACK_ACTION
        assert was_remapped is True

    def test_random_gibberish_defaults_to_novel(self):
        validated, was_remapped = validate_action("asdfjkl123")
        assert validated == DEFAULT_FALLBACK_ACTION
        assert was_remapped is True

    def test_empty_string_defaults_to_novel(self):
        validated, was_remapped = validate_action("")
        assert validated == DEFAULT_FALLBACK_ACTION
        assert was_remapped is True

    def test_numeric_string_defaults_to_novel(self):
        validated, was_remapped = validate_action("42")
        assert validated == DEFAULT_FALLBACK_ACTION
        assert was_remapped is True

    def test_unknown_action_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="scripts.refinement_loop"):
            validate_action("totally_bogus_action_xyz")
        assert any("unknown action" in r.message.lower() for r in caplog.records)


# ── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Whitespace, case variations, and hyphens should be handled."""

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped before lookup."""
        validated, was_remapped = validate_action("  novel  ")
        assert validated == "novel"
        assert was_remapped is False

    def test_whitespace_stripped_for_alias(self):
        validated, was_remapped = validate_action("  create  ")
        assert validated == "novel"
        assert was_remapped is True

    def test_case_insensitive_alias(self):
        """Aliases should be case-insensitive."""
        validated, was_remapped = validate_action("NEW_STRATEGY")
        assert validated == "novel"
        assert was_remapped is True

    def test_mixed_case_alias(self):
        validated, was_remapped = validate_action("New_Strategy")
        assert validated == "novel"
        assert was_remapped is True

    def test_hyphen_converted_to_underscore(self):
        """Hyphens should be normalised to underscores."""
        # "new-strategy" should normalise to "new_strategy" and map to "novel"
        validated, was_remapped = validate_action("new-strategy")
        assert validated == "novel"
        assert was_remapped is True

    def test_whitespace_only_defaults_to_novel(self):
        validated, was_remapped = validate_action("   ")
        assert validated == DEFAULT_FALLBACK_ACTION
        assert was_remapped is True


# ── Integration: validate_action result is always in VALID_ACTIONS ──────────

class TestAlwaysReturnsValidAction:
    """No matter the input, validate_action must always return a valid action."""

    @pytest.mark.parametrize("raw_action", [
        "novel", "new_strategy", "create", "", "iterate", "simplify",
        "modify", "refine", "refine_params", "propose", "propose_strategy",
        "totally_unknown", "  ", "NEW-STRATEGY", "Replace_Indicator",
        "42", "add_filter", "full_rewrite", "change_entry_logic",
    ])
    def test_result_always_in_valid_actions(self, raw_action):
        validated, _ = validate_action(raw_action)
        assert validated in VALID_ACTIONS, (
            f"validate_action('{raw_action}') returned '{validated}' "
            f"which is not in VALID_ACTIONS"
        )
