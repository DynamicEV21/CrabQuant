"""Tests for Phase 5.6.2 — Parallel Strategy Spawning.

Covers:
- get_variant_bias_text() in prompts.py
- get_parallel_prompt_variants() in prompts.py
- compute_composite_score() in prompts.py
- "balanced" mode in config.py
- call_llm_inventor variant bias injection in llm_api.py
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from crabquant.refinement.prompts import (
    PARALLEL_VARIANT_FOCI,
    get_parallel_prompt_variants,
    get_variant_bias_text,
    compute_composite_score,
)
from crabquant.refinement.config import RefinementConfig


# ─── PARALLEL_VARIANT_FOCI ──────────────────────────────────────────────────

class TestParallelVariantFoci:
    """Tests for the PARALLEL_VARIANT_FOCI constant."""

    def test_has_at_least_3_variants(self):
        assert len(PARALLEL_VARIANT_FOCI) >= 3

    def test_each_variant_has_required_keys(self):
        for v in PARALLEL_VARIANT_FOCI:
            assert "name" in v, f"Variant missing 'name': {v}"
            assert "bias" in v, f"Variant missing 'bias': {v}"

    def test_each_bias_is_non_empty_string(self):
        for v in PARALLEL_VARIANT_FOCI:
            assert isinstance(v["bias"], str)
            assert len(v["bias"]) > 20  # Bias should be descriptive

    def test_foci_names_are_unique(self):
        names = [v["name"] for v in PARALLEL_VARIANT_FOCI]
        assert len(names) == len(set(names)), "Duplicate variant names found"


# ─── get_variant_bias_text ─────────────────────────────────────────────────

class TestGetVariantBiasText:
    """Tests for get_variant_bias_text() helper."""

    def test_returns_empty_for_count_1(self):
        result = get_variant_bias_text(0, 1)
        assert result == ""

    def test_returns_empty_for_count_0(self):
        result = get_variant_bias_text(0, 0)
        assert result == ""

    def test_returns_empty_for_negative_index(self):
        result = get_variant_bias_text(-1, 3)
        assert result == ""

    def test_returns_bias_for_valid_index(self):
        result = get_variant_bias_text(0, 3)
        assert "Parallel Variant Focus" in result or "parallel" in result.lower()

    def test_different_indices_return_different_foci(self):
        biases = [get_variant_bias_text(i, 5) for i in range(5)]
        # At least the first two should differ
        assert biases[0] != biases[1]

    def test_wraps_around_when_index_exceeds_count(self):
        """Index > len(foci) should cycle using modulo."""
        n = len(PARALLEL_VARIANT_FOCI)
        bias_0 = get_variant_bias_text(0, n + 1)
        bias_n = get_variant_bias_text(n, n + 1)
        assert bias_0 == bias_n

    def test_all_valid_indices_return_non_empty(self):
        for i in range(len(PARALLEL_VARIANT_FOCI)):
            result = get_variant_bias_text(i, len(PARALLEL_VARIANT_FOCI))
            assert len(result) > 20


# ─── get_parallel_prompt_variants ───────────────────────────────────────────

class TestGetParallelPromptVariants:
    """Tests for get_parallel_prompt_variants() batch function."""

    def test_count_1_returns_single_unmodified(self):
        base = "Generate a strategy"
        result = get_parallel_prompt_variants(base, 1)
        assert result == [base]

    def test_count_3_returns_3_variants(self):
        base = "Generate a strategy"
        result = get_parallel_prompt_variants(base, 3)
        assert len(result) == 3

    def test_each_variant_contains_base_prompt(self):
        base = "Generate a strategy for SPY"
        result = get_parallel_prompt_variants(base, 3)
        for v in result:
            assert "Generate a strategy for SPY" in v

    def test_each_variant_contains_bias_text(self):
        base = "Generate a strategy"
        result = get_parallel_prompt_variants(base, 3)
        for v in result:
            assert "Parallel Variant Focus" in v or "parallel" in v.lower()

    def test_variants_differ_from_each_other(self):
        base = "Generate a strategy"
        result = get_parallel_prompt_variants(base, 3)
        assert len(set(result)) == 3

    def test_count_exceeding_foci_is_clamped(self):
        """Count > available foci should be clamped to foci length."""
        base = "Generate a strategy"
        n_foci = len(PARALLEL_VARIANT_FOCI)
        result = get_parallel_prompt_variants(base, n_foci + 10)
        assert len(result) == n_foci


# ─── compute_composite_score ───────────────────────────────────────────────

class TestComputeCompositeScore:
    """Tests for compute_composite_score()."""

    def test_zero_sharpe_returns_zero(self):
        assert compute_composite_score(0.0, 10, 0.1) == 0.0

    def test_negative_sharpe_returns_zero(self):
        assert compute_composite_score(-1.0, 10, 0.1) == 0.0

    def test_zero_trades_returns_zero(self):
        assert compute_composite_score(2.0, 0, 0.1) == 0.0

    def test_good_strategy_has_positive_score(self):
        score = compute_composite_score(2.0, 50, 0.1)
        assert score > 0

    def test_higher_sharpe_gives_higher_score(self):
        s1 = compute_composite_score(1.0, 30, 0.1)
        s2 = compute_composite_score(2.0, 30, 0.1)
        assert s2 > s1

    def test_more_trades_gives_higher_score(self):
        s1 = compute_composite_score(2.0, 10, 0.1)
        s2 = compute_composite_score(2.0, 50, 0.1)
        assert s2 > s1

    def test_larger_drawdown_gives_lower_score(self):
        s1 = compute_composite_score(2.0, 30, 0.05)
        s2 = compute_composite_score(2.0, 30, 0.3)
        assert s1 > s2

    def test_negative_drawdown_treated_as_positive(self):
        """Max drawdown is typically positive but handle edge case."""
        s1 = compute_composite_score(2.0, 30, 0.1)
        s2 = compute_composite_score(2.0, 30, -0.1)
        # abs(-0.1) = 0.1, so same score
        assert s1 == s2


# ─── Config "balanced" mode ────────────────────────────────────────────────

class TestBalancedMode:
    """Tests for the 'balanced' preset mode in RefinementConfig."""

    def test_balanced_enables_cross_run_learning(self):
        cfg = RefinementConfig()
        cfg.apply_mode("balanced")
        assert cfg.cross_run_learning is True

    def test_balanced_enables_parallel_invention(self):
        cfg = RefinementConfig()
        cfg.apply_mode("balanced")
        assert cfg.parallel_invention is True

    def test_balanced_disables_soft_promote(self):
        cfg = RefinementConfig()
        cfg.apply_mode("balanced")
        assert cfg.soft_promote is False

    def test_balanced_preserves_parallel_count(self):
        """balanced mode should not change parallel_invention_count."""
        cfg = RefinementConfig(parallel_invention_count=5)
        cfg.apply_mode("balanced")
        assert cfg.parallel_invention_count == 5

    def test_balanced_returns_self(self):
        cfg = RefinementConfig()
        result = cfg.apply_mode("balanced")
        assert result is cfg

    def test_explorer_differs_from_balanced(self):
        """Explorer enables soft_promote, balanced does not."""
        cfg_b = RefinementConfig()
        cfg_b.apply_mode("balanced")
        cfg_e = RefinementConfig()
        cfg_e.apply_mode("explorer")
        assert cfg_b.soft_promote != cfg_e.soft_promote

    def test_conservative_disables_parallel(self):
        cfg = RefinementConfig()
        cfg.apply_mode("conservative")
        assert cfg.parallel_invention is False
        assert cfg.cross_run_learning is False

    def test_fast_enables_cross_run_only(self):
        cfg = RefinementConfig()
        cfg.apply_mode("fast")
        assert cfg.cross_run_learning is True
        assert cfg.parallel_invention is False
        assert cfg.soft_promote is False

    def test_custom_leaves_toggles_unchanged(self):
        cfg = RefinementConfig(
            cross_run_learning=True,
            parallel_invention=True,
            soft_promote=True,
        )
        cfg.apply_mode("custom")
        assert cfg.cross_run_learning is True
        assert cfg.parallel_invention is True
        assert cfg.soft_promote is True

    def test_mode_is_case_insensitive(self):
        cfg = RefinementConfig()
        cfg.apply_mode("BALANCED")
        assert cfg.parallel_invention is True

    def test_mode_strips_whitespace(self):
        cfg = RefinementConfig()
        cfg.apply_mode("  balanced  ")
        assert cfg.parallel_invention is True


# ─── Config parallel defaults ──────────────────────────────────────────────

class TestParallelDefaults:
    """Tests for parallel invention defaults in RefinementConfig."""

    def test_parallel_invention_default_false(self):
        cfg = RefinementConfig()
        assert cfg.parallel_invention is False

    def test_parallel_invention_count_default_3(self):
        cfg = RefinementConfig()
        assert cfg.parallel_invention_count == 3

    def test_parallel_invention_count_is_int(self):
        cfg = RefinementConfig()
        assert isinstance(cfg.parallel_invention_count, int)

    def test_parallel_invention_is_bool(self):
        cfg = RefinementConfig()
        assert isinstance(cfg.parallel_invention, bool)


# ─── call_llm_inventor variant bias injection ──────────────────────────────

class TestLLMVariantBiasInjection:
    """Tests that call_llm_inventor injects variant bias into the user prompt."""

    def _make_context(self, **overrides):
        base = {
            "mandate": {"name": "test", "tickers": ["SPY"]},
        }
        base.update(overrides)
        return base

    @staticmethod
    def _get_user_content(mock_llm):
        """Extract user message content from the mock call."""
        messages = mock_llm.call_args.kwargs["messages"]
        return messages[1]["content"]

    @patch("crabquant.refinement.llm_api.call_zai_llm")
    def test_no_variant_keys_no_bias(self, mock_llm):
        """Without parallel variant keys, no bias is injected."""
        mock_llm.return_value = '{"action": "novel", "new_strategy_code": "pass=1"}'
        from crabquant.refinement.llm_api import call_llm_inventor
        context = self._make_context()
        call_llm_inventor(context=context)
        user_content = self._get_user_content(mock_llm)
        assert "Parallel Variant" not in user_content

    @patch("crabquant.refinement.llm_api.call_zai_llm")
    def test_variant_keys_inject_bias(self, mock_llm):
        """With parallel variant keys, bias text is appended to user message."""
        mock_llm.return_value = '{"action": "novel", "new_strategy_code": "pass=1"}'
        from crabquant.refinement.llm_api import call_llm_inventor
        context = self._make_context(
            parallel_variant_index=0,
            parallel_variant_count=3,
        )
        call_llm_inventor(context=context)
        user_content = self._get_user_content(mock_llm)
        # Should contain variant bias text
        assert "Variant" in user_content or "variant" in user_content

    @patch("crabquant.refinement.llm_api.call_zai_llm")
    def test_variant_count_1_no_bias(self, mock_llm):
        """variant_count=1 should not inject bias (sequential mode)."""
        mock_llm.return_value = '{"action": "novel", "new_strategy_code": "pass=1"}'
        from crabquant.refinement.llm_api import call_llm_inventor
        context = self._make_context(
            parallel_variant_index=0,
            parallel_variant_count=1,
        )
        call_llm_inventor(context=context)
        user_content = self._get_user_content(mock_llm)
        # Count=1 means no parallel, so no bias
        assert "Parallel Variant Focus" not in user_content

    @patch("crabquant.refinement.llm_api.call_zai_llm")
    def test_different_variants_get_different_bias(self, mock_llm):
        """Variant 0 and variant 1 should get different bias texts."""
        mock_llm.return_value = '{"action": "novel", "new_strategy_code": "pass=1"}'
        from crabquant.refinement.llm_api import call_llm_inventor

        context_0 = self._make_context(parallel_variant_index=0, parallel_variant_count=3)
        context_1 = self._make_context(parallel_variant_index=1, parallel_variant_count=3)

        call_llm_inventor(context=context_0)
        content_0 = self._get_user_content(mock_llm)

        call_llm_inventor(context=context_1)
        content_1 = self._get_user_content(mock_llm)

        # The bias portions should differ (last part of the prompt)
        assert content_0 != content_1

    @patch("crabquant.refinement.llm_api.call_zai_llm")
    def test_bias_appended_after_mandate(self, mock_llm):
        """Bias should be appended at the end of the user message."""
        mock_llm.return_value = '{"action": "novel", "new_strategy_code": "pass=1"}'
        from crabquant.refinement.llm_api import call_llm_inventor
        context = self._make_context(
            parallel_variant_index=2,
            parallel_variant_count=3,
        )
        call_llm_inventor(context=context)
        user_content = self._get_user_content(mock_llm)
        # Mandate should appear before the variant bias
        mandate_pos = user_content.find("Mandate")
        variant_pos = user_content.find("Variant", mandate_pos if mandate_pos >= 0 else 0)
        if mandate_pos >= 0 and variant_pos >= 0:
            assert variant_pos > mandate_pos
