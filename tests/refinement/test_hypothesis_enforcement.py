"""Tests for crabquant.refinement.hypothesis_enforcement — validate LLM output has non-trivial hypothesis."""

import pytest

from crabquant.refinement.hypothesis_enforcement import (
    HypothesisCheckResult,
    check_hypothesis,
    check_hypothesis_from_modification,
    GENERIC_PATTERNS,
)


class TestGenericPatterns:
    """Verify the built-in generic pattern list catches common lazy hypotheses."""

    def test_patterns_is_nonempty(self):
        assert len(GENERIC_PATTERNS) > 0

    def test_catches_improve_performance(self):
        assert any("improve performance" in p.lower() for p in GENERIC_PATTERNS)

    def test_catches_adjust_parameters(self):
        assert any("adjust parameter" in p.lower() for p in GENERIC_PATTERNS)

    def test_catches_generic_optimize(self):
        assert any("optim" in p.lower() for p in GENERIC_PATTERNS)


class TestCheckHypothesis:
    """Main hypothesis enforcement logic."""

    def test_substantive_hypothesis_passes(self):
        result = check_hypothesis(
            "Volume threshold too strict — loosening to 1.0x avg will generate more entry signals"
        )
        assert result.valid is True
        assert result.reason == ""

    def test_specific_hypothesis_with_numbers_passes(self):
        result = check_hypothesis(
            "Adding a 20-period EMA filter will reduce whipsaws in ranging markets by filtering out low-trend signals"
        )
        assert result.valid is True

    def test_empty_hypothesis_fails(self):
        result = check_hypothesis("")
        assert result.valid is False
        assert any("empty" in r.lower() or "missing" in r.lower() for r in [result.reason])

    def test_none_hypothesis_fails(self):
        result = check_hypothesis(None)
        assert result.valid is False

    def test_whitespace_only_fails(self):
        result = check_hypothesis("   ")
        assert result.valid is False

    def test_improve_performance_generic_fails(self):
        result = check_hypothesis("improve performance by adjusting parameters")
        assert result.valid is False
        assert "generic" in result.reason.lower()

    def test_adjust_parameters_generic_fails(self):
        result = check_hypothesis("Adjust parameters to improve strategy")
        assert result.valid is False

    def test_optimize_strategy_generic_fails(self):
        result = check_hypothesis("Optimize the strategy for better returns")
        assert result.valid is False

    def test_too_short_hypothesis_fails(self):
        result = check_hypothesis("Fix it")
        assert result.valid is False
        assert any("short" in r.lower() or "too short" in r.lower() for r in [result.reason])

    def test_min_length_boundary(self):
        """Exactly 20 chars should pass length check (if not generic)."""
        result = check_hypothesis("Add RSI divergence filter to catch reversal signals early")
        assert result.valid is True

    def test_case_insensitive_generic_check(self):
        result = check_hypothesis("IMPROVE PERFORMANCE")
        assert result.valid is False

    def test_custom_generic_patterns(self):
        extra = ["tweak settings"]
        result = check_hypothesis("tweak settings", extra_generic_patterns=extra)
        assert result.valid is False

    def test_returns_hypothesis_check_result(self):
        result = check_hypothesis("A good causal hypothesis about why changing the exit to ATR-based trailing stops reduces premature exits in volatile conditions")
        assert isinstance(result, HypothesisCheckResult)
        assert isinstance(result.valid, bool)
        assert isinstance(result.reason, str)

    def test_missing_field_in_modification_dict(self):
        result = check_hypothesis_from_modification({"action": "modify_params"})
        assert result.valid is False

    def test_valid_modification_dict(self):
        mod = {
            "hypothesis": "RSI period too long at 14 — reducing to 7 will catch momentum shifts earlier",
            "action": "modify_params",
            "addresses_failure": "too_few_trades",
        }
        result = check_hypothesis_from_modification(mod)
        assert result.valid is True

    def test_generic_modification_dict(self):
        mod = {
            "hypothesis": "improve performance",
            "action": "modify_params",
            "addresses_failure": "too_few_trades",
        }
        result = check_hypothesis_from_modification(mod)
        assert result.valid is False
