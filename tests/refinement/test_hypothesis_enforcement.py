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


class TestGenericPatternsExpanded:
    """Verify each individual generic pattern is caught."""

    @pytest.mark.parametrize("pattern", [
        "improve the strategy",
        "adjust parameter",
        "optimize the strategy",
        "optimize strategy",
        "make it better",
        "increase returns",
        "reduce losses",
        "fine-tune",
        "improve results",
        "get better results",
        "improve sharpe",
        "improve profitability",
    ])
    def test_each_generic_pattern_rejected(self, pattern):
        """Each built-in generic pattern should be rejected even if padded to >20 chars."""
        hypothesis = f"{pattern} by doing some things with the strategy parameters"
        result = check_hypothesis(hypothesis)
        assert result.valid is False
        assert "generic" in result.reason.lower()


class TestCheckHypothesisExpanded:
    """Additional edge cases for check_hypothesis."""

    def test_none_modification_dict_fails(self):
        result = check_hypothesis_from_modification(None)
        assert result.valid is False

    def test_empty_modification_dict_fails(self):
        result = check_hypothesis_from_modification({})
        assert result.valid is False

    def test_modification_dict_forwards_kwargs(self):
        mod = {"hypothesis": "improve performance by adjusting the entry parameters"}
        result = check_hypothesis_from_modification(mod, min_length=100)
        assert result.valid is False

    def test_custom_min_length_shorter_than_default(self):
        result = check_hypothesis("A valid hypothesis", min_length=10)
        assert result.valid is True

    def test_custom_min_length_longer_than_default(self):
        result = check_hypothesis("This is a specific but somewhat short hypothesis", min_length=100)
        assert result.valid is False
        assert "short" in result.reason.lower()

    def test_exactly_at_min_length_boundary(self):
        """Exactly 20 chars should pass length check (if not generic)."""
        # 20 chars exactly
        hypothesis = "12345678901234567890"
        result = check_hypothesis(hypothesis, min_length=20)
        assert result.valid is True

    def test_one_char_below_min_length(self):
        hypothesis = "1234567890123456789"  # 19 chars
        result = check_hypothesis(hypothesis, min_length=20)
        assert result.valid is False
        assert "19" in result.reason

    def test_whitespace_trimmed_before_length_check(self):
        result = check_hypothesis("   " + "A" * 18 + "   ", min_length=20)
        # "   AAAA...AAAA   " strip → 18 chars
        assert result.valid is False

    def test_tabs_and_newlines_stripped(self):
        result = check_hypothesis("\t\n")
        assert result.valid is False

    def test_newline_in_hypothesis_still_valid(self):
        result = check_hypothesis(
            "Volume spikes precede breakouts.\nLowering the volume threshold captures these moves."
        )
        assert result.valid is True

    def test_generic_pattern_with_leading_whitespace(self):
        result = check_hypothesis("   improve performance by doing more things")
        assert result.valid is False

    def test_generic_embedded_in_longer_sentence(self):
        result = check_hypothesis(
            "The strategy should improve performance by adjusting various parameters and settings"
        )
        assert result.valid is False

    def test_extra_generic_patterns_empty_list(self):
        extra = []
        result = check_hypothesis("improve performance", extra_generic_patterns=extra)
        # Should still fail from built-in patterns
        assert result.valid is False

    def test_extra_generic_patterns_with_regex_special_chars(self):
        extra = ["tweak (settings)"]
        result = check_hypothesis("We need to tweak (settings) to get better outcomes here", extra_generic_patterns=extra)
        assert result.valid is False

    def test_extra_patterns_checked_before_builtin(self):
        # A hypothesis that is NOT in built-in patterns but IS in extra
        extra = ["do something vague"]
        result = check_hypothesis("We should do something vague with the parameters", extra_generic_patterns=extra)
        assert result.valid is False
        assert "generic" in result.reason.lower()

    def test_extra_patterns_dont_affect_valid_hypothesis(self):
        extra = ["completely unrelated pattern xyz123"]
        result = check_hypothesis(
            "Adding a 20-period EMA filter will reduce whipsaws in ranging markets by filtering out low-trend signals",
            extra_generic_patterns=extra,
        )
        assert result.valid is True

    def test_unicode_hypothesis_valid(self):
        result = check_hypothesis(
            "The strategy fails in high-volatility regimes — adding a volatility filter using ATR × 1.5 will prevent entry during turbulent periods"
        )
        assert result.valid is True

    def test_unicode_in_generic_pattern(self):
        extra = ["tweak™ settings"]
        result = check_hypothesis("We should tweak™ settings to optimize", extra_generic_patterns=extra)
        assert result.valid is False

    def test_result_dataclass_fields(self):
        result = check_hypothesis("Valid causal hypothesis about changing exit logic to use ATR trailing stops")
        assert hasattr(result, "valid")
        assert hasattr(result, "reason")
        assert isinstance(result.valid, bool)
        assert isinstance(result.reason, str)

    def test_valid_hypothesis_has_empty_reason(self):
        result = check_hypothesis(
            "Volume threshold too strict — loosening to 1.0x avg will generate more entry signals"
        )
        assert result.reason == ""

    def test_invalid_hypothesis_has_nonempty_reason(self):
        result = check_hypothesis("")
        assert len(result.reason) > 0

    def test_multiple_generic_patterns_in_one_hypothesis(self):
        result = check_hypothesis("improve performance and increase returns simultaneously")
        assert result.valid is False

    def test_case_insensitive_upper(self):
        result = check_hypothesis("ADJUST PARAMETERS TO IMPROVE RESULTS")
        assert result.valid is False

    def test_case_insensitive_mixed(self):
        result = check_hypothesis("ImPrOvE pErFoRmAnCe")
        assert result.valid is False

    def test_hypothesis_with_numbers_still_checked(self):
        result = check_hypothesis("improve performance by 50 percent")
        assert result.valid is False

    def test_very_long_valid_hypothesis(self):
        long_hyp = "The current RSI period of 14 is too slow for this timeframe. "
        long_hyp += "Reducing to 7 will capture momentum shifts earlier, "
        long_hyp += "generating signals 2-3 bars before significant price moves. "
        long_hyp += "This directly addresses the lag issue identified in backtest analysis."
        result = check_hypothesis(long_hyp)
        assert result.valid is True

    def test_min_length_zero(self):
        result = check_hypothesis("Fix it", min_length=0)
        assert result.valid is True

    def test_min_length_negative(self):
        result = check_hypothesis("Fix it", min_length=-5)
        assert result.valid is True

    def test_check_hypothesis_from_modification_with_extra_patterns(self):
        mod = {
            "hypothesis": "just tweak things a bit",
            "action": "modify_params",
        }
        result = check_hypothesis_from_modification(
            mod, extra_generic_patterns=["tweak things"]
        )
        assert result.valid is False
