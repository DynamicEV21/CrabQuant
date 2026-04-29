"""Comprehensive tests for crabquant.refinement.prompts — prompt generation module.

Covers all functions and constants NOT already tested in test_prompt_refinement.py:
- PARALLEL_VARIANT_FOCI structure
- get_parallel_prompt_variants()
- get_variant_bias_text()
- compute_composite_score()
- _format_window_breakdown()
- build_failure_guidance() (all failure modes + edge cases)
- format_previous_attempts_section() (failure-mode-specific notes)
- build_turn1_prompt() (winner_examples, archetype_section, minimal defaults)
- build_refinement_prompt() (winner_examples, validation_failed, edge cases)
- format_stagnation_suffix() (empty string edge case)
- format_tier2_section() (missing keys)
"""

import pytest

from crabquant.refinement.prompts import (
    PARALLEL_VARIANT_FOCI,
    VALID_ACTIONS,
    compute_composite_score,
    get_parallel_prompt_variants,
    get_variant_bias_text,
    build_failure_guidance,
    build_turn1_prompt,
    build_refinement_prompt,
    format_previous_attempts_section,
    format_stagnation_suffix,
    format_tier2_section,
    _format_window_breakdown,
)


# ─── PARALLEL_VARIANT_FOCI constant ──────────────────────────────────────────


class TestParallelVariantFoci:
    """Verify the PARALLEL_VARIANT_FOCI constant structure."""

    def test_has_five_variants(self):
        assert len(PARALLEL_VARIANT_FOCI) == 5

    def test_each_variant_has_required_keys(self):
        for variant in PARALLEL_VARIANT_FOCI:
            assert "name" in variant, f"Missing 'name' key in variant: {variant}"
            assert "bias" in variant, f"Missing 'bias' key in variant: {variant}"

    def test_expected_variant_names(self):
        names = {v["name"] for v in PARALLEL_VARIANT_FOCI}
        assert "momentum" in names
        assert "mean_reversion" in names
        assert "volatility_breakout" in names
        assert "volume_confirmation" in names
        assert "multi_signal" in names

    def test_bias_text_contains_heading(self):
        for variant in PARALLEL_VARIANT_FOCI:
            assert "Parallel Variant Focus" in variant["bias"], (
                f"Bias for '{variant['name']}' missing heading"
            )

    def test_momentum_bias_prefers_correct_indicators(self):
        momentum = PARALLEL_VARIANT_FOCI[0]
        assert momentum["name"] == "momentum"
        assert "MACD" in momentum["bias"]
        assert "ROC" in momentum["bias"]


# ─── get_parallel_prompt_variants ────────────────────────────────────────────


class TestGetParallelPromptVariants:
    def test_count_one_returns_list_with_base(self):
        base = "Invent a strategy."
        result = get_parallel_prompt_variants(base, count=1)
        assert result == [base]

    def test_count_zero_returns_list_with_base(self):
        base = "Invent a strategy."
        result = get_parallel_prompt_variants(base, count=0)
        assert result == [base]

    def test_count_negative_returns_list_with_base(self):
        base = "Invent a strategy."
        result = get_parallel_prompt_variants(base, count=-5)
        assert result == [base]

    def test_count_two_returns_two_variants(self):
        base = "Invent a strategy."
        result = get_parallel_prompt_variants(base, count=2)
        assert len(result) == 2
        # First variant should be momentum bias
        assert "MOMENTUM" in result[0]
        # Second should be mean_reversion
        assert "MEAN REVERSION" in result[1]

    def test_count_five_returns_all_five(self):
        base = "Invent a strategy."
        result = get_parallel_prompt_variants(base, count=5)
        assert len(result) == 5
        for variant, focus in zip(result, PARALLEL_VARIANT_FOCI):
            assert focus["bias"].strip() in variant

    def test_count_exceeds_available_clamps(self):
        """If count > 5 (available foci), clamp to 5."""
        base = "Invent a strategy."
        result = get_parallel_prompt_variants(base, count=10)
        assert len(result) == 5

    def test_base_prompt_is_preserved_in_variant(self):
        base = "Invent a NEW strategy."
        result = get_parallel_prompt_variants(base, count=3)
        for variant in result:
            assert "Invent a NEW strategy." in variant

    def test_variant_appends_bias_with_newlines(self):
        base = "Base prompt."
        result = get_parallel_prompt_variants(base, count=2)
        # Should end with the bias text
        assert result[0].strip().endswith("Avoid: rsi, stoch, bbands.")


# ─── get_variant_bias_text ───────────────────────────────────────────────────


class TestGetVariantBiasText:
    def test_single_variant_returns_empty(self):
        assert get_variant_bias_text(0, 1) == ""

    def test_negative_index_returns_empty(self):
        assert get_variant_bias_text(-1, 3) == ""

    def test_zero_count_returns_empty(self):
        assert get_variant_bias_text(0, 0) == ""

    def test_first_variant_is_momentum(self):
        bias = get_variant_bias_text(0, 2)
        assert "MOMENTUM" in bias

    def test_second_variant_is_mean_reversion(self):
        bias = get_variant_bias_text(1, 3)
        assert "MEAN REVERSION" in bias

    def test_index_wraps_around(self):
        """Index beyond available should cycle via modulo."""
        bias_5 = get_variant_bias_text(5, 6)
        bias_0 = get_variant_bias_text(0, 2)
        # index 5 % 5 = 0 → momentum
        assert "MOMENTUM" in bias_5


# ─── compute_composite_score ─────────────────────────────────────────────────


class TestComputeCompositeScore:
    def test_basic_calculation(self):
        """sharpe=1.0, trades=20, max_drawdown=0.0 → 1.0 * 1.0 * 1.0 = 1.0"""
        result = compute_composite_score(1.0, 20, 0.0)
        assert result == pytest.approx(1.0)

    def test_trade_factor_sqrt(self):
        """trades=80 → sqrt(80/20) = sqrt(4) = 2.0, so score = sharpe * 2.0 * dd_penalty"""
        result = compute_composite_score(1.0, 80, 0.0)
        assert result == pytest.approx(2.0)

    def test_drawdown_penalty(self):
        """drawdown=0.5 → penalty = 1 - 0.5 = 0.5"""
        result = compute_composite_score(1.0, 20, 0.5)
        assert result == pytest.approx(0.5)

    def test_negative_sharpe_returns_zero(self):
        assert compute_composite_score(-0.5, 20, 0.0) == 0.0

    def test_zero_sharpe_returns_zero(self):
        assert compute_composite_score(0.0, 20, 0.0) == 0.0

    def test_zero_trades_returns_zero(self):
        assert compute_composite_score(1.0, 0, 0.0) == 0.0

    def test_negative_trades_returns_zero(self):
        assert compute_composite_score(1.0, -5, 0.0) == 0.0

    def test_large_drawdown_clamped_to_one(self):
        """drawdown > 1.0 should be clamped: penalty = 1 - 1.0 = 0.0"""
        result = compute_composite_score(2.0, 50, 5.0)
        assert result == pytest.approx(0.0)

    def test_negative_drawdown_uses_abs(self):
        """Negative drawdown should use abs, same as positive."""
        pos = compute_composite_score(1.0, 20, 0.3)
        neg = compute_composite_score(1.0, 20, -0.3)
        assert pos == pytest.approx(neg)

    def test_single_trade(self):
        """trades=1 → sqrt(1/20) ≈ 0.2236"""
        result = compute_composite_score(1.0, 1, 0.0)
        expected = (1 / 20) ** 0.5
        assert result == pytest.approx(expected)

    def test_very_few_trades_heavily_penalized(self):
        """2 trades → sqrt(2/20) = sqrt(0.1) ≈ 0.316"""
        result = compute_composite_score(3.0, 2, 0.1)
        trade_factor = (2 / 20) ** 0.5
        dd_penalty = 1.0 - 0.1
        expected = 3.0 * trade_factor * dd_penalty
        assert result == pytest.approx(expected, rel=1e-6)


# ─── _format_window_breakdown ────────────────────────────────────────────────


class TestFormatWindowBreakdown:
    def test_empty_validation(self):
        assert _format_window_breakdown({}) == ""

    def test_empty_window_results(self):
        assert _format_window_breakdown({"window_results": []}) == ""

    def test_single_passed_window(self):
        validation = {
            "window_results": [{
                "window": 1,
                "train_sharpe": 2.0,
                "test_sharpe": 1.5,
                "degradation": 0.25,
                "passed": True,
            }]
        }
        result = _format_window_breakdown(validation)
        assert "PASS" in result
        assert "1/1 windows passed" in result

    def test_single_failed_window(self):
        validation = {
            "window_results": [{
                "window": 1,
                "train_sharpe": 2.0,
                "test_sharpe": 0.5,
                "degradation": 0.75,
                "passed": False,
            }]
        }
        result = _format_window_breakdown(validation)
        assert "FAIL" in result
        assert "ALL windows failed" in result

    def test_mixed_results(self):
        validation = {
            "window_results": [
                {"window": 1, "train_sharpe": 2.0, "test_sharpe": 1.5, "degradation": 0.25, "passed": True},
                {"window": 2, "train_sharpe": 2.0, "test_sharpe": 0.3, "degradation": 0.85, "passed": False},
                {"window": 3, "train_sharpe": 2.0, "test_sharpe": 0.8, "degradation": 0.60, "passed": False},
            ]
        }
        result = _format_window_breakdown(validation)
        assert "PASS" in result
        assert "FAIL" in result
        assert "1/3 windows passed" in result

    def test_all_pass(self):
        validation = {
            "window_results": [
                {"window": i, "train_sharpe": 2.0, "test_sharpe": 1.5, "degradation": 0.25, "passed": True}
                for i in range(1, 4)
            ]
        }
        result = _format_window_breakdown(validation)
        assert "3/3 windows passed" in result

    def test_window_with_error(self):
        validation = {
            "window_results": [{
                "window": 1,
                "train_sharpe": 0,
                "test_sharpe": 0,
                "degradation": 0,
                "passed": False,
                "error": "RuntimeError: division by zero",
            }]
        }
        result = _format_window_breakdown(validation)
        assert "ERROR" in result
        assert "division by zero" in result

    def test_table_headers_present(self):
        validation = {
            "window_results": [{"window": 1, "train_sharpe": 1.0, "test_sharpe": 0.5, "degradation": 0.5, "passed": False}]
        }
        result = _format_window_breakdown(validation)
        assert "Window" in result
        assert "Train Sharpe" in result
        assert "Test Sharpe" in result
        assert "Degradation" in result

    def test_close_but_not_enough_passed(self):
        """2/3 windows passed → 'Close but not robust enough' warning."""
        validation = {
            "window_results": [
                {"window": 1, "train_sharpe": 2.0, "test_sharpe": 1.5, "degradation": 0.25, "passed": True},
                {"window": 2, "train_sharpe": 2.0, "test_sharpe": 1.3, "degradation": 0.35, "passed": True},
                {"window": 3, "train_sharpe": 2.0, "test_sharpe": 0.5, "degradation": 0.75, "passed": False},
            ]
        }
        result = _format_window_breakdown(validation)
        assert "2/3 windows passed" in result
        assert "Close but not robust enough" in result


# ─── build_failure_guidance ──────────────────────────────────────────────────


class TestBuildFailureGuidance:
    def test_too_few_trades_for_validation(self):
        result = build_failure_guidance("too_few_trades_for_validation", total_trades=3)
        assert "Increase Trade Frequency" in result
        assert "3 trades" in result
        assert "REMOVE conditions" in result

    def test_validation_failed_without_validation_data(self):
        result = build_failure_guidance("validation_failed", total_trades=30)
        assert "Fix Out-of-Sample Failure" in result
        assert "REDUCE complexity" in result

    def test_validation_failed_with_window_breakdown(self):
        validation = {
            "window_results": [
                {"window": 1, "train_sharpe": 2.0, "test_sharpe": 0.5, "degradation": 0.75, "passed": False}
            ]
        }
        result = build_failure_guidance("validation_failed", total_trades=30, validation=validation)
        assert "Fix Out-of-Sample Failure" in result
        assert "Rolling Walk-Forward" in result

    def test_regime_fragility(self):
        result = build_failure_guidance("regime_fragility", total_trades=40)
        assert "Regime-Dependent" in result
        assert "regime detection" in result

    def test_low_sharpe_many_trades(self):
        """low_sharpe with >= 15 trades → no curve-fitting warning."""
        result = build_failure_guidance("low_sharpe", total_trades=50)
        assert "Improve Sharpe Ratio" in result
        assert "CRITICAL" not in result
        assert "Very few trades" not in result

    def test_low_sharpe_very_few_trades(self):
        """low_sharpe with < 10 trades → CRITICAL curve-fit warning."""
        result = build_failure_guidance("low_sharpe", total_trades=5)
        assert "CRITICAL" in result
        assert "curve-fit" in result
        assert "5 trades" in result

    def test_low_sharpe_between_10_and_15(self):
        """low_sharpe with 12 trades → 'Very few trades' warning."""
        result = build_failure_guidance("low_sharpe", total_trades=12)
        assert "Very few trades" in result
        assert "12" in result
        assert "CRITICAL" not in result

    def test_unknown_failure_mode_returns_empty(self):
        result = build_failure_guidance("nonexistent_mode", total_trades=30)
        assert result == ""

    def test_empty_failure_mode_returns_empty(self):
        result = build_failure_guidance("", total_trades=30)
        assert result == ""

    def test_default_trades_zero(self):
        """total_trades defaults to 0."""
        result = build_failure_guidance("too_few_trades_for_validation")
        assert "0 trades" in result

    def test_excessive_drawdown(self):
        """excessive_drawdown has actionable guidance for risk management."""
        result = build_failure_guidance("excessive_drawdown", total_trades=30)
        assert "Excessive Drawdown" in result
        assert "STOP LOSS" in result
        assert "TREND FILTER" in result
        assert "VOLATILITY FILTER" in result
        assert "max drawdown < 25%" in result

    def test_flat_signal(self):
        """flat_signal has actionable guidance for getting signals to fire."""
        result = build_failure_guidance("flat_signal", total_trades=0)
        assert "Flat Signal" in result
        assert "SIMPLER signal" in result
        assert "NaN handling" in result
        assert "fillna" in result
        assert "at least 10 trades" in result

    def test_overtrading(self):
        """overtrading has actionable guidance for reducing trade frequency."""
        result = build_failure_guidance("overtrading", total_trades=500)
        assert "Overtrading" in result
        assert "COOLDOWN" in result
        assert "LONGER indicator periods" in result
        assert "CONFIRMATION" in result
        assert "20-100 trades" in result


# ─── format_previous_attempts_section (failure-mode-specific notes) ──────────


class TestFormatPreviousAttemptsSectionNotes:
    """Test the inline notes for specific failure modes."""

    def test_too_few_trades_note(self):
        attempts = [{
            "turn": 1, "sharpe": 2.0, "failure_mode": "too_few_trades_for_validation",
            "action": "add_filter", "hypothesis": "h1", "params_used": {},
            "delta_from_prev": "Initial", "num_trades": 5,
        }]
        result = format_previous_attempts_section(attempts)
        assert "OPEN UP conditions" in result
        assert "5 trades" in result

    def test_validation_failed_note_without_validation_data(self):
        attempts = [{
            "turn": 2, "sharpe": 1.5, "failure_mode": "validation_failed",
            "action": "modify_params", "hypothesis": "h2", "params_used": {},
            "delta_from_prev": "Changed p",
        }]
        result = format_previous_attempts_section(attempts)
        assert "FAILED out-of-sample" in result
        assert "REDUCE complexity" in result

    def test_validation_failed_note_with_validation_data(self):
        attempts = [{
            "turn": 2, "sharpe": 1.5, "failure_mode": "validation_failed",
            "action": "modify_params", "hypothesis": "h2", "params_used": {},
            "delta_from_prev": "Changed p",
            "validation": {
                "avg_test_sharpe": 0.3,
                "windows_passed": 1,
                "num_windows": 5,
                "window_results": [
                    {"window": 1, "train_sharpe": 2.0, "test_sharpe": 0.5, "degradation": 0.75, "passed": False},
                ],
            },
        }]
        result = format_previous_attempts_section(attempts)
        assert "avg test Sharpe=0.3" in result
        assert "1/5 windows passed" in result
        assert "Rolling Walk-Forward" in result

    def test_low_sharpe_curve_fit_warning(self):
        attempts = [{
            "turn": 1, "sharpe": 0.2, "failure_mode": "low_sharpe",
            "action": "novel", "hypothesis": "h1", "params_used": {},
            "delta_from_prev": "Initial", "num_trades": 7,
        }]
        result = format_previous_attempts_section(attempts)
        assert "CURVE-FITTING RISK" in result
        assert "7 trades" in result

    def test_low_sharpe_without_curve_fit_warning(self):
        """low_sharpe with >= 10 trades → no curve-fitting note."""
        attempts = [{
            "turn": 1, "sharpe": 0.2, "failure_mode": "low_sharpe",
            "action": "novel", "hypothesis": "h1", "params_used": {},
            "delta_from_prev": "Initial", "num_trades": 25,
        }]
        result = format_previous_attempts_section(attempts)
        assert "CURVE-FITTING RISK" not in result

    def test_regime_fragility_note(self):
        attempts = [{
            "turn": 3, "sharpe": 0.8, "failure_mode": "regime_fragility",
            "action": "change_entry_logic", "hypothesis": "h3", "params_used": {},
            "delta_from_prev": "Changed entry",
        }]
        result = format_previous_attempts_section(attempts)
        assert "REGIME WARNING" in result
        assert "regime detection" in result

    def test_unknown_failure_mode_no_note(self):
        attempts = [{
            "turn": 1, "sharpe": 0.5, "failure_mode": "unknown_mode",
            "action": "novel", "hypothesis": "h1", "params_used": {},
            "delta_from_prev": "Initial",
        }]
        result = format_previous_attempts_section(attempts)
        assert "NOTE" not in result
        assert "WARNING" not in result

    def test_missing_optional_keys_use_defaults(self):
        """Entry missing many keys should not crash."""
        attempts = [{"turn": 1}]  # minimal
        result = format_previous_attempts_section(attempts)
        assert "Turn 1" in result


# ─── build_turn1_prompt (additional coverage) ────────────────────────────────


class TestBuildTurn1PromptAdditional:
    def test_with_winner_examples(self):
        mandate = {"name": "test", "tickers": ["AAPL"], "period": "1y"}
        winners = [{
            "name": "golden_cross",
            "sharpe": 2.1,
            "trades": 45,
            "ticker": "SPY",
            "source_code": "def generate_signals(df, p): pass",
        }]
        prompt = build_turn1_prompt(
            mandate=mandate, current_turn=1, max_turns=7,
            winner_examples=winners,
        )
        assert "Proven Strategies" in prompt
        assert "golden_cross" in prompt
        assert "2.10" in prompt
        assert "45 trades" in prompt
        assert "(SPY)" in prompt

    def test_winner_without_ticker(self):
        mandate = {"name": "test", "tickers": ["AAPL"], "period": "1y"}
        winners = [{
            "name": "simple_rsi",
            "sharpe": 1.8,
            "trades": 30,
            "source_code": "def generate_signals(df, p): pass",
        }]
        prompt = build_turn1_prompt(
            mandate=mandate, current_turn=1, max_turns=7,
            winner_examples=winners,
        )
        assert "simple_rsi" in prompt
        assert "(SPY)" not in prompt  # no ticker

    def test_explicit_archetype_section(self):
        mandate = {"name": "test", "tickers": ["AAPL"], "period": "1y"}
        prompt = build_turn1_prompt(
            mandate=mandate, current_turn=1, max_turns=7,
            archetype_section="## Custom Archetype\nUse Bollinger Bands.",
        )
        assert "Custom Archetype" in prompt
        assert "Bollinger Bands" in prompt

    def test_minimal_mandate_uses_defaults(self):
        """Mandate with minimal keys should use sensible defaults."""
        mandate = {"name": "test"}
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=7)
        assert "unnamed" not in prompt  # name is provided
        assert "test" in prompt
        # Defaults from mandate.get() fallbacks
        assert "1.5" in prompt  # default sharpe_target
        assert "2y" in prompt  # default period

    def test_no_optional_args(self):
        """Only required args: mandate, current_turn, max_turns."""
        mandate = {"name": "minimal", "tickers": ["MSFT"], "period": "1y"}
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=5)
        assert "minimal" in prompt
        assert "MSFT" in prompt
        assert "new_strategy" in prompt
        assert "novel" in prompt

    def test_seed_code_without_seed_name(self):
        """If seed_code is provided but not seed_strategy_name, no seed section."""
        mandate = {"name": "test", "tickers": ["AAPL"], "period": "1y"}
        prompt = build_turn1_prompt(
            mandate=mandate, current_turn=1, max_turns=7,
            seed_code="def generate_signals(df, p): pass",
        )
        # seed_section requires both name and code
        assert "Seed Strategy" not in prompt

    def test_seed_params_defaults_to_empty_dict(self):
        mandate = {"name": "test", "tickers": ["AAPL"], "period": "1y"}
        prompt = build_turn1_prompt(
            mandate=mandate, current_turn=1, max_turns=7,
            seed_strategy_name="s1",
            seed_code="def generate_signals(df, p): pass",
        )
        assert "Seed Strategy: s1" in prompt
        assert "Default params: {}" in prompt

    def test_indicator_reference_not_in_user_prompt(self):
        """indicator_reference goes into system prompt, not user message."""
        mandate = {"name": "test", "tickers": ["AAPL"], "period": "1y"}
        prompt = build_turn1_prompt(
            mandate=mandate, current_turn=1, max_turns=7,
            indicator_reference="Full ref text here",
        )
        # The turn1 prompt doesn't use indicator_reference directly
        # (it goes into SYSTEM_PROMPT, not TURN1_PROMPT)
        assert "Full ref text here" not in prompt


# ─── build_refinement_prompt (additional coverage) ───────────────────────────


class TestBuildRefinementPromptAdditional:
    def _make_report(self, **overrides):
        defaults = {
            "sharpe_ratio": 1.0,
            "total_return_pct": 0.1,
            "max_drawdown_pct": -0.05,
            "win_rate": 0.5,
            "total_trades": 20,
            "profit_factor": 1.3,
            "calmar_ratio": 1.0,
            "sortino_ratio": 1.5,
            "composite_score": 1.0,
            "failure_mode": "low_sharpe",
            "failure_details": "Sharpe below target",
            "sharpe_by_year": {},
            "stagnation_score": 0.0,
            "stagnation_trend": "improving",
            "previous_sharpes": [],
            "previous_actions": [],
            "guardrail_violations": [],
            "guardrail_warnings": [],
            "current_strategy_code": "def generate_signals(df, params): pass",
            "current_params": {"period": 14},
            "previous_attempts": [],
            "consecutive_modify_params": 0,
        }
        defaults.update(overrides)
        return defaults

    def test_with_winner_examples(self):
        report = self._make_report()
        winners = [{
            "name": "bb_squeeze",
            "sharpe": 1.9,
            "trades": 35,
            "ticker": "QQQ",
            "source_code": "def generate_signals(df, p): pass",
        }]
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=2, max_turns=7,
            sharpe_target=1.5, best_sharpe=0.8, best_turn=1,
            winner_examples=winners,
        )
        assert "Proven Strategies" in prompt
        assert "bb_squeeze" in prompt
        assert "(QQQ)" in prompt

    def test_validation_failed_includes_window_breakdown(self):
        report = self._make_report(
            failure_mode="validation_failed",
            failure_details="Out-of-sample failure",
            previous_attempts=[{
                "turn": 1,
                "sharpe": 1.8,
                "failure_mode": "validation_failed",
                "action": "novel",
                "hypothesis": "h1",
                "params_used": {},
                "delta_from_prev": "Initial",
                "validation": {
                    "avg_test_sharpe": 0.2,
                    "windows_passed": 0,
                    "num_windows": 4,
                    "window_results": [
                        {"window": i, "train_sharpe": 2.0, "test_sharpe": 0.3,
                         "degradation": 0.85, "passed": False}
                        for i in range(1, 5)
                    ],
                },
            }],
        )
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=2, max_turns=7,
            sharpe_target=1.5, best_sharpe=1.8, best_turn=1,
        )
        assert "Fix Out-of-Sample Failure" in prompt
        assert "Rolling Walk-Forward" in prompt
        assert "ALL windows failed" in prompt

    def test_no_sharpe_by_year(self):
        report = self._make_report(sharpe_by_year={})
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=2, max_turns=7,
            sharpe_target=1.5, best_sharpe=0.5, best_turn=1,
        )
        assert "not available" in prompt

    def test_turn_and_max_turn_in_prompt(self):
        report = self._make_report()
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=4, max_turns=10,
            sharpe_target=2.0, best_sharpe=1.0, best_turn=2,
        )
        assert "Turn 4/10" in prompt

    def test_stagnation_section(self):
        report = self._make_report(stagnation_score=0.85, stagnation_trend="stuck")
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=5, max_turns=7,
            sharpe_target=1.5, best_sharpe=0.5, best_turn=1,
            stagnation_suffix="PIVOT NOW",
        )
        assert "0.85" in prompt
        assert "stuck" in prompt
        assert "PIVOT NOW" in prompt

    def test_best_so_far_displayed(self):
        report = self._make_report()
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=3, max_turns=7,
            sharpe_target=1.5, best_sharpe=2.3, best_turn=2,
        )
        assert "2.30" in prompt
        assert "turn 2" in prompt

    def test_current_params_displayed(self):
        report = self._make_report(current_params={"fast": 10, "slow": 30})
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=2, max_turns=7,
            sharpe_target=1.5, best_sharpe=0.5, best_turn=1,
        )
        assert "fast" in prompt
        assert "slow" in prompt

    def test_failure_guidance_included_for_known_mode(self):
        report = self._make_report(
            failure_mode="too_few_trades_for_validation",
            total_trades=3,
        )
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=2, max_turns=7,
            sharpe_target=1.5, best_sharpe=0.3, best_turn=1,
        )
        assert "Increase Trade Frequency" in prompt
        assert "3 trades" in prompt

    def test_feature_importance_section(self):
        report = self._make_report(
            feature_importance_section="### Feature Importance\n  RSI: 0.4\n  MACD: 0.3",
        )
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=2, max_turns=7,
            sharpe_target=1.5, best_sharpe=0.5, best_turn=1,
        )
        assert "Feature Importance" in prompt
        assert "RSI" in prompt

    def test_no_strategy_examples(self):
        report = self._make_report()
        prompt = build_refinement_prompt(
            tier1_report=report, current_turn=2, max_turns=7,
            sharpe_target=1.5, best_sharpe=0.5, best_turn=1,
        )
        # Should still contain the section heading
        assert "Example Strategies" in prompt


# ─── format_stagnation_suffix (additional edge cases) ───────────────────────


class TestFormatStagnationSuffixAdditional:
    def test_empty_string_constraint(self):
        """Empty string constraint should return empty (like None)."""
        result = format_stagnation_suffix("", "some suffix")
        assert result == ""

    def test_non_normal_constraint_without_suffix(self):
        """Non-normal constraint but no prompt_suffix → empty."""
        result = format_stagnation_suffix("pivot", "")
        assert result == ""

    def test_non_normal_constraint_with_none_suffix(self):
        result = format_stagnation_suffix("nuclear", None)
        assert result == ""


# ─── format_tier2_section (additional edge cases) ───────────────────────────


class TestFormatTier2SectionAdditional:
    def test_missing_keys(self):
        """Dict with no tier2 keys at all should return empty."""
        result = format_tier2_section({})
        assert result == ""

    def test_falsy_zero_benchmark(self):
        """benchmark_return_pct=0 is falsy but should be included (is not None)."""
        report = {"benchmark_return_pct": 0.0}
        result = format_tier2_section(report)
        assert "Benchmark" in result
        assert "0.0%" in result

    def test_empty_regime_dict(self):
        """Empty regime_sharpe dict is falsy → section skipped."""
        report = {"regime_sharpe": {}}
        result = format_tier2_section(report)
        assert result == ""

    def test_empty_drawdowns_list(self):
        """Empty top_drawdowns list is falsy → section skipped."""
        report = {"top_drawdowns": []}
        result = format_tier2_section(report)
        assert result == ""

    def test_multiple_drawdowns(self):
        report = {
            "top_drawdowns": [
                {"depth_pct": -0.20, "duration_bars": 30},
                {"depth_pct": -0.10, "duration_bars": 15},
            ],
        }
        result = format_tier2_section(report)
        assert "20.0%" in result
        assert "30 bars" in result
        assert "10.0%" in result
        assert "15 bars" in result
