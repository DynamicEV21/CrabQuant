"""Tests for build_targeted_task_guidance in crabquant.refinement.prompts."""

import pytest

from crabquant.refinement.prompts import build_targeted_task_guidance


# ── helpers ──────────────────────────────────────────────────────────────

def _result(**kwargs) -> str:
    """Call build_targeted_task_guidance with defaults overridden by kwargs."""
    return build_targeted_task_guidance("low_sharpe", **kwargs)


# ── 1. Base instructions always present (steps 1-3) ─────────────────────

@pytest.mark.parametrize("failure_mode", [
    "low_sharpe",
    "too_few_trades",
    "too_few_trades_for_validation",
    "regime_fragility",
    "excessive_drawdown",
    "validation_failed",
    "flat_signal",
    "overtrading",
    "backtest_crash",
    "module_load_failed",
    "unknown",
    "some_random_string",
])
def test_base_instructions_always_present(failure_mode):
    result = build_targeted_task_guidance(failure_mode)
    assert "1. Read the CURRENT STRATEGY CODE above" in result
    assert "2. Read the FAILURE DIAGNOSIS" in result
    assert "3. Read PREVIOUS ATTEMPTS" in result


# ── 2. Each failure mode produces specific step 4 content ───────────────

class TestFailureModeStep4Content:
    def test_low_sharpe(self):
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=0.0, sharpe_target=1.5)
        assert "LARGE gap to target" in result
        assert "4." in result

    def test_too_few_trades(self):
        result = build_targeted_task_guidance("too_few_trades", total_trades=5)
        assert "increase trade frequency" in result
        assert "Widen ONE threshold" in result
        assert "SIMPLIFY" in result

    def test_regime_fragility(self):
        result = build_targeted_task_guidance("regime_fragility")
        assert "REGIME FILTER" in result
        assert "don't replace the signal" in result
        assert "ADX" in result

    def test_excessive_drawdown(self):
        result = build_targeted_task_guidance("excessive_drawdown")
        assert "RISK MANAGEMENT" in result
        assert "don't change the entry signal" in result
        assert "ATR-based stop loss" in result

    def test_validation_failed(self):
        result = build_targeted_task_guidance("validation_failed")
        assert "OVERFIT" in result
        assert "REMOVE" in result
        assert "Widen ALL thresholds" in result

    def test_flat_signal(self):
        result = build_targeted_task_guidance("flat_signal")
        assert "ZERO signals" in result
        assert "`and`/`or` instead of `&`/`|`" in result
        assert "SIMPLEST possible entry" in result

    def test_overtrading(self):
        result = build_targeted_task_guidance("overtrading")
        assert "too many signals" in result
        assert "COOLDOWN" in result
        assert "LONGER indicator periods" in result

    def test_backtest_crash(self):
        result = build_targeted_task_guidance("backtest_crash")
        assert "bug" in result
        assert "function signatures" in result
        assert "pd.Series[bool]" in result

    def test_module_load_failed(self):
        result = build_targeted_task_guidance("module_load_failed")
        assert "bug" in result
        assert "function signatures" in result
        assert "pd.Series[bool]" in result

    def test_unknown(self):
        result = build_targeted_task_guidance("unknown")
        assert "targeted modification with a causal hypothesis" in result

    def test_arbitrary_string(self):
        result = build_targeted_task_guidance("anything_else")
        assert "targeted modification with a causal hypothesis" in result


# ── 3. low_sharpe gap tiers ─────────────────────────────────────────────

class TestLowSharpeGapTiers:
    def test_close_gap(self):
        """gap < 0.5 → CLOSE to target"""
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=1.2, sharpe_target=1.5)
        assert "CLOSE to target" in result
        assert "ONE small" in result
        # gap = 0.30
        assert "gap: 0.30" in result
        assert "LARGE gap" not in result
        assert "MODERATE gap" not in result

    def test_close_gap_boundary_just_under(self):
        """gap = 0.49 → still CLOSE"""
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=1.01, sharpe_target=1.5)
        assert "CLOSE to target" in result

    def test_moderate_gap(self):
        """0.5 <= gap < 1.5 → MODERATE"""
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=0.5, sharpe_target=1.5)
        assert "MODERATE gap" in result
        assert "ONE specific weakness" in result
        assert "Sharpe 0.50" in result
        assert "1.50" in result
        assert "CLOSE to target" not in result
        assert "LARGE gap" not in result

    def test_large_gap(self):
        """gap >= 1.5 → LARGE"""
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=0.0, sharpe_target=1.5)
        assert "LARGE gap" in result
        assert "core signal may need replacement" in result
        assert "Sharpe 0.00" in result

    def test_large_gap_boundary(self):
        """gap = 1.5 exactly → LARGE (boundary is >=)"""
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=0.0, sharpe_target=1.5)
        assert "LARGE gap" in result

    def test_zero_sharpe(self):
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=0.0, sharpe_target=1.5)
        assert "LARGE gap" in result

    def test_negative_sharpe(self):
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=-1.0, sharpe_target=1.5)
        assert "LARGE gap" in result


# ── 4. too_few_trades and too_few_trades_for_validation both work ──────

class TestTooFewTradesVariants:
    def test_too_few_trades(self):
        result = build_targeted_task_guidance("too_few_trades", total_trades=3)
        assert "increase trade frequency" in result
        assert "20+" in result

    def test_too_few_trades_for_validation(self):
        result = build_targeted_task_guidance("too_few_trades_for_validation", total_trades=8)
        assert "increase trade frequency" in result
        assert "20+" in result

    def test_both_produce_same_step4_structure(self):
        r1 = build_targeted_task_guidance("too_few_trades", total_trades=5)
        r2 = build_targeted_task_guidance("too_few_trades_for_validation", total_trades=5)
        # Same total_trades → identical output
        assert r1 == r2


# ── 5. Urgency when remaining <= 2 turns ────────────────────────────────

class TestUrgency:
    @pytest.mark.parametrize("turn_num, max_turns", [
        (5, 7),   # remaining = 2
        (6, 7),   # remaining = 1
        (7, 7),   # remaining = 0
        (1, 3),   # remaining = 2
        (2, 3),   # remaining = 1
        (3, 3),   # remaining = 0
        (4, 4),   # remaining = 0
    ])
    def test_urgency_present(self, turn_num, max_turns):
        result = build_targeted_task_guidance("low_sharpe", turn_num=turn_num, max_turns=max_turns)
        remaining = max_turns - turn_num
        assert f"{remaining} turn(s) remaining" in result
        assert "⚠️" in result
        assert "full_rewrite" in result
        assert "novel" in result

    def test_urgency_shows_correct_count(self):
        result = build_targeted_task_guidance("low_sharpe", turn_num=6, max_turns=7)
        assert "1 turn(s) remaining" in result

    def test_urgency_zero_remaining(self):
        result = build_targeted_task_guidance("low_sharpe", turn_num=7, max_turns=7)
        assert "0 turn(s) remaining" in result


# ── 6. No urgency when remaining > 2 turns ──────────────────────────────

@pytest.mark.parametrize("turn_num, max_turns", [
    (1, 7),  # remaining = 6
    (2, 7),  # remaining = 5
    (3, 7),  # remaining = 4
    (4, 7),  # remaining = 3
    (1, 4),  # remaining = 3
])
def test_no_urgency_when_remaining_gt_2(turn_num, max_turns):
    result = build_targeted_task_guidance("low_sharpe", turn_num=turn_num, max_turns=max_turns)
    assert "turn(s) remaining" not in result
    assert "⚠️" not in result


# ── 7. Trade count shown in too_few_trades guidance ─────────────────────

class TestTradeCountInGuidance:
    def test_shows_trade_count(self):
        result = build_targeted_task_guidance("too_few_trades", total_trades=7)
        assert "from 7 to 20+" in result

    def test_shows_zero_trades(self):
        result = build_targeted_task_guidance("too_few_trades", total_trades=0)
        assert "from 0 to 20+" in result

    def test_shows_large_trade_count(self):
        result = build_targeted_task_guidance("too_few_trades", total_trades=999)
        assert "from 999 to 20+" in result

    def test_for_validation_variant_also_shows_count(self):
        result = build_targeted_task_guidance("too_few_trades_for_validation", total_trades=12)
        assert "from 12 to 20+" in result


# ── 8. Sharpe values shown in low_sharpe guidance ───────────────────────

class TestSharpeValuesInGuidance:
    def test_close_gap_shows_gap_value(self):
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=1.1, sharpe_target=1.5)
        assert "gap: 0.40" in result

    def test_moderate_gap_shows_both_values(self):
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=0.8, sharpe_target=1.5)
        assert "Sharpe 0.80" in result
        assert "1.50" in result

    def test_large_gap_shows_both_values(self):
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=-0.5, sharpe_target=2.0)
        assert "Sharpe -0.50" in result
        assert "2.00" in result

    def test_custom_sharpe_target(self):
        result = build_targeted_task_guidance("low_sharpe", sharpe_ratio=0.0, sharpe_target=3.0)
        assert "3.00" in result


# ── 9. Backtest crash / module load failed map to same guidance ────────

class TestCrashModesIdentical:
    def test_same_output(self):
        r1 = build_targeted_task_guidance("backtest_crash")
        r2 = build_targeted_task_guidance("module_load_failed")
        assert r1 == r2

    def test_both_mention_bug(self):
        for mode in ("backtest_crash", "module_load_failed"):
            result = build_targeted_task_guidance(mode)
            assert "bug" in result


# ── 10. Unknown failure mode gets generic guidance ──────────────────────

class TestUnknownFailureMode:
    def test_generic_message(self):
        result = build_targeted_task_guidance("unknown")
        assert "targeted modification with a causal hypothesis" in result

    def test_still_has_base(self):
        result = build_targeted_task_guidance("unknown")
        assert "1. Read the CURRENT STRATEGY CODE" in result
        assert "2. Read the FAILURE DIAGNOSIS" in result
        assert "3. Read PREVIOUS ATTEMPTS" in result

    def test_empty_string(self):
        result = build_targeted_task_guidance("")
        assert "targeted modification with a causal hypothesis" in result

    def test_random_string(self):
        result = build_targeted_task_guidance("banana_stand")
        assert "targeted modification with a causal hypothesis" in result


# ── 11. All steps are numbered (1-5) ───────────────────────────────────

class TestStepNumbering:
    @pytest.mark.parametrize("failure_mode", [
        "low_sharpe",
        "too_few_trades",
        "regime_fragility",
        "excessive_drawdown",
        "validation_failed",
        "flat_signal",
        "overtrading",
        "backtest_crash",
        "module_load_failed",
        "unknown",
    ])
    def test_steps_1_through_5_present(self, failure_mode):
        result = build_targeted_task_guidance(failure_mode)
        # Check each numbered step exists
        assert "1. " in result
        assert "2. " in result
        assert "3. " in result
        assert "4. " in result
        assert "5. " in result

    def test_step_4_and_5_on_separate_lines(self):
        result = build_targeted_task_guidance("regime_fragility")
        lines = result.split("\n")
        step_lines = [l for l in lines if l.strip().startswith(("1.", "2.", "3.", "4.", "5."))]
        assert len(step_lines) >= 5
        # Verify steps are in order
        assert step_lines[0].strip().startswith("1.")
        assert step_lines[1].strip().startswith("2.")
        assert step_lines[2].strip().startswith("3.")
        assert step_lines[3].strip().startswith("4.")
        assert step_lines[4].strip().startswith("5.")


# ── 12. Return type is always str ───────────────────────────────────────

@pytest.mark.parametrize("failure_mode", [
    "low_sharpe",
    "too_few_trades",
    "too_few_trades_for_validation",
    "regime_fragility",
    "excessive_drawdown",
    "validation_failed",
    "flat_signal",
    "overtrading",
    "backtest_crash",
    "module_load_failed",
    "unknown",
    "",
    "anything",
])
def test_return_type_is_str(failure_mode):
    result = build_targeted_task_guidance(
        failure_mode,
        sharpe_ratio=1.0,
        sharpe_target=2.0,
        total_trades=10,
        turn_num=3,
        max_turns=7,
    )
    assert isinstance(result, str)
    assert len(result) > 0
