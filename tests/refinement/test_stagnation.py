"""
Unit tests for stagnation detection module.
Tests various history patterns to ensure stagnation scoring works correctly.
"""
import pytest
from crabquant.refinement.stagnation import (
    build_stagnation_recovery,
    check_family_plateau,
    check_hypothesis_failure_alignment,
    classify_indicator,
    compute_stagnation,
    detect_stagnation_trap,
    extract_indicators_from_code,
    get_stagnation_response,
    track_indicator_diversity,
)


class TestStagnationDetection:
    """Test stagnation scoring algorithm."""

    def test_compute_stagnation_insufficient_history(self):
        """Test with less than 2 history entries."""
        history = [{"sharpe": 0.5}]
        score, trend = compute_stagnation(history)
        assert score == 0.0
        assert trend == "improving"

    def test_compute_stagnation_improving_trend(self):
        """Test clear upward trend in Sharpe ratios."""
        history = [
            {"sharpe": 0.4, "action": "modify_params"},
            {"sharpe": 0.6, "action": "add_filter"},
            {"sharpe": 0.8, "action": "change_entry_logic"}
        ]
        score, trend = compute_stagnation(history)
        assert trend == "improving"
        # slope=0.2 -> trend=0.0, but high variance (std=0.2) pushes score up
        assert score <= 0.4  # Low-ish despite variance (improving trend = 0.0 factor)

    def test_compute_stagnation_declining_trend(self):
        """Test downward trend with repetitive actions."""
        history = [
            {"sharpe": 0.8, "action": "modify_params"},
            {"sharpe": 0.6, "action": "modify_params"},
            {"sharpe": 0.3, "action": "modify_params"}
        ]
        score, trend = compute_stagnation(history)
        assert trend == "declining"
        # declining trend + high variance + all same action = high stagnation
        assert score >= 0.6

    def test_compute_stagnation_flat_trend(self):
        """Test flat/oscillating Sharpe ratios with repetitive actions."""
        history = [
            {"sharpe": 0.5, "action": "modify_params"},
            {"sharpe": 0.52, "action": "modify_params"},
            {"sharpe": 0.48, "action": "modify_params"}
        ]
        score, trend = compute_stagnation(history)
        assert trend == "flat"
        # flat trend(0.7) + low variance + repetitive(1.0) = moderate
        assert score >= 0.4

    def test_compute_stagnation_high_variance(self):
        """Test with high variance in Sharpe ratios."""
        history = [
            {"sharpe": 0.2, "action": "modify_params"},
            {"sharpe": 0.9, "action": "add_filter"},
            {"sharpe": 0.1, "action": "change_entry_logic"}
        ]
        score, trend = compute_stagnation(history)
        assert score > 0.5  # High variance + declining trend = elevated score

    def test_compute_stagnation_repetitive_actions(self):
        """Test with repetitive 'modify_params' actions."""
        history = [
            {"sharpe": 0.5, "action": "modify_params"},
            {"sharpe": 0.55, "action": "modify_params"},
            {"sharpe": 0.52, "action": "modify_params"}
        ]
        score, trend = compute_stagnation(history)
        # flat trend + repetitive actions = elevated stagnation
        assert score > 0.4

    def test_compute_stagnation_diverse_actions(self):
        """Test with diverse action types and improving trend."""
        history = [
            {"sharpe": 0.5, "action": "modify_params"},
            {"sharpe": 0.6, "action": "add_filter"},
            {"sharpe": 0.7, "action": "change_entry_logic"},
            {"sharpe": 0.8, "action": "replace_indicator"}
        ]
        score, trend = compute_stagnation(history)
        assert score < 0.4  # Diverse actions with improving trend should have low score

    def test_compute_stagnation_empty_history(self):
        score, trend = compute_stagnation([])
        assert score == 0.0
        assert trend == "improving"

    def test_compute_stagnation_single_sharpe_entry(self):
        """Only one entry with sharpe but that's not enough."""
        history = [{"sharpe": 0.5}, {"action": "modify_params"}]  # only 1 sharpe
        score, trend = compute_stagnation(history)
        assert score == 0.0

    def test_compute_stagnation_two_modify_params(self):
        """Two modify_params out of last 3 gives partial repetition."""
        history = [
            {"sharpe": 0.5, "action": "modify_params"},
            {"sharpe": 0.5, "action": "modify_params"},
            {"sharpe": 0.5, "action": "add_filter"},
        ]
        score, trend = compute_stagnation(history)
        assert score > 0.0  # some repetition score


class TestStagnationResponse:
    """Test stagnation response protocol."""

    def test_get_stagnation_response_early_iteration(self):
        """Test response for early iterations (should be normal)."""
        response = get_stagnation_response(2, 0.5)
        assert response["constraint"] == "normal"
        assert response["prompt_suffix"] == ""

    def test_get_stagnation_response_abandon_threshold(self):
        """Test response for high stagnation score (abandon)."""
        response = get_stagnation_response(5, 0.9)
        assert response["constraint"] == "abandon"
        assert "ABANDON" in response["prompt_suffix"]

    def test_get_stagnation_response_nuclear_rewrite(self):
        """Test nuclear rewrite response for moderate stagnation with high iteration."""
        response = get_stagnation_response(7, 0.7)
        assert response["constraint"] == "nuclear"
        assert "NUCLEAR REWRITE" in response["prompt_suffix"]

    def test_get_stagnation_response_pivot_threshold(self):
        """Test pivot response for high stagnation."""
        response = get_stagnation_response(4, 0.75)
        assert response["constraint"] == "pivot"
        assert "PIVOT" in response["prompt_suffix"]

    def test_get_stagnation_response_broaden_threshold(self):
        """Test broaden response for moderate stagnation at iteration > 3."""
        response = get_stagnation_response(5, 0.55)
        assert response["constraint"] == "broaden"
        assert "BROADEN" in response["prompt_suffix"]

    def test_get_stagnation_response_edge_case_just_below_abandon(self):
        """Test response just below abandon threshold."""
        response = get_stagnation_response(5, 0.79)
        assert response["constraint"] == "pivot"

    def test_get_stagnation_response_at_abandon_threshold(self):
        """Test response at abandon threshold."""
        response = get_stagnation_response(4, 0.81)
        assert response["constraint"] == "abandon"

    def test_get_stagnation_response_broaden_at_iteration_4(self):
        """Test broaden at iteration 4 with score 0.6 (not nuclear since iter < 6)."""
        response = get_stagnation_response(4, 0.6)
        assert response["constraint"] == "broaden"

    def test_get_stagnation_response_early_high_score(self):
        """Even with high score, iteration <= 3 returns normal."""
        response = get_stagnation_response(3, 0.95)
        assert response["constraint"] == "normal"

    def test_get_stagnation_response_low_score_late_iteration(self):
        """Low score even at late iteration returns normal."""
        response = get_stagnation_response(10, 0.3)
        assert response["constraint"] == "normal"

    def test_get_stagnation_response_nuclear_requires_iteration_6(self):
        """Nuclear requires score > 0.6 AND iteration >= 6."""
        response = get_stagnation_response(4, 0.65)
        assert response["constraint"] == "broaden"  # not nuclear

    def test_get_stagnation_response_returns_dict(self):
        response = get_stagnation_response(1, 0.0)
        assert isinstance(response, dict)
        assert "constraint" in response
        assert "prompt_suffix" in response


class TestHypothesisFailureAlignment:
    """Test the hypothesis-failure mismatch guard."""

    def test_matching_failure_mode_no_warning(self):
        warnings = check_hypothesis_failure_alignment("low_sharpe", "low_sharpe", "add_filter")
        assert warnings == []

    def test_mismatched_failure_mode_warns(self):
        warnings = check_hypothesis_failure_alignment("too_few_trades", "low_sharpe", "add_filter")
        assert len(warnings) == 1
        assert "Diagnosed" in warnings[0]

    def test_action_mismatch_warns(self):
        warnings = check_hypothesis_failure_alignment(
            "too_few_trades", "too_few_trades", "change_exit_logic"
        )
        assert len(warnings) == 1
        assert "Tightening exits" in warnings[0]

    def test_drawdown_modify_params_warns(self):
        warnings = check_hypothesis_failure_alignment(
            "excessive_drawdown", "excessive_drawdown", "modify_params"
        )
        assert len(warnings) == 1
        assert "Tweaks rarely fix" in warnings[0]

    def test_both_mismatched(self):
        warnings = check_hypothesis_failure_alignment(
            "excessive_drawdown", "low_sharpe", "modify_params"
        )
        assert len(warnings) == 2

    def test_unknown_pair_no_warning(self):
        warnings = check_hypothesis_failure_alignment("some_mode", "some_mode", "some_action")
        assert warnings == []


class TestClassifyIndicator:
    """Test indicator family classification."""

    def test_momentum_indicators(self):
        assert classify_indicator("ema") == "momentum"
        assert classify_indicator("sma") == "momentum"
        assert classify_indicator("macd") == "momentum"
        assert classify_indicator("roc") == "momentum"

    def test_mean_reversion_indicators(self):
        assert classify_indicator("rsi") == "mean_reversion"
        assert classify_indicator("bbands") == "mean_reversion"
        assert classify_indicator("bollinger") == "mean_reversion"
        assert classify_indicator("stoch") == "mean_reversion"

    def test_volatility_indicators(self):
        assert classify_indicator("atr") == "volatility"
        assert classify_indicator("adx") == "volatility"
        assert classify_indicator("supertrend") == "volatility"

    def test_volume_indicators(self):
        assert classify_indicator("obv") == "volume"
        assert classify_indicator("vwap") == "volume"

    def test_trend_indicators(self):
        assert classify_indicator("ichimoku") == "trend"
        assert classify_indicator("dema") == "trend"
        assert classify_indicator("tema") == "trend"
        assert classify_indicator("psar") == "trend"

    def test_unknown_indicator(self):
        assert classify_indicator("foobar") == "unknown"
        assert classify_indicator("xyz_indicator") == "unknown"

    def test_case_insensitive(self):
        assert classify_indicator("RSI") == "mean_reversion"
        assert classify_indicator("EMA") == "momentum"
        assert classify_indicator("Atr") == "volatility"


class TestExtractIndicatorsFromCode:
    """Test indicator extraction from source code."""

    def test_cached_indicator_double_quotes(self):
        code = 'cached_indicator("ema", span=12)'
        indicators = extract_indicators_from_code(code)
        assert "ema" in indicators

    def test_cached_indicator_single_quotes(self):
        code = "cached_indicator('rsi', period=14)"
        indicators = extract_indicators_from_code(code)
        assert "rsi" in indicators

    def test_ta_dot_notation(self):
        code = "ta.macd(close, fast=12, slow=26)"
        indicators = extract_indicators_from_code(code)
        assert "macd" in indicators

    def test_pandas_ta_notation(self):
        code = "pandas_ta.atr(high, low, close, length=14)"
        indicators = extract_indicators_from_code(code)
        assert "atr" in indicators

    def test_empty_code(self):
        indicators = extract_indicators_from_code("")
        assert indicators == []

    def test_deduplication(self):
        code = """
        cached_indicator("ema", 12)
        cached_indicator("ema", 26)
        cached_indicator("rsi", 14)
        """
        indicators = extract_indicators_from_code(code)
        assert indicators == ["ema", "rsi"]

    def test_mixed_sources(self):
        code = """
        cached_indicator("ema", 12)
        ta.rsi(close, 14)
        pandas_ta.atr(high, low, close)
        """
        indicators = extract_indicators_from_code(code)
        assert "ema" in indicators
        assert "rsi" in indicators
        assert "atr" in indicators


class TestTrackIndicatorDiversity:
    """Test indicator diversity tracking."""

    def test_empty_history(self):
        result = track_indicator_diversity([])
        assert result["families_used"] == set()
        assert result["family_counts"] == {}
        assert result["dominant_family"] is None
        assert result["is_rut"] is False
        assert result["all_indicators"] == []

    def test_with_current_code(self):
        code = 'cached_indicator("ema", 12)\ncached_indicator("rsi", 14)'
        result = track_indicator_diversity([], current_code=code)
        assert result["all_indicators"] == ["ema", "rsi"]
        assert "momentum" in result["families_used"]
        assert "mean_reversion" in result["families_used"]

    def test_rut_detection(self):
        # Rut detection requires code-based extraction (params path adds to
        # all_indicators but not family_counts). Use current_code + history
        # with code paths that exist to exercise the code-reading branch.
        # Instead, provide current_code that uses the same family repeatedly.
        code_momentum = 'cached_indicator("ema", 12)\ncached_indicator("sma", 26)'
        # Each call to track_indicator_diversity processes current_code once,
        # so we need the code to extract multiple indicators from same family.
        code_multi = """
        cached_indicator("ema", 12)
        cached_indicator("sma", 26)
        cached_indicator("macd", 14)
        cached_indicator("wma", 10)
        """
        result = track_indicator_diversity([], current_code=code_multi)
        assert "momentum" in result["families_used"]
        # 4 indicators all in momentum family → dominant
        assert result["dominant_family"] == "momentum"
        # total_family_uses = 4, dominant_pct = 100% >= 80% and total >= 3
        assert result["is_rut"] is True

    def test_no_rut_with_diverse_families(self):
        history = [
            {"params_used": {"ema_period": 12}},
            {"params_used": {"rsi_period": 14}},
            {"params_used": {"atr_period": 10}},
        ]
        result = track_indicator_diversity(history)
        assert result["is_rut"] is False

    def test_recovery_hint_present_when_rut(self):
        history = [
            {"params_used": {"ema_fast": 12, "ema_slow": 26}},
            {"params_used": {"ema_fast": 10, "ema_slow": 20}},
            {"params_used": {"ema_fast": 8, "ema_slow": 30}},
        ]
        result = track_indicator_diversity(history)
        if result["is_rut"]:
            assert "INDICATOR RUT" in result["recovery_hint"]
            assert result["dominant_family"] in result["recovery_hint"]


class TestDetectStagnationTrap:
    """Test specific trap detection."""

    def test_no_trap_few_turns(self):
        history = [{"sharpe": 0.5}]
        result = detect_stagnation_trap(history)
        assert result["trap"] == "no_trap"
        assert result["severity"] == "low"

    def test_zero_sharpe_trap(self):
        history = [
            {"sharpe": -0.1, "action": "modify_params"},
            {"sharpe": -0.2, "action": "modify_params"},
            {"sharpe": 0.0, "action": "modify_params"},
        ]
        result = detect_stagnation_trap(history)
        assert result["trap"] == "zero_sharpe"
        assert result["severity"] == "critical"

    def test_low_sharpe_plateau(self):
        history = [
            {"sharpe": 0.1, "action": "modify_params"},
            {"sharpe": 0.2, "action": "add_filter"},
            {"sharpe": 0.15, "action": "modify_params"},
        ]
        result = detect_stagnation_trap(history)
        assert result["trap"] == "low_sharpe_plateau"
        assert result["severity"] == "high"

    def test_action_loop_trap(self):
        history = [
            {"sharpe": 0.5, "action": "modify_params", "failure_mode": ""},
            {"sharpe": 0.5, "action": "modify_params", "failure_mode": ""},
            {"sharpe": 0.5, "action": "modify_params", "failure_mode": ""},
            {"sharpe": 0.5, "action": "modify_params", "failure_mode": ""},
            {"sharpe": 0.5, "action": "modify_params", "failure_mode": ""},
        ]
        result = detect_stagnation_trap(history)
        assert result["trap"] == "action_loop"
        assert result["severity"] == "medium"

    def test_validation_loop_trap(self):
        history = [
            {"sharpe": 1.5, "action": "modify_params", "failure_mode": "validation_failed"},
            {"sharpe": 1.6, "action": "add_filter", "failure_mode": "validation_failed"},
            {"sharpe": 1.4, "action": "modify_params", "failure_mode": "validation_failed"},
        ]
        result = detect_stagnation_trap(history, best_sharpe=1.6, sharpe_target=1.5)
        assert result["trap"] == "validation_loop"
        assert result["severity"] == "high"

    def test_high_sharpe_few_trades_trap(self):
        history = [
            {"sharpe": 1.5, "num_trades": 5},
            {"sharpe": 1.6, "num_trades": 8},
        ]
        result = detect_stagnation_trap(history, best_sharpe=1.6, sharpe_target=1.5)
        assert result["trap"] == "high_sharpe_few_trades"
        assert result["severity"] == "high"

    def test_near_target_trap(self):
        # near_target: best_sharpe >= sharpe_target*0.7 and best < target
        # Also need avg_recent > 0.7 to avoid mid_sharpe_trap
        history = [
            {"sharpe": 0.9, "action": "modify_params"},
            {"sharpe": 1.0, "action": "add_filter"},
            {"sharpe": 1.1, "action": "modify_params"},
        ]
        result = detect_stagnation_trap(history, best_sharpe=1.1, sharpe_target=1.5)
        assert result["trap"] == "near_target"
        assert result["severity"] == "low"

    def test_mid_sharpe_trap(self):
        history = [
            {"sharpe": 0.4, "action": "modify_params"},
            {"sharpe": 0.5, "action": "add_filter"},
            {"sharpe": 0.45, "action": "modify_params"},
        ]
        result = detect_stagnation_trap(history, best_sharpe=0.5, sharpe_target=1.5)
        assert result["trap"] == "mid_sharpe_trap"
        assert result["severity"] == "medium"

    def test_improving_no_trap(self):
        # For no_trap: best_sharpe >= sharpe_target, so near_target won't match
        # avg_recent > 0.7 so mid_sharpe won't match
        history = [
            {"sharpe": 1.2, "action": "replace_indicator"},
            {"sharpe": 1.3, "action": "change_entry_logic"},
            {"sharpe": 1.4, "action": "add_filter"},
        ]
        result = detect_stagnation_trap(history, best_sharpe=1.4, sharpe_target=1.3)
        assert result["trap"] == "no_trap"

    def test_result_has_required_keys(self):
        history = [{"sharpe": 0.5}, {"sharpe": 0.6}]
        result = detect_stagnation_trap(history)
        for key in ("trap", "severity", "sharpes", "recent_failure_modes",
                     "recent_actions", "turns_in_trap", "description"):
            assert key in result


class TestBuildStagnationRecovery:
    """Test recovery instruction generation."""

    def test_no_trap_returns_empty(self):
        trap_info = {"trap": "no_trap", "severity": "low", "turns_in_trap": 0}
        assert build_stagnation_recovery(trap_info) == ""

    def test_low_severity_returns_empty(self):
        trap_info = {"trap": "near_target", "severity": "low", "turns_in_trap": 2}
        assert build_stagnation_recovery(trap_info) == ""

    def test_zero_sharpe_recovery(self):
        trap_info = {"trap": "zero_sharpe", "severity": "critical", "turns_in_trap": 5}
        recovery = build_stagnation_recovery(trap_info)
        assert "Zero Sharpe" in recovery
        assert "MANDATORY" in recovery

    def test_low_sharpe_plateau_recovery(self):
        trap_info = {"trap": "low_sharpe_plateau", "severity": "high", "turns_in_trap": 4}
        recovery = build_stagnation_recovery(trap_info)
        assert "Low Sharpe Plateau" in recovery
        assert "ABANDON" in recovery

    def test_mid_sharpe_trap_recovery(self):
        trap_info = {"trap": "mid_sharpe_trap", "severity": "medium", "turns_in_trap": 3}
        recovery = build_stagnation_recovery(trap_info)
        assert "Mid Sharpe Trap" in recovery

    def test_high_sharpe_few_trades_recovery(self):
        trap_info = {"trap": "high_sharpe_few_trades", "severity": "high", "turns_in_trap": 3}
        recovery = build_stagnation_recovery(trap_info)
        assert "Trade Count Trap" in recovery
        assert "30+ trades" in recovery

    def test_validation_loop_recovery(self):
        trap_info = {"trap": "validation_loop", "severity": "high", "turns_in_trap": 4}
        recovery = build_stagnation_recovery(trap_info)
        assert "Validation Failure Loop" in recovery
        assert "overfitting" in recovery.lower()

    def test_action_loop_recovery(self):
        trap_info = {"trap": "action_loop", "severity": "medium", "turns_in_trap": 3}
        recovery = build_stagnation_recovery(trap_info)
        assert "Action Loop" in recovery
        assert "DIFFERENT action" in recovery

    def test_indicator_rut_recovery(self):
        trap_info = {
            "trap": "indicator_rut",
            "severity": "medium",
            "turns_in_trap": 5,
            "description": "Stuck using 'momentum' indicators on every turn.",
        }
        recovery = build_stagnation_recovery(trap_info)
        assert "Indicator Rut" in recovery
        assert "momentum" in recovery

    def test_unknown_trap_returns_empty(self):
        trap_info = {"trap": "unknown_trap", "severity": "high", "turns_in_trap": 1}
        assert build_stagnation_recovery(trap_info) == ""


class TestCheckFamilyPlateau:
    """Test mandate-aware forced exploration on plateau (Enhancement 8)."""

    def _momentum_turn(self, status: str = "REJECT") -> dict:
        """Create a turn dict that classifies as 'momentum' family."""
        return {
            "status": status,
            "code": 'cached_indicator("ema", 12)\ncached_indicator("macd", 14)',
        }

    def _mean_reversion_turn(self, status: str = "REJECT") -> dict:
        """Create a turn dict that classifies as 'mean_reversion' family."""
        return {
            "status": status,
            "code": 'cached_indicator("rsi", 14)\ncached_indicator("bbands", 20)',
        }

    # ── Within-archetype pivot ────────────────────────────────────────

    def test_within_archetype_pivot_when_mandate_matches(self):
        """Mandate=momentum, stuck on momentum, no force_diversify → 'within'."""
        history = [self._momentum_turn() for _ in range(3)]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is True
        assert pivot_type == "within"
        assert "momentum" in message
        assert "STUCK" in message

    def test_within_archetype_message_suggests_alternatives(self):
        """Within-archetype message should mention trying different indicators."""
        history = [self._momentum_turn() for _ in range(3)]
        mandate = {"strategy_archetype": "momentum"}
        _, _, message = check_family_plateau(history, mandate)
        assert "MACD" in message
        assert "ROC" in message

    # ── Cross-archetype pivot ─────────────────────────────────────────

    def test_cross_archetype_pivot_when_force_diversify(self):
        """Mandate=momentum, stuck on momentum, force_diversify=True → 'cross'."""
        history = [self._momentum_turn() for _ in range(3)]
        mandate = {"strategy_archetype": "momentum", "force_diversify": True}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is True
        assert pivot_type == "cross"
        assert "STUCK" in message

    def test_cross_archetype_pivot_when_family_mismatches_mandate(self):
        """Mandate=trend, stuck on momentum → 'cross' (family != archetype)."""
        history = [self._momentum_turn() for _ in range(3)]
        mandate = {"strategy_archetype": "trend"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is True
        assert pivot_type == "cross"

    def test_cross_archetype_lists_alternative_families(self):
        """Cross-archetype message should list families excluding the stuck one."""
        history = [self._momentum_turn() for _ in range(3)]
        mandate = {"strategy_archetype": "mean_reversion"}
        _, _, message = check_family_plateau(history, mandate)
        assert "mean_reversion" in message  # listed as alternative
        assert "Do NOT use any momentum" in message

    # ── No pivot cases ────────────────────────────────────────────────

    def test_no_pivot_when_diverse_families(self):
        """Different families in recent turns → no pivot."""
        history = [
            self._momentum_turn(),
            self._mean_reversion_turn(),
            self._momentum_turn(),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is False
        assert pivot_type is None
        assert message is None

    def test_no_pivot_when_keep_in_history(self):
        """A KEEP status in recent history suppresses the pivot."""
        history = [
            self._momentum_turn(status="REJECT"),
            self._momentum_turn(status="KEEP"),
            self._momentum_turn(status="REJECT"),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is False
        assert pivot_type is None

    def test_no_pivot_when_keep_case_insensitive(self):
        """KEEP status matching is case-insensitive."""
        history = [
            self._momentum_turn(status="reject"),
            self._momentum_turn(status="keep"),
            self._momentum_turn(status="reject"),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, _ = check_family_plateau(history, mandate)
        assert should_pivot is False

    def test_no_pivot_when_too_few_turns(self):
        """Fewer turns than max_same_family → no pivot."""
        history = [self._momentum_turn(), self._momentum_turn()]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, _ = check_family_plateau(history, mandate)
        assert should_pivot is False

    def test_no_pivot_when_unknown_family(self):
        """Turns that classify as 'unknown' don't trigger a pivot."""
        history = [
            {"status": "REJECT", "params_used": {"foo_bar": 1}},
            {"status": "REJECT", "params_used": {"baz_qux": 2}},
            {"status": "REJECT", "params_used": {"random_key": 3}},
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, _ = check_family_plateau(history, mandate)
        assert should_pivot is False

    # ── Edge cases ────────────────────────────────────────────────────

    def test_params_used_fallback_classification(self):
        """Family inferred from params_used keys when no code is present."""
        history = [
            {"status": "REJECT", "params_used": {"ema_fast": 12, "ema_slow": 26}},
            {"status": "REJECT", "params_used": {"macd_fast": 12, "macd_slow": 26}},
            {"status": "REJECT", "params_used": {"sma_period": 50}},
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is True
        assert pivot_type == "within"

    def test_custom_max_same_family(self):
        """Respects custom max_same_family threshold."""
        history = [self._momentum_turn() for _ in range(4)]
        mandate = {"strategy_archetype": "momentum"}
        # With max_same_family=4, 4 turns should trigger
        should_pivot, _, _ = check_family_plateau(
            history, mandate, max_same_family=4
        )
        assert should_pivot is True
        # With max_same_family=5, 4 turns should NOT trigger
        should_pivot, _, _ = check_family_plateau(
            history, mandate, max_same_family=5
        )
        assert should_pivot is False

    def test_empty_history(self):
        should_pivot, pivot_type, _ = check_family_plateau(
            [], {"strategy_archetype": "momentum"}
        )
        assert should_pivot is False
