"""
Unit tests for stagnation detection module.
Tests various history patterns to ensure stagnation scoring works correctly.
"""
import pytest
from crabquant.refinement.stagnation import (
    compute_stagnation,
    get_stagnation_response,
    check_hypothesis_failure_alignment,
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
