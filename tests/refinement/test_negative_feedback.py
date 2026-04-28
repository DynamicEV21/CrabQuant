"""Tests for negative example feedback loop (Task 4).

Tests that format_previous_attempts_section and build_failure_guidance
provide the correct actionable guidance for each failure mode, including
the rolling walk-forward window breakdown for validation_failed.
"""

import pytest

from crabquant.refinement.prompts import (
    build_failure_guidance,
    format_previous_attempts_section,
    _format_window_breakdown,
)


# ── Test: _format_window_breakdown ───────────────────────────────────────


class TestFormatWindowBreakdown:
    """Tests for the per-window breakdown formatter."""

    def test_empty_validation(self):
        result = _format_window_breakdown({})
        assert result == ""

    def test_no_window_results(self):
        result = _format_window_breakdown({"avg_test_sharpe": 0.5})
        assert result == ""

    def test_all_windows_pass(self):
        validation = {
            "window_results": [
                {"window": 1, "train_sharpe": 1.5, "test_sharpe": 1.2, "degradation": 0.2, "passed": True},
                {"window": 2, "train_sharpe": 1.8, "test_sharpe": 1.0, "degradation": 0.44, "passed": True},
                {"window": 3, "train_sharpe": 1.3, "test_sharpe": 0.9, "degradation": 0.31, "passed": True},
            ]
        }
        result = _format_window_breakdown(validation)
        assert "Window | Train Sharpe | Test Sharpe" in result
        assert "✅ PASS" in result
        assert "❌ FAIL" not in result
        assert "3/3 windows passed" in result

    def test_all_windows_fail(self):
        validation = {
            "window_results": [
                {"window": 1, "train_sharpe": 2.0, "test_sharpe": -0.5, "degradation": 1.0, "passed": False},
                {"window": 2, "train_sharpe": 1.5, "test_sharpe": 0.1, "degradation": 0.93, "passed": False},
            ]
        }
        result = _format_window_breakdown(validation)
        assert "ALL windows failed" in result
        assert "SIMPLIFY drastically" in result

    def test_mixed_windows(self):
        validation = {
            "window_results": [
                {"window": 1, "train_sharpe": 1.5, "test_sharpe": 1.0, "degradation": 0.33, "passed": True},
                {"window": 2, "train_sharpe": 2.0, "test_sharpe": -0.3, "degradation": 1.0, "passed": False},
                {"window": 3, "train_sharpe": 1.8, "test_sharpe": 0.5, "degradation": 0.72, "passed": False},
                {"window": 4, "train_sharpe": 1.2, "test_sharpe": 0.8, "degradation": 0.33, "passed": True},
            ]
        }
        result = _format_window_breakdown(validation)
        assert "2/4 windows passed" in result
        assert "reduce parameters" in result.lower()

    def test_one_window_pass(self):
        validation = {
            "window_results": [
                {"window": 1, "train_sharpe": 2.0, "test_sharpe": 1.5, "degradation": 0.25, "passed": True},
                {"window": 2, "train_sharpe": 1.5, "test_sharpe": -0.5, "degradation": 1.0, "passed": False},
                {"window": 3, "train_sharpe": 1.8, "test_sharpe": 0.2, "degradation": 0.89, "passed": False},
            ]
        }
        result = _format_window_breakdown(validation)
        assert "Only 1/3 windows passed" in result
        assert "heavily overfit" in result

    def test_error_window(self):
        validation = {
            "window_results": [
                {"window": 1, "train_sharpe": 1.5, "test_sharpe": 1.0, "degradation": 0.33, "passed": True},
                {"window": 2, "train_sharpe": 0, "test_sharpe": 0, "degradation": 1.0, "passed": False, "error": "Not enough bars for training"},
            ]
        }
        result = _format_window_breakdown(validation)
        assert "💥 ERROR" in result
        assert "Not enough bars" in result

    def test_sharpe_values_formatted(self):
        validation = {
            "window_results": [
                {"window": 1, "train_sharpe": 1.23, "test_sharpe": 0.45, "degradation": 0.63, "passed": True},
            ]
        }
        result = _format_window_breakdown(validation)
        assert "1.23" in result
        assert "0.45" in result
        assert "63%" in result


# ── Test: build_failure_guidance ─────────────────────────────────────────


class TestBuildFailureGuidance:

    def test_unknown_mode_returns_empty(self):
        result = build_failure_guidance("unknown_mode", 50)
        assert result == ""

    def test_too_few_trades_for_validation(self):
        result = build_failure_guidance("too_few_trades_for_validation", 5)
        assert "Increase Trade Frequency" in result
        assert "5 trades" in result
        assert "REMOVE conditions" in result
        assert "WIDEN thresholds" in result
        assert "30+ trades" in result

    def test_validation_failed_without_validation_data(self):
        result = build_failure_guidance("validation_failed", 45)
        assert "Fix Out-of-Sample Failure" in result
        assert "REDUCE complexity" in result
        assert "Rolling Walk-Forward" not in result  # No window breakdown without data

    def test_validation_failed_with_window_breakdown(self):
        validation = {
            "avg_test_sharpe": 0.3,
            "windows_passed": 2,
            "num_windows": 6,
            "window_results": [
                {"window": 1, "train_sharpe": 1.5, "test_sharpe": 1.0, "degradation": 0.33, "passed": True},
                {"window": 2, "train_sharpe": 1.8, "test_sharpe": 0.8, "degradation": 0.56, "passed": True},
                {"window": 3, "train_sharpe": 2.0, "test_sharpe": -0.2, "degradation": 1.0, "passed": False},
                {"window": 4, "train_sharpe": 1.5, "test_sharpe": 0.1, "degradation": 0.93, "passed": False},
                {"window": 5, "train_sharpe": 1.7, "test_sharpe": -0.1, "degradation": 1.0, "passed": False},
                {"window": 6, "train_sharpe": 1.3, "test_sharpe": 0.0, "degradation": 1.0, "passed": False},
            ],
        }
        result = build_failure_guidance("validation_failed", 45, validation)
        assert "Fix Out-of-Sample Failure" in result
        assert "Rolling Walk-Forward Window Breakdown" in result
        assert "2/6 windows passed" in result
        assert "Window | Train Sharpe | Test Sharpe" in result

    def test_regime_fragility(self):
        result = build_failure_guidance("regime_fragility", 30)
        assert "Regime-Dependent" in result
        assert "regime detection" in result
        assert "adaptive parameters" in result

    def test_low_sharpe_very_few_trades_curve_fit_warning(self):
        result = build_failure_guidance("low_sharpe", 3)
        assert "CRITICAL" in result
        assert "curve-fit" in result
        assert "3 trades" in result
        assert "REPLACE the entire strategy" in result
        assert "30+ times" in result

    def test_low_sharpe_few_trades_unreliable(self):
        result = build_failure_guidance("low_sharpe", 12)
        assert "Very few trades (12)" in result
        assert "unreliable" in result
        assert "CRITICAL" not in result  # Not < 10

    def test_low_sharpe_enough_trades_no_warning(self):
        result = build_failure_guidance("low_sharpe", 25)
        assert "CRITICAL" not in result
        assert "Very few trades" not in result
        assert "Improve Sharpe Ratio" in result


# ── Test: format_previous_attempts_section ───────────────────────────────


class TestFormatPreviousAttemptsSection:

    def test_empty_returns_hint(self):
        result = format_previous_attempts_section([])
        assert "no previous attempts" in result

    def test_too_few_trades_has_guidance(self):
        attempts = [
            {
                "turn": 2,
                "sharpe": 2.5,
                "failure_mode": "too_few_trades_for_validation",
                "action": "novel",
                "hypothesis": "test hypothesis",
                "params_used": {"rsi_period": 14},
                "delta_from_prev": "Initial",
                "num_trades": 4,
            }
        ]
        result = format_previous_attempts_section(attempts)
        assert "Only 4 trades" in result
        assert "OPEN UP conditions" in result
        assert "remove filters" in result
        assert "widen thresholds" in result

    def test_validation_failed_has_window_breakdown(self):
        attempts = [
            {
                "turn": 3,
                "sharpe": 1.8,
                "failure_mode": "validation_failed",
                "action": "modify_params",
                "hypothesis": "widen RSI threshold",
                "params_used": {"rsi_period": 14},
                "delta_from_prev": "Changed RSI from 20 to 25",
                "num_trades": 35,
                "validation": {
                    "avg_test_sharpe": 0.2,
                    "windows_passed": 1,
                    "num_windows": 6,
                    "window_results": [
                        {"window": 1, "train_sharpe": 2.0, "test_sharpe": 1.2, "degradation": 0.4, "passed": True},
                        {"window": 2, "train_sharpe": 1.8, "test_sharpe": -0.5, "degradation": 1.0, "passed": False},
                        {"window": 3, "train_sharpe": 1.5, "test_sharpe": 0.1, "degradation": 0.93, "passed": False},
                    ],
                },
            }
        ]
        result = format_previous_attempts_section(attempts)
        assert "FAILED out-of-sample validation" in result
        assert "avg test Sharpe=0.2" in result
        assert "1/6 windows passed" in result
        assert "Rolling Walk-Forward Window Breakdown" in result
        assert "Window | Train Sharpe | Test Sharpe" in result
        assert "Only 1/3 windows passed" in result  # From window breakdown summary

    def test_validation_failed_without_validation_data(self):
        attempts = [
            {
                "turn": 3,
                "sharpe": 1.8,
                "failure_mode": "validation_failed",
                "action": "modify_params",
                "hypothesis": "test",
                "params_used": {},
                "delta_from_prev": "Changed",
                "num_trades": 30,
            }
        ]
        result = format_previous_attempts_section(attempts)
        assert "FAILED out-of-sample validation" in result
        assert "REDUCE complexity" in result
        assert "Rolling Walk-Forward" not in result  # No breakdown without data

    def test_low_sharpe_curve_fitting_warning(self):
        attempts = [
            {
                "turn": 2,
                "sharpe": 0.1,
                "failure_mode": "low_sharpe",
                "action": "novel",
                "hypothesis": "test",
                "params_used": {},
                "delta_from_prev": "Initial",
                "num_trades": 5,
            }
        ]
        result = format_previous_attempts_section(attempts)
        assert "CURVE-FITTING RISK" in result
        assert "5 trades" in result
        assert "fitting noise" in result
        assert "SIMPLER" in result
        assert "30+ times" in result

    def test_low_sharpe_enough_trades_no_curve_fit_warning(self):
        attempts = [
            {
                "turn": 2,
                "sharpe": 0.3,
                "failure_mode": "low_sharpe",
                "action": "modify_params",
                "hypothesis": "test",
                "params_used": {},
                "delta_from_prev": "Changed",
                "num_trades": 25,
            }
        ]
        result = format_previous_attempts_section(attempts)
        assert "CURVE-FITTING RISK" not in result

    def test_regime_fragility_has_guidance(self):
        attempts = [
            {
                "turn": 4,
                "sharpe": 1.5,
                "failure_mode": "regime_fragility",
                "action": "add_filter",
                "hypothesis": "test",
                "params_used": {},
                "delta_from_prev": "Added regime filter",
                "num_trades": 20,
            }
        ]
        result = format_previous_attempts_section(attempts)
        assert "REGIME WARNING" in result
        assert "specific market conditions" in result
        assert "regime detection" in result
        assert "more robust indicator" in result

    def test_multiple_attempts_mixed_failures(self):
        attempts = [
            {
                "turn": 1,
                "sharpe": 2.0,
                "failure_mode": "too_few_trades_for_validation",
                "action": "novel",
                "hypothesis": "test 1",
                "params_used": {"rsi": 14},
                "delta_from_prev": "Initial",
                "num_trades": 3,
            },
            {
                "turn": 2,
                "sharpe": 0.5,
                "failure_mode": "low_sharpe",
                "action": "modify_params",
                "hypothesis": "test 2",
                "params_used": {"rsi": 20},
                "delta_from_prev": "Widened RSI",
                "num_trades": 5,
            },
            {
                "turn": 3,
                "sharpe": 1.8,
                "failure_mode": "validation_failed",
                "action": "change_entry_logic",
                "hypothesis": "test 3",
                "params_used": {"rsi": 30},
                "delta_from_prev": "Changed entry",
                "num_trades": 40,
                "validation": {
                    "avg_test_sharpe": 0.1,
                    "windows_passed": 1,
                    "num_windows": 6,
                    "window_results": [
                        {"window": 1, "train_sharpe": 2.0, "test_sharpe": 1.0, "degradation": 0.5, "passed": True},
                    ],
                },
            },
        ]
        result = format_previous_attempts_section(attempts)
        # Turn 1: too_few_trades
        assert "Only 3 trades" in result
        # Turn 2: low_sharpe curve-fit warning (5 < 10)
        assert "CURVE-FITTING RISK" in result
        # Turn 3: validation_failed with window breakdown
        assert "FAILED out-of-sample validation" in result
        assert "Rolling Walk-Forward Window Breakdown" in result

    def test_basic_fields_present(self):
        attempts = [
            {
                "turn": 1,
                "sharpe": 0.5,
                "failure_mode": "low_sharpe",
                "action": "novel",
                "hypothesis": "RSI mean reversion",
                "params_used": {"rsi_period": 14, "oversold": 30},
                "delta_from_prev": "Initial",
                "num_trades": 50,
            }
        ]
        result = format_previous_attempts_section(attempts)
        assert "Turn 1: Sharpe 0.50" in result
        assert "Failure: low_sharpe" in result
        assert "Action: novel" in result
        assert "Hypothesis: \"RSI mean reversion\"" in result
        assert "rsi_period" in result
