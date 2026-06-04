"""Tests for crabquant.refinement.sharpe_diagnosis — Sharpe Root Cause Analyzer.

Tests the diagnose_low_sharpe() function which analyzes backtest metrics
to identify specific root causes of low Sharpe and provide actionable guidance.
"""

import pytest

from crabquant.refinement.sharpe_diagnosis import diagnose_low_sharpe


class TestDiagnoseLowSharpe_Basic:
    """Basic behavior tests."""

    def test_returns_empty_string_when_no_metrics(self):
        result = diagnose_low_sharpe()
        assert result == ""

    def test_returns_empty_for_zero_trades(self):
        result = diagnose_low_sharpe(sharpe_ratio=0.3, total_trades=0)
        assert result == ""

    def test_returns_empty_for_single_trade(self):
        result = diagnose_low_sharpe(sharpe_ratio=0.3, total_trades=1)
        assert result == ""

    def test_header_contains_key_metrics(self):
        result = diagnose_low_sharpe(
            sharpe_ratio=0.5,
            win_rate=0.40,
            profit_factor=0.8,
            total_return_pct=0.05,
            max_drawdown_pct=-0.10,
            total_trades=20,
        )
        assert "🔍 Sharpe Root Cause Analysis" in result
        assert "Current Sharpe: 0.50" in result
        assert "Target: 1.5" in result
        assert "Win Rate: 40%" in result
        assert "Profit Factor: 0.80" in result


class TestDiagnoseLowSharpe_LosingMoney:
    """Strategy with negative return."""

    def test_negative_return_with_enough_trades(self):
        result = diagnose_low_sharpe(
            total_return_pct=-0.05,
            total_trades=10,
        )
        assert "Strategy is losing money" in result
        assert "Flip the signal direction" in result
        assert "fundamentally wrong" in result

    def test_negative_return_with_few_trades_no_diagnosis(self):
        """Not enough trades to be confident about losing."""
        result = diagnose_low_sharpe(
            total_return_pct=-0.05,
            total_trades=3,
        )
        assert "losing money" not in result


class TestDiagnoseLowSharpe_VeryLowWinRate:
    """Win rate < 35% — noisy/false entries."""

    def test_very_low_win_rate(self):
        result = diagnose_low_sharpe(
            win_rate=0.25,
            total_trades=20,
            total_return_pct=0.02,
        )
        assert "Very low win rate (25%)" in result
        assert "TREND FILTER" in result
        assert "WIDEN thresholds" in result
        assert "VOLATILITY FILTER" in result

    def test_very_low_win_rate_needs_minimum_trades(self):
        """Below 10 trades → no win rate diagnosis."""
        result = diagnose_low_sharpe(
            win_rate=0.20,
            total_trades=5,
            total_return_pct=0.01,
        )
        assert "Very low win rate" not in result


class TestDiagnoseLowSharpe_ModerateLowWinRate:
    """Win rate 35-45% — weak signal."""

    def test_moderate_low_win_rate(self):
        result = diagnose_low_sharpe(
            win_rate=0.40,
            total_trades=20,
            total_return_pct=0.02,
        )
        assert "Low win rate (40%)" in result
        assert "confirmation filter" in result


class TestDiagnoseLowSharpe_ProfitFactor:
    """Profit factor diagnostics."""

    def test_profit_factor_below_1(self):
        result = diagnose_low_sharpe(
            profit_factor=0.7,
            total_trades=15,
            win_rate=0.50,
        )
        assert "Profit factor 0.70 < 1.0" in result
        assert "STOP-LOSS" in result
        assert "TRAILING STOPS" in result
        assert "TIME STOP" in result

    def test_marginal_profit_factor(self):
        result = diagnose_low_sharpe(
            profit_factor=1.15,
            total_trades=20,
            win_rate=0.50,
        )
        assert "Marginal profit factor (1.15)" in result
        assert "EXIT logic" in result

    def test_zero_profit_factor_no_diagnosis(self):
        """profit_factor == 0 → no diagnosis."""
        result = diagnose_low_sharpe(
            profit_factor=0.0,
            total_trades=15,
        )
        assert "Profit factor" not in result


class TestDiagnoseLowSharpe_ExcessiveDrawdown:
    """Drawdown too large relative to return."""

    def test_high_dd_return_ratio(self):
        result = diagnose_low_sharpe(
            total_return_pct=0.05,
            max_drawdown_pct=-0.25,
            total_trades=20,
            win_rate=0.50,
            profit_factor=1.2,
        )
        assert "Drawdown too large relative to return" in result
        assert "DD/return ratio" in result
        assert "VOLATILITY STOP" in result

    def test_low_dd_return_ratio_no_diagnosis(self):
        """Return is high enough relative to drawdown."""
        result = diagnose_low_sharpe(
            total_return_pct=0.30,
            max_drawdown_pct=-0.15,
            total_trades=20,
            win_rate=0.50,
            profit_factor=1.5,
        )
        assert "Drawdown too large" not in result

    def test_negative_return_no_dd_diagnosis(self):
        """DD diagnosis only when total_return > 0."""
        result = diagnose_low_sharpe(
            total_return_pct=-0.05,
            max_drawdown_pct=-0.30,
            total_trades=10,
        )
        assert "Drawdown too large relative to return" not in result


class TestDiagnoseLowSharpe_Sortino:
    """Sortino << Sharpe → concentrated downside."""

    def test_sortino_much_lower_than_sharpe(self):
        result = diagnose_low_sharpe(
            sharpe_ratio=1.5,
            sortino_ratio=0.5,
            total_trades=20,
            win_rate=0.55,
            profit_factor=1.5,
        )
        assert "Sortino (0.50) much lower than Sharpe (1.50)" in result
        assert "DOWNSIDE PROTECTION" in result

    def test_sortino_similar_to_sharpe_no_diagnosis(self):
        result = diagnose_low_sharpe(
            sharpe_ratio=1.0,
            sortino_ratio=0.9,
            total_trades=20,
            win_rate=0.50,
            profit_factor=1.3,
        )
        assert "Sortino" not in result

    def test_zero_sortino_no_diagnosis(self):
        result = diagnose_low_sharpe(
            sharpe_ratio=1.0,
            sortino_ratio=0.0,
            total_trades=20,
        )
        assert "Sortino" not in result


class TestDiagnoseLowSharpe_InconsistentReturns:
    """Positive return but low Sharpe — inconsistent."""

    def test_positive_return_low_sharpe(self):
        result = diagnose_low_sharpe(
            total_return_pct=0.15,
            sharpe_ratio=0.3,
            win_rate=0.50,
            total_trades=20,
            profit_factor=1.2,
        )
        assert "Returns are positive (15.0%)" in result
        assert "inconsistent" in result
        assert "CONVICTION FILTER" in result


class TestDiagnoseLowSharpe_Whipsaw:
    """Short holding periods + low win rate → whipsaw."""

    def test_whipsaw_pattern(self):
        result = diagnose_low_sharpe(
            avg_holding_bars=2.5,
            win_rate=0.35,
            total_trades=25,
            total_return_pct=0.01,
        )
        assert "Whipsaw pattern" in result
        assert "2.5 bars" in result
        assert "INCREASE indicator lengths" in result

    def test_short_hold_good_win_rate_no_whipsaw(self):
        """Short holds with good win rate is not whipsaw."""
        result = diagnose_low_sharpe(
            avg_holding_bars=3.0,
            win_rate=0.55,
            total_trades=30,
            total_return_pct=0.08,
        )
        assert "Whipsaw" not in result

    def test_no_avg_holding_bars_no_whipsaw(self):
        """Without avg_holding_bars, whipsaw can't be diagnosed."""
        result = diagnose_low_sharpe(
            win_rate=0.30,
            total_trades=25,
        )
        assert "Whipsaw" not in result


class TestDiagnoseLowSharpe_LongHolds:
    """Long holding periods + low return → no edge captured."""

    def test_long_holds_low_return(self):
        result = diagnose_low_sharpe(
            avg_holding_bars=50,
            total_return_pct=0.03,
            total_trades=10,
            win_rate=0.50,
        )
        assert "Long holds with little gain" in result
        assert "50 bars" in result
        assert "ENTRY TIMING" in result
        assert "TIME STOP" in result


class TestDiagnoseLowSharpe_ByYear:
    """Sharpe-by-year inconsistency."""

    def test_all_years_negative(self):
        result = diagnose_low_sharpe(
            sharpe_by_year={2020: -0.5, 2021: -1.0, 2022: -0.3},
            total_trades=20,
            win_rate=0.40,
        )
        assert "loses in ALL years" in result
        assert "completely different approach" in result

    def test_one_good_year_out_of_three(self):
        result = diagnose_low_sharpe(
            sharpe_by_year={2020: 1.5, 2021: -0.5, 2022: -0.8},
            total_trades=20,
            win_rate=0.40,
        )
        assert "only works in 1/3 years" in result
        assert "REGIME DETECTION" in result

    def test_two_good_years_not_flagged(self):
        """2 out of 3 years positive → not flagged as regime-dependent."""
        result = diagnose_low_sharpe(
            sharpe_by_year={2020: 1.5, 2021: 0.8, 2022: -0.5},
            total_trades=20,
            win_rate=0.45,
        )
        assert "only works in 1/" not in result

    def test_single_year_no_diagnosis(self):
        """Need at least 2 years of data."""
        result = diagnose_low_sharpe(
            sharpe_by_year={2022: -0.5},
            total_trades=20,
        )
        assert "loses in ALL" not in result
        assert "only works in" not in result


class TestDiagnoseLowSharpe_GapAnalysis:
    """When no specific root cause found, gap analysis provides guidance."""

    def test_small_gap(self):
        result = diagnose_low_sharpe(
            sharpe_ratio=1.3,
            sharpe_target=1.5,
            total_trades=20,
            win_rate=0.50,
            profit_factor=1.4,
            total_return_pct=0.10,
        )
        assert "Close to target" in result
        assert "only 0.20 below target" in result

    def test_moderate_gap(self):
        result = diagnose_low_sharpe(
            sharpe_ratio=0.8,
            sharpe_target=1.5,
            total_trades=20,
            win_rate=0.50,
            profit_factor=1.3,
            total_return_pct=0.10,
        )
        assert "Moderate Sharpe gap" in result

    def test_large_gap(self):
        result = diagnose_low_sharpe(
            sharpe_ratio=0.3,
            sharpe_target=1.5,
            total_trades=20,
            win_rate=0.55,
            profit_factor=1.5,
            total_return_pct=0.03,
        )
        assert "Large Sharpe gap" in result
        assert "FULL REWRITE" in result


class TestDiagnoseLowSharpe_MultipleDiagnoses:
    """Multiple root causes can be diagnosed simultaneously."""

    def test_low_win_rate_plus_bad_profit_factor(self):
        result = diagnose_low_sharpe(
            win_rate=0.30,
            profit_factor=0.6,
            total_trades=20,
            total_return_pct=0.02,
        )
        # Both diagnoses should be present
        assert "Very low win rate" in result
        assert "Profit factor" in result
        # Should have two separate diagnosis blocks
        assert result.count("**Root cause:") == 2

    def test_whipsaw_plus_low_win_rate(self):
        result = diagnose_low_sharpe(
            win_rate=0.30,
            avg_holding_bars=3.0,
            total_trades=25,
            total_return_pct=0.01,
        )
        assert "Very low win rate" in result
        assert "Whipsaw pattern" in result


class TestDiagnoseLowSharpe_IntegrationWithBuildFailureGuidance:
    """Test that build_failure_guidance includes sharpe diagnosis."""

    def test_low_sharpe_includes_diagnosis(self):
        from crabquant.refinement.prompts import build_failure_guidance

        result = build_failure_guidance(
            "low_sharpe",
            total_trades=30,
            sharpe_ratio=0.4,
            sharpe_target=1.5,
            total_return_pct=0.02,
            max_drawdown_pct=-0.10,
            win_rate=0.30,
            profit_factor=0.7,
            sortino_ratio=0.2,
            calmar_ratio=0.2,
        )
        assert "Improve Sharpe Ratio" in result
        assert "🔍 Sharpe Root Cause Analysis" in result
        assert "Very low win rate" in result
        assert "Profit factor" in result

    def test_low_sharpe_few_trades_includes_both_warnings(self):
        from crabquant.refinement.prompts import build_failure_guidance

        result = build_failure_guidance(
            "low_sharpe",
            total_trades=5,
            sharpe_ratio=0.3,
            win_rate=0.40,
            profit_factor=0.8,
        )
        assert "CRITICAL" in result
        assert "5 trades" in result
        # Still includes diagnosis
        assert "🔍 Sharpe Root Cause Analysis" in result

    def test_non_low_sharpe_unchanged(self):
        """Other failure modes should not include sharpe diagnosis."""
        from crabquant.refinement.prompts import build_failure_guidance

        result = build_failure_guidance(
            "regime_fragility",
            total_trades=30,
            sharpe_ratio=0.4,
            win_rate=0.30,
        )
        assert "🔍 Sharpe Root Cause Analysis" not in result
        assert "Regime-Dependent" in result

    def test_unknown_failure_mode_empty(self):
        from crabquant.refinement.prompts import build_failure_guidance

        result = build_failure_guidance("nonexistent_mode")
        assert result == ""

    def test_backward_compatible_positional_args(self):
        """Original positional call signature still works."""
        from crabquant.refinement.prompts import build_failure_guidance

        # Should not raise — uses defaults for new params
        result = build_failure_guidance("low_sharpe", 20)
        assert "Improve Sharpe Ratio" in result
        # Without metrics, diagnosis has minimal info but shouldn't crash
        # Gap analysis will still fire based on default sharpe 0.0 vs target 1.5
