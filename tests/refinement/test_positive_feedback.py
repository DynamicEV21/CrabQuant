"""Tests for the positive feedback analyzer."""

import pytest

from crabquant.refinement.positive_feedback import (
    PositiveFeedback,
    analyze_positive_feedback,
    format_positive_feedback_for_prompt,
    _build_regression_risk,
)


class TestAnalyzePositiveFeedback:
    """Test analyze_positive_feedback with various metric combinations."""

    def test_all_zeros_returns_empty_strengths(self):
        """When all metrics are zero, no strengths should be identified."""
        fb = analyze_positive_feedback()
        assert fb.strengths == []
        assert fb.preserve_warnings == []
        assert "no clearly positive attributes" in fb.overall_assessment

    def test_positive_returns_identified(self):
        """Strategies with positive returns should get a strength."""
        fb = analyze_positive_feedback(total_return_pct=0.15)
        assert any("Positive returns" in s for s in fb.strengths)
        assert any("PRESERVE" in w for w in fb.preserve_warnings)

    def test_moderate_positive_returns(self):
        """Small positive returns should be noted but not get preserve warning."""
        fb = analyze_positive_feedback(total_return_pct=0.03)
        assert any("Modest positive returns" in s for s in fb.strengths)
        assert not fb.preserve_warnings

    def test_negative_returns_no_return_strength(self):
        """Negative returns should not get a return strength."""
        fb = analyze_positive_feedback(total_return_pct=-0.05)
        assert not any("return" in s.lower() for s in fb.strengths)

    def test_good_sharpe_half_target(self):
        """Sharpe at 50% of target should be noted."""
        fb = analyze_positive_feedback(sharpe_ratio=0.75, sharpe_target=1.5)
        assert any("50% of target" in s for s in fb.strengths)

    def test_strong_sharpe_preserve_warning(self):
        """Sharpe > 0.5 should trigger preserve warning."""
        fb = analyze_positive_feedback(sharpe_ratio=1.0)
        assert any("PRESERVE" in w and "Sharpe" in w for w in fb.preserve_warnings)

    def test_healthy_win_rate(self):
        """Win rate in 45-70% range should be identified as healthy."""
        fb = analyze_positive_feedback(win_rate=0.55)
        assert any("Healthy win rate" in s for s in fb.strengths)
        assert any("win rate" in w.lower() for w in fb.preserve_warnings)

    def test_high_win_rate(self):
        """Win rate > 70% should warn about potential curve-fitting."""
        fb = analyze_positive_feedback(win_rate=0.80)
        assert any("High win rate" in s for s in fb.strengths)

    def test_low_win_rate_no_strength(self):
        """Win rate < 35% should not get a strength."""
        fb = analyze_positive_feedback(win_rate=0.30)
        assert not any("win rate" in s.lower() for s in fb.strengths)

    def test_acceptable_win_rate(self):
        """Win rate 35-45% should be noted as acceptable."""
        fb = analyze_positive_feedback(win_rate=0.40)
        assert any("Acceptable win rate" in s for s in fb.strengths)

    def test_strong_profit_factor(self):
        """Profit factor > 1.5 should be identified."""
        fb = analyze_positive_feedback(profit_factor=2.0)
        assert any("Strong profit factor" in s for s in fb.strengths)
        assert any("PRESERVE" in w for w in fb.preserve_warnings)

    def test_positive_profit_factor(self):
        """Profit factor 1.0-1.5 should be noted."""
        fb = analyze_positive_feedback(profit_factor=1.2)
        assert any("Positive profit factor" in s for s in fb.strengths)

    def test_bad_profit_factor_no_strength(self):
        """Profit factor < 1.0 should not get a strength."""
        fb = analyze_positive_feedback(profit_factor=0.8)
        assert not any("profit factor" in s.lower() for s in fb.strengths)

    def test_good_sortino(self):
        """Sortino > 1.0 should be identified."""
        fb = analyze_positive_feedback(sortino_ratio=1.5)
        assert any("Good Sortino" in s for s in fb.strengths)
        assert any("downside" in w.lower() for w in fb.preserve_warnings)

    def test_controlled_drawdown(self):
        """Drawdown > -15% should be identified as controlled."""
        fb = analyze_positive_feedback(max_drawdown_pct=-0.10)
        assert any("Controlled drawdown" in s for s in fb.strengths)

    def test_moderate_drawdown(self):
        """Drawdown -15% to -25% should be noted as moderate."""
        fb = analyze_positive_feedback(max_drawdown_pct=-0.20)
        assert any("Moderate drawdown" in s for s in fb.strengths)

    def test_severe_drawdown_no_strength(self):
        """Drawdown < -25% should not get a drawdown strength."""
        fb = analyze_positive_feedback(max_drawdown_pct=-0.35)
        assert not any("drawdown" in s.lower() for s in fb.strengths)

    def test_good_trade_frequency(self):
        """20-100 trades should be identified as good."""
        fb = analyze_positive_feedback(total_trades=50)
        assert any("Good trade frequency" in s for s in fb.strengths)

    def test_low_trade_frequency(self):
        """10-19 trades should be noted as adequate."""
        fb = analyze_positive_feedback(total_trades=15)
        assert any("Adequate trade frequency" in s for s in fb.strengths)

    def test_high_trade_frequency(self):
        """> 100 trades should note potential cost issues."""
        fb = analyze_positive_feedback(total_trades=200)
        assert any("High trade frequency" in s for s in fb.strengths)

    def test_very_few_trades_no_strength(self):
        """< 10 trades should not get a frequency strength."""
        fb = analyze_positive_feedback(total_trades=5)
        assert not any("trade" in s.lower() for s in fb.strengths)

    def test_reasonable_holding_period(self):
        """3-30 bars holding period should be noted."""
        fb = analyze_positive_feedback(avg_holding_bars=10.0)
        assert any("holding period" in s.lower() for s in fb.strengths)

    def test_long_holding_period(self):
        """> 30 bars should be noted as longer."""
        fb = analyze_positive_feedback(avg_holding_bars=50.0)
        assert any("Longer holding period" in s for s in fb.strengths)

    def test_no_holding_period(self):
        """None holding period should not crash."""
        fb = analyze_positive_feedback(avg_holding_bars=None)
        assert not any("holding" in s.lower() for s in fb.strengths)

    def test_profitable_every_year(self):
        """All positive years should get highest praise."""
        fb = analyze_positive_feedback(
            sharpe_by_year={"2021": 1.0, "2022": 0.5, "2023": 0.8}
        )
        assert any("Profitable every year" in s for s in fb.strengths)
        assert any("PRESERVE" in w and "ALL years" in w for w in fb.preserve_warnings)

    def test_profitable_most_years(self):
        """70%+ positive years should be noted."""
        fb = analyze_positive_feedback(
            sharpe_by_year={"2021": 1.0, "2022": -0.5, "2023": 0.8, "2024": 0.3}
        )
        assert any("Profitable in most years" in s for s in fb.strengths)

    def test_profitable_one_year(self):
        """1+ positive years should identify best year."""
        fb = analyze_positive_feedback(
            sharpe_by_year={"2021": 1.5, "2022": -1.0}
        )
        assert any("Best year: 2021" in s for s in fb.strengths)

    def test_all_negative_years(self):
        """All negative years should not get a year consistency strength."""
        fb = analyze_positive_feedback(
            sharpe_by_year={"2021": -0.5, "2022": -1.0}
        )
        assert not any("year" in s.lower() for s in fb.strengths)

    def test_single_year_no_consistency(self):
        """Single year should not trigger consistency analysis."""
        fb = analyze_positive_feedback(
            sharpe_by_year={"2023": 1.0}
        )
        assert not any("year" in s.lower() for s in fb.strengths)

    def test_good_calmar_ratio(self):
        """Calmar > 1.0 should be noted."""
        fb = analyze_positive_feedback(calmar_ratio=1.5)
        assert any("Good Calmar" in s for s in fb.strengths)

    def test_multiple_strengths_overall_assessment(self):
        """3+ strengths should get the strong assessment."""
        fb = analyze_positive_feedback(
            total_return_pct=0.15,
            sharpe_ratio=1.0,
            win_rate=0.55,
            profit_factor=2.0,
        )
        assert "3 positive attributes" in fb.overall_assessment or "4 positive attributes" in fb.overall_assessment

    def test_one_strength_overall_assessment(self):
        """1 strength should get the moderate assessment."""
        fb = analyze_positive_feedback(total_return_pct=0.15)
        assert "1 positive attribute" in fb.overall_assessment


class TestBuildRegressionRisk:
    """Test the regression risk warning system."""

    def test_low_sharpe_with_positive_returns(self):
        """low_sharpe + positive returns should warn about changing signal."""
        risk = _build_regression_risk(
            failure_mode="low_sharpe",
            has_positive_returns=True,
            has_good_win_rate=False,
            has_profit_factor=False,
            sharpe_ratio=0.3,
            preserve_warnings=[],
        )
        assert "REGRESSION RISK" in risk
        assert "positive returns" in risk.lower()
        assert "trend filter" in risk.lower()

    def test_regime_fragility_with_positive_returns(self):
        """regime_fragility + positive returns should suggest regime gate."""
        risk = _build_regression_risk(
            failure_mode="regime_fragility",
            has_positive_returns=True,
            has_good_win_rate=False,
            has_profit_factor=False,
            sharpe_ratio=0.5,
            preserve_warnings=[],
        )
        assert "REGRESSION RISK" in risk
        assert "regime gate" in risk.lower()

    def test_too_few_trades_with_good_win_rate(self):
        """too_few_trades + good win rate should suggest loosening."""
        risk = _build_regression_risk(
            failure_mode="too_few_trades",
            has_positive_returns=False,
            has_good_win_rate=True,
            has_profit_factor=False,
            sharpe_ratio=0.0,
            preserve_warnings=[],
        )
        assert "REGRESSION RISK" in risk
        assert "LOOSEN" in risk

    def test_too_few_trades_with_profit_factor(self):
        """too_few_trades + good profit factor should suggest widening."""
        risk = _build_regression_risk(
            failure_mode="too_few_trades",
            has_positive_returns=False,
            has_good_win_rate=False,
            has_profit_factor=True,
            sharpe_ratio=0.0,
            preserve_warnings=[],
        )
        assert "REGRESSION RISK" in risk
        assert "profitable" in risk.lower()

    def test_excessive_drawdown_with_good_sharpe(self):
        """excessive_drawdown + good Sharpe should suggest risk management."""
        risk = _build_regression_risk(
            failure_mode="excessive_drawdown",
            has_positive_returns=True,
            has_good_win_rate=True,
            has_profit_factor=True,
            sharpe_ratio=1.0,
            preserve_warnings=[],
        )
        assert "REGRESSION RISK" in risk
        assert "stop loss" in risk.lower()

    def test_validation_failed_with_good_sharpe(self):
        """validation_failed + good Sharpe should suggest simplification."""
        risk = _build_regression_risk(
            failure_mode="validation_failed",
            has_positive_returns=True,
            has_good_win_rate=True,
            has_profit_factor=True,
            sharpe_ratio=1.2,
            preserve_warnings=[],
        )
        assert "REGRESSION RISK" in risk
        assert "OVERFIT" in risk
        assert "1.2" in risk

    def test_preserve_warnings_fallback(self):
        """If no specific risk but preserve warnings exist, should warn generally."""
        risk = _build_regression_risk(
            failure_mode="flat_signal",
            has_positive_returns=False,
            has_good_win_rate=False,
            has_profit_factor=False,
            sharpe_ratio=0.0,
            preserve_warnings=["PRESERVE: keep this thing"],
        )
        assert "REGRESSION RISK" in risk
        assert "TARGETED changes" in risk

    def test_no_risk_no_warnings(self):
        """No specific risk and no warnings should return empty string."""
        risk = _build_regression_risk(
            failure_mode="unknown",
            has_positive_returns=False,
            has_good_win_rate=False,
            has_profit_factor=False,
            sharpe_ratio=0.0,
            preserve_warnings=[],
        )
        assert risk == ""


class TestFormatPositiveFeedbackForPrompt:
    """Test the prompt formatting function."""

    def test_empty_strengths_returns_empty(self):
        """No strengths should return empty string."""
        fb = PositiveFeedback(
            strengths=[],
            preserve_warnings=[],
            overall_assessment="nothing good",
            regression_risk="",
        )
        result = format_positive_feedback_for_prompt(fb)
        assert result == ""

    def test_single_strength_formatted(self):
        """Single strength should be formatted as list item."""
        fb = PositiveFeedback(
            strengths=["**Good win rate** (55%)"],
            preserve_warnings=[],
            overall_assessment="decent",
            regression_risk="",
        )
        result = format_positive_feedback_for_prompt(fb)
        assert "### ✅ What's Working" in result
        assert "Good win rate" in result
        assert "decent" in result

    def test_preserve_warnings_section(self):
        """Preserve warnings should get a sub-header."""
        fb = PositiveFeedback(
            strengths=["something good"],
            preserve_warnings=["PRESERVE: keep this"],
            overall_assessment="ok",
            regression_risk="",
        )
        result = format_positive_feedback_for_prompt(fb)
        assert "Preservation Rules" in result
        assert "PRESERVE: keep this" in result

    def test_regression_risk_included(self):
        """Regression risk should be included when present."""
        fb = PositiveFeedback(
            strengths=["something"],
            preserve_warnings=[],
            overall_assessment="ok",
            regression_risk="⚠️ REGRESSION RISK: don't break it",
        )
        result = format_positive_feedback_for_prompt(fb)
        assert "REGRESSION RISK" in result

    def test_full_output_structure(self):
        """Verify the full output has all expected sections."""
        fb = PositiveFeedback(
            strengths=["s1", "s2", "s3"],
            preserve_warnings=["p1"],
            overall_assessment="good strategy",
            regression_risk="⚠️ REGRESSION RISK: be careful",
        )
        result = format_positive_feedback_for_prompt(fb)
        # Check section order
        assert result.index("### ✅ What's Working") < result.index("Preservation Rules")
        assert result.index("Preservation Rules") < result.index("good strategy")
        assert result.index("good strategy") < result.index("REGRESSION RISK")


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""

    def test_nearly_there_strategy(self):
        """Strategy that's close to target — should get lots of positive feedback."""
        fb = analyze_positive_feedback(
            sharpe_ratio=1.2,
            sharpe_target=1.5,
            total_return_pct=0.25,
            max_drawdown_pct=-0.12,
            win_rate=0.58,
            profit_factor=1.8,
            sortino_ratio=1.5,
            calmar_ratio=2.0,
            total_trades=45,
            avg_holding_bars=12.0,
            sharpe_by_year={"2021": 1.0, "2022": 0.5, "2023": 1.2, "2024": 0.8},
            failure_mode="low_sharpe",
        )
        # Should identify many strengths
        assert len(fb.strengths) >= 5
        # Should have preserve warnings
        assert len(fb.preserve_warnings) >= 3
        # Should have regression risk (low_sharpe + positive returns)
        assert "REGRESSION RISK" in fb.regression_risk

    def test_bad_strategy_minimal_feedback(self):
        """Terrible strategy — should get minimal positive feedback."""
        fb = analyze_positive_feedback(
            sharpe_ratio=-0.5,
            sharpe_target=1.5,
            total_return_pct=-0.20,
            max_drawdown_pct=-0.40,
            win_rate=0.30,
            profit_factor=0.6,
            sortino_ratio=-0.3,
            total_trades=5,
            failure_mode="low_sharpe",
        )
        # Should have very few or no strengths
        assert len(fb.strengths) <= 1
        # Should not have regression risk
        assert fb.regression_risk == ""

    def test_regime_fragility_good_base(self):
        """Strategy with regime fragility but good base metrics."""
        fb = analyze_positive_feedback(
            sharpe_ratio=0.8,
            sharpe_target=1.5,
            total_return_pct=0.20,
            max_drawdown_pct=-0.18,
            win_rate=0.52,
            profit_factor=1.6,
            total_trades=35,
            sharpe_by_year={"2021": 2.0, "2022": -1.5, "2023": 0.5},
            failure_mode="regime_fragility",
        )
        assert any("Best year: 2021" in s for s in fb.strengths)
        assert "REGRESSION RISK" in fb.regression_risk
        assert "regime gate" in fb.regression_risk.lower()

    def test_too_few_trades_good_quality(self):
        """Strategy with few trades but good quality per trade."""
        fb = analyze_positive_feedback(
            sharpe_ratio=2.0,
            sharpe_target=1.5,
            total_return_pct=0.30,
            win_rate=0.65,
            profit_factor=2.5,
            total_trades=8,
            failure_mode="too_few_trades",
        )
        # Despite low trades, should recognize the quality
        assert any("Positive returns" in s for s in fb.strengths)
        assert any("Strong profit factor" in s for s in fb.strengths)
        assert "REGRESSION RISK" in fb.regression_risk
        assert "LOOSEN" in fb.regression_risk
