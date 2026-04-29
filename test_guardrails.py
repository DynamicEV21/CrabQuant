"""
Direct unit tests for crabquant.guardrails module.

Tests GuardrailConfig presets, check_guardrails violations/warnings/score,
and OverfittingDetector edge cases. These complement the existing
tests/refinement/test_guardrails_integration.py which tests the wrapper layer.
"""

import pytest

from crabquant.guardrails import (
    GuardrailConfig,
    GuardrailReport,
    check_guardrails,
    OverfittingDetector,
)
from crabquant.engine.backtest import BacktestResult


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_result(**overrides) -> BacktestResult:
    """Factory with sensible passing defaults (aggressive preset)."""
    defaults = dict(
        ticker="SPY",
        strategy_name="test",
        iteration=0,
        sharpe=1.8,
        total_return=0.20,
        max_drawdown=-0.10,
        win_rate=0.55,
        num_trades=50,
        avg_trade_return=0.02,
        calmar_ratio=1.5,
        sortino_ratio=2.0,
        profit_factor=1.5,
        avg_holding_bars=5.0,
        best_trade=500.0,
        worst_trade=-200.0,
        passed=True,
        score=1.2,
        notes="ok",
        params={},
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def make_results(num=5, **common_overrides):
    """Create multiple BacktestResults with varying returns."""
    results = []
    for i in range(num):
        results.append(make_result(
            total_return=0.05 * (i + 1),
            iteration=i,
            **common_overrides,
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GuardrailConfig — preset factories
# ═══════════════════════════════════════════════════════════════════════════════

class TestGuardrailConfigDefaults:
    def test_default_values(self):
        cfg = GuardrailConfig()
        assert cfg.max_drawdown == 0.25
        assert cfg.min_trades == 5
        assert cfg.min_win_rate == 0.30
        assert cfg.min_sharpe == 1.0
        assert cfg.min_return == 0.05
        assert cfg.max_holding_days == 100
        assert cfg.min_profit_factor == 1.0

    def test_custom_values(self):
        cfg = GuardrailConfig(max_drawdown=0.5, min_trades=1)
        assert cfg.max_drawdown == 0.5
        assert cfg.min_trades == 1


class TestGuardrailConfigConservative:
    def test_values(self):
        cfg = GuardrailConfig.conservative()
        assert cfg.max_drawdown == 0.10
        assert cfg.min_trades == 30
        assert cfg.min_sharpe == 1.5
        assert cfg.min_win_rate == 0.45
        assert cfg.min_return == 0.10
        assert cfg.min_profit_factor == 1.5

    def test_stricter_than_default(self):
        cons = GuardrailConfig.conservative()
        default = GuardrailConfig()
        assert cons.min_trades > default.min_trades
        assert cons.min_sharpe > default.min_sharpe
        assert cons.max_drawdown < default.max_drawdown


class TestGuardrailConfigModerate:
    def test_values(self):
        cfg = GuardrailConfig.moderate()
        assert cfg.max_drawdown == 0.20
        assert cfg.min_trades == 15
        assert cfg.min_sharpe == 1.0
        assert cfg.min_win_rate == 0.35
        assert cfg.min_return == 0.08

    def test_between_conservative_and_aggressive(self):
        mod = GuardrailConfig.moderate()
        cons = GuardrailConfig.conservative()
        agg = GuardrailConfig.aggressive()
        assert cons.min_trades > mod.min_trades > agg.min_trades


class TestGuardrailConfigAggressive:
    def test_values(self):
        cfg = GuardrailConfig.aggressive()
        assert cfg.max_drawdown == 0.30
        assert cfg.min_trades == 5
        assert cfg.min_sharpe == 0.5
        assert cfg.min_win_rate == 0.25
        assert cfg.min_return == 0.05

    def test_most_permissive(self):
        agg = GuardrailConfig.aggressive()
        cons = GuardrailConfig.conservative()
        assert agg.max_drawdown > cons.max_drawdown
        assert agg.min_trades < cons.min_trades


# ═══════════════════════════════════════════════════════════════════════════════
# GuardrailReport
# ═══════════════════════════════════════════════════════════════════════════════

class TestGuardrailReport:
    def test_passed_report(self):
        r = GuardrailReport(passed=True, violations=[], warnings=[], score_adjustment=0.0)
        assert r.passed is True
        assert r.violations == []
        assert r.warnings == []
        assert r.score_adjustment == 0.0

    def test_failed_report(self):
        r = GuardrailReport(
            passed=False,
            violations=["Too few trades"],
            warnings=["Low trade count"],
            score_adjustment=-0.1,
        )
        assert r.passed is False
        assert len(r.violations) == 1
        assert len(r.warnings) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# check_guardrails — passing cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckGuardrailsPassing:
    def test_perfect_result_passes(self):
        result = make_result(sharpe=5.0, num_trades=100, total_return=0.50,
                             win_rate=0.70, profit_factor=3.0, max_drawdown=-0.05)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.passed
        assert report.violations == []
        assert report.score_adjustment == 0.0

    def test_barely_passes_aggressive(self):
        """Result exactly at aggressive thresholds should pass."""
        result = make_result(
            sharpe=0.5, num_trades=5, total_return=0.05,
            win_rate=0.25, profit_factor=1.0, max_drawdown=-0.30,
        )
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.passed

    def test_good_result_default_config(self):
        result = make_result()
        report = check_guardrails(result, GuardrailConfig())
        assert report.passed


# ═══════════════════════════════════════════════════════════════════════════════
# check_guardrails — individual violations
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckGuardrailsViolations:
    def test_few_trades(self):
        result = make_result(num_trades=2)
        report = check_guardrails(result, GuardrailConfig())
        assert not report.passed
        assert any("Too few trades" in v for v in report.violations)

    def test_low_sharpe(self):
        result = make_result(sharpe=0.3)
        report = check_guardrails(result, GuardrailConfig())
        assert not report.passed
        assert any("Sharpe too low" in v for v in report.violations)

    def test_low_return(self):
        result = make_result(total_return=0.01)
        report = check_guardrails(result, GuardrailConfig())
        assert not report.passed
        assert any("Return too low" in v for v in report.violations)

    def test_low_win_rate(self):
        result = make_result(win_rate=0.10)
        report = check_guardrails(result, GuardrailConfig())
        assert not report.passed
        assert any("Win rate too low" in v for v in report.violations)

    def test_low_profit_factor(self):
        result = make_result(profit_factor=0.5)
        report = check_guardrails(result, GuardrailConfig())
        assert not report.passed
        assert any("Profit factor too low" in v for v in report.violations)

    def test_deep_drawdown(self):
        result = make_result(max_drawdown=-0.50)
        report = check_guardrails(result, GuardrailConfig())
        assert not report.passed
        assert any("Drawdown too deep" in v for v in report.violations)

    def test_long_holding_period(self):
        result = make_result(avg_holding_bars=200)
        report = check_guardrails(result, GuardrailConfig())
        assert not report.passed
        assert any("Holding too long" in v for v in report.violations)

    @pytest.mark.parametrize("attr,bad_val,key", [
        ("num_trades", 0, "Too few trades"),
        ("sharpe", -1.0, "Sharpe too low"),
        ("total_return", -0.5, "Return too low"),
        ("win_rate", 0.0, "Win rate too low"),
        ("profit_factor", 0.0, "Profit factor too low"),
        ("max_drawdown", -1.0, "Drawdown too deep"),
    ])
    def test_zero_or_negative_values(self, attr, bad_val, key):
        """Extreme bad values should produce violations."""
        result = make_result(**{attr: bad_val})
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert not report.passed
        assert any(key in v for v in report.violations)


# ═══════════════════════════════════════════════════════════════════════════════
# check_guardrails — warnings
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckGuardrailsWarnings:
    def test_low_trade_count_warning(self):
        """Trades between min_trades and 2*min_trades should warn."""
        result = make_result(num_trades=7)  # Default min_trades=5, so 7 < 10
        report = check_guardrails(result, GuardrailConfig())
        assert any("Low trade count" in w for w in report.warnings)

    def test_sharpe_near_threshold_warning(self):
        """Sharpe between min_sharpe and 1.2*min_sharpe should warn."""
        result = make_result(sharpe=1.1)  # Default min_sharpe=1.0, so 1.1 < 1.2
        report = check_guardrails(result, GuardrailConfig())
        assert any("Sharpe near threshold" in w for w in report.warnings)

    def test_drawdown_approaching_warning(self):
        """Drawdown between 70% and 100% of max_drawdown should warn."""
        # Default max_drawdown=0.25. Range: -0.175 to -0.25
        result = make_result(max_drawdown=-0.20)
        report = check_guardrails(result, GuardrailConfig())
        assert any("approaching limit" in w for w in report.warnings)

    def test_win_rate_below_50_warning(self):
        """Win rate < 50% with trades > 0 should warn."""
        result = make_result(win_rate=0.45, num_trades=10)
        report = check_guardrails(result, GuardrailConfig())
        assert any("below 50%" in w for w in report.warnings)

    def test_no_warnings_for_strong_result(self):
        """Strong result should have no warnings."""
        result = make_result(
            sharpe=5.0, num_trades=100, max_drawdown=-0.02, win_rate=0.80,
        )
        report = check_guardrails(result, GuardrailConfig())
        assert report.warnings == []


# ═══════════════════════════════════════════════════════════════════════════════
# check_guardrails — score adjustment
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckGuardrailsScoreAdjustment:
    def test_zero_violations_zero_adjustment(self):
        result = make_result()
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.score_adjustment == 0.0

    def test_one_violation(self):
        result = make_result(sharpe=0.3)
        report = check_guardrails(result, GuardrailConfig())
        assert report.score_adjustment == pytest.approx(-0.1)

    def test_multiple_violations(self):
        result = make_result(sharpe=0.3, num_trades=2, total_return=-0.5)
        report = check_guardrails(result, GuardrailConfig())
        assert report.score_adjustment <= -0.3

    def test_capped_at_negative_one(self):
        """Score adjustment should never go below -1.0."""
        result = make_result(
            sharpe=-5.0, num_trades=0, total_return=-1.0,
            win_rate=0.0, profit_factor=0.0, max_drawdown=-0.99,
            avg_holding_bars=999,
        )
        report = check_guardrails(result, GuardrailConfig())
        assert report.score_adjustment == pytest.approx(-1.0)

    def test_10_violations_capped(self):
        """10 violations * -0.1 = -1.0, should be capped."""
        result = make_result(
            sharpe=-5.0, num_trades=0, total_return=-1.0,
            win_rate=0.0, profit_factor=0.0, max_drawdown=-0.99,
            avg_holding_bars=999,
        )
        report = check_guardrails(result, GuardrailConfig())
        assert report.score_adjustment >= -1.0


# ═══════════════════════════════════════════════════════════════════════════════
# check_guardrails — boundary conditions
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckGuardrailsBoundaries:
    def test_exact_min_trades_passes(self):
        result = make_result(num_trades=5)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.passed  # 5 >= 5

    def test_exact_min_sharpe_passes(self):
        result = make_result(sharpe=0.5)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.passed  # 0.5 >= 0.5

    def test_exact_min_return_passes(self):
        result = make_result(total_return=0.05)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.passed  # 0.05 >= 0.05

    def test_exact_min_win_rate_passes(self):
        result = make_result(win_rate=0.25)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.passed  # 0.25 >= 0.25

    def test_exact_max_drawdown_passes(self):
        """Drawdown exactly at -max_drawdown should pass."""
        result = make_result(max_drawdown=-0.30)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.passed  # -0.30 >= -0.30

    def test_just_below_min_trades_fails(self):
        result = make_result(num_trades=4)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert not report.passed

    def test_just_below_min_sharpe_fails(self):
        result = make_result(sharpe=0.49)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert not report.passed

    def test_zero_trades(self):
        result = make_result(num_trades=0)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert not report.passed
        assert any("Too few trades" in v for v in report.violations)


# ═══════════════════════════════════════════════════════════════════════════════
# check_guardrails — different config presets
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckGuardrailsPresets:
    def test_passes_aggressive_fails_conservative(self):
        """Result good enough for aggressive but not conservative."""
        result = make_result(
            sharpe=0.8, num_trades=10, total_return=0.06,
            win_rate=0.30, max_drawdown=-0.15, profit_factor=1.0,
        )
        agg_report = check_guardrails(result, GuardrailConfig.aggressive())
        cons_report = check_guardrails(result, GuardrailConfig.conservative())
        assert agg_report.passed
        assert not cons_report.passed

    def test_passes_conservative_passes_all(self):
        """Result passing conservative should pass all presets."""
        result = make_result(
            sharpe=3.0, num_trades=100, total_return=0.30,
            win_rate=0.60, max_drawdown=-0.05, profit_factor=2.5,
        )
        for preset in [GuardrailConfig.aggressive(), GuardrailConfig.moderate(),
                       GuardrailConfig.conservative()]:
            report = check_guardrails(result, preset)
            assert report.passed, f"Failed for preset with min_trades={preset.min_trades}"


# ═══════════════════════════════════════════════════════════════════════════════
# OverfittingDetector — detect_curve_fitting
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverfittingDetector:
    @pytest.fixture
    def detector(self):
        return OverfittingDetector()

    def test_insufficient_results(self, detector):
        is_overfit, reason = detector.detect_curve_fitting([make_result()])
        assert is_overfit is False
        assert "Insufficient" in reason

    def test_empty_results(self, detector):
        is_overfit, reason = detector.detect_curve_fitting([])
        assert is_overfit is False
        assert "Insufficient" in reason

    def test_uniform_good_results(self, detector):
        """All results with similar good returns → no overfitting."""
        results = make_results(5, sharpe=2.0, total_return=0.20)
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False
        assert reason == ""

    def test_performance_degradation(self, detector):
        """Top half positive, bottom half zero/negative → overfitting."""
        results = [
            make_result(total_return=0.30, iteration=0),
            make_result(total_return=0.25, iteration=1),
            make_result(total_return=0.00, iteration=2),
            make_result(total_return=-0.05, iteration=3),
        ]
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is True
        assert "Performance degradation" in reason

    def test_suspicious_high_sharpe_few_trades(self, detector):
        """One result with Sharpe > 3.0 and < 30 trades → overfitting."""
        results = [
            make_result(sharpe=5.0, num_trades=10, iteration=0),
            make_result(sharpe=1.0, num_trades=50, iteration=1),
        ]
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is True
        assert "Suspiciously high Sharpe" in reason

    def test_high_variance_across_params(self, detector):
        """High return std > mean with positive mean → overfitting."""
        results = [
            make_result(total_return=0.50, iteration=0),
            make_result(total_return=-0.40, iteration=1),
            make_result(total_return=0.30, iteration=2),
            make_result(total_return=-0.30, iteration=3),
        ]
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is True
        assert "High return variance" in reason

    def test_low_sharpe_with_many_trades_ok(self, detector):
        """Sharpe > 3 with many trades should NOT trigger overfitting."""
        results = [
            make_result(sharpe=4.0, num_trades=100, iteration=0),
            make_result(sharpe=1.0, num_trades=100, iteration=1),
        ]
        is_overfit, reason = detector.detect_curve_fitting(results)
        # High sharpe but >= 30 trades, should not be suspicious
        assert is_overfit is False

    def test_zero_trades_high_sharpe_ignored(self, detector):
        """Zero trades with high Sharpe should NOT trigger overfitting check."""
        results = [
            make_result(sharpe=5.0, num_trades=0, iteration=0),
            make_result(sharpe=1.0, num_trades=50, iteration=1),
        ]
        is_overfit, reason = detector.detect_curve_fitting(results)
        # 0 trades means the suspicious check (trades > 0) won't fire
        assert is_overfit is False

    def test_two_results_positive_returns(self, detector):
        """Exactly 2 results, both positive → no overfitting."""
        results = [
            make_result(total_return=0.10, iteration=0),
            make_result(total_return=0.15, iteration=1),
        ]
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False

    def test_all_negative_returns_no_degradation(self, detector):
        """All negative returns → top half positive check fails → no overfit."""
        results = [
            make_result(total_return=-0.05, iteration=0),
            make_result(total_return=-0.10, iteration=1),
        ]
        is_overfit, reason = detector.detect_curve_fitting(results)
        # top_avg <= 0, so the degradation check doesn't fire
        assert is_overfit is False


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: check_guardrails + OverfittingDetector
# ═══════════════════════════════════════════════════════════════════════════════

class TestGuardrailsOverfittingIntegration:
    def test_good_result_no_overfitting(self):
        """A strategy that passes guardrails and isn't overfit."""
        result = make_result(sharpe=2.0, num_trades=50)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.passed

        detector = OverfittingDetector()
        results = make_results(5, sharpe=2.0)
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert not is_overfit

    def test_overfit_result_still_passes_guardrails(self):
        """Overfitting detection is separate from guardrail checks."""
        # A single good result passes guardrails
        result = make_result(sharpe=2.0, num_trades=20)
        report = check_guardrails(result, GuardrailConfig.aggressive())
        assert report.passed

        # But multiple results show instability
        detector = OverfittingDetector()
        results = [
            make_result(total_return=0.40, iteration=0),
            make_result(total_return=-0.30, iteration=1),
        ]
        is_overfit, _ = detector.detect_curve_fitting(results)
        assert is_overfit
