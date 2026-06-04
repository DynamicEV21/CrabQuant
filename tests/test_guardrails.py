"""Tests for CrabQuant guardrails module."""

import pytest
from dataclasses import dataclass, field
from datetime import datetime

from crabquant.guardrails import (
    GuardrailConfig,
    GuardrailReport,
    check_guardrails,
    OverfittingDetector,
)


# ── Helper: build a BacktestResult-like object ──

@dataclass
class FakeBacktestResult:
    """Mimics BacktestResult for testing without needing VectorBT."""
    ticker: str = "AAPL"
    strategy_name: str = "test"
    iteration: int = 0
    sharpe: float = 2.0
    total_return: float = 0.15
    max_drawdown: float = -0.10
    win_rate: float = 0.55
    num_trades: int = 20
    avg_trade_return: float = 0.02
    calmar_ratio: float = 2.0
    sortino_ratio: float = 2.5
    profit_factor: float = 1.8
    avg_holding_bars: float = 5.0
    best_trade: float = 0.08
    worst_trade: float = -0.03
    passed: bool = True
    score: float = 3.0
    notes: str = ""
    params: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def _make_result(**overrides) -> FakeBacktestResult:
    """Build a FakeBacktestResult with defaults, overriding specified fields."""
    defaults = dict(
        ticker="AAPL", strategy_name="test", iteration=0,
        sharpe=2.0, total_return=0.15, max_drawdown=-0.10,
        win_rate=0.55, num_trades=20, avg_trade_return=0.02,
        calmar_ratio=2.0, sortino_ratio=2.5, profit_factor=1.8,
        avg_holding_bars=5.0, best_trade=0.08, worst_trade=-0.03,
        passed=True, score=3.0, notes="",
    )
    defaults.update(overrides)
    return FakeBacktestResult(**defaults)


# ═══════════════════════════════════════════════════════════════
# GuardrailConfig
# ═══════════════════════════════════════════════════════════════

class TestGuardrailConfig:
    def test_default_config(self):
        c = GuardrailConfig()
        assert c.max_drawdown == 0.25
        assert c.min_trades == 5
        assert c.min_win_rate == 0.30
        assert c.min_sharpe == 1.0
        assert c.min_return == 0.05
        assert c.max_holding_days == 100
        assert c.min_profit_factor == 1.0

    def test_conservative_preset(self):
        c = GuardrailConfig.conservative()
        assert c.max_drawdown == 0.10
        assert c.min_trades == 30
        assert c.min_sharpe == 1.5
        assert c.min_win_rate == 0.45
        assert c.min_return == 0.10
        assert c.min_profit_factor == 1.5

    def test_moderate_preset(self):
        c = GuardrailConfig.moderate()
        assert c.max_drawdown == 0.20
        assert c.min_trades == 15
        assert c.min_sharpe == 1.0
        assert c.min_win_rate == 0.35
        assert c.min_return == 0.08

    def test_aggressive_preset(self):
        c = GuardrailConfig.aggressive()
        assert c.max_drawdown == 0.30
        assert c.min_trades == 5
        assert c.min_sharpe == 0.5
        assert c.min_win_rate == 0.25
        assert c.min_return == 0.05

    def test_presets_are_strictest_to_loosest(self):
        """Conservative should be stricter than moderate, which is stricter than aggressive."""
        con = GuardrailConfig.conservative()
        mod = GuardrailConfig.moderate()
        agg = GuardrailConfig.aggressive()

        # Conservative has tighter thresholds
        assert con.min_trades > mod.min_trades > agg.min_trades
        assert con.min_sharpe > mod.min_sharpe > agg.min_sharpe
        assert con.max_drawdown < mod.max_drawdown < agg.max_drawdown

    def test_conservative_defaults_for_unset_fields(self):
        """Fields not explicitly set in conservative() fall back to class defaults."""
        c = GuardrailConfig.conservative()
        assert c.max_holding_days == 100  # default
        assert c.min_profit_factor == 1.5  # explicitly set

    def test_moderate_defaults_for_unset_fields(self):
        """Fields not explicitly set in moderate() fall back to class defaults."""
        c = GuardrailConfig.moderate()
        assert c.max_holding_days == 100  # default
        assert c.min_profit_factor == 1.0  # default

    def test_aggressive_defaults_for_unset_fields(self):
        """Fields not explicitly set in aggressive() fall back to class defaults."""
        c = GuardrailConfig.aggressive()
        assert c.max_holding_days == 100  # default
        assert c.min_profit_factor == 1.0  # default

    def test_custom_config_overrides(self):
        """Users can pass custom values to the constructor."""
        c = GuardrailConfig(min_trades=50, max_drawdown=0.05)
        assert c.min_trades == 50
        assert c.max_drawdown == 0.05
        # Others stay at default
        assert c.min_sharpe == 1.0

    def test_is_dataclass(self):
        """GuardrailConfig should be a dataclass instance."""
        c = GuardrailConfig()
        assert hasattr(c, '__dataclass_fields__')

    def test_presets_return_same_type(self):
        """All preset classmethods should return GuardrailConfig instances."""
        assert isinstance(GuardrailConfig.conservative(), GuardrailConfig)
        assert isinstance(GuardrailConfig.moderate(), GuardrailConfig)
        assert isinstance(GuardrailConfig.aggressive(), GuardrailConfig)


# ═══════════════════════════════════════════════════════════════
# GuardrailReport
# ═══════════════════════════════════════════════════════════════

class TestGuardrailReport:
    def test_report_fields(self):
        report = GuardrailReport(
            passed=True, violations=[], warnings=[], score_adjustment=0.0
        )
        assert report.passed is True
        assert report.violations == []
        assert report.warnings == []
        assert report.score_adjustment == 0.0

    def test_report_with_violations(self):
        report = GuardrailReport(
            passed=False,
            violations=["Sharpe too low: 0.50 < 1.0"],
            warnings=["Win rate below 50%: 45.0%"],
            score_adjustment=-0.1,
        )
        assert report.passed is False
        assert len(report.violations) == 1
        assert len(report.warnings) == 1
        assert report.score_adjustment == -0.1

    def test_report_is_dataclass(self):
        report = GuardrailReport(True, [], [], 0.0)
        assert hasattr(report, '__dataclass_fields__')


# ═══════════════════════════════════════════════════════════════
# check_guardrails — Hard Violations
# ═══════════════════════════════════════════════════════════════

class TestCheckGuardrailsViolations:
    def test_clean_result_passes(self):
        """A good result should pass all guardrails."""
        result = _make_result()
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        assert isinstance(report, GuardrailReport)
        assert report.passed is True
        assert len(report.violations) == 0
        assert report.score_adjustment == 0.0

    def test_low_sharpe_violation(self):
        result = _make_result(sharpe=0.5)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("Sharpe" in v for v in report.violations)

    def test_low_return_violation(self):
        result = _make_result(total_return=0.02)
        config = GuardrailConfig(min_return=0.05)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("Return" in v for v in report.violations)

    def test_deep_drawdown_violation(self):
        result = _make_result(max_drawdown=-0.35)
        config = GuardrailConfig(max_drawdown=0.25)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("Drawdown" in v or "drawdown" in v for v in report.violations)

    def test_too_few_trades_violation(self):
        result = _make_result(num_trades=2)
        config = GuardrailConfig(min_trades=5)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("trades" in v.lower() for v in report.violations)

    def test_low_win_rate_violation(self):
        result = _make_result(win_rate=0.20)
        config = GuardrailConfig(min_win_rate=0.30)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("Win rate" in v for v in report.violations)

    def test_low_profit_factor_violation(self):
        result = _make_result(profit_factor=0.7)
        config = GuardrailConfig(min_profit_factor=1.0)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("Profit factor" in v for v in report.violations)

    def test_long_holding_violation(self):
        result = _make_result(avg_holding_bars=150)
        config = GuardrailConfig(max_holding_days=100)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("Holding" in v for v in report.violations)

    def test_multiple_violations_accumulate(self):
        """Multiple failing checks should produce multiple violations."""
        result = _make_result(sharpe=0.3, total_return=0.01, num_trades=1)
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        assert report.passed is False
        assert len(report.violations) >= 3

    def test_zero_trades_violation(self):
        """Zero trades should violate min_trades."""
        result = _make_result(num_trades=0)
        config = GuardrailConfig(min_trades=5)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("Too few trades" in v for v in report.violations)

    def test_zero_return_violation(self):
        """Zero total return should violate min_return."""
        result = _make_result(total_return=0.0)
        config = GuardrailConfig(min_return=0.05)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("Return" in v for v in report.violations)

    def test_negative_return_violation(self):
        """Negative total return should violate min_return."""
        result = _make_result(total_return=-0.10)
        config = GuardrailConfig(min_return=0.05)
        report = check_guardrails(result, config)

        assert report.passed is False
        assert any("Return" in v for v in report.violations)

    def test_zero_drawdown_no_violation(self):
        """Zero drawdown (no loss) should not violate."""
        result = _make_result(max_drawdown=0.0)
        config = GuardrailConfig(max_drawdown=0.25)
        report = check_guardrails(result, config)

        assert not any("drawdown" in v.lower() for v in report.violations)

    def test_exact_drawdown_boundary(self):
        """Drawdown exactly at limit should not violate."""
        result = _make_result(max_drawdown=-0.25)
        config = GuardrailConfig(max_drawdown=0.25)
        report = check_guardrails(result, config)

        # -0.25 < -0.25 is False, so no violation
        assert not any("drawdown" in v.lower() for v in report.violations)

    def test_exact_sharpe_boundary(self):
        """Sharpe exactly at threshold should not violate."""
        result = _make_result(sharpe=1.0)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)

        assert not any("Sharpe" in v for v in report.violations)

    def test_exact_win_rate_boundary(self):
        """Win rate exactly at threshold should not violate."""
        result = _make_result(win_rate=0.30)
        config = GuardrailConfig(min_win_rate=0.30)
        report = check_guardrails(result, config)

        assert not any("Win rate" in v for v in report.violations)

    def test_exact_min_trades_boundary(self):
        """Trades exactly at min should not violate."""
        result = _make_result(num_trades=5)
        config = GuardrailConfig(min_trades=5)
        report = check_guardrails(result, config)

        assert not any("trades" in v.lower() for v in report.violations)

    def test_exact_profit_factor_boundary(self):
        """Profit factor exactly at threshold should not violate."""
        result = _make_result(profit_factor=1.0)
        config = GuardrailConfig(min_profit_factor=1.0)
        report = check_guardrails(result, config)

        assert not any("Profit factor" in v for v in report.violations)

    def test_exact_holding_boundary(self):
        """Holding exactly at max should not violate."""
        result = _make_result(avg_holding_bars=100)
        config = GuardrailConfig(max_holding_days=100)
        report = check_guardrails(result, config)

        assert not any("Holding" in v for v in report.violations)


# ═══════════════════════════════════════════════════════════════
# check_guardrails — Warnings
# ═══════════════════════════════════════════════════════════════

class TestCheckGuardrailsWarnings:
    def test_win_rate_below_50_warning(self):
        """Win rate below 50% should trigger a warning."""
        result = _make_result(win_rate=0.40)
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        assert report.passed is True  # 40% > 30% threshold
        assert len(report.warnings) > 0
        assert any("Win rate" in w for w in report.warnings)

    def test_win_rate_above_50_no_warning(self):
        """Win rate above 50% should not trigger the below-50 warning."""
        result = _make_result(win_rate=0.60)
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        assert not any("Win rate" in w for w in report.warnings)

    def test_low_trade_count_warning(self):
        """Trades between min_trades and 2*min_trades should trigger a warning."""
        result = _make_result(num_trades=7)
        config = GuardrailConfig(min_trades=5)
        report = check_guardrails(result, config)

        assert report.passed is True
        assert any("Low trade count" in w for w in report.warnings)

    def test_no_low_trade_count_warning_when_enough(self):
        """Trades >= 2*min_trades should not trigger low trade count warning."""
        result = _make_result(num_trades=15)
        config = GuardrailConfig(min_trades=5)
        report = check_guardrails(result, config)

        assert not any("Low trade count" in w for w in report.warnings)

    def test_sharpe_near_threshold_warning(self):
        """Sharpe between min and 1.2*min should trigger a warning."""
        result = _make_result(sharpe=1.1)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)

        assert report.passed is True
        assert any("Sharpe near threshold" in w for w in report.warnings)

    def test_no_sharpe_warning_when_high(self):
        """Sharpe well above threshold should not trigger near-threshold warning."""
        result = _make_result(sharpe=2.0)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)

        assert not any("Sharpe near" in w for w in report.warnings)

    def test_drawdown_approaching_limit_warning(self):
        """Drawdown between 70% and 100% of limit should warn."""
        result = _make_result(max_drawdown=-0.20)
        config = GuardrailConfig(max_drawdown=0.25)
        # -0.25 < -0.20 < -0.175 → in the warning zone
        report = check_guardrails(result, config)

        assert report.passed is True
        assert any("Drawdown approaching" in w for w in report.warnings)

    def test_no_drawdown_warning_when_small(self):
        """Small drawdown should not trigger approaching-limit warning."""
        result = _make_result(max_drawdown=-0.05)
        config = GuardrailConfig(max_drawdown=0.25)
        # -0.05 is not between -0.25 and -0.175
        report = check_guardrails(result, config)

        assert not any("Drawdown approaching" in w for w in report.warnings)

    def test_zero_trades_no_win_rate_warning(self):
        """When num_trades is 0, win rate below 50% warning should not fire."""
        result = _make_result(num_trades=0, win_rate=0.30)
        config = GuardrailConfig(min_trades=0)  # avoid violation
        report = check_guardrails(result, config)

        assert not any("Win rate below 50%" in w for w in report.warnings)

    def test_warnings_and_violations_can_coexist(self):
        """A result can have both violations and warnings simultaneously."""
        result = _make_result(sharpe=0.5, win_rate=0.40, num_trades=7)
        config = GuardrailConfig(min_trades=5, min_sharpe=1.0, min_win_rate=0.30)
        report = check_guardrails(result, config)

        assert report.passed is False  # Sharpe violation
        assert len(report.violations) >= 1
        assert len(report.warnings) >= 1


# ═══════════════════════════════════════════════════════════════
# check_guardrails — Score Adjustment
# ═══════════════════════════════════════════════════════════════

class TestCheckGuardrailsScore:
    def test_score_adjustment_per_violation(self):
        """Score adjustment should be -0.1 per violation, capped at -1.0."""
        result = _make_result(sharpe=0.3, total_return=0.01, num_trades=1)
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        expected = max(-1.0, -0.1 * len(report.violations))
        assert report.score_adjustment == expected

    def test_score_adjustment_capped_at_minus_one(self):
        """Even with many violations, score_adjustment should not exceed -1.0."""
        result = _make_result(
            sharpe=0.0, total_return=0.0, max_drawdown=-0.50,
            win_rate=0.0, num_trades=0, profit_factor=0.0,
            avg_holding_bars=200,
        )
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        assert report.score_adjustment == max(-1.0, -0.1 * len(report.violations))
        assert report.score_adjustment >= -1.0

    def test_no_violations_zero_adjustment(self):
        """Clean result should have zero score adjustment."""
        result = _make_result()
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        assert report.score_adjustment == 0.0

    def test_single_violation_adjustment(self):
        """One violation should give exactly -0.1 adjustment."""
        result = _make_result(sharpe=0.5)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)

        assert report.score_adjustment == -0.1

    def test_two_violations_adjustment(self):
        """Two violations should give exactly -0.2 adjustment."""
        result = _make_result(sharpe=0.5, total_return=0.01)
        config = GuardrailConfig(min_sharpe=1.0, min_return=0.05)
        report = check_guardrails(result, config)

        assert report.score_adjustment == -0.2

    def test_warnings_do_not_affect_score(self):
        """Warnings should not change the score adjustment."""
        result = _make_result(win_rate=0.40, sharpe=1.1, num_trades=7)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)

        # Has warnings but no violations
        assert report.passed is True
        assert report.score_adjustment == 0.0


# ═══════════════════════════════════════════════════════════════
# check_guardrails — Preset Configs Integration
# ═══════════════════════════════════════════════════════════════

class TestCheckGuardrailsPresets:
    def test_conservative_rejects_marginal_result(self):
        """A result that passes default config should fail conservative."""
        result = _make_result(sharpe=1.2, total_return=0.08, num_trades=20, win_rate=0.40)
        config = GuardrailConfig()
        report_default = check_guardrails(result, config)
        assert report_default.passed is True

        config_con = GuardrailConfig.conservative()
        report_con = check_guardrails(result, config_con)
        # Conservative requires min_sharpe=1.5, min_return=0.10, min_win_rate=0.45
        assert report_con.passed is False

    def test_aggressive_accepts_weaker_result(self):
        """A result that fails default config should pass aggressive."""
        result = _make_result(sharpe=0.7, total_return=0.06, win_rate=0.30)
        config = GuardrailConfig()
        report_default = check_guardrails(result, config)
        assert report_default.passed is False  # sharpe 0.7 < 1.0

        config_agg = GuardrailConfig.aggressive()
        report_agg = check_guardrails(result, config_agg)
        # Aggressive allows min_sharpe=0.5
        assert report_agg.passed is True


# ═══════════════════════════════════════════════════════════════
# OverfittingDetector
# ═══════════════════════════════════════════════════════════════

class TestOverfittingDetector:
    def _make_results(self, returns, sharpes, trades):
        return [
            _make_result(total_return=r, sharpe=s, num_trades=t, iteration=i)
            for i, (r, s, t) in enumerate(zip(returns, sharpes, trades))
        ]

    def test_no_overfit_consistent_results(self):
        """Consistent results across params should not trigger overfitting."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.10, 0.12, 0.11, 0.09, 0.10],
            sharpes=[1.5, 1.6, 1.4, 1.3, 1.5],
            trades=[25, 30, 28, 22, 26],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False
        assert reason == ""

    def test_performance_degradation_detected(self):
        """If top half avg > 0 but bottom half <= 0, flag overfitting."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.10, 0.08, 0.06, -0.02, -0.04, -0.06],
            sharpes=[1.5, 1.3, 1.1, 0.7, 0.5, 0.3],
            trades=[40, 40, 40, 40, 40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is True
        assert "degradation" in reason.lower()

    def test_suspicious_sharpe_detected(self):
        """High Sharpe with few trades should be flagged."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.20, 0.10, 0.08],
            sharpes=[5.0, 1.5, 1.2],
            trades=[5, 30, 25],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is True
        assert "suspicious" in reason.lower()

    def test_high_variance_detected(self):
        """High std > mean across returns should be flagged."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.50, 0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01],
            sharpes=[1.5, 1.0, 1.1, 1.0, 1.2, 1.0, 1.1, 1.0],
            trades=[40, 40, 40, 40, 40, 40, 40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is True
        assert "variance" in reason.lower()

    def test_single_result_returns_no_overfit(self):
        """Single result is insufficient for overfitting detection."""
        detector = OverfittingDetector()
        results = self._make_results(returns=[0.15], sharpes=[4.0], trades=[3])
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False
        assert "insufficient" in reason.lower()

    def test_empty_results_returns_no_overfit(self):
        """Empty list should return insufficient results."""
        detector = OverfittingDetector()
        is_overfit, reason = detector.detect_curve_fitting([])
        assert is_overfit is False
        assert "insufficient" in reason.lower()

    def test_two_results_no_degradation(self):
        """Two results both positive should not trigger degradation."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.15, 0.10],
            sharpes=[1.5, 1.2],
            trades=[30, 25],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False

    def test_two_results_degradation(self):
        """Two results: one positive, one negative should trigger degradation."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.20, -0.05],
            sharpes=[1.5, 0.3],
            trades=[30, 30],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        # Sorted: [0.20, -0.05], top=[0.20], bottom=[-0.05]
        # top_avg=0.20 > 0 and bottom_avg=-0.05 <= 0 → degradation
        assert is_overfit is True
        assert "degradation" in reason.lower()

    def test_sharpe_exactly_30_trades_no_suspicious(self):
        """Sharpe > 3.0 with exactly 30 trades should NOT trigger (needs < 30)."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.15, 0.10],
            sharpes=[4.0, 1.5],
            trades=[30, 30],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False

    def test_sharpe_high_zero_trades_no_suspicious(self):
        """Sharpe > 3.0 with 0 trades should NOT trigger (needs > 0)."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.20, 0.10],
            sharpes=[5.0, 1.5],
            trades=[0, 30],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        # 0 trades doesn't satisfy num_trades > 0
        assert is_overfit is False

    def test_all_negative_returns_no_variance_flag(self):
        """All negative returns: mean < 0 so variance check skips (requires mean > 0)."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[-0.50, -0.01, -0.02, -0.01, -0.03],
            sharpes=[0.3, 0.5, 0.4, 0.5, 0.4],
            trades=[30, 30, 30, 30, 30],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        # Mean is negative, so variance check won't fire
        # Degradation check: top half avg > 0? No (all negative) → no degradation
        assert is_overfit is False

    def test_identical_returns_no_variance_flag(self):
        """All identical returns: std=0, so variance check won't trigger."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.10, 0.10, 0.10, 0.10, 0.10],
            sharpes=[1.5, 1.5, 1.5, 1.5, 1.5],
            trades=[30, 30, 30, 30, 30],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False

    def test_bottom_half_exactly_zero_no_degradation(self):
        """Bottom half avg exactly 0 should trigger degradation (<= 0)."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.15, 0.10, 0.00, 0.00],
            sharpes=[1.5, 1.2, 0.5, 0.5],
            trades=[30, 30, 30, 30],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        # Sorted: [0.15, 0.10, 0.00, 0.00]
        # top=[0.15, 0.10] avg=0.125 > 0, bottom=[0.00, 0.00] avg=0.0 <= 0
        assert is_overfit is True
        assert "degradation" in reason.lower()

    def test_returns_tuple_type(self):
        """detect_curve_fitting should return a tuple of (bool, str)."""
        detector = OverfittingDetector()
        results = self._make_results([0.10], [1.5], [20])
        result = detector.detect_curve_fitting(results)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_degradation_triggers_before_variance(self):
        """When both degradation and high variance apply, degradation fires first."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.50, 0.30, -0.10, -0.20, -0.30, -0.40],
            sharpes=[2.0, 1.5, 0.5, 0.3, 0.2, 0.1],
            trades=[30, 30, 30, 30, 30, 30],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is True
        # Degradation is checked first in the code
        assert "degradation" in reason.lower()
