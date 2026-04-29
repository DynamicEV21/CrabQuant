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


# ── GuardrailConfig presets ──

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

    def test_conervative_preset(self):
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

    def test_moderate_preset_max_holding_days_default(self):
        """Moderate preset should inherit default max_holding_days."""
        c = GuardrailConfig.moderate()
        assert c.max_holding_days == 100  # inherited default

    def test_aggressive_preset_max_holding_days_default(self):
        """Aggressive preset should inherit default max_holding_days."""
        c = GuardrailConfig.aggressive()
        assert c.max_holding_days == 100  # inherited default

    def test_moderate_preset_min_profit_factor_default(self):
        """Moderate preset should inherit default min_profit_factor."""
        c = GuardrailConfig.moderate()
        assert c.min_profit_factor == 1.0  # inherited default

    def test_aggressive_preset_min_profit_factor_default(self):
        """Aggressive preset should inherit default min_profit_factor."""
        c = GuardrailConfig.aggressive()
        assert c.min_profit_factor == 1.0  # inherited default

    def test_presets_are_independent_instances(self):
        """Each preset call should return a fresh instance."""
        c1 = GuardrailConfig.conservative()
        c2 = GuardrailConfig.conservative()
        assert c1 is not c2
        c1.max_drawdown = 0.99
        assert c2.max_drawdown == 0.10  # c2 unaffected

    def test_custom_config_overrides(self):
        """Custom config values should override defaults."""
        c = GuardrailConfig(min_trades=50, max_drawdown=0.05)
        assert c.min_trades == 50
        assert c.max_drawdown == 0.05
        # Other defaults still intact
        assert c.min_sharpe == 1.0
        assert c.min_return == 0.05

    def test_all_presets_are_dataclass_instances(self):
        """All presets should be GuardrailConfig instances."""
        for factory in [GuardrailConfig, GuardrailConfig.conservative,
                        GuardrailConfig.moderate, GuardrailConfig.aggressive]:
            assert isinstance(factory(), GuardrailConfig)


# ── check_guardrails ──

class TestCheckGuardrails:
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

    def test_score_adjustment_per_violation(self):
        """Score adjustment should be -0.1 per violation, capped at -1.0."""
        result = _make_result(sharpe=0.3, total_return=0.01, num_trades=1)
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        expected = max(-1.0, -0.1 * len(report.violations))
        assert report.score_adjustment == expected

    def test_score_adjustment_capped_at_minus_one(self):
        """Even with many violations, score_adjustment should not exceed -1.0."""
        # 7 checks that each produce a violation → cap at -1.0
        result = _make_result(
            sharpe=0.0, total_return=0.0, max_drawdown=-0.50,
            win_rate=0.0, num_trades=0, profit_factor=0.0,
            avg_holding_bars=200,
        )
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        # All 7 checks fail → -0.7, capped formula: max(-1.0, -0.1 * 7) = -0.7
        # To actually hit the cap we need >10 violations; with 7 the cap isn't reached.
        # Verify the formula works: adjustment = -0.1 * count, max at -1.0
        assert report.score_adjustment == max(-1.0, -0.1 * len(report.violations))
        assert report.score_adjustment >= -1.0

    def test_warnings_generated(self):
        """Warnings should be generated for marginal metrics."""
        # Win rate below 50% should trigger a warning
        result = _make_result(win_rate=0.40)
        config = GuardrailConfig()
        report = check_guardrails(result, config)

        assert report.passed is True  # 40% > 30% threshold
        assert len(report.warnings) > 0
        assert any("Win rate" in w for w in report.warnings)

    # ── Boundary condition tests ──

    @pytest.mark.parametrize("field_name,value,threshold_field", [
        ("sharpe", 1.0, "min_sharpe"),
        ("total_return", 0.05, "min_return"),
        ("win_rate", 0.30, "min_win_rate"),
        ("profit_factor", 1.0, "min_profit_factor"),
        ("num_trades", 5, "min_trades"),
    ])
    def test_exact_threshold_passes(self, field_name, value, threshold_field):
        """Metrics exactly at threshold should pass (not violate)."""
        result = _make_result(**{field_name: value})
        config = GuardrailConfig()
        report = check_guardrails(result, config)
        assert report.passed is True

    def test_drawdown_exact_threshold_passes(self):
        """Drawdown exactly at -max_drawdown should pass."""
        result = _make_result(max_drawdown=-0.25)
        config = GuardrailConfig(max_drawdown=0.25)
        report = check_guardrails(result, config)
        assert report.passed is True

    def test_holding_bars_exact_threshold_passes(self):
        """Holding bars exactly at max_holding_days should pass."""
        result = _make_result(avg_holding_bars=100.0)
        config = GuardrailConfig(max_holding_days=100)
        report = check_guardrails(result, config)
        assert report.passed is True

    def test_just_below_threshold_fails(self):
        """Metrics epsilon below threshold should fail."""
        result = _make_result(sharpe=0.999)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)
        assert report.passed is False

    def test_just_above_drawdown_limit_fails(self):
        """Drawdown just past -max_drawdown should fail."""
        result = _make_result(max_drawdown=-0.251)
        config = GuardrailConfig(max_drawdown=0.25)
        report = check_guardrails(result, config)
        assert report.passed is False

    def test_just_above_holding_limit_fails(self):
        """Holding bars just above max_holding_days should fail."""
        result = _make_result(avg_holding_bars=100.1)
        config = GuardrailConfig(max_holding_days=100)
        report = check_guardrails(result, config)
        assert report.passed is False

    # ── Zero / negative edge cases ──

    def test_zero_trades(self):
        """Zero trades should violate min_trades."""
        result = _make_result(num_trades=0)
        config = GuardrailConfig(min_trades=5)
        report = check_guardrails(result, config)
        assert report.passed is False
        assert any("Too few trades" in v for v in report.violations)

    def test_negative_sharpe(self):
        """Negative sharpe should violate min_sharpe."""
        result = _make_result(sharpe=-1.5)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)
        assert report.passed is False

    def test_negative_total_return(self):
        """Negative total return should violate min_return."""
        result = _make_result(total_return=-0.10)
        config = GuardrailConfig(min_return=0.05)
        report = check_guardrails(result, config)
        assert report.passed is False

    def test_zero_profit_factor(self):
        """Zero profit factor should violate min_profit_factor."""
        result = _make_result(profit_factor=0.0)
        config = GuardrailConfig(min_profit_factor=1.0)
        report = check_guardrails(result, config)
        assert report.passed is False

    # ── Report structure tests ──

    def test_report_violations_are_strings(self):
        """All violations in the report should be strings."""
        result = _make_result(sharpe=0.0, num_trades=0)
        config = GuardrailConfig()
        report = check_guardrails(result, config)
        assert all(isinstance(v, str) for v in report.violations)

    def test_report_warnings_are_strings(self):
        """All warnings in the report should be strings."""
        result = _make_result(win_rate=0.40)
        config = GuardrailConfig()
        report = check_guardrails(result, config)
        assert all(isinstance(w, str) for w in report.warnings)

    def test_passed_true_with_warnings(self):
        """Strategy can pass with warnings present."""
        result = _make_result(win_rate=0.40, num_trades=6, sharpe=1.05)
        config = GuardrailConfig(min_trades=5, min_sharpe=1.0)
        report = check_guardrails(result, config)
        assert report.passed is True
        assert len(report.warnings) > 0

    def test_passed_false_no_warnings_possible(self):
        """A failing result can have zero warnings."""
        # Zero trades: no low-trade warning (condition requires num_trades > 0)
        # All other metrics far from warning thresholds
        result = _make_result(num_trades=0, sharpe=5.0, win_rate=0.60,
                              max_drawdown=-0.01, total_return=0.50)
        config = GuardrailConfig(min_trades=5)
        report = check_guardrails(result, config)
        assert report.passed is False
        assert len(report.violations) > 0
        assert len(report.warnings) == 0

    # ── Score adjustment edge cases ──

    def test_score_adjustment_zero_when_no_violations(self):
        """Zero violations should yield score_adjustment of 0.0."""
        result = _make_result()
        config = GuardrailConfig()
        report = check_guardrails(result, config)
        assert report.score_adjustment == 0.0

    def test_score_adjustment_single_violation(self):
        """Single violation should yield -0.1."""
        result = _make_result(sharpe=0.5)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)
        assert report.score_adjustment == -0.1

    # ── Warning-specific tests ──

    def test_warning_low_trade_count(self):
        """Trade count between 0 and min_trades*2 should warn."""
        # min_trades=5, so 0 < 6 < 10 → should warn
        result = _make_result(num_trades=6)
        config = GuardrailConfig(min_trades=5)
        report = check_guardrails(result, config)
        assert report.passed is True
        assert any("Low trade count" in w for w in report.warnings)

    def test_warning_low_trade_count_zero_trades(self):
        """Zero trades should NOT trigger low trade count warning."""
        result = _make_result(num_trades=0)
        config = GuardrailConfig(min_trades=5)
        report = check_guardrails(result, config)
        assert not any("Low trade count" in w for w in report.warnings)

    def test_warning_sharpe_near_threshold(self):
        """Sharpe between 0 and min_sharpe*1.2 should warn."""
        # min_sharpe=1.0, so 0 < 1.05 < 1.2 → should warn
        result = _make_result(sharpe=1.05)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)
        assert report.passed is True
        assert any("Sharpe near threshold" in w for w in report.warnings)

    def test_no_warning_sharpe_well_above(self):
        """Sharpe well above 1.2x threshold should not warn."""
        result = _make_result(sharpe=3.0)
        config = GuardrailConfig(min_sharpe=1.0)
        report = check_guardrails(result, config)
        assert not any("Sharpe near threshold" in w for w in report.warnings)

    def test_warning_drawdown_approaching_limit(self):
        """Drawdown between -max_dd*0.7 and -max_dd should warn."""
        # max_drawdown=0.25, so -0.25 < dd < -0.175
        result = _make_result(max_drawdown=-0.20)
        config = GuardrailConfig(max_drawdown=0.25)
        report = check_guardrails(result, config)
        assert report.passed is True
        assert any("Drawdown approaching" in w for w in report.warnings)

    def test_no_warning_drawdown_shallow(self):
        """Shallow drawdown should not trigger approaching-limit warning."""
        result = _make_result(max_drawdown=-0.10)
        config = GuardrailConfig(max_drawdown=0.25)
        report = check_guardrails(result, config)
        assert not any("Drawdown approaching" in w for w in report.warnings)

    def test_warning_win_rate_below_50_with_zero_trades(self):
        """Win rate below 50% with zero trades should NOT trigger win rate warning."""
        result = _make_result(win_rate=0.40, num_trades=0)
        config = GuardrailConfig()
        report = check_guardrails(result, config)
        assert not any("Win rate below 50%" in w for w in report.warnings)

    def test_no_warnings_excellent_metrics(self):
        """Excellent metrics should produce no warnings."""
        result = _make_result(
            num_trades=50, sharpe=3.0, win_rate=0.70,
            max_drawdown=-0.02, total_return=0.30,
        )
        config = GuardrailConfig()
        report = check_guardrails(result, config)
        assert len(report.warnings) == 0

    # ── Integration: using different presets ──

    def test_conservative_config_fails_moderate_passing_result(self):
        """A result that passes moderate should fail conservative."""
        result = _make_result(
            num_trades=20, sharpe=1.2, win_rate=0.40,
            total_return=0.09, max_drawdown=-0.15,
        )
        mod_config = GuardrailConfig.moderate()
        con_config = GuardrailConfig.conservative()

        assert check_guardrails(result, mod_config).passed is True
        assert check_guardrails(result, con_config).passed is False

    def test_aggressive_config_passes_conservative_failing_result(self):
        """A result that fails conservative should pass aggressive."""
        result = _make_result(
            num_trades=10, sharpe=0.6, win_rate=0.30,
            total_return=0.06, max_drawdown=-0.25,
        )
        con_config = GuardrailConfig.conservative()
        agg_config = GuardrailConfig.aggressive()

        assert check_guardrails(result, con_config).passed is False
        assert check_guardrails(result, agg_config).passed is True

    def test_violation_messages_contain_values(self):
        """Violation messages should include the actual metric values."""
        result = _make_result(sharpe=0.3, total_return=0.01, num_trades=1)
        config = GuardrailConfig()
        report = check_guardrails(result, config)
        for v in report.violations:
            # Messages should contain actual values, not just labels
            assert any(char.isdigit() for char in v)


# ── OverfittingDetector ──

class TestOverfittingDetector:
    def _make_results(self, returns: list[float], sharpes: list[float], trades: list[int]):
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
        # When top half is positive and bottom half is negative, degradation
        # naturally implies high variance. The detector checks degradation first,
        # so this should always trigger as degradation.
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
        # All returns positive (no degradation) but with huge spread:
        # One outlier at 0.50, rest near 0.01-0.03. Mean≈0.076, std≈0.160.
        # Both halves positive → no degradation. std > mean → variance triggers.
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

    # ── Additional OverfittingDetector tests ──

    def test_empty_list_returns_no_overfit(self):
        """Empty list should return insufficient results."""
        detector = OverfittingDetector()
        is_overfit, reason = detector.detect_curve_fitting([])
        assert is_overfit is False
        assert "insufficient" in reason.lower()

    def test_two_results_enough_for_detection(self):
        """Two results should be enough for overfitting detection."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.20, -0.05],
            sharpes=[2.0, 0.5],
            trades=[40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        # Top half avg=0.20 > 0, bottom half avg=-0.05 <= 0 → degradation
        assert is_overfit is True

    def test_three_results_split(self):
        """3 results: mid=1, so top=1, bottom=2."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.20, 0.01, -0.10],
            sharpes=[2.0, 1.0, 0.5],
            trades=[40, 40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        # Sorted desc: [0.20, 0.01, -0.10], mid=1
        # top=[0.20] avg=0.20, bottom=[0.01, -0.10] avg=-0.045
        # top_avg > 0, bottom_avg <= 0 → degradation
        assert is_overfit is True
        assert "degradation" in reason.lower()

    def test_all_negative_returns_no_degradation(self):
        """All negative returns: top_avg <= 0, so no degradation check fires."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[-0.01, -0.02, -0.03, -0.04],
            sharpes=[0.5, 0.4, 0.3, 0.2],
            trades=[40, 40, 40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        # top_avg=-0.01 (not > 0), no degradation
        # mean=-0.025 (not > 0), no variance
        # No high sharpe with few trades
        assert is_overfit is False

    def test_high_sharpe_enough_trades_not_flagged(self):
        """High sharpe with enough trades (>=30) should not be suspicious."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.20, 0.15],
            sharpes=[5.0, 4.0],
            trades=[30, 35],  # exactly 30 and above — should be fine
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False

    def test_high_sharpe_zero_trades_not_flagged(self):
        """High sharpe with zero trades should NOT trigger suspicious check."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.20, 0.15],
            sharpes=[5.0, 4.0],
            trades=[0, 35],  # 0 trades → condition requires > 0
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False

    def test_identical_returns_no_variance(self):
        """All identical returns: std=0, no variance issue."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.10, 0.10, 0.10, 0.10],
            sharpes=[1.5, 1.5, 1.5, 1.5],
            trades=[40, 40, 40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is False

    def test_returns_mean_zero_no_variance_flag(self):
        """Mean of zero returns should skip variance check (requires mean > 0)."""
        detector = OverfittingDetector()
        # Returns: [-0.10, 0.10] → mean=0.0, std≈0.1
        # mean not > 0 → variance check skipped
        results = self._make_results(
            returns=[0.10, -0.10],
            sharpes=[1.5, 0.5],
            trades=[40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        # top_avg=0.10 > 0, bottom_avg=-0.10 <= 0 → degradation fires
        assert is_overfit is True
        assert "degradation" in reason.lower()

    def test_suspicious_sharpe_reason_includes_iteration(self):
        """Suspicious sharpe reason should include the iteration number."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.20, 0.10],
            sharpes=[5.0, 1.5],
            trades=[5, 30],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is True
        assert "iter=0" in reason

    def test_variance_reason_includes_std_and_mean(self):
        """High variance reason should include std and mean values."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.50, 0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01],
            sharpes=[1.5, 1.0, 1.1, 1.0, 1.2, 1.0, 1.1, 1.0],
            trades=[40, 40, 40, 40, 40, 40, 40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert "std=" in reason
        assert "mean=" in reason

    def test_low_positive_variance_not_flagged(self):
        """Small variance relative to mean should not be flagged."""
        detector = OverfittingDetector()
        # Returns tightly clustered around 0.10
        results = self._make_results(
            returns=[0.11, 0.10, 0.09, 0.10, 0.11, 0.10],
            sharpes=[1.5, 1.4, 1.3, 1.4, 1.5, 1.4],
            trades=[40, 40, 40, 40, 40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        # std≈0.007, mean≈0.102 → std < mean → no variance flag
        assert is_overfit is False

    def test_degradation_reason_includes_averages(self):
        """Degradation reason should include top and bottom half averages."""
        detector = OverfittingDetector()
        results = self._make_results(
            returns=[0.20, 0.10, -0.05, -0.10],
            sharpes=[2.0, 1.5, 0.5, 0.3],
            trades=[40, 40, 40, 40],
        )
        is_overfit, reason = detector.detect_curve_fitting(results)
        assert is_overfit is True
        # Reason should mention the actual averages
        assert "avg return" in reason.lower()

    def test_detector_is_instantiable(self):
        """OverfittingDetector should be instantiable without arguments."""
        detector = OverfittingDetector()
        assert hasattr(detector, 'detect_curve_fitting')
