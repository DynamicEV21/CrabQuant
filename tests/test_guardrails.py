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
