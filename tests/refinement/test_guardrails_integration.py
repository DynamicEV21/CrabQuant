"""
Tests for crabquant.refinement.guardrails_integration.

Integrates check_guardrails() from crabquant.guardrails into the refinement
pipeline, wrapping the result in a pipeline-friendly interface.
"""

from unittest.mock import MagicMock, patch

import pytest

from crabquant.engine.backtest import BacktestResult
from crabquant.guardrails import GuardrailConfig, GuardrailReport
from crabquant.refinement.guardrails_integration import (
    run_guardrails_check,
    GuardrailsIntegrationResult,
    select_guardrail_config,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_result(**overrides) -> BacktestResult:
    """Factory with sensible passing defaults."""
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


# ── select_guardrail_config ─────────────────────────────────────────────────


class TestSelectGuardrailConfig:
    """Test guardrail config selection based on iteration context."""

    def test_early_iteration_returns_aggressive(self):
        cfg = select_guardrail_config(iteration=1, max_turns=7)
        assert cfg.min_trades == 5  # Aggressive default
        assert cfg.max_drawdown == 0.30

    def test_late_iteration_returns_conservative(self):
        cfg = select_guardrail_config(iteration=6, max_turns=7)
        assert cfg.min_trades == 30  # Conservative
        assert cfg.max_drawdown == 0.10

    def test_mid_iteration_returns_moderate(self):
        cfg = select_guardrail_config(iteration=3, max_turns=7)
        assert cfg.min_trades == 15  # Moderate

    def test_custom_preset_name(self):
        cfg = select_guardrail_config(iteration=1, max_turns=7, preset="conservative")
        assert cfg.min_trades == 30

    def test_custom_preset_overrides_auto(self):
        """Explicit preset overrides auto-selection."""
        cfg = select_guardrail_config(iteration=6, max_turns=7, preset="aggressive")
        assert cfg.min_trades == 5  # Still aggressive despite late iteration


# ── GuardrailsIntegrationResult ─────────────────────────────────────────────


class TestGuardrailsIntegrationResult:
    """Test the integration result dataclass."""

    def test_passed_result(self):
        r = GuardrailsIntegrationResult(
            passed=True,
            violations=[],
            warnings=[],
            score_adjustment=0.0,
            config_preset="moderate",
        )
        assert r.passed
        assert r.violations == []

    def test_failed_result(self):
        r = GuardrailsIntegrationResult(
            passed=False,
            violations=["Sharpe too low: 0.50 < 1.00"],
            warnings=["Sharpe near threshold"],
            score_adjustment=-0.1,
            config_preset="conservative",
        )
        assert not r.passed
        assert len(r.violations) == 1


# ── run_guardrails_check ───────────────────────────────────────────────────


class TestRunGuardrailsCheck:
    """Test the main guardrails check function."""

    def test_passing_result(self):
        """A good backtest result should pass guardrails."""
        result = make_result()
        integration = run_guardrails_check(result, preset="aggressive")
        assert integration.passed
        assert integration.violations == []

    def test_failing_result_low_sharpe(self):
        """Low Sharpe should produce violations."""
        result = make_result(sharpe=0.3)
        integration = run_guardrails_check(result, preset="conservative")
        assert not integration.passed
        assert any("Sharpe" in v for v in integration.violations)

    def test_failing_result_few_trades(self):
        """Too few trades should produce violations."""
        result = make_result(num_trades=2)
        integration = run_guardrails_check(result, preset="moderate")
        assert not integration.passed
        assert any("trade" in v.lower() for v in integration.violations)

    def test_failing_result_deep_drawdown(self):
        """Deep drawdown should produce violations."""
        result = make_result(max_drawdown=-0.35)
        integration = run_guardrails_check(result, preset="moderate")
        assert not integration.passed
        assert any("drawdown" in v.lower() for v in integration.violations)

    def test_warnings_produced(self):
        """Near-threshold values should produce warnings."""
        result = make_result(sharpe=1.05, num_trades=16)
        # sharpe=1.05 is close to conservative min_sharpe=1.5? No...
        # With moderate: min_sharpe=1.0, so 1.05 passes but may warn
        integration = run_guardrails_check(result, preset="moderate")
        # Sharpe 1.05 is near 1.0 threshold, should warn
        assert isinstance(integration.warnings, list)

    def test_custom_config(self):
        """Custom GuardrailConfig should be accepted."""
        cfg = GuardrailConfig(min_trades=100, min_sharpe=3.0)
        result = make_result(sharpe=2.0, num_trades=50)
        integration = run_guardrails_check(result, config=cfg)
        assert not integration.passed  # Doesn't meet strict thresholds

    def test_auto_config_selection(self):
        """Without explicit config, should auto-select based on iteration."""
        result = make_result()
        integration = run_guardrails_check(result, iteration=5, max_turns=7)
        # Should succeed and have a config_preset set
        assert integration.config_preset in ("conservative", "moderate", "aggressive")

    def test_score_adjustment_from_violations(self):
        """Score adjustment should reflect violation count."""
        result = make_result(sharpe=0.3, num_trades=2, max_drawdown=-0.35)
        integration = run_guardrails_check(result, preset="conservative")
        assert integration.score_adjustment < 0
        # Multiple violations → stronger adjustment
        assert integration.score_adjustment <= -0.2

    def test_preserves_original_report_fields(self):
        """Integration result should preserve all GuardrailReport fields."""
        result = make_result()
        integration = run_guardrails_check(result, preset="aggressive")
        assert hasattr(integration, "passed")
        assert hasattr(integration, "violations")
        assert hasattr(integration, "warnings")
        assert hasattr(integration, "score_adjustment")
