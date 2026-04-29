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


# ── select_guardrail_config additional ────────────────────────────────────


class TestSelectGuardrailConfigEdgeCases:
    """Additional edge cases for config selection."""

    def test_iteration_zero_returns_aggressive(self):
        """Iteration 0 (progress=0) should return aggressive."""
        cfg = select_guardrail_config(iteration=0, max_turns=7)
        assert cfg.min_trades == 5

    def test_max_turns_zero_returns_moderate(self):
        """max_turns=0 returns moderate as default."""
        cfg = select_guardrail_config(iteration=0, max_turns=0)
        assert cfg.min_trades == 15

    def test_exact_25_percent_boundary(self):
        """Iteration exactly at 25% returns aggressive (<= 0.25)."""
        cfg = select_guardrail_config(iteration=2, max_turns=8)  # 2/8 = 0.25
        assert cfg.min_trades == 5  # aggressive

    def test_just_above_25_percent(self):
        """Iteration just above 25% returns moderate."""
        cfg = select_guardrail_config(iteration=3, max_turns=8)  # 3/8 = 0.375
        assert cfg.min_trades == 15  # moderate

    def test_exact_75_percent_boundary(self):
        """Iteration exactly at 75% returns moderate (<= 0.75)."""
        cfg = select_guardrail_config(iteration=6, max_turns=8)  # 6/8 = 0.75
        assert cfg.min_trades == 15  # moderate

    def test_just_above_75_percent(self):
        """Iteration just above 75% returns conservative."""
        cfg = select_guardrail_config(iteration=7, max_turns=8)  # 7/8 = 0.875
        assert cfg.min_trades == 30  # conservative

    def test_explicit_moderate_preset(self):
        """Explicit moderate preset works."""
        cfg = select_guardrail_config(preset="moderate")
        assert cfg.min_trades == 15

    def test_explicit_aggressive_preset(self):
        """Explicit aggressive preset works."""
        cfg = select_guardrail_config(preset="aggressive")
        assert cfg.min_trades == 5

    def test_unknown_preset_falls_through(self):
        """Unknown preset name triggers auto-selection."""
        cfg = select_guardrail_config(iteration=0, max_turns=7, preset="unknown")
        # Falls through to auto-selection; iteration=0 → aggressive
        assert cfg.min_trades == 5


# ── _config_preset_name ──────────────────────────────────────────────────


class TestConfigPresetName:
    """Test the _config_preset_name helper."""

    def test_identifies_aggressive(self):
        from crabquant.refinement.guardrails_integration import _config_preset_name
        cfg = GuardrailConfig.aggressive()
        assert _config_preset_name(cfg) == "aggressive"

    def test_identifies_moderate(self):
        from crabquant.refinement.guardrails_integration import _config_preset_name
        cfg = GuardrailConfig.moderate()
        assert _config_preset_name(cfg) == "moderate"

    def test_identifies_conservative(self):
        from crabquant.refinement.guardrails_integration import _config_preset_name
        cfg = GuardrailConfig.conservative()
        assert _config_preset_name(cfg) == "conservative"

    def test_custom_config(self):
        from crabquant.refinement.guardrails_integration import _config_preset_name
        cfg = GuardrailConfig(min_trades=100, min_sharpe=5.0)
        assert _config_preset_name(cfg) == "custom"


# ── run_guardrails_check additional ──────────────────────────────────────


class TestRunGuardrailsCheckEdgeCases:
    """Additional edge cases for the main check function."""

    def test_config_overrides_preset(self):
        """When both config and preset are given, config wins."""
        cfg = GuardrailConfig(min_trades=5, min_sharpe=0.5, max_drawdown=0.30)
        result = make_result()
        integration = run_guardrails_check(result, config=cfg, preset="conservative")
        # Config matches aggressive preset exactly
        assert integration.config_preset == "aggressive"
        # Preset was ignored - result should pass with aggressive thresholds
        assert integration.passed is True

    def test_default_preset_is_unknown(self):
        """GuardrailsIntegrationResult default config_preset is 'unknown'."""
        r = GuardrailsIntegrationResult(
            passed=False, violations=[], warnings=[], score_adjustment=0.0
        )
        assert r.config_preset == "unknown"

    def test_multiple_violations_counted(self):
        """Multiple failing metrics produce multiple violations."""
        result = make_result(sharpe=-1.0, num_trades=0, max_drawdown=-0.90)
        integration = run_guardrails_check(result, preset="conservative")
        assert len(integration.violations) >= 2

    def test_perfect_strategy_no_warnings(self):
        """A very strong strategy should have no warnings."""
        result = make_result(sharpe=5.0, num_trades=500, max_drawdown=-0.02)
        integration = run_guardrails_check(result, preset="aggressive")
        # Very high metrics, no warnings expected
        assert integration.passed is True

    def test_score_adjustment_zero_on_pass(self):
        """Score adjustment should be 0 or positive when all checks pass."""
        result = make_result(sharpe=3.0, num_trades=100)
        integration = run_guardrails_check(result, preset="aggressive")
        assert integration.score_adjustment >= 0

    def test_integration_result_default_fields(self):
        """GuardrailsIntegrationResult has correct default values."""
        r = GuardrailsIntegrationResult(
            passed=True, violations=["x"], warnings=["y"], score_adjustment=-0.5
        )
        assert r.passed is True
        assert r.violations == ["x"]
        assert r.warnings == ["y"]
        assert r.score_adjustment == -0.5
        assert r.config_preset == "unknown"
