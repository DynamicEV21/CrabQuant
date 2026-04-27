"""
Guardrails integration for the refinement pipeline.

Wraps check_guardrails() from crabquant.guardrails in a pipeline-friendly
interface with auto-config selection based on iteration context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crabquant.engine.backtest import BacktestResult
from crabquant.guardrails import GuardrailConfig, GuardrailReport, check_guardrails


@dataclass
class GuardrailsIntegrationResult:
    """Pipeline-friendly wrapper around GuardrailReport."""

    passed: bool
    violations: list[str]
    warnings: list[str]
    score_adjustment: float
    config_preset: str = "unknown"


def select_guardrail_config(
    iteration: int = 0,
    max_turns: int = 7,
    preset: str | None = None,
) -> GuardrailConfig:
    """Select a GuardrailConfig based on iteration context.

    Auto-selection logic:
    - Early iterations (<= 25% of max_turns): aggressive (allow exploration)
    - Mid iterations (25-75%): moderate
    - Late iterations (> 75%): conservative (only accept robust strategies)

    Args:
        iteration: Current iteration number (0-indexed).
        max_turns: Maximum number of iterations.
        preset: Explicit preset name. If provided, auto-selection is skipped.

    Returns:
        GuardrailConfig with appropriate thresholds.
    """
    if preset == "conservative":
        return GuardrailConfig.conservative()
    if preset == "moderate":
        return GuardrailConfig.moderate()
    if preset == "aggressive":
        return GuardrailConfig.aggressive()

    # Auto-select based on iteration progress
    if max_turns <= 0:
        return GuardrailConfig.moderate()

    progress = iteration / max_turns
    if progress <= 0.25:
        return GuardrailConfig.aggressive()
    elif progress <= 0.75:
        return GuardrailConfig.moderate()
    else:
        return GuardrailConfig.conservative()


def _config_preset_name(config: GuardrailConfig) -> str:
    """Determine which preset a config matches (or 'custom')."""
    presets = {
        "aggressive": GuardrailConfig.aggressive(),
        "moderate": GuardrailConfig.moderate(),
        "conservative": GuardrailConfig.conservative(),
    }
    for name, preset_cfg in presets.items():
        if (
            config.min_trades == preset_cfg.min_trades
            and config.max_drawdown == preset_cfg.max_drawdown
            and config.min_sharpe == preset_cfg.min_sharpe
        ):
            return name
    return "custom"


def run_guardrails_check(
    result: BacktestResult,
    *,
    config: GuardrailConfig | None = None,
    preset: str | None = None,
    iteration: int = 0,
    max_turns: int = 7,
) -> GuardrailsIntegrationResult:
    """Run guardrails check on a backtest result.

    Args:
        result: BacktestResult from the engine.
        config: Explicit GuardrailConfig. If provided, preset is ignored.
        preset: Guardrail preset name ("conservative", "moderate", "aggressive").
        iteration: Current iteration for auto-config selection.
        max_turns: Max turns for auto-config selection.

    Returns:
        GuardrailsIntegrationResult with violations, warnings, and score adjustment.
    """
    if config is None:
        config = select_guardrail_config(
            iteration=iteration, max_turns=max_turns, preset=preset
        )

    report = check_guardrails(result, config)

    return GuardrailsIntegrationResult(
        passed=report.passed,
        violations=report.violations,
        warnings=report.warnings,
        score_adjustment=report.score_adjustment,
        config_preset=_config_preset_name(config),
    )
