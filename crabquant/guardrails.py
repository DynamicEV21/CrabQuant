"""
CrabQuant Strategy Guardrails

Prevents garbage strategies from being promoted to winners.
Provides configurable risk thresholds, overfitting detection, and validation gates.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GuardrailConfig:
    """Configurable risk thresholds for strategy validation."""

    max_drawdown: float = 0.25      # Max allowed drawdown (25%)
    min_trades: int = 5             # Min trades for statistical significance
    min_win_rate: float = 0.30      # Min win rate
    min_sharpe: float = 1.0         # Min Sharpe ratio (pre-filter, lower than final)
    min_return: float = 0.05        # Min total return (5%)
    max_holding_days: int = 100     # Max average holding period (days)
    min_profit_factor: float = 1.0  # Min profit factor

    @classmethod
    def conservative(cls):
        return cls(
            max_drawdown=0.10, min_trades=30, min_sharpe=1.5,
            min_win_rate=0.45, min_return=0.10, min_profit_factor=1.5,
        )

    @classmethod
    def moderate(cls):
        return cls(
            max_drawdown=0.20, min_trades=15, min_sharpe=1.0,
            min_win_rate=0.35, min_return=0.08,
        )

    @classmethod
    def aggressive(cls):
        return cls(
            max_drawdown=0.30, min_trades=5, min_sharpe=0.5,
            min_win_rate=0.25, min_return=0.05,
        )


@dataclass
class GuardrailReport:
    """Result of guardrail checks against a backtest result."""

    passed: bool
    violations: list[str]
    warnings: list[str]
    score_adjustment: float  # -1.0 to 0.0, applied to composite score


def check_guardrails(result, config: GuardrailConfig) -> GuardrailReport:
    """
    Check a BacktestResult against all guardrail thresholds.

    Args:
        result: BacktestResult from the engine
        config: GuardrailConfig with thresholds

    Returns:
        GuardrailReport with violations, warnings, and score adjustment
    """
    violations = []
    warnings = []

    # ── Hard violations (fail the guardrail) ──

    if result.num_trades < config.min_trades:
        violations.append(
            f"Too few trades: {result.num_trades} < {config.min_trades}"
        )

    if result.sharpe < config.min_sharpe:
        violations.append(
            f"Sharpe too low: {result.sharpe:.2f} < {config.min_sharpe}"
        )

    if result.total_return < config.min_return:
        violations.append(
            f"Return too low: {result.total_return:.1%} < {config.min_return:.0%}"
        )

    if result.win_rate < config.min_win_rate:
        violations.append(
            f"Win rate too low: {result.win_rate:.1%} < {config.min_win_rate:.0%}"
        )

    if result.profit_factor < config.min_profit_factor:
        violations.append(
            f"Profit factor too low: {result.profit_factor:.2f} < {config.min_profit_factor}"
        )

    # Drawdown: stored as negative float in BacktestResult (e.g., -0.15)
    if result.max_drawdown < -config.max_drawdown:
        violations.append(
            f"Drawdown too deep: {result.max_drawdown:.1%} > {config.max_drawdown:.0%}"
        )

    if result.avg_holding_bars > config.max_holding_days:
        violations.append(
            f"Holding too long: {result.avg_holding_bars:.0f} bars > {config.max_holding_days}"
        )

    # ── Warnings (pass but worth flagging) ──

    if 0 < result.num_trades < config.min_trades * 2:
        warnings.append(
            f"Low trade count: {result.num_trades} (marginal statistical significance)"
        )

    if 0 < result.sharpe < config.min_sharpe * 1.2:
        warnings.append(
            f"Sharpe near threshold: {result.sharpe:.2f} (close to {config.min_sharpe})"
        )

    if -config.max_drawdown < result.max_drawdown < -config.max_drawdown * 0.7:
        warnings.append(
            f"Drawdown approaching limit: {result.max_drawdown:.1%}"
        )

    if result.num_trades > 0 and result.win_rate < 0.50:
        warnings.append(
            f"Win rate below 50%: {result.win_rate:.1%} (relies on large winners)"
        )

    # ── Score adjustment: -0.1 per violation, capped at -1.0 ──
    score_adjustment = max(-1.0, -0.1 * len(violations))

    return GuardrailReport(
        passed=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        score_adjustment=score_adjustment,
    )


class OverfittingDetector:
    """Detect overfitting across multiple param combos of the same strategy+ticker."""

    def detect_curve_fitting(self, results: list) -> tuple[bool, str]:
        """
        Check for overfitting across multiple param combos.

        Args:
            results: List of BacktestResult objects (same strategy+ticker, different params)

        Returns:
            (is_overfit, reason) tuple
        """
        if len(results) < 2:
            return False, "Insufficient results for overfitting detection"

        # ── 1. Performance degradation: first half vs second half ──
        sorted_results = sorted(results, key=lambda r: r.total_return, reverse=True)
        mid = len(sorted_results) // 2

        if mid == 0:
            top_half = sorted_results
            bottom_half = sorted_results
        else:
            top_half = sorted_results[:mid]
            bottom_half = sorted_results[mid:]

        top_avg = np.mean([r.total_return for r in top_half])
        bottom_avg = np.mean([r.total_return for r in bottom_half])

        if top_avg > 0 and bottom_avg <= 0:
            return True, (
                f"Performance degradation: top half avg return {top_avg:.1%} "
                f"vs bottom half {bottom_avg:.1%}"
            )

        # ── 2. Suspiciously high Sharpe with few trades ──
        for r in results:
            if r.sharpe > 3.0 and r.num_trades < 30 and r.num_trades > 0:
                return True, (
                    f"Suspiciously high Sharpe ({r.sharpe:.2f}) with only "
                    f"{r.num_trades} trades (iter={r.iteration})"
                )

        # ── 3. High variance across param combos ──
        returns = [r.total_return for r in results]
        returns_std = np.std(returns)
        returns_mean = np.mean(returns)

        if returns_mean > 0 and returns_std > abs(returns_mean):
            return True, (
                f"High return variance: std={returns_std:.1%} > mean={returns_mean:.1%} "
                f"(unstable across params)"
            )

        return False, ""
