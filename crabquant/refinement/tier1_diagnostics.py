"""
Tier 1 Diagnostics — Enhanced Tier 1 diagnostic report building.

Extends the base diagnostics with:
- sharpe_by_year in the report
- previous_attempts with params and deltas
- consecutive modify_params detection for cosmetic guard
- Formatted previous attempts string for LLM context

These are always computed (~0.01s extra) and included in every BacktestReport.
"""

from typing import Any


def format_previous_attempts(history: list[dict]) -> str:
    """Format previous attempts for LLM context.

    Each attempt shows turn, sharpe, failure, action, hypothesis, params, and delta.

    Args:
        history: List of history dicts from RunState.

    Returns:
        Formatted multi-line string suitable for LLM prompt.
    """
    if not history:
        return ""

    lines = []
    for entry in history:
        turn = entry.get("turn", "?")
        sharpe = entry.get("sharpe", 0.0)
        failure = entry.get("failure_mode", "unknown")
        action = entry.get("action", "unknown")
        hypothesis = entry.get("hypothesis", "N/A")
        params = entry.get("params_used", {})
        delta = entry.get("delta_from_prev", "N/A")

        lines.append(
            f"Turn {turn}: Sharpe {sharpe:.2f}\n"
            f"  Failure: {failure}\n"
            f"  Action taken: {action}\n"
            f"  Hypothesis: \"{hypothesis}\"\n"
            f"  Params used: {params}\n"
            f"  What changed from prior: {delta}"
        )

    return "\n".join(lines)


def compute_consecutive_modify_params(history: list[dict]) -> int:
    """Count consecutive modify_params actions at the end of history.

    Used by the cosmetic improvement guard to force structural intervention.

    Args:
        history: List of history dicts from RunState.

    Returns:
        Number of consecutive modify_params at the tail of history.
    """
    if not history:
        return 0

    count = 0
    for entry in reversed(history):
        if entry.get("action") == "modify_params":
            count += 1
        else:
            break
    return count


def build_tier1_report(
    *,
    backtest_result: Any,
    failure_mode: str,
    failure_details: str,
    sharpe_by_year: dict[str, float],
    stagnation_score: float,
    stagnation_trend: str,
    history: list[dict],
    guardrail_violations: list[str],
    guardrail_warnings: list[str] | None = None,
    current_strategy_code: str = "",
    current_params: dict | None = None,
    strategy_id: str = "",
    iteration: int = 0,
) -> dict:
    """Build a Tier 1 diagnostic report dict.

    This is the always-computed (~0.01s) diagnostic payload that gets
    included in every BacktestReport and sent to the LLM.

    Args:
        backtest_result: BacktestResult from BacktestEngine.
        failure_mode: Classified failure mode string.
        failure_details: Human-readable failure explanation.
        sharpe_by_year: Annual Sharpe ratios dict.
        stagnation_score: Current stagnation score (0-1).
        stagnation_trend: "improving" | "flat" | "declining".
        history: RunState history entries.
        guardrail_violations: List of guardrail violation strings.
        guardrail_warnings: Optional list of guardrail warning strings.
        current_strategy_code: Full source of current strategy.
        current_params: Params used in this backtest.
        strategy_id: Strategy/run identifier.
        iteration: Current iteration number.

    Returns:
        Dict with all Tier 1 fields, ready for JSON serialization.
    """
    return {
        # Identity
        "strategy_id": strategy_id,
        "iteration": iteration,

        # Core metrics (mapped from BacktestResult field names)
        "sharpe_ratio": backtest_result.sharpe,
        "total_return_pct": backtest_result.total_return,
        "max_drawdown_pct": backtest_result.max_drawdown,
        "win_rate": backtest_result.win_rate,
        "total_trades": backtest_result.num_trades,
        "profit_factor": backtest_result.profit_factor,
        "calmar_ratio": backtest_result.calmar_ratio,
        "sortino_ratio": backtest_result.sortino_ratio,
        "composite_score": backtest_result.score,

        # Failure classification
        "failure_mode": failure_mode,
        "failure_details": failure_details,

        # Temporal resolution
        "sharpe_by_year": sharpe_by_year,

        # Stagnation context
        "stagnation_score": stagnation_score,
        "stagnation_trend": stagnation_trend,
        "previous_sharpes": [h.get("sharpe", 0.0) for h in history],
        "previous_actions": [h.get("action", "") for h in history],

        # Guardrail results
        "guardrail_violations": guardrail_violations,
        "guardrail_warnings": guardrail_warnings or [],

        # Current strategy context
        "current_strategy_code": current_strategy_code,
        "current_params": current_params or {},

        # Previous attempts (last 3 with full detail)
        "previous_attempts": history[-3:] if history else [],

        # Cosmetic guard: consecutive modify_params count
        "consecutive_modify_params": compute_consecutive_modify_params(history),
    }
