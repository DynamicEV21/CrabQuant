"""
Deterministic failure mode classifier for backtested strategies.

Rules are checked in priority order — first match wins. No LLM calls.
"""

from crabquant.engine.backtest import BacktestResult
from crabquant.guardrails import GuardrailReport


def classify_failure(
    result: BacktestResult,
    guardrails: GuardrailReport,
    sharpe_by_year: dict[str, float],
    data_length: int = 500,
    sharpe_target: float = 1.5,
) -> tuple[str, str]:
    """
    Classify why a strategy failed.

    Returns (failure_mode, details). Rules evaluated in priority order;
    first match wins.

    Args:
        result: BacktestResult from the engine.
        guardrails: GuardrailReport (reserved for future use).
        sharpe_by_year: Per-year Sharpe ratios for regime fragility check.
        data_length: Number of bars in the backtest DataFrame.
        sharpe_target: Sharpe ratio target for the low_sharpe catch-all.
    """

    # 1. Too few trades (below minimum for statistical significance)
    # Threshold: 10 trades. Previously 20, lowered because 56% of recent
    # mandate failures were too_few_trades — strategies were structurally
    # sound but over-filtered. 10 trades provides adequate statistical
    # significance on daily-frequency 2y backtests while not penalising
    # legitimate conservative strategies.
    if result.num_trades < 10:
        return (
            "too_few_trades",
            f"Only {result.num_trades} trades (min 10 required). "
            f"Strategy is too selective — use shorter lookback windows (5-15 periods), "
            f"lower entry thresholds, or add a re-entry mechanism after stop-loss/cooldown.",
        )

    # 2. Flat signal (no meaningful activity)
    if result.num_trades == 0 or (result.total_return == 0 and result.sharpe == 0):
        return (
            "flat_signal",
            "Strategy produced zero meaningful signals or returns.",
        )

    # 3. Excessive drawdown
    if result.max_drawdown < -0.30:
        return (
            "excessive_drawdown",
            f"Max drawdown {result.max_drawdown:.1%} exceeds 30% threshold.",
        )

    # 4. Regime fragility
    if len(sharpe_by_year) >= 2:
        sharpe_values = list(sharpe_by_year.values())
        sharpe_range = max(sharpe_values) - min(sharpe_values)
        has_negative = any(s < 0 for s in sharpe_values)
        min_sharpe = min(sharpe_values)
        if (sharpe_range > 2.5 and has_negative) or (sharpe_range > 3.0 and min_sharpe < 0.3):
            return (
                "regime_fragility",
                f"Sharpe varies {sharpe_range:.1f} across years "
                f"({sharpe_by_year}). Strategy is regime-dependent.",
            )

    # 5. Overtrading (relative to data length)
    if data_length > 0 and result.num_trades > data_length * 0.5:
        return (
            "overtrading",
            f"{result.num_trades} trades on {data_length} bars "
            f"({result.num_trades / data_length:.0%}) "
            f"— transaction costs likely dominate.",
        )

    # 6. Low sharpe (default catch-all)
    return (
        "low_sharpe",
        f"Sharpe {result.sharpe:.2f} < target {sharpe_target}. "
        f"Return {result.total_return:.1%}, MaxDD {result.max_drawdown:.1%}."
        if result.sharpe < sharpe_target
        else "Below target but no specific failure pattern detected.",
    )
