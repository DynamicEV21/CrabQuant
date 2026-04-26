"""
Batch Confirmation

Runs confirmation backtests across multiple time periods and slippage levels
to determine if a strategy is ROBUST, FRAGILE, or FAILED.
"""

import logging
from typing import Optional

import pandas as pd

from crabquant.confirm import ConfirmationResult, CONFIRM_THRESHOLDS
from crabquant.confirm.runner import run_confirmation
from crabquant.data import load_data

logger = logging.getLogger(__name__)

# Periods to test across
PERIODS = ["2y", "1y", "6mo"]

# Slippage levels to stress-test
SLIPPAGE_LEVELS = [0.0, 0.001, 0.002]  # 0%, 0.1%, 0.2%


def _aggregate_results(results: list[ConfirmationResult]) -> tuple[bool, bool, bool]:
    """
    Analyze batch results to determine verdict.

    Returns:
        (base_pass, robust_pass, fragile)
        - base_pass: passes at 0% slippage on primary period
        - robust_pass: passes at 0.2% slippage on primary period
        - fragile: passes base but fails robust
    """
    # Primary period is first result at each slippage level
    # Results are ordered: period0_slip0, period0_slip1, period0_slip2, period1_slip0, ...

    # Extract results by slippage level for the primary period (2y)
    primary_results = results[:len(SLIPPAGE_LEVELS)]  # First period's results

    if not primary_results:
        return False, False, False

    # Base pass: passes at 0% slippage on primary period
    base_pass = primary_results[0].passed if len(primary_results) > 0 else False

    # Robust pass: passes at 0.2% slippage on primary period
    robust_pass = primary_results[2].passed if len(primary_results) > 2 else False

    # Also check: passes on at least 2 out of 3 periods at 0% slippage
    period_zero_slip = [results[i * len(SLIPPAGE_LEVELS)] for i in range(len(PERIODS))
                        if i * len(SLIPPAGE_LEVELS) < len(results)]
    period_pass_count = sum(1 for r in period_zero_slip if r.passed)
    multi_period_pass = period_pass_count >= 2

    fragile = base_pass and not robust_pass
    overall_robust = robust_pass and multi_period_pass

    return base_pass, overall_robust, fragile


def batch_confirm(
    winner: dict,
    n_periods: int = 3,
) -> ConfirmationResult:
    """
    Run batch confirmation on a winner from winners.json.

    Tests the strategy across multiple time periods and slippage levels
    to determine if the VectorBT result holds up under realistic conditions.

    Args:
        winner: Dict with keys: strategy, ticker, params, (optional: sharpe, return, etc.)
        n_periods: Number of time periods to test (default 3)

    Returns:
        ConfirmationResult with verdict: ROBUST, FRAGILE, or FAILED
    """
    strategy_name = winner["strategy"]
    ticker = winner["ticker"]
    params = winner["params"]

    periods = PERIODS[:n_periods]
    all_results = []
    all_notes = []

    # Load data once per period and test all slippage levels
    for period in periods:
        try:
            df = load_data(ticker, period=period)
        except Exception as e:
            all_notes.append(f"Failed to load {ticker} ({period}): {e}")
            # Pad with failed results
            for slip in SLIPPAGE_LEVELS:
                all_results.append(ConfirmationResult(
                    notes=[f"No data for {period}"], passed=False
                ))
            continue

        if len(df) < 100:
            all_notes.append(f"Insufficient data for {ticker} ({period}): {len(df)} bars")
            for slip in SLIPPAGE_LEVELS:
                all_results.append(ConfirmationResult(
                    notes=[f"Insufficient data: {len(df)} bars"], passed=False
                ))
            continue

        for slip in SLIPPAGE_LEVELS:
            result = run_confirmation(
                strategy_name, ticker, params,
                df=df.copy(),
                slippage_pct=slip,
            )
            result.notes.insert(0, f"{period} @ {slip*100:.1f}% slip")
            all_results.append(result)
            logger.debug(f"  {period} slip={slip*100:.1f}%: sharpe={result.sharpe:.2f} "
                        f"ret={result.total_return:.2%} dd={result.max_dd:.2%} "
                        f"trades={result.trades} passed={result.passed}")

    # Determine verdict
    base_pass, robust_pass, fragile = _aggregate_results(all_results)

    if robust_pass:
        verdict = "ROBUST"
    elif fragile:
        verdict = "FRAGILE"
    else:
        verdict = "FAILED"

    # Use the primary period, 0% slippage result as the base stats
    primary = all_results[0] if all_results else ConfirmationResult()

    # Aggregate notes
    summary_notes = [f"Verdict: {verdict}"]
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        summary_notes.append(f"  {' | '.join(r.notes[:1])}: {status}")

    # Check if any result had trades
    any_trades = any(r.trades > 0 for r in all_results)

    return ConfirmationResult(
        sharpe=primary.sharpe,
        total_return=primary.total_return,
        max_dd=primary.max_dd,
        trades=primary.trades,
        win_rate=primary.win_rate,
        profit_factor=primary.profit_factor,
        expectancy=primary.expectancy,
        passed=verdict in ("ROBUST", "FRAGILE"),
        verdict=verdict,
        notes=summary_notes,
    )
