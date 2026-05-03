"""
HODL Baseline Comparison for CrabQuant Scoring Pipeline.

A strategy that can't beat buy-and-hold is useless.  This module computes
the buy-and-hold return and Sharpe ratio over the same price data that a
strategy was backtested on, then provides a penalty function and a
pass/fail check for the scoring pipeline.

Enhancement #11 — see VISION.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hodl_baseline(data: pd.DataFrame) -> dict:
    """Compute buy-and-hold metrics over a price DataFrame.

    Buy at the first close, sell at the last close.  Sharpe is computed
    from daily log returns (annualised at 252 trading days).

    Args:
        data: DataFrame with at least a ``'close'`` column.

    Returns:
        ``{"hodl_return": float, "hodl_sharpe": float}`` — return is the
        simple total return (last_close / first_close - 1), sharpe is the
        annualised Sharpe ratio of daily log returns.  Both default to
        ``0.0`` on edge cases.
    """
    # ── Edge cases ────────────────────────────────────────────────────────
    if data is None or len(data) < 2:
        return {"hodl_return": 0.0, "hodl_sharpe": 0.0}

    close = data["close"].astype(float)

    # Zero or negative prices → cannot compute meaningful returns
    if (close <= 0).any():
        return {"hodl_return": 0.0, "hodl_sharpe": 0.0}

    # ── Total return ──────────────────────────────────────────────────────
    first_close = float(close.iloc[0])
    last_close = float(close.iloc[-1])
    hodl_return = (last_close / first_close) - 1.0

    # ── Sharpe ratio from daily log returns ───────────────────────────────
    log_returns = np.log(close.values[1:] / close.values[:-1])
    n = len(log_returns)

    if n < 2:
        return {"hodl_return": hodl_return, "hodl_sharpe": 0.0}

    mean_lr = float(np.mean(log_returns))
    std_lr = float(np.std(log_returns, ddof=1))

    if std_lr < 1e-12:
        # Zero volatility → Sharpe is 0 (no risk-adjusted information)
        return {"hodl_return": hodl_return, "hodl_sharpe": 0.0}

    # Annualise: daily mean * 252 / (daily std * sqrt(252))
    hodl_sharpe = (mean_lr / std_lr) * np.sqrt(252)

    return {"hodl_return": hodl_return, "hodl_sharpe": hodl_sharpe}


def hodl_penalty(
    strategy_return: float,
    benchmark_return: float,
    threshold: float = 0.8,
) -> float:
    """Return a penalty when the strategy underperforms buy-and-hold.

    If the benchmark (HODL) return is positive and the strategy return is
    below ``threshold * benchmark_return``, apply a score penalty.

    Args:
        strategy_return: Total return of the strategy (e.g. 0.12 for 12 %).
        benchmark_return: Total return of buy-and-hold over the same period.
        threshold: Fraction of benchmark return the strategy must at least
            achieve to avoid a penalty (default 0.8).

    Returns:
        Penalty value (negative).  ``0.0`` when the strategy is acceptable.
    """
    if benchmark_return > 0 and strategy_return < benchmark_return * threshold:
        return -0.3
    return 0.0


def check_hodl_outperformance(
    strategy_sharpe: float,
    data: pd.DataFrame,
    margin: float = 1.1,
) -> tuple[bool, str]:
    """Check whether a strategy outperforms the HODL baseline by a margin.

    The strategy's Sharpe ratio must exceed ``margin * hodl_sharpe``.
    When HODL Sharpe is negative (downtrending market) the strategy
    automatically passes — any positive-Sharpe strategy beats a losing
    baseline.

    Args:
        strategy_sharpe: Sharpe ratio of the backtested strategy.
        data: Price DataFrame with a ``'close'`` column (same data the
            strategy was tested on).
        margin: Minimum ratio of strategy_sharpe / hodl_sharpe required
            (default 1.1 → strategy must be 10 % better than HODL).

    Returns:
        ``(passed, notes)`` — passed is ``True`` when the strategy
        sufficiently outperforms HODL.
    """
    baseline = hodl_baseline(data)
    hodl_sharpe = baseline["hodl_sharpe"]
    hodl_return = baseline["hodl_return"]

    # Edge case: could not compute HODL metrics
    if hodl_sharpe == 0.0 and hodl_return == 0.0:
        return (True, "HODL baseline unavailable — skipped")

    # HODL is losing money → any non-negative strategy beats it
    if hodl_sharpe <= 0:
        if strategy_sharpe > 0:
            return (True, f"Passed HODL check (strategy {strategy_sharpe:.2f} > 0, HODL {hodl_sharpe:.2f})")
        return (
            False,
            f"Strategy Sharpe {strategy_sharpe:.2f} does not beat HODL "
            f"(HODL Sharpe {hodl_sharpe:.2f}, HODL return {hodl_return:.1%})",
        )

    # Normal case: strategy must exceed HODL by margin
    required = hodl_sharpe * margin
    if strategy_sharpe >= required:
        return (
            True,
            f"Passed HODL check (strategy {strategy_sharpe:.2f} >= "
            f"{required:.2f} = HODL {hodl_sharpe:.2f} × {margin})",
        )

    return (
        False,
        f"Strategy Sharpe {strategy_sharpe:.2f} < {required:.2f} required "
        f"(HODL Sharpe {hodl_sharpe:.2f}, margin {margin})",
    )
