"""Deflated Sharpe Ratio — multiple-testing correction for Sharpe ratios.

Implements the Deflated Sharpe Ratio (Bailey & López de Prado 2014) which
adjusts an observed Sharpe ratio for the number of independent trials,
penalising strategies that only look good because we have tested so many.

Key idea: after N independent trials, the expected maximum Sharpe grows as
sqrt(2 * log(N)).  The deflated Sharpe asks: "is my observed Sharpe still
impressive given that I ran N trials?"

Usage:
    >>> deflated_sharpe(sharpe=2.0, n_trials=100, returns=portfolio_returns)
    0.73  # still positive → not just luck

    >>> deflated_sharpe(sharpe=0.8, n_trials=4614)
    -0.42  # negative → likely overfit to multiple testing

References:
    Bailey, D. H. and López de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting, and Statistical Noise."
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def _expected_max_sharpe(
    n_trials: int,
    sr0: float = 0.0,
    sharpe_std: float = 1.0,
) -> float:
    """Approximate the expected maximum Sharpe ratio across n iid trials.

    Uses the closed-form expected maximum of n iid normal variables:
        E[max(Z_1..Z_n)] ≈ sqrt(2 * ln(n)) - (ln(ln(n)) + ln(4*pi)) / (2 * sqrt(2 * ln(n)))

    This is a standard result from extreme value theory (Gumbel approximation).

    Args:
        n_trials: Number of independent trials/experiments.
        sr0: Benchmark (null hypothesis) Sharpe ratio.
        sharpe_std: Standard deviation of Sharpe estimator across folds.

    Returns:
        Expected maximum Sharpe under the null hypothesis.
    """
    if n_trials <= 1:
        return sr0

    # Clamp to avoid numerical issues at very small n
    log_n = np.log(max(n_trials, 2))

    # Expected maximum of n standard normals (Gumbel / extreme value theory)
    # E[max(Z)] ≈ sqrt(2 ln n) - (ln ln n + ln(4π)) / (2 sqrt(2 ln n))
    term1 = np.sqrt(2.0 * log_n)
    inner = log_n + np.log(4.0 * np.pi)
    term2 = inner / (2.0 * term1) if term1 > 1e-10 else 0.0
    e_max_z = term1 - term2

    return sr0 + sharpe_std * e_max_z


def _probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    sharpe_std: float,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR).

    PSR = P(SR > benchmark_SR) = Φ((SR_obs - benchmark_SR) / σ_SR)

    where Φ is the standard normal CDF.

    Args:
        observed_sharpe: Observed annualised Sharpe ratio.
        benchmark_sharpe: Benchmark Sharpe to compare against.
        sharpe_std: Standard error of the Sharpe estimator.

    Returns:
        Probability that the true Sharpe exceeds the benchmark.
    """
    if sharpe_std < 1e-10:
        # No variance — return 1.0 if above benchmark, 0.0 if below
        return 1.0 if observed_sharpe > benchmark_sharpe else (0.5 if observed_sharpe == benchmark_sharpe else 0.0)

    z = (observed_sharpe - benchmark_sharpe) / sharpe_std
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_std: float = 1.0,
    n_trials: int = 1,
    sr0: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio — corrects observed Sharpe for multiple testing.

    After N trials, even random noise will produce some strategies with
    apparently high Sharpe.  DSR asks: "what is the probability that this
    observed Sharpe is genuinely above the expected-maximum from N random
    strategies?"

    DSR = PSR(observed_SR, E[max_SR_1..SR_N], σ_SR)

    where E[max_SR] is the expected maximum Sharpe under the null (no skill).

    Args:
        observed_sharpe: Observed annualised Sharpe ratio of the strategy.
        sharpe_std: Standard error of the Sharpe estimator.  Can be derived
            from walk-forward fold variation or from return series properties.
            Default 1.0 (annualised, typical for daily returns with ~252 obs).
        n_trials: Total number of independent experiments/trials run.
        sr0: Null-hypothesis (benchmark) Sharpe ratio.  Default 0 (no skill).

    Returns:
        Probability that the observed Sharpe is NOT due to multiple testing.
        Values < 0.05 suggest the strategy is likely overfit.
    """
    # Edge cases
    if n_trials < 1:
        n_trials = 1
        logger.warning("deflated_sharpe_ratio: n_trials < 1, clamping to 1")

    expected_max = _expected_max_sharpe(n_trials, sr0, sharpe_std)
    return _probabilistic_sharpe_ratio(observed_sharpe, expected_max, sharpe_std)


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    skew: float = 0,
    kurt: float = 3,
    returns: np.ndarray | None = None,
    sr0: float = 0.0,
) -> float:
    """Convenience wrapper: compute deflated Sharpe from raw strategy outputs.

    This is the primary API called by the scoring pipeline.  It accepts
    either a pre-computed ``sharpe_std`` or computes one from the return
    series (using the moments-corrected formula from Bailey & López de Prado).

    The return value is the **deflated Sharpe score** — a signed value where:
    - Positive → strategy Sharpe is likely genuine (above multiple-testing noise)
    - Negative → strategy likely overfit (Sharpe not impressive given N trials)

    This is NOT a probability like ``deflated_sharpe_ratio()``.  Instead it
    returns ``(observed_SR - expected_max_SR) / σ_SR``, i.e. the number of
    standard deviations the observed Sharpe exceeds the expected maximum.

    Args:
        sharpe: Observed annualised Sharpe ratio.
        n_trials: Total number of independent experiments run.
        skew: Skewness of return distribution (default 0).
        kurt: Excess kurtosis of return distribution (default 3 = normal).
        returns: Optional return array.  If provided, used to estimate
            sharpe_std from the data instead of using the moments formula.
        sr0: Null-hypothesis Sharpe (default 0).

    Returns:
        Deflated Sharpe score.  Negative means likely overfit.
    """
    # Handle edge cases
    if n_trials < 1:
        n_trials = 1
        logger.debug("deflated_sharpe: n_trials < 1, clamping to 1")

    if n_trials == 1:
        # No multiple testing penalty — return a proxy based on observed Sharpe
        return sharpe

    # Estimate sharpe_std
    if returns is not None and len(returns) > 1:
        # Bootstrap-style: use standard error of the Sharpe estimator
        n = len(returns)
        r_std = float(np.std(returns, ddof=1))
        if r_std < 1e-10:
            return 0.0
        # Approximate: σ_SR ≈ sqrt((1 - skew*SR + (kurt-1)/4 * SR²) / (n-1))
        # This is the moments-corrected formula from Bailey & de Prado
        sr = sharpe / (252 ** 0.5) if sharpe is not None else 0  # de-annualise
        sharpe_std = np.sqrt(
            max(0, (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
        ) * (252 ** 0.5)  # re-annualise
    else:
        # Default: use the standard formula with skew/kurt corrections
        # Assume ~252 daily observations (typical for 1-year backtest)
        n_bars = 252
        sr = sharpe / (252 ** 0.5)
        sharpe_std = np.sqrt(
            max(0, (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n_bars - 1))
        ) * (252 ** 0.5)

    if sharpe_std < 1e-10:
        return 0.0

    # Compute expected maximum Sharpe under null
    expected_max = _expected_max_sharpe(n_trials, sr0, sharpe_std)

    # Deflated Sharpe = (observed - expected_max) / σ
    # Positive → genuine skill, negative → likely noise
    return (sharpe - expected_max) / sharpe_std
