"""
Strategy Correlation Analyzer

Computes signal correlation between strategies to detect redundant or
overlapping strategies in the registry. Useful for:
- Avoiding promotion of near-duplicate strategies
- Building diversified portfolios
- Understanding indicator overlap in the strategy library

This is a Phase 7 prep item (portfolio correlation gate) that can be
used immediately for analysis.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from crabquant.data import load_data

logger = logging.getLogger(__name__)


def compute_signal_correlation(
    df: pd.DataFrame,
    signals_a: pd.Series,
    signals_b: pd.Series,
) -> dict:
    """Compute correlation between two signal series.

    Args:
        df: Price dataframe (unused directly, but ensures date alignment).
        signals_a: First signal series (boolean or 1/0/-1).
        signals_b: Second signal series (boolean or 1/0/-1).

    Returns:
        Dict with correlation metrics:
            pearson: Pearson correlation of the two signals
            agreement_rate: Fraction of bars where both signals agree (both long or both flat/short)
            jaccard_long: Jaccard similarity of long-entry bars only
            anti_correlation: Whether signals are anti-correlated (< -0.3)
    """
    # Align signals to the union of both indices
    combined_index = signals_a.index.union(signals_b.index)
    s1 = signals_a.reindex(combined_index).fillna(0).infer_objects(copy=False).astype(float)
    s2 = signals_b.reindex(combined_index).fillna(0).infer_objects(copy=False).astype(float)

    # Compute pearson — handle constant signals (std=0 → NaN from corrcoef)
    if len(s1) <= 1:
        pearson = 0.0
    else:
        std1 = np.std(s1.values, ddof=1)
        std2 = np.std(s2.values, ddof=1)
        if std1 < 1e-10 or std2 < 1e-10:
            # Constant signal: correlation is 1.0 if both constant with same value, else 0.0
            pearson = 1.0 if np.all(s1.values == s2.values) else 0.0
        else:
            pearson = float(np.corrcoef(s1.values, s2.values)[0, 1])

    # Agreement rate: both positive or both zero/negative
    agreement = ((s1 > 0) & (s2 > 0)) | ((s1 <= 0) & (s2 <= 0))
    agreement_rate = float(agreement.sum() / len(agreement)) if len(agreement) > 0 else 0.0

    # Jaccard similarity of long-entry bars
    long_a = set(s1[s1 > 0].index)
    long_b = set(s2[s2 > 0].index)
    union = long_a | long_b
    jaccard_long = len(long_a & long_b) / len(union) if union else 0.0

    return {
        "pearson": round(pearson, 4),
        "agreement_rate": round(agreement_rate, 4),
        "jaccard_long": round(jaccard_long, 4),
        "anti_correlated": pearson < -0.3,
    }


def analyze_strategy_pair(
    strategy_a: str,
    strategy_b: str,
    ticker: str = "SPY",
    period: str = "1y",
    params_a: Optional[dict] = None,
    params_b: Optional[dict] = None,
) -> dict:
    """Analyze correlation between two strategies.

    Loads data, runs both strategies, and computes signal correlation.

    Args:
        strategy_a: Name of first strategy (must be in STRATEGY_REGISTRY).
        strategy_b: Name of second strategy.
        ticker: Ticker to analyze on.
        period: Data period.
        params_a: Override params for strategy A (uses defaults if None).
        params_b: Override params for strategy B.

    Returns:
        Dict with:
            strategy_a, strategy_b, ticker
            correlation: dict from compute_signal_correlation
            returns_correlation: Pearson correlation of daily returns
    """
    from crabquant.strategies import STRATEGY_REGISTRY

    df = load_data(ticker, period=period)

    # Get strategies
    entry_a = STRATEGY_REGISTRY.get(strategy_a)
    entry_b = STRATEGY_REGISTRY.get(strategy_b)

    if entry_a is None:
        raise ValueError(f"Strategy '{strategy_a}' not in registry")
    if entry_b is None:
        raise ValueError(f"Strategy '{strategy_b}' not in registry")

    # Extract generate_signals function
    gen_a = entry_a[0] if isinstance(entry_a, tuple) else entry_a["generate_signals"]
    gen_b = entry_b[0] if isinstance(entry_b, tuple) else entry_b["generate_signals"]

    # Get params
    p_a = params_a or (entry_a[1] if isinstance(entry_a, tuple) else entry_a.get("params", {}))
    p_b = params_b or (entry_b[1] if isinstance(entry_b, tuple) else entry_b.get("params", {}))

    # Generate signals
    signals_a = gen_a(df.copy(), p_a)
    signals_b = gen_b(df.copy(), p_b)

    # Compute signal correlation
    signal_corr = compute_signal_correlation(df, signals_a, signals_b)

    # Compute returns correlation
    returns_a = df["close"].pct_change() * signals_a.reindex(df.index).fillna(0).astype(float)
    returns_b = df["close"].pct_change() * signals_b.reindex(df.index).fillna(0).astype(float)

    _ra = returns_a.dropna().values
    _rb = returns_b.dropna().values
    if len(_ra) > 1 and np.std(_ra, ddof=1) > 1e-10 and np.std(_rb, ddof=1) > 1e-10:
        returns_corr = float(np.corrcoef(_ra, _rb)[0, 1])
    else:
        returns_corr = 0.0

    return {
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "ticker": ticker,
        "period": period,
        "signal_correlation": signal_corr,
        "returns_correlation": round(returns_corr, 4),
        "is_duplicate": signal_corr["pearson"] > 0.8 and signal_corr["jaccard_long"] > 0.7,
    }


def scan_registry_correlations(
    ticker: str = "SPY",
    period: str = "6mo",
    max_strategies: int = 20,
    duplicate_threshold: float = 0.8,
) -> dict:
    """Scan all strategies in the registry and find highly correlated pairs.

    Args:
        ticker: Ticker to analyze on.
        period: Data period (shorter for speed).
        max_strategies: Maximum strategies to analyze (N^2 pairs).
        duplicate_threshold: Pearson threshold to flag as duplicate.

    Returns:
        Dict with:
            total_analyzed: number of strategies checked
            total_pairs: number of pairs compared
            duplicates: list of (name_a, name_b, correlation) tuples above threshold
            high_correlations: list of (name_a, name_b, correlation) for 0.5-0.8 range
            all_results: full matrix of correlations
    """
    from crabquant.strategies import STRATEGY_REGISTRY

    names = list(STRATEGY_REGISTRY.keys())[:max_strategies]
    n = len(names)
    logger.info(f"Analyzing correlations for {n} strategies...")

    duplicates = []
    high_corr = []
    all_results = []

    for i in range(n):
        for j in range(i + 1, n):
            try:
                result = analyze_strategy_pair(names[i], names[j], ticker, period)
                pearson = result["signal_correlation"]["pearson"]

                pair = (names[i], names[j], pearson)
                all_results.append(pair)

                if result["is_duplicate"]:
                    duplicates.append(pair)
                elif pearson > 0.5:
                    high_corr.append(pair)

            except Exception as e:
                logger.warning(f"Error comparing {names[i]} vs {names[j]}: {e}")

    # Sort by correlation descending
    duplicates.sort(key=lambda x: -x[2])
    high_corr.sort(key=lambda x: -x[2])
    all_results.sort(key=lambda x: -x[2])

    return {
        "ticker": ticker,
        "period": period,
        "total_analyzed": n,
        "total_pairs": n * (n - 1) // 2,
        "duplicates": duplicates,
        "high_correlations": high_corr,
        "all_results": all_results[:50],  # Top 50
    }
