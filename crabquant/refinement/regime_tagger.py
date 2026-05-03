"""
Regime Tagger — compute preferred_regimes for strategies based on per-regime Sharpe.

After a strategy passes validation, compute its Sharpe in each market regime
and tag it with the regimes where it performs well.  This powers regime-aware
strategy selection in the execution layer (Phase 7).
"""

import logging
from typing import Optional

import pandas as pd

from crabquant.data import load_data
from crabquant.engine import BacktestEngine
from crabquant.regime import MarketRegime

logger = logging.getLogger(__name__)

# Thresholds for regime tagging
SHARPE_GOOD_THRESHOLD = 0.8  # Sharpe above this → "good" regime
SHARPE_ACCEPTABLE_THRESHOLD = 0.3  # Sharpe above this → "acceptable"
MIN_BARS_PER_REGIME = 20  # Need at least this many bars to trust regime Sharpe


def compute_strategy_regime_tags(
    strategy_fn,
    params: dict,
    ticker: str = "SPY",
    period: str = "3y",
    *,
    engine: Optional[BacktestEngine] = None,
) -> dict:
    """Compute per-regime Sharpe and return regime tags for a strategy.

    Runs the strategy on the full period, segments the equity curve by regime,
    and tags the strategy with regimes where it performs well.

    Args:
        strategy_fn: Strategy function (df, params) -> (entries, exits).
        params: Strategy parameters.
        ticker: Ticker to use (default SPY for market-level regime analysis).
        period: Data period (default 3y).
        engine: Optional BacktestEngine.

    Returns:
        Dict with keys:
            preferred_regimes: list[str] — regimes where Sharpe > GOOD threshold
            acceptable_regimes: list[str] — regimes where Sharpe > ACCEPTABLE threshold
            weak_regimes: list[str] — regimes where Sharpe < 0 (avoid)
            regime_sharpes: dict[str, float] — Sharpe per regime
            is_regime_specific: bool — True if performance varies significantly
    """
    if engine is None:
        engine = BacktestEngine()

    # Load data and run backtest
    try:
        df = load_data(ticker, period=period)
    except Exception as e:
        logger.warning("Could not load data for regime tagging: %s", e)
        return _empty_result()

    if len(df) < 200:
        return _empty_result()

    # Load SPY for regime detection
    try:
        spy_df = load_data("SPY", period=period)
    except Exception:
        spy_df = None

    # Run strategy
    try:
        entries, exits = strategy_fn(df, params)
        result = engine.run(df, entries, exits, strategy_fn.__name__, ticker, params=params)
    except Exception as e:
        logger.warning("Strategy failed during regime tagging: %s", e)
        return _empty_result()

    if result.num_trades < 5:
        return _empty_result()

    # Get regime labels for each bar
    regime_df = _label_regimes(df, spy_df)

    # Compute per-regime Sharpe using equity curve
    regime_sharpes = _compute_per_regime_sharpe(df, entries, result)

    if not regime_sharpes:
        return _empty_result()

    # Classify regimes
    preferred = [r for r, s in regime_sharpes.items() if s >= SHARPE_GOOD_THRESHOLD]
    acceptable = [r for r, s in regime_sharpes.items()
                  if SHARPE_ACCEPTABLE_THRESHOLD <= s < SHARPE_GOOD_THRESHOLD]
    weak = [r for r, s in regime_sharpes.items() if s < 0]

    # Determine if strategy is regime-specific
    sharpes = list(regime_sharpes.values())
    sharpe_range = max(sharpes) - min(sharpes) if len(sharpes) > 1 else 0
    any_negative = any(s < 0 for s in sharpes)
    is_regime_specific = sharpe_range > 1.5 or any_negative

    return {
        "preferred_regimes": sorted(preferred),
        "acceptable_regimes": sorted(acceptable),
        "weak_regimes": sorted(weak),
        "regime_sharpes": regime_sharpes,
        "is_regime_specific": is_regime_specific,
    }


def get_regime_strategies(
    current_regime: str,
    registry: dict,
    *,
    min_sharpe: float = 0.5,
) -> list[dict]:
    """Query the strategy registry for strategies that work in a given regime.

    Args:
        current_regime: Current market regime string (e.g., "trending_up").
        registry: STRATEGY_REGISTRY dict with regime metadata.
        min_sharpe: Minimum Sharpe threshold in the target regime.

    Returns:
        List of dicts with name, regime_sharpe, description, sorted by Sharpe desc.
    """
    results = []
    for name, entry in registry.items():
        # New format: entry is a dict with metadata
        if isinstance(entry, dict):
            regime_sharpes = entry.get("regime_sharpes", {})
            regime_sharpe = regime_sharpes.get(current_regime, 0)
            if regime_sharpe >= min_sharpe:
                results.append({
                    "name": name,
                    "regime_sharpe": regime_sharpe,
                    "all_regime_sharpes": regime_sharpes,
                    "description": entry.get("description", ""),
                })
        # Legacy format: entry is a tuple — no regime data available
        elif isinstance(entry, tuple) and len(entry) >= 4:
            # Can't filter by regime without data — include all
            results.append({
                "name": name,
                "regime_sharpe": None,
                "all_regime_sharpes": {},
                "description": entry[3] if entry[3] else "",
            })

    # Sort: strategies with known regime Sharpe first (highest first), then unknowns
    results.sort(key=lambda x: (x["regime_sharpe"] is None, -(x["regime_sharpe"] or 0)))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    return {
        "preferred_regimes": [],
        "acceptable_regimes": [],
        "weak_regimes": [],
        "regime_sharpes": {},
        "is_regime_specific": False,
    }


def _label_regimes(df: pd.DataFrame, spy_df: pd.DataFrame | None) -> pd.Series:
    """Assign a regime label to each bar in the DataFrame.

    Uses SPY data for regime detection when available, falls back to
    the ticker's own data.
    """
    from crabquant.regime import detect_regime

    source_df = spy_df if spy_df is not None and len(spy_df) > 0 else df

    # Align source to target index
    if source_df is not df:
        common_idx = df.index.intersection(source_df.index)
        source_df = source_df.loc[common_idx]
        df = df.loc[common_idx]

    labels = pd.Series("unknown", index=df.index)

    # Sliding window regime detection
    lookback = 50
    for i in range(lookback, len(df)):
        window = source_df.iloc[max(0, i - lookback):i]
        if len(window) < 20:
            continue
        try:
            regime, _ = detect_regime(window)
            labels.iloc[i] = regime.value
        except Exception:
            pass

    return labels


def _compute_per_regime_sharpe(
    df: pd.DataFrame,
    entries: pd.Series,
    result,
) -> dict[str, float]:
    """Compute annualized Sharpe for each contiguous regime segment.

    Segments the equity curve by regime transitions and computes per-segment
    Sharpe, then averages across segments of the same regime.
    """
    spy_df = None
    try:
        spy_df = load_data("SPY", period="3y")
    except Exception:
        pass

    labels = _label_regimes(df, spy_df)

    # Group by regime and compute per-regime returns
    regime_returns: dict[str, list[float]] = {}

    for regime_val in labels.unique():
        if regime_val == "unknown":
            continue
        mask = labels == regime_val
        if mask.sum() < MIN_BARS_PER_REGIME:
            continue

        # Compute daily returns for this regime segment
        regime_close = df.loc[mask, "close"]
        if len(regime_close) < 2:
            continue

        # Use strategy signal returns where available
        regime_entries = entries.reindex(regime_close.index).fillna(False)
        daily_ret = regime_close.pct_change().dropna()

        # Strategy return: hold when entry signal is True
        strat_ret = daily_ret.copy()
        # If strategy says exit, we're flat (0 return)
        strat_ret[~regime_entries.reindex(strat_ret.index).fillna(False)] = 0.0

        if len(strat_ret) < 10 or strat_ret.std() == 0:
            continue

        # Annualized Sharpe
        annual_factor = 252
        sharpe = (strat_ret.mean() / strat_ret.std()) * (annual_factor ** 0.5)
        regime_returns.setdefault(regime_val, []).append(sharpe)

    # Average across segments of the same regime
    regime_sharpes = {}
    for regime, sharpe_list in regime_returns.items():
        regime_sharpes[regime] = sum(sharpe_list) / len(sharpe_list)

    return regime_sharpes
