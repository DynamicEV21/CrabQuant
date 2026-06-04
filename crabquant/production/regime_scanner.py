"""
Regime-Aware Scanner (5.5.6)

Scans the strategy registry and production strategies, filtering by the
current detected market regime.  Returns ranked strategy lists that can be
fed into the portfolio regime router for position sizing and allocation.

Inspired by QuantFactory's regime-based strategy selection — only run
strategies that are known to work in the current regime, and weight them
by their historical regime Sharpe.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from crabquant.regime import MarketRegime, detect_regime
from crabquant.data import load_data

logger = logging.getLogger(__name__)


def detect_current_regime(
    ticker: str = "SPY",
    period: str = "6mo",
) -> tuple[MarketRegime, dict]:
    """Detect the current market regime from price data.

    Args:
        ticker: Ticker to use for regime detection (default SPY).
        period: Data period to load.

    Returns:
        (MarketRegime, metadata) — regime enum and confidence scores.
    """
    try:
        df = load_data(ticker, period=period)
        regime, meta = detect_regime(df)
        return regime, meta
    except Exception as e:
        logger.warning("Regime detection failed: %s — defaulting to LOW_VOLATILITY", e)
        return MarketRegime.LOW_VOLATILITY, {"confidence": 0.0, "reason": "detection_failed"}


def scan_regime_strategies(
    registry: dict,
    current_regime: MarketRegime | str | None = None,
    *,
    min_regime_sharpe: float = 0.3,
    max_strategies: int = 20,
) -> dict:
    """Scan the strategy registry and rank by current regime fitness.

    Args:
        registry: STRATEGY_REGISTRY dict (name -> tuple or dict entry).
        current_regime: Override regime (auto-detects if None).
        min_regime_sharpe: Minimum Sharpe in current regime to include.
        max_strategies: Cap on returned strategies.

    Returns:
        Dict with keys:
            regime: str — detected regime name
            regime_confidence: float — detection confidence
            timestamp: str — scan time
            strategies: list[dict] — ranked strategies, each with:
                name, regime_sharpe, all_regime_sharpes, description,
                is_regime_specific, preferred_regimes, weak_regimes
            skipped_regime_specific: list[str] — strategies weak in current regime
            skipped_no_data: list[str] — strategies with no regime data (legacy)
    """
    # Detect regime
    if current_regime is None:
        regime_enum, meta = detect_current_regime()
        regime_name = regime_enum.value if isinstance(regime_enum, MarketRegime) else str(regime_enum)
        confidence = meta.get("confidence", 0.0)
    else:
        regime_name = current_regime.value if isinstance(current_regime, MarketRegime) else str(current_regime)
        confidence = 1.0
        regime_enum = current_regime

    matched = []
    skipped_weak = []
    skipped_no_data = []

    for name, entry in registry.items():
        if isinstance(entry, dict):
            regime_sharpes = entry.get("regime_sharpes", {})
            regime_sharpe = regime_sharpes.get(regime_name, 0.0)
            is_regime_specific = entry.get("is_regime_specific", False)
            weak_regimes = entry.get("weak_regimes", [])
            preferred = entry.get("preferred_regimes", [])

            # Skip strategies that are weak in the current regime
            if regime_name in weak_regimes:
                skipped_weak.append(name)
                continue

            # Include if Sharpe is above threshold, or if not regime-specific (generalists)
            if regime_sharpe >= min_regime_sharpe or (not is_regime_specific and regime_sharpe >= 0):
                matched.append({
                    "name": name,
                    "regime_sharpe": regime_sharpe,
                    "all_regime_sharpes": regime_sharpes,
                    "description": entry.get("description", ""),
                    "is_regime_specific": is_regime_specific,
                    "preferred_regimes": preferred,
                    "weak_regimes": weak_regimes,
                })

        elif isinstance(entry, tuple) and len(entry) >= 4:
            # Legacy tuple format — no regime data
            skipped_no_data.append(name)

    # Sort by regime Sharpe descending (None/0 last)
    matched.sort(key=lambda x: -x["regime_sharpe"])

    # Cap
    matched = matched[:max_strategies]

    return {
        "regime": regime_name,
        "regime_confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies": matched,
        "skipped_regime_specific": skipped_weak,
        "skipped_no_data": skipped_no_data,
        "total_registry": len(registry),
        "matched_count": len(matched),
    }


def scan_production_by_regime(
    current_regime: MarketRegime | str | None = None,
    *,
    min_regime_sharpe: float = 0.3,
) -> dict:
    """Scan production strategies and filter by current regime.

    Unlike scan_regime_strategies (which uses the in-memory STRATEGY_REGISTRY),
    this reads from the production registry file on disk.

    Returns:
        Same structure as scan_regime_strategies but from production registry.
    """
    from crabquant.production import get_production_strategies
    from crabquant.strategies import STRATEGY_REGISTRY

    # Detect regime
    if current_regime is None:
        regime_enum, meta = detect_current_regime()
        regime_name = regime_enum.value if isinstance(regime_enum, MarketRegime) else str(regime_enum)
        confidence = meta.get("confidence", 0.0)
    else:
        regime_name = current_regime.value if isinstance(current_regime, MarketRegime) else str(current_regime)
        confidence = 1.0

    # Get production strategies
    prod_strategies = get_production_strategies()

    # Enrich with regime data from STRATEGY_REGISTRY
    matched = []
    for ps in prod_strategies:
        key = ps["key"]
        # Try to find regime data in the strategy registry
        regime_data = {}
        # Key format: "strategy_name|ticker|hash"
        strategy_name = key.split("|")[0] if "|" in key else key
        if strategy_name in STRATEGY_REGISTRY:
            entry = STRATEGY_REGISTRY[strategy_name]
            if isinstance(entry, dict):
                regime_data = {
                    "regime_sharpes": entry.get("regime_sharpes", {}),
                    "is_regime_specific": entry.get("is_regime_specific", False),
                    "preferred_regimes": entry.get("preferred_regimes", []),
                    "weak_regimes": entry.get("weak_regimes", []),
                }

        regime_sharpes = regime_data.get("regime_sharpes", {})
        regime_sharpe = regime_sharpes.get(regime_name, 0.0)
        weak_regimes = regime_data.get("weak_regimes", [])

        if regime_name in weak_regimes:
            continue

        if regime_sharpe >= min_regime_sharpe or not regime_data:
            matched.append({
                "key": key,
                "strategy_name": ps["strategy_name"],
                "ticker": ps["ticker"],
                "regime_sharpe": regime_sharpe,
                "all_regime_sharpes": regime_sharpes,
                "promoted_at": ps.get("promoted_at", ""),
                **regime_data,
            })

    matched.sort(key=lambda x: -(x.get("regime_sharpe") or 0))

    return {
        "regime": regime_name,
        "regime_confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies": matched,
        "total_production": len(prod_strategies),
        "matched_count": len(matched),
    }
