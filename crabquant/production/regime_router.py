"""
Portfolio Regime Router (5.5.7)

Routes capital allocation across strategies based on the current detected
market regime.  Uses regime Sharpe data to weight strategy selection and
sizing — strategies with higher regime Sharpe get more capital.

This is the execution layer that ties regime detection to actual portfolio
construction.  Inspired by QuantFactory's regime-weighted allocation but
adapted for CrabQuant's mandate-based architecture.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from crabquant.regime import MarketRegime, detect_regime
from crabquant.data import load_data
from crabquant.production.regime_scanner import detect_current_regime, scan_regime_strategies

logger = logging.getLogger(__name__)

# Default allocation parameters
DEFAULT_MAX_POSITIONS = 5
DEFAULT_MIN_WEIGHT = 0.05  # 5% minimum per position
DEFAULT_MAX_WEIGHT = 0.40  # 40% maximum per position (no single-strategy dominance)
DEFAULT_SHARPE_FLOOR = 0.0  # Below this, exclude from allocation


def route_portfolio(
    registry: dict,
    total_capital: float = 100_000.0,
    current_regime: MarketRegime | str | None = None,
    *,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    sharpe_floor: float = DEFAULT_SHARPE_FLOOR,
) -> dict:
    """Build a regime-weighted portfolio allocation.

    Detects the current market regime, scans the strategy registry for
    regime-fit strategies, and computes capital allocations based on
    regime Sharpe scores.

    Args:
        registry: STRATEGY_REGISTRY dict.
        total_capital: Total portfolio value in dollars.
        current_regime: Override regime (auto-detects if None).
        max_positions: Maximum number of strategies to allocate to.
        min_weight: Minimum allocation weight per strategy (0.0-1.0).
        max_weight: Maximum allocation weight per strategy (0.0-1.0).
        sharpe_floor: Minimum regime Sharpe to include.

    Returns:
        Dict with keys:
            regime: str — detected regime
            regime_confidence: float — detection confidence
            timestamp: str — routing time
            total_capital: float
            allocations: list[dict] — each with:
                name, weight, capital, regime_sharpe, is_regime_specific
            total_allocated: float — sum of allocated capital
            cash_reserve: float — unallocated capital
            num_strategies: int
    """
    # Scan strategies for current regime
    scan_result = scan_regime_strategies(
        registry,
        current_regime=current_regime,
        min_regime_sharpe=sharpe_floor,
        max_strategies=max_positions * 2,  # Get more than we need for filtering
    )

    strategies = scan_result["strategies"]
    regime_name = scan_result["regime"]
    confidence = scan_result["regime_confidence"]

    if not strategies:
        logger.info("No strategies matched regime %s — full cash reserve", regime_name)
        return {
            "regime": regime_name,
            "regime_confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_capital": total_capital,
            "allocations": [],
            "total_allocated": 0.0,
            "cash_reserve": total_capital,
            "num_strategies": 0,
            "reason": "no_strategies_matched_regime",
        }

    # Filter by Sharpe floor and take top N
    eligible = [s for s in strategies if (s.get("regime_sharpe") or 0) >= sharpe_floor]
    eligible = eligible[:max_positions]

    if not eligible:
        return {
            "regime": regime_name,
            "regime_confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_capital": total_capital,
            "allocations": [],
            "total_allocated": 0.0,
            "cash_reserve": total_capital,
            "num_strategies": 0,
            "reason": "no_strategies_above_sharpe_floor",
        }

    # Compute Sharpe-based weights
    sharpes = np.array([max(s.get("regime_sharpe", 0.0), 0.01) for s in eligible])

    # Use Sharpe directly as weight with a floor to avoid zero-weight
    weights = sharpes / sharpes.sum()

    # Apply min/max constraints
    allocations = []
    for i, strategy in enumerate(eligible):
        w = weights[i]

        # Enforce max weight — redistribute excess
        if w > max_weight:
            excess = w - max_weight
            w = max_weight
            # Redistribute excess proportionally to others
            other_indices = [j for j in range(len(weights)) if j != i]
            if other_indices:
                other_sum = weights[other_indices].sum()
                for j in other_indices:
                    weights[j] += excess * (weights[j] / other_sum) if other_sum > 0 else 0

        # Skip if below minimum weight (too small to matter)
        if w < min_weight:
            continue

        capital = w * total_capital
        allocations.append({
            "name": strategy["name"],
            "weight": round(w, 4),
            "capital": round(capital, 2),
            "regime_sharpe": strategy.get("regime_sharpe", 0.0),
            "is_regime_specific": strategy.get("is_regime_specific", False),
            "preferred_regimes": strategy.get("preferred_regimes", []),
        })

    # Re-normalize weights after filtering
    total_weight = sum(a["weight"] for a in allocations)
    if total_weight > 0:
        for a in allocations:
            a["weight"] = round(a["weight"] / total_weight, 4)
            a["capital"] = round(a["weight"] * total_capital, 2)

    total_allocated = sum(a["capital"] for a in allocations)

    return {
        "regime": regime_name,
        "regime_confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_capital": total_capital,
        "allocations": allocations,
        "total_allocated": round(total_allocated, 2),
        "cash_reserve": round(total_capital - total_allocated, 2),
        "num_strategies": len(allocations),
    }


def get_regime_summary() -> dict:
    """Quick summary of current regime and registry fitness.

    Returns a lightweight dict suitable for logging or dashboard display.
    Does NOT require network calls if regime data is cached.

    Returns:
        Dict with regime, strategy counts, and top matches.
    """
    from crabquant.strategies import STRATEGY_REGISTRY

    regime_enum, meta = detect_current_regime()
    regime_name = regime_enum.value

    scan = scan_regime_strategies(STRATEGY_REGISTRY, current_regime=regime_enum)

    return {
        "current_regime": regime_name,
        "confidence": meta.get("confidence", 0.0),
        "registry_total": len(STRATEGY_REGISTRY),
        "matched_strategies": scan["matched_count"],
        "skipped_weak": len(scan["skipped_regime_specific"]),
        "skipped_no_data": len(scan["skipped_no_data"]),
        "top_3": [
            {"name": s["name"], "sharpe": s["regime_sharpe"]}
            for s in scan["strategies"][:3]
        ],
    }
