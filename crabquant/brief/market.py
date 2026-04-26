"""
Market regime detection for the daily brief.

Wraps crabquant.regime.detect_regime() with SPY data loading,
returning a simple dict suitable for BriefData.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from crabquant.data import load_data
from crabquant.regime import MarketRegime, detect_regime, get_strategy_ranking

logger = logging.getLogger(__name__)

RESULTS_DIR = None  # lazily resolved


def _get_results_dir() -> str:
    """Resolve the CrabQuant results directory."""
    global RESULTS_DIR
    if RESULTS_DIR is None:
        from pathlib import Path

        RESULTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "results")
    return RESULTS_DIR


def get_market_regime() -> dict:
    """
    Detect current market regime using SPY data.

    Returns:
        Dict with keys: regime, confidence, spy_20d_return, realized_vol, scores
    """
    spy_data = load_data("SPY", period="2mo")

    regime, metadata = detect_regime(spy_data)

    # Calculate 20-day return
    close = spy_data["close"]
    spy_20d_return = None
    if len(close) >= 21:
        spy_20d_return = (close.iloc[-1] / close.iloc[-21] - 1) * 100

    return {
        "regime": regime.value,
        "confidence": metadata.get("confidence", 0.0),
        "spy_20d_return": round(spy_20d_return, 1) if spy_20d_return is not None else None,
        "realized_vol": metadata.get("realized_vol"),
        "scores": metadata.get("scores", {}),
    }


def get_best_strategies_for_regime(regime_name: str, top_n: int = 5) -> list[dict]:
    """
    Get top production strategies suited for the current regime.

    Returns:
        List of dicts with keys: ticker, strategy_name, sharpe, total_return, verdict
    """
    import json
    from pathlib import Path

    results_dir = _get_results_dir()
    registry_file = Path(results_dir).parent / "strategies" / "production" / "registry.json"
    confirmed_file = Path(results_dir) / "confirmed" / "confirmed.json"

    # Load confirmed strategies (these are the real "production" ones)
    confirmed = []
    if confirmed_file.exists():
        with open(confirmed_file) as f:
            confirmed = json.load(f)

    if not confirmed:
        return []

    # Map regime name to enum
    regime_map = {
        "trending_up": MarketRegime.TRENDING_UP,
        "trending_down": MarketRegime.TRENDING_DOWN,
        "mean_reversion": MarketRegime.MEAN_REVERSION,
        "high_volatility": MarketRegime.HIGH_VOLATILITY,
        "low_volatility": MarketRegime.LOW_VOLATILITY,
    }

    regime_enum = regime_map.get(regime_name.lower())
    if regime_enum is None:
        # Fall back to all strategies
        ranked = confirmed
    else:
        # Get available strategy types
        available_types = list({s["strategy"] for s in confirmed})
        ranking = get_strategy_ranking(regime_enum, available_types)
        rank_map = {name: score for name, score in ranking}

        # Sort confirmed by regime affinity, then by confirm_sharpe
        ranked = sorted(
            confirmed,
            key=lambda s: (rank_map.get(s["strategy"], 0), s.get("confirm_sharpe", 0)),
            reverse=True,
        )

    result = []
    for s in ranked[:top_n]:
        result.append({
            "ticker": s.get("ticker", "?"),
            "strategy_name": s.get("strategy", "?"),
            "sharpe": round(s.get("confirm_sharpe", 0), 2),
            "total_return": round(s.get("confirm_return", 0) * 100, 1),
            "verdict": s.get("verdict", ""),
            "discovery_regime": s.get("discovery_regime", ""),
        })

    return result
