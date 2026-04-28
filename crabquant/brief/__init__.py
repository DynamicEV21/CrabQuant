"""
CrabQuant Daily Market Brief

Generates a concise Telegram-friendly summary of market regime,
production strategies, recent discoveries, and cron health.
"""

import logging

from crabquant.brief.discoveries import (
    get_cron_status,
    get_promotion_metrics,
    get_recent_promotions,
    get_recent_winners,
    get_retirements,
)
from crabquant.brief.formatter import format_brief
from crabquant.brief.market import get_best_strategies_for_regime, get_market_regime
from crabquant.brief.models import BriefData

logger = logging.getLogger(__name__)


def generate_brief() -> str:
    """
    Generate the daily market brief.

    Returns:
        Formatted brief string, or "NO_REPLY" if nothing to report.
    """
    brief = BriefData()

    # ── Market regime ──
    try:
        regime_result = get_market_regime()
        brief.regime = regime_result["regime"]
        brief.regime_confidence = regime_result.get("confidence", 0.0)
        brief.spy_20d_return = regime_result.get("spy_20d_return")
        brief.realized_vol = regime_result.get("realized_vol")
    except Exception as e:
        logger.warning(f"Failed to get market regime: {e}")

    # ── Top production strategies for regime ──
    try:
        brief.top_production = get_best_strategies_for_regime(brief.regime)
        brief.regime_strategy_suggestions = _get_regime_suggestions(brief.regime)
    except Exception as e:
        logger.warning(f"Failed to get production strategies: {e}")

    # ── Recent activity (24h) ──
    try:
        recent = get_recent_winners(hours=24)
        brief.recent_winners_count = recent["count"]
        brief.total_combos_tested = recent.get("total_tested", 0)
    except Exception as e:
        logger.warning(f"Failed to get recent winners: {e}")

    try:
        brief.recent_promotions_count = get_recent_promotions(hours=24)
    except Exception as e:
        logger.warning(f"Failed to get promotions: {e}")

    try:
        brief.recent_retirements_count = get_retirements(hours=24)
    except Exception as e:
        logger.warning(f"Failed to get retirements: {e}")

    # ── Cron status ──
    try:
        cron = get_cron_status()
        brief.cron_active = cron["active"]
        brief.cron_total = cron["total"]
    except Exception as e:
        logger.warning(f"Failed to get cron status: {e}")

    # ── Promotion funnel ──
    try:
        brief.promotion_metrics = get_promotion_metrics()
    except Exception as e:
        logger.warning(f"Failed to get promotion metrics: {e}")

    # ── Format ──
    formatted = format_brief(brief)

    # If truly empty, signal no reply
    if not formatted.strip() or formatted.strip() == "NO_REPLY":
        return "NO_REPLY"

    return formatted


def _get_regime_suggestions(regime_name: str) -> list[tuple[str, float]]:
    """Get top strategy type suggestions for a regime."""
    from crabquant.regime import MarketRegime, REGIME_STRATEGY_AFFINITY

    # Map brief regime names back to enum
    regime_map = {
        "TRENDING_UP": MarketRegime.TRENDING_UP,
        "TRENDING_DOWN": MarketRegime.TRENDING_DOWN,
        "MEAN_REVERSION": MarketRegime.MEAN_REVERSION,
        "HIGH_VOLATILITY": MarketRegime.HIGH_VOLATILITY,
        "LOW_VOLATILITY": MarketRegime.LOW_VOLATILITY,
    }

    regime_enum = regime_map.get(regime_name)
    if regime_enum is None:
        return []

    affinity = REGIME_STRATEGY_AFFINITY.get(regime_enum, {})
    return sorted(affinity.items(), key=lambda x: x[1], reverse=True)[:5]


__all__ = ["generate_brief", "BriefData"]
