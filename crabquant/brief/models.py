"""
Data class for the daily brief.
"""

from dataclasses import dataclass, field


@dataclass
class BriefData:
    """All data needed to compose the daily brief."""

    regime: str = "UNKNOWN"
    regime_confidence: float = 0.0
    spy_20d_return: float | None = None
    realized_vol: float | None = None
    top_production: list[dict] = field(default_factory=list)
    recent_winners_count: int = 0
    recent_promotions_count: int = 0
    recent_retirements_count: int = 0
    total_combos_tested: int = 0
    cron_active: int = 0
    cron_total: int = 0
    regime_strategy_suggestions: list[tuple[str, float]] = field(default_factory=list)
