"""
Tests for crabquant.brief.models module.
"""

from crabquant.brief.models import BriefData


class TestBriefData:

    def test_default_creation(self):
        brief = BriefData()
        assert brief.regime == "UNKNOWN"
        assert brief.regime_confidence == 0.0
        assert brief.spy_20d_return is None
        assert brief.realized_vol is None
        assert brief.top_production == []
        assert brief.recent_winners_count == 0
        assert brief.recent_promotions_count == 0
        assert brief.recent_retirements_count == 0
        assert brief.total_combos_tested == 0
        assert brief.cron_active == 0
        assert brief.cron_total == 0
        assert brief.regime_strategy_suggestions == []
        assert brief.promotion_metrics == {}

    def test_custom_values(self):
        brief = BriefData(
            regime="trending_up",
            regime_confidence=0.9,
            spy_20d_return=3.5,
            realized_vol=0.18,
            top_production=[{"ticker": "SPY"}],
            recent_winners_count=5,
            cron_active=3,
            cron_total=4,
        )
        assert brief.regime == "trending_up"
        assert brief.regime_confidence == 0.9
        assert brief.spy_20d_return == 3.5
        assert brief.realized_vol == 0.18
        assert brief.top_production == [{"ticker": "SPY"}]
        assert brief.recent_winners_count == 5
        assert brief.cron_active == 3
        assert brief.cron_total == 4

    def test_mutable_defaults_are_independent(self):
        """Each instance should get its own list/dict, not share defaults."""
        b1 = BriefData()
        b2 = BriefData()
        b1.top_production.append({"ticker": "SPY"})
        b1.promotion_metrics["key"] = "value"
        assert b2.top_production == []
        assert b2.promotion_metrics == {}

    def test_regime_strategy_suggestions_type(self):
        brief = BriefData(regime_strategy_suggestions=[("sma_cross", 0.9)])
        assert brief.regime_strategy_suggestions[0] == ("sma_cross", 0.9)

    def test_dataclass_repr(self):
        brief = BriefData(regime="trending_up", cron_active=2, cron_total=3)
        r = repr(brief)
        assert "BriefData" in r
        assert "trending_up" in r
