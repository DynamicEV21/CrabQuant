"""
Tests for crabquant.brief.formatter module.
"""

import pytest

from crabquant.brief.formatter import _regime_short_tag, format_brief
from crabquant.brief.models import BriefData


class TestRegimeShortTag:

    def test_trending_up(self):
        assert _regime_short_tag("trending_up") == "BULL"

    def test_trending_down(self):
        assert _regime_short_tag("trending_down") == "BEAR"

    def test_mean_reversion(self):
        assert _regime_short_tag("mean_reversion") == "MR"

    def test_high_volatility(self):
        assert _regime_short_tag("high_volatility") == "HVOL"

    def test_low_volatility(self):
        assert _regime_short_tag("low_volatility") == "LVOL"

    def test_unknown(self):
        assert _regime_short_tag("unknown") == "???"

    def test_unrecognized_truncates(self):
        result = _regime_short_tag("some_custom_regime")
        # .upper()[:5] on "some_custom_regime" → "SOME_"
        assert result == "SOME_"


class TestFormatBrief:

    def _minimal_brief(self, **overrides):
        defaults = {
            "regime": "trending_up",
            "regime_confidence": 0.8,
            "spy_20d_return": 2.5,
            "realized_vol": 0.15,
            "top_production": [],
            "recent_winners_count": 0,
            "recent_promotions_count": 0,
            "recent_retirements_count": 0,
            "total_combos_tested": 0,
            "cron_active": 2,
            "cron_total": 5,
        }
        defaults.update(overrides)
        return BriefData(**defaults)

    def test_contains_header(self):
        brief = self._minimal_brief()
        result = format_brief(brief)
        assert "📊 CrabQuant Daily Brief" in result

    def test_contains_regime(self):
        brief = self._minimal_brief(regime="trending_up")
        result = format_brief(brief)
        assert "TRENDING UP" in result

    def test_spy_20d_return_displayed(self):
        brief = self._minimal_brief(spy_20d_return=3.7)
        result = format_brief(brief)
        assert "SPY 20d: +3.7%" in result

    def test_spy_20d_negative(self):
        brief = self._minimal_brief(spy_20d_return=-1.2)
        result = format_brief(brief)
        assert "SPY 20d: -1.2%" in result

    def test_spy_20d_none_omitted(self):
        brief = self._minimal_brief(spy_20d_return=None)
        result = format_brief(brief)
        assert "SPY 20d" not in result

    def test_volatility_displayed(self):
        brief = self._minimal_brief(realized_vol=0.18)
        result = format_brief(brief)
        assert "Vol: 18.0" in result

    def test_volatility_none_omitted(self):
        brief = self._minimal_brief(realized_vol=None)
        result = format_brief(brief)
        assert "Vol:" not in result

    def test_no_production_strategies_message(self):
        brief = self._minimal_brief(top_production=[])
        result = format_brief(brief)
        assert "No production strategies yet" in result

    def test_top_production_displayed(self):
        brief = self._minimal_brief(top_production=[
            {"ticker": "SPY", "strategy_name": "sma_cross", "sharpe": 1.5, "total_return": 12.3,
             "discovery_regime": "trending_up"}
        ])
        result = format_brief(brief)
        assert "🏆 Top Production:" in result
        assert "SPY/sma_cross" in result
        assert "Sharpe 1.5" in result
        assert "+12.3%" in result
        assert "[BULL]" in result

    def test_regime_strategy_suggestions(self):
        brief = self._minimal_brief(
            top_production=[{"ticker": "SPY", "strategy_name": "x", "sharpe": 1.0, "total_return": 5.0,
                             "discovery_regime": ""}],
            regime_strategy_suggestions=[("strat_a", 0.9), ("strat_b", 0.7)],
        )
        result = format_brief(brief)
        assert "💡 Best for [BULL]:" in result
        assert "strat_a" in result

    def test_recent_activity_section(self):
        brief = self._minimal_brief(
            recent_winners_count=3,
            recent_promotions_count=1,
            recent_retirements_count=1,
            total_combos_tested=42,
        )
        result = format_brief(brief)
        assert "📈 Last 24h:" in result
        assert "3 new winners" in result
        assert "1 promoted" in result
        assert "1 retired" in result
        assert "42 combos tested" in result

    def test_no_activity_message(self):
        brief = self._minimal_brief()
        result = format_brief(brief)
        assert "No new discoveries" in result

    def test_cron_status(self):
        brief = self._minimal_brief(cron_active=3, cron_total=7)
        result = format_brief(brief)
        assert "🤖 Crons: 3/7 active" in result

    def test_promotion_metrics_displayed(self):
        brief = self._minimal_brief(promotion_metrics={
            "total_winners": 10,
            "walk_forward_passed_count": 7,
            "confirmed_count": 5,
            "promoted_count": 3,
            "promotion_rate": 0.3,
        })
        result = format_brief(brief)
        assert "🔬 Pipeline Conversion:" in result
        assert "Backtest Winners: 10" in result
        assert "Walk-Forward Passed: 7" in result
        assert "Confirmed: 5" in result
        assert "Promoted to Registry: 3" in result
        assert "Conversion Rate: 30.0%" in result

    def test_promotion_metrics_no_winners_omitted(self):
        brief = self._minimal_brief(promotion_metrics={"total_winners": 0})
        result = format_brief(brief)
        assert "Pipeline Conversion" not in result

    def test_under_800_chars(self):
        brief = self._minimal_brief(
            top_production=[
                {"ticker": f"T{i}", "strategy_name": f"s{i}", "sharpe": 1.0,
                 "total_return": 5.0, "discovery_regime": "trending_up"}
                for i in range(20)
            ],
            promotion_metrics={
                "total_winners": 100,
                "walk_forward_passed_count": 80,
                "confirmed_count": 60,
                "promoted_count": 40,
                "promotion_rate": 0.4,
            },
            regime_strategy_suggestions=[(f"strat_{i}", 0.5) for i in range(10)],
        )
        result = format_brief(brief)
        assert len(result) <= 800

    def test_unknown_regime_tag(self):
        brief = self._minimal_brief(regime="some_weird_regime")
        result = format_brief(brief)
        assert "SOME_WEIRD_REGIME" in result or "SOME W" in result

    def test_no_regime_tag_for_unknown_discovery(self):
        brief = self._minimal_brief(top_production=[
            {"ticker": "SPY", "strategy_name": "x", "sharpe": 1.0, "total_return": 5.0,
             "discovery_regime": "unknown"}
        ])
        result = format_brief(brief)
        assert "[BULL]" not in result  # No tag for unknown

    def test_negative_total_return_format(self):
        brief = self._minimal_brief(top_production=[
            {"ticker": "SPY", "strategy_name": "x", "sharpe": -0.5, "total_return": -3.2,
             "discovery_regime": ""}
        ])
        result = format_brief(brief)
        assert "-3.2%" in result
