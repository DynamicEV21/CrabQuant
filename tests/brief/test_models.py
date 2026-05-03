"""
Tests for crabquant.brief.models module.
"""

import copy
import pickle

import pytest

from crabquant.brief.models import BriefData


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def default_brief():
    return BriefData()


@pytest.fixture
def populated_brief():
    return BriefData(
        regime="trending_up",
        regime_confidence=0.92,
        spy_20d_return=3.5,
        realized_vol=0.18,
        top_production=[{"ticker": "SPY", "strategy_name": "sma_cross", "sharpe": 1.5, "total_return": 12.5}],
        recent_winners_count=7,
        recent_promotions_count=2,
        recent_retirements_count=1,
        total_combos_tested=500,
        cron_active=3,
        cron_total=4,
        regime_strategy_suggestions=[("sma_cross", 0.9), ("momentum", 0.7)],
        promotion_metrics={"total_winners": 10, "confirmed_count": 3, "promotion_rate": 0.3},
    )


# ── Default Creation ──────────────────────────────────────────────────────


class TestBriefDataDefaultCreation:

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

    def test_all_defaults_are_zero_or_none_or_empty(self, default_brief):
        assert default_brief.regime_confidence == 0.0
        assert default_brief.recent_winners_count == 0
        assert default_brief.recent_promotions_count == 0
        assert default_brief.recent_retirements_count == 0
        assert default_brief.total_combos_tested == 0
        assert default_brief.cron_active == 0
        assert default_brief.cron_total == 0
        assert default_brief.spy_20d_return is None
        assert default_brief.realized_vol is None
        assert default_brief.top_production == []
        assert default_brief.regime_strategy_suggestions == []
        assert default_brief.promotion_metrics == {}


# ── Custom Values ──────────────────────────────────────────────────────────


class TestBriefDataCustomValues:

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

    def test_regime_can_be_any_string(self):
        brief = BriefData(regime="custom_regime")
        assert brief.regime == "custom_regime"

    def test_regime_confidence_accepts_float(self):
        brief = BriefData(regime_confidence=0.123456)
        assert brief.regime_confidence == pytest.approx(0.123456)

    def test_regime_confidence_accepts_integer(self):
        brief = BriefData(regime_confidence=1)
        assert brief.regime_confidence == 1.0

    def test_spy_20d_return_accepts_positive(self):
        brief = BriefData(spy_20d_return=10.5)
        assert brief.spy_20d_return == 10.5

    def test_spy_20d_return_accepts_negative(self):
        brief = BriefData(spy_20d_return=-5.2)
        assert brief.spy_20d_return == -5.2

    def test_spy_20d_return_accepts_zero(self):
        brief = BriefData(spy_20d_return=0.0)
        assert brief.spy_20d_return == 0.0

    def test_realized_vol_accepts_various_values(self):
        for vol in [0.0, 0.01, 0.5, 1.0, 2.5]:
            brief = BriefData(realized_vol=vol)
            assert brief.realized_vol == vol

    def test_count_fields_accept_large_integers(self):
        brief = BriefData(
            recent_winners_count=999999,
            recent_promotions_count=999999,
            recent_retirements_count=999999,
            total_combos_tested=999999,
        )
        assert brief.recent_winners_count == 999999
        assert brief.recent_promotions_count == 999999
        assert brief.recent_retirements_count == 999999
        assert brief.total_combos_tested == 999999

    def test_top_production_accepts_multiple_entries(self):
        entries = [{"ticker": f"T{i}", "strategy_name": f"s{i}"} for i in range(10)]
        brief = BriefData(top_production=entries)
        assert len(brief.top_production) == 10
        assert brief.top_production[5]["ticker"] == "T5"

    def test_top_production_with_complex_dicts(self):
        complex_entry = {
            "ticker": "SPY",
            "strategy_name": "sma_cross",
            "sharpe": 2.1,
            "total_return": 15.5,
            "verdict": "ROBUST",
            "discovery_regime": "trending_up",
        }
        brief = BriefData(top_production=[complex_entry])
        assert brief.top_production[0]["sharpe"] == 2.1

    def test_regime_strategy_suggestions_with_multiple_entries(self):
        suggestions = [("sma_cross", 0.9), ("momentum", 0.7), ("mean_rev", 0.5)]
        brief = BriefData(regime_strategy_suggestions=suggestions)
        assert len(brief.regime_strategy_suggestions) == 3
        assert brief.regime_strategy_suggestions[1] == ("momentum", 0.7)

    def test_promotion_metrics_with_full_funnel(self):
        metrics = {
            "total_winners": 100,
            "backtest_only_count": 60,
            "walk_forward_passed_count": 30,
            "confirmed_count": 15,
            "promoted_count": 8,
            "promotion_rate": 0.08,
        }
        brief = BriefData(promotion_metrics=metrics)
        assert brief.promotion_metrics["total_winners"] == 100
        assert brief.promotion_metrics["promotion_rate"] == 0.08


# ── Mutable Defaults Independence ─────────────────────────────────────────


class TestMutableDefaults:

    def test_mutable_defaults_are_independent(self):
        """Each instance should get its own list/dict, not share defaults."""
        b1 = BriefData()
        b2 = BriefData()
        b1.top_production.append({"ticker": "SPY"})
        b1.promotion_metrics["key"] = "value"
        assert b2.top_production == []
        assert b2.promotion_metrics == {}

    def test_top_production_mutation_does_not_affect_new_instance(self):
        b1 = BriefData()
        b1.top_production.append({"ticker": "AAPL"})
        b2 = BriefData()
        assert b2.top_production == []

    def test_regime_strategy_suggestions_mutation_isolation(self):
        b1 = BriefData()
        b1.regime_strategy_suggestions.append(("strat", 0.5))
        b2 = BriefData()
        assert b2.regime_strategy_suggestions == []

    def test_promotion_metrics_mutation_does_not_affect_new_instance(self):
        b1 = BriefData()
        b1.promotion_metrics["rate"] = 0.42
        b2 = BriefData()
        assert b2.promotion_metrics == {}

    def test_multiple_instances_all_independent(self):
        instances = [BriefData() for _ in range(5)]
        for i, inst in enumerate(instances):
            inst.top_production.append({"idx": i})
            inst.promotion_metrics[f"key_{i}"] = i
        for i, inst in enumerate(instances):
            assert len(inst.top_production) == 1
            assert inst.top_production[0]["idx"] == i


# ── Dataclass Behavior ────────────────────────────────────────────────────


class TestDataclassBehavior:

    def test_dataclass_repr(self):
        brief = BriefData(regime="trending_up", cron_active=2, cron_total=3)
        r = repr(brief)
        assert "BriefData" in r
        assert "trending_up" in r

    def test_repr_contains_all_fields(self, populated_brief):
        r = repr(populated_brief)
        assert "trending_up" in r
        assert "3.5" in r

    def test_equality_same_values(self):
        b1 = BriefData(regime="test", regime_confidence=0.5)
        b2 = BriefData(regime="test", regime_confidence=0.5)
        assert b1 == b2

    def test_inequality_different_values(self):
        b1 = BriefData(regime="test", regime_confidence=0.5)
        b2 = BriefData(regime="other", regime_confidence=0.5)
        assert b1 != b2

    def test_not_equal_to_non_briefdata(self):
        brief = BriefData()
        assert brief != "BriefData"
        assert brief != {"regime": "UNKNOWN"}
        assert brief != None

    def test_str_returns_repr(self, default_brief):
        s = str(default_brief)
        assert "BriefData" in s

    def test_field_assignment(self):
        brief = BriefData()
        brief.regime = "mean_reversion"
        brief.regime_confidence = 0.8
        assert brief.regime == "mean_reversion"
        assert brief.regime_confidence == 0.8

    def test_copy_deepcopy(self):
        brief = BriefData(
            top_production=[{"ticker": "SPY"}],
            promotion_metrics={"key": "val"},
        )
        import copy
        deep = copy.deepcopy(brief)
        deep.top_production[0]["ticker"] = "QQQ"
        deep.promotion_metrics["key"] = "changed"
        assert brief.top_production[0]["ticker"] == "SPY"
        assert brief.promotion_metrics["key"] == "val"

    def test_pickle_roundtrip(self):
        brief = BriefData(
            regime="high_volatility",
            regime_confidence=0.75,
            spy_20d_return=-2.3,
            realized_vol=0.45,
            top_production=[{"ticker": "AAPL"}],
            recent_winners_count=3,
            recent_promotions_count=1,
            recent_retirements_count=0,
            total_combos_tested=200,
            cron_active=2,
            cron_total=4,
            regime_strategy_suggestions=[("vol_break", 0.8)],
            promotion_metrics={"total_winners": 5},
        )
        data = pickle.dumps(brief)
        restored = pickle.loads(data)
        assert restored == brief

    def test_has_dataclass_fields(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(BriefData)}
        assert "regime" in fields
        assert "regime_confidence" in fields
        assert "spy_20d_return" in fields
        assert "realized_vol" in fields
        assert "top_production" in fields
        assert "recent_winners_count" in fields
        assert "recent_promotions_count" in fields
        assert "recent_retirements_count" in fields
        assert "total_combos_tested" in fields
        assert "cron_active" in fields
        assert "cron_total" in fields
        assert "regime_strategy_suggestions" in fields
        assert "promotion_metrics" in fields

    def test_field_count_is_thirteen(self):
        import dataclasses
        assert len(dataclasses.fields(BriefData)) == 13

    def test_asdict_conversion(self):
        import dataclasses
        brief = BriefData(regime="test")
        d = dataclasses.asdict(brief)
        assert isinstance(d, dict)
        assert d["regime"] == "test"
        assert d["top_production"] == []
        assert d["promotion_metrics"] == {}

    def test_regime_strategy_suggestions_type(self):
        brief = BriefData(regime_strategy_suggestions=[("sma_cross", 0.9)])
        assert brief.regime_strategy_suggestions[0] == ("sma_cross", 0.9)

    def test_regime_strategy_suggestions_empty_list_is_default(self, default_brief):
        assert default_brief.regime_strategy_suggestions == []

    def test_partial_population_preserves_other_defaults(self):
        brief = BriefData(regime="trending_down", cron_active=1)
        assert brief.regime == "trending_down"
        assert brief.cron_active == 1
        assert brief.regime_confidence == 0.0
        assert brief.spy_20d_return is None
        assert brief.realized_vol is None
        assert brief.top_production == []
        assert brief.recent_winners_count == 0
        assert brief.cron_total == 0
