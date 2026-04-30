"""Tests for Phase 6 Mandate Prioritization.

Covers:
- score_mandate
- prioritize_mandates
- convergence history, portfolio gaps, regime match
- budget truncation, top_n, tiebreaking
"""

from crabquant.refinement.mandate_generator import (
    score_mandate,
    prioritize_mandates,
)


def _mandate(archetype: str = "momentum", name: str = "test") -> dict:
    return {
        "name": name,
        "strategy_archetype": archetype,
        "primary_ticker": "SPY",
        "sharpe_target": 1.0,
    }


# ── score_mandate ───────────────────────────────────────────────────────


class TestScoreMandate:

    def test_high_convergence_scores_higher(self):
        """Mandates with higher convergence history get higher scores."""
        conv_high = {"momentum": 0.8}
        conv_low = {"momentum": 0.2}

        score_high = score_mandate(
            _mandate("momentum"), convergence_history=conv_high,
        )
        score_low = score_mandate(
            _mandate("momentum"), convergence_history=conv_low,
        )
        assert score_high > score_low

    def test_portfolio_gap_bonus(self):
        """Mandates for underrepresented archetypes get higher scores."""
        gaps = {"breakout": 0.1, "trend": 0.9}  # breakout = big gap, trend = covered

        score_gap = score_mandate(
            _mandate("breakout"), portfolio_gaps=gaps,
        )
        score_covered = score_mandate(
            _mandate("trend"), portfolio_gaps=gaps,
        )
        assert score_gap > score_covered

    def test_regime_match_bonus(self):
        """Mandates matching the current regime get higher scores."""
        score_match = score_mandate(
            _mandate("trend"), current_regime="trending",
        )
        score_mismatch = score_mandate(
            _mandate("mean_reversion"), current_regime="trending",
        )
        assert score_match > score_mismatch

    def test_score_in_range(self):
        """Score is always between 0 and 1."""
        score = score_mandate(_mandate("momentum"))
        assert 0.0 <= score <= 1.0

    def test_empty_history_defaults_neutral(self):
        """Missing convergence history and portfolio gaps use neutral defaults."""
        score = score_mandate(_mandate("unknown_archetype"))
        assert 0.0 <= score <= 1.0
        # With all defaults at 0.5, score should be ~0.50
        assert 0.3 <= score <= 0.7

    def test_unknown_regime_neutral(self):
        """Unknown regime uses neutral affinity scores."""
        score = score_mandate(_mandate("momentum"), current_regime="nonexistent")
        assert 0.0 <= score <= 1.0

    def test_custom_regime_affinity(self):
        """Custom regime affinity overrides defaults."""
        custom = {
            "bull": {"momentum": 1.0, "trend": 0.0},
        }
        score_momentum = score_mandate(
            _mandate("momentum"), current_regime="bull", regime_affinity=custom,
        )
        score_trend = score_mandate(
            _mandate("trend"), current_regime="bull", regime_affinity=custom,
        )
        assert score_momentum > score_trend


# ── prioritize_mandates ────────────────────────────────────────────────


class TestPrioritizeMandates:

    def test_sort_order(self):
        """Mandates are returned sorted by score descending."""
        mandates = [
            _mandate("mean_reversion", "a"),
            _mandate("trend", "b"),
            _mandate("breakout", "c"),
        ]
        prioritized = prioritize_mandates(
            mandates,
            convergence_history={"trend": 0.9, "breakout": 0.7, "mean_reversion": 0.3},
            current_regime="trending",
        )
        scores = [m["_priority_score"] for m in prioritized]
        assert scores == sorted(scores, reverse=True)

    def test_returns_same_count(self):
        """By default, all mandates are returned."""
        mandates = [_mandate("momentum", f"m{i}") for i in range(5)]
        result = prioritize_mandates(mandates)
        assert len(result) == 5

    def test_empty_mandates(self):
        """Empty input returns empty list."""
        assert prioritize_mandates([]) == []

    def test_top_n_limit(self):
        """top_n truncates the result."""
        mandates = [_mandate("momentum", f"m{i}") for i in range(10)]
        result = prioritize_mandates(mandates, top_n=3)
        assert len(result) == 3

    def test_budget_truncation(self):
        """Low api_budget_remaining returns fewer mandates."""
        mandates = [_mandate("momentum", f"m{i}") for i in range(10)]
        result = prioritize_mandates(mandates, api_budget_remaining=0.3)
        assert len(result) < 10

    def test_budget_minimum_one(self):
        """Even with very low budget, at least 1 mandate is returned."""
        mandates = [_mandate("momentum", f"m{i}") for i in range(10)]
        result = prioritize_mandates(mandates, api_budget_remaining=0.01)
        assert len(result) >= 1

    def test_top_n_overrides_budget(self):
        """When both top_n and budget are set, top_n takes precedence."""
        mandates = [_mandate("momentum", f"m{i}") for i in range(10)]
        result = prioritize_mandates(mandates, api_budget_remaining=1.0, top_n=2)
        assert len(result) == 2

    def test_tiebreaking_by_name(self):
        """Tied scores are broken by mandate name alphabetically."""
        # All same archetype → same score → sorted by name
        mandates = [
            _mandate("momentum", "charlie"),
            _mandate("momentum", "alpha"),
            _mandate("momentum", "bravo"),
        ]
        result = prioritize_mandates(mandates)
        names = [m["name"] for m in result]
        assert names == ["alpha", "bravo", "charlie"]

    def test_priority_score_attached(self):
        """Each mandate gets a _priority_score field."""
        mandates = [_mandate("momentum", "m1")]
        result = prioritize_mandates(mandates)
        assert "_priority_score" in result[0]
        assert isinstance(result[0]["_priority_score"], float)

    def test_high_volatility_regime_favors_breakout(self):
        """In high_volatility regime, breakout strategies score highest."""
        mandates = [
            _mandate("breakout", "bo"),
            _mandate("trend", "tr"),
            _mandate("mean_reversion", "mr"),
            _mandate("momentum", "mo"),
        ]
        result = prioritize_mandates(
            mandates,
            current_regime="high_volatility",
        )
        # Breakout should be first (highest affinity in high_volatility)
        assert result[0]["strategy_archetype"] == "breakout"
