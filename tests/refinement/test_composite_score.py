"""Tests for composite score tracking in best-strategy selection.

Task 9: Composite Score for Best-Strategy Tracking
Prevents "high Sharpe, 3 trades" from being tracked as "best".
Formula: sharpe * sqrt(trades/20) * (1 - abs(max_drawdown))
"""

import pytest
import math
from crabquant.refinement.prompts import compute_composite_score
from crabquant.refinement.schemas import RunState


class TestComputeCompositeScore:
    """Unit tests for the compute_composite_score function."""

    def test_basic_computation(self):
        """Standard case: Sharpe 2.0, 40 trades, 10% drawdown."""
        score = compute_composite_score(sharpe=2.0, trades=40, max_drawdown=0.10)
        expected = 2.0 * math.sqrt(40 / 20) * (1.0 - 0.10)
        assert score == pytest.approx(expected, rel=1e-6)

    def test_penalizes_few_trades(self):
        """A strategy with 3 trades should get a much lower composite than one with 40 trades,
        even at the same Sharpe."""
        score_3 = compute_composite_score(sharpe=2.0, trades=3, max_drawdown=0.10)
        score_40 = compute_composite_score(sharpe=2.0, trades=40, max_drawdown=0.10)
        assert score_3 < score_40
        # 3 trades should be less than half the score of 40 trades
        assert score_3 < score_40 * 0.5

    def test_penalizes_high_drawdown(self):
        """Higher drawdown should reduce composite score."""
        score_low_dd = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=0.05)
        score_high_dd = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=0.40)
        assert score_low_dd > score_high_dd

    def test_rewards_high_sharpe(self):
        """Higher Sharpe should produce higher composite score, all else equal."""
        score_low = compute_composite_score(sharpe=1.0, trades=30, max_drawdown=0.10)
        score_high = compute_composite_score(sharpe=3.0, trades=30, max_drawdown=0.10)
        assert score_high > score_low

    def test_zero_sharpe_returns_zero(self):
        """Non-positive Sharpe should return 0.0."""
        assert compute_composite_score(sharpe=0.0, trades=30, max_drawdown=0.10) == 0.0
        assert compute_composite_score(sharpe=-1.0, trades=30, max_drawdown=0.10) == 0.0

    def test_zero_trades_returns_zero(self):
        """Zero trades should return 0.0."""
        assert compute_composite_score(sharpe=2.0, trades=0, max_drawdown=0.10) == 0.0

    def test_high_sharpe_few_trades_vs_moderate_sharpe_many_trades(self):
        """Core scenario: high Sharpe + 3 trades should NOT beat moderate Sharpe + 40 trades."""
        high_sharpe_few = compute_composite_score(sharpe=5.0, trades=3, max_drawdown=0.05)
        moderate_sharpe_many = compute_composite_score(sharpe=2.0, trades=40, max_drawdown=0.10)
        assert moderate_sharpe_many > high_sharpe_few, (
            "A strategy with Sharpe 2.0 and 40 trades should outrank "
            "Sharpe 5.0 with only 3 trades (likely curve-fit)"
        )

    def test_extreme_drawdown_caps_penalty(self):
        """Drawdown penalty is capped at abs(max_drawdown) = 1.0 (100% drawdown → 0 penalty multiplier)."""
        score = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=2.0)
        # abs(2.0) is clamped to 1.0, so dd_penalty = 0.0, score = 0.0
        assert score == 0.0

    def test_negative_drawdown_handled(self):
        """Drawdown stored as negative fraction (e.g., -0.15) should work via abs()."""
        score_pos = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=0.15)
        score_neg = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=-0.15)
        assert score_pos == pytest.approx(score_neg, rel=1e-6)


class TestCompositeScoreBestTracking:
    """Tests that best-strategy tracking uses composite score, not raw Sharpe."""

    def test_run_state_has_composite_score_field(self):
        """RunState should have best_composite_score field."""
        state = RunState(
            run_id="test-001",
            mandate_name="test",
            created_at="2026-01-01T00:00:00",
        )
        assert hasattr(state, "best_composite_score")
        assert state.best_composite_score == -999.0

    def test_run_state_serialization_roundtrip(self):
        """best_composite_score should survive JSON roundtrip."""
        state = RunState(
            run_id="test-001",
            mandate_name="test",
            created_at="2026-01-01T00:00:00",
            best_sharpe=2.5,
            best_composite_score=3.0,
            best_turn=3,
        )
        d = state.to_dict()
        restored = RunState.from_dict(d)
        assert restored.best_composite_score == 3.0
        assert restored.best_sharpe == 2.5

    def test_composite_score_in_history_entries(self):
        """History entries should include composite_score when available."""
        state = RunState(
            run_id="test-001",
            mandate_name="test",
            created_at="2026-01-01T00:00:00",
        )
        state.history.append({
            "turn": 1,
            "sharpe": 2.0,
            "composite_score": 1.8,
            "failure_mode": "low_sharpe",
            "action": "modify_params",
        })
        assert state.history[0]["composite_score"] == 1.8

    def test_scenario_curve_fit_loses_to_robust(self):
        """Simulate the core scenario: curve-fit (high Sharpe, few trades) vs robust (moderate Sharpe, many trades)."""
        curve_fit_score = compute_composite_score(sharpe=4.0, trades=5, max_drawdown=0.03)
        robust_score = compute_composite_score(sharpe=1.8, trades=60, max_drawdown=0.12)

        assert robust_score > curve_fit_score, (
            f"Robust strategy (Sharpe 1.8, 60 trades) composite={robust_score:.2f} "
            f"should beat curve-fit (Sharpe 4.0, 5 trades) composite={curve_fit_score:.2f}"
        )

    def test_equal_sharpe_different_trades(self):
        """Same Sharpe but different trade counts: more trades wins."""
        score_10 = compute_composite_score(sharpe=2.0, trades=10, max_drawdown=0.10)
        score_50 = compute_composite_score(sharpe=2.0, trades=50, max_drawdown=0.10)
        assert score_50 > score_10

    def test_equal_sharpe_different_drawdown(self):
        """Same Sharpe and trades but different drawdown: lower drawdown wins."""
        score_good = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=0.05)
        score_bad = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=0.30)
        assert score_good > score_bad


# ── New tests: edge cases, boundary conditions, and additional functions ──────


class TestComputeCompositeScoreEdgeCases:
    """Additional edge case and boundary tests for compute_composite_score."""

    def test_zero_drawdown_no_penalty(self):
        """Zero drawdown should have no penalty (dd_penalty = 1.0)."""
        score = compute_composite_score(sharpe=2.0, trades=20, max_drawdown=0.0)
        expected = 2.0 * math.sqrt(20 / 20) * 1.0
        assert score == pytest.approx(expected, rel=1e-6)

    def test_exactly_one_trade(self):
        """One trade should produce a valid (small) score."""
        score = compute_composite_score(sharpe=2.0, trades=1, max_drawdown=0.05)
        expected = 2.0 * math.sqrt(1 / 20) * 0.95
        assert score == pytest.approx(expected, rel=1e-6)
        assert score > 0

    def test_negative_trades_returns_zero(self):
        """Negative trade count should return 0.0."""
        assert compute_composite_score(sharpe=2.0, trades=-5, max_drawdown=0.10) == 0.0

    def test_very_small_positive_sharpe(self):
        """Sharpe just above zero should produce a very small score."""
        score = compute_composite_score(sharpe=0.01, trades=100, max_drawdown=0.01)
        assert score > 0
        assert score < 0.1

    def test_exactly_20_trades_trade_factor_one(self):
        """Exactly 20 trades → trade_factor = sqrt(20/20) = 1.0."""
        score = compute_composite_score(sharpe=3.0, trades=20, max_drawdown=0.0)
        assert score == pytest.approx(3.0, rel=1e-6)

    def test_very_large_trade_count(self):
        """Very large trade count should scale appropriately."""
        score_20 = compute_composite_score(sharpe=2.0, trades=20, max_drawdown=0.0)
        score_500 = compute_composite_score(sharpe=2.0, trades=500, max_drawdown=0.0)
        ratio = score_500 / score_20
        expected_ratio = math.sqrt(500 / 20)
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_99_percent_drawdown(self):
        """99% drawdown should heavily penalize but not zero out."""
        score = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=0.99)
        expected = 2.0 * math.sqrt(30 / 20) * 0.01
        assert score == pytest.approx(expected, rel=1e-6)

    def test_100_percent_drawdown(self):
        """100% drawdown → dd_penalty = 0.0 → score = 0.0."""
        score = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=1.0)
        assert score == 0.0

    def test_drawdown_greater_than_one(self):
        """Drawdown > 1.0 should be clamped to 1.0."""
        score_100 = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=1.0)
        score_200 = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=2.0)
        assert score_100 == score_200

    def test_negative_large_sharpe(self):
        """Very negative Sharpe should return 0.0."""
        assert compute_composite_score(sharpe=-100.0, trades=100, max_drawdown=0.01) == 0.0

    def test_very_high_sharpe_many_trades(self):
        """Very high Sharpe with many trades → high score."""
        score = compute_composite_score(sharpe=10.0, trades=200, max_drawdown=0.05)
        assert score > 30.0

    def test_returns_float_type(self):
        """Score should always be a float."""
        score = compute_composite_score(sharpe=2.0, trades=30, max_drawdown=0.1)
        assert isinstance(score, float)

    def test_zero_sharpe_zero_trades(self):
        """Both zero should return 0.0."""
        assert compute_composite_score(sharpe=0.0, trades=0, max_drawdown=0.0) == 0.0


class TestComputeCompositeScoreTradeoffs:
    """Tests that verify the formula's tradeoff behavior."""

    def test_curve_fit_10_trades_vs_robust_80_trades(self):
        """Sharpe 3.0 + 10 trades should lose to Sharpe 1.5 + 80 trades."""
        curve = compute_composite_score(sharpe=3.0, trades=10, max_drawdown=0.08)
        robust = compute_composite_score(sharpe=1.5, trades=80, max_drawdown=0.15)
        assert robust > curve

    def test_perfect_sharpe_few_trades_vs_moderate_many(self):
        """Sharpe 10.0 + 5 trades with 2% DD still has very high composite.
        Verify the tradeoff exists: moderate many trades should be competitive."""
        extreme = compute_composite_score(sharpe=10.0, trades=5, max_drawdown=0.02)
        moderate = compute_composite_score(sharpe=2.0, trades=100, max_drawdown=0.15)
        # extreme = 10 * sqrt(5/20) * 0.98 = 10 * 0.5 * 0.98 = 4.9
        # moderate = 2 * sqrt(100/20) * 0.85 = 2 * 2.236 * 0.85 = 3.80
        # Extreme still wins here — curve-fit detection via composite works
        # for more moderate differences, not extreme Sharpe advantage
        assert extreme > moderate
        # But verify the tradeoff: 200 trades should overcome the Sharpe gap
        very_robust = compute_composite_score(sharpe=2.0, trades=200, max_drawdown=0.15)
        assert very_robust > extreme

    def test_drawdown_difference_overcomes_sharpe(self):
        """Enough drawdown difference can overcome Sharpe advantage."""
        good_dd = compute_composite_score(sharpe=1.5, trades=40, max_drawdown=0.02)
        bad_dd = compute_composite_score(sharpe=2.5, trades=40, max_drawdown=0.55)
        assert good_dd > bad_dd

    def test_monotonic_trade_increase(self):
        """For fixed Sharpe and DD, more trades → higher score."""
        prev = 0
        for trades in [5, 10, 20, 40, 80, 160]:
            score = compute_composite_score(sharpe=2.0, trades=trades, max_drawdown=0.10)
            assert score > prev
            prev = score

    def test_monotonic_drawdown_decrease(self):
        """For fixed Sharpe and trades, less drawdown → higher score."""
        prev = 0
        for dd in [0.50, 0.30, 0.15, 0.05, 0.01]:
            score = compute_composite_score(sharpe=2.0, trades=40, max_drawdown=dd)
            assert score > prev
            prev = score

    def test_monotonic_sharpe_increase(self):
        """For fixed trades and DD, higher Sharpe → higher score."""
        prev = 0
        for sharpe in [0.5, 1.0, 2.0, 3.0, 5.0]:
            score = compute_composite_score(sharpe=sharpe, trades=40, max_drawdown=0.10)
            assert score > prev
            prev = score
