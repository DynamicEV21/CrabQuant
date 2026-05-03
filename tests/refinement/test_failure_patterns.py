"""Tests for crabquant.refinement.failure_patterns — Phase 6."""

import pytest

from crabquant.refinement.failure_patterns import (
    _RECOMMENDATIONS,
    analyze_failure_patterns,
    format_failure_patterns_for_prompt,
    get_auto_adjustments,
)
from crabquant.refinement.action_effectiveness import SKIP_MANDATES


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_entry(
    mandate: str = "momentum_spy",
    turn: int = 1,
    action: str = "novel",
    sharpe: float = 0.3,
    success: bool = False,
    failure_mode: str = "low_sharpe",
) -> dict:
    return {
        "mandate": mandate,
        "turn": turn,
        "action": action,
        "sharpe": sharpe,
        "success": success,
        "failure_mode": failure_mode,
        "timestamp": "2026-04-29T12:00:00",
    }


def _entries_with_mode(mode: str, count: int, **kwargs) -> list[dict]:
    """Generate *count* failure entries all with the given mode."""
    return [
        _make_entry(
            mandate=f"mandate_{i}",
            turn=1,
            failure_mode=mode,
            **kwargs,
        )
        for i in range(count)
    ]


# ── Distribution computation ─────────────────────────────────────────────────


class TestDistributionComputation:

    def test_distribution_computation(self):
        """Distribution percentages sum to 1.0 and modes are present."""
        history = (
            _entries_with_mode("low_sharpe", 30)
            + _entries_with_mode("too_few_trades", 20)
            + _entries_with_mode("excessive_drawdown", 10)
        )
        result = analyze_failure_patterns(history)

        dist = result["distribution"]
        assert "low_sharpe" in dist
        assert "too_few_trades" in dist
        assert "excessive_drawdown" in dist

        # Percentages should sum to ~1.0
        total_pct = sum(dist.values())
        assert total_pct == pytest.approx(1.0, abs=0.01)

        assert result["total_failures"] == 60

    def test_distribution_values_correct(self):
        """Check exact percentage values."""
        history = _entries_with_mode("low_sharpe", 60) + _entries_with_mode("too_few_trades", 40)
        result = analyze_failure_patterns(history)

        assert result["distribution"]["low_sharpe"] == pytest.approx(0.6)
        assert result["distribution"]["too_few_trades"] == pytest.approx(0.4)


# ── Dominant mode detection ──────────────────────────────────────────────────


class TestDominantModeDetection:

    def test_dominant_mode_detection(self):
        """A mode >40% is detected as dominant."""
        history = _entries_with_mode("low_sharpe", 50) + _entries_with_mode("too_few_trades", 10)
        result = analyze_failure_patterns(history)

        assert result["dominant_mode"] == "low_sharpe"
        assert result["dominant_pct"] > 0.40

    def test_no_dominant_when_balanced(self):
        """No dominant mode when failures are evenly spread."""
        history = (
            _entries_with_mode("low_sharpe", 10)
            + _entries_with_mode("too_few_trades", 10)
            + _entries_with_mode("excessive_drawdown", 10)
        )
        result = analyze_failure_patterns(history)

        assert result["dominant_mode"] is None

    def test_exactly_40_percent_not_dominant(self):
        """Exactly 40% should NOT be dominant (threshold is >40%)."""
        # Three modes each at 33.3% — none exceeds 40%
        history = (
            _entries_with_mode("low_sharpe", 40)
            + _entries_with_mode("too_few_trades", 40)
            + _entries_with_mode("excessive_drawdown", 40)
        )
        result = analyze_failure_patterns(history)

        assert result["dominant_mode"] is None


# ── Recommendations ──────────────────────────────────────────────────────────


class TestRecommendations:

    def test_recommendations_too_few_trades(self):
        """too_few_trades dominant mode produces relevant recommendations."""
        history = _entries_with_mode("too_few_trades", 50) + _entries_with_mode("low_sharpe", 5)
        result = analyze_failure_patterns(history)

        assert result["dominant_mode"] == "too_few_trades"
        assert len(result["recommendations"]) > 0
        # Check that at least one recommendation mentions trades or entries
        rec_text = " ".join(result["recommendations"]).lower()
        assert any(word in rec_text for word in ["trade", "entry", "signal"])

    def test_recommendations_low_sharpe(self):
        """low_sharpe dominant mode produces quality-focused recommendations."""
        history = _entries_with_mode("low_sharpe", 50) + _entries_with_mode("too_few_trades", 5)
        result = analyze_failure_patterns(history)

        assert result["dominant_mode"] == "low_sharpe"
        assert len(result["recommendations"]) > 0
        rec_text = " ".join(result["recommendations"]).lower()
        assert any(word in rec_text for word in ["quality", "simplif", "sharpe"])

    def test_recommendations_regime_fragility(self):
        """regime_fragility dominant mode produces regime-aware recommendations."""
        history = _entries_with_mode("regime_fragility", 50) + _entries_with_mode("low_sharpe", 5)
        result = analyze_failure_patterns(history)

        assert result["dominant_mode"] == "regime_fragility"
        assert len(result["recommendations"]) > 0
        rec_text = " ".join(result["recommendations"]).lower()
        assert any(word in rec_text for word in ["regime", "adaptive", "market state"])


# ── Auto adjustments ─────────────────────────────────────────────────────────


class TestAutoAdjustments:

    def test_auto_adjustment_trades(self):
        """too_few_trades produces entry-frequency hints and threshold adjustments."""
        analysis = analyze_failure_patterns(
            _entries_with_mode("too_few_trades", 50) + _entries_with_mode("low_sharpe", 5)
        )
        adj = get_auto_adjustments(analysis)

        assert len(adj["prompt_hints"]) > 0
        assert adj["threshold_adjustments"]["min_trades_hint"] == 8
        assert "change_entry_logic" in adj["priority_actions"]

    def test_auto_adjustment_sharpe(self):
        """low_sharpe produces quality-over-quantity hints."""
        analysis = analyze_failure_patterns(
            _entries_with_mode("low_sharpe", 50) + _entries_with_mode("too_few_trades", 5)
        )
        adj = get_auto_adjustments(analysis)

        assert len(adj["prompt_hints"]) > 0
        assert adj["threshold_adjustments"]["quality_over_quantity"] is True
        hint_text = " ".join(adj["prompt_hints"]).lower()
        assert any(word in hint_text for word in ["quality", "simplif"])

    def test_auto_adjustment_regime_fragility(self):
        """regime_fragility produces regime-aware adjustments."""
        analysis = analyze_failure_patterns(
            _entries_with_mode("regime_fragility", 50) + _entries_with_mode("low_sharpe", 5)
        )
        adj = get_auto_adjustments(analysis)

        assert adj["threshold_adjustments"]["regime_aware"] is True
        assert "add_regime_filter" in adj["priority_actions"]

    def test_auto_adjustment_excessive_drawdown(self):
        """excessive_drawdown produces risk management adjustments."""
        analysis = analyze_failure_patterns(
            _entries_with_mode("excessive_drawdown", 50) + _entries_with_mode("low_sharpe", 5)
        )
        adj = get_auto_adjustments(analysis)

        assert "max_drawdown_pct" in adj["threshold_adjustments"]
        assert "stop_loss_pct" in adj["threshold_adjustments"]
        assert "add_stop_loss" in adj["priority_actions"]

    def test_no_adjustment_balanced_failures(self):
        """Balanced failures (no dominant) produce empty adjustments."""
        history = (
            _entries_with_mode("low_sharpe", 10)
            + _entries_with_mode("too_few_trades", 10)
            + _entries_with_mode("excessive_drawdown", 10)
        )
        analysis = analyze_failure_patterns(history)
        adj = get_auto_adjustments(analysis)

        assert adj["prompt_hints"] == []
        assert adj["threshold_adjustments"] == {}
        assert adj["priority_actions"] == []


# ── Recent trend ─────────────────────────────────────────────────────────────


class TestRecentTrend:

    def test_recent_trend_increasing(self):
        """When dominant mode frequency rises in recent window, trend is 'increasing'."""
        # Build history: first 50 have low_sharpe at ~20%, last 50 at ~80%
        history = (
            _entries_with_mode("low_sharpe", 10)
            + _entries_with_mode("too_few_trades", 40)   # older half: 10/50 = 20%
            + _entries_with_mode("low_sharpe", 40)
            + _entries_with_mode("too_few_trades", 10)   # newer half: 40/50 = 80%
        )
        result = analyze_failure_patterns(history)

        assert result["recent_trend"]["direction"] == "increasing"
        assert "rose" in result["recent_trend"]["detail"].lower()

    def test_recent_trend_decreasing(self):
        """When dominant mode frequency drops, trend is 'decreasing'."""
        # Build history: first 50 have low_sharpe at ~80%, last 50 at ~20%
        history = (
            _entries_with_mode("low_sharpe", 40)
            + _entries_with_mode("too_few_trades", 10)   # older half: 40/50 = 80%
            + _entries_with_mode("low_sharpe", 10)
            + _entries_with_mode("too_few_trades", 40)   # newer half: 10/50 = 20%
        )
        result = analyze_failure_patterns(history)

        assert result["recent_trend"]["direction"] == "decreasing"
        assert "fell" in result["recent_trend"]["detail"].lower()

    def test_recent_trend_stable(self):
        """When dominant mode frequency is roughly the same, trend is 'stable'."""
        history = (
            _entries_with_mode("low_sharpe", 25)
            + _entries_with_mode("too_few_trades", 25)   # older half: 25/50 = 50%
            + _entries_with_mode("low_sharpe", 25)
            + _entries_with_mode("too_few_trades", 25)   # newer half: 25/50 = 50%
        )
        result = analyze_failure_patterns(history)

        assert result["recent_trend"]["direction"] == "stable"


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_history(self):
        """Empty history returns safe defaults."""
        result = analyze_failure_patterns([])

        assert result["total_failures"] == 0
        assert result["distribution"] == {}
        assert result["dominant_mode"] is None
        assert result["dominant_pct"] == 0.0
        assert result["recommendations"] == []
        assert result["recent_trend"]["direction"] == "stable"

    def test_smoke_entries_filtered(self):
        """smoke_test and other SKIP_MANDATES entries are excluded."""
        smoke_entries = [
            _make_entry(mandate="smoke_test", failure_mode="low_sharpe"),
            _make_entry(mandate="test_mandate", failure_mode="low_sharpe"),
            _make_entry(mandate="e2e_stress_test", failure_mode="too_few_trades"),
            _make_entry(mandate="single_turn", failure_mode="excessive_drawdown"),
            _make_entry(mandate="module_fail", failure_mode="low_sharpe"),
        ]
        result = analyze_failure_patterns(smoke_entries)

        assert result["total_failures"] == 0
        assert result["distribution"] == {}

    def test_successful_entries_excluded(self):
        """Entries with empty failure_mode (successes) are not counted."""
        history = [
            _make_entry(failure_mode=""),
            _make_entry(failure_mode=""),
            _make_entry(failure_mode="low_sharpe"),
        ]
        result = analyze_failure_patterns(history)

        assert result["total_failures"] == 1
        assert "low_sharpe" in result["distribution"]

    def test_window_limits_entries(self):
        """Only the last *window* entries are analysed."""
        # 150 entries: 60 too_few_trades + 30 low_sharpe + 30 excessive_drawdown + 30 regime_fragility
        # With window=100, last 100 = 10 too_few + 30 low_sharpe + 30 excessive + 30 regime
        # None exceeds 40%: 10%, 30%, 30%, 30%
        history = (
            _entries_with_mode("too_few_trades", 60)
            + _entries_with_mode("low_sharpe", 30)
            + _entries_with_mode("excessive_drawdown", 30)
            + _entries_with_mode("regime_fragility", 30)  # older entries
        )
        result = analyze_failure_patterns(history, window=100)

        assert result["total_failures"] == 100
        assert result["dominant_mode"] is None  # no mode > 40%


# ── Format for prompt ────────────────────────────────────────────────────────


class TestFormatPromptContent:

    def test_format_prompt_content(self):
        """Formatted output includes distribution, dominant mode, and recommendations."""
        history = _entries_with_mode("low_sharpe", 50) + _entries_with_mode("too_few_trades", 10)
        analysis = analyze_failure_patterns(history)
        formatted = format_failure_patterns_for_prompt(analysis)

        assert "Failure Pattern Analysis" in formatted
        assert "low_sharpe" in formatted
        assert "too_few_trades" in formatted
        assert "Recommendations" in formatted
        assert "Dominant failure mode" in formatted

    def test_format_empty_returns_empty(self):
        """Empty analysis returns empty string."""
        formatted = format_failure_patterns_for_prompt(
            analyze_failure_patterns([])
        )
        assert formatted == ""

    def test_format_includes_bar_chart(self):
        """Distribution includes a visual bar."""
        history = _entries_with_mode("low_sharpe", 60) + _entries_with_mode("too_few_trades", 40)
        analysis = analyze_failure_patterns(history)
        formatted = format_failure_patterns_for_prompt(analysis)

        assert "█" in formatted

    def test_format_includes_trend_when_increasing(self):
        """Trend info is included when direction is not stable."""
        history = (
            _entries_with_mode("low_sharpe", 10)
            + _entries_with_mode("too_few_trades", 40)
            + _entries_with_mode("low_sharpe", 40)
            + _entries_with_mode("too_few_trades", 10)
        )
        analysis = analyze_failure_patterns(history)
        formatted = format_failure_patterns_for_prompt(analysis)

        assert "Trend:" in formatted


# ── All failure modes have recommendations ───────────────────────────────────


class TestAllFailureModesHaveRecommendations:

    def test_all_failure_modes_have_recommendations(self):
        """Every mode in _RECOMMENDATIONS has at least one recommendation."""
        for mode, recs in _RECOMMENDATIONS.items():
            assert isinstance(recs, list)
            assert len(recs) >= 1, f"{mode} has no recommendations"
            for rec in recs:
                assert isinstance(rec, str)
                assert len(rec) > 10, f"{mode} recommendation too short: {rec!r}"

    def test_known_modes_covered(self):
        """The major failure modes are all covered in _RECOMMENDATIONS."""
        expected_modes = [
            "too_few_trades",
            "low_sharpe",
            "regime_fragility",
            "excessive_drawdown",
            "backtest_crash",
        ]
        for mode in expected_modes:
            assert mode in _RECOMMENDATIONS, f"{mode} missing from recommendations"

    def test_unknown_mode_gets_generic(self):
        """An unknown dominant mode still gets recommendations (generic)."""
        history = _entries_with_mode("totally_unknown_mode", 50) + _entries_with_mode("low_sharpe", 5)
        result = analyze_failure_patterns(history)

        assert result["dominant_mode"] == "totally_unknown_mode"
        assert len(result["recommendations"]) > 0
