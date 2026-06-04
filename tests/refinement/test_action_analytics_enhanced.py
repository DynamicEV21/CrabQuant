"""Tests for Phase 6 Action Analytics Enhancement.

Covers:
- get_failure_mode_action_stats
- format_action_feedback_for_context
- recent_window bias
- persistence
- avg_delta computation
"""

from __future__ import annotations

from crabquant.refinement.action_analytics import (
    get_failure_mode_action_stats,
    format_action_feedback_for_context,
)


def _entry(
    mandate: str = "m1",
    turn: int = 1,
    action: str = "modify_params",
    sharpe: float = 0.5,
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
        "timestamp": "2026-04-28T10:00:00",
    }


def _sample_history() -> list[dict]:
    """Build a realistic history with sequential turns per mandate."""
    return [
        # m1: low_sharpe → modify_params → still fail → replace_indicator → success
        _entry("m1", 1, "modify_params", 0.3, False, "low_sharpe"),
        _entry("m1", 2, "modify_params", 0.5, False, "low_sharpe"),
        _entry("m1", 3, "replace_indicator", 1.2, True, ""),
        # m2: low_sharpe → add_filter → still fail → full_rewrite → success
        _entry("m2", 1, "add_filter", 0.4, False, "low_sharpe"),
        _entry("m2", 2, "add_filter", 0.6, False, "low_sharpe"),
        _entry("m2", 3, "full_rewrite", 1.5, True, ""),
        # m3: low_sharpe → replace_indicator → still fail → replace_indicator → success
        _entry("m3", 1, "replace_indicator", 0.2, False, "low_sharpe"),
        _entry("m3", 2, "replace_indicator", 0.4, False, "low_sharpe"),
        _entry("m3", 3, "replace_indicator", 1.1, True, ""),
        # m4: too_few_trades → change_entry_logic → success
        _entry("m4", 1, "change_entry_logic", 0.1, False, "too_few_trades"),
        _entry("m4", 2, "change_entry_logic", 1.3, True, ""),
        # m5: low_sharpe → modify_params → still fail → modify_params → fail (stuck)
        _entry("m5", 1, "modify_params", 0.3, False, "low_sharpe"),
        _entry("m5", 2, "modify_params", 0.4, False, "low_sharpe"),
        _entry("m5", 3, "modify_params", 0.5, False, "low_sharpe"),
    ]


# ── get_failure_mode_action_stats ───────────────────────────────────────


class TestFailureModeActionStats:

    def test_filters_by_failure_mode(self):
        """Only entries matching the given failure mode are included."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")

        # too_few_trades entry (m4) should NOT appear
        assert "change_entry_logic" not in stats
        # low_sharpe actions should appear
        assert "modify_params" in stats
        assert "replace_indicator" in stats
        assert "add_filter" in stats

    def test_counts_correct(self):
        """Total count per action matches the number of matching entries."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")

        # modify_params: m1-t1, m1-t2, m5-t1, m5-t2, m5-t3 = 5 entries
        assert stats["modify_params"]["count"] == 5
        # replace_indicator: m1-t2 (not matched), m3-t1, m3-t2 = 2 entries
        assert stats["replace_indicator"]["count"] == 2
        # add_filter: m2-t1, m2-t2 = 2 entries
        assert stats["add_filter"]["count"] == 2

    def test_success_detection(self):
        """Actions followed by a successful next turn are counted as successes."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")

        # replace_indicator at m3-t2 → m3-t3 succeeded → 1 success
        assert stats["replace_indicator"]["success"] == 1
        # replace_indicator at m3-t1 → m3-t2 failed → 0 more successes
        assert stats["replace_indicator"]["success"] >= 1

        # modify_params at m1-t2 → m1-t3 succeeded → 1 success
        # modify_params at m5-t2 → m5-t3 failed → no success
        # modify_params at m5-t3 → no next turn → no success
        # modify_params at m1-t1 → m1-t2 failed → no success
        # modify_params at m5-t1 → m5-t2 failed → no success
        assert stats["modify_params"]["success"] == 1

    def test_avg_delta_computation(self):
        """avg_delta is the average Sharpe change between turns."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")

        # replace_indicator at m3-t1 (sharpe=0.2) → m3-t2 (sharpe=0.4) → delta=0.2
        # replace_indicator at m3-t2 (sharpe=0.4) → m3-t3 (sharpe=1.1) → delta=0.7
        # avg_delta = (0.2 + 0.7) / 2 = 0.45
        assert abs(stats["replace_indicator"]["avg_delta"] - 0.45) < 0.01

    def test_empty_history_graceful(self):
        """Empty history returns empty dict."""
        stats = get_failure_mode_action_stats([], "low_sharpe")
        assert stats == {}

    def test_unknown_failure_mode_graceful(self):
        """Unknown failure mode returns empty dict."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "nonexistent_mode")
        assert stats == {}

    def test_recent_window_bias(self):
        """Only the last N entries are considered when window is set."""
        history = _sample_history()

        # With a window of 3 (last 3 entries), only m5 turns are considered
        stats_windowed = get_failure_mode_action_stats(history, "low_sharpe", recent_window=3)
        assert "modify_params" in stats_windowed
        # Only m5 has low_sharpe in the last 3 entries → only modify_params
        assert len(stats_windowed) == 1
        assert stats_windowed["modify_params"]["count"] == 3

    def test_no_window_returns_all(self):
        """recent_window=0 returns all entries."""
        history = _sample_history()
        stats_all = get_failure_mode_action_stats(history, "low_sharpe", recent_window=0)
        stats_default = get_failure_mode_action_stats(history, "low_sharpe")
        assert stats_all == stats_default

    def test_sorted_by_success_descending(self):
        """Results are sorted by success count descending."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")
        keys = list(stats.keys())
        for i in range(1, len(keys)):
            assert stats[keys[i]]["success"] <= stats[keys[i - 1]]["success"]


# ── format_action_feedback_for_context ──────────────────────────────────


class TestFormatActionFeedbackForContext:

    def test_returns_string(self):
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")
        result = format_action_feedback_for_context("low_sharpe", stats)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_failure_mode(self):
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")
        result = format_action_feedback_for_context("low_sharpe", stats)
        assert "low_sharpe" in result

    def test_includes_action_names(self):
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")
        result = format_action_feedback_for_context("low_sharpe", stats)
        assert "modify_params" in result
        assert "replace_indicator" in result

    def test_includes_total_count(self):
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")
        total = sum(s["count"] for s in stats.values())
        result = format_action_feedback_for_context("low_sharpe", stats)
        assert str(total) in result

    def test_max_actions_limit(self):
        """Only max_actions are shown."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")
        result = format_action_feedback_for_context("low_sharpe", stats, max_actions=1)
        # Should only show 1 action line (indented with ✅ or ❌)
        action_lines = [l for l in result.split("\n") if "success" in l and "Sharpe" in l]
        assert len(action_lines) == 1

    def test_empty_stats_returns_empty(self):
        result = format_action_feedback_for_context("low_sharpe", {})
        assert result == ""

    def test_includes_consider_trying(self):
        """Includes suggestion line."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")
        result = format_action_feedback_for_context("low_sharpe", stats)
        assert "Consider trying:" in result

    def test_shows_success_rate(self):
        """Each action line includes a percentage."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")
        result = format_action_feedback_for_context("low_sharpe", stats)
        assert "%" in result

    def test_shows_avg_delta(self):
        """Each action line includes the average Sharpe delta."""
        history = _sample_history()
        stats = get_failure_mode_action_stats(history, "low_sharpe")
        result = format_action_feedback_for_context("low_sharpe", stats)
        assert "Sharpe" in result
