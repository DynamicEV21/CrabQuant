"""Tests for crabquant.refinement.action_analytics — Phase 3."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crabquant.refinement.action_analytics import (
    ActionStats,
    track_action_result,
    aggregate_action_stats,
    compute_action_success_rates,
    generate_llm_context,
    load_run_history,
    RUN_HISTORY_FILE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_history_entry(
    action: str = "modify_params",
    sharpe: float = 1.0,
    success: bool = True,
    mandate: str = "test_mandate",
    turn: int = 1,
) -> dict:
    return {
        "mandate": mandate,
        "turn": turn,
        "action": action,
        "sharpe": sharpe,
        "success": success,
        "failure_mode": "" if success else "low_sharpe",
        "timestamp": "2026-04-26T10:00:00",
    }


def sample_run_history() -> list[dict]:
    """A realistic set of run history entries for testing."""
    return [
        make_history_entry("modify_params", 0.5, False, "m1", 1),
        make_history_entry("modify_params", 0.6, False, "m1", 2),
        make_history_entry("change_entry_logic", 1.2, True, "m1", 3),
        make_history_entry("add_filter", 0.8, False, "m2", 1),
        make_history_entry("add_filter", 1.5, True, "m2", 2),
        make_history_entry("replace_indicator", 0.3, False, "m3", 1),
        make_history_entry("replace_indicator", 0.4, False, "m3", 2),
        make_history_entry("full_rewrite", 1.8, True, "m3", 3),
        make_history_entry("modify_params", 0.7, False, "m4", 1),
        make_history_entry("modify_params", 0.9, False, "m4", 2),
        make_history_entry("modify_params", 1.1, True, "m4", 3),
        make_history_entry("change_exit_logic", 1.6, True, "m5", 1),
    ]


# ── track_action_result ───────────────────────────────────────────────────────

class TestTrackActionResult:

    def test_appends_to_history_file(self):
        """Tracking an action should append a JSON line to the history file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = f.name

        try:
            with patch("crabquant.refinement.action_analytics.RUN_HISTORY_FILE", tmp_path):
                track_action_result(
                    mandate="test_m",
                    turn=1,
                    action="modify_params",
                    sharpe=1.2,
                    success=True,
                    failure_mode="",
                )

            entries = load_run_history(tmp_path)
            assert len(entries) == 1
            assert entries[0]["action"] == "modify_params"
            assert entries[0]["success"] is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_multiple_appends(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = f.name

        try:
            with patch("crabquant.refinement.action_analytics.RUN_HISTORY_FILE", tmp_path):
                track_action_result("m1", 1, "modify_params", 0.5, False, "low_sharpe")
                track_action_result("m1", 2, "change_entry_logic", 1.5, True, "")

            entries = load_run_history(tmp_path)
            assert len(entries) == 2
            assert entries[0]["action"] == "modify_params"
            assert entries[1]["action"] == "change_entry_logic"
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ── aggregate_action_stats ────────────────────────────────────────────────────

class TestAggregateActionStats:

    def test_basic_aggregation(self):
        history = sample_run_history()
        stats = aggregate_action_stats(history)

        assert isinstance(stats, dict)
        assert "modify_params" in stats
        assert "change_entry_logic" in stats

    def test_stats_include_count_and_success_rate(self):
        history = sample_run_history()
        stats = aggregate_action_stats(history)

        mp = stats["modify_params"]
        assert mp["total"] == 5
        assert mp["successes"] == 1
        assert 0 <= mp["success_rate"] <= 1

    def test_avg_sharpe_computed(self):
        history = sample_run_history()
        stats = aggregate_action_stats(history)

        mp = stats["modify_params"]
        assert "avg_sharpe" in mp
        expected_avg = (0.5 + 0.6 + 0.7 + 0.9 + 1.1) / 5
        assert abs(mp["avg_sharpe"] - expected_avg) < 0.01

    def test_empty_history_returns_empty(self):
        stats = aggregate_action_stats([])
        assert stats == {}


# ── compute_action_success_rates ──────────────────────────────────────────────

class TestComputeActionSuccessRates:

    def test_returns_sorted_list(self):
        history = sample_run_history()
        rates = compute_action_success_rates(history)

        assert isinstance(rates, list)
        assert len(rates) > 0
        # Should be sorted by success_rate descending
        for i in range(1, len(rates)):
            assert rates[i]["success_rate"] <= rates[i - 1]["success_rate"]

    def test_each_entry_has_required_fields(self):
        history = sample_run_history()
        rates = compute_action_success_rates(history)

        for entry in rates:
            assert "action" in entry
            assert "success_rate" in entry
            assert "total" in entry
            assert "successes" in entry
            assert "avg_sharpe" in entry

    def test_best_action_has_highest_rate(self):
        history = sample_run_history()
        rates = compute_action_success_rates(history)

        # change_exit_logic and full_rewrite should be 100%
        best = rates[0]
        assert best["success_rate"] == 1.0

    def test_empty_history_returns_empty(self):
        rates = compute_action_success_rates([])
        assert rates == []


# ── generate_llm_context ─────────────────────────────────────────────────────

class TestGenerateLLMContext:

    def test_returns_string(self):
        history = sample_run_history()
        ctx = generate_llm_context(history)
        assert isinstance(ctx, str)

    def test_includes_action_types(self):
        history = sample_run_history()
        ctx = generate_llm_context(history)
        assert "modify_params" in ctx
        assert "change_entry_logic" in ctx

    def test_includes_success_rates(self):
        history = sample_run_history()
        ctx = generate_llm_context(history)
        assert "success rate" in ctx.lower() or "Success rate" in ctx

    def test_empty_history_returns_fallback(self):
        ctx = generate_llm_context([])
        assert isinstance(ctx, str)
        assert len(ctx) > 0


# ── load_run_history ──────────────────────────────────────────────────────────

class TestLoadRunHistory:

    def test_loads_from_valid_file(self):
        history = sample_run_history()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in history:
                f.write(json.dumps(entry) + "\n")
            tmp_path = f.name

        try:
            loaded = load_run_history(tmp_path)
            assert len(loaded) == len(history)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_returns_empty_for_missing_file(self):
        loaded = load_run_history("/nonexistent/path.jsonl")
        assert loaded == []

    def test_skips_malformed_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"action": "modify_params", "sharpe": 1.0}\n')
            f.write("NOT JSON\n")
            f.write('{"action": "add_filter", "sharpe": 1.5}\n')
            tmp_path = f.name

        try:
            loaded = load_run_history(tmp_path)
            assert len(loaded) == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_skips_empty_lines(self):
        """Empty lines in JSONL are silently skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"action": "modify_params", "sharpe": 1.0}\n')
            f.write("\n")
            f.write('{"action": "add_filter", "sharpe": 1.5}\n')
            f.write("\n\n")
            tmp_path = f.name

        try:
            loaded = load_run_history(tmp_path)
            assert len(loaded) == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_all_malformed_returns_empty(self):
        """File with only garbage lines returns empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("garbage\n")
            f.write("more garbage\n")
            tmp_path = f.name

        try:
            loaded = load_run_history(tmp_path)
            assert loaded == []
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ── track_action_result additional ───────────────────────────────────────


class TestTrackActionResultEdgeCases:
    """Additional tests for track_action_result."""

    def test_creates_parent_directories(self, tmp_path):
        """Track creates parent directories if they don't exist."""
        deep_path = str(tmp_path / "sub" / "dir" / "history.jsonl")
        track_action_result("m1", 1, "modify_params", 1.0, True, path=deep_path)

        loaded = load_run_history(deep_path)
        assert len(loaded) == 1

    def test_explicit_path_parameter(self):
        """Explicit path parameter is used instead of default."""
        with tempfile.NamedTemporaryFile(mode="w", suffix="..jsonl", delete=False) as f:
            tmp_path = f.name

        try:
            track_action_result("m1", 1, "full_rewrite", 2.0, True, path=tmp_path)
            loaded = load_run_history(tmp_path)
            assert loaded[0]["action"] == "full_rewrite"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_entry_has_timestamp(self):
        """Tracked entry includes a timestamp field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = f.name

        try:
            track_action_result("m1", 1, "modify_params", 1.0, False, "low_sharpe", path=tmp_path)
            loaded = load_run_history(tmp_path)
            assert "timestamp" in loaded[0]
            assert loaded[0]["failure_mode"] == "low_sharpe"
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ── aggregate_action_stats additional ────────────────────────────────────


class TestAggregateActionStatsEdgeCases:
    """Additional edge cases for aggregation."""

    def test_missing_action_key_defaults_to_unknown(self):
        """Entries without 'action' key default to 'unknown'."""
        history = [{"sharpe": 1.0, "success": True}]
        stats = aggregate_action_stats(history)
        assert "unknown" in stats
        assert stats["unknown"]["total"] == 1

    def test_missing_success_key_defaults_to_false(self):
        """Entries without 'success' key are treated as failures."""
        history = [{"action": "modify_params", "sharpe": 1.0}]
        stats = aggregate_action_stats(history)
        assert stats["modify_params"]["successes"] == 0
        assert stats["modify_params"]["failures"] == 1

    def test_missing_sharpe_key_defaults_to_zero(self):
        """Entries without 'sharpe' key default to 0.0 for avg."""
        history = [
            {"action": "modify_params", "success": True},  # no sharpe
            {"action": "modify_params", "success": True, "sharpe": 2.0},
        ]
        stats = aggregate_action_stats(history)
        assert stats["modify_params"]["avg_sharpe"] == 1.0  # (0 + 2) / 2

    def test_single_entry(self):
        """Single history entry produces correct stats."""
        history = [make_history_entry("add_filter", 1.5, True)]
        stats = aggregate_action_stats(history)
        assert stats["add_filter"]["total"] == 1
        assert stats["add_filter"]["success_rate"] == 1.0
        assert stats["add_filter"]["avg_sharpe"] == 1.5

    def test_all_failures(self):
        """All-failure history produces 0% success rate."""
        history = [
            make_history_entry("modify_params", 0.3, False),
            make_history_entry("modify_params", 0.4, False),
            make_history_entry("modify_params", 0.5, False),
        ]
        stats = aggregate_action_stats(history)
        assert stats["modify_params"]["success_rate"] == 0.0
        assert stats["modify_params"]["failures"] == 3

    def test_all_successes(self):
        """All-success history produces 100% success rate."""
        history = [
            make_history_entry("full_rewrite", 1.5, True),
            make_history_entry("full_rewrite", 2.0, True),
        ]
        stats = aggregate_action_stats(history)
        assert stats["full_rewrite"]["success_rate"] == 1.0
        assert stats["full_rewrite"]["successes"] == 2


# ── generate_llm_context additional ─────────────────────────────────────


class TestGenerateLLMContextEdgeCases:
    """Additional tests for LLM context generation."""

    def test_includes_recommendation_line(self):
        """Context includes recommendation for best and worst actions."""
        history = sample_run_history()
        ctx = generate_llm_context(history)
        assert "Recommendation" in ctx

    def test_includes_avg_sharpe_values(self):
        """Context includes avg Sharpe for each action."""
        history = sample_run_history()
        ctx = generate_llm_context(history)
        assert "avg Sharpe" in ctx or "Sharpe" in ctx

    def test_single_action_history(self):
        """Context works with only one action type."""
        history = [
            make_history_entry("modify_params", 1.0, True),
            make_history_entry("modify_params", 0.5, False),
        ]
        ctx = generate_llm_context(history)
        assert "modify_params" in ctx
        assert "Recommendation" in ctx

    def test_fallback_message_for_empty(self):
        """Empty history returns first-run message."""
        ctx = generate_llm_context([])
        assert "first run" in ctx.lower()


# ── RUN_HISTORY_FILE constant ───────────────────────────────────────────


class TestConstants:
    """Test module constants."""

    def test_run_history_file_default(self):
        """RUN_HISTORY_FILE points to results/run_history.jsonl."""
        assert "run_history.jsonl" in RUN_HISTORY_FILE

    def test_action_stats_is_dict(self):
        """ActionStats is an alias for dict."""
        assert ActionStats is dict


