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


