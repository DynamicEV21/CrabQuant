"""Tests for crabquant.refinement.action_effectiveness — Phase 6."""

import json
import tempfile
from pathlib import Path

import pytest

from crabquant.refinement.action_effectiveness import (
    SKIP_MANDATES,
    analyze_action_effectiveness,
    format_action_effectiveness_for_prompt,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_history(entries: list[dict], tmp_path: Path) -> str:
    """Write history entries to a temp JSONL file and return the path."""
    p = tmp_path / "run_history.jsonl"
    with p.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return str(p)


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


def _sample_history() -> list[dict]:
    """Realistic multi-failure-mode history for testing."""
    return [
        # low_sharpe failures
        _make_entry("momentum_spy", 1, "novel", 0.3, False, "low_sharpe"),
        _make_entry("momentum_spy", 2, "novel", 0.5, False, "low_sharpe"),
        _make_entry("momentum_spy", 3, "replace_indicator", 0.4, False, "low_sharpe"),
        _make_entry("momentum_spy", 4, "replace_indicator", 0.6, False, "low_sharpe"),
        _make_entry("momentum_spy", 5, "modify_params", 0.7, False, "low_sharpe"),
        _make_entry("momentum_spy", 6, "modify_params", 0.8, True, "low_sharpe"),
        _make_entry("momentum_spy", 7, "novel", 1.0, True, "low_sharpe"),
        _make_entry("momentum_spy", 8, "novel", 0.2, False, "low_sharpe"),
        _make_entry("mean_rev_aapl", 1, "novel", 0.1, False, "low_sharpe"),
        _make_entry("mean_rev_aapl", 2, "novel", 1.5, True, "low_sharpe"),
        # too_few_trades failures
        _make_entry("momentum_spy", 1, "change_entry_logic", 0.5, False, "too_few_trades"),
        _make_entry("momentum_spy", 2, "change_entry_logic", 1.2, True, "too_few_trades"),
        _make_entry("momentum_spy", 3, "full_rewrite", 0.8, False, "too_few_trades"),
        _make_entry("mean_rev_aapl", 1, "change_entry_logic", 0.3, False, "too_few_trades"),
        _make_entry("mean_rev_aapl", 2, "change_entry_logic", 0.4, False, "too_few_trades"),
        # excessive_drawdown failures
        _make_entry("breakout_qqq", 1, "add_filter", 0.9, False, "excessive_drawdown"),
        _make_entry("breakout_qqq", 2, "add_filter", 1.0, True, "excessive_drawdown"),
        _make_entry("breakout_qqq", 3, "add_regime_filter", 1.1, True, "excessive_drawdown"),
        # backtest_crash failures
        _make_entry("test_real", 1, "novel", 0.0, False, "backtest_crash"),
        _make_entry("test_real", 2, "novel", 0.0, False, "backtest_crash"),
        _make_entry("test_real", 3, "novel", 1.5, True, "backtest_crash"),
        # successful entries (no failure_mode) — should be excluded from per-mode stats
        _make_entry("winner_strategy", 1, "novel", 2.0, True, ""),
        _make_entry("winner_strategy", 2, "modify_params", 1.8, True, ""),
        # Test mandates — should be filtered out
        _make_entry("smoke_test", 1, "novel", 1.0, True, ""),
        _make_entry("test_mandate", 1, "modify_params", 0.5, False, "low_sharpe"),
        _make_entry("e2e_stress_test", 1, "full_rewrite", 0.3, False, "low_sharpe"),
        _make_entry("e2e_test_momentum", 1, "novel", 0.4, False, "low_sharpe"),
        _make_entry("single_turn", 1, "novel", 0.5, False, "low_sharpe"),
        _make_entry("multi_ticker", 1, "novel", 0.6, False, "low_sharpe"),
        _make_entry("history_test", 1, "novel", 0.7, False, "low_sharpe"),
        _make_entry("modify_test", 1, "novel", 0.8, False, "low_sharpe"),
        _make_entry("module_fail", 1, "novel", 0.9, False, "low_sharpe"),
        _make_entry("unreachable_target", 1, "novel", 0.2, False, "low_sharpe"),
        _make_entry("high_target", 1, "novel", 0.3, False, "low_sharpe"),
    ]


# ── SKIP_MANDATES constant ──────────────────────────────────────────────────


class TestSkipMandates:
    """Test that the skip list contains all expected mandates."""

    def test_smoke_test_in_skip(self):
        assert "smoke_test" in SKIP_MANDATES

    def test_test_mandate_in_skip(self):
        assert "test_mandate" in SKIP_MANDATES

    def test_e2e_stress_test_in_skip(self):
        assert "e2e_stress_test" in SKIP_MANDATES

    def test_e2e_test_momentum_in_skip(self):
        assert "e2e_test_momentum" in SKIP_MANDATES

    def test_single_turn_in_skip(self):
        assert "single_turn" in SKIP_MANDATES

    def test_multi_ticker_in_skip(self):
        assert "multi_ticker" in SKIP_MANDATES

    def test_history_test_in_skip(self):
        assert "history_test" in SKIP_MANDATES

    def test_modify_test_in_skip(self):
        assert "modify_test" in SKIP_MANDATES

    def test_module_fail_in_skip(self):
        assert "module_fail" in SKIP_MANDATES

    def test_unreachable_target_in_skip(self):
        assert "unreachable_target" in SKIP_MANDATES

    def test_high_target_in_skip(self):
        assert "high_target" in SKIP_MANDATES

    def test_is_frozenset(self):
        assert isinstance(SKIP_MANDATES, frozenset)


# ── analyze_action_effectiveness ────────────────────────────────────────────


class TestAnalyzeActionEffectiveness:

    def test_returns_dict(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        assert isinstance(result, dict)

    def test_total_entries_excludes_skip_mandates(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        # 34 total - 11 skip mandates = 23 real entries
        assert result["total_entries"] == 23

    def test_total_entries_excludes_success_entries_from_modes(self, tmp_path):
        """Successful entries (empty failure_mode) are counted in total_entries
        but excluded from by_failure_mode."""
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        # 23 real entries, 2 with empty failure_mode
        total_in_modes = sum(
            m["total"] for m in result["by_failure_mode"].values()
        )
        assert total_in_modes == 21  # only failure-mode entries

    def test_has_by_failure_mode_key(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        assert "by_failure_mode" in result

    def test_low_sharpe_mode_present(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        assert "low_sharpe" in result["by_failure_mode"]

    def test_too_few_trades_mode_present(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        assert "too_few_trades" in result["by_failure_mode"]

    def test_excessive_drawdown_mode_present(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        assert "excessive_drawdown" in result["by_failure_mode"]

    def test_backtest_crash_mode_present(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        assert "backtest_crash" in result["by_failure_mode"]

    def test_low_sharpe_novel_action_stats(self, tmp_path):
        """novel action for low_sharpe: 6 real entries, 2 successes."""
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        novel = result["by_failure_mode"]["low_sharpe"]["actions"]["novel"]
        assert novel["total"] == 6
        assert novel["successes"] == 2

    def test_low_sharpe_success_rate(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        novel = result["by_failure_mode"]["low_sharpe"]["actions"]["novel"]
        assert novel["success_rate"] == pytest.approx(2 / 6)

    def test_ranked_actions_sorted_by_success_rate(self, tmp_path):
        """ranked_actions should be sorted by success_rate descending."""
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        ranked = result["by_failure_mode"]["low_sharpe"]["ranked_actions"]
        for i in range(1, len(ranked)):
            assert ranked[i][1] <= ranked[i - 1][1]

    def test_ranked_actions_are_tuples(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        ranked = result["by_failure_mode"]["low_sharpe"]["ranked_actions"]
        for item in ranked:
            assert isinstance(item, tuple)
            assert len(item) == 4  # (action, rate, total, successes)

    def test_too_few_trades_change_entry_logic_stats(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        cel = result["by_failure_mode"]["too_few_trades"]["actions"]["change_entry_logic"]
        assert cel["total"] == 4
        assert cel["successes"] == 1

    def test_excessive_drawdown_add_filter_stats(self, tmp_path):
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        af = result["by_failure_mode"]["excessive_drawdown"]["actions"]["add_filter"]
        assert af["total"] == 2
        assert af["successes"] == 1

    def test_skip_mandate_not_in_results(self, tmp_path):
        """smoke_test and test_mandate entries should be filtered out."""
        path = _write_history(_sample_history(), tmp_path)
        result = analyze_action_effectiveness(path)
        # The test_mandate low_sharpe entry should NOT be counted
        novel = result["by_failure_mode"]["low_sharpe"]["actions"]["novel"]
        assert novel["total"] == 6  # momentum_spy(4) + mean_rev_aapl(2)

    def test_missing_file_returns_empty(self):
        result = analyze_action_effectiveness("/nonexistent/path.jsonl")
        assert result["total_entries"] == 0
        assert result["by_failure_mode"] == {}

    def test_empty_history_returns_empty(self, tmp_path):
        path = _write_history([], tmp_path)
        result = analyze_action_effectiveness(path)
        assert result["total_entries"] == 0
        assert result["by_failure_mode"] == {}

    def test_all_skip_mandates_returns_empty_modes(self, tmp_path):
        """If every entry is a skip mandate, by_failure_mode is empty."""
        entries = [
            _make_entry("smoke_test", 1, "novel", 0.3, False, "low_sharpe"),
            _make_entry("test_mandate", 1, "novel", 0.3, False, "low_sharpe"),
        ]
        path = _write_history(entries, tmp_path)
        result = analyze_action_effectiveness(path)
        assert result["total_entries"] == 0
        assert result["by_failure_mode"] == {}

    def test_all_successes_returns_empty_modes(self, tmp_path):
        """If every entry is a success (empty failure_mode), by_failure_mode is empty."""
        entries = [
            _make_entry("real_mandate", 1, "novel", 2.0, True, ""),
            _make_entry("real_mandate", 2, "modify_params", 1.5, True, ""),
        ]
        path = _write_history(entries, tmp_path)
        result = analyze_action_effectiveness(path)
        assert result["total_entries"] == 2
        assert result["by_failure_mode"] == {}

    def test_missing_action_defaults_to_unknown(self, tmp_path):
        """Entries without 'action' key default to 'unknown'."""
        entries = [{"mandate": "real", "failure_mode": "low_sharpe", "success": False}]
        path = _write_history(entries, tmp_path)
        result = analyze_action_effectiveness(path)
        assert "unknown" in result["by_failure_mode"]["low_sharpe"]["actions"]

    def test_missing_success_defaults_to_false(self, tmp_path):
        """Entries without 'success' key are treated as failures."""
        entries = [
            {"mandate": "real", "action": "novel", "failure_mode": "low_sharpe"},
        ]
        path = _write_history(entries, tmp_path)
        result = analyze_action_effectiveness(path)
        novel = result["by_failure_mode"]["low_sharpe"]["actions"]["novel"]
        assert novel["successes"] == 0

    def test_single_entry(self, tmp_path):
        path = _write_history(
            [_make_entry("real", 1, "novel", 0.3, False, "low_sharpe")],
            tmp_path,
        )
        result = analyze_action_effectiveness(path)
        assert result["total_entries"] == 1
        novel = result["by_failure_mode"]["low_sharpe"]["actions"]["novel"]
        assert novel["total"] == 1
        assert novel["successes"] == 0

    def test_malformed_json_lines_skipped(self, tmp_path):
        """Malformed JSON lines are silently skipped."""
        p = tmp_path / "run_history.jsonl"
        with p.open("w") as f:
            f.write('{"mandate":"real","action":"novel","failure_mode":"low_sharpe","success":false}\n')
            f.write("NOT JSON\n")
            f.write('{"mandate":"real","action":"novel","failure_mode":"low_sharpe","success":true}\n')
        result = analyze_action_effectiveness(str(p))
        assert result["total_entries"] == 2


# ── format_action_effectiveness_for_prompt ──────────────────────────────────


class TestFormatActionEffectivenessForPrompt:

    def _get_effectiveness(self, tmp_path, entries=None):
        """Helper to get effectiveness data from sample history."""
        entries = entries or _sample_history()
        path = _write_history(entries, tmp_path)
        return analyze_action_effectiveness(path)

    def test_returns_string(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert isinstance(result, str)

    def test_includes_header_with_total(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "Action Effectiveness" in result
        assert "23 real attempts" in result

    def test_includes_failure_mode_name(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "low_sharpe" in result

    def test_includes_action_names(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "novel" in result
        assert "replace_indicator" in result
        assert "modify_params" in result

    def test_includes_success_counts(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        # novel: 2/6
        assert "2/6" in result

    def test_includes_percentage(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "33.3%" in result  # novel success rate (2/6)

    def test_includes_best_so_far_marker(self, tmp_path):
        """The highest success_rate action should be marked 'best so far'."""
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        # modify_params has 1/1 = 100% for low_sharpe, should be marked best
        assert "best so far" in result

    def test_includes_recommendation(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "Recommended" in result

    def test_recommendation_includes_backticks(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "`" in result  # action names in backticks

    def test_empty_data_returns_empty_string(self):
        result = format_action_effectiveness_for_prompt({}, "low_sharpe")
        assert result == ""

    def test_no_data_for_failure_mode_returns_empty(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "nonexistent_mode")
        assert result == ""

    def test_missing_by_failure_mode_key_returns_empty(self):
        result = format_action_effectiveness_for_prompt(
            {"total_entries": 100}, "low_sharpe"
        )
        assert result == ""

    def test_too_few_trades_formatting(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "too_few_trades")
        assert "too_few_trades" in result
        assert "change_entry_logic" in result

    def test_excessive_drawdown_formatting(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "excessive_drawdown")
        assert "excessive_drawdown" in result
        assert "add_filter" in result
        assert "add_regime_filter" in result

    def test_backtest_crash_formatting(self, tmp_path):
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "backtest_crash")
        assert "backtest_crash" in result
        assert "novel" in result

    def test_all_zero_success_rates(self, tmp_path):
        """When all actions have 0% success, still shows recommendation."""
        entries = [
            _make_entry("real", 1, "novel", 0.1, False, "low_sharpe"),
            _make_entry("real", 2, "novel", 0.2, False, "low_sharpe"),
            _make_entry("real", 3, "replace_indicator", 0.3, False, "low_sharpe"),
        ]
        eff = self._get_effectiveness(tmp_path, entries)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "0.0%" in result
        assert "Recommended" in result
        # Should recommend the most-tried action
        assert "novel" in result

    def test_single_action_formatting(self, tmp_path):
        """Single action type still produces valid output."""
        entries = [
            _make_entry("real", 1, "novel", 0.5, True, "low_sharpe"),
            _make_entry("real", 2, "novel", 0.3, False, "low_sharpe"),
        ]
        eff = self._get_effectiveness(tmp_path, entries)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "novel" in result
        assert "1/2" in result
        assert "Recommended" in result

    def test_no_best_marker_when_all_zero(self, tmp_path):
        """No 'best so far' marker when all success rates are 0%."""
        entries = [
            _make_entry("real", 1, "novel", 0.1, False, "low_sharpe"),
            _make_entry("real", 2, "replace_indicator", 0.2, False, "low_sharpe"),
        ]
        eff = self._get_effectiveness(tmp_path, entries)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "best so far" not in result

    def test_recommendation_single_action(self, tmp_path):
        """When there's only one action, recommendation has single item."""
        entries = [
            _make_entry("real", 1, "novel", 0.5, True, "low_sharpe"),
        ]
        eff = self._get_effectiveness(tmp_path, entries)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert "Recommended: start with `novel`" in result
        assert " or " not in result

    def test_recommendation_two_actions(self, tmp_path):
        """When there are multiple successful actions, recommends top 2."""
        entries = [
            _make_entry("real", 1, "novel", 0.5, True, "low_sharpe"),
            _make_entry("real", 2, "modify_params", 0.5, True, "low_sharpe"),
            _make_entry("real", 3, "replace_indicator", 0.3, False, "low_sharpe"),
        ]
        eff = self._get_effectiveness(tmp_path, entries)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        assert " or " in result

    def test_empty_failure_mode_string(self, tmp_path):
        """Empty failure_mode string returns empty."""
        eff = self._get_effectiveness(tmp_path)
        result = format_action_effectiveness_for_prompt(eff, "")
        assert result == ""

    def test_output_format_matches_spec(self, tmp_path):
        """Output roughly matches the spec format."""
        entries = [
            _make_entry("real", 1, "novel", 0.3, False, "low_sharpe"),
            _make_entry("real", 2, "novel", 0.5, True, "low_sharpe"),
            _make_entry("real", 3, "replace_indicator", 0.4, False, "low_sharpe"),
        ]
        eff = self._get_effectiveness(tmp_path, entries)
        result = format_action_effectiveness_for_prompt(eff, "low_sharpe")
        # Check format structure
        assert result.startswith("### Action Effectiveness")
        assert "For low_sharpe failures:" in result
        assert "- novel:" in result
        assert "- replace_indicator:" in result
        assert "Recommended:" in result


# ── Integration: analyze → format pipeline ──────────────────────────────────


class TestIntegrationPipeline:

    def test_full_pipeline(self, tmp_path):
        """End-to-end: write history → analyze → format for prompt."""
        entries = _sample_history()
        path = _write_history(entries, tmp_path)

        eff = analyze_action_effectiveness(path)
        assert eff["total_entries"] > 0

        for mode in ["low_sharpe", "too_few_trades", "excessive_drawdown", "backtest_crash"]:
            formatted = format_action_effectiveness_for_prompt(eff, mode)
            assert isinstance(formatted, str)
            if formatted:
                assert mode in formatted
                assert "Recommended" in formatted

    def test_empty_file_pipeline(self, tmp_path):
        """Empty history produces no output."""
        path = _write_history([], tmp_path)
        eff = analyze_action_effectiveness(path)
        assert eff["total_entries"] == 0
        assert format_action_effectiveness_for_prompt(eff, "low_sharpe") == ""

    def test_only_skip_mandates_pipeline(self, tmp_path):
        """Only skip mandates produces no output."""
        entries = [
            _make_entry("smoke_test", 1, "novel", 1.0, True, ""),
            _make_entry("test_mandate", 1, "novel", 0.5, False, "low_sharpe"),
        ]
        path = _write_history(entries, tmp_path)
        eff = analyze_action_effectiveness(path)
        assert eff["total_entries"] == 0
