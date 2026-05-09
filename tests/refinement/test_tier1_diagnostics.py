"""Tests for crabquant.refinement.tier1_diagnostics — enhanced Tier 1 diagnostics."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from crabquant.refinement.tier1_diagnostics import (
    build_tier1_report,
    format_previous_attempts,
    compute_consecutive_modify_params,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_backtest_result(**overrides):
    """Create a mock BacktestResult-like object."""
    defaults = dict(
        sharpe=1.2,
        total_return=0.15,
        max_drawdown=-0.10,
        win_rate=0.55,
        num_trades=30,
        profit_factor=1.5,
        calmar_ratio=1.5,
        sortino_ratio=2.0,
        expected_value=0.0,
        score=1.2,
        passed=False,
        params={"period": 14, "threshold": 0.5},
        ticker="AAPL",
        strategy_name="test_strategy",
        iteration=3,
    )
    defaults.update(overrides)
    return type("BacktestResult", (), defaults)()


def make_portfolio_mock(years=None, n_per_year=252, seed=42):
    """Mock vbt.Portfolio with .returns() spanning given years."""
    rng = np.random.default_rng(seed)
    if years is None:
        years = [2023, 2024]
    idx_parts = []
    for yr in years:
        idx_parts.append(pd.date_range(f"{yr}-01-03", periods=n_per_year, freq="B"))
    idx = idx_parts[0]
    for p in idx_parts[1:]:
        idx = idx.append(p)
    returns = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
    pf = MagicMock()
    pf.returns.return_value = returns
    return pf


# ── format_previous_attempts ─────────────────────────────────────────────────

class TestFormatPreviousAttempts:

    def test_empty_history(self):
        result = format_previous_attempts([])
        assert result == ""

    def test_single_attempt(self):
        history = [{
            "turn": 1,
            "sharpe": 0.45,
            "failure_mode": "too_few_trades",
            "action": "add_filter",
            "hypothesis": "Volume filter will generate more signals",
            "params_used": {"period": 14},
            "delta_from_prev": "Initial strategy (no prior version)",
        }]
        result = format_previous_attempts(history)
        assert "Turn 1" in result
        assert "0.45" in result
        assert "too_few_trades" in result
        assert "add_filter" in result
        assert "Volume filter" in result
        assert "period" in result

    def test_multiple_attempts(self):
        history = [
            {
                "turn": 1, "sharpe": 0.3, "failure_mode": "too_few_trades",
                "action": "add_filter", "hypothesis": "h1",
                "params_used": {"a": 1}, "delta_from_prev": "Initial",
            },
            {
                "turn": 2, "sharpe": 0.8, "failure_mode": "low_sharpe",
                "action": "modify_params", "hypothesis": "h2",
                "params_used": {"a": 2}, "delta_from_prev": "Action: add_filter",
            },
        ]
        result = format_previous_attempts(history)
        assert "Turn 1" in result
        assert "Turn 2" in result
        assert "0.30" in result
        assert "0.80" in result

    def test_missing_optional_keys_no_crash(self):
        history = [{"turn": 1, "sharpe": 0.5}]
        result = format_previous_attempts(history)
        assert "Turn 1" in result
        assert "0.50" in result

    def test_sharpe_formatted_to_two_decimals(self):
        history = [{"turn": 1, "sharpe": 0.123456}]
        result = format_previous_attempts(history)
        assert "0.12" in result

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_empty_dict_in_history(self):
        """Completely empty dict entry uses defaults."""
        history = [{}]
        result = format_previous_attempts(history)
        assert "Turn ?" in result
        assert "unknown" in result
        assert "N/A" in result

    def test_negative_sharpe(self):
        history = [{"turn": 1, "sharpe": -0.75}]
        result = format_previous_attempts(history)
        assert "-0.75" in result

    def test_zero_sharpe(self):
        history = [{"turn": 1, "sharpe": 0.0}]
        result = format_previous_attempts(history)
        assert "0.00" in result

    def test_large_sharpe(self):
        history = [{"turn": 1, "sharpe": 5.6789}]
        result = format_previous_attempts(history)
        assert "5.68" in result

    def test_multiline_hypothesis(self):
        history = [{
            "turn": 1, "sharpe": 1.0,
            "hypothesis": "Line 1\nLine 2\nLine 3",
        }]
        result = format_previous_attempts(history)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_special_chars_in_failure_mode(self):
        history = [{"turn": 1, "sharpe": 0.5, "failure_mode": "max_dd > -25% (exceeded)"}]
        result = format_previous_attempts(history)
        assert "max_dd > -25%" in result

    def test_empty_params_used(self):
        history = [{"turn": 1, "sharpe": 0.5, "params_used": {}}]
        result = format_previous_attempts(history)
        assert "Params used: {}" in result

    def test_complex_params_used(self):
        history = [{
            "turn": 1, "sharpe": 0.5,
            "params_used": {"fast": 12, "slow": 26, "threshold": 0.75},
        }]
        result = format_previous_attempts(history)
        assert "fast" in result
        assert "slow" in result
        assert "threshold" in result

    def test_each_entry_on_new_line(self):
        history = [
            {"turn": 1, "sharpe": 0.5},
            {"turn": 2, "sharpe": 0.8},
        ]
        result = format_previous_attempts(history)
        lines = result.strip().split("\n")
        # Each entry has 6 lines, so 2 entries = 12 lines
        assert len(lines) == 12

    def test_delta_from_prev_default(self):
        """Missing delta_from_prev defaults to 'N/A'."""
        history = [{"turn": 1, "sharpe": 0.5}]
        result = format_previous_attempts(history)
        assert "N/A" in result

    def test_returns_string_type(self):
        history = [{"turn": 1, "sharpe": 0.5}]
        result = format_previous_attempts(history)
        assert isinstance(result, str)

    def test_unicode_in_hypothesis(self):
        history = [{"turn": 1, "sharpe": 0.5, "hypothesis": "Strategy fails during bear 🐻"}]
        result = format_previous_attempts(history)
        assert "🐻" in result


# ── compute_consecutive_modify_params ────────────────────────────────────────

class TestComputeConsecutiveModifyParams:

    def test_empty_history(self):
        assert compute_consecutive_modify_params([]) == 0

    def test_no_modify_params(self):
        history = [
            {"action": "add_filter"},
            {"action": "change_entry_logic"},
        ]
        assert compute_consecutive_modify_params(history) == 0

    def test_one_modify_params(self):
        history = [{"action": "modify_params"}]
        assert compute_consecutive_modify_params(history) == 1

    def test_two_consecutive(self):
        history = [
            {"action": "add_filter"},
            {"action": "modify_params"},
            {"action": "modify_params"},
        ]
        assert compute_consecutive_modify_params(history) == 2

    def test_three_consecutive(self):
        history = [
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "modify_params"},
        ]
        assert compute_consecutive_modify_params(history) == 3

    def test_broken_streak(self):
        history = [
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "add_filter"},
            {"action": "modify_params"},
        ]
        assert compute_consecutive_modify_params(history) == 1

    def test_missing_action_key(self):
        # Empty dict has no action → breaks the streak
        history = [{"action": "modify_params"}, {}]
        assert compute_consecutive_modify_params(history) == 0

    def test_missing_action_at_start(self):
        # Empty dict at start, then modify_params → streak is 1
        history = [{}, {"action": "modify_params"}]
        assert compute_consecutive_modify_params(history) == 1

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_all_different_actions(self):
        history = [
            {"action": "add_filter"},
            {"action": "change_entry_logic"},
            {"action": "full_rewrite"},
        ]
        assert compute_consecutive_modify_params(history) == 0

    def test_long_streak(self):
        history = [{"action": "modify_params"} for _ in range(10)]
        assert compute_consecutive_modify_params(history) == 10

    def test_modify_params_not_at_end(self):
        """Only tail consecutive modify_params counts."""
        history = [
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "add_filter"},
        ]
        assert compute_consecutive_modify_params(history) == 0

    def test_single_non_modify_action(self):
        history = [{"action": "full_rewrite"}]
        assert compute_consecutive_modify_params(history) == 0

    def test_returns_int(self):
        result = compute_consecutive_modify_params([{"action": "modify_params"}])
        assert isinstance(result, int)

    def test_none_value_for_action(self):
        """None value for action key should break the streak."""
        history = [{"action": "modify_params"}, {"action": None}]
        assert compute_consecutive_modify_params(history) == 0

    def test_empty_string_action(self):
        """Empty string for action should break the streak."""
        history = [{"action": "modify_params"}, {"action": ""}]
        assert compute_consecutive_modify_params(history) == 0

    def test_interleaved_streaks(self):
        """Only the last streak counts."""
        history = [
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "add_filter"},
            {"action": "modify_params"},
            {"action": "modify_params"},
            {"action": "modify_params"},
        ]
        assert compute_consecutive_modify_params(history) == 3


# ── build_tier1_report ──────────────────────────────────────────────────────

class TestBuildTier1Report:

    def _make_context(self, **overrides):
        """Build minimal context dict for build_tier1_report."""
        defaults = {
            "backtest_result": make_backtest_result(),
            "failure_mode": "low_sharpe",
            "failure_details": "Sharpe 1.2 < target 1.5",
            "sharpe_by_year": {"2023": 1.8, "2024": 0.3},
            "stagnation_score": 0.4,
            "stagnation_trend": "flat",
            "history": [
                {"turn": 1, "sharpe": 0.45, "failure_mode": "too_few_trades",
                 "action": "add_filter", "hypothesis": "h1",
                 "params_used": {"p": 1}, "delta_from_prev": "Initial"},
                {"turn": 2, "sharpe": 0.8, "failure_mode": "low_sharpe",
                 "action": "modify_params", "hypothesis": "h2",
                 "params_used": {"p": 2}, "delta_from_prev": "Changed params"},
            ],
            "guardrail_violations": ["max_drawdown exceeded"],
            "guardrail_warnings": ["low win rate"],
            "current_strategy_code": "def generate_signals(df, params):\n    pass",
            "current_params": {"period": 14},
            "strategy_id": "run_001",
            "iteration": 3,
        }
        defaults.update(overrides)
        return defaults

    def test_returns_dict_with_required_keys(self):
        ctx = self._make_context()
        report = build_tier1_report(**ctx)
        assert isinstance(report, dict)
        # All Tier 1 fields present
        assert "strategy_id" in report
        assert "iteration" in report
        assert "sharpe_ratio" in report
        assert "total_return_pct" in report
        assert "max_drawdown_pct" in report
        assert "win_rate" in report
        assert "total_trades" in report
        assert "failure_mode" in report
        assert "sharpe_by_year" in report
        assert "stagnation_score" in report
        assert "stagnation_trend" in report
        assert "previous_sharpes" in report
        assert "previous_actions" in report
        assert "guardrail_violations" in report
        assert "current_strategy_code" in report
        assert "current_params" in report
        assert "previous_attempts" in report
        assert "consecutive_modify_params" in report

    def test_sharpe_mapping(self):
        ctx = self._make_context(backtest_result=make_backtest_result(sharpe=2.5))
        report = build_tier1_report(**ctx)
        assert report["sharpe_ratio"] == 2.5

    def test_return_mapping(self):
        ctx = self._make_context(backtest_result=make_backtest_result(total_return=0.22))
        report = build_tier1_report(**ctx)
        assert report["total_return_pct"] == 0.22

    def test_drawdown_mapping(self):
        ctx = self._make_context(backtest_result=make_backtest_result(max_drawdown=-0.18))
        report = build_tier1_report(**ctx)
        assert report["max_drawdown_pct"] == -0.18

    def test_trades_mapping(self):
        ctx = self._make_context(backtest_result=make_backtest_result(num_trades=42))
        report = build_tier1_report(**ctx)
        assert report["total_trades"] == 42

    def test_previous_sharpes_from_history(self):
        ctx = self._make_context()
        report = build_tier1_report(**ctx)
        assert report["previous_sharpes"] == [0.45, 0.8]

    def test_previous_actions_from_history(self):
        ctx = self._make_context()
        report = build_tier1_report(**ctx)
        assert report["previous_actions"] == ["add_filter", "modify_params"]

    def test_consecutive_modify_params_in_report(self):
        history = [
            {"action": "modify_params"},
            {"action": "modify_params"},
        ]
        ctx = self._make_context(history=history)
        report = build_tier1_report(**ctx)
        assert report["consecutive_modify_params"] == 2

    def test_previous_attempts_last_three(self):
        history = [
            {"turn": i, "sharpe": float(i)} for i in range(1, 6)
        ]
        ctx = self._make_context(history=history)
        report = build_tier1_report(**ctx)
        # Should include last 3 entries
        assert len(report["previous_attempts"]) == 3
        assert report["previous_attempts"][0]["turn"] == 3

    def test_empty_history(self):
        ctx = self._make_context(history=[])
        report = build_tier1_report(**ctx)
        assert report["previous_sharpes"] == []
        assert report["previous_actions"] == []
        assert report["previous_attempts"] == []
        assert report["consecutive_modify_params"] == 0

    def test_no_guardrails(self):
        ctx = self._make_context(guardrail_violations=[], guardrail_warnings=[])
        report = build_tier1_report(**ctx)
        assert report["guardrail_violations"] == []
        assert report["guardrail_warnings"] == []

    def test_sharpe_by_year_preserved(self):
        ctx = self._make_context(sharpe_by_year={"2023": 2.5, "2024": 0.1})
        report = build_tier1_report(**ctx)
        assert report["sharpe_by_year"] == {"2023": 2.5, "2024": 0.1}

    def test_calmar_and_sortino_mapped(self):
        ctx = self._make_context(
            backtest_result=make_backtest_result(calmar_ratio=3.0, sortino_ratio=4.0)
        )
        report = build_tier1_report(**ctx)
        assert report["calmar_ratio"] == 3.0
        assert report["sortino_ratio"] == 4.0

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_default_strategy_id_and_iteration(self):
        """Default strategy_id and iteration when not provided."""
        ctx = self._make_context(strategy_id="", iteration=0)
        report = build_tier1_report(**ctx)
        assert report["strategy_id"] == ""
        assert report["iteration"] == 0

    def test_win_rate_mapping(self):
        ctx = self._make_context(backtest_result=make_backtest_result(win_rate=0.72))
        report = build_tier1_report(**ctx)
        assert report["win_rate"] == 0.72

    def test_profit_factor_mapping(self):
        ctx = self._make_context(backtest_result=make_backtest_result(profit_factor=2.5))
        report = build_tier1_report(**ctx)
        assert report["profit_factor"] == 2.5

    def test_composite_score_mapping(self):
        ctx = self._make_context(backtest_result=make_backtest_result(score=0.95))
        report = build_tier1_report(**ctx)
        assert report["composite_score"] == 0.95

    def test_failure_mode_and_details(self):
        ctx = self._make_context(
            failure_mode="high_drawdown",
            failure_details="Max drawdown of -35% exceeded threshold",
        )
        report = build_tier1_report(**ctx)
        assert report["failure_mode"] == "high_drawdown"
        assert report["failure_details"] == "Max drawdown of -35% exceeded threshold"

    def test_stagnation_score_and_trend(self):
        ctx = self._make_context(stagnation_score=0.85, stagnation_trend="declining")
        report = build_tier1_report(**ctx)
        assert report["stagnation_score"] == 0.85
        assert report["stagnation_trend"] == "declining"

    def test_all_stagnation_trends(self):
        for trend in ("improving", "flat", "declining"):
            ctx = self._make_context(stagnation_trend=trend)
            report = build_tier1_report(**ctx)
            assert report["stagnation_trend"] == trend

    def test_guardrail_warnings_none_becomes_empty_list(self):
        """None guardrail_warnings should default to empty list."""
        ctx = self._make_context(guardrail_warnings=None)
        report = build_tier1_report(**ctx)
        assert report["guardrail_warnings"] == []

    def test_current_strategy_code_preserved(self):
        code = "def go(df):\n    return df\n"
        ctx = self._make_context(current_strategy_code=code)
        report = build_tier1_report(**ctx)
        assert report["current_strategy_code"] == code

    def test_current_params_none_becomes_empty_dict(self):
        ctx = self._make_context(current_params=None)
        report = build_tier1_report(**ctx)
        assert report["current_params"] == {}

    def test_current_params_preserved(self):
        params = {"fast": 10, "slow": 30, "signal": "macd"}
        ctx = self._make_context(current_params=params)
        report = build_tier1_report(**ctx)
        assert report["current_params"] == params

    def test_previous_sharpes_defaults_when_missing(self):
        """History entries without sharpe key default to 0.0."""
        history = [{"turn": 1}, {"turn": 2}]
        ctx = self._make_context(history=history)
        report = build_tier1_report(**ctx)
        assert report["previous_sharpes"] == [0.0, 0.0]

    def test_previous_actions_defaults_when_missing(self):
        """History entries without action key default to empty string."""
        history = [{"turn": 1}, {"turn": 2}]
        ctx = self._make_context(history=history)
        report = build_tier1_report(**ctx)
        assert report["previous_actions"] == ["", ""]

    def test_sharpe_by_year_empty(self):
        ctx = self._make_context(sharpe_by_year={})
        report = build_tier1_report(**ctx)
        assert report["sharpe_by_year"] == {}

    def test_negative_sharpe_values(self):
        ctx = self._make_context(backtest_result=make_backtest_result(sharpe=-0.5))
        report = build_tier1_report(**ctx)
        assert report["sharpe_ratio"] == -0.5

    def test_zero_trades(self):
        ctx = self._make_context(backtest_result=make_backtest_result(num_trades=0))
        report = build_tier1_report(**ctx)
        assert report["total_trades"] == 0

    def test_multiple_guardrail_violations(self):
        violations = ["max_drawdown_exceeded", "min_trades_not_met", "overfitting_risk"]
        ctx = self._make_context(guardrail_violations=violations)
        report = build_tier1_report(**ctx)
        assert report["guardrail_violations"] == violations

    def test_history_with_many_entries(self):
        """With 10 history entries, previous_attempts should be last 3."""
        history = [{"turn": i, "sharpe": float(i)} for i in range(1, 11)]
        ctx = self._make_context(history=history)
        report = build_tier1_report(**ctx)
        assert len(report["previous_attempts"]) == 3
        assert report["previous_attempts"][-1]["turn"] == 10

    def test_history_with_less_than_three(self):
        """With 2 history entries, previous_attempts should be all 2."""
        history = [{"turn": 1, "sharpe": 0.5}, {"turn": 2, "sharpe": 0.8}]
        ctx = self._make_context(history=history)
        report = build_tier1_report(**ctx)
        assert len(report["previous_attempts"]) == 2

    def test_negative_max_drawdown(self):
        ctx = self._make_context(backtest_result=make_backtest_result(max_drawdown=-0.50))
        report = build_tier1_report(**ctx)
        assert report["max_drawdown_pct"] == -0.50

    def test_negative_total_return(self):
        ctx = self._make_context(backtest_result=make_backtest_result(total_return=-0.10))
        report = build_tier1_report(**ctx)
        assert report["total_return_pct"] == -0.10

    def test_report_is_json_serializable(self):
        """Report dict should be JSON-serializable."""
        import json
        ctx = self._make_context()
        report = build_tier1_report(**ctx)
        blob = json.dumps(report)
        assert isinstance(blob, str)

    def test_zero_stagnation_score(self):
        ctx = self._make_context(stagnation_score=0.0)
        report = build_tier1_report(**ctx)
        assert report["stagnation_score"] == 0.0

    def test_max_stagnation_score(self):
        ctx = self._make_context(stagnation_score=1.0)
        report = build_tier1_report(**ctx)
        assert report["stagnation_score"] == 1.0
