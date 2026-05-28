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
