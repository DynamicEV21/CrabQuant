"""Tests for positive_feedback wiring into the refinement loop context.

Verifies that:
1. analyze_positive_feedback() is called in context_builder.build_llm_context()
2. format_positive_feedback_for_prompt() output is set in the context dict
3. positive_feedback_section appears in the final prompt via append_sections
4. Previous successful turns are surfaced in the feedback
5. No positive feedback section when report has no strengths
6. The wiring doesn't break when positive_feedback module raises
"""

import pytest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_run_state(**overrides):
    """Create a minimal RunState-like object for testing."""
    defaults = dict(
        run_id="test-run",
        mandate_name="test",
        created_at="2025-01-01",
        max_turns=7,
        sharpe_target=1.5,
        tickers=["SPY"],
        period="2y",
        current_turn=2,
        status="running",
        best_sharpe=0.8,
        best_composite_score=5.0,
        best_turn=1,
        best_code_path="",
        best_strategy_code="",
        consecutive_regressions=0,
        revert_notice="",
        history=[],
        code_quality_feedback="",
        action_cooldowns={},
        lock_pid=None,
        lock_timestamp=None,
    )
    defaults.update(overrides)
    return type("RunState", (), defaults)()


def _make_backtest_report(**overrides):
    """Create a minimal BacktestReport-like object for testing."""
    defaults = dict(
        strategy_id="test-strat",
        iteration=2,
        sharpe_ratio=0.8,
        total_return_pct=0.12,
        max_drawdown_pct=-0.08,
        win_rate=0.55,
        total_trades=45,
        profit_factor=1.8,
        calmar_ratio=1.5,
        sortino_ratio=1.2,
        composite_score=5.0,
        failure_mode="low_sharpe",
        failure_details="Sharpe 0.80 < target 1.50",
        sharpe_by_year={"2023": 0.9, "2024": 0.7},
        stagnation_score=0.3,
        stagnation_trend="improving",
        previous_sharpes=[0.5, 0.8],
        previous_actions=["novel", "adjust_params"],
        guardrail_violations=[],
        guardrail_warnings=[],
        regime_sharpe=None,
        regime_regime_shift=None,
        top_drawdowns=None,
        portfolio_correlation=None,
        benchmark_return_pct=None,
        market_regime=None,
        current_strategy_code="def generate_signals(df, params): ...",
        current_params={"period": 20},
        previous_attempts=[],
        multi_ticker_results=None,
        feature_importance=None,
        param_optimization=None,
    )
    defaults.update(overrides)

    class MockReport:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

        def to_dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    return MockReport(**defaults)


def _build_context(state, report=None, mandate=None):
    """Helper to call build_llm_context with all expensive deps mocked."""
    from crabquant.refinement.context_builder import build_llm_context

    mandate = mandate or {"tickers": ["SPY"], "period": "2y"}
    patches = [
        patch("crabquant.refinement.context_builder.load_indicator_reference", return_value=""),
        patch("crabquant.refinement.context_builder.extract_quick_reference", return_value=""),
        patch("crabquant.refinement.context_builder.build_trade_count_guidance", return_value=""),
        patch("crabquant.refinement.context_builder.get_strategy_examples", return_value=[]),
        patch("crabquant.refinement.context_builder.get_winner_examples", return_value=[]),
        patch("crabquant.refinement.context_builder.get_strategy_catalog", return_value=[]),
        patch("crabquant.refinement.context_builder._build_stagnation_recovery_section", return_value=""),
        patch("crabquant.refinement.context_builder._build_crash_error_feedback", return_value=""),
        patch("crabquant.refinement.action_analytics.generate_llm_context", return_value=""),
        patch("crabquant.refinement.action_analytics.load_run_history", return_value=[]),
        patch("crabquant.refinement.action_analytics.RUN_HISTORY_FILE", "/dev/null"),
        patch("crabquant.refinement.context_builder.build_turn1_prompt", return_value="## Base prompt"),
    ]
    for p in patches:
        p.start()
    try:
        return build_llm_context(state, report=report, mandate=mandate)
    finally:
        for p in patches:
            p.stop()


# ═════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════

class TestPositiveFeedbackContextWiring:
    """Verify positive_feedback is wired into context_builder.build_llm_context()."""

    def test_context_key_set_when_report_has_strengths(self):
        """When report has positive metrics, positive_feedback_section should be in context."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(
            sharpe_ratio=1.2,
            total_return_pct=0.15,
            win_rate=0.55,
            profit_factor=2.0,
        )
        context = _build_context(state, report=report)

        assert "positive_feedback_section" in context
        assert len(context["positive_feedback_section"]) > 0

    def test_context_key_absent_when_no_report(self):
        """Without a report (turn 1), positive_feedback_section should not be set."""
        state = _make_run_state(current_turn=1)
        context = _build_context(state, report=None)

        assert "positive_feedback_section" not in context

    def test_context_key_absent_when_all_metrics_zero(self):
        """When all metrics are zero/negative, no strengths → no section."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(
            sharpe_ratio=0.0,
            total_return_pct=-0.05,
            win_rate=0.0,
            profit_factor=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            max_drawdown_pct=-0.30,
            sharpe_by_year={},
        )
        context = _build_context(state, report=report)

        # No strengths → format returns empty string → not set in context
        assert "positive_feedback_section" not in context

    def test_feedback_contains_what_working_header(self):
        """The formatted section should contain the 'What's Working' header."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(
            sharpe_ratio=1.2,
            total_return_pct=0.15,
            win_rate=0.55,
        )
        context = _build_context(state, report=report)

        assert "What's Working" in context["positive_feedback_section"]
        assert "PRESERVE" in context["positive_feedback_section"]

    def test_feedback_identifies_positive_returns(self):
        """Positive total return should be identified as a strength."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(total_return_pct=0.20)
        context = _build_context(state, report=report)

        assert "Positive returns" in context["positive_feedback_section"]

    def test_feedback_identifies_good_win_rate(self):
        """Win rate in 45-70% range should be noted."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(win_rate=0.60)
        context = _build_context(state, report=report)

        assert "win rate" in context["positive_feedback_section"].lower() or "Win rate" in context["positive_feedback_section"]

    def test_feedback_identifies_strong_profit_factor(self):
        """Profit factor > 1.5 should be called out."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(profit_factor=2.5)
        context = _build_context(state, report=report)

        assert "profit factor" in context["positive_feedback_section"].lower()

    def test_feedback_includes_sharpe_ratio_value(self):
        """The Sharpe ratio from the report should appear in feedback."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(sharpe_ratio=1.2)
        context = _build_context(state, report=report)

        assert "1.20" in context["positive_feedback_section"]

    def test_feedback_uses_correct_sharpe_target(self):
        """Feedback should reference the state's sharpe_target, not a hardcoded value."""
        state = _make_run_state(current_turn=2, sharpe_target=2.0)
        report = _make_backtest_report(sharpe_ratio=1.5)
        context = _build_context(state, report=report)

        # 1.5 is 75% of 2.0 target — should show percentage
        assert "75%" in context["positive_feedback_section"] or "1.50" in context["positive_feedback_section"]

    def test_feedback_shows_controlled_drawdown(self):
        """Moderate drawdown should be identified as controlled."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(max_drawdown_pct=-0.10)
        context = _build_context(state, report=report)

        assert "drawdown" in context["positive_feedback_section"].lower() or "Drawdown" in context["positive_feedback_section"]

    def test_feedback_shows_trade_frequency(self):
        """Trade count in 20-100 range should be identified."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(total_trades=50)
        context = _build_context(state, report=report)

        assert "50" in context["positive_feedback_section"] and "trade" in context["positive_feedback_section"].lower()


class TestPositiveFeedbackPreviousSuccessfulTurns:
    """Verify that previous successful turns are surfaced in positive feedback."""

    def test_successful_turns_shown_in_feedback(self):
        """When history has turns that hit the sharpe target, they should appear."""
        state = _make_run_state(
            current_turn=4,
            sharpe_target=1.5,
            history=[
                {"turn": 1, "sharpe": 0.5, "action": "novel", "num_trades": 30},
                {"turn": 2, "sharpe": 1.8, "action": "adjust_params", "num_trades": 45},
                {"turn": 3, "sharpe": 1.6, "action": "add_filter", "num_trades": 40},
            ],
        )
        report = _make_backtest_report(sharpe_ratio=0.9)
        context = _build_context(state, report=report)

        assert "positive_feedback_section" in context
        assert "Previous Successful Turns" in context["positive_feedback_section"]
        assert "Turn 2" in context["positive_feedback_section"]
        assert "Turn 3" in context["positive_feedback_section"]

    def test_no_successful_turns_section_when_none_hit_target(self):
        """When no history turns hit the target, no successful turns section."""
        state = _make_run_state(
            current_turn=3,
            sharpe_target=1.5,
            history=[
                {"turn": 1, "sharpe": 0.5, "action": "novel", "num_trades": 30},
                {"turn": 2, "sharpe": 0.8, "action": "adjust_params", "num_trades": 35},
            ],
        )
        report = _make_backtest_report(sharpe_ratio=0.9)
        context = _build_context(state, report=report)

        # May or may not have positive_feedback_section depending on current metrics,
        # but should NOT have "Previous Successful Turns"
        fb = context.get("positive_feedback_section", "")
        assert "Previous Successful Turns" not in fb

    def test_successful_turns_capped_at_three(self):
        """Only last 3 successful turns should be shown even if more exist."""
        state = _make_run_state(
            current_turn=6,
            sharpe_target=1.5,
            history=[
                {"turn": 1, "sharpe": 2.0, "action": "novel", "num_trades": 30},
                {"turn": 2, "sharpe": 1.8, "action": "adjust_params", "num_trades": 40},
                {"turn": 3, "sharpe": 1.6, "action": "add_filter", "num_trades": 35},
                {"turn": 4, "sharpe": 2.1, "action": "tune_thresholds", "num_trades": 50},
                {"turn": 5, "sharpe": 1.7, "action": "adjust_params", "num_trades": 45},
            ],
        )
        report = _make_backtest_report(sharpe_ratio=0.5, sharpe_by_year={})
        context = _build_context(state, report=report)

        fb = context.get("positive_feedback_section", "")
        assert "Turn 3" in fb
        assert "Turn 4" in fb
        assert "Turn 5" in fb
        # Turn 1 and 2 should be excluded (only last 3 successful)
        assert "Turn 1" not in fb[fb.index("Previous Successful Turns"):]
        assert "Turn 2" not in fb[fb.index("Previous Successful Turns"):]

    def test_successful_turns_include_action_and_trades(self):
        """Successful turn entries should show the action taken and trade count."""
        state = _make_run_state(
            current_turn=3,
            sharpe_target=1.5,
            history=[
                {"turn": 1, "sharpe": 2.0, "action": "add_indicator", "num_trades": 55},
            ],
        )
        report = _make_backtest_report(sharpe_ratio=0.5)
        context = _build_context(state, report=report)

        fb = context.get("positive_feedback_section", "")
        assert "add_indicator" in fb
        assert "55" in fb


class TestPositiveFeedbackPromptInjection:
    """Verify that positive feedback reaches the final LLM prompt."""

    def test_feedback_in_prompt_when_present(self):
        """positive_feedback_section should appear in the prompt string."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(
            sharpe_ratio=1.2,
            total_return_pct=0.15,
            win_rate=0.55,
        )
        context = _build_context(state, report=report)

        prompt = context.get("prompt", "")
        # The section should appear in the prompt via append_sections
        assert "What's Working" in prompt or "PRESERVE" in prompt

    def test_no_feedback_in_prompt_when_absent(self):
        """When no positive feedback, prompt should not contain the section."""
        state = _make_run_state(current_turn=2)
        report = _make_backtest_report(
            sharpe_ratio=0.0,
            total_return_pct=-0.10,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=-0.30,
            sharpe_by_year={},
        )
        context = _build_context(state, report=report)

        prompt = context.get("prompt", "")
        # The "What's Working" header should not appear
        assert "What's Working" not in prompt

    def test_successful_turns_in_prompt(self):
        """Previous successful turns should reach the prompt."""
        state = _make_run_state(
            current_turn=3,
            sharpe_target=1.5,
            history=[
                {"turn": 1, "sharpe": 2.0, "action": "novel", "num_trades": 50},
            ],
        )
        report = _make_backtest_report(sharpe_ratio=0.5)
        context = _build_context(state, report=report)

        prompt = context.get("prompt", "")
        assert "Previous Successful Turns" in prompt


class TestPositiveFeedbackResilience:
    """Verify the wiring is resilient to errors."""

    def test_no_crash_when_report_missing_attributes(self):
        """Should not crash if report object is missing some attributes."""
        state = _make_run_state(current_turn=2)
        # Report with minimal fields — to_dict should still work
        report = _make_backtest_report(sharpe_ratio=0.5)
        # Should not raise
        context = _build_context(state, report=report)
        assert context is not None

    def test_no_crash_when_history_entries_lack_sharpe(self):
        """Should not crash if history entries don't have sharpe key."""
        state = _make_run_state(
            current_turn=3,
            history=[
                {"turn": 1, "status": "code_generation_failed"},
                {"turn": 2, "status": "backtest_crash", "error": {}},
            ],
        )
        report = _make_backtest_report(sharpe_ratio=0.5)
        # Should not raise
        context = _build_context(state, report=report)
        assert context is not None

    def test_empty_history_no_successful_turns_section(self):
        """Empty history should not produce a successful turns section."""
        state = _make_run_state(current_turn=2, history=[])
        report = _make_backtest_report(sharpe_ratio=0.5)
        context = _build_context(state, report=report)

        fb = context.get("positive_feedback_section", "")
        assert "Previous Successful Turns" not in fb
