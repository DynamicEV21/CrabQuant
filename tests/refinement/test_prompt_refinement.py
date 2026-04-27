"""Tests for crabquant.refinement.prompts — Refined LLM prompts."""

import pytest
from unittest.mock import MagicMock

from crabquant.refinement.prompts import (
    SYSTEM_PROMPT,
    TURN1_PROMPT,
    REFINEMENT_PROMPT,
    build_turn1_prompt,
    build_refinement_prompt,
    VALID_ACTIONS,
    format_stagnation_suffix,
    format_tier2_section,
    format_previous_attempts_section,
)


class TestConstants:
    """Verify prompt constants exist and are well-formed."""

    def test_system_prompt_contains_required_sections(self):
        assert "hypothesis" in SYSTEM_PROMPT.lower()
        assert "generate_signals" in SYSTEM_PROMPT
        assert "DEFAULT_PARAMS" in SYSTEM_PROMPT
        assert "PARAM_GRID" in SYSTEM_PROMPT
        assert "DESCRIPTION" in SYSTEM_PROMPT
        assert "pandas_ta" in SYSTEM_PROMPT
        assert "pd.Series[bool]" in SYSTEM_PROMPT

    def test_valid_actions_includes_all_eight(self):
        expected = {
            "replace_indicator", "add_filter", "modify_params",
            "change_entry_logic", "change_exit_logic",
            "add_regime_filter", "full_rewrite", "novel",
        }
        assert expected.issubset(set(VALID_ACTIONS))

    def test_turn1_prompt_has_placeholders(self):
        assert "{mandate_name}" in TURN1_PROMPT
        assert "{sharpe_target}" in TURN1_PROMPT
        assert "{tickers}" in TURN1_PROMPT

    def test_refinement_prompt_has_placeholders(self):
        assert "{sharpe_ratio" in REFINEMENT_PROMPT
        assert "{failure_mode}" in REFINEMENT_PROMPT
        assert "{current_strategy_code}" in REFINEMENT_PROMPT
        assert "{sharpe_by_year}" in REFINEMENT_PROMPT


class TestFormatStagnationSuffix:
    def test_normal_constraint(self):
        result = format_stagnation_suffix("normal", "")
        assert result == ""

    def test_pivot_constraint(self):
        result = format_stagnation_suffix("pivot", "PIVOT: try something different")
        assert "PIVOT" in result

    def test_nuclear_constraint(self):
        result = format_stagnation_suffix("nuclear", "NUCLEAR REWRITE: ...")
        assert "NUCLEAR" in result

    def test_abandon_constraint(self):
        result = format_stagnation_suffix("abandon", "ABANDON: ...")
        assert "ABANDON" in result

    def test_none_inputs(self):
        result = format_stagnation_suffix(None, None)
        assert result == ""


class TestFormatTier2Section:
    def test_empty_tier2(self):
        report = {"regime_sharpe": None, "top_drawdowns": None, "benchmark_return_pct": None}
        result = format_tier2_section(report)
        assert result == ""

    def test_with_regime_sharpe(self):
        report = {
            "regime_sharpe": {"uptrend": 2.1, "downtrend": -0.5},
            "top_drawdowns": None,
            "benchmark_return_pct": None,
        }
        result = format_tier2_section(report)
        assert "uptrend" in result
        assert "2.1" in result

    def test_with_drawdowns(self):
        report = {
            "regime_sharpe": None,
            "top_drawdowns": [{"depth_pct": -0.15, "duration_bars": 20}],
            "benchmark_return_pct": None,
        }
        result = format_tier2_section(report)
        assert "15.0%" in result

    def test_with_benchmark(self):
        report = {
            "regime_sharpe": None,
            "top_drawdowns": None,
            "benchmark_return_pct": 0.22,
        }
        result = format_tier2_section(report)
        assert "22.0%" in result

    def test_with_all_tier2(self):
        report = {
            "regime_sharpe": {"sideways": 0.3},
            "top_drawdowns": [{"depth_pct": -0.10}],
            "benchmark_return_pct": 0.15,
        }
        result = format_tier2_section(report)
        assert "Regime" in result
        assert "Drawdown" in result
        assert "Benchmark" in result


class TestFormatPreviousAttemptsSection:
    def test_empty(self):
        # Empty list returns the fallback message (for prompt context)
        result = format_previous_attempts_section([])
        assert "first refinement" in result

    def test_single_attempt(self):
        attempts = [{
            "turn": 1, "sharpe": 0.5, "failure_mode": "too_few_trades",
            "action": "add_filter", "hypothesis": "h1",
            "params_used": {"p": 1}, "delta_from_prev": "Initial",
        }]
        result = format_previous_attempts_section(attempts)
        assert "Turn 1" in result
        assert "0.50" in result
        assert "add_filter" in result

    def test_multiple_attempts(self):
        attempts = [
            {"turn": 1, "sharpe": 0.3, "action": "novel", "hypothesis": "h1",
             "params_used": {"a": 1}, "delta_from_prev": "Initial"},
            {"turn": 2, "sharpe": 0.9, "action": "modify_params", "hypothesis": "h2",
             "params_used": {"a": 2}, "delta_from_prev": "Changed a from 1 to 2"},
        ]
        result = format_previous_attempts_section(attempts)
        assert "Turn 1" in result
        assert "Turn 2" in result


class TestBuildTurn1Prompt:
    def test_basic_mandate(self):
        mandate = {
            "name": "momentum_spy",
            "strategy_archetype": "momentum",
            "sharpe_target": 1.5,
            "tickers": ["SPY", "AAPL"],
            "period": "2y",
        }
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=7)
        assert "momentum_spy" in prompt
        assert "1.5" in prompt
        assert "SPY" in prompt
        assert "momentum" in prompt
        assert "new_strategy" in prompt

    def test_with_seed_strategy(self):
        mandate = {
            "name": "test",
            "strategy_archetype": "momentum",
            "sharpe_target": 1.5,
            "tickers": ["AAPL"],
            "period": "2y",
        }
        seed_code = "def generate_signals(df, params): pass"
        seed_params = {"period": 14}
        prompt = build_turn1_prompt(
            mandate=mandate, current_turn=1, max_turns=7,
            seed_strategy_name="macd_momentum",
            seed_code=seed_code,
            seed_params=seed_params,
        )
        assert "macd_momentum" in prompt
        assert "generate_signals" in prompt

    def test_with_strategy_examples(self):
        mandate = {"name": "test", "tickers": ["AAPL"], "period": "1y"}
        examples = [{"name": "rsi_crossover", "source_code": "pass", "default_params": {}, "description": "desc"}]
        prompt = build_turn1_prompt(
            mandate=mandate, current_turn=1, max_turns=7,
            strategy_examples=examples,
        )
        assert "rsi_crossover" in prompt

    def test_with_strategy_catalog(self):
        mandate = {"name": "test", "tickers": ["AAPL"], "period": "1y"}
        catalog = [{"name": "s1", "description": "d1"}, {"name": "s2", "description": "d2"}]
        prompt = build_turn1_prompt(
            mandate=mandate, current_turn=1, max_turns=7,
            strategy_catalog=catalog,
        )
        assert "s1" in prompt
        assert "s2" in prompt


class TestBuildRefinementPrompt:
    def _make_tier1_report(self, **overrides):
        defaults = {
            "sharpe_ratio": 1.2,
            "total_return_pct": 0.15,
            "max_drawdown_pct": -0.10,
            "win_rate": 0.55,
            "total_trades": 30,
            "profit_factor": 1.5,
            "calmar_ratio": 1.5,
            "sortino_ratio": 2.0,
            "composite_score": 1.2,
            "failure_mode": "low_sharpe",
            "failure_details": "Sharpe 1.2 < target 1.5",
            "sharpe_by_year": {"2023": 1.8, "2024": 0.3},
            "stagnation_score": 0.4,
            "stagnation_trend": "flat",
            "previous_sharpes": [0.45, 0.8],
            "previous_actions": ["add_filter", "modify_params"],
            "guardrail_violations": [],
            "guardrail_warnings": [],
            "current_strategy_code": "def generate_signals(df, params): pass",
            "current_params": {"period": 14},
            "previous_attempts": [],
            "consecutive_modify_params": 0,
        }
        defaults.update(overrides)
        return defaults

    def test_basic_refinement(self):
        report = self._make_tier1_report()
        prompt = build_refinement_prompt(
            tier1_report=report,
            current_turn=2,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=0.8,
            best_turn=1,
        )
        assert "1.2" in prompt  # current sharpe
        assert "1.5" in prompt  # target
        assert "low_sharpe" in prompt
        assert "generate_signals" in prompt  # current code included

    def test_with_stagnation(self):
        report = self._make_tier1_report()
        prompt = build_refinement_prompt(
            tier1_report=report,
            current_turn=5,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=0.8,
            best_turn=1,
            stagnation_suffix="PIVOT: Try something different",
        )
        assert "PIVOT" in prompt

    def test_with_previous_attempts(self):
        report = self._make_tier1_report(
            previous_attempts=[{
                "turn": 1, "sharpe": 0.5, "failure_mode": "too_few_trades",
                "action": "add_filter", "hypothesis": "h1",
                "params_used": {"p": 1}, "delta_from_prev": "Initial",
            }],
        )
        prompt = build_refinement_prompt(
            tier1_report=report,
            current_turn=2,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=0.5,
            best_turn=1,
        )
        assert "Turn 1" in prompt
        assert "add_filter" in prompt

    def test_with_tier2_data(self):
        report = self._make_tier1_report(
            regime_sharpe={"uptrend": 2.1, "downtrend": -0.5},
            top_drawdowns=[{"depth_pct": -0.15}],
            benchmark_return_pct=0.22,
        )
        prompt = build_refinement_prompt(
            tier1_report=report,
            current_turn=4,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=1.0,
            best_turn=2,
        )
        assert "Regime" in prompt or "uptrend" in prompt

    def test_with_strategy_examples(self):
        report = self._make_tier1_report()
        examples = [{"name": "rsi_crossover", "source_code": "pass", "default_params": {}, "description": "desc"}]
        prompt = build_refinement_prompt(
            tier1_report=report,
            current_turn=2,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=0.8,
            best_turn=1,
            strategy_examples=examples,
        )
        assert "rsi_crossover" in prompt

    def test_sharpe_by_year_in_prompt(self):
        report = self._make_tier1_report(
            sharpe_by_year={"2023": 2.5, "2024": 0.1}
        )
        prompt = build_refinement_prompt(
            tier1_report=report,
            current_turn=2,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=1.0,
            best_turn=1,
        )
        assert "2023" in prompt
        assert "2024" in prompt
