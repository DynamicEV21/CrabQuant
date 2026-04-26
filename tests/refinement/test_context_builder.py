"""Tests for crabquant.refinement.context_builder"""

import json
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from pathlib import Path


# Minimal RunState-like object for testing
@dataclass
class FakeRunState:
    current_turn: int = 0
    max_turns: int = 7
    sharpe_target: float = 1.5
    tickers: list = field(default_factory=lambda: ["AAPL", "SPY"])
    history: list = field(default_factory=list)
    best_sharpe: float = 0.0
    best_turn: int = 0


@dataclass
class FakeReport:
    sharpe_ratio: float = 0.8
    total_return_pct: float = 0.05
    total_trades: int = 12
    failure_mode: str = "low_sharpe"
    stagnation_score: float = 0.0
    current_strategy_code: str = "def generate_signals(df, params): pass"
    current_params: dict = field(default_factory=lambda: {"rsi_period": 14})


class TestGetStrategyCatalog:
    """Test strategy catalog retrieval."""

    def test_returns_list_of_dicts(self):
        from crabquant.refinement.context_builder import get_strategy_catalog

        catalog = get_strategy_catalog()
        assert isinstance(catalog, list)
        assert len(catalog) > 0
        for entry in catalog:
            assert "name" in entry
            assert "description" in entry

    def test_contains_known_strategies(self):
        from crabquant.refinement.context_builder import get_strategy_catalog

        catalog = get_strategy_catalog()
        names = [e["name"] for e in catalog]
        # Should have at least some of the known strategies
        assert "rsi_crossover" in names
        assert "macd_momentum" in names

    def test_each_entry_has_string_description(self):
        from crabquant.refinement.context_builder import get_strategy_catalog

        catalog = get_strategy_catalog()
        for entry in catalog:
            assert isinstance(entry["name"], str)
            assert isinstance(entry["description"], str)


class TestGetStrategyExamples:
    """Test strategy example retrieval with full source code."""

    def test_returns_examples_for_momentum(self):
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("momentum")
        assert len(examples) >= 1
        for ex in examples:
            assert "name" in ex
            assert "source_code" in ex
            assert "default_params" in ex

    def test_returns_examples_for_unknown_archetype(self):
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("nonexistent_type")
        # Should fall back to defaults
        assert len(examples) >= 1

    def test_default_archetype(self):
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("any")
        assert len(examples) >= 1

    def test_examples_have_python_code(self):
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("momentum")
        for ex in examples:
            assert "def " in ex["source_code"]

    def test_examples_have_default_params_dict(self):
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("momentum")
        for ex in examples:
            assert isinstance(ex["default_params"], dict)

    def test_mean_reversion_archetype(self):
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("mean_reversion")
        names = [e["name"] for e in examples]
        assert "rsi_crossover" in names

    def test_breakout_archetype(self):
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("breakout")
        assert len(examples) >= 1

    def test_trend_archetype(self):
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("trend")
        assert len(examples) >= 1


class TestComputeDelta:
    """Test delta computation between strategy versions."""

    def test_initial_strategy_no_prev(self):
        from crabquant.refinement.context_builder import compute_delta

        result = compute_delta("some code", "full_rewrite", "testing")
        assert result == "Initial strategy (no prior version)"

    def test_detects_added_indicators(self, tmp_path):
        from crabquant.refinement.context_builder import compute_delta

        prev = tmp_path / "prev.py"
        prev.write_text("cached_indicator('rsi')\ncached_indicator('macd')\n")
        
        current = "cached_indicator('rsi')\ncached_indicator('macd')\ncached_indicator('atr')\n"
        result = compute_delta(current, "add_filter", "add ATR filter", str(prev))
        assert "Added indicators" in result
        assert "atr" in result

    def test_detects_removed_indicators(self, tmp_path):
        from crabquant.refinement.context_builder import compute_delta

        prev = tmp_path / "prev.py"
        prev.write_text("cached_indicator('rsi')\ncached_indicator('macd')\n")
        
        current = "cached_indicator('rsi')\n"
        result = compute_delta(current, "replace_indicator", "remove MACD", str(prev))
        assert "Removed indicators" in result
        assert "macd" in result

    def test_same_indicators_logic_changed(self, tmp_path):
        from crabquant.refinement.context_builder import compute_delta

        prev = tmp_path / "prev.py"
        prev.write_text("cached_indicator('rsi')\n")
        
        current = "cached_indicator('rsi')\n# changed logic\n"
        result = compute_delta(current, "modify_params", "tweak params", str(prev))
        assert "Same indicators" in result

    def test_handles_missing_prev_file(self):
        from crabquant.refinement.context_builder import compute_delta

        result = compute_delta("code", "modify_params", "test", "/nonexistent/path.py")
        assert "not found" in result or "Action" in result

    def test_includes_action_and_hypothesis(self, tmp_path):
        from crabquant.refinement.context_builder import compute_delta

        prev = tmp_path / "prev.py"
        prev.write_text("cached_indicator('rsi')\n")
        
        result = compute_delta("code", "full_rewrite", "complete overhaul", str(prev))
        assert "Action: full_rewrite" in result
        assert "Hypothesis: complete overhaul" in result


class TestBuildLlmContext:
    """Test full context assembly."""

    def test_basic_context_without_report(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        context = build_llm_context(state, report=None, mandate={"strategy_archetype": "momentum"})
        
        assert context["current_turn"] == 1
        assert context["max_turns"] == 7
        assert context["sharpe_target"] == 1.5
        assert context["tickers"] == ["AAPL", "SPY"]
        assert "strategy_examples" in context
        assert "strategy_catalog" in context

    def test_context_with_report(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=2, history=[{"turn": 1, "sharpe": 0.5}])
        report = FakeReport()
        context = build_llm_context(state, report=report, mandate={})
        
        assert "backtest_report" in context
        assert context["current_strategy_code"] == "def generate_signals(df, params): pass"
        assert context["current_params"] == {"rsi_period": 14}

    def test_previous_attempts_limited_to_3(self):
        from crabquant.refinement.context_builder import build_llm_context

        history = [{"turn": i, "sharpe": 0.5 + i * 0.1} for i in range(10)]
        state = FakeRunState(history=history)
        context = build_llm_context(state)
        
        assert len(context["previous_attempts"]) == 3
        assert context["previous_attempts"][0]["turn"] == 7

    def test_mandate_passed_through(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        mandate = {"name": "test_mandate", "strategy_archetype": "breakout"}
        context = build_llm_context(state, mandate=mandate)
        
        assert context["mandate"]["name"] == "test_mandate"

    def test_best_sharpe_included(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(best_sharpe=1.8, best_turn=3)
        context = build_llm_context(state)
        
        assert context["best_sharpe_so_far"] == 1.8
        assert context["best_turn"] == 3

    def test_no_report_means_no_current_code(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        context = build_llm_context(state, report=None)
        
        assert "current_strategy_code" not in context
        assert "current_params" not in context

    def test_report_with_to_dict_method(self):
        from crabquant.refinement.context_builder import build_llm_context

        class CustomReport:
            def to_dict(self):
                return {"sharpe_ratio": 1.2, "custom": True}
            current_strategy_code = "code"
            current_params = {"a": 1}

        state = FakeRunState()
        context = build_llm_context(state, report=CustomReport())
        
        assert context["backtest_report"]["custom"] is True

    def test_empty_mandate_uses_defaults(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        context = build_llm_context(state)
        
        assert context["mandate"] == {}
        assert context["current_turn"] == 1
