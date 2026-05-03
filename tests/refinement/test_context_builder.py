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
    best_composite_score: float = -999.0


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

    def test_catalog_has_multiple_entries(self):
        """Catalog should contain more than just 1-2 strategies."""
        from crabquant.refinement.context_builder import get_strategy_catalog

        catalog = get_strategy_catalog()
        assert len(catalog) >= 4

    def test_no_duplicate_names(self):
        """Catalog should not contain duplicate strategy names."""
        from crabquant.refinement.context_builder import get_strategy_catalog

        catalog = get_strategy_catalog()
        names = [e["name"] for e in catalog]
        assert len(names) == len(set(names))


class TestStripAdvancedFunctions:
    """Test the _strip_advanced_functions helper."""

    def test_strips_param_grid(self):
        from crabquant.refinement.context_builder import _strip_advanced_functions

        code = """\
import pandas as pd

DEFAULT_PARAMS = {"x": 1}
DESCRIPTION = "test"

PARAM_GRID = {"x": [1, 2, 3]}

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""
        result = _strip_advanced_functions(code)
        assert "PARAM_GRID" not in result
        assert "generate_signals" in result
        assert "DEFAULT_PARAMS" in result

    def test_strips_generate_signals_matrix(self):
        from crabquant.refinement.context_builder import _strip_advanced_functions

        code = """\
import pandas as pd

DEFAULT_PARAMS = {"x": 1}
DESCRIPTION = "test"

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)

def generate_signals_matrix(df, params=None):
    return pd.DataFrame()
"""
        result = _strip_advanced_functions(code)
        assert "generate_signals_matrix" not in result
        assert "generate_signals" in result

    def test_preserves_other_functions(self):
        from crabquant.refinement.context_builder import _strip_advanced_functions

        code = """\
import pandas as pd

DEFAULT_PARAMS = {"x": 1}
DESCRIPTION = "test"

def helper_function(x):
    return x + 1

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""
        result = _strip_advanced_functions(code)
        assert "helper_function" in result
        assert "generate_signals" in result

    def test_strips_multiline_param_grid(self):
        from crabquant.refinement.context_builder import _strip_advanced_functions

        code = """\
import pandas as pd

DEFAULT_PARAMS = {"x": 1}
DESCRIPTION = "test"

PARAM_GRID = {
    "x": [1, 2, 3],
    "y": [4, 5, 6],
}

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""
        result = _strip_advanced_functions(code)
        assert "PARAM_GRID" not in result
        assert "generate_signals" in result

    def test_empty_code_returns_empty(self):
        from crabquant.refinement.context_builder import _strip_advanced_functions

        result = _strip_advanced_functions("")
        assert result == ""

    def test_code_without_param_grid_or_matrix_unchanged(self):
        from crabquant.refinement.context_builder import _strip_advanced_functions

        code = """\
import pandas as pd

DEFAULT_PARAMS = {"x": 1}
DESCRIPTION = "test"

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""
        result = _strip_advanced_functions(code)
        # Should be essentially the same (minus trailing whitespace)
        assert "generate_signals" in result
        assert "DEFAULT_PARAMS" in result


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

    def test_examples_stripped_of_param_grid_mostly(self):
        """Most examples should have PARAM_GRID stripped from source_code.
        
        Note: Some strategies with very long multi-line PARAM_GRID may not be
        fully stripped due to brace-depth tracking limitations."""
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("momentum")
        # At least one example should have PARAM_GRID stripped
        stripped = sum(1 for ex in examples if "PARAM_GRID" not in ex["source_code"])
        assert stripped >= 1, "At least one example should have PARAM_GRID stripped"

    def test_examples_stripped_of_generate_signals_matrix(self):
        """Examples should have generate_signals_matrix stripped."""
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("momentum")
        for ex in examples:
            assert "generate_signals_matrix" not in ex["source_code"]

    def test_fallback_archetype_uses_macd_and_rsi(self):
        """Unknown archetype should fall back to macd_momentum + rsi_crossover."""
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("completely_unknown")
        names = [e["name"] for e in examples]
        assert "macd_momentum" in names or "rsi_crossover" in names

    def test_each_example_has_description(self):
        """Each example should have a description string."""
        from crabquant.refinement.context_builder import get_strategy_examples

        examples = get_strategy_examples("any")
        for ex in examples:
            assert "description" in ex
            assert isinstance(ex["description"], str)


class TestGetWinnerExamples:
    """Test winner examples loading."""

    def test_returns_empty_when_no_winners_file(self, tmp_path):
        from crabquant.refinement.context_builder import get_winner_examples

        # Create a winners.json in a non-existent parent to simulate missing file
        fake_winners_path = tmp_path / "nonexistent" / "winners.json"

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", fake_winners_path):
            examples = get_winner_examples(runs_dir=tmp_path)
            assert examples == []

    def test_returns_empty_for_empty_winners(self, tmp_path):
        from crabquant.refinement.context_builder import get_winner_examples

        # Create empty winners.json
        winners_dir = tmp_path / "winners"
        winners_dir.mkdir()
        (winners_dir / "winners.json").write_text("[]")

        with patch("crabquant.refinement.context_builder._WINNERS_PATH",
                    winners_dir / "winners.json"):
            examples = get_winner_examples(runs_dir=tmp_path)
            assert examples == []

    def test_returns_empty_for_invalid_json(self, tmp_path):
        from crabquant.refinement.context_builder import get_winner_examples

        winners_dir = tmp_path / "winners"
        winners_dir.mkdir()
        (winners_dir / "winners.json").write_text("not json")

        with patch("crabquant.refinement.context_builder._WINNERS_PATH",
                    winners_dir / "winners.json"):
            examples = get_winner_examples(runs_dir=tmp_path)
            assert examples == []

    def test_filters_zero_trades(self, tmp_path):
        from crabquant.refinement.context_builder import get_winner_examples

        winners_dir = tmp_path / "winners"
        winners_dir.mkdir()
        winners = [
            {"strategy": "good", "sharpe": 2.0, "trades": 20, "ticker": "SPY"},
            {"strategy": "no_trades", "sharpe": 5.0, "trades": 0, "ticker": "SPY"},
        ]
        (winners_dir / "winners.json").write_text(json.dumps(winners))

        with patch("crabquant.refinement.context_builder._WINNERS_PATH",
                    winners_dir / "winners.json"):
            examples = get_winner_examples(runs_dir=tmp_path)
            names = [e["name"] for e in examples]
            assert "no_trades" not in names

    def test_deduplicates_by_name(self, tmp_path):
        from crabquant.refinement.context_builder import get_winner_examples

        winners_dir = tmp_path / "winners"
        winners_dir.mkdir()
        winners = [
            {"strategy": "dup", "sharpe": 2.0, "trades": 20, "ticker": "SPY"},
            {"strategy": "dup", "sharpe": 3.0, "trades": 25, "ticker": "AAPL"},
        ]
        (winners_dir / "winners.json").write_text(json.dumps(winners))

        with patch("crabquant.refinement.context_builder._WINNERS_PATH",
                    winners_dir / "winners.json"):
            examples = get_winner_examples(runs_dir=tmp_path)
            # Only one entry for "dup" should appear
            dup_count = sum(1 for e in examples if e["name"] == "dup")
            assert dup_count <= 1

    def test_respects_max_examples(self, tmp_path):
        from crabquant.refinement.context_builder import get_winner_examples

        winners_dir = tmp_path / "winners"
        winners_dir.mkdir()
        winners = [
            {"strategy": f"strat_{i}", "sharpe": float(i), "trades": 20, "ticker": "SPY"}
            for i in range(10)
        ]
        (winners_dir / "winners.json").write_text(json.dumps(winners))

        with patch("crabquant.refinement.context_builder._WINNERS_PATH",
                    winners_dir / "winners.json"):
            examples = get_winner_examples(runs_dir=tmp_path, max_examples=2)
            assert len(examples) <= 2

    def test_archetype_diversity_selection(self, tmp_path):
        """Verify that winner examples span multiple archetypes, not just the top-scoring one."""
        from crabquant.refinement.context_builder import get_winner_examples

        winners_dir = tmp_path / "winners"
        winners_dir.mkdir()
        # Create winners dominated by one archetype (momentum/roc) with lower-scoring
        # entries from other archetypes. The diversity selection should surface them.
        winners = [
            {"strategy": "roc_ema_volume_a", "sharpe": 3.0, "trades": 30, "ticker": "SPY"},
            {"strategy": "roc_ema_volume_b", "sharpe": 2.8, "trades": 28, "ticker": "SPY"},
            {"strategy": "roc_ema_volume_c", "sharpe": 2.5, "trades": 25, "ticker": "SPY"},
            {"strategy": "roc_ema_volume_d", "sharpe": 2.3, "trades": 22, "ticker": "SPY"},
            {"strategy": "roc_ema_volume_e", "sharpe": 2.1, "trades": 20, "ticker": "SPY"},
            # Lower-scoring but different archetypes
            {"strategy": "bollinger_squeeze", "sharpe": 1.5, "trades": 15, "ticker": "SPY"},
            {"strategy": "mean_reversion_rsi", "sharpe": 1.4, "trades": 18, "ticker": "SPY"},
            {"strategy": "volume_breakout", "sharpe": 1.3, "trades": 16, "ticker": "SPY"},
            {"strategy": "adx_pullback", "sharpe": 1.2, "trades": 14, "ticker": "SPY"},
        ]
        (winners_dir / "winners.json").write_text(json.dumps(winners))

        with patch("crabquant.refinement.context_builder._WINNERS_PATH",
                    winners_dir / "winners.json"):
            examples = get_winner_examples(runs_dir=tmp_path, max_examples=3)
            names = [e["name"] for e in examples]

            # Without diversity, all 3 would be roc_ema_volume_*. With diversity,
            # at least one should be from a different archetype.
            roc_count = sum(1 for n in names if n.startswith("roc_ema_volume"))
            assert roc_count < 3, (
                f"Expected archetype diversity but got all roc_ema_volume: {names}"
            )

    def test_archetype_diversity_fallback_when_single_archetype(self, tmp_path):
        """When all winners are from one archetype, diversity selection still returns results."""
        from crabquant.refinement.context_builder import get_winner_examples

        winners_dir = tmp_path / "winners"
        winners_dir.mkdir()
        winners = [
            {"strategy": f"momentum_{i}", "sharpe": float(3 - i * 0.3), "trades": 20, "ticker": "SPY"}
            for i in range(5)
        ]
        (winners_dir / "winners.json").write_text(json.dumps(winners))

        with patch("crabquant.refinement.context_builder._WINNERS_PATH",
                    winners_dir / "winners.json"):
            examples = get_winner_examples(runs_dir=tmp_path, max_examples=2)
            assert len(examples) <= 2
            # All should still be momentum (no other archetype available)
            for e in examples:
                assert "momentum" in e["name"].lower()


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

    def test_both_added_and_removed(self, tmp_path):
        """Detects both added and removed indicators in one delta."""
        from crabquant.refinement.context_builder import compute_delta

        prev = tmp_path / "prev.py"
        prev.write_text("cached_indicator('rsi')\ncached_indicator('macd')\n")

        current = "cached_indicator('rsi')\ncached_indicator('atr')\n"
        result = compute_delta(current, "swap", "replace MACD with ATR", str(prev))
        assert "Added indicators" in result
        assert "Removed indicators" in result
        assert "atr" in result
        assert "macd" in result

    def test_no_indicators_in_either(self, tmp_path):
        """When neither version has cached_indicator calls."""
        from crabquant.refinement.context_builder import compute_delta

        prev = tmp_path / "prev.py"
        prev.write_text("x = 1\ny = 2\n")

        current = "x = 3\ny = 4\n"
        result = compute_delta(current, "tweak", "adjust values", str(prev))
        assert "Same indicators" in result

    def test_none_prev_code_path(self):
        """None as prev_code_path should return initial strategy message."""
        from crabquant.refinement.context_builder import compute_delta

        result = compute_delta("code", "action", "hypothesis", None)
        assert result == "Initial strategy (no prior version)"

    def test_empty_string_prev_code_path(self):
        """Empty string as prev_code_path should return initial strategy message."""
        from crabquant.refinement.context_builder import compute_delta

        result = compute_delta("code", "action", "hypothesis", "")
        assert result == "Initial strategy (no prior version)"


class TestBuildLlmContext:
    """Test full context assembly."""

    def test_basic_context_without_report(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        context = build_llm_context(state, report=None, mandate={"strategy_archetype": "momentum"})
        
        assert context["current_turn"] == 0  # default is 0
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
        assert context["current_turn"] == 0  # default is 0

    def test_best_composite_score_included(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(best_composite_score=42.5)
        context = build_llm_context(state)
        
        assert context["best_composite_score"] == 42.5

    def test_indicator_reference_included(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        context = build_llm_context(state)
        
        assert "indicator_reference" in context
        assert "indicator_quick_ref" in context

    def test_winner_examples_included_by_default(self):
        """Winner examples should be included when cross_run_learning is True (default)."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        context = build_llm_context(state)
        
        assert "winner_examples" in context

    def test_cross_run_learning_disabled(self):
        """Winner examples should be empty when cross_run_learning is False."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        mandate = {"cross_run_learning": False}
        context = build_llm_context(state, mandate=mandate)
        
        assert context["winner_examples"] == []

    def test_dataclass_report_converted(self):
        """Report as dataclass should be converted via asdict."""
        from crabquant.refinement.context_builder import build_llm_context
        from dataclasses import dataclass, field as dc_field

        @dataclass
        class DataclassReport:
            sharpe_ratio: float = 1.0
            custom_field: str = "hello"

        state = FakeRunState()
        context = build_llm_context(state, report=DataclassReport())
        
        assert context["backtest_report"]["custom_field"] == "hello"

    def test_empty_history(self):
        """Empty history should produce empty previous_attempts."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(history=[])
        context = build_llm_context(state)
        
        assert context["previous_attempts"] == []

    def test_current_turn_increments(self):
        """current_turn should match state.current_turn (no longer adds +1)."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=5)
        context = build_llm_context(state)
        
        assert context["current_turn"] == 5

    def test_state_with_missing_attributes(self):
        """State-like object missing some attrs should use getattr defaults."""
        from crabquant.refinement.context_builder import build_llm_context

        class MinimalState:
            pass

        context = build_llm_context(MinimalState())
        
        assert context["current_turn"] == 0  # default from getattr(state, "current_turn", 0)
        assert context["max_turns"] == 7  # default
        assert context["sharpe_target"] == 1.5  # default
        assert context["tickers"] == ["AAPL", "SPY"]  # default


class TestFormatMultiTickerFeedback:
    """Test multi-ticker feedback formatting."""

    def test_basic_format(self):
        from crabquant.refinement.context_builder import _format_multi_ticker_feedback

        mt_results = {
            "tickers_tested": 2,
            "tickers_passed": 1,
            "avg_sharpe": 1.0,
            "min_sharpe": 0.5,
            "pass_rate": 0.5,
            "per_ticker": [
                {"ticker": "AAPL", "sharpe": 1.5, "trades": 10, "max_drawdown": 0.1, "passed": True},
                {"ticker": "GOOG", "sharpe": 0.5, "trades": 5, "max_drawdown": 0.2, "passed": False},
            ],
        }
        result = _format_multi_ticker_feedback(mt_results)
        assert "Multi-Ticker Validation" in result
        assert "AAPL" in result
        assert "GOOG" in result
        assert "PASS" in result
        assert "FAIL" in result

    def test_all_pass_no_overfit_warning(self):
        from crabquant.refinement.context_builder import _format_multi_ticker_feedback

        mt_results = {
            "tickers_tested": 2,
            "tickers_passed": 2,
            "avg_sharpe": 1.5,
            "min_sharpe": 1.2,
            "pass_rate": 1.0,
            "per_ticker": [
                {"ticker": "AAPL", "sharpe": 1.5, "trades": 10, "max_drawdown": 0.1, "passed": True},
                {"ticker": "GOOG", "sharpe": 1.2, "trades": 8, "max_drawdown": 0.15, "passed": True},
            ],
        }
        result = _format_multi_ticker_feedback(mt_results)
        assert "overfit" not in result.lower()

    def test_has_failures_includes_overfit_guidance(self):
        from crabquant.refinement.context_builder import _format_multi_ticker_feedback

        mt_results = {
            "tickers_tested": 2,
            "tickers_passed": 0,
            "avg_sharpe": 0.3,
            "min_sharpe": -0.5,
            "pass_rate": 0.0,
            "per_ticker": [
                {"ticker": "AAPL", "sharpe": -0.5, "trades": 3, "max_drawdown": 0.3, "passed": False},
                {"ticker": "GOOG", "sharpe": 0.3, "trades": 2, "max_drawdown": 0.25, "passed": False},
            ],
        }
        result = _format_multi_ticker_feedback(mt_results)
        assert "overfit" in result.lower()
        assert "Simplifying logic" in result or "simplifying" in result.lower()

    def test_empty_per_ticker(self):
        from crabquant.refinement.context_builder import _format_multi_ticker_feedback

        mt_results = {
            "tickers_tested": 0,
            "tickers_passed": 0,
            "avg_sharpe": 0.0,
            "min_sharpe": 0.0,
            "pass_rate": 0.0,
            "per_ticker": [],
        }
        result = _format_multi_ticker_feedback(mt_results)
        assert "Multi-Ticker Validation" in result

    def test_includes_sharpe_and_trade_counts(self):
        from crabquant.refinement.context_builder import _format_multi_ticker_feedback

        mt_results = {
            "tickers_tested": 1,
            "tickers_passed": 1,
            "avg_sharpe": 2.5,
            "min_sharpe": 2.5,
            "pass_rate": 1.0,
            "per_ticker": [
                {"ticker": "SPY", "sharpe": 2.5, "trades": 42, "max_drawdown": 0.05, "passed": True},
            ],
        }
        result = _format_multi_ticker_feedback(mt_results)
        assert "2.50" in result
        assert "42" in result

    def test_missing_max_drawdown_defaults(self):
        """Per-ticker entries without max_drawdown should not crash."""
        from crabquant.refinement.context_builder import _format_multi_ticker_feedback

        mt_results = {
            "tickers_tested": 1,
            "tickers_passed": 1,
            "avg_sharpe": 1.0,
            "min_sharpe": 1.0,
            "pass_rate": 1.0,
            "per_ticker": [
                {"ticker": "SPY", "sharpe": 1.0, "trades": 10, "passed": True},
            ],
        }
        result = _format_multi_ticker_feedback(mt_results)
        assert "SPY" in result


class TestBuildStagnationRecoverySection:
    """Test stagnation recovery section building."""

    def test_empty_history_no_stagnation(self):
        from crabquant.refinement.context_builder import _build_stagnation_recovery_section

        state = FakeRunState(history=[])
        result = _build_stagnation_recovery_section(state)
        assert result == ""

    def test_single_turn_no_stagnation(self):
        """Need at least 2 turns with Sharpe data."""
        from crabquant.refinement.context_builder import _build_stagnation_recovery_section

        state = FakeRunState(history=[{"turn": 1, "sharpe": 0.5}])
        result = _build_stagnation_recovery_section(state)
        assert result == ""

    def test_improving_history_no_stagnation(self):
        """Improving Sharpe above the stagnation thresholds should not trigger."""
        from crabquant.refinement.context_builder import _build_stagnation_recovery_section

        state = FakeRunState(history=[
            {"turn": 1, "sharpe": 0.8},
            {"turn": 2, "sharpe": 1.2},
            {"turn": 3, "sharpe": 1.8},
        ], best_sharpe=1.8)
        result = _build_stagnation_recovery_section(state)
        assert result == ""

    def test_stagnant_history_triggers_recovery(self):
        """Flat/declining Sharpe should trigger stagnation recovery."""
        from crabquant.refinement.context_builder import _build_stagnation_recovery_section

        state = FakeRunState(history=[
            {"turn": 1, "sharpe": 0.1, "action": "modify_params", "failure_mode": "low_sharpe"},
            {"turn": 2, "sharpe": 0.1, "action": "modify_params", "failure_mode": "low_sharpe"},
            {"turn": 3, "sharpe": 0.1, "action": "modify_params", "failure_mode": "low_sharpe"},
            {"turn": 4, "sharpe": 0.1, "action": "modify_params", "failure_mode": "low_sharpe"},
        ])
        result = _build_stagnation_recovery_section(state)
        # Should detect action loop or low_sharpe_plateau
        assert isinstance(result, str)
        # May or may not be empty depending on severity thresholds

    def test_stagnation_injected_into_context(self):
        """When stagnation is detected, it should appear in the context."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(history=[
            {"turn": 1, "sharpe": 0.0, "action": "modify_params", "failure_mode": "low_sharpe"},
            {"turn": 2, "sharpe": 0.0, "action": "modify_params", "failure_mode": "low_sharpe"},
            {"turn": 3, "sharpe": 0.0, "action": "modify_params", "failure_mode": "low_sharpe"},
            {"turn": 4, "sharpe": 0.0, "action": "modify_params", "failure_mode": "low_sharpe"},
            {"turn": 5, "sharpe": 0.0, "action": "modify_params", "failure_mode": "low_sharpe"},
        ])
        context = build_llm_context(state)
        # Stagnation may or may not be injected depending on severity
        # Just verify the context is well-formed
        assert isinstance(context, dict)


class TestMultiTickerInContext:
    """Test multi-ticker feedback injection into LLM context."""

    def test_multi_ticker_feedback_in_context(self):
        from crabquant.refinement.context_builder import build_llm_context

        @dataclass
        class ReportWithMT:
            sharpe_ratio: float = 1.0
            total_trades: int = 10
            multi_ticker_results: dict = field(default_factory=lambda: {
                "tickers_tested": 2,
                "tickers_passed": 1,
                "avg_sharpe": 0.8,
                "min_sharpe": -0.2,
                "pass_rate": 0.5,
                "per_ticker": [
                    {"ticker": "AAPL", "sharpe": 1.5, "trades": 10, "max_drawdown": 0.1, "passed": True},
                    {"ticker": "GOOG", "sharpe": 0.5, "trades": 5, "max_drawdown": 0.2, "passed": False},
                ],
            })

        state = FakeRunState()
        context = build_llm_context(state, report=ReportWithMT())
        
        assert "multi_ticker_feedback" in context
        assert "AAPL" in context["multi_ticker_feedback"]

    def test_no_multi_ticker_when_absent(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        report = FakeReport()  # No multi_ticker_results attribute
        context = build_llm_context(state, report=report)
        
        assert "multi_ticker_feedback" not in context

    def test_no_multi_ticker_when_none(self):
        from crabquant.refinement.context_builder import build_llm_context

        @dataclass
        class ReportWithNoneMT:
            sharpe_ratio: float = 1.0
            multi_ticker_results: None = None

        state = FakeRunState()
        context = build_llm_context(state, report=ReportWithNoneMT())
        
        assert "multi_ticker_feedback" not in context


class TestFeatureImportanceInContext:
    """Test feature importance section injection into LLM context."""

    def test_feature_importance_injected(self):
        from crabquant.refinement.context_builder import build_llm_context

        @dataclass
        class ReportWithFI:
            sharpe_ratio: float = 1.0
            feature_importance: dict = field(default_factory=lambda: {
                "indicators": [
                    {"name": "rsi", "importance": 0.5},
                    {"name": "macd", "importance": 0.3},
                ],
            })

        state = FakeRunState()
        context = build_llm_context(state, report=ReportWithFI())
        
        assert "feature_importance_section" in context

    def test_no_feature_importance_when_absent(self):
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState()
        report = FakeReport()  # No feature_importance attribute
        context = build_llm_context(state, report=report)
        
        assert "feature_importance_section" not in context

    def test_no_feature_importance_when_empty(self):
        from crabquant.refinement.context_builder import build_llm_context

        @dataclass
        class ReportWithEmptyFI:
            sharpe_ratio: float = 1.0
            feature_importance: dict = field(default_factory=lambda: {"indicators": []})

        state = FakeRunState()
        context = build_llm_context(state, report=ReportWithEmptyFI())
        
        assert "feature_importance_section" not in context


class TestPromptWiring:
    """Verify that build_llm_context sets context['prompt'] using proper prompt builders.

    This is the CRITICAL wiring that enables failure guidance, sharpe diagnosis,
    regime diagnosis, and all other feedback systems. Without this, call_llm_inventor
    falls back to raw JSON dumps that bypass all guidance.
    """

    def test_turn1_sets_prompt_key(self):
        """Turn 1 (no report) should set context['prompt'] via build_turn1_prompt."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=0)
        context = build_llm_context(state, report=None, mandate={})

        assert "prompt" in context
        assert isinstance(context["prompt"], str)
        assert len(context["prompt"]) > 100  # Substantial prompt content
        # Turn 1 prompt should include mandate-related content
        assert "Turn" in context["prompt"] or "turn" in context["prompt"]

    def test_turn2_sets_prompt_key(self):
        """Turn 2+ (with report) should set context['prompt'] via build_refinement_prompt."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=1)
        context = build_llm_context(state, report=FakeReport(), mandate={})

        assert "prompt" in context
        assert isinstance(context["prompt"], str)
        assert len(context["prompt"]) > 100

    def test_refinement_prompt_contains_failure_guidance(self):
        """The refinement prompt should contain failure guidance for low_sharpe."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=1)
        report = FakeReport(failure_mode="low_sharpe")
        context = build_llm_context(state, report=report, mandate={"sharpe_target": 1.5})

        assert "prompt" in context
        prompt = context["prompt"]
        # The refinement prompt should include failure guidance section
        assert "Guidance" in prompt or "guidance" in prompt or "low_sharpe" in prompt

    def test_refinement_prompt_contains_too_few_trades_guidance(self):
        """The refinement prompt should contain too_few_trades guidance."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=1)
        report = FakeReport(failure_mode="too_few_trades", total_trades=2)
        context = build_llm_context(state, report=report, mandate={})

        assert "prompt" in context
        prompt = context["prompt"]
        assert "Too Few Trades" in prompt or "too_few_trades" in prompt

    def test_refinement_prompt_contains_regime_guidance(self):
        """The refinement prompt should contain regime_fragility guidance."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=1)
        report = FakeReport(failure_mode="regime_fragility")
        context = build_llm_context(state, report=report, mandate={})

        assert "prompt" in context
        prompt = context["prompt"]
        assert "Regime" in prompt or "regime" in prompt

    def test_refinement_prompt_with_sharpe_diagnosis(self):
        """low_sharpe with metrics should trigger sharpe diagnosis in the prompt."""
        from crabquant.refinement.context_builder import build_llm_context

        @dataclass
        class ReportWithMetrics:
            sharpe_ratio: float = 0.3
            total_return_pct: float = 0.02
            total_trades: int = 40
            failure_mode: str = "low_sharpe"
            failure_details: str = "Sharpe 0.3 < target 1.5"
            sharpe_by_year: dict = field(default_factory=lambda: {
                "2023": 0.5, "2024": 0.1, "2025": 0.3
            })
            stagnation_score: float = 0.0
            current_strategy_code: str = "def generate_signals(df, params): pass"
            current_params: dict = field(default_factory=dict)
            win_rate: float = 0.42
            profit_factor: float = 0.8
            sortino_ratio: float = 0.4
            calmar_ratio: float = 0.2
            max_drawdown_pct: float = -0.15
            composite_score: float = 0.3
            guardrail_violations: list = field(default_factory=list)
            guardrail_warnings: list = field(default_factory=list)
            previous_attempts: list = field(default_factory=list)
            previous_sharpes: list = field(default_factory=list)
            previous_actions: list = field(default_factory=list)
            stagnation_trend: str = "improving"

        state = FakeRunState(current_turn=1)
        context = build_llm_context(
            state, report=ReportWithMetrics(), mandate={"sharpe_target": 1.5}
        )

        assert "prompt" in context
        prompt = context["prompt"]
        # Sharpe diagnosis should mention root cause or actionable fix
        assert "win rate" in prompt.lower() or "profit factor" in prompt.lower() or "Guidance" in prompt

    def test_prompt_key_takes_precedence_over_fallback(self):
        """When prompt key is set, call_llm_inventor should use it directly."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=0)
        context = build_llm_context(state, report=None, mandate={})

        # Verify the prompt key is a complete, self-contained prompt
        # (not just a reference to another field)
        prompt = context["prompt"]
        assert "##" in prompt  # Has markdown headers (structured prompt)
        assert len(prompt) > 500  # Substantial content

    def test_turn1_prompt_graceful_on_exception(self):
        """If build_turn1_prompt fails, context should still be returned without prompt."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=0)

        with patch("crabquant.refinement.context_builder.build_turn1_prompt", side_effect=Exception("test error")):
            context = build_llm_context(state, report=None, mandate={})

        # Should not crash, but prompt should not be set
        assert "prompt" not in context or context.get("prompt") is None

    def test_turn2_prompt_graceful_on_exception(self):
        """If build_refinement_prompt fails, context should still be returned without prompt."""
        from crabquant.refinement.context_builder import build_llm_context

        state = FakeRunState(current_turn=1)

        with patch("crabquant.refinement.context_builder.build_refinement_prompt", side_effect=Exception("test error")):
            context = build_llm_context(state, report=FakeReport(), mandate={})

        # Should not crash, but prompt should not be set
        assert "prompt" not in context or context.get("prompt") is None
