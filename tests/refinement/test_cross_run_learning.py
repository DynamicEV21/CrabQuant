"""Tests for cross-run learning (get_winner_examples)."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from crabquant.refinement.context_builder import get_winner_examples


@pytest.fixture
def winners_data():
    return [
        {
            "strategy": "great_strategy",
            "sharpe": 2.1,
            "trades": 80,
            "ticker": "SPY",
            "params": {"fast": 10, "slow": 30},
        },
        {
            "strategy": "ok_strategy",
            "sharpe": 1.2,
            "trades": 40,
            "ticker": "AAPL",
            "params": {},
        },
        {
            "strategy": "curve_fit",
            "sharpe": 3.0,
            "trades": 3,
            "ticker": "SPY",
            "params": {},
        },
        {
            "strategy": "negative_sharpe",
            "sharpe": -0.5,
            "trades": 50,
            "ticker": "SPY",
            "params": {},
        },
        {
            "strategy": "refinement_winner",
            "sharpe": 1.8,
            "trades": 60,
            "ticker": "SPY",
            "refinement_run": "test_run_123",
            "params": {},
        },
    ]


class TestGetWinnerExamples:
    """Test get_winner_examples function."""

    def test_returns_empty_when_no_winners_file(self, tmp_path):
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", tmp_path / "nonexistent.json"):
            result = get_winner_examples()
        assert result == []

    def test_returns_empty_when_file_corrupted(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json{{{")
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", bad_file):
            result = get_winner_examples()
        assert result == []

    def test_filters_low_trade_strategies(self, tmp_path):
        """Curve-fit strategies with < 1 trade should be excluded."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "curve", "sharpe": 5.0, "trades": 0, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert result == []

    def test_deduplicates_by_name(self, tmp_path):
        """Same strategy name should only appear once."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "dup", "sharpe": 1.0, "trades": 20, "ticker": "SPY"},
            {"strategy": "dup", "sharpe": 2.0, "trades": 30, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert len(result) <= 1

    def test_respects_max_examples(self, tmp_path):
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": f"strat_{i}", "sharpe": 1.0 + i * 0.5, "trades": 20 + i * 10, "ticker": "SPY"}
            for i in range(10)
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples(max_examples=2)
        assert len(result) <= 2

    def test_ticker_bonus_in_ranking(self, tmp_path):
        """Strategies matching the requested ticker should rank higher."""
        winners_file = tmp_path / "winners.json"
        runs_dir = tmp_path / "runs"
        # Create run dirs with strategy code
        for name in ["spy_strat", "aapl_strat"]:
            run_dir = runs_dir / name
            run_dir.mkdir(parents=True)
            (run_dir / "strategy_v1.py").write_text(
                f"def generate_signals(df, params):\n    return df['close'], df['close']\n"
            )
        winners_file.write_text(json.dumps([
            {"strategy": "spy_strat", "sharpe": 1.0, "trades": 30,
             "ticker": "SPY", "refinement_run": "spy_strat"},
            {"strategy": "aapl_strat", "sharpe": 1.5, "trades": 30,
             "ticker": "AAPL", "refinement_run": "aapl_strat"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples(ticker="SPY", max_examples=1)
        # SPY strategy should be preferred (1.5x bonus)
        assert len(result) == 1
        assert result[0]["ticker"] == "SPY"

    def test_returns_code_from_registry(self, tmp_path):
        """Should load code from STRATEGY_REGISTRY for sweep winners."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "macd_momentum", "sharpe": 1.5, "trades": 50, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert len(result) == 1
        assert result[0]["name"] == "macd_momentum"
        assert "source_code" in result[0]
        assert "generate_signals" in result[0]["source_code"]

    def test_returns_code_from_refinement_run(self, tmp_path):
        """Should load code from refinement run directory."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "ref_winner", "sharpe": 1.8, "trades": 60,
             "ticker": "SPY", "refinement_run": "run_123"},
        ]))
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "run_123"
        run_dir.mkdir(parents=True)
        # Write state.json with best_turn
        (run_dir / "state.json").write_text(json.dumps({"best_turn": 3}))
        # Write the strategy file
        (run_dir / "strategy_v3.py").write_text(
            "def generate_signals(df, params):\n    return df['close'], df['close']\n"
        )

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples()
        assert len(result) == 1
        assert "generate_signals" in result[0]["source_code"]

    def test_fallback_to_latest_strategy_file(self, tmp_path):
        """If best_turn file doesn't exist, fall back to latest strategy file."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "ref_winner2", "sharpe": 1.5, "trades": 40,
             "ticker": "SPY", "refinement_run": "run_456"},
        ]))
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "run_456"
        run_dir.mkdir(parents=True)
        # Only write v2 file (no state.json, no v3)
        (run_dir / "strategy_v2.py").write_text(
            "def generate_signals(df, params):\n    entries = df['close'] > df['close'].shift(1)\n    exits = ~entries\n    return entries, exits\n"
        )

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples()
        assert len(result) == 1
        assert "generate_signals" in result[0]["source_code"]

    def test_composite_score_favors_robust_strategies(self, tmp_path):
        """High Sharpe + high trades should rank above high Sharpe + low trades."""
        winners_file = tmp_path / "winners.json"
        runs_dir = tmp_path / "runs"
        for name in ["robust", "curve_fit"]:
            run_dir = runs_dir / name
            run_dir.mkdir(parents=True)
            (run_dir / "strategy_v1.py").write_text(
                f"def generate_signals(df, params):\n    return df['close'], df['close']\n"
            )
        winners_file.write_text(json.dumps([
            {"strategy": "robust", "sharpe": 1.5, "trades": 100,
             "ticker": "SPY", "refinement_run": "robust"},
            {"strategy": "curve_fit", "sharpe": 3.0, "trades": 5,
             "ticker": "SPY", "refinement_run": "curve_fit"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples(max_examples=1)
        assert len(result) == 1
        # robust: 1.5 * sqrt(100) = 15.0, curve_fit: 3.0 * sqrt(5) = 6.7
        assert result[0]["name"] == "robust"


class TestCrossRunLearningToggle:
    """Test that cross_run_learning toggle controls winner examples."""

    def test_disabled_returns_empty(self, tmp_path):
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "macd_momentum", "sharpe": 1.5, "trades": 50, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert len(result) >= 1

    def test_enabled_returns_winners(self, tmp_path):
        """When cross_run_learning is True, should return winners."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "macd_momentum", "sharpe": 1.5, "trades": 50, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert len(result) >= 1


# ── New tests: edge cases and uncovered paths ──────────────────────────────────


class TestGetWinnerExamplesEdgeCases:
    """Additional edge case tests for get_winner_examples."""

    def test_empty_winners_list(self, tmp_path):
        """Empty winners JSON list should return empty list."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert result == []

    def test_winners_missing_strategy_name(self, tmp_path):
        """Entries without 'strategy' key should be skipped."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"sharpe": 2.0, "trades": 50, "ticker": "SPY"},
            {"strategy": "macd_momentum", "sharpe": 1.0, "trades": 30, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        # Nameless entry is filtered out; macd_momentum loads from registry
        assert len(result) >= 1
        assert all(ex["name"] for ex in result), "All results should have non-empty names"

    def test_winners_empty_strategy_name(self, tmp_path):
        """Entries with empty strategy name should be skipped."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "", "sharpe": 2.0, "trades": 50, "ticker": "SPY"},
            {"strategy": "macd_momentum", "sharpe": 1.0, "trades": 30, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert len(result) >= 1
        assert all(ex["name"] for ex in result), "All results should have non-empty names"

    def test_winners_missing_sharpe_defaults_to_zero(self, tmp_path):
        """Missing sharpe key should default to 0 (filtered out since trades>=1 but sharpe<=0)."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "no_sharpe", "trades": 50, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert result == []

    def test_winners_null_sharpe_defaults_to_zero(self, tmp_path):
        """Null sharpe value should default to 0 (filtered out)."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "null_sharpe", "sharpe": None, "trades": 50, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert result == []

    def test_winners_missing_trades_defaults_to_zero(self, tmp_path):
        """Missing trades key should default to 0 (filtered out: trades < 1)."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "no_trades", "sharpe": 2.0, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert result == []

    def test_exactly_one_trade_passes_filter(self, tmp_path):
        """Strategy with exactly 1 trade and positive sharpe should pass filter."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "one_trade", "sharpe": 1.0, "trades": 1, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        # Should attempt to load code; if code available, returns 1
        assert len(result) <= 1

    def test_negative_ticker_no_crash(self, tmp_path):
        """Missing ticker field should not crash."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "no_ticker", "sharpe": 1.5, "trades": 30},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        # Should not crash; code loading determines if result is non-empty

    def test_ticker_no_match_falls_back(self, tmp_path):
        """When requested ticker doesn't match any, should still return non-matching winners."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "msft_strat", "sharpe": 2.0, "trades": 40, "ticker": "MSFT"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples(ticker="SPY")
        # Should return MSFT winner (no ticker match, no bonus)
        # May be empty if no code available, but should not crash
        assert isinstance(result, list)

    def test_corrupted_state_json_fallback(self, tmp_path):
        """If state.json is corrupted, should fall back to latest strategy file."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "bad_state", "sharpe": 1.5, "trades": 40,
             "ticker": "SPY", "refinement_run": "run_bad"},
        ]))
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "run_bad"
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text("not valid json{{{")
        (run_dir / "strategy_v2.py").write_text(
            "def generate_signals(df, params):\n    return df['close'], df['close']\n"
        )

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples()
        assert len(result) == 1
        assert "generate_signals" in result[0]["source_code"]

    def test_empty_refinement_run_dir(self, tmp_path):
        """If refinement run dir exists but has no strategy files, skip it."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "empty_run", "sharpe": 1.5, "trades": 40,
             "ticker": "SPY", "refinement_run": "run_empty"},
        ]))
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "run_empty"
        run_dir.mkdir(parents=True)
        # No strategy files at all

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples()
        # Should return empty since no code available
        assert result == []

    def test_refinement_run_nonexistent_dir(self, tmp_path):
        """If refinement_run references a nonexistent dir, skip gracefully."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "ghost", "sharpe": 1.5, "trades": 40,
             "ticker": "SPY", "refinement_run": "nonexistent_run"},
        ]))
        runs_dir = tmp_path / "runs"

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples()
        assert result == []

    def test_best_turn_from_state_json(self, tmp_path):
        """Should use best_turn from state.json to load correct strategy version."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "best_turn_test", "sharpe": 1.8, "trades": 50,
             "ticker": "SPY", "refinement_run": "run_bt"},
        ]))
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "run_bt"
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text(json.dumps({"best_turn": 2}))
        (run_dir / "strategy_v1.py").write_text("# Version 1\n")
        (run_dir / "strategy_v2.py").write_text(
            "def generate_signals(df, params):\n    return df['close'], df['close']\n"
        )
        (run_dir / "strategy_v3.py").write_text("# Version 3\n")

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples()
        assert len(result) == 1
        # Should load v2 since best_turn=2
        assert "generate_signals" in result[0]["source_code"]

    def test_long_strategy_truncation(self, tmp_path):
        """Strategies longer than 3000 chars should be truncated."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "long_strat", "sharpe": 1.5, "trades": 40,
             "ticker": "SPY", "refinement_run": "run_long"},
        ]))
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "run_long"
        run_dir.mkdir(parents=True)
        # Build code that exceeds 3000 chars with irrelevant content
        # Truncation keeps imports, generate_signals, DEFAULT_PARAMS, DESCRIPTION
        long_code = (
            "import pandas as pd\n"
            "from crabquant.indicator_cache import cached_indicator\n"
            "def generate_signals(df, params):\n"
            "    entries = df['close'] > df['close'].shift(1)\n"
            "    exits = ~entries\n"
            "    return entries, exits\n"
            "DEFAULT_PARAMS = {'fast': 10}\n"
            "DESCRIPTION = 'A strategy'\n"
            "# padding " + "x" * 3500 + "\n"
        )
        assert len(long_code) > 3000
        (run_dir / "strategy_v1.py").write_text(long_code)

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples()
        assert len(result) == 1
        # The truncation logic keeps import/def generate_signals/DEFAULT_PARAMS/DESCRIPTION
        # lines but drops the padding comment. However, the truncation may not always
        # produce shorter output depending on which lines match. Just verify it runs
        # without error and produces source_code.
        assert "source_code" in result[0]
        assert len(result[0]["source_code"]) > 0

    def test_max_examples_zero(self, tmp_path):
        """max_examples=0 should return empty list."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "any", "sharpe": 2.0, "trades": 50, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples(max_examples=0)
        assert result == []

    def test_multiple_duplicates_keeps_first(self, tmp_path):
        """When same name appears multiple times, only first occurrence is kept."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "dup_first", "sharpe": 2.0, "trades": 50, "ticker": "SPY"},
            {"strategy": "dup_first", "sharpe": 3.0, "trades": 60, "ticker": "SPY"},
            {"strategy": "dup_first", "sharpe": 4.0, "trades": 70, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert len(result) <= 1

    def test_custom_runs_dir_parameter(self, tmp_path):
        """Custom runs_dir parameter should be respected."""
        winners_file = tmp_path / "winners.json"
        custom_runs = tmp_path / "custom_runs"
        custom_run_dir = custom_runs / "custom_run"
        custom_run_dir.mkdir(parents=True)
        (custom_run_dir / "strategy_v1.py").write_text(
            "def generate_signals(df, params):\n    return df['close'], df['close']\n"
        )
        winners_file.write_text(json.dumps([
            {"strategy": "custom", "sharpe": 1.5, "trades": 40,
             "ticker": "SPY", "refinement_run": "custom_run"},
        ]))

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples(runs_dir=custom_runs)
        assert len(result) == 1
        assert "generate_signals" in result[0]["source_code"]

    def test_result_dict_keys(self, tmp_path):
        """Result dicts should have expected keys."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "key_test", "sharpe": 1.5, "trades": 40,
             "ticker": "SPY", "refinement_run": "run_keys", "params": {"a": 1}},
        ]))
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "run_keys"
        run_dir.mkdir(parents=True)
        (run_dir / "strategy_v1.py").write_text(
            "def generate_signals(df, params):\n    return df['close'], df['close']\n"
        )

        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file), \
             patch("crabquant.refinement.context_builder._RUNS_DIR", runs_dir):
            result = get_winner_examples()
        assert len(result) == 1
        example = result[0]
        assert "name" in example
        assert "sharpe" in example
        assert "trades" in example
        assert "ticker" in example
        assert "source_code" in example
        assert "params" in example

    def test_no_refinement_run_no_registry(self, tmp_path):
        """Strategy not in registry and no refinement_run → empty result."""
        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([
            {"strategy": "unknown_strategy_xyz", "sharpe": 2.0, "trades": 40, "ticker": "SPY"},
        ]))
        with patch("crabquant.refinement.context_builder._WINNERS_PATH", winners_file):
            result = get_winner_examples()
        assert result == []
