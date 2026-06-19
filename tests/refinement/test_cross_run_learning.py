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
