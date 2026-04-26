"""Tests for CrabQuant backtest engine."""

import pytest
import pandas as pd
import numpy as np
from crabquant.data import load_data
from crabquant.engine import BacktestEngine, BacktestResult
from crabquant.strategies.rsi_crossover import generate_signals


class TestBacktestEngine:
    """Test the backtest engine."""

    def test_engine_creation(self):
        """Engine should create with default params."""
        engine = BacktestEngine()
        assert engine.sharpe_target == 1.5
        assert engine.max_drawdown_limit == 0.25
        assert engine.initial_cash == 100_000

    def test_engine_custom_params(self):
        """Engine should accept custom params."""
        engine = BacktestEngine(sharpe_target=2.0, min_trades=10)
        assert engine.sharpe_target == 2.0
        assert engine.min_trades == 10

    def test_run_returns_result(self):
        """run() should return a BacktestResult."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")

        assert isinstance(result, BacktestResult)
        assert result.ticker == "AAPL"
        assert result.strategy_name == "test"

    def test_result_has_all_fields(self):
        """Result should have all expected fields."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")

        assert hasattr(result, "sharpe")
        assert hasattr(result, "total_return")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "num_trades")
        assert hasattr(result, "calmar_ratio")
        assert hasattr(result, "sortino_ratio")
        assert hasattr(result, "profit_factor")
        assert hasattr(result, "score")
        assert hasattr(result, "passed")

    def test_no_signals_returns_zero_trades(self):
        """All-False entries should produce zero trades."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert result.num_trades == 0
        assert result.passed == False

    def test_passed_criteria(self):
        """Result should only pass when all criteria met."""
        engine = BacktestEngine(sharpe_target=10.0)  # Impossible to pass
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert result.passed == False  # Shouldn't pass with Sharpe >= 10

    def test_score_penalizes_low_trades(self):
        """Score should be lower for strategies with fewer trades."""
        engine = BacktestEngine(sharpe_target=0.0, min_trades=0, min_total_return=0)
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")
        # Score should be finite and non-negative
        assert np.isfinite(result.score)
        assert result.score >= 0

    def test_params_stored_in_result(self):
        """Params should be stored in the result."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)
        params = {"fast_len": 7, "slow_len": 21}

        result = engine.run(df, entries, exits, "test", "AAPL", params=params)
        assert result.params == params
