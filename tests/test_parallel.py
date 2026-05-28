"""Tests for CrabQuant parallel backtesting."""

import time

import numpy as np
import pytest

from crabquant.data import load_data
from crabquant.engine import BacktestEngine, BacktestResult
from crabquant.engine.parallel import parallel_backtest
from crabquant.strategies import STRATEGY_REGISTRY
from crabquant.strategies.rsi_crossover import generate_signals_matrix, PARAM_GRID


@pytest.fixture
def small_grid():
    """Small param grid for fast tests."""
    return {k: v[:2] for k, v in PARAM_GRID.items()}


@pytest.fixture
def tickers():
    """Small set of tickers for parallel tests (uses cached data)."""
    return ["AAPL", "MSFT"]


class TestParallelBacktest:
    """Test the parallel_backtest function."""

    def test_returns_correct_count(self, tickers, small_grid):
        """parallel_backtest should return results for all tickers."""
        results = parallel_backtest("rsi_crossover", tickers, small_grid, max_workers=2)

        # Each ticker should produce results equal to the grid size
        expected_cols = 1
        for v in small_grid.values():
            expected_cols *= len(v)

        # Total results should be tickers * combos_per_ticker
        assert len(results) == len(tickers) * expected_cols

    def test_results_are_backtest_result(self, tickers, small_grid):
        """All results should be BacktestResult instances."""
        results = parallel_backtest("rsi_crossover", tickers, small_grid, max_workers=2)

        assert all(isinstance(r, BacktestResult) for r in results)

    def test_results_have_correct_tickers(self, tickers, small_grid):
        """Each result should have the correct ticker assigned."""
        results = parallel_backtest("rsi_crossover", tickers, small_grid, max_workers=2)

        result_tickers = {r.ticker for r in results}
        assert result_tickers == set(tickers)

    def test_results_have_correct_strategy(self, tickers, small_grid):
        """Each result should have the correct strategy name."""
        results = parallel_backtest("rsi_crossover", tickers, small_grid, max_workers=2)

        assert all(r.strategy_name == "rsi_crossover" for r in results)

    def test_matches_sequential(self, tickers, small_grid):
        """Parallel results should match sequential execution."""
        # Sequential
        engine = BacktestEngine()
        seq_results = []
        for ticker in tickers:
            df = load_data(ticker, period="2y")
            entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)
            ticker_results = engine.run_vectorized(df, entries_df, exits_df, param_list, "rsi_crossover", ticker)
            seq_results.extend(ticker_results)

        # Parallel
        par_results = parallel_backtest("rsi_crossover", tickers, small_grid, max_workers=2)

        assert len(seq_results) == len(par_results)

        # Match by ticker and params (order may differ between parallel runs)
        seq_by_key = {(r.ticker, str(sorted(r.params.items()))): r for r in seq_results}
        par_by_key = {(r.ticker, str(sorted(r.params.items()))): r for r in par_results}

        assert set(seq_by_key.keys()) == set(par_by_key.keys())

        for key in seq_by_key:
            sr = seq_by_key[key]
            pr = par_by_key[key]
            assert abs(sr.sharpe - pr.sharpe) < 0.001, \
                f"Sharpe mismatch for {key}: seq={sr.sharpe}, par={pr.sharpe}"
            assert abs(sr.total_return - pr.total_return) < 0.001, \
                f"Return mismatch for {key}: seq={sr.total_return}, par={pr.total_return}"
            assert sr.num_trades == pr.num_trades, \
                f"Trade count mismatch for {key}: seq={sr.num_trades}, par={pr.num_trades}"

    def test_single_ticker(self, small_grid):
        """Single ticker should work (degenerate case)."""
        results = parallel_backtest("rsi_crossover", ["AAPL"], small_grid, max_workers=1)

        expected_cols = 1
        for v in small_grid.values():
            expected_cols *= len(v)

        assert len(results) == expected_cols

    def test_unknown_strategy(self, tickers, small_grid):
        """Unknown strategy should return empty list."""
        results = parallel_backtest("nonexistent_strategy", tickers, small_grid, max_workers=2)
        assert results == []

    def test_fallback_to_sequential_on_small_count(self, small_grid):
        """max_workers=1 should still produce correct results."""
        results = parallel_backtest("rsi_crossover", ["AAPL"], small_grid, max_workers=1)

        expected_cols = 1
        for v in small_grid.values():
            expected_cols *= len(v)

        assert len(results) == expected_cols
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_three_tickers_parallel(self, small_grid):
        """Three tickers with 2 workers should produce correct results."""
        tickers = ["AAPL", "MSFT", "SPY"]
        results = parallel_backtest("rsi_crossover", tickers, small_grid, max_workers=2)

        expected_cols = 1
        for v in small_grid.values():
            expected_cols *= len(v)

        assert len(results) == len(tickers) * expected_cols

        result_tickers = {r.ticker for r in results}
        assert result_tickers == set(tickers)
