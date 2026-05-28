"""Tests for CrabQuant validation suite."""

import pytest
from crabquant.data import load_data
from crabquant.engine import BacktestEngine
from crabquant.strategies.rsi_crossover import generate_signals, DEFAULT_PARAMS
from crabquant.validation import walk_forward_test, cross_ticker_validation


class TestWalkForward:
    """Test walk-forward validation."""

    def test_walk_forward_returns_result(self):
        """walk_forward_test should return a WalkForwardResult."""
        result = walk_forward_test(generate_signals, "AAPL", DEFAULT_PARAMS)

        assert result.strategy_name == "generate_signals"
        assert result.ticker == "AAPL"
        assert hasattr(result, "train_sharpe")
        assert hasattr(result, "test_sharpe")
        assert hasattr(result, "degradation")
        assert hasattr(result, "robust")

    def test_walk_forward_data_split(self):
        """Walk-forward should split data into train/test periods."""
        result = walk_forward_test(generate_signals, "MSFT", DEFAULT_PARAMS,
                                   train_months=18, test_months=6)
        # Just verify it doesn't crash and returns valid numbers
        assert isinstance(result.train_sharpe, float)
        assert isinstance(result.test_sharpe, float)


class TestCrossTicker:
    """Test cross-ticker validation."""

    def test_cross_ticker_returns_result(self):
        """cross_ticker_validation should return a CrossTickerResult."""
        result = cross_ticker_validation(
            generate_signals, DEFAULT_PARAMS, ["AAPL", "MSFT", "GOOGL"]
        )

        assert result.tickers_tested >= 1
        assert result.tickers_profitable >= 0
        assert hasattr(result, "avg_sharpe")
        assert hasattr(result, "median_sharpe")
        assert hasattr(result, "sharpe_std")
        assert hasattr(result, "robust")

    def test_cross_ticker_invalid_tickers_handled(self):
        """Invalid tickers should be skipped gracefully."""
        result = cross_ticker_validation(
            generate_signals, DEFAULT_PARAMS, ["INVALID1", "INVALID2"]
        )
        # Should not crash, just report 0 results
        assert result.tickers_profitable == 0
