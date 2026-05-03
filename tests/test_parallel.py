"""Tests for CrabQuant parallel backtesting."""

import logging
import time
from concurrent.futures import Future
from dataclasses import asdict
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from crabquant.data import load_data
from crabquant.engine import BacktestEngine, BacktestResult
from crabquant.engine.parallel import parallel_backtest, _worker_backtest
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


def _make_result_dict(ticker="AAPL", strategy="rsi_crossover", **overrides):
    """Create a valid BacktestResult dict for mocking worker output."""
    base = {
        "ticker": ticker,
        "strategy_name": strategy,
        "iteration": 0,
        "sharpe": 1.5,
        "total_return": 0.25,
        "max_drawdown": -0.10,
        "win_rate": 0.55,
        "num_trades": 20,
        "avg_trade_return": 0.012,
        "calmar_ratio": 2.5,
        "sortino_ratio": 3.0,
        "profit_factor": 1.5,
        "avg_holding_bars": 5.0,
        "best_trade": 0.05,
        "worst_trade": -0.03,
        "passed": True,
        "score": 75.0,
        "notes": "",
        "params": {"rsi_period": 14},
    }
    base.update(overrides)
    return base


def _make_error_dict(ticker="AAPL", strategy="rsi_crossover", error_msg="boom"):
    """Create an error result dict (as returned by _worker_backtest on failure)."""
    return {"_error": True, "ticker": ticker, "strategy": strategy, "error": error_msg}


class FakeExecutor:
    """A fake ProcessPoolExecutor that runs tasks synchronously in the caller process."""

    def __init__(self, max_workers=None, fail_tickers=None, empty_tickers=None):
        self.max_workers = max_workers
        self.fail_tickers = fail_tickers or set()
        self.empty_tickers = empty_tickers or set()
        self._shutdown = False

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        try:
            args_tuple = args[0] if args else kwargs.get("args", ())
            ticker = args_tuple[1] if args_tuple else "UNKNOWN"
            if ticker in self.fail_tickers:
                # Simulate worker exception at the future level
                fut.set_exception(RuntimeError(f"Worker crash for {ticker}"))
            elif ticker in self.empty_tickers:
                fut.set_result([])
            else:
                result = fn(args_tuple) if args else fn(*args, **kwargs)
                fut.set_result(result)
        except Exception as exc:
            fut.set_exception(exc)
        return fut

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._shutdown = True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Original integration tests (require cached data)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParallelBacktest:
    """Test the parallel_backtest function — integration tests with real data."""

    def test_returns_correct_count(self, tickers, small_grid):
        """parallel_backtest should return results for all tickers."""
        results = parallel_backtest("rsi_crossover", tickers, small_grid, max_workers=2)

        expected_cols = 1
        for v in small_grid.values():
            expected_cols *= len(v)

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
        engine = BacktestEngine()
        seq_results = []
        for ticker in tickers:
            df = load_data(ticker, period="2y")
            entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)
            ticker_results = engine.run_vectorized(df, entries_df, exits_df, param_list, "rsi_crossover", ticker)
            seq_results.extend(ticker_results)

        par_results = parallel_backtest("rsi_crossover", tickers, small_grid, max_workers=2)

        assert len(seq_results) == len(par_results)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Mocked unit tests — no real data / processes required
# ═══════════════════════════════════════════════════════════════════════════════

class TestParallelBacktestMocked:
    """Unit tests for parallel_backtest using mocked executor + worker."""

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_empty_ticker_list(self, mock_workers, mock_executor_cls):
        """Empty ticker list should return empty results."""
        results = parallel_backtest("rsi_crossover", [], PARAM_GRID, max_workers=2)
        assert results == []

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=1)
    def test_single_ticker_mocked(self, mock_workers, mock_executor_cls):
        """Single ticker with mocked executor returns results."""
        rd = _make_result_dict(ticker="SPY")
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=_immediate_future([rd]))
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ["SPY"], PARAM_GRID, max_workers=1)
        assert len(results) == 1
        assert results[0].ticker == "SPY"
        assert isinstance(results[0], BacktestResult)

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_mixed_success_and_error_results(self, mock_workers, mock_executor_cls):
        """Worker errors ( dicts) are filtered out; only valid results returned."""
        def submit_side_effect(fn, args):
            ticker = args[1]
            if ticker == "MSFT":
                return _immediate_future([_make_error_dict(ticker="MSFT", error_msg="data fail")])
            return _immediate_future([_make_result_dict(ticker="AAPL")])

        fake_exec = FakeExecutor()
        fake_exec.submit = submit_side_effect
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ["AAPL", "MSFT"], PARAM_GRID, max_workers=2)
        assert len(results) == 1
        assert results[0].ticker == "AAPL"

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_all_workers_fail(self, mock_workers, mock_executor_cls):
        """When every worker returns error dicts, result list is empty."""
        rds = [_make_error_dict(ticker="AAPL"), _make_error_dict(ticker="MSFT")]
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(side_effect=lambda fn, args: _immediate_future(rds))
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ["AAPL", "MSFT"], PARAM_GRID, max_workers=2)
        assert results == []

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_future_exception_handled(self, mock_workers, mock_executor_cls):
        """If future.result() raises, it should be caught and not propagate."""
        fut = Future()
        fut.set_exception(RuntimeError("process died"))
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=fut)
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ["AAPL"], PARAM_GRID, max_workers=2)
        # Error is caught; result list is empty
        assert results == []

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_multiple_results_per_ticker(self, mock_workers, mock_executor_cls):
        """Worker returning multiple result dicts should all appear in output."""
        rds = [
            _make_result_dict(ticker="AAPL", iteration=0),
            _make_result_dict(ticker="AAPL", iteration=1),
            _make_result_dict(ticker="AAPL", iteration=2),
        ]
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(side_effect=lambda fn, args: _immediate_future(rds))
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ["AAPL"], PARAM_GRID, max_workers=2)
        assert len(results) == 3
        assert all(r.ticker == "AAPL" for r in results)

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=4)
    def test_workers_capped_to_ticker_count(self, mock_workers, mock_executor_cls):
        """actual_workers should not exceed number of tickers."""
        rd = _make_result_dict(ticker="SPY")
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=_immediate_future([rd]))
        mock_executor_cls.return_value = fake_exec

        parallel_backtest("rsi_crossover", ["SPY"], PARAM_GRID, max_workers=8)

        # Executor should have been created with max_workers <= len(tickers)
        mock_executor_cls.assert_called_once()
        call_kw = mock_executor_cls.call_args
        assert call_kw[1]["max_workers"] <= 1  # only 1 ticker

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_default_max_workers_uses_cpu_count(self, mock_workers, mock_executor_cls):
        """When max_workers=None, it defaults to min(cpu_count, len(tickers))."""
        rd = _make_result_dict(ticker="SPY")
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=_immediate_future([rd]))
        mock_executor_cls.return_value = fake_exec

        with patch("crabquant.engine.parallel.os.cpu_count", return_value=4):
            parallel_backtest("rsi_crossover", ["SPY"], PARAM_GRID)

        mock_executor_cls.assert_called_once()
        # Should be min(4, 1) = 1
        assert mock_executor_cls.call_args[1]["max_workers"] == 1

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_custom_period_passed_to_workers(self, mock_workers, mock_executor_cls):
        """The period parameter is forwarded to worker args."""
        captured_args = []

        def capture_submit(fn, args):
            captured_args.append(args)
            return _immediate_future([])

        fake_exec = FakeExecutor()
        fake_exec.submit = capture_submit
        mock_executor_cls.return_value = fake_exec

        parallel_backtest("rsi_crossover", ["AAPL"], PARAM_GRID, period="5y")
        assert len(captured_args) == 1
        assert captured_args[0][3] == "5y"  # period is 4th element

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_custom_resource_monitor_accepted(self, mock_workers, mock_executor_cls):
        """Passing an external ResourceMonitor should be accepted without error."""
        from crabquant.engine.resource_monitor import ResourceMonitor

        rd = _make_result_dict(ticker="SPY")
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=_immediate_future([rd]))
        mock_executor_cls.return_value = fake_exec

        monitor = ResourceMonitor()
        results = parallel_backtest(
            "rsi_crossover", ["SPY"], PARAM_GRID, max_workers=2,
            resource_monitor=monitor,
        )
        assert len(results) == 1

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_passed_filter_in_results(self, mock_workers, mock_executor_cls):
        """Results with passed=False are still returned (not filtered out)."""
        rds = [
            _make_result_dict(ticker="AAPL", passed=True, score=80),
            _make_result_dict(ticker="AAPL", passed=False, score=20),
        ]
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(side_effect=lambda fn, args: _immediate_future(rds))
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ["AAPL"], PARAM_GRID, max_workers=2)
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False


class TestParallelBacktestLogging:
    """Verify that parallel_backtest emits expected log messages."""

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_logs_start_info(self, mock_workers, mock_executor_cls, caplog):
        """Should log 'Starting parallel backtest' at INFO level."""
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=_immediate_future([]))
        mock_executor_cls.return_value = fake_exec

        with caplog.at_level(logging.INFO, logger="crabquant.engine.parallel"):
            parallel_backtest("rsi_crossover", ["AAPL"], PARAM_GRID, max_workers=2)

        assert any("Starting parallel backtest" in r.message for r in caplog.records)

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_logs_completion_summary(self, mock_workers, mock_executor_cls, caplog):
        """Should log 'Parallel backtest complete' with stats."""
        rds = [_make_result_dict(ticker="AAPL", num_trades=10, passed=True)]
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(side_effect=lambda fn, args: _immediate_future(rds))
        mock_executor_cls.return_value = fake_exec

        with caplog.at_level(logging.INFO, logger="crabquant.engine.parallel"):
            parallel_backtest("rsi_crossover", ["AAPL"], PARAM_GRID, max_workers=2)

        assert any("Parallel backtest complete" in r.message for r in caplog.records)
        assert any("1 results" in r.message for r in caplog.records)

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_logs_error_warning(self, mock_workers, mock_executor_cls, caplog):
        """Should log WARNING when workers return errors."""
        rds = [_make_error_dict(ticker="AAPL", error_msg="bad data")]
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(side_effect=lambda fn, args: _immediate_future(rds))
        mock_executor_cls.return_value = fake_exec

        with caplog.at_level(logging.WARNING, logger="crabquant.engine.parallel"):
            parallel_backtest("rsi_crossover", ["AAPL"], PARAM_GRID, max_workers=2)

        assert any("Error:" in r.message or "error" in r.message.lower() for r in caplog.records if r.levelno >= logging.WARNING)

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_logs_future_exception_as_error(self, mock_workers, mock_executor_cls, caplog):
        """Future-level exceptions should be logged at ERROR."""
        fut = Future()
        fut.set_exception(RuntimeError("kaboom"))
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=fut)
        mock_executor_cls.return_value = fake_exec

        with caplog.at_level(logging.ERROR, logger="crabquant.engine.parallel"):
            parallel_backtest("rsi_crossover", ["AAPL"], PARAM_GRID, max_workers=2)

        assert any("kaboom" in r.message for r in caplog.records)


class TestWorkerBacktest:
    """Direct tests for the _worker_backtest helper (runs in-process)."""

    def test_unknown_strategy_returns_empty(self):
        """_worker_backtest with unknown strategy returns empty list."""
        result = _worker_backtest(("no_such_strategy", "AAPL", PARAM_GRID, "2y"))
        assert result == []

    def test_empty_param_grid_returns_empty(self):
        """_worker_backtest with empty param_grid returns empty list."""
        result = _worker_backtest(("rsi_crossover", "AAPL", {}, "2y"))
        assert result == []

    def test_valid_strategy_returns_list(self):
        """_worker_backtest with known strategy returns a list of dicts."""
        # This test uses real data (cached); if unavailable, skip
        try:
            result = _worker_backtest(("rsi_crossover", "AAPL", {k: [v[0]] for k, v in PARAM_GRID.items()}, "2y"))
        except FileNotFoundError:
            pytest.skip("No cached data available")
        assert isinstance(result, list)

    def test_exception_in_worker_returns_error_dict(self):
        """If the worker encounters an exception, it returns an error dict."""
        # load_data is imported inside _worker_backtest from crabquant.data
        with patch("crabquant.data.load_data", side_effect=FileNotFoundError("no cache")):
            result = _worker_backtest(("rsi_crossover", "BOGUS", PARAM_GRID, "2y"))
        assert len(result) == 1
        assert result[0].get("_error") is True
        assert "BOGUS" in result[0]["ticker"]


class TestParallelBacktestEdgeCases:
    """Additional edge-case tests for parallel_backtest."""

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_large_number_of_tickers(self, mock_workers, mock_executor_cls):
        """Many tickers should all produce results."""
        ticker_list = [f"TK{i:03d}" for i in range(20)]
        rds_per_ticker = [_make_result_dict(ticker=t) for t in ticker_list]

        call_count = 0
        def submit_side_effect(fn, args):
            nonlocal call_count
            call_count += 1
            ticker = args[1]
            return _immediate_future([_make_result_dict(ticker=ticker)])

        fake_exec = FakeExecutor()
        fake_exec.submit = submit_side_effect
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ticker_list, PARAM_GRID, max_workers=4)
        assert len(results) == 20
        assert call_count == 20

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=1)
    def test_zero_max_workers(self, mock_workers, mock_executor_cls):
        """When compute_optimal_workers returns 0, actual_workers is capped to 0 by len(tickers)."""
        mock_workers.return_value = 0
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=_immediate_future([]))
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", [], PARAM_GRID, max_workers=0)
        assert results == []

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_result_fields_preserved(self, mock_workers, mock_executor_cls):
        """All fields from worker result dicts should be preserved in BacktestResult."""
        custom = _make_result_dict(
            ticker="GOOG",
            sharpe=2.34,
            total_return=0.567,
            max_drawdown=-0.15,
            win_rate=0.65,
            num_trades=42,
            passed=False,
            score=10.0,
            params={"fast": 5, "slow": 20},
        )
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=_immediate_future([custom]))
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ["GOOG"], PARAM_GRID, max_workers=2)
        assert len(results) == 1
        r = results[0]
        assert r.ticker == "GOOG"
        assert r.sharpe == 2.34
        assert r.total_return == 0.567
        assert r.max_drawdown == -0.15
        assert r.win_rate == 0.65
        assert r.num_trades == 42
        assert r.passed is False
        assert r.score == 10.0
        assert r.params == {"fast": 5, "slow": 20}

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_duplicate_tickers_allowed(self, mock_workers, mock_executor_cls):
        """Duplicate tickers in list should be processed (no dedup by framework)."""
        fake_exec = FakeExecutor()
        call_count = 0
        def submit_side_effect(fn, args):
            nonlocal call_count
            call_count += 1
            return _immediate_future([_make_result_dict(ticker=args[1])])
        fake_exec.submit = submit_side_effect
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ["AAPL", "AAPL"], PARAM_GRID, max_workers=2)
        assert len(results) == 2
        assert call_count == 2

    @patch("crabquant.engine.parallel.ProcessPoolExecutor")
    @patch("crabquant.engine.parallel.compute_optimal_workers", return_value=2)
    def test_no_trades_still_in_results(self, mock_workers, mock_executor_cls):
        """Results with num_trades=0 should still be included in the returned list."""
        rd = _make_result_dict(ticker="AAPL", num_trades=0)
        fake_exec = FakeExecutor()
        fake_exec.submit = MagicMock(return_value=_immediate_future([rd]))
        mock_executor_cls.return_value = fake_exec

        results = parallel_backtest("rsi_crossover", ["AAPL"], PARAM_GRID, max_workers=2)
        assert len(results) == 1
        assert results[0].num_trades == 0


def _immediate_future(value):
    """Helper: create an already-resolved Future."""
    fut = Future()
    fut.set_result(value)
    return fut
