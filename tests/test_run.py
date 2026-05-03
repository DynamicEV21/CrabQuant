"""
Comprehensive unit tests for crabquant/run.py
"""

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from crabquant.run import (
    DEFAULT_TICKERS,
    LOGS_DIR,
    RESULTS_DIR,
    VALIDATION_DIR,
    WINNERS_DIR,
    load_winners_from_log,
    main,
    mutate_params,
    print_result,
    print_summary,
    run_discovery,
    run_validation,
    sample_params,
    save_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(**overrides):
    """Create a mock result object with sensible defaults."""
    defaults = dict(
        sharpe=1.5,
        total_return=0.1,
        max_drawdown=-0.05,
        num_trades=20,
        win_rate=0.6,
        calmar_ratio=2.0,
        sortino_ratio=1.8,
        profit_factor=1.5,
        score=3.0,
        passed=True,
        ticker="SPY",
        strategy_name="macd",
        iteration=0,
        params={},
        notes="test",
        timestamp="2026-01-01",
        avg_trade_return=0.005,
        avg_holding_bars=10.0,
        best_trade=0.05,
        worst_trade=-0.02,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants:
    """Verify module-level constants are sane."""

    def test_default_tickers_is_list_of_str(self):
        assert isinstance(DEFAULT_TICKERS, list)
        assert all(isinstance(t, str) for t in DEFAULT_TICKERS)

    def test_default_tickers_non_empty(self):
        assert len(DEFAULT_TICKERS) > 0

    def test_default_tickers_has_spy(self):
        assert "SPY" in DEFAULT_TICKERS

    def test_results_dir_is_path(self):
        assert isinstance(RESULTS_DIR, Path)

    def test_logs_dir_under_results(self):
        assert LOGS_DIR.parent == RESULTS_DIR

    def test_winners_dir_under_results(self):
        assert WINNERS_DIR.parent == RESULTS_DIR

    def test_validation_dir_under_results(self):
        assert VALIDATION_DIR.parent == RESULTS_DIR


# ===========================================================================
# sample_params
# ===========================================================================

class TestSampleParams:
    def test_basic(self):
        grid = {"a": [1, 2, 3], "b": ["x", "y"]}
        assert sample_params(grid) == {"a": 1, "b": "x"}

    def test_single_value(self):
        grid = {"k": [42]}
        assert sample_params(grid) == {"k": 42}

    def test_empty_grid(self):
        assert sample_params({}) == {}

    def test_string_values(self):
        grid = {"period": ["fast", "slow"], "source": ["close"]}
        assert sample_params(grid) == {"period": "fast", "source": "close"}


# ===========================================================================
# mutate_params
# ===========================================================================

class TestMutateParams:
    def test_basic_shift(self):
        grid = {"a": [1, 2, 3, 4, 5]}
        params = {"a": 1}
        result = mutate_params(params, grid, 0)
        # iteration=0: shift = 1*(1+0) = 1 → idx 0+1=1
        assert result["a"] == 2

    def test_iteration_one_shift(self):
        grid = {"a": [1, 2, 3, 4, 5]}
        params = {"a": 3}
        result = mutate_params(params, grid, 1)
        # iteration=1: shift = -1*(1+1) = -2 → idx 2-2=0
        assert result["a"] == 1

    def test_clamp_at_start(self):
        grid = {"a": [10, 20, 30]}
        params = {"a": 10}
        result = mutate_params(params, grid, 1)
        # iteration=1: shift=-2 → idx 0-2 clamped to 0
        assert result["a"] == 10

    def test_clamp_at_end(self):
        grid = {"a": [10, 20, 30]}
        params = {"a": 30}
        result = mutate_params(params, grid, 0)
        # iteration=0: shift=1 → idx 2+1 clamped to 2
        assert result["a"] == 30

    def test_single_value_grid(self):
        grid = {"a": [42]}
        params = {"a": 42}
        result = mutate_params(params, grid, 0)
        assert result["a"] == 42

    def test_value_not_in_grid_defaults_to_first(self):
        grid = {"a": [1, 2, 3]}
        params = {"a": 999}
        result = mutate_params(params, grid, 0)
        # 999 not in grid → idx=0, shift=1 → idx=1
        assert result["a"] == 2

    def test_multiple_keys(self):
        grid = {"fast": [5, 10, 15], "slow": [20, 30, 40]}
        params = {"fast": 5, "slow": 30}
        result = mutate_params(params, grid, 0)
        assert "fast" in result
        assert "slow" in result

    def test_iteration_two(self):
        grid = {"a": [1, 2, 3, 4, 5]}
        params = {"a": 2}
        result = mutate_params(params, grid, 2)
        # iteration=2 (even): shift = 1*(1+0) = 1 → idx 1+1=2
        assert result["a"] == 3

    def test_iteration_three(self):
        grid = {"a": [1, 2, 3, 4, 5]}
        params = {"a": 4}
        result = mutate_params(params, grid, 3)
        # iteration=3 (odd): shift = -1*(1+1) = -2 → idx 3-2=1
        assert result["a"] == 2

    def test_empty_grid(self):
        result = mutate_params({}, {}, 0)
        assert result == {}

    def test_key_missing_from_params(self):
        grid = {"a": [1, 2, 3]}
        result = mutate_params({}, grid, 0)
        # key not in params → defaults to values[0]=1, shift=1 → idx=1
        assert result["a"] == 2


# ===========================================================================
# save_result / load_winners_from_log
# ===========================================================================

class TestSaveResultAndLoad:
    def test_save_creates_file(self, tmp_path):
        save_result(_make_result(), tmp_path)
        assert (tmp_path / "backtest_log.jsonl").exists()

    def test_save_appends_multiple(self, tmp_path):
        r1 = _make_result(ticker="AAPL")
        r2 = _make_result(ticker="MSFT")
        save_result(r1, tmp_path)
        save_result(r2, tmp_path)
        lines = (tmp_path / "backtest_log.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_save_json_structure(self, tmp_path):
        save_result(_make_result(ticker="TSLA"), tmp_path)
        data = json.loads((tmp_path / "backtest_log.jsonl").read_text().strip())
        assert data["ticker"] == "TSLA"
        assert data["passed"] is True
        assert "timestamp" in data

    def test_roundtrip_passed_only(self, tmp_path):
        passed = _make_result(ticker="GOOD", passed=True)
        failed = _make_result(ticker="BAD", passed=False)
        save_result(passed, tmp_path)
        save_result(failed, tmp_path)

        with patch("crabquant.run.LOGS_DIR", tmp_path):
            winners = load_winners_from_log()
        assert len(winners) == 1
        assert winners[0].ticker == "GOOD"

    def test_load_empty_file(self, tmp_path):
        (tmp_path / "backtest_log.jsonl").write_text("")
        with patch("crabquant.run.LOGS_DIR", tmp_path):
            winners = load_winners_from_log()
        assert winners == []

    def test_load_missing_file(self, tmp_path):
        with patch("crabquant.run.LOGS_DIR", tmp_path):
            winners = load_winners_from_log()
        assert winners == []

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        save_result(_make_result(), nested)
        assert nested.exists()

    def test_save_preserves_all_fields(self, tmp_path):
        r = _make_result(sharpe=2.5, total_return=0.3, params={"fast": 10})
        save_result(r, tmp_path)
        data = json.loads((tmp_path / "backtest_log.jsonl").read_text().strip())
        assert data["sharpe"] == 2.5
        assert data["total_return"] == 0.3
        assert data["params"]["fast"] == 10

    def test_roundtrip_multiple_winners(self, tmp_path):
        for t in ["AAPL", "GOOGL", "MSFT"]:
            save_result(_make_result(ticker=t, passed=True), tmp_path)
        save_result(_make_result(ticker="BAD", passed=False), tmp_path)

        with patch("crabquant.run.LOGS_DIR", tmp_path):
            winners = load_winners_from_log()
        tickers = [w.ticker for w in winners]
        assert tickers == ["AAPL", "GOOGL", "MSFT"]


# ===========================================================================
# print_result
# ===========================================================================

class TestPrintResult:
    def test_passed_result(self, capsys):
        print_result(_make_result(passed=True))
        output = capsys.readouterr().out
        assert "PASS" in output
        assert "SPY" in output

    def test_failed_result(self, capsys):
        print_result(_make_result(passed=False))
        output = capsys.readouterr().out
        assert "MISS" in output

    def test_contains_metrics(self, capsys):
        print_result(_make_result(sharpe=2.5, num_trades=50))
        output = capsys.readouterr().out
        assert "2.50" in output
        assert "50" in output

    def test_does_not_raise(self):
        print_result(_make_result())


# ===========================================================================
# print_summary
# ===========================================================================

class TestPrintSummary:
    def test_empty_list(self, capsys):
        print_summary([])
        output = capsys.readouterr().out
        assert "No results" in output

    def test_with_passed_results(self, capsys, tmp_path):
        results = [_make_result(ticker="AAPL"), _make_result(ticker="MSFT")]
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary(results)
        output = capsys.readouterr().out
        assert "DISCOVERY SUMMARY" in output
        assert "2" in output  # total combos

    def test_creates_summary_json(self, tmp_path):
        results = [_make_result(ticker="AAPL")]
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary(results)
        summary_file = tmp_path / "summary.json"
        assert summary_file.exists()
        data = json.loads(summary_file.read_text())
        assert data["total_combos"] == 1
        assert data["passed"] == 1

    def test_empty_does_not_create_file(self, tmp_path):
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary([])
        assert not (tmp_path / "summary.json").exists()

    def test_summary_contains_timestamp(self, tmp_path):
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary([_make_result()])
        data = json.loads((tmp_path / "summary.json").read_text())
        assert "timestamp" in data

    def test_top_winners_sorted_by_score(self, tmp_path):
        results = [
            _make_result(ticker="LOW", score=1.0),
            _make_result(ticker="HIGH", score=5.0),
        ]
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary(results)
        data = json.loads((tmp_path / "summary.json").read_text())
        assert data["top_winners"][0]["ticker"] == "HIGH"

    def test_mixed_passed_and_failed(self, capsys, tmp_path):
        results = [
            _make_result(ticker="PASS", passed=True),
            _make_result(ticker="FAIL", passed=False, sharpe=-0.5),
        ]
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary(results)
        output = capsys.readouterr().out
        assert "Passed target: 1" in output

    def test_all_valid_section(self, capsys, tmp_path):
        results = [
            _make_result(ticker="A", sharpe=2.0, passed=True),
            _make_result(ticker="B", sharpe=1.0, passed=False, num_trades=10),
        ]
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary(results)
        output = capsys.readouterr().out
        assert "Top 10 by Sharpe" in output

    def test_summary_no_passed(self, tmp_path):
        results = [_make_result(passed=False)]
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary(results)
        data = json.loads((tmp_path / "summary.json").read_text())
        assert data["passed"] == 0
        assert data["top_winners"] == []


# ===========================================================================
# run_discovery
# ===========================================================================

class TestRunDiscovery:
    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "test_strat": (lambda df, p: (None, None), {}, {"a": [1, 2]}, "Test strategy", None)
    })
    @patch("crabquant.run.print_summary")
    @patch("crabquant.run.save_result")
    @patch("crabquant.run.print_result")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.load_data")
    def test_basic_flow(self, mock_load, mock_engine_cls, mock_print, mock_save, mock_summary):
        mock_df = MagicMock()
        mock_load.return_value = mock_df
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.run.return_value = _make_result(passed=True)

        results = run_discovery(strategies=["test_strat"], tickers=["SPY"], max_iterations=2)

        assert len(results) == 1
        assert results[0].ticker == "SPY"

    @patch("crabquant.run.STRATEGY_REGISTRY", {})
    @patch("crabquant.run.print_summary")
    def test_unknown_strategy_skipped(self, mock_summary):
        results = run_discovery(strategies=["ghost"], tickers=["SPY"], max_iterations=1)
        assert results == []

    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "s": (lambda df, p: (None, None), {}, {"a": [1]}, "desc", None)
    })
    @patch("crabquant.run.print_summary")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.load_data", side_effect=Exception("no data"))
    def test_load_data_failure_skipped(self, mock_load, mock_engine_cls, mock_summary):
        results = run_discovery(strategies=["s"], tickers=["BAD"], max_iterations=1)
        assert results == []

    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "s": (lambda df, p: (None, None), {}, {"a": [1]}, "desc", None)
    })
    @patch("crabquant.run.print_summary")
    @patch("crabquant.run.save_result")
    @patch("crabquant.run.print_result")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.load_data")
    def test_iteration_error_continues(self, mock_load, mock_engine_cls, mock_print, mock_save, mock_summary):
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.run.side_effect = [Exception("boom"), _make_result(passed=True)]

        results = run_discovery(strategies=["s"], tickers=["SPY"], max_iterations=3)
        # Should still get a result from the successful iteration
        assert len(results) >= 1

    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "s": (lambda df, p: (None, None), {}, {"a": [1, 2, 3]}, "desc", None)
    })
    @patch("crabquant.run.print_summary")
    @patch("crabquant.run.save_result")
    @patch("crabquant.run.print_result")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.load_data")
    def test_calls_save_and_print_each_iteration(self, mock_load, mock_engine_cls, mock_print, mock_save, mock_summary):
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.run.return_value = _make_result()

        run_discovery(strategies=["s"], tickers=["SPY"], max_iterations=3)

        assert mock_save.call_count == 3
        assert mock_print.call_count == 3

    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "s": (lambda df, p: (None, None), {}, {"a": [1]}, "desc", None)
    })
    @patch("crabquant.run.print_summary")
    @patch("crabquant.run.save_result")
    @patch("crabquant.run.print_result")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.load_data")
    def test_best_result_kept_when_all_fail(self, mock_load, mock_engine_cls, mock_print, mock_save, mock_summary):
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.run.return_value = _make_result(passed=False, sharpe=0.8)

        results = run_discovery(strategies=["s"], tickers=["SPY"], max_iterations=2)
        # best is set when sharpe > 0 and (best is None or sharpe > best.sharpe)
        assert len(results) == 1
        assert results[0].sharpe == 0.8

    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "s": (lambda df, p: (None, None), {}, {"a": [1]}, "desc", None)
    })
    @patch("crabquant.run.print_summary")
    @patch("crabquant.run.save_result")
    @patch("crabquant.run.print_result")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.load_data")
    def test_no_results_when_all_negative_sharpe(self, mock_load, mock_engine_cls, mock_print, mock_save, mock_summary):
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.run.return_value = _make_result(passed=False, sharpe=-1.0)

        results = run_discovery(strategies=["s"], tickers=["SPY"], max_iterations=1)
        assert results == []

    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "s1": (lambda df, p: (None, None), {}, {"a": [1]}, "s1 desc", None),
        "s2": (lambda df, p: (None, None), {}, {"b": [2]}, "s2 desc", None),
    })
    @patch("crabquant.run.print_summary")
    @patch("crabquant.run.save_result")
    @patch("crabquant.run.print_result")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.load_data")
    def test_multiple_strategies(self, mock_load, mock_engine_cls, mock_print, mock_save, mock_summary):
        mock_load.return_value = MagicMock()
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.run.return_value = _make_result(passed=True)

        results = run_discovery(strategies=["s1", "s2"], tickers=["SPY"], max_iterations=1)
        assert len(results) == 2


# ===========================================================================
# run_validation
# ===========================================================================

class TestRunValidation:
    def _make_wf(self, **kw):
        defaults = dict(
            train_sharpe=1.5, train_return=0.12, test_sharpe=1.0, test_return=0.08,
            degradation=0.33, robust=True, notes="ok",
        )
        defaults.update(kw)
        return types.SimpleNamespace(**defaults)

    def _make_ct(self, **kw):
        defaults = dict(
            tickers_tested=10, tickers_profitable=7, avg_sharpe=0.9,
            median_sharpe=0.8, sharpe_std=0.3, robust=True,
            win_rate_across_tickers=0.7,
        )
        defaults.update(kw)
        return types.SimpleNamespace(**defaults)

    @patch("crabquant.run.VALIDATION_DIR")
    @patch("crabquant.run.walk_forward_test")
    @patch("crabquant.run.cross_ticker_validation")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "macd": (lambda df, p: (None, None), {}, {}, "desc", None)
    })
    def test_basic_flow(self, mock_engine_cls, mock_ct, mock_wf, mock_val_dir, tmp_path):
        mock_val_dir.mkdir = MagicMock()
        mock_val_dir.__truediv__ = lambda self, x: tmp_path / x

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_wf.return_value = self._make_wf()
        mock_ct.return_value = self._make_ct()

        winner = _make_result(strategy_name="macd", ticker="SPY", params={"fast": 10})
        results = run_validation(winning_results=[winner])

        assert len(results) == 1
        assert results[0]["ticker"] == "SPY"
        assert results[0]["overall_robust"] is True

    @patch("crabquant.run.load_winners_from_log", return_value=[])
    @patch("crabquant.run.BacktestEngine")
    def test_no_winners_returns_empty(self, mock_engine_cls, mock_load):
        results = run_validation()
        assert results == []

    @patch("crabquant.run.load_winners_from_log")
    @patch("crabquant.run.VALIDATION_DIR")
    @patch("crabquant.run.walk_forward_test")
    @patch("crabquant.run.cross_ticker_validation")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.STRATEGY_REGISTRY", {})
    def test_unknown_strategy_skipped(self, mock_engine_cls, mock_ct, mock_wf, mock_val_dir, mock_load):
        winner = _make_result(strategy_name="ghost")
        mock_load.return_value = [winner]
        results = run_validation()
        assert results == []

    @patch("crabquant.run.VALIDATION_DIR")
    @patch("crabquant.run.walk_forward_test")
    @patch("crabquant.run.cross_ticker_validation")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "macd": (lambda df, p: (None, None), {}, {}, "desc", None)
    })
    def test_not_robust_when_wf_fails(self, mock_engine_cls, mock_ct, mock_wf, mock_val_dir, tmp_path):
        mock_val_dir.mkdir = MagicMock()
        mock_val_dir.__truediv__ = lambda self, x: tmp_path / x

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_wf.return_value = self._make_wf(robust=False)
        mock_ct.return_value = self._make_ct(robust=True)

        winner = _make_result(strategy_name="macd", ticker="SPY", params={})
        results = run_validation(winning_results=[winner])
        assert results[0]["overall_robust"] is False

    @patch("crabquant.run.VALIDATION_DIR")
    @patch("crabquant.run.walk_forward_test")
    @patch("crabquant.run.cross_ticker_validation")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "macd": (lambda df, p: (None, None), {}, {}, "desc", None)
    })
    def test_saves_validation_json(self, mock_engine_cls, mock_ct, mock_wf, mock_val_dir, tmp_path):
        mock_val_dir.mkdir = MagicMock()
        mock_val_dir.__truediv__ = lambda self, x: tmp_path / x

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_wf.return_value = self._make_wf()
        mock_ct.return_value = self._make_ct()

        winner = _make_result(strategy_name="macd", ticker="SPY", params={})
        run_validation(winning_results=[winner])

        out_file = tmp_path / "validation_results.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data) == 1

    @patch("crabquant.run.VALIDATION_DIR")
    @patch("crabquant.run.walk_forward_test")
    @patch("crabquant.run.cross_ticker_validation")
    @patch("crabquant.run.BacktestEngine")
    @patch("crabquant.run.STRATEGY_REGISTRY", {
        "s1": (lambda df, p: (None, None), {}, {}, "d1", None),
        "s2": (lambda df, p: (None, None), {}, {}, "d2", None),
    })
    def test_multiple_winners(self, mock_engine_cls, mock_ct, mock_wf, mock_val_dir, tmp_path):
        mock_val_dir.mkdir = MagicMock()
        mock_val_dir.__truediv__ = lambda self, x: tmp_path / x

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_wf.return_value = self._make_wf()
        mock_ct.return_value = self._make_ct()

        w1 = _make_result(strategy_name="s1", ticker="AAPL", params={})
        w2 = _make_result(strategy_name="s2", ticker="MSFT", params={})
        results = run_validation(winning_results=[w1, w2])
        assert len(results) == 2


# ===========================================================================
# main (CLI)
# ===========================================================================

class TestMain:
    @patch("crabquant.run.run_validation")
    @patch("crabquant.run.argparse.ArgumentParser.parse_args")
    def test_validate_flag(self, mock_parse, mock_run_val):
        mock_parse.return_value = types.SimpleNamespace(
            validate=True, strategy=None, ticker=None, tickers=None, iterations=5
        )
        main()
        mock_run_val.assert_called_once()

    @patch("crabquant.run.run_discovery")
    @patch("crabquant.run.argparse.ArgumentParser.parse_args")
    def test_strategy_flag(self, mock_parse, mock_run_disc):
        mock_parse.return_value = types.SimpleNamespace(
            validate=False, strategy="macd", ticker=None, tickers=None, iterations=3
        )
        main()
        mock_run_disc.assert_called_once_with(strategies=["macd"], tickers=None, max_iterations=3)

    @patch("crabquant.run.run_discovery")
    @patch("crabquant.run.argparse.ArgumentParser.parse_args")
    def test_ticker_flag(self, mock_parse, mock_run_disc):
        mock_parse.return_value = types.SimpleNamespace(
            validate=False, strategy=None, ticker="AAPL,MSFT", tickers=None, iterations=5
        )
        main()
        mock_run_disc.assert_called_once_with(
            strategies=None, tickers=["AAPL", "MSFT"], max_iterations=5
        )

    @patch("crabquant.run.run_discovery")
    @patch("crabquant.run.argparse.ArgumentParser.parse_args")
    def test_default_no_args(self, mock_parse, mock_run_disc):
        mock_parse.return_value = types.SimpleNamespace(
            validate=False, strategy=None, ticker=None, tickers=None, iterations=5
        )
        main()
        mock_run_disc.assert_called_once_with(max_iterations=5)

    @patch("crabquant.run.run_discovery")
    @patch("crabquant.run.argparse.ArgumentParser.parse_args")
    def test_iterations_flag(self, mock_parse, mock_run_disc):
        mock_parse.return_value = types.SimpleNamespace(
            validate=False, strategy=None, ticker=None, tickers=None, iterations=10
        )
        main()
        mock_run_disc.assert_called_once_with(max_iterations=10)


# ===========================================================================
# Edge cases & integration-style
# ===========================================================================

class TestEdgeCases:
    def test_mutate_params_empty_params_empty_grid(self):
        assert mutate_params({}, {}, 99) == {}

    def test_mutate_params_key_in_grid_not_in_params(self):
        grid = {"x": [10, 20, 30]}
        result = mutate_params({}, grid, 0)
        # defaults to values[0]=10, shift=1 → idx=1 → 20
        assert result["x"] == 20

    def test_sample_params_preserves_order(self):
        grid = {"z": [1], "a": [2], "m": [3]}
        result = sample_params(grid)
        assert list(result.keys()) == ["z", "a", "m"]

    def test_save_result_with_complex_params(self, tmp_path):
        r = _make_result(params={"nested": {"a": 1}, "list": [1, 2, 3]})
        save_result(r, tmp_path)
        data = json.loads((tmp_path / "backtest_log.jsonl").read_text().strip())
        assert data["params"]["nested"]["a"] == 1

    def test_print_summary_single_result(self, capsys, tmp_path):
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary([_make_result()])
        output = capsys.readouterr().out
        assert "1" in output  # total combos = 1

    def test_load_winners_from_log_with_valid_data(self, tmp_path):
        log = tmp_path / "backtest_log.jsonl"
        log.write_text('{"passed": true, "ticker": "A"}\n{"passed": false, "ticker": "B"}\n')
        with patch("crabquant.run.LOGS_DIR", tmp_path):
            winners = load_winners_from_log()
        # Only passed lines should load
        assert len(winners) == 1
        assert winners[0].ticker == "A"

    def test_print_summary_with_zero_trades_excluded(self, capsys, tmp_path):
        results = [_make_result(num_trades=0, sharpe=-1.0, passed=False)]
        with patch("crabquant.run.RESULTS_DIR", tmp_path):
            print_summary(results)
        output = capsys.readouterr().out
        assert "Valid results (trades > 0): 0" in output
