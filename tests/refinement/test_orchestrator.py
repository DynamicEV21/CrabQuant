"""Tests for scripts.refinement_loop — the main orchestrator."""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

# Import the orchestrator functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from refinement_loop import (
    load_json,
    write_json,
    save_state,
    load_state,
    save_report,
    load_report,
    create_run_directory,
    acquire_lock,
    release_lock,
    run_full_validation,
    clear_cache,
    _build_retry_feedback,
    _make_result_proxy,
    _promote_post_loop,
    _write_dashboard,
    refinement_loop,
)
# Stagnation functions now live in the module (Phase 4)
from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response
from crabquant.refinement.schemas import RunState, BacktestReport, StrategyModification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy_code():
    return ("def generate_signals(df, params):\n"
            "    return df['close'] > 0, df['close'] < 0\n"
            "DEFAULT_PARAMS = {'period': 14}\n"
            "PARAM_GRID = {}\n"
            "DESCRIPTION = 'test'")


def _make_mandate_file(tmp_path, **overrides):
    mandate = {
        "name": "test_mandate",
        "tickers": ["AAPL"],
        "primary_ticker": "AAPL",
        "period": "1y",
        "strategy_archetype": "momentum",
    }
    mandate.update(overrides)
    path = tmp_path / "mandate.json"
    path.write_text(json.dumps(mandate))
    return str(path)


def _mock_backtest_result(sharpe=0.5, passed=False, num_trades=20):
    result = MagicMock()
    result.sharpe = sharpe
    result.total_return = 0.1
    result.max_drawdown = -0.15
    result.win_rate = 0.55
    result.num_trades = num_trades
    result.profit_factor = 1.2
    result.calmar_ratio = 0.8
    result.sortino_ratio = 0.9
    result.score = 0.6
    result.passed = passed
    result.params = {"rsi_period": 14}
    result.ticker = "AAPL"
    return result


def _mock_llm_response(action="novel", code=None):
    return {
        "action": action,
        "hypothesis": "test hypothesis",
        "new_strategy_code": code or _make_strategy_code(),
        "params": {},
        "expected_impact": "higher",
        "reasoning": "test reasoning",
        "addresses_failure": "",
    }


def _setup_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return run_dir


# ===========================================================================
# TestLoadJson
# ===========================================================================

class TestLoadJson:
    def test_loads_dict(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        result = load_json(str(f))
        assert result == {"key": "value"}

    def test_loads_nested(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"a": {"b": 1}}')
        result = load_json(str(f))
        assert result["a"]["b"] == 1

    def test_loads_list(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('[1, 2, 3]')
        result = load_json(str(f))
        assert result == [1, 2, 3]

    def test_loads_empty_object(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{}')
        result = load_json(str(f))
        assert result == {}

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(str(tmp_path / "nonexistent.json"))

    def test_raises_on_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_json(str(f))


# ===========================================================================
# TestWriteJson
# ===========================================================================

class TestWriteJson:
    def test_writes_json(self, tmp_path):
        f = tmp_path / "out.json"
        write_json(f, {"x": 1})
        assert json.loads(f.read_text()) == {"x": 1}

    def test_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "sub" / "dir" / "out.json"
        write_json(f, {"y": 2})
        assert f.exists()

    def test_handles_non_serializable(self, tmp_path):
        f = tmp_path / "out.json"
        write_json(f, {"ts": datetime(2024, 1, 1)})
        data = json.loads(f.read_text())
        assert "2024" in data["ts"]

    def test_writes_nested_structures(self, tmp_path):
        f = tmp_path / "out.json"
        data = {"a": [1, 2], "b": {"c": None}}
        write_json(f, data)
        assert json.loads(f.read_text()) == data

    def test_overwrites_existing(self, tmp_path):
        f = tmp_path / "out.json"
        write_json(f, {"v": 1})
        write_json(f, {"v": 2})
        assert json.loads(f.read_text())["v"] == 2


# ===========================================================================
# TestSaveLoadState
# ===========================================================================

class TestSaveLoadState:
    def test_roundtrip(self, tmp_path):
        state = RunState(run_id="test", mandate_name="test_mandate", created_at="2024-01-01T00:00:00")
        save_state(tmp_path, state)
        loaded = load_state(tmp_path)
        assert loaded is not None
        assert loaded.run_id == "test"
        assert loaded.mandate_name == "test_mandate"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_state(tmp_path / "nonexistent") is None

    def test_roundtrip_preserves_history(self, tmp_path):
        state = RunState(
            run_id="test", mandate_name="test_mandate", created_at="2024-01-01T00:00:00",
            history=[{"turn": 1, "sharpe": 0.5}],
            best_sharpe=0.5,
            best_turn=1,
        )
        save_state(tmp_path, state)
        loaded = load_state(tmp_path)
        assert loaded.history == [{"turn": 1, "sharpe": 0.5}]
        assert loaded.best_sharpe == 0.5

    def test_roundtrip_all_fields(self, tmp_path):
        state = RunState(
            run_id="r1", mandate_name="m1", created_at="2024-06-01T00:00:00",
            max_turns=5, sharpe_target=2.0, tickers=["TSLA"], period="1y",
            current_turn=3, status="running", best_sharpe=1.2,
            best_composite_score=1.0, best_turn=2, best_code_path="/tmp/strat.py",
            history=[{"turn": 1}, {"turn": 2}],
        )
        save_state(tmp_path, state)
        loaded = load_state(tmp_path)
        assert loaded.run_id == "r1"
        assert loaded.max_turns == 5
        assert loaded.sharpe_target == 2.0
        assert loaded.tickers == ["TSLA"]
        assert loaded.period == "1y"
        assert loaded.current_turn == 3
        assert loaded.status == "running"
        assert loaded.best_composite_score == 1.0
        assert loaded.best_code_path == "/tmp/strat.py"
        assert len(loaded.history) == 2

    def test_load_corrupt_json_returns_none(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("{{{invalid json")
        assert load_state(tmp_path) is None


# ===========================================================================
# TestSaveLoadReport
# ===========================================================================

class TestSaveLoadReport:
    def test_roundtrip(self, tmp_path):
        report = BacktestReport(
            strategy_id="test_run", iteration=1,
            sharpe_ratio=1.5, total_return_pct=0.1, max_drawdown_pct=-0.1,
            win_rate=0.6, total_trades=30, profit_factor=1.3,
            calmar_ratio=0.9, sortino_ratio=1.0, expected_value=0.0, composite_score=0.7,
            failure_mode="low_sharpe", failure_details="below target",
            sharpe_by_year={"2024": 1.5}, stagnation_score=0.1,
            stagnation_trend="improving", previous_sharpes=[], previous_actions=[],
            guardrail_violations=[], guardrail_warnings=[],
            regime_sharpe=None, regime_regime_shift=None, top_drawdowns=None,
            portfolio_correlation=None, benchmark_return_pct=None, market_regime=None,
            current_strategy_code="pass", current_params={}, previous_attempts=[],
        )
        save_report(tmp_path, 1, report)
        loaded = load_report(tmp_path, 1)
        assert loaded is not None
        assert loaded.sharpe_ratio == 1.5
        assert loaded.total_trades == 30

    def test_load_missing_returns_none(self, tmp_path):
        assert load_report(tmp_path, 99) is None

    def test_load_corrupt_returns_none(self, tmp_path):
        report_file = tmp_path / "report_v1.json"
        report_file.write_text("bad json")
        assert load_report(tmp_path, 1) is None


# ===========================================================================
# TestLocking
# ===========================================================================

class TestLocking:
    def test_acquire_succeeds(self, tmp_path):
        assert acquire_lock(tmp_path) is True
        assert (tmp_path / "lock.json").exists()

    def test_acquire_fails_if_locked(self, tmp_path):
        lock_data = {"pid": 999, "timestamp": datetime.now(timezone.utc).isoformat()}
        (tmp_path / "lock.json").write_text(json.dumps(lock_data))
        assert acquire_lock(tmp_path) is False

    def test_acquire_overrides_stale_lock(self, tmp_path):
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        lock_data = {"pid": 999, "timestamp": old_time}
        (tmp_path / "lock.json").write_text(json.dumps(lock_data))
        assert acquire_lock(tmp_path) is True

    def test_release_removes_lock(self, tmp_path):
        acquire_lock(tmp_path)
        assert (tmp_path / "lock.json").exists()
        release_lock(tmp_path)
        assert not (tmp_path / "lock.json").exists()

    def test_release_nonexistent_lock(self, tmp_path):
        # Should not raise
        release_lock(tmp_path)
        assert not (tmp_path / "lock.json").exists()

    def test_acquire_creates_valid_json(self, tmp_path):
        acquire_lock(tmp_path)
        lock_data = json.loads((tmp_path / "lock.json").read_text())
        assert "pid" in lock_data
        assert "timestamp" in lock_data

    def test_acquire_stale_lock_boundary(self, tmp_path):
        # Just under 1 hour — should still be locked
        time_59m = (datetime.now(timezone.utc) - timedelta(minutes=59)).isoformat()
        (tmp_path / "lock.json").write_text(json.dumps({"pid": 999, "timestamp": time_59m}))
        assert acquire_lock(tmp_path) is False

    def test_acquire_corrupt_lock(self, tmp_path):
        (tmp_path / "lock.json").write_text("not json")
        # Corrupt lock should be overridden (bare except in acquire_lock passes)
        assert acquire_lock(tmp_path) is True


# ===========================================================================
# TestCreateRunDirectory
# ===========================================================================

class TestCreateRunDirectory:
    def test_creates_directory(self, tmp_path):
        with patch("refinement_loop.project_root", tmp_path):
            d = create_run_directory({"name": "test_mandate"})
            assert d.exists()
            assert "test_mandate" in d.name

    def test_creates_with_timestamp(self, tmp_path):
        with patch("refinement_loop.project_root", tmp_path):
            d = create_run_directory({"name": "test"})
            assert d.name.startswith("test_")

    def test_replaces_spaces_in_name(self, tmp_path):
        with patch("refinement_loop.project_root", tmp_path):
            d = create_run_directory({"name": "My Strategy"})
            assert "my_strategy" in d.name

    def test_uses_unknown_for_missing_name(self, tmp_path):
        with patch("refinement_loop.project_root", tmp_path):
            d = create_run_directory({})
            assert "unknown" in d.name


# ===========================================================================
# TestComputeStagnation
# ===========================================================================

class TestComputeStagnation:
    def test_empty_history(self):
        score, trend = compute_stagnation([])
        assert score == 0.0

    def test_single_entry(self):
        score, trend = compute_stagnation([{"sharpe": 0.5}])
        assert score == 0.0

    def test_improving(self):
        history = [{"sharpe": s} for s in [0.1, 0.3, 0.6]]
        score, trend = compute_stagnation(history)
        assert trend == "improving"
        assert score < 0.5

    def test_flat(self):
        history = [{"sharpe": s} for s in [0.5, 0.5, 0.5]]
        score, trend = compute_stagnation(history)
        assert trend == "improving"
        assert score < 0.2

    def test_slow_progress(self):
        history = [{"sharpe": s} for s in [0.5, 0.55, 0.58]]
        score, trend = compute_stagnation(history)
        assert trend in ("improving", "flat")

    def test_declining(self):
        history = [{"sharpe": s} for s in [0.8, 0.4, 0.1]]
        score, trend = compute_stagnation(history)
        assert trend == "declining"
        assert score > 0.5

    def test_two_entries_improving(self):
        score, trend = compute_stagnation([{"sharpe": 0.3}, {"sharpe": 0.8}])
        assert trend == "improving"

    def test_no_sharpe_key(self):
        # Entries without 'sharpe' should not crash
        score, trend = compute_stagnation([{"turn": 1}, {"turn": 2}])
        # Depends on implementation — should not crash
        assert isinstance(score, float)


# ===========================================================================
# TestGetStagnationResponse
# ===========================================================================

class TestGetStagnationResponse:
    def test_no_constraint(self):
        resp = get_stagnation_response(2, 0.3)
        assert resp["constraint"] == "normal"

    def test_force_structural(self):
        resp = get_stagnation_response(4, 0.75)
        assert resp["constraint"] == "pivot"

    def test_abandon(self):
        resp = get_stagnation_response(5, 0.95)
        assert resp["constraint"] == "abandon"

    def test_no_abandon_early(self):
        resp = get_stagnation_response(3, 0.95)
        assert resp["constraint"] == "normal"

    def test_low_iteration_moderate_score(self):
        resp = get_stagnation_response(1, 0.5)
        assert resp["constraint"] == "normal"

    def test_response_has_prompt_suffix(self):
        resp = get_stagnation_response(3, 0.5)
        assert "prompt_suffix" in resp


# ===========================================================================
# TestRunFullValidation
# ===========================================================================

class TestRunFullValidation:
    def test_returns_ok_on_success(self, tmp_path):
        state = RunState(
            run_id="test", mandate_name="test", created_at="2024-01-01T00:00:00",
            best_code_path=str(tmp_path / "strategy.py"),
        )
        (tmp_path / "strategy.py").write_text(_make_strategy_code())
        result = run_full_validation(state, tmp_path)
        assert result["status"] == "ok"

    def test_returns_error_on_missing_file(self):
        state = RunState(
            run_id="test", mandate_name="test", created_at="2024-01-01T00:00:00",
            best_code_path="/nonexistent/strategy.py",
        )
        result = run_full_validation(state, Path("/tmp"))
        assert result["status"] == "error"


# ===========================================================================
# TestBuildRetryFeedback
# ===========================================================================

class TestBuildRetryFeedback:
    def test_empty_errors(self):
        fb = _build_retry_feedback([])
        assert "no specific error" in fb.lower()

    def test_syntax_error(self):
        fb = _build_retry_feedback(["SyntaxError: invalid syntax"])
        assert "SyntaxError" in fb
        assert "syntax" in fb.lower()

    def test_import_error(self):
        fb = _build_retry_feedback(["ImportError: No module named 'foo'"])
        assert "ImportError" in fb

    def test_signal_error(self):
        fb = _build_retry_feedback(["zero entry signals detected"])
        assert "zero entry" in fb

    def test_backtest_error(self):
        fb = _build_retry_feedback(["backtest failed: column not found"])
        assert "backtest" in fb.lower()

    def test_generic_error(self):
        fb = _build_retry_feedback(["something unexpected happened"])
        assert "something unexpected happened" in fb

    def test_multiple_errors(self):
        fb = _build_retry_feedback(["SyntaxError: bad", "zero entry signals"])
        assert "2 error" in fb


# ===========================================================================
# TestMakeResultProxy
# ===========================================================================

class TestMakeResultProxy:
    def test_with_report(self):
        state = RunState(
            run_id="test", mandate_name="test_mandate", created_at="2024-01-01T00:00:00",
            best_sharpe=1.5, tickers=["AAPL"], best_turn=3,
        )
        report = BacktestReport(
            strategy_id="test", iteration=1,
            sharpe_ratio=1.5, total_return_pct=0.1, max_drawdown_pct=-0.1,
            win_rate=0.6, total_trades=25, profit_factor=1.2,
            calmar_ratio=0.8, sortino_ratio=0.9, expected_value=0.0, composite_score=0.7,
            failure_mode="", failure_details="",
            sharpe_by_year={}, stagnation_score=0, stagnation_trend="",
            previous_sharpes=[], previous_actions=[],
            guardrail_violations=[], guardrail_warnings=[],
            regime_sharpe=None, regime_regime_shift=None, top_drawdowns=None,
            portfolio_correlation=None, benchmark_return_pct=None, market_regime=None,
            current_strategy_code="", current_params={"p": 1}, previous_attempts=[],
        )
        proxy = _make_result_proxy(state, report)
        assert proxy.sharpe == 1.5
        assert proxy.total_return == 0.1
        assert proxy.max_drawdown == -0.1
        assert proxy.num_trades == 25
        assert proxy.params == {"p": 1}

    def test_without_report(self):
        state = RunState(
            run_id="test", mandate_name="test_mandate", created_at="2024-01-01T00:00:00",
            best_sharpe=1.0, tickers=["SPY"], best_turn=2,
        )
        proxy = _make_result_proxy(state, None)
        assert proxy.sharpe == 1.0
        assert proxy.total_return == 0.0
        assert proxy.max_drawdown == 0.0
        assert proxy.num_trades == 0
        assert proxy.params == {}
        assert proxy.passed is True
        assert proxy.ticker == "SPY"

    def test_strategy_name_format(self):
        state = RunState(
            run_id="test", mandate_name="my_strat", created_at="2024-01-01T00:00:00",
            tickers=["AAPL"],
        )
        proxy = _make_result_proxy(state, None)
        assert proxy.strategy_name == "refined_my_strat"


# ===========================================================================
# TestClearCache
# ===========================================================================

class TestClearCache:
    def test_does_not_raise(self):
        clear_cache()  # Should be a no-op


# ===========================================================================
# TestWriteDashboard
# ===========================================================================

class TestWriteDashboard:
    def test_does_not_raise_on_missing_dir(self, tmp_path):
        # Should not raise even if run_dir doesn't exist
        _write_dashboard(tmp_path / "nonexistent")

    @patch("refinement_loop.generate_dashboard", side_effect=Exception("fail"))
    def test_does_not_raise_on_dashboard_error(self, mock_dash, tmp_path):
        _write_dashboard(tmp_path)  # Should swallow the exception


# ===========================================================================
# TestRefinementLoop — integration tests (heavily mocked)
# ===========================================================================

class TestRefinementLoop:
    """Integration tests for the main loop (heavily mocked)."""

    def _make_mandate(self, tmp_path, **overrides):
        return _make_mandate_file(tmp_path, **overrides)

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.run_full_validation", return_value={"status": "ok"})
    @patch("refinement_loop.check_guardrails")
    @patch("refinement_loop.classify_failure", return_value=("low_sharpe", "below target"))
    @patch("refinement_loop.compute_sharpe_by_year", return_value={"2024": 0.5})
    @patch("refinement_loop.run_backtest_safely")
    @patch("refinement_loop.load_strategy_module")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.call_llm_inventor")
    def test_success_path(self, mock_llm, mock_gates, mock_loader,
                          mock_backtest, mock_sharpe_year, mock_classify,
                          mock_guardrails, mock_validate, mock_promote,
                          mock_release, tmp_path):
        """Test that loop succeeds when Sharpe target is hit."""
        mock_llm.return_value = _mock_llm_response()
        mock_module = MagicMock()
        mock_module.DEFAULT_PARAMS = {"period": 14}
        mock_module.generate_signals = MagicMock(return_value=(MagicMock(), MagicMock()))
        mock_loader.return_value = mock_module
        result = _mock_backtest_result(sharpe=2.0, passed=True)
        mock_backtest.return_value = (result, MagicMock(), MagicMock(), None)
        mock_guardrails.return_value = MagicMock(violations=[], warnings=[])

        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"):
            _setup_run_dir(tmp_path)
            state = refinement_loop(mandate_path, max_turns=2, sharpe_target=1.0)

        assert state.status == "success"
        assert state.best_sharpe == 2.0

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.call_llm_inventor", return_value=None)
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    def test_llm_failure_advances_turn(self, mock_gates, mock_llm,
                                        mock_release, tmp_path):
        """Test that LLM failure doesn't crash the loop."""
        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"), \
             patch("refinement_loop.CircuitBreaker") as mock_cb_cls:
            mock_cb = MagicMock()
            mock_cb.is_open.return_value = False
            mock_cb.record.return_value = None
            mock_cb_cls.return_value = mock_cb
            _setup_run_dir(tmp_path)
            state = refinement_loop(mandate_path, max_turns=2, sharpe_target=1.0)

        assert state.status in ("abandoned", "max_turns_exhausted")

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.load_strategy_module", return_value=None)
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.call_llm_inventor")
    def test_module_load_failure(self, mock_llm, mock_gates, mock_loader,
                                  mock_release, tmp_path):
        """Test that module load failure is handled gracefully."""
        mock_llm.return_value = _mock_llm_response()
        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"):
            _setup_run_dir(tmp_path)
            state = refinement_loop(mandate_path, max_turns=2, sharpe_target=1.0)

        assert state.status == "max_turns_exhausted"

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.run_backtest_safely", return_value=None)
    @patch("refinement_loop.load_strategy_module")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.call_llm_inventor")
    def test_backtest_crash_handled(self, mock_llm, mock_gates, mock_loader,
                                     mock_backtest, mock_release, tmp_path):
        """Test that backtest crash is handled gracefully."""
        mock_llm.return_value = _mock_llm_response()
        mock_module = MagicMock()
        mock_module.DEFAULT_PARAMS = {}
        mock_loader.return_value = mock_module

        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"):
            _setup_run_dir(tmp_path)
            state = refinement_loop(mandate_path, max_turns=1, sharpe_target=1.0)

        assert state.status == "max_turns_exhausted"
        assert state.history[0]["status"] == "backtest_crash"

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.load_strategy_module")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.call_llm_inventor")
    def test_locked_run_returns_early(self, mock_llm, mock_gates, mock_loader,
                                       mock_release, tmp_path):
        """Test that a locked run directory returns immediately."""
        mandate_path = self._make_mandate(tmp_path)
        run_dir = _setup_run_dir(tmp_path)
        # Pre-create a lock
        lock_data = {"pid": 999, "timestamp": datetime.now(timezone.utc).isoformat()}
        (run_dir / "lock.json").write_text(json.dumps(lock_data))

        with patch("refinement_loop.create_run_directory", return_value=run_dir):
            state = refinement_loop(mandate_path, max_turns=2, sharpe_target=1.0)

        assert state.status == "pending"
        mock_llm.assert_not_called()

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.check_guardrails")
    @patch("refinement_loop.run_backtest_safely")
    @patch("refinement_loop.load_strategy_module")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.call_llm_inventor")
    def test_circuit_breaker_halts_loop(self, mock_llm, mock_gates, mock_loader,
                                         mock_backtest, mock_guardrails,
                                         mock_release, tmp_path):
        """Test that circuit breaker opening halts the loop."""
        mock_llm.return_value = _mock_llm_response()
        mock_module = MagicMock()
        mock_module.DEFAULT_PARAMS = {}
        mock_loader.return_value = mock_module
        mock_backtest.return_value = (
            _mock_backtest_result(sharpe=0.3), MagicMock(), MagicMock(), None
        )
        mock_guardrails.return_value = MagicMock(violations=[], warnings=[])

        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"), \
             patch("refinement_loop.CircuitBreaker") as mock_cb_cls:
            mock_cb = MagicMock()
            # Open after turn 1
            mock_cb.is_open.side_effect = [False, True]
            mock_cb.record.return_value = None
            mock_cb.summary.return_value = "pass rate too low"
            mock_cb.pass_rate = 0.0
            mock_cb_cls.return_value = mock_cb
            _setup_run_dir(tmp_path)
            state = refinement_loop(mandate_path, max_turns=3, sharpe_target=2.0)

        assert state.status == "abandoned"
        assert any(h.get("status") == "circuit_breaker_open" for h in state.history)

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.check_guardrails")
    @patch("refinement_loop.classify_failure", return_value=("low_sharpe", "below target"))
    @patch("refinement_loop.compute_sharpe_by_year", return_value={})
    @patch("refinement_loop.run_backtest_safely")
    @patch("refinement_loop.load_strategy_module")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.call_llm_inventor")
    def test_max_turns_exhausted(self, mock_llm, mock_gates, mock_loader,
                                  mock_backtest, mock_sharpe, mock_classify,
                                  mock_guardrails, mock_promote, mock_release,
                                  tmp_path):
        """Test loop exhausting all turns without hitting target."""
        mock_llm.return_value = _mock_llm_response()
        mock_module = MagicMock()
        mock_module.DEFAULT_PARAMS = {}
        mock_loader.return_value = mock_module
        mock_backtest.return_value = (
            _mock_backtest_result(sharpe=0.3), MagicMock(), MagicMock(), None
        )
        mock_guardrails.return_value = MagicMock(violations=[], warnings=[])

        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"):
            _setup_run_dir(tmp_path)
            state = refinement_loop(mandate_path, max_turns=2, sharpe_target=2.0)

        assert state.status == "max_turns_exhausted"

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.run_backtest_safely")
    @patch("refinement_loop.load_strategy_module")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.call_llm_inventor")
    def test_gate_failure_advances_turn(self, mock_llm, mock_gates, mock_loader,
                                         mock_backtest, mock_release, tmp_path):
        """Test that validation gate failure is handled gracefully."""
        mock_llm.return_value = _mock_llm_response()
        mock_module = MagicMock()
        mock_module.DEFAULT_PARAMS = {}
        mock_loader.return_value = mock_module
        mock_gates.return_value = (False, ["SyntaxError: bad syntax"])

        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"), \
             patch("refinement_loop.CircuitBreaker") as mock_cb_cls:
            mock_cb = MagicMock()
            mock_cb.is_open.return_value = False
            mock_cb.record.return_value = None
            mock_cb_cls.return_value = mock_cb
            _setup_run_dir(tmp_path)
            state = refinement_loop(mandate_path, max_turns=1, sharpe_target=1.0)

        assert state.status == "max_turns_exhausted"
        assert any(h.get("status") == "code_generation_failed" for h in state.history)

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.run_full_validation", return_value={"status": "ok", "passed": True})
    @patch("refinement_loop.auto_promote", return_value={"registered": True, "strategy_name": "test_strat"})
    @patch("refinement_loop.check_guardrails")
    @patch("refinement_loop.classify_failure", return_value=("", ""))
    @patch("refinement_loop.compute_sharpe_by_year", return_value={"2024": 2.0})
    @patch("refinement_loop.run_backtest_safely")
    @patch("refinement_loop.load_strategy_module")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.call_llm_inventor")
    def test_auto_promote_on_success(self, mock_llm, mock_gates, mock_loader,
                                      mock_backtest, mock_sharpe, mock_classify,
                                      mock_guardrails, mock_auto, mock_validate,
                                      mock_promote, mock_release, tmp_path):
        """Test that auto_promote is called when validation passes."""
        mock_llm.return_value = _mock_llm_response()
        mock_module = MagicMock()
        mock_module.DEFAULT_PARAMS = {"period": 14}
        mock_module.generate_signals = MagicMock(return_value=(MagicMock(), MagicMock()))
        mock_loader.return_value = mock_module
        result = _mock_backtest_result(sharpe=2.0, passed=True, num_trades=25)
        mock_backtest.return_value = (result, MagicMock(), MagicMock(), None)
        mock_guardrails.return_value = MagicMock(violations=[], warnings=[])

        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"), \
             patch("refinement_loop.run_full_validation_check", return_value={"status": "ok", "passed": True}):
            _setup_run_dir(tmp_path)
            state = refinement_loop(mandate_path, max_turns=2, sharpe_target=1.0)

        assert state.status == "success"
        mock_auto.assert_called_once()
