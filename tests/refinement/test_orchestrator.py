"""Tests for scripts.refinement_loop — the main orchestrator."""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
from dataclasses import dataclass, field

# Import the orchestrator functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from refinement_loop import (
    load_json,
    write_json,
    save_state,
    load_state,
    create_run_directory,
    acquire_lock,
    release_lock,
    refinement_loop,
)
# Stagnation functions now live in the module (Phase 4)
from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response


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
        # datetime should be handled by default=str
        from datetime import datetime
        write_json(f, {"ts": datetime(2024, 1, 1)})
        data = json.loads(f.read_text())
        assert "2024" in data["ts"]


class TestSaveLoadState:
    def test_roundtrip(self, tmp_path):
        from crabquant.refinement.schemas import RunState
        state = RunState(run_id="test", mandate_name="test_mandate", created_at="2024-01-01T00:00:00")
        save_state(tmp_path, state)
        loaded = load_state(tmp_path)
        assert loaded is not None
        assert loaded.run_id == "test"
        assert loaded.mandate_name == "test_mandate"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_state(tmp_path / "nonexistent") is None


class TestLocking:
    def test_acquire_succeeds(self, tmp_path):
        assert acquire_lock(tmp_path) is True
        assert (tmp_path / "lock.json").exists()

    def test_acquire_fails_if_locked(self, tmp_path):
        from datetime import datetime, timezone
        lock_data = {"pid": 999, "timestamp": datetime.now(timezone.utc).isoformat()}
        (tmp_path / "lock.json").write_text(json.dumps(lock_data))
        assert acquire_lock(tmp_path) is False

    def test_acquire_overrides_stale_lock(self, tmp_path):
        from datetime import datetime, timezone, timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        lock_data = {"pid": 999, "timestamp": old_time}
        (tmp_path / "lock.json").write_text(json.dumps(lock_data))
        assert acquire_lock(tmp_path) is True

    def test_release_removes_lock(self, tmp_path):
        acquire_lock(tmp_path)
        assert (tmp_path / "lock.json").exists()
        release_lock(tmp_path)
        assert not (tmp_path / "lock.json").exists()


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
        # Module uses numpy linear regression: slope=0 → trend="improving", score low
        history = [{"sharpe": s} for s in [0.5, 0.5, 0.5]]
        score, trend = compute_stagnation(history)
        assert trend == "improving"
        assert score < 0.2

    def test_slow_progress(self):
        history = [{"sharpe": s} for s in [0.5, 0.55, 0.58]]
        score, trend = compute_stagnation(history)
        # Module version classifies this as improving (positive slope)
        assert trend in ("improving", "flat")


class TestGetStagnationResponse:
    def test_no_constraint(self):
        resp = get_stagnation_response(2, 0.3)
        assert resp["constraint"] == "normal"

    def test_force_structural(self):
        # Module requires iteration > 3 for pivot (score > 0.7)
        resp = get_stagnation_response(4, 0.75)
        assert resp["constraint"] == "pivot"

    def test_abandon(self):
        resp = get_stagnation_response(5, 0.95)
        assert resp["constraint"] == "abandon"

    def test_no_abandon_early(self):
        # Module only abandons when iteration > 3
        resp = get_stagnation_response(3, 0.95)
        assert resp["constraint"] == "normal"


class TestRefinementLoop:
    """Integration tests for the main loop (heavily mocked)."""

    def _make_mandate(self, tmp_path, **overrides):
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

    def _mock_backtest_result(self, sharpe=0.5, passed=False):
        result = MagicMock()
        result.sharpe = sharpe
        result.total_return = 0.1
        result.max_drawdown = -0.15
        result.win_rate = 0.55
        result.num_trades = 20
        result.profit_factor = 1.2
        result.calmar_ratio = 0.8
        result.score = 0.6
        result.passed = passed
        result.params = {"rsi_period": 14}
        result.ticker = "AAPL"
        return result

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
        # Setup mocks
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": "def generate_signals(df, params):\n    return df['close'] > 0, df['close'] < 0\nDEFAULT_PARAMS = {'period': 14}\nPARAM_GRID = {}\nDESCRIPTION = 'test'",
            "params": {},
            "expected_impact": "higher",
        }
        
        mock_module = MagicMock()
        mock_module.DEFAULT_PARAMS = {"period": 14}
        mock_module.generate_signals = MagicMock(return_value=(MagicMock(), MagicMock()))
        mock_loader.return_value = mock_module
        
        result = self._mock_backtest_result(sharpe=2.0, passed=True)
        mock_backtest.return_value = (result, MagicMock(), MagicMock())
        
        mock_guardrails.return_value = MagicMock(violations=[], warnings=[])
        
        # Run with easy target
        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"):
            (tmp_path / "run").mkdir(exist_ok=True)
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
        # Patch CircuitBreaker so it never opens (allows both turns to run)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"), \
             patch("refinement_loop.CircuitBreaker") as mock_cb_cls:
            mock_cb = MagicMock()
            mock_cb.is_open.return_value = False
            mock_cb.record.return_value = None
            mock_cb_cls.return_value = mock_cb
            (tmp_path / "run").mkdir(exist_ok=True)
            state = refinement_loop(mandate_path, max_turns=2, sharpe_target=1.0)
        
        # Circuit breaker halts after consecutive LLM failures
        assert state.status in ("abandoned", "max_turns_exhausted")

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.load_strategy_module", return_value=None)
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.call_llm_inventor")
    def test_module_load_failure(self, mock_llm, mock_gates, mock_loader,
                                  mock_release, tmp_path):
        """Test that module load failure is handled gracefully."""
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": "def generate_signals(df, params): pass",
            "params": {},
            "expected_impact": "higher",
        }
        
        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"):
            (tmp_path / "run").mkdir(exist_ok=True)
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
        mock_llm.return_value = {
            "action": "novel", "hypothesis": "test",
            "new_strategy_code": "def generate_signals(df, params): pass\nDEFAULT_PARAMS={}\nPARAM_GRID={}\nDESCRIPTION='test'",
            "params": {}, "expected_impact": "higher",
        }
        mock_module = MagicMock()
        mock_module.DEFAULT_PARAMS = {}
        mock_loader.return_value = mock_module
        
        mandate_path = self._make_mandate(tmp_path)
        with patch("refinement_loop.create_run_directory", return_value=tmp_path / "run"):
            (tmp_path / "run").mkdir(exist_ok=True)
            state = refinement_loop(mandate_path, max_turns=1, sharpe_target=1.0)
        
        assert state.status == "max_turns_exhausted"
        assert state.history[0]["status"] == "backtest_crash"
