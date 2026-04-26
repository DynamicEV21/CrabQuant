"""End-to-end smoke test: single mandate, max_turns=2, easy sharpe_target.

Verifies the full pipeline loop: invents → validates → backtests → classifies → reports.
All external dependencies (LLM, data, backtest engine) are mocked.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root))


class TestE2ESmoke:
    """Smoke test: run refinement_loop end-to-end with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def setup_mandate(self, tmp_path):
        """Create a minimal mandate file."""
        mandate = {
            "name": "smoke_test",
            "tickers": ["AAPL"],
            "period": "1y",
            "primary_ticker": "AAPL",
        }
        self.mandate_path = tmp_path / "smoke_mandate.json"
        self.mandate_path.write_text(json.dumps(mandate))
        self.tmp_path = tmp_path

    def _make_mock_strategy_module(self):
        """Create a mock strategy module with required attributes."""
        mod = MagicMock()
        mod.generate_signals = MagicMock(return_value=(None, None))
        mod.DEFAULT_PARAMS = {"fast": 10, "slow": 20}
        mod.PARAM_GRID = {"fast": [5, 10, 15], "slow": [15, 20, 25]}
        mod.DESCRIPTION = "Smoke test strategy"
        return mod

    def _make_mock_backtest_result(self, sharpe=2.0):
        """Create a mock BacktestResult."""
        result = MagicMock()
        result.sharpe = sharpe
        result.total_return = sharpe * 10
        result.max_drawdown = 5.0
        result.win_rate = 0.6
        result.num_trades = 20
        result.profit_factor = 1.5
        result.calmar_ratio = sharpe * 0.8
        result.score = sharpe
        result.params = {"fast": 10, "slow": 20}
        result.sortino_ratio = sharpe * 0.9
        return result

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.save_report")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.run_full_validation")
    @patch("refinement_loop.check_guardrails")
    @patch("refinement_loop.classify_failure")
    @patch("refinement_loop.compute_sharpe_by_year")
    @patch("refinement_loop.run_backtest_safely")
    @patch("refinement_loop.run_validation_gates")
    @patch("refinement_loop.build_llm_context")
    @patch("refinement_loop.load_strategy_module")
    @patch("refinement_loop.acquire_lock")
    @patch("refinement_loop.call_llm_inventor")
    @patch("refinement_loop.clear_cache")
    def test_success_path(
        self, mock_clear, mock_inventor, mock_lock, mock_load_mod,
        mock_build_ctx, mock_gates, mock_backtest, mock_sharpe_yr,
        mock_classify, mock_guardrails, mock_validation, mock_write_json,
        mock_load_report, mock_save_report, mock_save_state,
        mock_promote, mock_release,
    ):
        """Full loop: LLM invents strategy → validate → backtest → success."""
        # Setup mocks
        mock_lock.return_value = True
        mock_clear.return_value = None
        mock_inventor.return_value = {
            "action": "novel",
            "hypothesis": "momentum breakout",
            "new_strategy_code": "def generate_signals(df, params): pass\nDEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'",
            "reasoning": "test",
            "addresses_failure": "",
            "expected_impact": "higher",
        }
        mock_build_ctx.return_value = {"mandate": "smoke", "context": "test"}
        mock_gates.return_value = (True, [])
        mock_load_mod.return_value = self._make_mock_strategy_module()

        mock_portfolio = MagicMock()
        mock_backtest.return_value = (self._make_mock_backtest_result(sharpe=2.0), MagicMock(), mock_portfolio)
        mock_sharpe_yr.return_value = {"2024": 2.0}
        mock_classify.return_value = ("none", {})
        mock_guardrails.return_value = MagicMock(violations=[], warnings=[])
        mock_validation.return_value = {"status": "ok"}
        mock_write_json.return_value = None
        mock_save_state.return_value = None
        mock_save_report.return_value = None
        mock_load_report.return_value = None

        # Run
        from refinement_loop import refinement_loop
        state = refinement_loop(str(self.mandate_path), max_turns=2, sharpe_target=0.1)

        # Verify success
        assert state.status == "success"
        assert state.best_sharpe >= 0.1
        assert state.current_turn == 1  # succeeded on first turn
        mock_inventor.assert_called_once()
        mock_gates.assert_called_once()
        mock_load_mod.assert_called_once()
        mock_backtest.assert_called_once()
        mock_promote.assert_called_once()

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_llm_failure_advances_turn(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """When LLM fails all attempts, the loop advances to the next turn."""
        mock_lock_ret = [True]
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        with patch("refinement_loop.acquire_lock", return_value=True) as mock_lock, \
             patch("refinement_loop.call_llm_inventor", return_value=None):

            from refinement_loop import refinement_loop
            state = refinement_loop(str(self.mandate_path), max_turns=2, sharpe_target=0.1)

        assert state.status == "max_turns_exhausted"
        assert state.current_turn == 2

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_gate_failure_advances_turn(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """When validation gates fail, the loop advances."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor", return_value={
                 "action": "novel", "hypothesis": "test",
                 "new_strategy_code": "bad code",
                 "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
             }), \
             patch("refinement_loop.run_validation_gates", return_value=(False, ["syntax error"])):

            from refinement_loop import refinement_loop
            state = refinement_loop(str(self.mandate_path), max_turns=2, sharpe_target=0.1)

        assert state.status == "max_turns_exhausted"

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_backtest_crash_handled(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """When backtest crashes (returns None), the loop advances."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor", return_value={
                 "action": "novel", "hypothesis": "test",
                 "new_strategy_code": "def generate_signals(df, params): pass\nDEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'",
                 "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
             }), \
             patch("refinement_loop.run_validation_gates", return_value=(True, [])), \
             patch("refinement_loop.load_strategy_module", return_value=self._make_mock_strategy_module()), \
             patch("refinement_loop.run_backtest_safely", return_value=None):

            from refinement_loop import refinement_loop
            state = refinement_loop(str(self.mandate_path), max_turns=2, sharpe_target=0.1)

        assert state.status == "max_turns_exhausted"
        # history should record backtest failure
        assert any("backtest" in str(h).lower() for h in state.history)
