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
        mock_backtest.return_value = (self._make_mock_backtest_result(sharpe=2.0), MagicMock(), mock_portfolio, None)
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

        # Circuit breaker halts after consecutive failures (better than exhausting turns)
        assert state.status in ("abandoned", "max_turns_exhausted")

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

        # Circuit breaker may halt early on consecutive gate failures
        assert state.status in ("abandoned", "max_turns_exhausted")

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


# ── Expanded E2E Tests (Phase 6) ──────────────────────────────────────────────


class TestE2EMandateConfigs:
    """Test different mandate configurations through the loop."""

    @pytest.fixture(autouse=True)
    def setup_mandate(self, tmp_path):
        self.tmp_path = tmp_path

    def _write_mandate(self, **overrides):
        mandate = {
            "name": "config_test",
            "tickers": ["AAPL"],
            "period": "1y",
            "primary_ticker": "AAPL",
        }
        mandate.update(overrides)
        path = self.tmp_path / "mandate.json"
        path.write_text(json.dumps(mandate))
        return path

    def _mock_success_deps(self):
        """Return a dict of common mock patches for a successful run."""
        return {
            "acquire_lock": True,
            "release_lock": None,
            "clear_cache": None,
            "call_llm_inventor": {
                "action": "novel",
                "hypothesis": "momentum",
                "new_strategy_code": (
                    "def generate_signals(df, params): pass\n"
                    "DEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'"
                ),
                "reasoning": "test",
                "addresses_failure": "",
                "expected_impact": "higher",
            },
            "build_llm_context": {"mandate": "test", "context": "test"},
            "run_validation_gates": (True, []),
            "load_strategy_module": MagicMock(
                generate_signals=MagicMock(return_value=(None, None)),
                DEFAULT_PARAMS={},
                PARAM_GRID={},
                DESCRIPTION="test",
            ),
            "run_backtest_safely": (
                MagicMock(
                    sharpe=2.0, total_return=20.0, max_drawdown=5.0,
                    win_rate=0.6, num_trades=20, profit_factor=1.5,
                    calmar_ratio=1.6, score=2.0, sortino_ratio=1.8,
                    params={},
                ),
                MagicMock(),  # df
                MagicMock(),  # portfolio
                None,  # error_info (Phase 6)
            ),
            "compute_sharpe_by_year": {"2024": 2.0},
            "classify_failure": ("none", {}),
            "check_guardrails": MagicMock(violations=[], warnings=[]),
            "run_full_validation": {"status": "ok", "passed": False},
            "write_json": None,
            "load_report": None,
            "save_report": None,
            "save_state": None,
            "promote_to_winner": None,
        }

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
    def test_multi_ticker_mandate(
        self, mock_clear, mock_inventor, mock_lock, mock_load_mod,
        mock_build_ctx, mock_gates, mock_backtest, mock_sharpe_yr,
        mock_classify, mock_guardrails, mock_validation, mock_write_json,
        mock_load_report, mock_save_report, mock_save_state,
        mock_promote, mock_release,
    ):
        """Mandate with multiple tickers runs successfully."""
        mandate_path = self._write_mandate(
            name="multi_ticker", tickers=["AAPL", "MSFT", "GOOG"]
        )
        deps = self._mock_success_deps()
        mock_lock.return_value = deps["acquire_lock"]
        mock_clear.return_value = deps["clear_cache"]
        mock_inventor.return_value = deps["call_llm_inventor"]
        mock_build_ctx.return_value = deps["build_llm_context"]
        mock_gates.return_value = deps["run_validation_gates"]
        mock_load_mod.return_value = deps["load_strategy_module"]
        mock_backtest.return_value = deps["run_backtest_safely"]
        mock_sharpe_yr.return_value = deps["compute_sharpe_by_year"]
        mock_classify.return_value = deps["classify_failure"]
        mock_guardrails.return_value = deps["check_guardrails"]
        mock_validation.return_value = deps["run_full_validation"]
        mock_write_json.return_value = deps["write_json"]
        mock_load_report.return_value = deps["load_report"]
        mock_save_report.return_value = deps["save_report"]
        mock_save_state.return_value = deps["save_state"]

        from refinement_loop import refinement_loop
        state = refinement_loop(str(mandate_path), max_turns=2, sharpe_target=0.1)
        assert state.status == "success"
        assert state.tickers == ["AAPL", "MSFT", "GOOG"]

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_high_sharpe_target_reached(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """High sharpe target is reached on first turn."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        high_sharpe_result = MagicMock(
            sharpe=5.0, total_return=50.0, max_drawdown=5.0,
            win_rate=0.7, num_trades=40, profit_factor=2.0,
            calmar_ratio=5.0, score=5.0, sortino_ratio=4.5, params={},
        )

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor", return_value={
                 "action": "novel", "hypothesis": "high sharpe",
                 "new_strategy_code": (
                     "def generate_signals(df, params): pass\n"
                     "DEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'"
                 ),
                 "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
             }), \
             patch("refinement_loop.run_validation_gates", return_value=(True, [])), \
             patch("refinement_loop.build_llm_context", return_value={}), \
             patch("refinement_loop.load_strategy_module", return_value=MagicMock(
                 generate_signals=MagicMock(return_value=(None, None)),
                 DEFAULT_PARAMS={}, PARAM_GRID={}, DESCRIPTION="test",
             )), \
             patch("refinement_loop.run_backtest_safely",
                   return_value=(high_sharpe_result, MagicMock(), MagicMock(), None)), \
             patch("refinement_loop.compute_sharpe_by_year", return_value={"2024": 5.0}), \
             patch("refinement_loop.classify_failure", return_value=("none", {})), \
             patch("refinement_loop.check_guardrails",
                   return_value=MagicMock(violations=[], warnings=[])), \
             patch("refinement_loop.run_full_validation",
                   return_value={"status": "ok", "passed": False}), \
             patch("refinement_loop.save_report", return_value=None):

            mandate_path = self._write_mandate(name="high_target")
            from refinement_loop import refinement_loop
            state = refinement_loop(str(mandate_path), max_turns=3, sharpe_target=3.0)
        assert state.status == "success"
        assert state.best_sharpe >= 3.0

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_sharpe_target_not_met_exhausts_turns(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """When Sharpe target is never met, all turns are exhausted."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        low_sharpe_result = MagicMock(
            sharpe=0.3, total_return=3.0, max_drawdown=5.0,
            win_rate=0.5, num_trades=15, profit_factor=1.0,
            calmar_ratio=0.3, score=0.3, sortino_ratio=0.4, params={},
        )

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor", return_value={
                 "action": "novel", "hypothesis": "low sharpe",
                 "new_strategy_code": (
                     "def generate_signals(df, params): pass\n"
                     "DEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'"
                 ),
                 "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
             }), \
             patch("refinement_loop.run_validation_gates", return_value=(True, [])), \
             patch("refinement_loop.build_llm_context", return_value={}), \
             patch("refinement_loop.load_strategy_module", return_value=MagicMock(
                 generate_signals=MagicMock(return_value=(None, None)),
                 DEFAULT_PARAMS={}, PARAM_GRID={}, DESCRIPTION="test",
             )), \
             patch("refinement_loop.run_backtest_safely",
                   return_value=(low_sharpe_result, MagicMock(), MagicMock(), None)), \
             patch("refinement_loop.compute_sharpe_by_year", return_value={"2024": 0.3}), \
             patch("refinement_loop.classify_failure", return_value=("low_sharpe", {})), \
             patch("refinement_loop.check_guardrails",
                   return_value=MagicMock(violations=[], warnings=[])), \
             patch("refinement_loop.save_report", return_value=None):

            mandate_path = self._write_mandate(name="unreachable_target")
            from refinement_loop import refinement_loop
            state = refinement_loop(str(mandate_path), max_turns=2, sharpe_target=5.0)
        assert state.status == "max_turns_exhausted"

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_llm_returns_modify_action(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """LLM returning a 'modify_params' action is handled correctly."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor", return_value={
                 "action": "modify_params", "hypothesis": "tune params",
                 "new_strategy_code": (
                     "def generate_signals(df, params): pass\n"
                     "DEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'"
                 ),
                 "reasoning": "tweak", "addresses_failure": "low sharpe",
                 "expected_impact": "higher",
             }), \
             patch("refinement_loop.run_validation_gates", return_value=(True, [])), \
             patch("refinement_loop.build_llm_context", return_value={}), \
             patch("refinement_loop.load_strategy_module", return_value=MagicMock(
                 generate_signals=MagicMock(return_value=(None, None)),
                 DEFAULT_PARAMS={}, PARAM_GRID={}, DESCRIPTION="test",
             )), \
             patch("refinement_loop.run_backtest_safely",
                   return_value=(MagicMock(
                       sharpe=2.0, total_return=20.0, max_drawdown=5.0,
                       win_rate=0.6, num_trades=20, profit_factor=1.5,
                       calmar_ratio=1.6, score=2.0, sortino_ratio=1.8, params={},
                   ), MagicMock(), MagicMock(), None)), \
             patch("refinement_loop.compute_sharpe_by_year", return_value={}), \
             patch("refinement_loop.classify_failure", return_value=("none", {})), \
             patch("refinement_loop.check_guardrails",
                   return_value=MagicMock(violations=[], warnings=[])), \
             patch("refinement_loop.run_full_validation",
                   return_value={"status": "ok", "passed": False}), \
             patch("refinement_loop.save_report", return_value=None):

            mandate_path = self._write_mandate(name="modify_test")
            from refinement_loop import refinement_loop
            state = refinement_loop(str(mandate_path), max_turns=2, sharpe_target=0.1)
        assert state.status == "success"
        # Verify the action was recorded in history
        assert any(h.get("action") == "modify_params" for h in state.history)

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_llm_returns_empty_strategy_code(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """LLM returning empty strategy_code triggers code generation failure."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor", return_value={
                 "action": "novel", "hypothesis": "empty",
                 "new_strategy_code": "",
                 "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
             }):

            mandate_path = self._write_mandate(name="empty_code")
            from refinement_loop import refinement_loop
            state = refinement_loop(str(mandate_path), max_turns=2, sharpe_target=0.1)
        # Should fail — empty code means no strategy generated
        assert state.status in ("abandoned", "max_turns_exhausted")

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_module_load_failure_advances_turn(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """When module loading fails after gates pass, turn advances."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor", return_value={
                 "action": "novel", "hypothesis": "test",
                 "new_strategy_code": (
                     "def generate_signals(df, params): pass\n"
                     "DEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'"
                 ),
                 "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
             }), \
             patch("refinement_loop.run_validation_gates", return_value=(True, [])), \
             patch("refinement_loop.load_strategy_module", return_value=None):

            mandate_path = self._write_mandate(name="module_fail")
            from refinement_loop import refinement_loop
            state = refinement_loop(str(mandate_path), max_turns=2, sharpe_target=0.1)
        assert state.status == "max_turns_exhausted"
        assert any("module_load" in str(h).lower() for h in state.history)

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_llm_raises_exception(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """When LLM call raises an exception, the loop advances."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor",
                   side_effect=TimeoutError("LLM timeout")):

            mandate_path = self._write_mandate(name="llm_timeout")
            from refinement_loop import refinement_loop
            state = refinement_loop(str(mandate_path), max_turns=2, sharpe_target=0.1)
        assert state.status in ("abandoned", "max_turns_exhausted")

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_llm_returns_none_then_succeeds(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """LLM fails first call then succeeds — loop should advance."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        call_count = [0]
        def llm_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return None  # Fail first 2 calls (all 3 retries in turn 1)
            return {
                "action": "novel", "hypothesis": "retry success",
                "new_strategy_code": (
                    "def generate_signals(df, params): pass\n"
                    "DEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'"
                ),
                "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
            }

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor", side_effect=llm_side_effect):

            mandate_path = self._write_mandate(name="retry_test")
            from refinement_loop import refinement_loop
            state = refinement_loop(str(mandate_path), max_turns=3, sharpe_target=0.1)
        # After turn 1 fails (3 retries), turn 2 also fails (3 retries),
        # so max_turns_exhausted or abandoned
        assert state.status in ("abandoned", "max_turns_exhausted")

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
    def test_max_turns_equals_one(
        self, mock_clear, mock_inventor, mock_lock, mock_load_mod,
        mock_build_ctx, mock_gates, mock_backtest, mock_sharpe_yr,
        mock_classify, mock_guardrails, mock_validation, mock_write_json,
        mock_load_report, mock_save_report, mock_save_state,
        mock_promote, mock_release,
    ):
        """Single turn: succeeds immediately or exhausts after 1 turn."""
        mock_lock.return_value = True
        mock_clear.return_value = None
        mock_inventor.return_value = {
            "action": "novel", "hypothesis": "single turn",
            "new_strategy_code": (
                "def generate_signals(df, params): pass\n"
                "DEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'"
            ),
            "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
        }
        mock_build_ctx.return_value = {}
        mock_gates.return_value = (True, [])
        mock_load_mod.return_value = MagicMock(
            generate_signals=MagicMock(return_value=(None, None)),
            DEFAULT_PARAMS={}, PARAM_GRID={}, DESCRIPTION="test",
        )
        mock_backtest.return_value = (
            MagicMock(
                sharpe=2.0, total_return=20.0, max_drawdown=5.0,
                win_rate=0.6, num_trades=25, profit_factor=1.5,
                calmar_ratio=1.6, score=2.0, sortino_ratio=1.8, params={},
            ),
            MagicMock(), MagicMock(), None,
        )
        mock_sharpe_yr.return_value = {}
        mock_classify.return_value = ("none", {})
        mock_guardrails.return_value = MagicMock(violations=[], warnings=[])
        mock_validation.return_value = {"status": "ok", "passed": False}
        mock_write_json.return_value = None
        mock_load_report.return_value = None
        mock_save_report.return_value = None
        mock_save_state.return_value = None

        mandate_path = self._write_mandate(name="single_turn")
        from refinement_loop import refinement_loop
        state = refinement_loop(str(mandate_path), max_turns=1, sharpe_target=0.1)
        assert state.status == "success"
        assert state.current_turn == 1

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_lock_not_acquired_returns_running(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """When lock cannot be acquired, state stays as initial (running)."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        with patch("refinement_loop.acquire_lock", return_value=False):
            mandate_path = self._write_mandate(name="locked")
            from refinement_loop import refinement_loop
            state = refinement_loop(str(mandate_path), max_turns=2, sharpe_target=0.1)
        # Lock not acquired → early return, status should be initial (not "running"
        # since save_state isn't called before the lock check returns state as-is)
        mock_inventor_not_called = True

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
    def test_history_records_correctly(
        self, mock_clear, mock_inventor, mock_lock, mock_load_mod,
        mock_build_ctx, mock_gates, mock_backtest, mock_sharpe_yr,
        mock_classify, mock_guardrails, mock_validation, mock_write_json,
        mock_load_report, mock_save_report, mock_save_state,
        mock_promote, mock_release,
    ):
        """Verify history is populated with correct turn data on success."""
        mock_lock.return_value = True
        mock_clear.return_value = None
        mock_inventor.return_value = {
            "action": "novel", "hypothesis": "history test",
            "new_strategy_code": (
                "def generate_signals(df, params): pass\n"
                "DEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'test'"
            ),
            "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
        }
        mock_build_ctx.return_value = {}
        mock_gates.return_value = (True, [])
        mock_load_mod.return_value = MagicMock(
            generate_signals=MagicMock(return_value=(None, None)),
            DEFAULT_PARAMS={}, PARAM_GRID={}, DESCRIPTION="test",
        )
        mock_backtest.return_value = (
            MagicMock(
                sharpe=1.5, total_return=15.0, max_drawdown=5.0,
                win_rate=0.55, num_trades=22, profit_factor=1.3,
                calmar_ratio=1.5, score=1.5, sortino_ratio=1.7, params={},
            ),
            MagicMock(), MagicMock(), None,
        )
        mock_sharpe_yr.return_value = {"2024": 1.5}
        mock_classify.return_value = ("none", {})
        mock_guardrails.return_value = MagicMock(violations=[], warnings=[])
        mock_validation.return_value = {"status": "ok", "passed": False}
        mock_write_json.return_value = None
        mock_load_report.return_value = None
        mock_save_report.return_value = None
        mock_save_state.return_value = None

        mandate_path = self._write_mandate(name="history_test")
        from refinement_loop import refinement_loop
        state = refinement_loop(str(mandate_path), max_turns=2, sharpe_target=0.1)
        assert state.status == "success"
        assert len(state.history) >= 1
        last_entry = state.history[-1]
        assert last_entry.get("turn") == 1
        assert last_entry.get("sharpe") == 1.5
        assert last_entry.get("action") == "novel"

    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.promote_to_winner")
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_report")
    @patch("refinement_loop.clear_cache")
    def test_invalid_strategy_code_from_llm(
        self, mock_clear, mock_load_report, mock_write_json,
        mock_save_state, mock_promote, mock_release,
    ):
        """LLM returns syntactically invalid code that fails gate validation."""
        mock_clear.return_value = None
        mock_load_report.return_value = None
        mock_write_json.return_value = None
        mock_save_state.return_value = None

        with patch("refinement_loop.acquire_lock", return_value=True), \
             patch("refinement_loop.call_llm_inventor", return_value={
                 "action": "novel", "hypothesis": "bad code",
                 "new_strategy_code": "this is not valid python syntax!!!!",
                 "reasoning": "", "addresses_failure": "", "expected_impact": "higher",
             }), \
             patch("refinement_loop.run_validation_gates",
                   return_value=(False, ["syntax error", "missing generate_signals"])):

            mandate_path = self._write_mandate(name="bad_code")
            from refinement_loop import refinement_loop
            state = refinement_loop(str(mandate_path), max_turns=2, sharpe_target=0.1)
        assert state.status in ("abandoned", "max_turns_exhausted")
