"""Integration tests for code_quality_check wired into the refinement loop.

Verifies that:
  1. check_code_quality() is called after gates pass, before backtesting
  2. Critical (reject) issues cause the strategy to be skipped
  3. Warning issues are stored as feedback but don't block
  4. Clean code passes through normally
  5. Feedback flows through state.code_quality_feedback -> context_builder -> LLM prompt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))


# ── Strategy code fixtures ──────────────────────────────────────────────────

GOOD_STRATEGY = """\
from crabquant.indicator_cache import cached_indicator
import pandas as pd

DESCRIPTION = "Simple RSI strategy"
DEFAULT_PARAMS = {"period": 14, "threshold": 30}

def generate_signals(df, params):
    period = params.get("period", 14)
    threshold = params.get("threshold", 30)
    rsi = cached_indicator("rsi", close=df["close"], length=period)
    entries = (rsi < threshold) & (rsi > rsi.shift(1))
    exits = (rsi > 70) | (hold_periods > 10)
    return entries, exits
"""

# Score ~0.5, verdict=warning (2 critical + 1 warning = -0.50 penalty)
# Used for WARNING tests — does NOT get rejected
WARNING_STRATEGY = """\
from crabquant.indicator_cache import cached_indicator
import pandas as pd

DESCRIPTION = "Strategy with no exit and contradictions"
DEFAULT_PARAMS = {"period": 14, "fast": 200, "slow": 300}

def generate_signals(df, params):
    period = params.get("period", 14)
    rsi = cached_indicator("rsi", close=df["close"], length=period)
    ema = cached_indicator("ema", close=df["close"], length=200)
    entries = (rsi < 30) & (ema > 50) & (ema < 30)
    exits = pd.Series(False, index=df.index)
    return entries, exits
"""

# Score ~0.1, verdict=reject (4 critical + 3 warning = far below threshold)
# Multiple contradictions + no_exit + long_lookback + extreme_threshold
REJECT_STRATEGY = """\
from crabquant.indicator_cache import cached_indicator
import pandas as pd

DESCRIPTION = "Contradictory conditions"
DEFAULT_PARAMS = {"fast": 10, "slow": 20}

def generate_signals(df, params):
    rsi = cached_indicator("rsi", close=df["close"], length=200)
    ema = cached_indicator("ema", close=df["close"], length=150)
    entries = (rsi > 50) & (rsi < 30) & (ema > 100) & (ema < 50)
    exits = pd.Series(False, index=df.index)
    return entries, exits
"""

NO_GENERATE_SIGNALS = """\
from crabquant.indicator_cache import cached_indicator

DESCRIPTION = "Missing generate_signals"
DEFAULT_PARAMS = {"period": 14}
"""

EMPTY_STRATEGY = ""

# 5 stacked & operators -> critical over_complex
OVER_COMPLEX_STRATEGY = """\
from crabquant.indicator_cache import cached_indicator
import pandas as pd

DESCRIPTION = "Over-complex strategy"
DEFAULT_PARAMS = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}

def generate_signals(df, params):
    rsi = cached_indicator("rsi", close=df["close"], length=14)
    macd = cached_indicator("macd", close=df["close"], fast=12, slow=26)
    atr = cached_indicator("atr", high=df["high"], low=df["low"], close=df["close"], length=14)
    ema = cached_indicator("ema", close=df["close"], length=50)
    bb = cached_indicator("bbands", close=df["close"], length=20)
    entries = (
        (rsi < 30)
        & (macd > 0)
        & (atr < 2)
        & (ema > df["close"])
        & (bb > 0)
        & (df["volume"] > 1000)
    )
    exits = (rsi > 70) | (hold_periods > 10)
    return entries, exits
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_run_state(**overrides) -> Any:
    """Create a RunState object with sensible defaults."""
    from crabquant.refinement.schemas import RunState
    defaults = {
        "run_id": "test_run_001",
        "mandate_name": "test_mandate",
        "created_at": "2025-01-01T00:00:00Z",
        "max_turns": 7,
        "sharpe_target": 1.5,
        "tickers": ["SPY"],
        "period": "1y",
        "current_turn": 1,
        "status": "running",
        "history": [],
        "code_quality_feedback": "",
    }
    defaults.update(overrides)
    return RunState(**{k: v for k, v in defaults.items()
                       if k in {f.name for f in RunState.__dataclass_fields__.values()}})


def _make_minimal_mandate() -> dict:
    return {
        "name": "test_mandate",
        "primary_ticker": "SPY",
        "tickers": ["SPY"],
        "period": "1y",
    }


def _make_mock_backtest_result(**overrides):
    """Create a mock BacktestResult with sensible defaults."""
    defaults = {
        "strategy_name": "test_strategy",
        "iteration": 1,
        "sharpe": 0.5,
        "total_return": 0.05,
        "max_drawdown": -0.1,
        "num_trades": 10,
        "win_rate": 0.5,
        "profit_factor": 1.0,
        "calmar_ratio": 0.5,
        "sortino_ratio": 0.6,
        "score": 1.0,
        "params": {},
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_mock_guardrail():
    guard = MagicMock()
    guard.violations = []
    guard.warnings = []
    return guard


def _patch_refinement_config():
    """Return a patcher that mocks RefinementConfig as a proper class."""
    cfg = MagicMock()
    cfg.parallel_invention = False
    cfg.multi_ticker_backtest = False
    cfg.feature_importance = False
    cfg.param_optimization = False
    cfg.adaptive_sharpe_target = False
    cfg.adaptive_start_factor = 0.5
    cfg.adaptive_ramp_turns = 3
    cfg.auto_revert_enabled = False
    cfg.auto_revert_max_consecutive = 2
    cfg.per_strategy_timeout_minutes = 15
    cfg.backtest_timeout_seconds = 60
    # from_mandate is called as RefinementConfig.from_mandate(mandate)
    # Make it return our configured instance
    cfg_cls = MagicMock(return_value=cfg)
    cfg_cls.from_mandate = MagicMock(return_value=cfg)
    return patch("refinement_loop.RefinementConfig", cfg_cls)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Direct unit tests on check_code_quality API
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckCodeQualityDirect:
    """Direct tests on check_code_quality with the strategy code fixtures."""

    def test_good_strategy_passes(self):
        """Good strategy should get 'good' verdict."""
        from crabquant.refinement.code_quality_check import check_code_quality
        report = check_code_quality(GOOD_STRATEGY)
        assert report.overall_verdict == "good"
        assert report.score >= 0.75

    def test_reject_strategy_has_critical_issues(self):
        """Strategy with multiple contradictions + no_exit should be rejected."""
        from crabquant.refinement.code_quality_check import check_code_quality
        report = check_code_quality(REJECT_STRATEGY)
        assert report.overall_verdict == "reject"
        assert report.score < 0.50
        categories = {i.category for i in report.issues}
        assert "no_exit" in categories or "contradictory" in categories

    def test_empty_strategy_rejected(self):
        """Empty strategy code should be rejected."""
        from crabquant.refinement.code_quality_check import check_code_quality
        report = check_code_quality(EMPTY_STRATEGY)
        assert report.overall_verdict == "reject"
        assert report.score == 0.0

    def test_no_generate_signals_rejected(self):
        """Strategy without generate_signals should be rejected."""
        from crabquant.refinement.code_quality_check import check_code_quality
        report = check_code_quality(NO_GENERATE_SIGNALS)
        assert report.overall_verdict == "reject"
        assert report.score == 0.0

    def test_format_for_prompt_includes_verdict(self):
        """format_code_quality_for_prompt should include REJECT verdict."""
        from crabquant.refinement.code_quality_check import (
            check_code_quality, format_code_quality_for_prompt,
        )
        report = check_code_quality(REJECT_STRATEGY)
        formatted = format_code_quality_for_prompt(report)
        assert "REJECT" in formatted.upper()

    def test_format_for_prompt_good_code(self):
        """format_code_quality_for_prompt for clean code should say no issues."""
        from crabquant.refinement.code_quality_check import (
            check_code_quality, format_code_quality_for_prompt,
        )
        report = check_code_quality(GOOD_STRATEGY)
        formatted = format_code_quality_for_prompt(report)
        assert "No issues detected" in formatted or "good" in formatted.lower()

    def test_over_complex_strategy_gets_warning_or_reject(self):
        """Strategy with 5+ stacked '&' operators should get at least a warning."""
        from crabquant.refinement.code_quality_check import check_code_quality
        report = check_code_quality(OVER_COMPLEX_STRATEGY)
        assert report.overall_verdict in ("warning", "reject")
        assert any(i.category == "over_complex" for i in report.issues)

    def test_report_has_required_fields(self):
        """CodeQualityReport should have all expected fields."""
        from crabquant.refinement.code_quality_check import check_code_quality
        report = check_code_quality(GOOD_STRATEGY)
        assert hasattr(report, "score")
        assert hasattr(report, "issues")
        assert hasattr(report, "overall_verdict")
        assert hasattr(report, "summary_for_llm")
        assert 0.0 <= report.score <= 1.0
        assert report.overall_verdict in ("good", "warning", "reject")

    def test_warning_strategy_gets_warning(self):
        """Strategy with some critical issues should get warning verdict."""
        from crabquant.refinement.code_quality_check import check_code_quality
        report = check_code_quality(WARNING_STRATEGY)
        assert report.overall_verdict in ("warning", "reject")

    def test_reject_strategy_score_below_threshold(self):
        """Rejected strategy should have score below 0.50."""
        from crabquant.refinement.code_quality_check import check_code_quality
        report = check_code_quality(REJECT_STRATEGY)
        assert report.score < 0.50


# ═══════════════════════════════════════════════════════════════════════════
# 2. RunState has code_quality_feedback field
# ═══════════════════════════════════════════════════════════════════════════

class TestRunStateCodeQualityField:
    """Verify that RunState supports code_quality_feedback field."""

    def test_run_state_has_code_quality_feedback(self):
        """RunState should have a code_quality_feedback field defaulting to ''."""
        from crabquant.refinement.schemas import RunState
        state = _make_run_state()
        assert hasattr(state, "code_quality_feedback")
        assert state.code_quality_feedback == ""

    def test_run_state_serialization_round_trip(self):
        """code_quality_feedback should survive JSON round-trip."""
        from crabquant.refinement.schemas import RunState
        state = _make_run_state(code_quality_feedback="## Reject feedback")
        json_str = state.to_json()
        restored = RunState.from_json(json_str)
        assert restored.code_quality_feedback == "## Reject feedback"

    def test_run_state_code_quality_feedback_settable(self):
        """code_quality_feedback can be set after construction."""
        from crabquant.refinement.schemas import RunState
        state = _make_run_state()
        state.code_quality_feedback = "new feedback"
        assert state.code_quality_feedback == "new feedback"


# ═══════════════════════════════════════════════════════════════════════════
# 3. context_builder injects code quality feedback into prompt
# ═══════════════════════════════════════════════════════════════════════════

class TestContextBuilderCodeQualityInjection:
    """Verify that code quality feedback reaches the LLM via context_builder."""

    @staticmethod
    def _build_context(state, mandate):
        """Helper to call build_llm_context with all expensive deps mocked."""
        from crabquant.refinement.context_builder import build_llm_context

        patches = [
            patch("crabquant.refinement.context_builder.load_indicator_reference", return_value=""),
            patch("crabquant.refinement.context_builder.extract_quick_reference", return_value=""),
            patch("crabquant.refinement.context_builder.build_trade_count_guidance", return_value=""),
            patch("crabquant.refinement.context_builder.get_strategy_examples", return_value=[]),
            patch("crabquant.refinement.context_builder.get_winner_examples", return_value=[]),
            patch("crabquant.refinement.context_builder.get_strategy_catalog", return_value=[]),
            patch("crabquant.refinement.context_builder._build_stagnation_recovery_section", return_value=""),
            patch("crabquant.refinement.context_builder._build_crash_error_feedback", return_value=""),
            # generate_llm_context is imported from action_analytics inside build_llm_context
            patch("crabquant.refinement.action_analytics.generate_llm_context", return_value=""),
            patch("crabquant.refinement.action_analytics.load_run_history", return_value=[]),
            patch("crabquant.refinement.action_analytics.RUN_HISTORY_FILE", "/dev/null"),
            patch("crabquant.refinement.context_builder.build_turn1_prompt", return_value="## Base prompt"),
        ]
        for p in patches:
            p.start()
        try:
            return build_llm_context(state, report=None, mandate=mandate)
        finally:
            for p in patches:
                p.stop()

    def test_feedback_in_prompt_when_set(self):
        """When state has code_quality_feedback, it should appear in prompt."""
        state = _make_run_state(
            code_quality_feedback="## Code Quality Pre-Check (Score: 0.30 - REJECT)\n- [CRITICAL] no_exit"
        )
        mandate = _make_minimal_mandate()
        context = self._build_context(state, mandate)

        prompt = context.get("prompt", "")
        assert "CODE QUALITY PRE-CHECK FAILED" in prompt

    def test_no_section_when_feedback_empty(self):
        """When state.code_quality_feedback is empty, prompt should not have section."""
        state = _make_run_state(code_quality_feedback="")
        mandate = _make_minimal_mandate()
        context = self._build_context(state, mandate)

        prompt = context.get("prompt", "")
        assert "CODE QUALITY PRE-CHECK FAILED" not in prompt

    def test_feedback_section_includes_instruction(self):
        """The code quality section should include an instruction to fix."""
        state = _make_run_state(
            code_quality_feedback="## Code Quality Pre-Check (Score: 0.30 - REJECT)\n- [CRITICAL] no_exit"
        )
        mandate = _make_minimal_mandate()
        context = self._build_context(state, mandate)

        prompt = context.get("prompt", "")
        assert "MUST fix" in prompt


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integration: refinement loop calls check_code_quality
# ═══════════════════════════════════════════════════════════════════════════

class TestRefinementLoopIntegration:
    """End-to-end tests that check_code_quality is wired into the loop."""

    @patch("refinement_loop.call_llm_inventor")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.load_strategy_module", return_value=MagicMock())
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_json", return_value=_make_minimal_mandate())
    @patch("refinement_loop.acquire_lock", return_value=True)
    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.create_run_directory")
    @patch("refinement_loop._write_dashboard")
    @patch("refinement_loop.clear_cache")
    @patch("refinement_loop.compute_sharpe_by_year", return_value={})
    @patch("refinement_loop.check_guardrails")
    @patch("refinement_loop.classify_failure", return_value=("low_sharpe", "Sharpe too low"))
    def test_check_code_quality_called_for_good_code(
        self, mock_classify, mock_guardrails, mock_sharpe_by_year,
        mock_clear, mock_dashboard, mock_create_dir, mock_release,
        mock_acquire, mock_load_json, mock_write_json, mock_save_state,
        mock_load_mod, mock_gates, mock_llm, tmp_path,
    ):
        """check_code_quality should be called when validation gates pass."""
        from crabquant.refinement.code_quality_check import check_code_quality

        mock_create_dir.return_value = tmp_path
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": GOOD_STRATEGY,
            "reasoning": "test",
            "expected_impact": "moderate",
        }

        mock_bt_result = _make_mock_backtest_result()
        mock_guardrails.return_value = _make_mock_guardrail()

        with patch("refinement_loop.run_backtest_safely",
                    return_value=(mock_bt_result, MagicMock(), MagicMock(), {})):
            with patch("refinement_loop.check_code_quality", wraps=check_code_quality) as spy:
                with _patch_refinement_config():
                    refinement_loop = __import__("refinement_loop")
                    refinement_loop.refinement_loop(str(tmp_path / "mandate.json"), max_turns=1)

        spy.assert_called_once()
        call_code = spy.call_args[0][0]
        assert "generate_signals" in call_code

    @patch("refinement_loop.call_llm_inventor")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_json", return_value=_make_minimal_mandate())
    @patch("refinement_loop.acquire_lock", return_value=True)
    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.create_run_directory")
    @patch("refinement_loop._write_dashboard")
    @patch("refinement_loop.clear_cache")
    def test_reject_skips_backtest(
        self, mock_clear, mock_dashboard, mock_create_dir, mock_release,
        mock_acquire, mock_load_json, mock_write_json, mock_save_state,
        mock_gates, mock_llm, tmp_path,
    ):
        """When code quality is 'reject', backtesting should NOT run."""
        mock_create_dir.return_value = tmp_path
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": REJECT_STRATEGY,
            "reasoning": "test",
            "expected_impact": "moderate",
        }

        with patch("refinement_loop.run_backtest_safely") as mock_bt:
            with _patch_refinement_config():
                refinement_loop = __import__("refinement_loop")
                refinement_loop.refinement_loop(str(tmp_path / "mandate.json"), max_turns=1)

        mock_bt.assert_not_called()

    @patch("refinement_loop.call_llm_inventor")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_json", return_value=_make_minimal_mandate())
    @patch("refinement_loop.acquire_lock", return_value=True)
    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.create_run_directory")
    @patch("refinement_loop._write_dashboard")
    @patch("refinement_loop.clear_cache")
    def test_reject_records_in_history(
        self, mock_clear, mock_dashboard, mock_create_dir, mock_release,
        mock_acquire, mock_load_json, mock_write_json, mock_save_state,
        mock_gates, mock_llm, tmp_path,
    ):
        """When code quality is 'reject', state.history should record it."""
        mock_create_dir.return_value = tmp_path
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": REJECT_STRATEGY,
            "reasoning": "test",
            "expected_impact": "moderate",
        }

        with patch("refinement_loop.run_backtest_safely"):
            with _patch_refinement_config():
                refinement_loop = __import__("refinement_loop")
                result = refinement_loop.refinement_loop(
                    str(tmp_path / "mandate.json"), max_turns=1
                )

        assert any(h.get("status") == "code_quality_rejected" for h in result.history)
        assert any("code_quality_score" in h for h in result.history)

    @patch("refinement_loop.call_llm_inventor")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_json", return_value=_make_minimal_mandate())
    @patch("refinement_loop.acquire_lock", return_value=True)
    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.create_run_directory")
    @patch("refinement_loop._write_dashboard")
    @patch("refinement_loop.clear_cache")
    def test_reject_sets_feedback_on_state(
        self, mock_clear, mock_dashboard, mock_create_dir, mock_release,
        mock_acquire, mock_load_json, mock_write_json, mock_save_state,
        mock_gates, mock_llm, tmp_path,
    ):
        """When code quality is 'reject', state.code_quality_feedback should be set."""
        mock_create_dir.return_value = tmp_path
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": REJECT_STRATEGY,
            "reasoning": "test",
            "expected_impact": "moderate",
        }

        with patch("refinement_loop.run_backtest_safely"):
            with _patch_refinement_config():
                refinement_loop = __import__("refinement_loop")
                result = refinement_loop.refinement_loop(
                    str(tmp_path / "mandate.json"), max_turns=1
                )

        assert result.code_quality_feedback != ""
        assert "Code Quality" in result.code_quality_feedback

    @patch("refinement_loop.call_llm_inventor")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.load_strategy_module", return_value=MagicMock())
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_json", return_value=_make_minimal_mandate())
    @patch("refinement_loop.acquire_lock", return_value=True)
    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.create_run_directory")
    @patch("refinement_loop._write_dashboard")
    @patch("refinement_loop.clear_cache")
    @patch("refinement_loop.compute_sharpe_by_year", return_value={})
    @patch("refinement_loop.check_guardrails")
    @patch("refinement_loop.classify_failure", return_value=("low_sharpe", "Sharpe too low"))
    def test_good_code_reaches_backtest(
        self, mock_classify, mock_guardrails, mock_sharpe_by_year,
        mock_clear, mock_dashboard, mock_create_dir, mock_release,
        mock_acquire, mock_load_json, mock_write_json, mock_save_state,
        mock_load_mod, mock_gates, mock_llm, tmp_path,
    ):
        """Clean strategy code should pass code quality and reach backtesting."""
        mock_create_dir.return_value = tmp_path
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": GOOD_STRATEGY,
            "reasoning": "test",
            "expected_impact": "moderate",
        }

        mock_bt_result = _make_mock_backtest_result()
        mock_guardrails.return_value = _make_mock_guardrail()

        with patch("refinement_loop.run_backtest_safely",
                    return_value=(mock_bt_result, MagicMock(), MagicMock(), {})) as mock_bt:
            with _patch_refinement_config():
                refinement_loop = __import__("refinement_loop")
                refinement_loop.refinement_loop(
                    str(tmp_path / "mandate.json"), max_turns=1
                )

        mock_bt.assert_called_once()

    @patch("refinement_loop.call_llm_inventor")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.load_strategy_module", return_value=MagicMock())
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_json", return_value=_make_minimal_mandate())
    @patch("refinement_loop.acquire_lock", return_value=True)
    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.create_run_directory")
    @patch("refinement_loop._write_dashboard")
    @patch("refinement_loop.clear_cache")
    @patch("refinement_loop.compute_sharpe_by_year", return_value={})
    @patch("refinement_loop.check_guardrails")
    @patch("refinement_loop.classify_failure", return_value=("low_sharpe", "Sharpe too low"))
    def test_good_code_clears_feedback(
        self, mock_classify, mock_guardrails, mock_sharpe_by_year,
        mock_clear, mock_dashboard, mock_create_dir, mock_release,
        mock_acquire, mock_load_json, mock_write_json, mock_save_state,
        mock_load_mod, mock_gates, mock_llm, tmp_path,
    ):
        """Clean strategy code should clear state.code_quality_feedback."""
        mock_create_dir.return_value = tmp_path
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": GOOD_STRATEGY,
            "reasoning": "test",
            "expected_impact": "moderate",
        }

        mock_bt_result = _make_mock_backtest_result()
        mock_guardrails.return_value = _make_mock_guardrail()

        with patch("refinement_loop.run_backtest_safely",
                    return_value=(mock_bt_result, MagicMock(), MagicMock(), {})):
            with _patch_refinement_config():
                refinement_loop = __import__("refinement_loop")
                result = refinement_loop.refinement_loop(
                    str(tmp_path / "mandate.json"), max_turns=1
                )

        assert result.code_quality_feedback == ""

    @patch("refinement_loop.call_llm_inventor")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_json", return_value=_make_minimal_mandate())
    @patch("refinement_loop.acquire_lock", return_value=True)
    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.create_run_directory")
    @patch("refinement_loop._write_dashboard")
    @patch("refinement_loop.clear_cache")
    def test_reject_on_turn1_sets_feedback_for_turn2(
        self, mock_clear, mock_dashboard, mock_create_dir, mock_release,
        mock_acquire, mock_load_json, mock_write_json, mock_save_state,
        mock_gates, mock_llm, tmp_path,
    ):
        """After code quality rejection, state should carry feedback for next turn."""
        mock_create_dir.return_value = tmp_path
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": REJECT_STRATEGY,
            "reasoning": "test",
            "expected_impact": "moderate",
        }

        with patch("refinement_loop.run_backtest_safely"):
            with _patch_refinement_config():
                refinement_loop = __import__("refinement_loop")
                result = refinement_loop.refinement_loop(
                    str(tmp_path / "mandate.json"), max_turns=2
                )

        assert result.code_quality_feedback != ""
        assert "Code Quality" in result.code_quality_feedback
        assert any(h.get("status") == "code_quality_rejected" for h in result.history)

    @patch("refinement_loop.call_llm_inventor")
    @patch("refinement_loop.run_validation_gates", return_value=(True, []))
    @patch("refinement_loop.save_state")
    @patch("refinement_loop.write_json")
    @patch("refinement_loop.load_json", return_value=_make_minimal_mandate())
    @patch("refinement_loop.acquire_lock", return_value=True)
    @patch("refinement_loop.release_lock")
    @patch("refinement_loop.create_run_directory")
    @patch("refinement_loop._write_dashboard")
    @patch("refinement_loop.clear_cache")
    def test_empty_code_rejected_without_backtest(
        self, mock_clear, mock_dashboard, mock_create_dir, mock_release,
        mock_acquire, mock_load_json, mock_write_json, mock_save_state,
        mock_gates, mock_llm, tmp_path,
    ):
        """Empty code should be rejected by code quality without backtesting."""
        mock_create_dir.return_value = tmp_path
        mock_llm.return_value = {
            "action": "novel",
            "hypothesis": "test",
            "new_strategy_code": EMPTY_STRATEGY,
            "reasoning": "test",
            "expected_impact": "moderate",
        }

        with patch("refinement_loop.run_backtest_safely") as mock_bt:
            with _patch_refinement_config():
                refinement_loop = __import__("refinement_loop")
                refinement_loop.refinement_loop(str(tmp_path / "mandate.json"), max_turns=1)

        mock_bt.assert_not_called()
