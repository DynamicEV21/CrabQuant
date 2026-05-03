"""Tests for Phase 6: crash error capture and feedback.

Covers:
- run_backtest_safely returns error_info dict on failure
- load_strategy_module returns error_info dict on failure
- _build_crash_error_feedback formats crash errors for LLM context
- get_crash_recovery_hints matches error patterns
- build_crash_guidance produces actionable guidance
- build_failure_guidance handles crash modes
"""

import numpy as np
import pandas as pd
import pytest
from types import ModuleType
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_error_info(
    error_type: str = "KeyError",
    error_message: str = "'Close'",
    error_traceback: str = "",
) -> dict:
    return {
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": error_traceback,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. run_backtest_safely — returns error_info on exception
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunBacktestSafelyErrorInfo:
    """Verify run_backtest_safely returns a 4-tuple with error_info on failure."""

    def test_returns_4tuple_on_keyerror(self):
        """KeyError during generate_signals → (None, None, None, error_info)."""
        from crabquant.refinement.diagnostics import run_backtest_safely

        bad_module = MagicMock()
        bad_module.DEFAULT_PARAMS = {"period": 14}
        bad_module.generate_signals.side_effect = KeyError("Close")

        with patch("crabquant.refinement.diagnostics.load_data") as mock_load:
            idx = pd.date_range("2022-01-03", periods=60, freq="B")
            mock_load.return_value = pd.DataFrame({
                "open": np.full(60, 100.0),
                "high": np.full(60, 105.0),
                "low": np.full(60, 95.0),
                "close": np.full(60, 100.0),
                "volume": np.full(60, 1e6),
            }, index=idx)

            result = run_backtest_safely(bad_module, "AAPL", "1y")

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert result[0] is None  # no BacktestResult
        assert result[1] is None  # no df
        assert result[2] is None  # no portfolio
        error_info = result[3]
        assert error_info is not None
        assert error_info["error_type"] == "KeyError"
        assert "Close" in error_info["error_message"]
        assert "error_traceback" in error_info

    def test_returns_4tuple_on_nameerror(self):
        """NameError during generate_signals → error_info with correct type."""
        from crabquant.refinement.diagnostics import run_backtest_safely

        bad_module = MagicMock()
        bad_module.DEFAULT_PARAMS = {}
        bad_module.generate_signals.side_effect = NameError("name 'ema' is not defined")

        with patch("crabquant.refinement.diagnostics.load_data") as mock_load:
            idx = pd.date_range("2022-01-03", periods=60, freq="B")
            mock_load.return_value = pd.DataFrame({
                "open": np.full(60, 100.0),
                "high": np.full(60, 105.0),
                "low": np.full(60, 95.0),
                "close": np.full(60, 100.0),
                "volume": np.full(60, 1e6),
            }, index=idx)

            result = run_backtest_safely(bad_module, "AAPL", "1y")

        error_info = result[3]
        assert error_info["error_type"] == "NameError"
        assert "ema" in error_info["error_message"]

    def test_returns_4tuple_on_typeerror(self):
        """TypeError during generate_signals → error_info with correct type."""
        from crabquant.refinement.diagnostics import run_backtest_safely

        bad_module = MagicMock()
        bad_module.DEFAULT_PARAMS = {}
        bad_module.generate_signals.side_effect = TypeError(
            "generate_signals() missing 1 required positional argument: 'params'"
        )

        with patch("crabquant.refinement.diagnostics.load_data") as mock_load:
            idx = pd.date_range("2022-01-03", periods=60, freq="B")
            mock_load.return_value = pd.DataFrame({
                "open": np.full(60, 100.0),
                "high": np.full(60, 105.0),
                "low": np.full(60, 95.0),
                "close": np.full(60, 100.0),
                "volume": np.full(60, 1e6),
            }, index=idx)

            result = run_backtest_safely(bad_module, "AAPL", "1y")

        error_info = result[3]
        assert error_info["error_type"] == "TypeError"

    def test_error_traceback_is_last_10_lines(self):
        """error_traceback should be limited to the last 10 lines."""
        from crabquant.refinement.diagnostics import run_backtest_safely, _build_error_info

        exc = RuntimeError("deep error")
        info = _build_error_info(exc)
        tb_lines = info["error_traceback"].strip().split("\n")
        assert len(tb_lines) <= 10

    def test_returns_4tuple_on_no_data(self):
        """When load_data returns None → error_info with ValueError type."""
        from crabquant.refinement.diagnostics import run_backtest_safely

        bad_module = MagicMock()

        with patch("crabquant.refinement.diagnostics.load_data", return_value=None):
            result = run_backtest_safely(bad_module, "BADTICKER", "1y")

        assert result[0] is None
        error_info = result[3]
        assert error_info["error_type"] == "ValueError"
        assert "BADTICKER" in error_info["error_message"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. load_strategy_module — returns error_info on failure
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadStrategyModuleErrorInfo:
    """Verify load_strategy_module returns (None, error_info) on failure."""

    def test_syntax_error_returns_error_info(self):
        """Strategy with syntax error → (None, error_info) with SyntaxError."""
        from crabquant.refinement.module_loader import load_strategy_module
        import tempfile, pathlib

        code = "def generate_signals(df, params):\n  return entries, exits\n"  # undefined vars
        # Actually use a real syntax error:
        code = "def generate_signals(df params):\n  pass\n"  # missing comma

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = pathlib.Path(f.name)

        try:
            result = load_strategy_module(path)
        finally:
            path.unlink(missing_ok=True)

        # Result should be a tuple (None, error_info)
        assert isinstance(result, tuple)
        assert result[0] is None
        error_info = result[1]
        assert error_info is not None
        assert "error_type" in error_info
        assert "error_message" in error_info
        # SyntaxError has a specific message
        assert error_info["error_type"] == "SyntaxError"

    def test_import_error_returns_error_info(self):
        """Strategy importing non-existent module → (None, error_info)."""
        from crabquant.refinement.module_loader import load_strategy_module
        import tempfile, pathlib

        code = (
            "import nonexistent_module_xyz\n"
            "DEFAULT_PARAMS = {}\n"
            "def generate_signals(df, params):\n"
            "    import pandas as pd\n"
            "    return pd.Series(False, index=df.index), pd.Series(False, index=df.index)\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = pathlib.Path(f.name)

        try:
            result = load_strategy_module(path)
        finally:
            path.unlink(missing_ok=True)

        assert isinstance(result, tuple)
        assert result[0] is None
        error_info = result[1]
        assert error_info is not None
        assert error_info["error_type"] == "ModuleNotFoundError"

    def test_missing_attrs_returns_error_info(self):
        """Strategy missing DEFAULT_PARAMS → (None, error_info)."""
        from crabquant.refinement.module_loader import load_strategy_module
        import tempfile, pathlib

        code = (
            "def generate_signals(df, params):\n"
            "    import pandas as pd\n"
            "    return pd.Series(False, index=df.index), pd.Series(False, index=df.index)\n"
            "# Missing DEFAULT_PARAMS!\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = pathlib.Path(f.name)

        try:
            result = load_strategy_module(path)
        finally:
            path.unlink(missing_ok=True)

        assert isinstance(result, tuple)
        assert result[0] is None
        error_info = result[1]
        assert error_info is not None
        assert error_info["error_type"] == "AttributeError"
        assert "DEFAULT_PARAMS" in error_info["error_message"]

    def test_valid_module_returns_module_only(self):
        """Valid strategy → returns module (no error_info tuple)."""
        from crabquant.refinement.module_loader import load_strategy_module
        import tempfile, pathlib

        code = (
            "DEFAULT_PARAMS = {'period': 14}\n"
            "def generate_signals(df, params):\n"
            "    import pandas as pd\n"
            "    return pd.Series(False, index=df.index), pd.Series(False, index=df.index)\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = pathlib.Path(f.name)

        try:
            result = load_strategy_module(path)
        finally:
            path.unlink(missing_ok=True)

        # load_strategy_module now returns (module, None) on success
        assert isinstance(result, tuple)
        module, err = result
        assert isinstance(module, ModuleType)
        assert err is None
        assert hasattr(module, "generate_signals")
        assert hasattr(module, "DEFAULT_PARAMS")

    def test_missing_file_returns_none(self):
        """Non-existent file → returns (None, error_info)."""
        from crabquant.refinement.module_loader import load_strategy_module

        result = load_strategy_module("/tmp/nonexistent_strategy_xyz.py")
        # Should return (None, error_info) tuple
        assert isinstance(result, tuple)
        module, err = result
        assert module is None
        assert err is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Context builder — includes crash errors in LLM context
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildContextWithCrashErrors:
    """Verify _build_crash_error_feedback and build_llm_context include crash info."""

    def test_no_crashes_returns_empty(self):
        """No crash history → empty string."""
        from crabquant.refinement.context_builder import _build_crash_error_feedback

        state = MagicMock()
        state.history = []
        result = _build_crash_error_feedback(state)
        assert result == ""

    def test_no_crashes_no_error_field(self):
        """History with crashes but no error field → empty string."""
        from crabquant.refinement.context_builder import _build_crash_error_feedback

        state = MagicMock()
        state.history = [
            {"turn": 1, "status": "backtest_crash"},
            {"turn": 2, "status": "module_load_failed"},
        ]
        result = _build_crash_error_feedback(state)
        assert result == ""

    def test_single_crash_included(self):
        """One crash with error → feedback includes error details."""
        from crabquant.refinement.context_builder import _build_crash_error_feedback

        state = MagicMock()
        state.history = [
            {
                "turn": 3,
                "status": "backtest_crash",
                "error": {
                    "error_type": "KeyError",
                    "error_message": "'Close'",
                    "error_traceback": "line 42 in generate_signals",
                },
            },
        ]
        result = _build_crash_error_feedback(state)
        assert "KeyError" in result
        assert "Close" in result
        assert "Turn 3" in result

    def test_multiple_crashes_truncated_to_3(self):
        """5 crashes → only last 3 shown."""
        from crabquant.refinement.context_builder import _build_crash_error_feedback

        state = MagicMock()
        state.history = [
            {"turn": i, "status": "backtest_crash",
             "error": {"error_type": "KeyError", "error_message": f"err{i}", "error_traceback": ""}}
            for i in range(1, 6)
        ]
        result = _build_crash_error_feedback(state)
        assert "Turn 3" in result
        assert "Turn 4" in result
        assert "Turn 5" in result
        assert "Turn 1" not in result
        assert "Turn 2" not in result

    def test_module_load_failed_included(self):
        """module_load_failed crash → included in feedback."""
        from crabquant.refinement.context_builder import _build_crash_error_feedback

        state = MagicMock()
        state.history = [
            {
                "turn": 2,
                "status": "module_load_failed",
                "error": {
                    "error_type": "SyntaxError",
                    "error_message": "invalid syntax",
                    "error_traceback": "",
                },
            },
        ]
        result = _build_crash_error_feedback(state)
        assert "SyntaxError" in result
        assert "Turn 2" in result

    def test_common_patterns_summary_included(self):
        """Feedback always includes common patterns summary when crashes exist."""
        from crabquant.refinement.context_builder import _build_crash_error_feedback

        state = MagicMock()
        state.history = [
            {"turn": 1, "status": "backtest_crash",
             "error": {"error_type": "RuntimeError", "error_message": "boom", "error_traceback": ""}},
        ]
        result = _build_crash_error_feedback(state)
        assert "Common crash patterns" in result
        assert "KeyError" in result  # common pattern listed
        assert "NameError" in result  # common pattern listed

    def test_recovery_hints_included(self):
        """Feedback includes recovery hints from get_crash_recovery_hints."""
        from crabquant.refinement.context_builder import _build_crash_error_feedback

        state = MagicMock()
        state.history = [
            {"turn": 1, "status": "backtest_crash",
             "error": {"error_type": "KeyError", "error_message": "'Close'", "error_traceback": ""}},
        ]
        result = _build_crash_error_feedback(state)
        assert "How to fix" in result or "lowercase" in result

    def test_build_llm_context_includes_crash_feedback(self):
        """build_llm_context includes crash_error_feedback key when crashes exist."""
        from crabquant.refinement.context_builder import build_llm_context

        state = MagicMock()
        state.current_turn = 2
        state.max_turns = 7
        state.sharpe_target = 1.5
        state.tickers = ["AAPL"]
        state.best_sharpe = 0.0
        state.best_composite_score = -999.0
        state.best_turn = 0
        state.history = [
            {"turn": 1, "status": "backtest_crash",
             "error": {"error_type": "KeyError", "error_message": "'Close'", "error_traceback": ""}},
        ]

        context = build_llm_context(state)
        assert "crash_error_feedback" in context
        assert "KeyError" in context["crash_error_feedback"]

    def test_build_llm_context_no_crash_feedback_when_clean(self):
        """build_llm_context omits crash_error_feedback when no crashes."""
        from crabquant.refinement.context_builder import build_llm_context

        state = MagicMock()
        state.current_turn = 2
        state.max_turns = 7
        state.sharpe_target = 1.5
        state.tickers = ["AAPL"]
        state.best_sharpe = 0.0
        state.best_composite_score = -999.0
        state.best_turn = 0
        state.history = [
            {"turn": 1, "status": "completed", "sharpe": 1.2},
        ]

        context = build_llm_context(state)
        assert "crash_error_feedback" not in context


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Error pattern recovery hints
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrashRecoveryHints:
    """Verify get_crash_recovery_hints and build_crash_guidance."""

    def test_keyerror_hint(self):
        """KeyError → column name hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("KeyError", "'Close'")
        assert hint != ""
        assert "lowercase" in hint or "column" in hint.lower()

    def test_nameerror_not_defined_hint(self):
        """NameError with 'not defined' → import hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("NameError", "name 'ema' is not defined")
        assert hint != ""
        assert "import" in hint.lower() or "indicator" in hint.lower()

    def test_typeerror_generate_signals_hint(self):
        """TypeError mentioning generate_signals → signature hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("TypeError", "generate_signals() missing 1 required")
        assert hint != ""
        assert "generate_signals" in hint
        assert "df" in hint and "params" in hint

    def test_attributeerror_hint(self):
        """AttributeError → bracket notation hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("AttributeError", "'DataFrame' object has no attribute 'Close'")
        assert hint != ""
        assert "df[" in hint or "bracket" in hint.lower()

    def test_importerror_hint(self):
        """ImportError → allowed modules hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("ImportError", "No module named 'talib'")
        assert hint != ""
        assert "standard library" in hint or "pandas" in hint

    def test_valueerror_array_hint(self):
        """ValueError with 'array' → NaN handling hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("ValueError", "array must not contain infs or NaNs")
        assert hint != ""
        assert "NaN" in hint or "fillna" in hint or "dropna" in hint

    def test_valueerror_truth_hint(self):
        """ValueError with 'Truth value' → boolean context hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("ValueError", "Truth value of a Series is ambiguous")
        assert hint != ""
        assert ".any()" in hint or ".all()" in hint

    def test_zerodivisionerror_hint(self):
        """ZeroDivisionError → epsilon hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("ZeroDivisionError", "division by zero")
        assert hint != ""
        assert "epsilon" in hint.lower() or "1e-10" in hint

    def test_syntaxerror_hint(self):
        """SyntaxError → syntax fix hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("SyntaxError", "invalid syntax")
        assert hint != ""
        assert "syntax" in hint.lower()

    def test_unknown_error_no_hint(self):
        """Completely unknown error type → empty hint."""
        from crabquant.refinement.prompts import get_crash_recovery_hints

        hint = get_crash_recovery_hints("CustomWeirdError", "something strange")
        assert hint == ""

    def test_build_crash_guidance_with_info(self):
        """build_crash_guidance produces formatted text."""
        from crabquant.refinement.prompts import build_crash_guidance

        info = {
            "error_type": "KeyError",
            "error_message": "'Close'",
            "error_traceback": "",
        }
        guidance = build_crash_guidance(info)
        assert "KeyError" in guidance
        assert "Close" in guidance
        assert "Fix:" in guidance or "fix" in guidance.lower()

    def test_build_crash_guidance_none(self):
        """build_crash_guidance with None → empty string."""
        from crabquant.refinement.prompts import build_crash_guidance

        assert build_crash_guidance(None) == ""

    def test_build_crash_guidance_unknown_error(self):
        """build_crash_guidance with unknown error → generic fix text."""
        from crabquant.refinement.prompts import build_crash_guidance

        info = {"error_type": "AlienError", "error_message": "not from this planet"}
        guidance = build_crash_guidance(info)
        assert "AlienError" in guidance
        assert "Fix:" in guidance or "fix" in guidance.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. build_failure_guidance — handles crash modes
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFailureGuidanceCrashModes:
    """Verify build_failure_guidance returns guidance for crash failure modes."""

    def test_backtest_crash_guidance(self):
        """failure_mode='backtest_crash' → actionable guidance."""
        from crabquant.refinement.prompts import build_failure_guidance

        guidance = build_failure_guidance("backtest_crash")
        assert guidance != ""
        assert "crash" in guidance.lower() or "code" in guidance.lower()
        assert "column" in guidance.lower() or "fix" in guidance.lower()

    def test_module_load_failed_guidance(self):
        """failure_mode='module_load_failed' → actionable guidance."""
        from crabquant.refinement.prompts import build_failure_guidance

        guidance = build_failure_guidance("module_load_failed")
        assert guidance != ""
        assert "syntax" in guidance.lower() or "import" in guidance.lower()

    def test_existing_modes_still_work(self):
        """Existing failure modes still return guidance (no regression)."""
        from crabquant.refinement.prompts import build_failure_guidance

        # These should all produce non-empty guidance
        for mode in ["too_few_trades_for_validation", "validation_failed",
                      "regime_fragility", "low_sharpe"]:
            guidance = build_failure_guidance(mode, total_trades=5)
            assert guidance != "", f"Expected guidance for mode={mode!r}"

    def test_unknown_mode_returns_empty(self):
        """Unknown failure mode → empty string."""
        from crabquant.refinement.prompts import build_failure_guidance

        assert build_failure_guidance("totally_unknown_mode") == ""
