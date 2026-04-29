"""Tests for validation_gates: Gate 1 (syntax+import) and Gate 2 (signal sanity)."""

import numpy as np
import pandas as pd
import pytest

from crabquant.refinement.validation_gates import (
    _load_module_from_code,
    gate_signal_sanity,
    gate_smoke_backtest,
    gate_syntax,
    run_validation_gates,
)


# ── Mock data factory ───────────────────────────────────────────────────────

def make_mock_df(n: int = 252) -> pd.DataFrame:
    """Deterministic OHLCV DataFrame for testing (no yfinance calls)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    returns = rng.normal(0, 0.01, n)
    close = 100.0 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


# ── Fixture strategy strings ────────────────────────────────────────────────

VALID_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {"step": 20}
PARAM_GRID = {"step": [10, 20, 30]}
DESCRIPTION = "Deterministic test strategy — signals every N bars."

def generate_signals(df: pd.DataFrame, params=None) -> tuple:
    p = {**DEFAULT_PARAMS, **(params or {})}
    n = len(df)
    entries = pd.Series(False, index=df.index, dtype=bool)
    exits = pd.Series(False, index=df.index, dtype=bool)
    step = p["step"]
    for i in range(step, n - step, step * 2):
        entries.iloc[i] = True
    exits.iloc[i + step] = True
    return entries, exits
"""

SYNTAX_ERROR_STRATEGY = """\
def generate_signals(df params=None):
    pass
"""

MISSING_GENERATE_SIGNALS = """\
import pandas as pd

DEFAULT_PARAMS = {"window": 20}
PARAM_GRID = {"window": [20]}
DESCRIPTION = "Missing generate_signals."
"""

MISSING_DEFAULT_PARAMS = """\
import pandas as pd

PARAM_GRID = {"window": [20]}
DESCRIPTION = "Missing DEFAULT_PARAMS."

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""

MISSING_PARAM_GRID = """\
import pandas as pd

DEFAULT_PARAMS = {"window": 20}
DESCRIPTION = "Missing PARAM_GRID."

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""

BAD_IMPORT_STRATEGY = """\
import pandas as pd
import nonexistent_package_xyz_12345

DEFAULT_PARAMS = {"window": 20}
PARAM_GRID = {"window": [20]}
DESCRIPTION = "Has bad import."

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""

ZERO_SIGNALS_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Always returns zero entry signals."

def generate_signals(df, params=None):
    entries = pd.Series(False, index=df.index, dtype=bool)
    exits = pd.Series(False, index=df.index, dtype=bool)
    return entries, exits
"""

OVERTRADING_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Fires on every bar — overtrading."

def generate_signals(df, params=None):
    entries = pd.Series(True, index=df.index, dtype=bool)
    exits = pd.Series(False, index=df.index, dtype=bool)
    return entries, exits
"""

NON_BOOL_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Returns float Series instead of bool."

def generate_signals(df, params=None):
    entries = df["close"].rolling(20).mean()
    exits = df["close"].rolling(10).mean()
    return entries, exits
"""

RUNTIME_ERROR_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {"window": 20}
PARAM_GRID = {"window": [20]}
DESCRIPTION = "Raises ValueError at runtime."

def generate_signals(df, params=None):
    raise ValueError("intentional runtime error")
"""

NAN_SIGNALS_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Injects None (NaN) into signals."

def generate_signals(df, params=None):
    n = len(df)
    # object dtype allows None, which becomes NaN
    entries = pd.Series([True if i % 5 == 0 else False for i in range(n)],
                        index=df.index, dtype=object)
    entries.iloc[5] = None
    exits = pd.Series(False, index=df.index)
    return entries, exits
"""

WRONG_INDEX_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Returns signals with wrong (RangeIndex) index."

def generate_signals(df, params=None):
    n = len(df)
    entries = pd.Series(False, index=range(n), dtype=bool)
    exits = pd.Series(False, index=range(n), dtype=bool)
    entries.iloc[10] = True
    return entries, exits
"""

MISSING_DESCRIPTION = """\
import pandas as pd

DEFAULT_PARAMS = {"window": 20}

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""

MISSING_ALL_REQUIRED = """\
import pandas as pd

def some_other_function():
    pass
"""

BAD_FROM_IMPORT_STRATEGY = """\
import pandas as pd
from nonexistent_xyz_123 import something

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Bad from-import."

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""

RETURNS_SINGLE_VALUE_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Returns single value instead of tuple."

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool)
"""

RETURNS_THREE_VALUES_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Returns 3-tuple instead of 2-tuple."

def generate_signals(df, params=None):
    return (pd.Series(False, index=df.index, dtype=bool),
            pd.Series(False, index=df.index, dtype=bool),
            "extra")
"""

NON_SERIES_ENTRIES_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Entries is a list, not a Series."

def generate_signals(df, params=None):
    return [True, False, True], pd.Series(False, index=df.index, dtype=bool)
"""

EXACT_50_PERCENT_SIGNALS = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Exactly 50% entries — boundary case."

def generate_signals(df, params=None):
    n = len(df)
    entries = pd.Series([i < n // 2 for i in range(n)], index=df.index, dtype=bool)
    exits = pd.Series(False, index=df.index, dtype=bool)
    return entries, exits
"""

JUST_UNDER_50_PERCENT_SIGNALS = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "49% entries — should not trigger overtrading."

def generate_signals(df, params=None):
    n = len(df)
    entries = pd.Series([i < int(n * 0.49) for i in range(n)], index=df.index, dtype=bool)
    exits = pd.Series(False, index=df.index, dtype=bool)
    return entries, exits
"""

NAN_EXITS_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Exits contain NaN values."

def generate_signals(df, params=None):
    n = len(df)
    entries = pd.Series(False, index=df.index, dtype=bool)
    exits = pd.Series([True if i % 10 == 0 else False for i in range(n)],
                      index=df.index, dtype=object)
    exits.iloc[3] = None
    return entries, exits
"""

WRONG_EXITS_INDEX_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Entries correct index but exits have wrong index."

def generate_signals(df, params=None):
    n = len(df)
    entries = pd.Series(False, index=df.index, dtype=bool)
    entries.iloc[10] = True
    exits = pd.Series(False, index=range(n), dtype=bool)
    return entries, exits
"""

EMPTY_STRATEGY = ""

WHITESPACE_ONLY_STRATEGY = "   \n\t\n  "

NONE_RETURN_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Returns None."

def generate_signals(df, params=None):
    return None
"""

INDENTATION_ERROR_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}

def generate_signals(df, params=None):
    entries = pd.Series(False, index=df.index)
  exits = pd.Series(False, index=df.index)
    return entries, exits

DESCRIPTION = "Bad indentation."
"""

TYPE_ERROR_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Raises TypeError at runtime."

def generate_signals(df, params=None):
    raise TypeError("intentional type error")
"""


# ── Gate 1: Syntax + Import ─────────────────────────────────────────────────

class TestGateSyntax:
    def test_valid_strategy_passes(self):
        ok, errors = gate_syntax(VALID_STRATEGY)
        assert ok is True
        assert errors == []

    def test_syntax_error_caught(self):
        ok, errors = gate_syntax(SYNTAX_ERROR_STRATEGY)
        assert ok is False
        assert any("SyntaxError" in e for e in errors)

    def test_missing_generate_signals(self):
        ok, errors = gate_syntax(MISSING_GENERATE_SIGNALS)
        assert ok is False
        assert any("generate_signals" in e for e in errors)

    def test_missing_default_params(self):
        ok, errors = gate_syntax(MISSING_DEFAULT_PARAMS)
        assert ok is False
        assert any("DEFAULT_PARAMS" in e for e in errors)

    def test_missing_param_grid_is_optional(self):
        """PARAM_GRID is not required for LLM-generated strategies."""
        ok, errors = gate_syntax(MISSING_PARAM_GRID)
        assert ok is True  # Should pass without PARAM_GRID
        assert not any("PARAM_GRID" in e for e in errors)

    def test_bad_import_caught(self):
        ok, errors = gate_syntax(BAD_IMPORT_STRATEGY)
        assert ok is False
        assert any("ImportError" in e for e in errors)

    def test_empty_code_fails(self):
        ok, errors = gate_syntax("")
        assert ok is False
        assert len(errors) > 0

    def test_stdlib_imports_not_flagged(self):
        code = """\
import os
import sys
from typing import Optional
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Uses stdlib — should not raise ImportError."

def generate_signals(df, params=None):
    return None, None
"""
        ok, errors = gate_syntax(code)
        # os / sys / typing must NOT appear in ImportError messages
        assert not any("ImportError" in e and ("os" in e or "sys" in e or "typing" in e)
                       for e in errors)

    def test_returns_tuple_of_bool_and_list(self):
        result = gate_syntax(VALID_STRATEGY)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_whitespace_only_code_fails(self):
        """Whitespace-only code should be treated as empty."""
        ok, errors = gate_syntax(WHITESPACE_ONLY_STRATEGY)
        assert ok is False
        assert any("Empty" in e or "empty" in e.lower() for e in errors)

    def test_missing_description_caught(self):
        """DESCRIPTION is a required attribute."""
        ok, errors = gate_syntax(MISSING_DESCRIPTION)
        assert ok is False
        assert any("DESCRIPTION" in e for e in errors)

    def test_missing_all_required_attrs(self):
        """Strategy missing generate_signals, DEFAULT_PARAMS, and DESCRIPTION."""
        ok, errors = gate_syntax(MISSING_ALL_REQUIRED)
        assert ok is False
        assert any("generate_signals" in e for e in errors)
        assert any("DEFAULT_PARAMS" in e for e in errors)
        assert any("DESCRIPTION" in e for e in errors)

    def test_bad_from_import_caught(self):
        """from X import Y where X is not installed."""
        ok, errors = gate_syntax(BAD_FROM_IMPORT_STRATEGY)
        assert ok is False
        assert any("ImportError" in e for e in errors)

    def test_syntax_error_reports_line_number(self):
        """Syntax errors should include line number information."""
        ok, errors = gate_syntax(SYNTAX_ERROR_STRATEGY)
        assert ok is False
        error_msg = errors[0]
        assert "line" in error_msg.lower() or "SyntaxError" in error_msg

    def test_numpy_import_not_flagged(self):
        """numpy is a real third-party package — should not be flagged."""
        code = """\
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Uses numpy."

def generate_signals(df, params=None):
    return None, None
"""
        ok, errors = gate_syntax(code)
        assert not any("ImportError" in e for e in errors)

    def test_multiple_errors_reported(self):
        """Both syntax error and missing attrs reported when code has syntax error."""
        # Syntax error prevents checking further attributes
        ok, errors = gate_syntax(SYNTAX_ERROR_STRATEGY)
        assert ok is False
        assert len(errors) >= 1

    def test_minimal_valid_strategy(self):
        """Strategy with only the three required attrs — no PARAM_GRID, no other imports."""
        code = """\
import pandas as pd

DEFAULT_PARAMS = {"x": 1}
DESCRIPTION = "Minimal."

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""
        ok, errors = gate_syntax(code)
        assert ok is True
        assert errors == []


# ── Gate 2: Signal Sanity ──────────────────────────────────────────────────

class TestGateSignalSanity:
    def setup_method(self):
        self.df = make_mock_df()

    def test_valid_strategy_passes(self):
        ok, errors = gate_signal_sanity(VALID_STRATEGY, df=self.df)
        assert ok is True, f"Expected pass but got errors: {errors}"
        assert errors == []

    def test_zero_signals_caught(self):
        ok, errors = gate_signal_sanity(ZERO_SIGNALS_STRATEGY, df=self.df)
        assert ok is False
        assert any("zero" in e.lower() or "signal" in e.lower() or "entr" in e.lower()
                   for e in errors)

    def test_overtrading_caught(self):
        ok, errors = gate_signal_sanity(OVERTRADING_STRATEGY, df=self.df)
        assert ok is False
        assert any("overtrad" in e.lower() for e in errors)

    def test_non_bool_entries_caught(self):
        ok, errors = gate_signal_sanity(NON_BOOL_STRATEGY, df=self.df)
        assert ok is False
        assert any("bool" in e.lower() or "entries" in e.lower() for e in errors)

    def test_runtime_error_caught(self):
        ok, errors = gate_signal_sanity(RUNTIME_ERROR_STRATEGY, df=self.df)
        assert ok is False
        assert any("runtime" in e.lower() or "error" in e.lower() for e in errors)

    def test_wrong_index_caught(self):
        ok, errors = gate_signal_sanity(WRONG_INDEX_STRATEGY, df=self.df)
        assert ok is False
        assert any("index" in e.lower() for e in errors)

    def test_nan_signals_caught(self):
        ok, errors = gate_signal_sanity(NAN_SIGNALS_STRATEGY, df=self.df)
        assert ok is False
        assert any("nan" in e.lower() or "na" in e.lower() for e in errors)

    def test_returns_tuple_of_bool_and_list(self):
        result = gate_signal_sanity(VALID_STRATEGY, df=self.df)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_import_exec_error_caught(self):
        # Code that compiles but fails on import (division by zero at module level)
        bad_code = """\
import pandas as pd
x = 1 / 0  # module-level error

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Crashes on import."

def generate_signals(df, params=None):
    return pd.Series(False, index=df.index, dtype=bool), pd.Series(False, index=df.index, dtype=bool)
"""
        ok, errors = gate_signal_sanity(bad_code, df=self.df)
        assert ok is False
        assert any("error" in e.lower() for e in errors)

    def test_nan_exits_caught(self):
        """NaN values in exits should also be caught."""
        ok, errors = gate_signal_sanity(NAN_EXITS_STRATEGY, df=self.df)
        assert ok is False
        assert any("nan" in e.lower() or "na" in e.lower() for e in errors)

    def test_wrong_exits_index_caught(self):
        """Exits with mismatched index should be caught."""
        ok, errors = gate_signal_sanity(WRONG_EXITS_INDEX_STRATEGY, df=self.df)
        assert ok is False
        assert any("index" in e.lower() for e in errors)

    def test_single_value_return_caught(self):
        """Returning a single value instead of a tuple should fail."""
        ok, errors = gate_signal_sanity(RETURNS_SINGLE_VALUE_STRATEGY, df=self.df)
        assert ok is False
        assert any("tuple" in e.lower() for e in errors)

    def test_three_values_return_caught(self):
        """Returning a 3-tuple instead of 2-tuple should fail."""
        ok, errors = gate_signal_sanity(RETURNS_THREE_VALUES_STRATEGY, df=self.df)
        assert ok is False
        assert any("tuple" in e.lower() for e in errors)

    def test_non_series_entries_caught(self):
        """Entries that are a list, not pd.Series, should fail."""
        ok, errors = gate_signal_sanity(NON_SERIES_ENTRIES_STRATEGY, df=self.df)
        assert ok is False
        assert any("entries" in e.lower() or "series" in e.lower() for e in errors)

    def test_none_return_caught(self):
        """Returning None from generate_signals should fail."""
        ok, errors = gate_signal_sanity(NONE_RETURN_STRATEGY, df=self.df)
        assert ok is False

    def test_type_error_caught(self):
        """TypeError raised at runtime should be caught."""
        ok, errors = gate_signal_sanity(TYPE_ERROR_STRATEGY, df=self.df)
        assert ok is False
        assert any("runtime" in e.lower() or "error" in e.lower() for e in errors)

    def test_syntax_error_code_fails_gate2(self):
        """Code with syntax errors cannot be loaded — should fail Gate 2."""
        ok, errors = gate_signal_sanity(SYNTAX_ERROR_STRATEGY, df=self.df)
        assert ok is False
        assert any("error" in e.lower() for e in errors)

    def test_empty_code_fails_gate2(self):
        """Empty code cannot produce signals."""
        ok, errors = gate_signal_sanity("", df=self.df)
        assert ok is False

    def test_exactly_50_percent_boundary(self):
        """Exactly 50% entries is at the boundary — should NOT trigger overtrading (uses >, not >=)."""
        ok, errors = gate_signal_sanity(EXACT_50_PERCENT_SIGNALS, df=self.df)
        # > 0.5 means exactly 50% passes
        assert ok is True, f"Expected pass but got errors: {errors}"

    def test_under_50_percent_not_overtrading(self):
        """49% entries should NOT trigger overtrading."""
        ok, errors = gate_signal_sanity(JUST_UNDER_50_PERCENT_SIGNALS, df=self.df)
        assert ok is True, f"Expected pass but got errors: {errors}"

    def test_multiple_dtype_and_index_errors(self):
        """Non-bool entries with wrong index should report both errors."""
        # NON_BOOL_STRATEGY has correct index but wrong dtype
        # Let's test a strategy with both wrong dtype and wrong index
        bad_code = """\
import pandas as pd

DEFAULT_PARAMS = {}
PARAM_GRID = {}
DESCRIPTION = "Wrong dtype AND wrong index."

def generate_signals(df, params=None):
    n = len(df)
    entries = df["close"].rolling(20).mean().iloc[:n]  # may have NaN at start
    exits = pd.Series(False, index=range(n), dtype=bool)
    return entries, exits
"""
        ok, errors = gate_signal_sanity(bad_code, df=self.df)
        assert ok is False
        assert len(errors) >= 2  # Should catch both dtype and index issues


# ── Gate 3: Smoke Backtest (placeholder) ───────────────────────────────────

class TestGateSmokeBacktest:
    def test_always_passes(self):
        ok, errors = gate_smoke_backtest(VALID_STRATEGY)
        assert ok is True
        assert errors == []

    def test_passes_even_for_broken_code(self):
        ok, errors = gate_smoke_backtest(SYNTAX_ERROR_STRATEGY)
        assert ok is True
        assert errors == []

    def test_passes_with_empty_code(self):
        ok, errors = gate_smoke_backtest("")
        assert ok is True
        assert errors == []

    def test_passes_with_none_code(self):
        """gate_smoke_backtest should handle any input (placeholder)."""
        ok, errors = gate_smoke_backtest("garbage")
        assert ok is True
        assert errors == []


# ── _load_module_from_code ──────────────────────────────────────────────────

class TestLoadModuleFromCode:
    def test_valid_code_returns_module(self):
        module, err = _load_module_from_code(VALID_STRATEGY)
        assert module is not None
        assert err == ""
        assert hasattr(module, "generate_signals")

    def test_syntax_error_returns_none(self):
        module, err = _load_module_from_code(SYNTAX_ERROR_STRATEGY)
        assert module is None
        assert len(err) > 0

    def test_runtime_error_in_function_still_loads(self):
        """Runtime errors inside functions don't prevent module loading."""
        module, err = _load_module_from_code(RUNTIME_ERROR_STRATEGY)
        # The module loads fine — the error only triggers when generate_signals is called
        assert module is not None
        assert err == ""

    def test_module_level_error_returns_none(self):
        """Module-level code that crashes should return None."""
        bad_code = "x = 1 / 0\n"
        module, err = _load_module_from_code(bad_code)
        assert module is None
        assert len(err) > 0

    def test_empty_code_loads_as_empty_module(self):
        """Empty code loads as a valid (empty) module."""
        module, err = _load_module_from_code("")
        # An empty module is technically loadable
        assert err == ""

    def test_module_has_correct_attributes(self):
        module, err = _load_module_from_code(VALID_STRATEGY)
        assert module is not None
        assert hasattr(module, "generate_signals")
        assert hasattr(module, "DEFAULT_PARAMS")
        assert hasattr(module, "DESCRIPTION")

    def test_module_cleanup_on_error(self):
        """Failed module loads should clean up sys.modules."""
        import sys
        bad_code = "x = 1 / 0\n"
        # Count modules starting with _vgate_ before
        vgate_before = [k for k in sys.modules if k.startswith("_vgate_")]
        module, err = _load_module_from_code(bad_code)
        assert module is None
        # After error, no new _vgate_ modules should linger
        vgate_after = [k for k in sys.modules if k.startswith("_vgate_")]
        assert len(vgate_after) == len(vgate_before)


# ── run_validation_gates ────────────────────────────────────────────────────

class TestRunValidationGates:
    def setup_method(self):
        self.df = make_mock_df()

    def test_valid_strategy_passes_all_gates(self):
        ok, errors = run_validation_gates(VALID_STRATEGY, df=self.df)
        assert ok is True, f"Expected pass but got errors: {errors}"
        assert errors == []

    def test_gate1_failure_short_circuits(self):
        ok, errors = run_validation_gates(SYNTAX_ERROR_STRATEGY, df=self.df)
        assert ok is False
        assert any("SyntaxError" in e for e in errors)

    def test_gate2_failure_reported(self):
        ok, errors = run_validation_gates(ZERO_SIGNALS_STRATEGY, df=self.df)
        assert ok is False
        assert len(errors) > 0

    def test_returns_tuple_of_bool_and_list(self):
        result = run_validation_gates(VALID_STRATEGY, df=self.df)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    def test_empty_code_fails(self):
        ok, errors = run_validation_gates("", df=self.df)
        assert ok is False

    def test_overtrading_strategy_fails(self):
        ok, errors = run_validation_gates(OVERTRADING_STRATEGY, df=self.df)
        assert ok is False

    def test_bad_import_fails_at_gate1(self):
        """Bad import should fail at Gate 1, never reaching Gate 2."""
        ok, errors = run_validation_gates(BAD_IMPORT_STRATEGY, df=self.df)
        assert ok is False
        assert any("ImportError" in e for e in errors)

    def test_missing_required_attrs_fails(self):
        """Missing generate_signals should fail at Gate 1."""
        ok, errors = run_validation_gates(MISSING_GENERATE_SIGNALS, df=self.df)
        assert ok is False

    def test_gate_chaining_order(self):
        """Gate 1 errors should be reported (not Gate 2 errors) for syntax-broken code."""
        ok, errors = run_validation_gates(SYNTAX_ERROR_STRATEGY, df=self.df)
        # Should be Gate 1 error (syntax), not Gate 2 (signal)
        assert any("SyntaxError" in e for e in errors)
        # Should NOT have signal-related errors
        assert not any("runtime" in e.lower() for e in errors)

    def test_custom_ticker_and_period_params(self):
        """Ticker and period should be accepted as parameters (Gate 3 placeholder)."""
        ok, errors = run_validation_gates(VALID_STRATEGY, ticker="AAPL", period="6mo", df=self.df)
        assert ok is True, f"Expected pass but got errors: {errors}"

    def test_whitespace_only_code_fails(self):
        ok, errors = run_validation_gates(WHITESPACE_ONLY_STRATEGY, df=self.df)
        assert ok is False
