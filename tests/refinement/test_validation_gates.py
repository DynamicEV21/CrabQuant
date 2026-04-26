"""Tests for validation_gates: Gate 1 (syntax+import) and Gate 2 (signal sanity)."""

import numpy as np
import pandas as pd
import pytest

from crabquant.refinement.validation_gates import (
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

    def test_missing_param_grid(self):
        ok, errors = gate_syntax(MISSING_PARAM_GRID)
        assert ok is False
        assert any("PARAM_GRID" in e for e in errors)

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
