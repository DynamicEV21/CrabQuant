"""
Three-gate pre-backtest validation for LLM-generated strategy code.

Gate 1 (~0.1s): Syntax + import check via AST.
Gate 2 (~1s):   Signal sanity — run generate_signals, verify output shape/dtype.
Gate 3 (~10s):  Smoke backtest — quick 6-month backtest, check metrics.
"""

import ast
import importlib
import importlib.util
import sys
import tempfile
import uuid
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


_STD_LIB: frozenset[str] = frozenset(sys.stdlib_module_names)
_REQUIRED_ATTRS: frozenset[str] = frozenset(
    {"generate_signals", "DEFAULT_PARAMS", "DESCRIPTION"}
)


def gate_syntax(code: str) -> tuple[bool, list[str]]:
    """Gate 1: AST parse + required attribute check (~0.1s)."""
    errors: list[str] = []

    if not code or not code.strip():
        return (False, ["Empty code"])

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (False, [f"SyntaxError: line {e.lineno}: {e.msg}"])

    # Verify all top-level imported packages are installed
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
        elif isinstance(node, ast.Import):
            module_name = node.names[0].name
        else:
            continue
        top = module_name.split(".")[0]
        if not top or top in _STD_LIB:
            continue
        try:
            importlib.import_module(top)
        except ImportError:
            errors.append(f"ImportError: '{top}' not installed")

    # Collect names defined at module level (functions + assignments)
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)

    missing = _REQUIRED_ATTRS - defined
    if missing:
        errors.append(f"Missing required: {sorted(missing)}")

    # Verify generate_signals has correct signature (df, params)
    if "generate_signals" in defined:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generate_signals":
                args = [a.arg for a in node.args.args]
                if args != ["df", "params"]:
                    errors.append(
                        f"generate_signals(df, params) signature expected, "
                        f"got ({', '.join(args)})"
                    )
                break

    return (len(errors) == 0, errors)


def _load_module_from_code(code: str) -> tuple[ModuleType | None, str]:
    """Write code to a temp file and import it. Returns (module, error_msg)."""
    module_name = f"_vgate_{uuid.uuid4().hex[:8]}"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="vgate_", delete=False
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)

        spec = importlib.util.spec_from_file_location(module_name, str(tmp_path))
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(tmp_path)
        sys.modules[module_name] = module

        source = tmp_path.read_text(encoding="utf-8")
        code_obj = compile(source, str(tmp_path), "exec")
        exec(code_obj, module.__dict__)

        return module, ""
    except Exception as e:
        if module_name in sys.modules:
            del sys.modules[module_name]
        return None, str(e)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def gate_signal_sanity(
    code: str,
    df: pd.DataFrame | None = None,
    ticker: str = "SPY",
    period: str = "1y",
) -> tuple[bool, list[str]]:
    """Gate 2: Load strategy module and verify signal output (~1s).

    Pass ``df`` directly to avoid a live yfinance call (useful in tests).
    """
    errors: list[str] = []

    module, err = _load_module_from_code(code)
    if module is None:
        return (False, [f"Import/exec error: {err}"])

    if df is None:
        from crabquant.data import load_data
        df = load_data(ticker, period=period)

    try:
        result = module.generate_signals(df, module.DEFAULT_PARAMS)
    except Exception as e:
        return (False, [f"Runtime error in generate_signals: {e}"])

    if not isinstance(result, tuple) or len(result) != 2:
        return (False, [f"generate_signals must return (entries, exits) tuple, got {type(result).__name__}"])

    entries, exits = result

    # dtype checks
    if not isinstance(entries, pd.Series) or entries.dtype != bool:
        errors.append(
            f"entries must be pd.Series[bool], got {type(entries).__name__}"
            + (f"[{entries.dtype}]" if isinstance(entries, pd.Series) else "")
        )
    if not isinstance(exits, pd.Series) or exits.dtype != bool:
        errors.append(
            f"exits must be pd.Series[bool], got {type(exits).__name__}"
            + (f"[{exits.dtype}]" if isinstance(exits, pd.Series) else "")
        )

    # Index alignment
    if isinstance(entries, pd.Series) and not entries.index.equals(df.index):
        errors.append("entries index does not match input DataFrame index")
    if isinstance(exits, pd.Series) and not exits.index.equals(df.index):
        errors.append("exits index does not match input DataFrame index")

    # NaN / NA check (runs even when dtype is wrong, for extra diagnostics)
    if isinstance(entries, pd.Series) and entries.isna().any():
        errors.append("entries contain NaN/NA values")
    if isinstance(exits, pd.Series) and exits.isna().any():
        errors.append("exits contain NaN/NA values")

    # Signal count checks — only when dtype is correct
    if isinstance(entries, pd.Series) and entries.dtype == bool:
        n_entries = int(entries.sum())
        if n_entries == 0:
            errors.append("Zero entry signals generated on test data")
        elif n_entries > len(entries) * 0.5:
            errors.append(
                f"Overtrading: {n_entries} entry signals on {len(entries)} bars"
            )

    return (len(errors) == 0, errors)


def gate_smoke_backtest(code: str, ticker: str = "SPY") -> tuple[bool, list[str]]:
    """Gate 3: quick 6-month smoke backtest validation.

    Delegates to the real implementation in gate3_smoke.
    """
    from crabquant.refinement.gate3_smoke import gate_smoke_backtest as _real

    return _real(code, ticker=ticker)


def run_validation_gates(
    strategy_code: str,
    ticker: str = "SPY",
    period: str = "1y",
    df: pd.DataFrame | None = None,
) -> tuple[bool, list[str]]:
    """Run all validation gates. Short-circuits on first failure.

    Returns:
        (passed, errors) — errors is empty when passed is True.
    """
    ok, errors = gate_syntax(strategy_code)
    if not ok:
        return (False, errors)

    ok, errors = gate_signal_sanity(strategy_code, df=df, ticker=ticker, period=period)
    if not ok:
        return (False, errors)

    ok, errors = gate_smoke_backtest(strategy_code, ticker=ticker)
    if not ok:
        return (False, errors)

    return (True, [])
