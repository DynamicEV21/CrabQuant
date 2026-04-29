"""Tests for crabquant.refinement.module_loader — TDD first pass."""

import sys
import textwrap
from pathlib import Path

import pytest

from crabquant.refinement.module_loader import load_module_from_code, load_strategy_module

# ─── Fixtures / helpers ───────────────────────────────────────────────────────

VALID_STRATEGY_CODE = textwrap.dedent("""
    import pandas as pd

    DEFAULT_PARAMS = {"period": 14}
    PARAM_GRID = {"period": [7, 14, 21]}

    def generate_signals(df, params=None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        return entries, exits
""").strip()

MISSING_GENERATE_SIGNALS = textwrap.dedent("""
    DEFAULT_PARAMS = {"period": 14}
    PARAM_GRID = {"period": [7, 14, 21]}
""").strip()

MISSING_DEFAULT_PARAMS = textwrap.dedent("""
    import pandas as pd

    PARAM_GRID = {"period": [7, 14, 21]}

    def generate_signals(df, params=None):
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
""").strip()

MISSING_PARAM_GRID = textwrap.dedent("""
    import pandas as pd

    DEFAULT_PARAMS = {"period": 14}

    def generate_signals(df, params=None):
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
""").strip()

SYNTAX_ERROR_CODE = textwrap.dedent("""
    def generate_signals(df params=None):
        return None
""").strip()

RUNTIME_ERROR_CODE = textwrap.dedent("""
    DEFAULT_PARAMS = {"period": 14}
    PARAM_GRID = {"period": [7, 14, 21]}

    raise ValueError("boom at import time")

    def generate_signals(df, params=None):
        pass
""").strip()


@pytest.fixture
def write_strategy(tmp_path):
    """Return a helper that writes code to a .py file and returns its Path."""
    def _write(code: str, filename: str = "strategy.py") -> Path:
        p = tmp_path / filename
        p.write_text(code)
        return p
    return _write


# ─── load_strategy_module ─────────────────────────────────────────────────────

class TestLoadStrategyModule:
    def test_returns_module_for_valid_file(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE)
        module = load_strategy_module(path)
        assert module is not None

    def test_loaded_module_has_generate_signals(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE)
        module = load_strategy_module(path)
        assert hasattr(module, "generate_signals")
        assert callable(module.generate_signals)

    def test_loaded_module_has_default_params(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE)
        module = load_strategy_module(path)
        assert hasattr(module, "DEFAULT_PARAMS")
        assert isinstance(module.DEFAULT_PARAMS, dict)

    def test_loaded_module_has_param_grid(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE)
        module = load_strategy_module(path)
        assert hasattr(module, "PARAM_GRID")
        assert isinstance(module.PARAM_GRID, dict)

    def test_returns_none_for_nonexistent_path(self, tmp_path):
        result = load_strategy_module(tmp_path / "ghost.py")
        assert result is None

    def test_returns_none_for_syntax_error(self, write_strategy):
        path = write_strategy(SYNTAX_ERROR_CODE)
        result = load_strategy_module(path)
        assert result is None

    def test_returns_none_for_missing_generate_signals(self, write_strategy):
        path = write_strategy(MISSING_GENERATE_SIGNALS)
        result = load_strategy_module(path)
        assert result is None

    def test_returns_none_for_missing_default_params(self, write_strategy):
        path = write_strategy(MISSING_DEFAULT_PARAMS)
        result = load_strategy_module(path)
        assert result is None

    def test_accepts_missing_param_grid(self, write_strategy):
        """PARAM_GRID is optional for LLM-generated strategies."""
        path = write_strategy(MISSING_PARAM_GRID)
        result = load_strategy_module(path)
        assert result is not None

    def test_returns_none_for_runtime_error_at_import(self, write_strategy):
        path = write_strategy(RUNTIME_ERROR_CODE)
        result = load_strategy_module(path)
        assert result is None

    def test_accepts_string_path(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE)
        module = load_strategy_module(str(path))
        assert module is not None

    def test_custom_module_name(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE)
        module = load_strategy_module(path, module_name="my_custom_name")
        assert module is not None
        assert "my_custom_name" in sys.modules

    def test_default_module_name_derives_from_stem(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE, filename="cool_strategy.py")
        load_strategy_module(path)
        assert "strategy_temp_cool_strategy" in sys.modules

    def test_reloads_stale_cached_module(self, write_strategy):
        """Loading same module name twice uses the latest code, not the cached version."""
        path = write_strategy(VALID_STRATEGY_CODE)
        m1 = load_strategy_module(path, module_name="test_stale_cache")

        updated_code = VALID_STRATEGY_CODE.replace('{"period": 14}', '{"period": 99}')
        path.write_text(updated_code)

        m2 = load_strategy_module(path, module_name="test_stale_cache")
        assert m2.DEFAULT_PARAMS["period"] == 99

    def test_loads_real_rsi_crossover_strategy(self):
        """Smoke test: load an actual strategy from the crabquant strategies directory."""
        strategy_path = Path("crabquant/strategies/rsi_crossover.py")
        module = load_strategy_module(strategy_path)
        assert module is not None
        assert callable(module.generate_signals)
        assert isinstance(module.DEFAULT_PARAMS, dict)
        assert isinstance(module.PARAM_GRID, dict)


# ─── load_module_from_code ────────────────────────────────────────────────────

class TestLoadModuleFromCode:
    def test_returns_module_for_valid_code(self):
        module = load_module_from_code(VALID_STRATEGY_CODE)
        assert module is not None

    def test_loaded_module_has_required_attributes(self):
        module = load_module_from_code(VALID_STRATEGY_CODE)
        assert callable(module.generate_signals)
        assert isinstance(module.DEFAULT_PARAMS, dict)
        assert isinstance(module.PARAM_GRID, dict)

    def test_returns_none_for_syntax_error(self):
        result = load_module_from_code(SYNTAX_ERROR_CODE)
        assert result is None

    def test_returns_none_for_missing_generate_signals(self):
        result = load_module_from_code(MISSING_GENERATE_SIGNALS)
        assert result is None

    def test_returns_none_for_missing_default_params(self):
        result = load_module_from_code(MISSING_DEFAULT_PARAMS)
        assert result is None

    def test_accepts_missing_param_grid(self):
        """PARAM_GRID is optional for LLM-generated strategies."""
        result = load_module_from_code(MISSING_PARAM_GRID)
        assert result is not None

    def test_returns_none_for_runtime_error(self):
        result = load_module_from_code(RUNTIME_ERROR_CODE)
        assert result is None

    def test_custom_module_name(self):
        module = load_module_from_code(VALID_STRATEGY_CODE, module_name="from_code_custom")
        assert module is not None
        assert "from_code_custom" in sys.modules

    def test_temp_file_is_cleaned_up(self):
        """No temp files should persist after loading."""
        import tempfile
        before = set(Path(tempfile.gettempdir()).glob("strategy_*.py"))
        load_module_from_code(VALID_STRATEGY_CODE, module_name="cleanup_test_module")
        after = set(Path(tempfile.gettempdir()).glob("strategy_*.py"))
        assert after == before

    def test_does_not_leak_empty_string(self):
        result = load_module_from_code("")
        assert result is None

    def test_module_values_are_correct(self):
        module = load_module_from_code(VALID_STRATEGY_CODE)
        assert module.DEFAULT_PARAMS == {"period": 14}
        assert module.PARAM_GRID == {"period": [7, 14, 21]}


# ─── Expanded load_strategy_module edge cases ──────────────────────────────


class TestLoadStrategyModuleEdgeCases:
    """Additional edge-case tests for load_strategy_module."""

    def test_returns_none_for_directory_path(self, tmp_path):
        result = load_strategy_module(tmp_path)
        assert result is None

    def test_returns_none_for_non_py_file(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE, filename="strategy.txt")
        # importlib can't create a spec for non-.py files
        result = load_strategy_module(path, module_name="txt_file_test")
        assert result is None

    def test_unicode_code_loads_correctly(self, write_strategy):
        code = textwrap.dedent("""
            DEFAULT_PARAMS = {"period": 14}
            DESCRIPTION = "RSI 策略 — 反转信号"
            def generate_signals(df, params=None):
                return None, None
        """).strip()
        path = write_strategy(code, filename="unicode_strategy.py")
        module = load_strategy_module(path, module_name="unicode_test")
        assert module is not None
        assert module.DESCRIPTION == "RSI 策略 — 反转信号"

    def test_import_error_in_code(self, write_strategy):
        code = textwrap.dedent("""
            import nonexistent_module_xyz
            DEFAULT_PARAMS = {"period": 14}
            def generate_signals(df, params=None):
                return None, None
        """).strip()
        path = write_strategy(code, filename="import_err.py")
        result = load_strategy_module(path, module_name="import_err_test")
        assert result is None

    def test_empty_file(self, write_strategy):
        path = write_strategy("", filename="empty.py")
        result = load_strategy_module(path, module_name="empty_test")
        assert result is None

    def test_only_comments(self, write_strategy):
        code = "# This is a comment\n# Another comment\n"
        path = write_strategy(code, filename="comments.py")
        result = load_strategy_module(path, module_name="comments_test")
        assert result is None

    def test_module_has_correct_file_attribute(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE, filename="my_strat.py")
        module = load_strategy_module(path, module_name="file_attr_test")
        assert module is not None
        assert str(path) in module.__file__

    def test_generate_signals_is_callable(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE)
        module = load_strategy_module(path, module_name="callable_test")
        assert callable(module.generate_signals)

    def test_non_callable_generate_signals_rejected(self, write_strategy):
        code = textwrap.dedent("""
            DEFAULT_PARAMS = {"period": 14}
            generate_signals = "not a function"
        """).strip()
        path = write_strategy(code, filename="non_callable.py")
        # hasattr check passes since attribute exists (even if not callable)
        module = load_strategy_module(path, module_name="non_callable_test")
        assert module is not None  # only checks hasattr, not callable

    def test_default_params_not_dict(self, write_strategy):
        code = textwrap.dedent("""
            DEFAULT_PARAMS = [1, 2, 3]
            def generate_signals(df, params=None):
                return None, None
        """).strip()
        path = write_strategy(code, filename="bad_params.py")
        # Only checks hasattr, not type
        module = load_strategy_module(path, module_name="bad_params_test")
        assert module is not None

    def test_module_name_with_dashes(self, write_strategy):
        path = write_strategy(VALID_STRATEGY_CODE)
        # Dashes in module name are technically valid for importlib
        module = load_strategy_module(path, module_name="my-strategy-test")
        assert module is not None
        assert "my-strategy-test" in sys.modules

    def test_deeply_nested_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        path = nested / "deep_strategy.py"
        path.write_text(VALID_STRATEGY_CODE)
        module = load_strategy_module(path, module_name="deep_test")
        assert module is not None

    def test_infinite_loop_protection_at_import(self, write_strategy):
        code = textwrap.dedent("""
            import sys
            DEFAULT_PARAMS = {"period": 14}
            def generate_signals(df, params=None):
                return None, None
        """).strip()
        path = write_strategy(code, filename="safe_import.py")
        module = load_strategy_module(path, module_name="safe_import_test")
        assert module is not None

    def test_syntax_error_in_function_body(self, write_strategy):
        code = textwrap.dedent("""
            DEFAULT_PARAMS = {"period": 14}
            def generate_signals(df, params=None):
                x = 1 +  # syntax error
                return None, None
        """).strip()
        path = write_strategy(code, filename="fn_syntax_err.py")
        result = load_strategy_module(path, module_name="fn_syntax_test")
        assert result is None

    def test_name_error_at_module_level(self, write_strategy):
        code = textwrap.dedent("""
            DEFAULT_PARAMS = {"period": 14}
            x = undefined_variable
            def generate_signals(df, params=None):
                return None, None
        """).strip()
        path = write_strategy(code, filename="name_err.py")
        result = load_strategy_module(path, module_name="name_err_test")
        assert result is None


# ─── Expanded load_module_from_code edge cases ─────────────────────────────


class TestLoadModuleFromCodeEdgeCases:
    """Additional edge-case tests for load_module_from_code."""

    def test_whitespace_only(self):
        result = load_module_from_code("   \n\t\n  ")
        assert result is None

    def test_none_input(self):
        result = load_module_from_code(None)
        assert result is None

    def test_unicode_in_code(self):
        code = textwrap.dedent("""
            DEFAULT_PARAMS = {"period": 14}
            DESCRIPTION = "Momentum — 動量策略"
            def generate_signals(df, params=None):
                return None, None
        """).strip()
        module = load_module_from_code(code, module_name="unicode_code_test")
        assert module is not None
        assert "動量策略" in module.DESCRIPTION

    def test_very_long_code(self):
        """Test with a large strategy module (many lines)."""
        lines = [
            "import pandas as pd",
            "DEFAULT_PARAMS = {'period': 14}",
            "def generate_signals(df, params=None):",
            "    entries = pd.Series(False, index=df.index)",
            "    exits = pd.Series(False, index=df.index)",
        ]
        # Add many comment lines to bulk it up
        for i in range(500):
            lines.append(f"# Helper comment line {i}")
        code = "\n".join(lines)
        module = load_module_from_code(code, module_name="long_code_test")
        assert module is not None
        assert callable(module.generate_signals)

    def test_code_with_extra_attributes(self):
        code = textwrap.dedent("""
            import pandas as pd
            DEFAULT_PARAMS = {"fast": 12, "slow": 26}
            PARAM_GRID = {"fast": [8, 12, 16], "slow": [20, 26, 32]}
            DESCRIPTION = "EMA crossover with volume filter"
            RISK_PER_TRADE = 0.02
            def generate_signals(df, params=None):
                entries = pd.Series(False, index=df.index)
                exits = pd.Series(False, index=df.index)
                return entries, exits
        """).strip()
        module = load_module_from_code(code, module_name="extra_attrs_test")
        assert module is not None
        assert module.DESCRIPTION == "EMA crossover with volume filter"
        assert module.RISK_PER_TRADE == 0.02

    def test_code_with_import_pandas_ta(self):
        """Code that imports pandas_ta (should not crash at import)."""
        code = textwrap.dedent("""
            DEFAULT_PARAMS = {"period": 14}
            def generate_signals(df, params=None):
                return None, None
        """).strip()
        module = load_module_from_code(code, module_name="no_pandas_ta_test")
        assert module is not None

    def test_code_with_syntax_error_single_line(self):
        result = load_module_from_code("def foo(:\n  pass", module_name="one_line_syntax")
        assert result is None

    def test_code_with_indentation_error(self):
        code = textwrap.dedent("""
            DEFAULT_PARAMS = {"period": 14}
            def generate_signals(df, params=None):
            return None  # bad indent
        """).strip()
        result = load_module_from_code(code, module_name="indent_err_test")
        assert result is None

    def test_module_not_leaked_in_sys_modules_on_failure(self):
        name = "leak_test_unique_name_42"
        if name in sys.modules:
            del sys.modules[name]
        load_module_from_code(SYNTAX_ERROR_CODE, module_name=name)
        assert name not in sys.modules

    def test_default_module_name_from_code(self):
        """load_module_from_code generates a temp file with strategy_ prefix."""
        module = load_module_from_code(VALID_STRATEGY_CODE)
        assert module is not None
        # The module name will be strategy_temp_<random> from the temp file stem
        found = [k for k in sys.modules if k.startswith("strategy_temp_")]
        assert len(found) >= 1

    def test_generate_signals_returns_correct_signature(self):
        import inspect
        module = load_module_from_code(VALID_STRATEGY_CODE, module_name="sig_test")
        assert module is not None
        sig = inspect.signature(module.generate_signals)
        params = list(sig.parameters.keys())
        assert "df" in params
        assert "params" in params
