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

    def test_returns_none_for_missing_param_grid(self, write_strategy):
        path = write_strategy(MISSING_PARAM_GRID)
        result = load_strategy_module(path)
        assert result is None

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

    def test_returns_none_for_missing_param_grid(self):
        result = load_module_from_code(MISSING_PARAM_GRID)
        assert result is None

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
