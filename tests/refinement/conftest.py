"""Conftest for tests/refinement — mock heavy strategy dependencies."""
import sys
import types
import importlib

# The crabquant.refinement.__init__ imports context_builder which imports
# the strategies package, which requires pandas_ta, vectorbt, etc.
# For unit tests that only need the stagnation / mandate modules,
# we mock these heavy C-extension dependencies so tests can run on
# any Python version without installing the full runtime stack.
# However, if the real package is importable, use it instead of a mock.
for _mod in ("pandas_ta", "vectorbt", "yfinance", "backtesting"):
    if _mod not in sys.modules:
        try:
            importlib.import_module(_mod)
        except ImportError:
            sys.modules[_mod] = types.ModuleType(_mod)
