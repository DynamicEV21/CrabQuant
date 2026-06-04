"""Safe strategy code loading via temp file + importlib."""

import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType

_REQUIRED_ATTRS = ("generate_signals", "DEFAULT_PARAMS")


def _build_load_error_info(exc: Exception) -> dict:
    """Build a structured error info dict from a module loading exception.

    Args:
        exc: The caught exception.

    Returns:
        Dict with error_type and error_message.
    """
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    last_lines = tb_lines[-10:] if len(tb_lines) > 10 else tb_lines
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_traceback": "".join(last_lines).strip(),
    }


def load_strategy_module(
    strategy_path: "Path | str",
    module_name: str | None = None,
) -> "ModuleType | None | tuple":
    """Load a strategy .py file as a Python module.

    Uses importlib.util.spec_from_file_location so crabquant imports resolve
    correctly (exec() on a raw string does not set __file__, breaking relative
    imports and indicator caches).

    Returns (module, None) on success, (None, error_info_dict) on failure.
    error_info contains error_type, error_message, and error_traceback.
    """
    strategy_path = Path(strategy_path)
    if not strategy_path.exists():
        return None, {"error_type": "FileNotFoundError", "error_message": f"File not found: {strategy_path}", "error_traceback": ""}

    if module_name is None:
        module_name = f"strategy_temp_{strategy_path.stem}"

    try:
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, str(strategy_path))
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(strategy_path)
        sys.modules[module_name] = module

        # Compile from source directly — bypasses .pyc cache so re-loading the
        # same path after an in-place write always picks up the latest code.
        source = strategy_path.read_text(encoding="utf-8")
        code_obj = compile(source, str(strategy_path), "exec")
        exec(code_obj, module.__dict__)

        for attr in _REQUIRED_ATTRS:
            if not hasattr(module, attr):
                error_info = {
                    "error_type": "AttributeError",
                    "error_message": f"Missing required attribute: {attr}",
                    "error_traceback": "",
                }
                print(f"  Module load error: Missing {attr}")
                del sys.modules[module_name]
                return None, error_info

        return module, None
    except Exception as e:
        print(f"  Module load error: {e}")
        if module_name in sys.modules:
            del sys.modules[module_name]
        return None, _build_load_error_info(e)


def load_module_from_code(
    code: str,
    module_name: str | None = None,
) -> "ModuleType | None":
    """Load LLM-generated strategy code (Python string) as a module.

    Writes the code to a temporary file, loads it via importlib, then deletes
    the temp file.  This is the safe approach — exec() on a raw string breaks
    crabquant's indicator cache and relative imports.
    """
    if not code or not code.strip():
        return None

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="strategy_",
            delete=False,
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)

        result = load_strategy_module(tmp_path, module_name=module_name)
        # Normalize: extract module from tuple if needed
        if isinstance(result, tuple):
            return result[0]  # module or None
        return result
    except Exception as e:
        print(f"  load_module_from_code error: {e}")
        return None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
