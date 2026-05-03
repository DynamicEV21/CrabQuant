"""
AST Safety Sanitizer for LLM-generated strategy code.

Walks the AST tree of strategy code to detect dangerous patterns:
- Forbidden module imports (os, sys, subprocess, socket, etc.)
- Dangerous builtin calls (exec, eval, compile, __import__, open)
- Attribute access to dangerous functions (os.system, os.popen, etc.)
- Look-ahead bias patterns in pandas operations
- Dunder attribute access that could bypass restrictions

Returns (is_safe: bool, violations: list[str]) with line numbers.
"""

from __future__ import annotations

import ast
import re


# ── Configurable blocklists ────────────────────────────────────────────────

BLOCKED_IMPORTS: dict[str, str] = {
    # module_prefix: description
    "os": "system-level access (os.system, os.popen, etc.)",
    "sys": "interpreter access (sys.exit, sys.path manipulation)",
    "subprocess": "process execution (subprocess.call, Popen, etc.)",
    "socket": "network socket access",
    "http": "HTTP client/server (http.server, http.client)",
    "urllib": "URL fetching (urllib.request, etc.)",
    "requests": "HTTP library for network requests",
    "shutil": "file operations (shutil.rmtree, copy, etc.)",
    "pathlib": "file system path operations",
    "ctypes": "foreign function interface (C library access)",
    "pickle": "arbitrary code deserialization",
    "marshal": "code object serialization (can execute arbitrary code)",
    "importlib": "dynamic module import (can load arbitrary modules)",
}

BLOCKED_BUILTINS: dict[str, str] = {
    # builtin_name: description
    "exec": "arbitrary code execution",
    "eval": "arbitrary expression evaluation",
    "compile": "code compilation (can be used with exec/eval)",
    "__import__": "dynamic module import (bypasses import restrictions)",
    "open": "file I/O access",
    "globals": "access to global namespace",
    "locals": "access to local namespace",
    "getattr": "dynamic attribute access (can bypass restrictions)",
    "setattr": "dynamic attribute modification",
    "delattr": "dynamic attribute deletion",
    "vars": "access to object __dict__",
}

# Dangerous attribute chains: (object, attribute) pairs
BLOCKED_ATTR_ACCESS: dict[str, set[str]] = {
    "os": {"system", "popen", "exec", "spawn", "fork", "kill", "remove", "unlink",
           "rename", "mkdir", "rmdir", "listdir", "walk", "environ"},
    "subprocess": {"call", "run", "Popen", "check_output", "check_call",
                   "getoutput", "getstatusoutput"},
    "sys": {"exit", "path"},
}

# Look-ahead bias patterns (regex-based, checked on raw source)
LOOKAHEAD_PATTERNS: list[dict[str, str]] = [
    {"pattern": r"\.shift\s*\(\s*-\s*\d+\s*\)", "desc": "pandas shift with negative N (look-ahead)"},
    {"pattern": r"\[:-\d+\]", "desc": "negative slice index (possible look-ahead)"},
]


def sanitize_strategy_code(code: str) -> tuple[bool, list[str]]:
    """
    Analyze strategy code for dangerous patterns via AST walk.

    Args:
        code: Python source code to sanitize.

    Returns:
        (is_safe, violations) where is_safe is True when no violations found,
        and violations is a list of human-readable strings with line numbers.
    """
    violations: list[str] = []

    if not code or not code.strip():
        return (False, ["Empty code"])

    # ── Phase 1: AST parse ──────────────────────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (False, [f"SyntaxError at line {e.lineno}: {e.msg}"])

    # ── Phase 2: AST walk for import violations ─────────────────────────
    for node in ast.walk(tree):
        # import os, sys, subprocess
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module in BLOCKED_IMPORTS:
                    violations.append(
                        f"Line {node.lineno}: Blocked import '{alias.name}' — "
                        f"{BLOCKED_IMPORTS[top_module]}"
                    )

        # from os import system
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top_module = module.split(".")[0]
            if top_module in BLOCKED_IMPORTS:
                violations.append(
                    f"Line {node.lineno}: Blocked import from '{module}' — "
                    f"{BLOCKED_IMPORTS[top_module]}"
                )
            # Also check individual names being imported from allowed modules
            for alias in node.names:
                if alias.name in BLOCKED_BUILTINS:
                    violations.append(
                        f"Line {node.lineno}: Blocked builtin '{alias.name}' "
                        f"imported from '{module}' — {BLOCKED_BUILTINS[alias.name]}"
                    )

    # ── Phase 3: AST walk for dangerous calls ───────────────────────────
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func

        # Direct builtin calls: exec(code), eval(expr), open("file")
        if isinstance(func, ast.Name) and func.id in BLOCKED_BUILTINS:
            violations.append(
                f"Line {node.lineno}: Blocked builtin call '{func.id}()' — "
                f"{BLOCKED_BUILTINS[func.id]}"
            )

        # __import__('os') — Name call to __import__
        # (already covered by BLOCKED_BUILTINS, but be explicit)

        # Attribute calls: os.system("rm -rf /"), subprocess.Popen(...)
        if isinstance(func, ast.Attribute):
            # Get the full attribute chain: os.system, subprocess.Popen
            chain = _get_attr_chain(func)
            if chain:
                obj_part = chain[0]
                attr_part = chain[-1]
                if obj_part in BLOCKED_ATTR_ACCESS:
                    blocked_attrs = BLOCKED_ATTR_ACCESS[obj_part]
                    if attr_part in blocked_attrs:
                        violations.append(
                            f"Line {node.lineno}: Blocked attribute call "
                            f"'{'.'.join(chain)}()' — dangerous {obj_part} method"
                        )

    # ── Phase 4: AST walk for dangerous dunder attribute access ─────────
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attr_name = node.attr
            # Block access to dangerous dunder attributes
            if attr_name.startswith("__") and attr_name.endswith("__"):
                # Allow common safe dunders
                safe_dunders = {
                    "__init__", "__name__", "__file__", "__doc__",
                    "__version__", "__all__", "__class__", "__dict__",
                    "__eq__", "__lt__", "__le__", "__gt__", "__ge__",
                    "__add__", "__sub__", "__mul__", "__truediv__",
                    "__neg__", "__pos__", "__abs__", "__len__",
                    "__getitem__", "__setitem__", "__contains__",
                    "__iter__", "__next__", "__repr__", "__str__",
                    "__float__", "__int__", "__bool__", "__hash__",
                    "__enter__", "__exit__", "__call__",
                    "__index__", "__round__", "__floor__", "__ceil__",
                }
                if attr_name not in safe_dunders:
                    violations.append(
                        f"Line {node.lineno}: Suspicious dunder access "
                        f"'{attr_name}' — may bypass security restrictions"
                    )

    # ── Phase 5: Regex-based look-ahead bias detection ──────────────────
    source_lines = code.splitlines()
    for rule in LOOKAHEAD_PATTERNS:
        pattern = re.compile(rule["pattern"])
        for i, line in enumerate(source_lines, start=1):
            if pattern.search(line):
                violations.append(
                    f"Line {i}: Possible look-ahead bias — {rule['desc']}"
                )

    return (len(violations) == 0, violations)


def _get_attr_chain(node: ast.Attribute) -> list[str]:
    """
    Recursively extract the full attribute chain from an ast.Attribute node.

    Examples:
        os.system → ["os", "system"]
        subprocess.Popen → ["subprocess", "Popen"]
        foo.bar.baz → ["foo", "bar", "baz"]
    """
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    return parts
