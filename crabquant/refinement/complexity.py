"""Complexity scoring for strategy code.

Uses Python's ast module to measure code complexity dimensions and flag
strategies that are likely to overfit. Lightweight and deterministic —
no external dependencies, runs on every strategy evaluation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, asdict


@dataclass
class ComplexityScore:
    """Breakdown of strategy complexity dimensions."""

    n_params: int = 0          # number of tunable parameters
    n_nodes: int = 0           # total AST nodes (code length proxy)
    n_functions: int = 0       # function/async-function definitions
    n_branches: int = 0        # if/elif/else branches
    max_nesting: int = 0       # maximum nesting depth of loops/if
    n_features: int = 0        # distinct data columns referenced
    total: float = 0.0         # weighted complexity score 0–100


# ── AST helpers ───────────────────────────────────────────────────────────────

def _count_nodes(tree: ast.AST) -> int:
    """Count total AST nodes in the tree."""
    return sum(1 for _ in ast.walk(tree))


def _count_functions(tree: ast.AST) -> int:
    """Count function and async function definitions."""
    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _count_branches(tree: ast.AST) -> int:
    """Count if/elif/else branch points."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            count += 1                     # each `if` counts as one branch point
            count += len(node.orelse)       # each elif/else adds a branch
    return count


def _max_nesting(tree: ast.AST) -> int:
    """Compute maximum nesting depth of for/while/if blocks."""
    max_depth = 0

    def _walk(node: ast.AST, depth: int) -> None:
        nonlocal max_depth
        if isinstance(node, (ast.For, ast.While, ast.If)):
            depth += 1
            max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            _walk(child, depth)

    _walk(tree, 0)
    return max_depth


def _count_features(tree: ast.AST) -> int:
    """Count distinct data column names referenced (close, high, low, volume, etc.)."""
    feature_names: set[str] = set()
    common_features = {
        "open", "high", "low", "close", "volume",
        "vwap", "returns", "log_returns",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            # Matches patterns like df["close"], data["high"]
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                if node.slice.value.lower() in common_features:
                    feature_names.add(node.slice.value.lower())
        # Also check df.column attribute access
        if isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.lower() in common_features:
                feature_names.add(node.attr.lower())
    return len(feature_names)


# ── Public API ────────────────────────────────────────────────────────────────

def complexity_score(code: str, params: dict | None = None) -> dict:
    """Compute a complexity score for strategy code.

    Parameters
    ----------
    code : str
        The strategy source code (Python).
    params : dict | None
        Tunable parameters dict. If ``None`` or empty, n_params = 0.

    Returns
    -------
    dict
        ``{
            "complexity": float (0–100),
            "flags": list[str],
            "breakdown": { ... }  # per-dimension counts
        }``
    """
    flags: list[str] = []
    breakdown = ComplexityScore()

    # ── Parameter count ──
    if params:
        breakdown.n_params = len(params)

    # ── Parse AST ──
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Can't parse — assign maximum penalty
        breakdown.total = 100.0
        flags.append("invalid_python")
        return _build_result(breakdown, flags)

    if not code.strip():
        return _build_result(breakdown, flags)

    # ── Compute dimensions ──
    breakdown.n_nodes = _count_nodes(tree)
    breakdown.n_functions = _count_functions(tree)
    breakdown.n_branches = _count_branches(tree)
    breakdown.max_nesting = _max_nesting(tree)
    breakdown.n_features = _count_features(tree)

    # ── Weighted score (0–100) ──
    # Each dimension contributes a 0–25 sub-score, total 0–100.
    # We use a logarithmic scale for n_nodes so real strategies don't
    # instantly max out.
    import math

    node_score = min(25.0, math.log1p(breakdown.n_nodes) * 2.0)
    func_score = min(25.0, breakdown.n_functions * 4.0)
    branch_score = min(25.0, breakdown.n_branches * 2.5)
    param_score = min(25.0, breakdown.n_params * 2.5)

    breakdown.total = round(min(100.0, node_score + func_score + branch_score + param_score), 1)

    # ── Flags ──
    if breakdown.total > 70:
        flags.append("high_complexity")
    if breakdown.n_params > 8:
        flags.append("too_many_params")
    if breakdown.max_nesting > 3:
        flags.append("deep_nesting")
    if breakdown.n_branches > 10:
        flags.append("too_many_branches")
    if breakdown.n_functions > 5:
        flags.append("too_many_functions")

    return _build_result(breakdown, flags)


def _build_result(score: ComplexityScore, flags: list[str]) -> dict:
    """Build the return dict from a ComplexityScore and flags list."""
    return {
        "complexity": score.total,
        "flags": flags,
        "breakdown": asdict(score),
    }


def complexity_penalty(score: float, base_threshold: float = 1.5) -> float:
    """Compute an adjusted promotion threshold based on complexity.

    Higher complexity → harder to get promoted.

    Parameters
    ----------
    score : float
        Complexity score (0–100) from ``complexity_score()``.
    base_threshold : float
        Base Sharpe threshold for promotion.

    Returns
    -------
    float
        Adjusted threshold.
    """
    # Map complexity 0–100 to a penalty factor of 0.0–0.5
    # so at complexity=100 the threshold is 1.5× the base.
    penalty = (score / 100.0) * 0.5
    return base_threshold * (1.0 + penalty)
