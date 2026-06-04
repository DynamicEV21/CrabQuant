"""
Registry access helpers — abstract over tuple (legacy) and dict (new) entries.

The STRATEGY_REGISTRY historically stored entries as 5-tuples:
    (fn, defaults, grid, description, matrix_fn)

New entries from the promotion pipeline store dicts:
    {"fn": ..., "defaults": ..., "grid": ..., "description": ..., "matrix_fn": ...,
     "preferred_regimes": [...], "weak_regimes": [...], "regime_sharpes": {...}}

These helpers let all existing code work transparently with either format.
"""

from typing import Any, Callable


def get_fn(entry: Any) -> Callable:
    """Get the generate_signals function from a registry entry."""
    if isinstance(entry, dict):
        return entry["fn"]
    return entry[0]


def get_defaults(entry: Any) -> dict:
    """Get DEFAULT_PARAMS from a registry entry."""
    if isinstance(entry, dict):
        return entry["defaults"]
    return entry[1] if len(entry) > 1 else {}


def get_grid(entry: Any) -> dict:
    """Get PARAM_GRID from a registry entry."""
    if isinstance(entry, dict):
        return entry["grid"]
    return entry[2] if len(entry) > 2 else {}


def get_description(entry: Any) -> str:
    """Get DESCRIPTION from a registry entry."""
    if isinstance(entry, dict):
        return entry.get("description", "")
    return entry[3] if len(entry) > 3 else ""


def get_matrix_fn(entry: Any) -> Callable | None:
    """Get generate_signals_matrix from a registry entry."""
    if isinstance(entry, dict):
        return entry.get("matrix_fn")
    return entry[4] if len(entry) > 4 else None


def get_regime_tags(entry: Any) -> dict:
    """Get regime tags from a registry entry.

    Returns dict with preferred_regimes, acceptable_regimes, weak_regimes,
    regime_sharpes, is_regime_specific.  Empty for legacy tuple entries.
    """
    if isinstance(entry, dict):
        return {
            "preferred_regimes": entry.get("preferred_regimes", []),
            "acceptable_regimes": entry.get("acceptable_regimes", []),
            "weak_regimes": entry.get("weak_regimes", []),
            "regime_sharpes": entry.get("regime_sharpes", {}),
            "is_regime_specific": entry.get("is_regime_specific", False),
        }
    return {
        "preferred_regimes": [],
        "acceptable_regimes": [],
        "weak_regimes": [],
        "regime_sharpes": {},
        "is_regime_specific": False,
    }


def to_tuple(entry: Any) -> tuple:
    """Convert a registry entry to the legacy 5-tuple format.

    Useful for backward compat in places that unpack:
        for name, (fn, defaults, grid, desc, matrix_fn) in registry.items()
    """
    if isinstance(entry, dict):
        return (
            entry["fn"],
            entry["defaults"],
            entry["grid"],
            entry.get("description", ""),
            entry.get("matrix_fn"),
        )
    return tuple(entry)
