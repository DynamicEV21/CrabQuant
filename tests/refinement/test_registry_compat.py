"""Tests for _registry_compat.py — abstract over tuple and dict registry entries."""

from unittest.mock import MagicMock

import pytest


def _make_dict_entry(**overrides):
    """Create a dict-style registry entry."""
    entry = {
        "fn": MagicMock(__name__="test_fn"),
        "defaults": {"fast": 12, "slow": 26},
        "grid": {"fast": [8, 12, 16]},
        "description": "Test strategy",
        "matrix_fn": MagicMock(),
        "preferred_regimes": ["trending_up"],
        "acceptable_regimes": [],
        "weak_regimes": ["high_volatility"],
        "regime_sharpes": {"trending_up": 1.5},
        "is_regime_specific": True,
    }
    entry.update(overrides)
    return entry


def _make_tuple_entry():
    """Create a legacy tuple-style registry entry."""
    return (
        MagicMock(__name__="test_fn"),
        {"fast": 12, "slow": 26},
        {"fast": [8, 12, 16]},
        "Test strategy",
        MagicMock(),
    )


class TestRegistryCompat:
    """Test that compat helpers work with both dict and tuple entries."""

    def test_get_fn_dict(self):
        from crabquant.strategies._registry_compat import get_fn
        entry = _make_dict_entry()
        assert get_fn(entry).__name__ == "test_fn"

    def test_get_fn_tuple(self):
        from crabquant.strategies._registry_compat import get_fn
        entry = _make_tuple_entry()
        assert get_fn(entry).__name__ == "test_fn"

    def test_get_defaults_dict(self):
        from crabquant.strategies._registry_compat import get_defaults
        entry = _make_dict_entry()
        assert get_defaults(entry) == {"fast": 12, "slow": 26}

    def test_get_defaults_tuple(self):
        from crabquant.strategies._registry_compat import get_defaults
        entry = _make_tuple_entry()
        assert get_defaults(entry) == {"fast": 12, "slow": 26}

    def test_get_description_dict(self):
        from crabquant.strategies._registry_compat import get_description
        entry = _make_dict_entry()
        assert get_description(entry) == "Test strategy"

    def test_get_description_tuple(self):
        from crabquant.strategies._registry_compat import get_description
        entry = _make_tuple_entry()
        assert get_description(entry) == "Test strategy"

    def test_get_matrix_fn_dict(self):
        from crabquant.strategies._registry_compat import get_matrix_fn
        entry = _make_dict_entry()
        assert get_matrix_fn(entry) is not None

    def test_get_matrix_fn_tuple(self):
        from crabquant.strategies._registry_compat import get_matrix_fn
        entry = _make_tuple_entry()
        assert get_matrix_fn(entry) is not None

    def test_get_matrix_fn_none_dict(self):
        from crabquant.strategies._registry_compat import get_matrix_fn
        entry = _make_dict_entry(matrix_fn=None)
        assert get_matrix_fn(entry) is None

    def test_get_regime_tags_dict(self):
        from crabquant.strategies._registry_compat import get_regime_tags
        entry = _make_dict_entry()
        tags = get_regime_tags(entry)
        assert tags["preferred_regimes"] == ["trending_up"]
        assert tags["is_regime_specific"] is True
        assert tags["regime_sharpes"] == {"trending_up": 1.5}

    def test_get_regime_tags_tuple(self):
        from crabquant.strategies._registry_compat import get_regime_tags
        entry = _make_tuple_entry()
        tags = get_regime_tags(entry)
        assert tags["preferred_regimes"] == []
        assert tags["is_regime_specific"] is False

    def test_to_tuple_from_dict(self):
        from crabquant.strategies._registry_compat import to_tuple
        entry = _make_dict_entry()
        t = to_tuple(entry)
        assert isinstance(t, tuple)
        assert len(t) == 5
        assert t[3] == "Test strategy"

    def test_to_tuple_from_tuple(self):
        from crabquant.strategies._registry_compat import to_tuple
        entry = _make_tuple_entry()
        t = to_tuple(entry)
        assert t == entry

    def test_get_defaults_empty_tuple(self):
        """Handles short tuple gracefully."""
        from crabquant.strategies._registry_compat import get_defaults, get_description
        entry = (MagicMock(),)  # only fn
        assert get_defaults(entry) == {}
        assert get_description(entry) == ""
