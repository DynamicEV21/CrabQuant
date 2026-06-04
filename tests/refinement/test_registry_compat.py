"""Tests for _registry_compat.py — abstract over tuple and dict registry entries."""

from unittest.mock import MagicMock, sentinel

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

    # ── New tests ──────────────────────────────────────────────────────

    def test_get_fn_same_object_dict(self):
        """get_fn should return the exact fn object from dict entry."""
        from crabquant.strategies._registry_compat import get_fn
        fn = MagicMock(__name__="my_fn")
        entry = _make_dict_entry(fn=fn)
        assert get_fn(entry) is fn

    def test_get_fn_same_object_tuple(self):
        """get_fn should return the exact fn object from tuple entry."""
        from crabquant.strategies._registry_compat import get_fn
        fn = MagicMock(__name__="my_fn")
        entry = (fn, {}, {}, "desc", None)
        assert get_fn(entry) is fn

    def test_get_grid_dict(self):
        from crabquant.strategies._registry_compat import get_grid
        entry = _make_dict_entry()
        assert get_grid(entry) == {"fast": [8, 12, 16]}

    def test_get_grid_tuple(self):
        from crabquant.strategies._registry_compat import get_grid
        entry = _make_tuple_entry()
        assert get_grid(entry) == {"fast": [8, 12, 16]}

    def test_get_grid_empty_dict(self):
        """Dict entry with empty grid."""
        from crabquant.strategies._registry_compat import get_grid
        entry = _make_dict_entry(grid={})
        assert get_grid(entry) == {}

    def test_get_grid_short_tuple(self):
        """Tuple shorter than 3 elements should return empty grid."""
        from crabquant.strategies._registry_compat import get_grid
        entry = (MagicMock(), {"a": 1})  # only 2 elements
        assert get_grid(entry) == {}

    def test_get_grid_single_element_tuple(self):
        """Tuple with only fn should return empty grid."""
        from crabquant.strategies._registry_compat import get_grid
        entry = (MagicMock(),)
        assert get_grid(entry) == {}

    def test_get_description_missing_key_dict(self):
        """Dict entry without description key should return empty string."""
        from crabquant.strategies._registry_compat import get_description
        entry = {"fn": MagicMock(), "defaults": {}, "grid": {}}
        assert get_description(entry) == ""

    def test_get_description_empty_string_dict(self):
        """Dict entry with empty description."""
        from crabquant.strategies._registry_compat import get_description
        entry = _make_dict_entry(description="")
        assert get_description(entry) == ""

    def test_get_description_short_tuple(self):
        """Tuple shorter than 4 elements should return empty string."""
        from crabquant.strategies._registry_compat import get_description
        entry = (MagicMock(), {}, {})
        assert get_description(entry) == ""

    def test_get_matrix_fn_missing_key_dict(self):
        """Dict entry without matrix_fn key should return None."""
        from crabquant.strategies._registry_compat import get_matrix_fn
        entry = {"fn": MagicMock(), "defaults": {}, "grid": {}}
        assert get_matrix_fn(entry) is None

    def test_get_matrix_fn_short_tuple(self):
        """Tuple shorter than 5 elements should return None."""
        from crabquant.strategies._registry_compat import get_matrix_fn
        entry = (MagicMock(), {}, {}, "desc")
        assert get_matrix_fn(entry) is None

    def test_get_matrix_fn_none_in_tuple(self):
        """Tuple with explicit None at index 4 should return None."""
        from crabquant.strategies._registry_compat import get_matrix_fn
        fn = MagicMock()
        entry = (fn, {}, {}, "desc", None)
        assert get_matrix_fn(entry) is None

    def test_get_regime_tags_empty_dict(self):
        """Dict entry without regime keys should return empty defaults."""
        from crabquant.strategies._registry_compat import get_regime_tags
        entry = {"fn": MagicMock(), "defaults": {}, "grid": {}}
        tags = get_regime_tags(entry)
        assert tags["preferred_regimes"] == []
        assert tags["acceptable_regimes"] == []
        assert tags["weak_regimes"] == []
        assert tags["regime_sharpes"] == {}
        assert tags["is_regime_specific"] is False

    def test_get_regime_tags_partial_dict(self):
        """Dict entry with some regime keys should return partial + defaults."""
        from crabquant.strategies._registry_compat import get_regime_tags
        entry = _make_dict_entry(
            preferred_regimes=["mean_reversion"],
            weak_regimes=[],
            # acceptable_regimes, regime_sharpes, is_regime_specific not set
        )
        # Remove keys not set
        entry.pop("acceptable_regimes", None)
        entry.pop("regime_sharpes", None)
        entry.pop("is_regime_specific", None)
        tags = get_regime_tags(entry)
        assert tags["preferred_regimes"] == ["mean_reversion"]
        assert tags["acceptable_regimes"] == []
        assert tags["weak_regimes"] == []
        assert tags["regime_sharpes"] == {}
        assert tags["is_regime_specific"] is False

    def test_get_regime_tags_tuple_structure(self):
        """Tuple entry should return all regime tag keys with defaults."""
        from crabquant.strategies._registry_compat import get_regime_tags
        entry = _make_tuple_entry()
        tags = get_regime_tags(entry)
        expected_keys = {"preferred_regimes", "acceptable_regimes", "weak_regimes", "regime_sharpes", "is_regime_specific"}
        assert set(tags.keys()) == expected_keys

    def test_get_regime_tags_is_regime_specific_false_dict(self):
        """Dict entry with is_regime_specific=False."""
        from crabquant.strategies._registry_compat import get_regime_tags
        entry = _make_dict_entry(is_regime_specific=False)
        tags = get_regime_tags(entry)
        assert tags["is_regime_specific"] is False

    def test_to_tuple_dict_preserves_matrix_fn_none(self):
        """to_tuple from dict with matrix_fn=None should have None at index 4."""
        from crabquant.strategies._registry_compat import to_tuple
        entry = _make_dict_entry(matrix_fn=None)
        t = to_tuple(entry)
        assert t[4] is None

    def test_to_tuple_dict_preserves_fn(self):
        """to_tuple from dict should preserve the fn reference."""
        from crabquant.strategies._registry_compat import to_tuple
        fn = MagicMock(__name__="special_fn")
        entry = _make_dict_entry(fn=fn)
        t = to_tuple(entry)
        assert t[0] is fn

    def test_to_tuple_dict_preserves_defaults_and_grid(self):
        """to_tuple from dict should preserve defaults and grid."""
        from crabquant.strategies._registry_compat import to_tuple
        entry = _make_dict_entry()
        t = to_tuple(entry)
        assert t[1] == {"fast": 12, "slow": 26}
        assert t[2] == {"fast": [8, 12, 16]}

    def test_to_tuple_from_tuple_identity(self):
        """to_tuple from tuple should return same tuple object."""
        from crabquant.strategies._registry_compat import to_tuple
        entry = _make_tuple_entry()
        t = to_tuple(entry)
        assert t is entry  # Should be same object

    def test_to_tuple_from_dict_is_real_tuple(self):
        """to_tuple result should be a real tuple (not a list or other)."""
        from crabquant.strategies._registry_compat import to_tuple
        entry = _make_dict_entry()
        t = to_tuple(entry)
        assert isinstance(t, tuple)
        assert not isinstance(t, list)

    def test_get_defaults_empty_dict_entry(self):
        """Dict entry with empty defaults."""
        from crabquant.strategies._registry_compat import get_defaults
        entry = _make_dict_entry(defaults={})
        assert get_defaults(entry) == {}

    def test_get_regime_tags_multiple_regimes(self):
        """Dict entry with multiple preferred and acceptable regimes."""
        from crabquant.strategies._registry_compat import get_regime_tags
        entry = _make_dict_entry(
            preferred_regimes=["trending_up", "low_volatility"],
            acceptable_regimes=["ranging"],
            weak_regimes=["high_volatility", "crash"],
            regime_sharpes={"trending_up": 1.5, "low_volatility": 1.2, "ranging": 0.5},
            is_regime_specific=True,
        )
        tags = get_regime_tags(entry)
        assert len(tags["preferred_regimes"]) == 2
        assert "ranging" in tags["acceptable_regimes"]
        assert len(tags["weak_regimes"]) == 2
        assert tags["regime_sharpes"]["trending_up"] == 1.5
        assert tags["is_regime_specific"] is True

    def test_mixed_access_patterns(self):
        """Test accessing multiple fields from the same entry."""
        from crabquant.strategies._registry_compat import (
            get_fn, get_defaults, get_grid, get_description,
            get_matrix_fn, get_regime_tags, to_tuple,
        )
        entry = _make_dict_entry()

        fn = get_fn(entry)
        defaults = get_defaults(entry)
        grid = get_grid(entry)
        desc = get_description(entry)
        matrix = get_matrix_fn(entry)
        tags = get_regime_tags(entry)
        t = to_tuple(entry)

        # All should work without errors
        assert callable(fn)
        assert isinstance(defaults, dict)
        assert isinstance(grid, dict)
        assert isinstance(desc, str)
        assert len(t) == 5
        assert isinstance(tags, dict)

    def test_get_defaults_returns_same_dict_reference(self):
        """get_defaults should return the actual dict from the entry."""
        from crabquant.strategies._registry_compat import get_defaults
        d = {"fast": 12}
        entry = _make_dict_entry(defaults=d)
        assert get_defaults(entry) is d

    def test_get_grid_returns_same_dict_reference(self):
        """get_grid should return the actual dict from the entry."""
        from crabquant.strategies._registry_compat import get_grid
        g = {"fast": [8, 12]}
        entry = _make_dict_entry(grid=g)
        assert get_grid(entry) is g

    def test_two_element_tuple_defaults(self):
        """Tuple with exactly 2 elements (fn, defaults)."""
        from crabquant.strategies._registry_compat import get_defaults, get_grid, get_description
        fn = MagicMock()
        defaults = {"a": 1}
        entry = (fn, defaults)
        assert get_defaults(entry) is defaults
        assert get_grid(entry) == {}
        assert get_description(entry) == ""

    def test_three_element_tuple(self):
        """Tuple with 3 elements (fn, defaults, grid)."""
        from crabquant.strategies._registry_compat import get_grid, get_description, get_matrix_fn
        fn = MagicMock()
        grid = {"x": [1, 2]}
        entry = (fn, {}, grid)
        assert get_grid(entry) is grid
        assert get_description(entry) == ""
        assert get_matrix_fn(entry) is None

    def test_four_element_tuple(self):
        """Tuple with 4 elements (fn, defaults, grid, description)."""
        from crabquant.strategies._registry_compat import get_description, get_matrix_fn
        fn = MagicMock()
        entry = (fn, {}, {}, "My Strategy")
        assert get_description(entry) == "My Strategy"
        assert get_matrix_fn(entry) is None
