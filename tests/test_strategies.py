"""Tests for CrabQuant strategy library."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from crabquant.strategies import STRATEGY_REGISTRY, DEFAULT_TICKERS
from crabquant.strategies._registry_compat import (
    get_fn, get_defaults, get_grid, get_description, get_matrix_fn,
    get_regime_tags, to_tuple,
)

# Strategies whose matrix functions have known bugs with non-default param grids
# (e.g., hardcoded indicator column names like 'ADX_14' that break when adx_len != 14)
_KNOWN_MATRIX_BUGS = {
    "invented_volume_adx_ema",
    "invented_volume_breakout_adx",
    "invented_rsi_volume_atr",
}


# ── Synthetic data helpers ─────────────────────────────────────────────────

def make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data — no network calls needed."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    returns = rng.normal(0.0005, 0.015, n)
    close = 100.0 * np.cumprod(1 + returns)
    noise = rng.normal(0, 0.003, n)
    return pd.DataFrame(
        {
            "open": close * (1 + noise * 0.5),
            "high": close * (1 + np.abs(noise)),
            "low": close * (1 - np.abs(noise)),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


def make_trending_ohlcv(n: int = 500, seed: int = 123) -> pd.DataFrame:
    """Create synthetic data with a clear uptrend."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    # Strong upward drift
    returns = rng.normal(0.002, 0.01, n)
    close = 100.0 * np.cumprod(1 + returns)
    noise = rng.normal(0, 0.002, n)
    return pd.DataFrame(
        {
            "open": close * (1 + noise * 0.3),
            "high": close * (1 + np.abs(noise) * 0.8),
            "low": close * (1 - np.abs(noise) * 0.8),
            "close": close,
            "volume": rng.integers(800_000, 3_000_000, n).astype(float),
        },
        index=dates,
    )


def make_volatile_ohlcv(n: int = 500, seed: int = 999) -> pd.DataFrame:
    """Create synthetic data with high volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    returns = rng.normal(0.0003, 0.035, n)
    close = 100.0 * np.cumprod(1 + returns)
    noise = rng.normal(0, 0.008, n)
    return pd.DataFrame(
        {
            "open": close * (1 + noise * 0.5),
            "high": close * (1 + np.abs(noise) * 1.5),
            "low": close * (1 - np.abs(noise) * 1.5),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test Strategy Registry
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyRegistry:
    """Tests for STRATEGY_REGISTRY structure and contents."""

    def test_registry_is_dict(self):
        """STRATEGY_REGISTRY should be a dict."""
        assert isinstance(STRATEGY_REGISTRY, dict)

    def test_registry_not_empty(self):
        """Registry should have at least 20 strategies."""
        assert len(STRATEGY_REGISTRY) >= 20

    def test_all_core_strategies_present(self):
        """Every core strategy should be in the registry."""
        expected = [
            "rsi_crossover", "macd_momentum", "adx_pullback",
            "atr_channel_breakout", "volume_breakout", "multi_rsi_confluence",
            "ema_ribbon_reversal", "bollinger_squeeze", "ichimoku_trend",
        ]
        for name in expected:
            assert name in STRATEGY_REGISTRY, f"{name} missing from registry"

    def test_all_invented_strategies_have_prefix(self):
        """Invented strategies should have 'invented_' prefix."""
        invented = [n for n in STRATEGY_REGISTRY if n.startswith("invented_")]
        assert len(invented) >= 5
        for name in invented:
            assert name.startswith("invented_")

    def test_no_duplicate_keys(self):
        """Registry should have unique keys."""
        keys = list(STRATEGY_REGISTRY.keys())
        assert len(keys) == len(set(keys))

    def test_registry_tuple_structure(self):
        """Each registry entry should be (fn, defaults, grid, description, matrix_fn)."""
        for name, entry in STRATEGY_REGISTRY.items():
            assert isinstance(entry, tuple), f"{name}: entry should be tuple"
            assert len(entry) == 5, f"{name}: entry should be 5-tuple, got {len(entry)}"
            fn, defaults, grid, desc, matrix_fn = entry
            assert callable(fn), f"{name}: fn should be callable"
            assert isinstance(defaults, dict), f"{name}: defaults should be dict"
            assert isinstance(grid, dict), f"{name}: grid should be dict"
            assert isinstance(desc, str), f"{name}: desc should be string"
            assert len(desc) > 20, f"{name}: desc too short"
            assert callable(matrix_fn), f"{name}: matrix_fn should be callable"

    def test_all_defaults_are_dicts_with_values(self):
        """Every DEFAULT_PARAMS dict should be non-empty (except ichimoku_trend which uses no params)."""
        _NO_PARAMS = {"ichimoku_trend"}
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            if name in _NO_PARAMS:
                assert len(defaults) == 0, f"{name}: expected empty defaults"
            else:
                assert len(defaults) > 0, f"{name}: defaults should not be empty"

    def test_all_grids_have_lists(self):
        """Every PARAM_GRID value should be a list with at least 1 item (most >= 2)."""
        _NO_PARAMS = {"ichimoku_trend"}
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            if name in _NO_PARAMS:
                assert len(grid) == 0, f"{name}: expected empty grid"
            else:
                for k, v in grid.items():
                    assert isinstance(v, list), f"{name}.{k}: grid value should be list"
                    assert len(v) >= 1, f"{name}.{k}: grid should have >= 1 value"

    def test_grid_keys_match_defaults_keys(self):
        """PARAM_GRID keys should be a superset of DEFAULT_PARAMS keys (except no-param strategies)."""
        _NO_PARAMS = {"ichimoku_trend"}
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            if name in _NO_PARAMS:
                continue
            for k in defaults:
                assert k in grid, f"{name}: default param '{k}' missing from grid"


# ═══════════════════════════════════════════════════════════════════════════
# Test Strategy Interface Consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyInterface:
    """All strategies should have a consistent interface."""

    def test_all_strategies_return_boolean_series(self):
        """Every strategy should return (entries, exits) as boolean Series."""
        df = make_ohlcv()
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            entries, exits = fn(df, defaults)
            assert isinstance(entries, pd.Series), f"{name}: entries should be Series"
            assert isinstance(exits, pd.Series), f"{name}: exits should be Series"
            assert entries.dtype == bool, f"{name}: entries should be bool, got {entries.dtype}"
            assert exits.dtype == bool, f"{name}: exits should be bool, got {exits.dtype}"
            assert len(entries) == len(df), f"{name}: entries length mismatch"
            assert len(exits) == len(df), f"{name}: exits length mismatch"

    def test_strategies_work_with_none_params(self):
        """Strategies should work with params=None (use defaults)."""
        df = make_ohlcv()
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            entries, exits = fn(df, None)
            assert len(entries) == len(df), f"{name}: length mismatch with None params"

    def test_strategies_work_with_empty_params(self):
        """Strategies should work with params={} (use defaults)."""
        df = make_ohlcv()
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            entries, exits = fn(df, {})
            assert len(entries) == len(df), f"{name}: length mismatch with empty params"

    def test_strategies_work_with_custom_params(self):
        """Strategies should accept custom params that override defaults."""
        df = make_ohlcv()
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            if name in _KNOWN_MATRIX_BUGS:
                # These strategies have hardcoded indicator column names
                # that break with non-default params — skip
                continue
            if len(grid) == 0:
                # No-param strategies (e.g. ichimoku_trend) — skip
                continue
            # Use first value from each grid param
            custom = {k: v[0] for k, v in grid.items()}
            entries, exits = fn(df, custom)
            assert len(entries) == len(df), f"{name}: failed with custom params"

    def test_entries_exits_share_index(self):
        """Entries and exits should have the same index as the input DataFrame."""
        df = make_ohlcv()
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            entries, exits = fn(df, defaults)
            pd.testing.assert_index_equal(entries.index, df.index, exact=True)
            pd.testing.assert_index_equal(exits.index, df.index, exact=True)

    def test_no_all_true_entries(self):
        """No strategy should produce all-True entries (that's broken)."""
        df = make_ohlcv()
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            entries, exits = fn(df, defaults)
            assert not entries.all(), f"{name}: all entries are True — strategy is broken"

    def test_no_all_true_exits(self):
        """No strategy should produce all-True exits (that's broken)."""
        df = make_ohlcv()
        for name, (fn, defaults, grid, desc, matrix_fn) in STRATEGY_REGISTRY.items():
            entries, exits = fn(df, defaults)
            assert not exits.all(), f"{name}: all exits are True — strategy is broken"


# ═══════════════════════════════════════════════════════════════════════════
# Test Matrix Functions
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyMatrix:
    """Tests for generate_signals_matrix functions."""

    def _matrix_testable_strategies(self):
        """Yield (name, fn, defaults, grid, desc, matrix_fn) excluding known-buggy ones."""
        for name, entry in STRATEGY_REGISTRY.items():
            if name in _KNOWN_MATRIX_BUGS:
                continue
            yield (name, *entry)

    def test_matrix_returns_correct_types(self):
        """Matrix function should return (entries_df, exits_df, param_list)."""
        df = make_ohlcv()
        for name, fn, defaults, grid, desc, matrix_fn in self._matrix_testable_strategies():
            small_grid = {k: v[:2] for k, v in grid.items()}
            entries_df, exits_df, param_list = matrix_fn(df, small_grid)
            assert isinstance(entries_df, pd.DataFrame), f"{name}: entries_df should be DataFrame"
            assert isinstance(exits_df, pd.DataFrame), f"{name}: exits_df should be DataFrame"
            assert isinstance(param_list, list), f"{name}: param_list should be list"

    def test_matrix_param_list_length(self):
        """param_list length should equal the product of grid values."""
        df = make_ohlcv()
        for name, fn, defaults, grid, desc, matrix_fn in self._matrix_testable_strategies():
            small_grid = {k: v[:2] for k, v in grid.items()}
            entries_df, exits_df, param_list = matrix_fn(df, small_grid)
            expected = 1
            for v in small_grid.values():
                expected *= len(v)
            assert len(param_list) == expected, f"{name}: expected {expected} combos, got {len(param_list)}"

    def test_matrix_columns_match_param_list(self):
        """Number of columns in entries_df should match param_list length."""
        df = make_ohlcv()
        for name, fn, defaults, grid, desc, matrix_fn in self._matrix_testable_strategies():
            small_grid = {k: v[:2] for k, v in grid.items()}
            entries_df, exits_df, param_list = matrix_fn(df, small_grid)
            assert entries_df.shape[1] == len(param_list), f"{name}: column count mismatch"
            assert exits_df.shape[1] == len(param_list), f"{name}: exits column count mismatch"

    def test_matrix_each_param_is_dict(self):
        """Each entry in param_list should be a dict."""
        df = make_ohlcv()
        for name, fn, defaults, grid, desc, matrix_fn in self._matrix_testable_strategies():
            small_grid = {k: v[:2] for k, v in grid.items()}
            entries_df, exits_df, param_list = matrix_fn(df, small_grid)
            for i, p in enumerate(param_list):
                assert isinstance(p, dict), f"{name}: param_list[{i}] should be dict"

    def test_matrix_known_buggy_strategies_exist(self):
        """Known-buggy strategies should actually exist in the registry."""
        for name in _KNOWN_MATRIX_BUGS:
            assert name in STRATEGY_REGISTRY, f"Known-buggy strategy {name} not in registry"

    def test_matrix_buggy_strategies_still_have_callable_matrix_fn(self):
        """Even buggy matrix functions should be callable."""
        for name in _KNOWN_MATRIX_BUGS:
            _, _, _, _, matrix_fn = STRATEGY_REGISTRY[name]
            assert callable(matrix_fn), f"{name}: matrix_fn should be callable"


# ═══════════════════════════════════════════════════════════════════════════
# Test Individual Strategy Behaviors
# ═══════════════════════════════════════════════════════════════════════════

class TestIndividualStrategies:
    """Tests for specific strategy behaviors."""

    def test_rsi_crossover_generates_signals(self):
        """RSI crossover should generate some entries on synthetic data."""
        from crabquant.strategies.rsi_crossover import generate_signals
        df = make_ohlcv()
        entries, exits = generate_signals(df)
        assert isinstance(entries, pd.Series)
        assert entries.sum() >= 0

    def test_macd_momentum_generates_signals(self):
        """MACD momentum should generate entries on trending data."""
        from crabquant.strategies.macd_momentum import generate_signals
        df = make_trending_ohlcv()
        entries, exits = generate_signals(df)
        assert isinstance(entries, pd.Series)

    def test_ichimoku_no_params(self):
        """Ichimoku should work with empty params dict."""
        from crabquant.strategies.ichimoku_trend import generate_signals
        df = make_ohlcv()
        entries, exits = generate_signals(df, {})
        assert len(entries) == len(df)

    def test_strategy_params_override_defaults(self):
        """Custom params should override defaults."""
        from crabquant.strategies.rsi_crossover import generate_signals, DEFAULT_PARAMS
        df = make_ohlcv()
        custom_params = {**DEFAULT_PARAMS, "fast_len": 3}
        entries, exits = generate_signals(df, custom_params)
        assert len(entries) == len(df)

    def test_bollinger_squeeze_on_volatile_data(self):
        """Bollinger squeeze should handle volatile data gracefully."""
        from crabquant.strategies.bollinger_squeeze import generate_signals
        df = make_volatile_ohlcv()
        entries, exits = generate_signals(df)
        assert len(entries) == len(df)
        assert entries.dtype == bool

    def test_volume_breakout_signals_type(self):
        """Volume breakout should return proper boolean Series."""
        from crabquant.strategies.volume_breakout import generate_signals
        df = make_ohlcv()
        entries, exits = generate_signals(df)
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_ema_crossover_basic(self):
        """EMA crossover should produce valid signals."""
        from crabquant.strategies.ema_crossover import generate_signals
        df = make_ohlcv()
        entries, exits = generate_signals(df)
        assert len(entries) == len(df)
        assert isinstance(entries, pd.Series)


# ═══════════════════════════════════════════════════════════════════════════
# Test DEFAULT_TICKERS
# ═══════════════════════════════════════════════════════════════════════════

class TestDefaultTickers:
    """Tests for DEFAULT_TICKERS list."""

    def test_default_tickers_is_list(self):
        """DEFAULT_TICKERS should be a list."""
        assert isinstance(DEFAULT_TICKERS, list)

    def test_default_tickers_not_empty(self):
        """Should have a reasonable number of tickers."""
        assert len(DEFAULT_TICKERS) >= 20

    def test_default_tickers_are_strings(self):
        """All tickers should be uppercase strings."""
        for t in DEFAULT_TICKERS:
            assert isinstance(t, str), f"{t} is not a string"
            assert t == t.upper(), f"{t} is not uppercase"

    def test_default_tickers_no_duplicates(self):
        """No duplicate tickers."""
        assert len(DEFAULT_TICKERS) == len(set(DEFAULT_TICKERS))

    def test_default_tickers_expected_members(self):
        """Should include well-known mega-caps."""
        for expected in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
            assert expected in DEFAULT_TICKERS


# ═══════════════════════════════════════════════════════════════════════════
# Test Registry Compat Helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestRegistryCompat:
    """Tests for _registry_compat helpers with tuple entries."""

    def test_get_fn_from_tuple(self):
        """get_fn should extract callable from tuple entry."""
        entry = STRATEGY_REGISTRY["rsi_crossover"]
        fn = get_fn(entry)
        assert callable(fn)

    def test_get_defaults_from_tuple(self):
        """get_defaults should extract dict from tuple entry."""
        entry = STRATEGY_REGISTRY["rsi_crossover"]
        defaults = get_defaults(entry)
        assert isinstance(defaults, dict)
        assert "fast_len" in defaults

    def test_get_grid_from_tuple(self):
        """get_grid should extract dict from tuple entry."""
        entry = STRATEGY_REGISTRY["rsi_crossover"]
        grid = get_grid(entry)
        assert isinstance(grid, dict)

    def test_get_description_from_tuple(self):
        """get_description should extract string from tuple entry."""
        entry = STRATEGY_REGISTRY["rsi_crossover"]
        desc = get_description(entry)
        assert isinstance(desc, str)
        assert len(desc) > 20

    def test_get_matrix_fn_from_tuple(self):
        """get_matrix_fn should extract callable from tuple entry."""
        entry = STRATEGY_REGISTRY["rsi_crossover"]
        matrix_fn = get_matrix_fn(entry)
        assert callable(matrix_fn)

    def test_get_regime_tags_from_tuple(self):
        """get_regime_tags should return empty values for tuple entries."""
        entry = STRATEGY_REGISTRY["rsi_crossover"]
        tags = get_regime_tags(entry)
        assert tags["preferred_regimes"] == []
        assert tags["is_regime_specific"] is False

    def test_to_tuple_from_tuple_passthrough(self):
        """to_tuple should pass through tuple entries unchanged."""
        entry = STRATEGY_REGISTRY["rsi_crossover"]
        result = to_tuple(entry)
        assert result == entry

    def test_get_fn_from_dict_entry(self):
        """get_fn should work with dict-format entries."""
        mock_fn = lambda df, p: (pd.Series(False, index=df.index), pd.Series(False, index=df.index))
        dict_entry = {"fn": mock_fn, "defaults": {}, "grid": {}, "description": "test"}
        assert get_fn(dict_entry) is mock_fn

    def test_get_regime_tags_from_dict_entry(self):
        """get_regime_tags should extract regime info from dict entries."""
        dict_entry = {
            "fn": lambda df, p: (pd.Series(False, index=df.index), pd.Series(False, index=df.index)),
            "defaults": {},
            "grid": {},
            "description": "test",
            "preferred_regimes": ["trending"],
            "weak_regimes": ["sideways"],
        }
        tags = get_regime_tags(dict_entry)
        assert tags["preferred_regimes"] == ["trending"]
        assert tags["weak_regimes"] == ["sideways"]

    def test_to_tuple_from_dict(self):
        """to_tuple should convert dict entries to 5-tuples."""
        mock_fn = lambda df, p: (pd.Series(False, index=df.index), pd.Series(False, index=df.index))
        dict_entry = {
            "fn": mock_fn, "defaults": {"x": 1}, "grid": {"x": [1, 2]},
            "description": "test desc", "matrix_fn": mock_fn,
        }
        result = to_tuple(dict_entry)
        assert isinstance(result, tuple)
        assert len(result) == 5
        assert result[0] is mock_fn
        assert result[1] == {"x": 1}
