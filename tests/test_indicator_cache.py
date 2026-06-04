"""Tests for crabquant.indicator_cache module."""

import numpy as np
import pandas as pd
import pytest

from crabquant.indicator_cache import (
    _hash_series,
    _hash_dataframe,
    _make_key,
    cached_indicator,
    clear_cache,
    cache_size,
    _cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure cache is empty before each test."""
    clear_cache()
    yield
    clear_cache()


class TestHashSeries:
    def test_same_series_same_hash(self):
        s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
        h1 = _hash_series(s)
        h2 = _hash_series(s)
        assert h1 == h2

    def test_different_values_different_hash(self):
        s1 = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
        s2 = pd.Series([4.0, 5.0, 6.0], index=pd.date_range("2020-01-01", periods=3))
        assert _hash_series(s1) != _hash_series(s2)

    def test_different_index_different_hash(self):
        idx1 = pd.date_range("2020-01-01", periods=3)
        idx2 = pd.date_range("2020-01-04", periods=3)
        s1 = pd.Series([1.0, 2.0, 3.0], index=idx1)
        s2 = pd.Series([1.0, 2.0, 3.0], index=idx2)
        assert _hash_series(s1) != _hash_series(s2)

    def test_different_name_different_hash(self):
        s1 = pd.Series([1.0, 2.0], name="close")
        s2 = pd.Series([1.0, 2.0], name="open")
        assert _hash_series(s1) != _hash_series(s2)

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        h = _hash_series(s)
        assert isinstance(h, str)
        assert len(h) == 32  # md5 hex digest


class TestHashDataFrame:
    def test_same_df_same_hash(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        assert _hash_dataframe(df) == _hash_dataframe(df)

    def test_different_columns_different_hash(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0]})
        df2 = pd.DataFrame({"b": [1.0, 2.0]})
        assert _hash_dataframe(df1) != _hash_dataframe(df2)

    def test_different_values_different_hash(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0]})
        df2 = pd.DataFrame({"a": [3.0, 4.0]})
        assert _hash_dataframe(df1) != _hash_dataframe(df2)

    def test_column_order_matters(self):
        df1 = pd.DataFrame({"a": [1.0], "b": [2.0]})
        df2 = pd.DataFrame({"b": [2.0], "a": [1.0]})
        # Different column iteration order → different hash
        assert _hash_dataframe(df1) != _hash_dataframe(df2)


class TestMakeKey:
    def test_basic_key(self):
        key = _make_key("rsi", (14,), {})
        assert key[0] == "rsi"
        assert key[1] == (14,)
        assert key[2] == ()

    def test_kwargs_sorted(self):
        key1 = _make_key("macd", (), {"fast": 12, "slow": 26})
        key2 = _make_key("macd", (), {"slow": 26, "fast": 12})
        assert key1 == key2

    def test_series_args_hashed(self):
        s = pd.Series([1.0, 2.0, 3.0])
        key = _make_key("rsi", (s,), {"length": 14})
        # The series should be hashed, not stored raw
        assert isinstance(key[1][0], str)

    def test_dataframe_args_hashed(self):
        df = pd.DataFrame({"close": [1.0, 2.0]})
        key = _make_key("macd", (df,), {})
        assert isinstance(key[1][0], str)

    def test_mixed_args(self):
        s = pd.Series([1.0, 2.0])
        key = _make_key("test", (s, 14, "literal"), {"x": 1})
        assert key[1][0] == _hash_series(s)  # series hashed
        assert key[1][1] == 14  # int passed through
        assert key[1][2] == "literal"  # str passed through


class TestCachedIndicator:
    def test_cache_miss_computes(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = cached_indicator("rsi", close, length=5)
        assert isinstance(result, pd.Series)
        assert cache_size() == 1

    def test_cache_hit_returns_same(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        r1 = cached_indicator("rsi", close, length=5)
        r2 = cached_indicator("rsi", close, length=5)
        assert r1 is r2  # same object reference
        assert cache_size() == 1

    def test_different_params_different_cache_entries(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        cached_indicator("rsi", close, length=5)
        cached_indicator("rsi", close, length=10)
        assert cache_size() == 2

    def test_different_indicators_different_cache_entries(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        cached_indicator("rsi", close, length=5)
        cached_indicator("ema", close, length=5)
        assert cache_size() == 2

    def test_different_data_different_cache_entries(self):
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        s2 = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        cached_indicator("rsi", s1, length=5)
        cached_indicator("rsi", s2, length=5)
        assert cache_size() == 2

    def test_invalid_indicator_raises(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(AttributeError, match="pandas_ta has no indicator 'nonexistent_indicator'"):
            cached_indicator("nonexistent_indicator", close)


class TestClearCache:
    def test_clears_all_entries(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        cached_indicator("rsi", close, length=5)
        cached_indicator("ema", close, length=5)
        assert cache_size() == 2

        clear_cache()
        assert cache_size() == 0

    def test_clear_allows_recompute(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        r1 = cached_indicator("rsi", close, length=5)
        clear_cache()
        r2 = cached_indicator("rsi", close, length=5)
        # After clear, it recomputes — may be equal but not same object
        assert isinstance(r2, pd.Series)
        assert cache_size() == 1


class TestCacheSize:
    def test_empty_cache(self):
        assert cache_size() == 0

    def test_size_increases(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert cache_size() == 0
        cached_indicator("rsi", close, length=5)
        assert cache_size() == 1
        cached_indicator("ema", close, length=5)
        assert cache_size() == 2


# ── _hash_series extended ─────────────────────────────────────────────────

class TestHashSeriesExtended:
    def test_range_index(self):
        """Hash should work with default RangeIndex."""
        s = pd.Series([1.0, 2.0, 3.0])
        h = _hash_series(s)
        assert isinstance(h, str)
        assert len(h) == 32

    def test_integer_index(self):
        """Hash should differ for different integer indices."""
        s1 = pd.Series([1.0, 2.0], index=[0, 1])
        s2 = pd.Series([1.0, 2.0], index=[10, 11])
        assert _hash_series(s1) != _hash_series(s2)

    def test_series_with_nans(self):
        """Hash should handle NaN values without error."""
        s = pd.Series([1.0, np.nan, 3.0])
        h = _hash_series(s)
        assert isinstance(h, str)
        assert len(h) == 32

    def test_series_with_all_nans(self):
        """Hash should work for all-NaN series."""
        s = pd.Series([np.nan, np.nan, np.nan])
        h = _hash_series(s)
        assert isinstance(h, str)

    def test_same_values_different_index_type(self):
        """Same values but different index type should differ."""
        s1 = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2))
        s2 = pd.Series([1.0, 2.0], index=[0, 1])
        assert _hash_series(s1) != _hash_series(s2)

    def test_long_series(self):
        """Hash should work for long series."""
        s = pd.Series(np.random.randn(10000))
        h = _hash_series(s)
        assert isinstance(h, str)
        assert len(h) == 32

    def test_unnamed_series(self):
        """Hash should work for unnamed series (name=None)."""
        s = pd.Series([1.0, 2.0, 3.0])  # default name is None
        h = _hash_series(s)
        assert isinstance(h, str)


# ── _hash_dataframe extended ──────────────────────────────────────────────

class TestHashDataFrameExtended:
    def test_single_column(self):
        """Hash should work with single-column DataFrame."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        h = _hash_dataframe(df)
        assert isinstance(h, str)
        assert len(h) == 32

    def test_empty_dataframe(self):
        """Hash should work for empty DataFrame."""
        df = pd.DataFrame()
        h = _hash_dataframe(df)
        assert isinstance(h, str)

    def test_datetime_index(self):
        """Hash should incorporate datetime index."""
        df1 = pd.DataFrame(
            {"a": [1.0, 2.0]},
            index=pd.date_range("2020-01-01", periods=2),
        )
        df2 = pd.DataFrame(
            {"a": [1.0, 2.0]},
            index=pd.date_range("2020-06-01", periods=2),
        )
        assert _hash_dataframe(df1) != _hash_dataframe(df2)

    def test_many_columns(self):
        """Hash should work with many columns."""
        df = pd.DataFrame({f"col_{i}": np.random.randn(100) for i in range(20)})
        h = _hash_dataframe(df)
        assert isinstance(h, str)
        assert len(h) == 32

    def test_same_data_renamed_column(self):
        """Renaming a column should change the hash."""
        df1 = pd.DataFrame({"a": [1.0, 2.0]})
        df2 = df1.rename(columns={"a": "b"})
        assert _hash_dataframe(df1) != _hash_dataframe(df2)


# ── _make_key extended ────────────────────────────────────────────────────

class TestMakeKeyExtended:
    def test_no_args_no_kwargs(self):
        """Key with no args or kwargs should work."""
        key = _make_key("test", (), {})
        assert key == ("test", (), ())

    def test_only_kwargs(self):
        """Key with only kwargs should sort them."""
        key = _make_key("macd", (), {"slow": 26, "fast": 12, "signal": 9})
        assert key[2] == (("fast", 12), ("signal", 9), ("slow", 26))

    def test_only_positional_args(self):
        """Key with only positional args (no pandas objects)."""
        key = _make_key("test", (1, 2, 3), {})
        assert key == ("test", (1, 2, 3), ())

    def test_kwargs_order_doesnt_matter(self):
        """Different kwargs order should produce same key."""
        k1 = _make_key("ind", (), {"z": 1, "a": 2, "m": 3})
        k2 = _make_key("ind", (), {"a": 2, "m": 3, "z": 1})
        assert k1 == k2

    def test_same_series_produces_same_hash_in_key(self):
        """Same series in args should produce same hashed key."""
        s = pd.Series([1.0, 2.0, 3.0])
        k1 = _make_key("rsi", (s,), {"length": 14})
        k2 = _make_key("rsi", (s,), {"length": 14})
        assert k1 == k2


# ── cached_indicator extended ─────────────────────────────────────────────

class TestCachedIndicatorExtended:
    def test_macd_returns_dataframe(self):
        """MACD indicator should return a DataFrame."""
        close = pd.Series(np.random.randn(50).cumsum() + 100)
        result = cached_indicator("macd", close, fast=12, slow=26, signal=9)
        assert isinstance(result, pd.DataFrame)

    def test_ema_returns_series(self):
        """EMA indicator should return a Series."""
        close = pd.Series([1.0] * 20)
        result = cached_indicator("ema", close, length=5)
        assert isinstance(result, pd.Series)

    def test_atr_indicator(self):
        """ATR indicator should compute and cache."""
        close = pd.Series(np.random.randn(30).cumsum() + 100)
        high = close + 1
        low = close - 1
        result = cached_indicator("atr", high, low, close, length=14)
        assert isinstance(result, pd.Series)
        assert cache_size() == 1

    def test_different_name_same_data(self):
        """Different indicator names should be separate cache entries."""
        close = pd.Series(np.random.randn(20).cumsum() + 100)
        cached_indicator("rsi", close, length=5)
        cached_indicator("ema", close, length=5)
        cached_indicator("sma", close, length=5)
        assert cache_size() == 3

    def test_cache_hit_is_same_object(self):
        """Cache hit should return the exact same object (identity)."""
        close = pd.Series(np.random.randn(20).cumsum() + 100)
        r1 = cached_indicator("ema", close, length=10)
        r2 = cached_indicator("ema", close, length=10)
        assert r1 is r2

    def test_cache_miss_is_different_object(self):
        """Cache miss should return a new object."""
        close = pd.Series(np.random.randn(20).cumsum() + 100)
        r1 = cached_indicator("ema", close, length=10)
        clear_cache()
        r2 = cached_indicator("ema", close, length=10)
        # After clear, it recomputes — value equal but not same object
        assert r1 is not r2

    def test_multiple_lengths_separate_entries(self):
        """Same indicator with different lengths should be separate entries."""
        close = pd.Series(np.random.randn(50).cumsum() + 100)
        for length in [5, 10, 20, 50]:
            cached_indicator("sma", close, length=length)
        assert cache_size() == 4

    def test_no_kwargs_indicator(self):
        """Indicator with no kwargs should work."""
        close = pd.Series(np.random.randn(20).cumsum() + 100)
        result = cached_indicator("log_return", close)
        assert isinstance(result, pd.Series)
        assert cache_size() == 1


# ── clear_cache extended ──────────────────────────────────────────────────

class TestClearCacheExtended:
    def test_clear_empty_no_error(self):
        """Clearing an already-empty cache should not error."""
        assert cache_size() == 0
        clear_cache()  # Should not raise
        assert cache_size() == 0

    def test_clear_multiple_times(self):
        """Clearing multiple times should be safe."""
        close = pd.Series(np.random.randn(20).cumsum() + 100)
        cached_indicator("rsi", close, length=5)
        assert cache_size() == 1

        clear_cache()
        assert cache_size() == 0

        clear_cache()
        assert cache_size() == 0

        clear_cache()
        assert cache_size() == 0


# ── _cache module-level behavior ──────────────────────────────────────────

class TestCacheModuleBehavior:
    def test_cache_is_module_level_dict(self):
        """_cache should be a module-level dict."""
        assert isinstance(_cache, dict)

    def test_cache_shared_across_imports(self):
        """Cache should be shared when importing from different paths."""
        from crabquant.indicator_cache import cached_indicator as ci1, _cache as c1
        import importlib
        mod = importlib.import_module("crabquant.indicator_cache")
        c2 = mod._cache

        # Should be the same dict object
        assert c1 is c2

    def test_cache_entries_are_tuple_keys(self):
        """Cache keys should be tuples."""
        close = pd.Series(np.random.randn(20).cumsum() + 100)
        cached_indicator("rsi", close, length=5)
        key = list(_cache.keys())[0]
        assert isinstance(key, tuple)
