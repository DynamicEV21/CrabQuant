"""Tests for crabquant.indicator_cache module."""

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
