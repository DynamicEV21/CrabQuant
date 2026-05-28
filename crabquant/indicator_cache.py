"""
Indicator Cache

Caches pandas_ta indicator results across strategy calls so that identical
(indicator_name, params, data_hash) lookups skip recomputation.

Usage in strategies:
    from crabquant.indicator_cache import cached_indicator, clear_cache

    # Inside generate_signals_matrix:
    rsi = cached_indicator("rsi", (close, 14))

    # Or as a drop-in for pandas_ta calls:
    rsi = cached_indicator("rsi", close, length=14)
    macd = cached_indicator("macd", close, fast=12, slow=26, signal=9)

The cache key is built from:
  1. The indicator function name
  2. A hash of the input data (index + values for Series, or column tuple for DataFrames)
  3. A tuple of sorted keyword arguments

Call clear_cache() between independent backtest runs to free memory.
"""

from __future__ import annotations

import hashlib
import pandas as pd
import pandas_ta

# Module-level dict cache: {(name, data_hash, params_key): result}
_cache: dict[tuple, pd.Series | pd.DataFrame] = {}


def _hash_series(s: pd.Series) -> str:
    """Create a stable hash for a pandas Series based on index + values."""
    h = hashlib.md5()
    # Hash index using its values (works for DatetimeIndex, RangeIndex, etc.)
    h.update(pd.util.hash_pandas_object(s.index, hash_key="crabquant").values.tobytes())
    # Hash the series name
    h.update(str(s.name).encode())
    # Hash the series values
    h.update(pd.util.hash_pandas_object(s, hash_key="crabquant").values.tobytes())
    return h.hexdigest()


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Create a stable hash for a pandas DataFrame based on index + columns."""
    h = hashlib.md5()
    # Hash index using hash_pandas_object (works for all index types)
    h.update(pd.util.hash_pandas_object(df.index, hash_key="crabquant").values.tobytes())
    for col in df.columns:
        h.update(col.encode())
        h.update(pd.util.hash_pandas_object(df[col], hash_key="crabquant").values.tobytes())
    return h.hexdigest()


def _make_key(name: str, args: tuple, kwargs: dict) -> tuple:
    """Build a cache key from indicator name, args, and sorted kwargs."""
    # Hash any pandas objects in args
    hashed_args = []
    for a in args:
        if isinstance(a, pd.Series):
            hashed_args.append(_hash_series(a))
        elif isinstance(a, pd.DataFrame):
            hashed_args.append(_hash_dataframe(a))
        else:
            hashed_args.append(a)
    # Sort kwargs for deterministic keys
    sorted_kwargs = tuple(sorted(kwargs.items()))
    return (name, tuple(hashed_args), sorted_kwargs)


def cached_indicator(name: str, *args, **kwargs) -> pd.Series | pd.DataFrame:
    """
    Compute a pandas_ta indicator with caching.

    Args:
        name: pandas_ta indicator name (e.g., "rsi", "macd", "ema", "atr")
        *args: Positional args (typically a Series and sometimes a length)
        **kwargs: Keyword args (e.g., length=14, fast=12, slow=26)

    Returns:
        The indicator result (Series or DataFrame)

    Example:
        rsi = cached_indicator("rsi", close, length=14)
        ema = cached_indicator("ema", close, length=50)
        macd = cached_indicator("macd", close, fast=12, slow=26, signal=9)
    """
    key = _make_key(name, args, kwargs)
    if key in _cache:
        return _cache[key]

    if not hasattr(pandas_ta, name):
        raise AttributeError(f"pandas_ta has no indicator '{name}'")

    result = getattr(pandas_ta, name)(*args, **kwargs)
    _cache[key] = result
    return result


def clear_cache() -> None:
    """Clear the indicator cache. Call between independent backtest runs."""
    _cache.clear()


def cache_size() -> int:
    """Return the number of cached indicator results."""
    return len(_cache)
