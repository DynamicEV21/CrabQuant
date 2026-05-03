"""
CrabQuant Data Loader

Handles OHLCV data loading from yfinance with caching.
Supports hybrid loading: massive.com parquet overlay + yfinance deep history.
Ported and simplified from QuantFactory's data_loader.py.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(os.environ.get("CRABQUANT_CACHE_DIR", os.path.expanduser("~/.cache/crabquant")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Massive.com parquet directory (for high-quality recent-year overlay)
MASSIVE_DATA_DIR = Path(
    os.environ.get("CRABQUANT_DATA_DIR", "/home/Zev/development/quant-projects/financial-data/stocks/daily/")
)

# Expected OHLCV columns
_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _load_massive_parquet(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load massive.com parquet data for a ticker if available.

    Reads {TICKER}.parquet from CRABQUANT_DATA_DIR, converts to the
    standard CrabQuant schema (DatetimeIndex named 'Date', no timezone,
    lowercase OHLCV columns only).

    Returns None if the file doesn't exist or can't be read.
    """
    parquet_path = MASSIVE_DATA_DIR / f"{ticker}.parquet"
    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path)
        if df.empty:
            logger.debug(f"Massive.com parquet for {ticker} is empty, skipping")
            return None

        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]

        # Identify the timestamp column (could be 'timestamp', 'date', etc.)
        ts_col = None
        for candidate in ("timestamp", "date", "datetime"):
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
            # Timestamp is already the index
            pass
        elif ts_col is not None:
            df = df.set_index(ts_col)

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.debug(f"Massive.com parquet for {ticker}: no DatetimeIndex, skipping")
            return None

        # Strip timezone and set index name to 'Date'
        df.index = df.index.tz_localize(None)
        df.index.name = "Date"

        # Keep only OHLCV columns
        available = [c for c in _OHLCV_COLS if c in df.columns]
        if not available:
            logger.debug(f"Massive.com parquet for {ticker}: no OHLCV columns, skipping")
            return None

        df = df[available].copy()

        # Ensure numeric types
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()
        # Sort chronologically and deduplicate
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        logger.debug(f"Loaded massive.com parquet for {ticker}: {len(df)} bars")
        return df

    except Exception as e:
        logger.warning(f"Failed to load massive.com parquet for {ticker}: {e}")
        return None


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize a DataFrame to CrabQuant OHLCV format."""
    df.columns = [col.lower() for col in df.columns]
    df = df[[c for c in _OHLCV_COLS if c in df.columns]].copy()
    df = df.dropna()
    df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df


def load_data(ticker: str, period: str = "2y", use_cache: bool = True) -> pd.DataFrame:
    """
    Load OHLCV data for a ticker.

    Uses hybrid loading: yfinance for full history with massive.com parquet
    overlay for higher-quality data on overlapping dates. The parquet data
    takes priority where dates overlap.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Data period to fetch (e.g., '1y', '2y', '5y')
        use_cache: Whether to use cached data

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume

    Raises:
        ValueError: If data cannot be fetched
    """
    cache_file = CACHE_DIR / f"{ticker}_{period}.pkl"

    # Try cache first
    if use_cache and cache_file.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        # Cache is valid for 7 days (OHLCV data changes slowly enough for validation)
        if cache_age < timedelta(days=7):
            logger.debug(f"Loading {ticker} from cache ({cache_age.total_seconds() / 3600:.0f}h old)")
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            return df

    # --- Hybrid fetch: yfinance (base) + massive.com parquet (overlay) ---

    # 1. Load massive.com parquet if available
    massive_df = _load_massive_parquet(ticker)

    # 2. Load yfinance for full history
    import yfinance as yf

    logger.info(f"Fetching {ticker} from yfinance (period={period})")
    t = yf.Ticker(ticker)
    yf_df = t.history(period=period)

    if yf_df.empty and massive_df is None:
        raise ValueError(f"No data returned from yfinance for '{ticker}' and no massive.com parquet")

    # 3. Standardize yfinance data
    if not yf_df.empty:
        yf_df = _standardize_df(yf_df)

    # 4. Merge: if both sources exist, overlay massive.com on top of yfinance
    if massive_df is not None and not yf_df.empty:
        # Combine: start with yfinance, then update with massive.com (massive takes priority)
        # Use combine_first so massive values fill in where both have data
        df = massive_df.combine_first(yf_df)
        # Ensure standard column order
        df = df[_OHLCV_COLS].copy()
        df = df.sort_index()
        logger.info(f"Hybrid merge for {ticker}: yfinance={len(yf_df)}, massive={len(massive_df)}, "
                     f"merged={len(df)}")
    elif massive_df is not None:
        df = massive_df
        logger.info(f"Using massive.com parquet only for {ticker}: {len(df)} bars")
    else:
        df = yf_df

    # Save to cache
    if use_cache:
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
        logger.debug(f"Cached {ticker} ({len(df)} rows)")

    logger.info(f"Loaded {ticker}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def load_multi(tickers: list[str], period: str = "2y", use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """
    Load data for multiple tickers.

    Args:
        tickers: List of ticker symbols
        period: Data period
        use_cache: Whether to use cached data

    Returns:
        Dict mapping ticker -> DataFrame
    """
    result = {}
    for ticker in tickers:
        try:
            result[ticker] = load_data(ticker, period, use_cache)
        except Exception as e:
            logger.warning(f"Failed to load {ticker}: {e}")
    return result


def clear_cache(ticker: Optional[str] = None):
    """Clear cached data. If ticker is None, clears all cache."""
    if ticker:
        for f in CACHE_DIR.glob(f"{ticker}_*.pkl"):
            f.unlink()
            logger.info(f"Cleared cache for {ticker}")
    else:
        for f in CACHE_DIR.glob("*.pkl"):
            f.unlink()
        logger.info("Cleared all cache")
