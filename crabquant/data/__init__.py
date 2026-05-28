"""
CrabQuant Data Loader

Handles OHLCV data loading from yfinance with caching.
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


def load_data(ticker: str, period: str = "2y", use_cache: bool = True) -> pd.DataFrame:
    """
    Load OHLCV data for a ticker.

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
        # Cache is valid for 1 day (market hours)
        if cache_age < timedelta(hours=20):
            logger.debug(f"Loading {ticker} from cache ({cache_age.total_seconds() / 3600:.0f}h old)")
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            return df

    # Fetch from yfinance
    import yfinance as yf

    logger.info(f"Fetching {ticker} from yfinance (period={period})")
    t = yf.Ticker(ticker)
    df = t.history(period=period)

    if df.empty:
        raise ValueError(f"No data returned from yfinance for '{ticker}'")

    # Standardize
    df.columns = [col.lower() for col in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    df.index = df.index.tz_localize(None)

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

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
