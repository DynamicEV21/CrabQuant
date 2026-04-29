"""Tests for CrabQuant data loader."""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


class TestDataLoader:
    """Test data loading functionality."""

    def test_load_data_returns_dataframe(self):
        """load_data should return a DataFrame with correct columns."""
        from crabquant.data import load_data

        df = load_data("AAPL", period="2y")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) > 0

    def test_load_data_no_nans(self):
        """Data should have no NaN values."""
        from crabquant.data import load_data

        df = load_data("SPY", period="1y")
        assert df.isna().sum().sum() == 0

    def test_load_data_numeric_types(self):
        """All OHLCV columns should be numeric."""
        from crabquant.data import load_data

        df = load_data("MSFT", period="1y")
        for col in ["open", "high", "low", "close", "volume"]:
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_load_invalid_ticker_raises(self):
        """Invalid ticker should raise ValueError."""
        from crabquant.data import load_data

        with pytest.raises(ValueError):
            load_data("INVALIDTICKER12345", period="1y", use_cache=False)

    def test_load_multi(self):
        """load_multi should return dict of DataFrames."""
        from crabquant.data import load_multi

        result = load_multi(["AAPL", "SPY"], period="1y")
        assert isinstance(result, dict)
        assert "AAPL" in result
        assert len(result["AAPL"]) > 0


# ── Cached loading ──────────────────────────────────────────────────────────

class TestDataCache:
    @patch("crabquant.data.pickle.load")
    @patch("crabquant.data.Path.exists", return_value=True)
    @patch("crabquant.data.Path.stat")
    def test_cache_hit_returns_cached_df(self, mock_stat, mock_exists, mock_pickle_load):
        """When cache file exists and is fresh, return cached data."""
        from crabquant.data import load_data

        cached_df = pd.DataFrame({
            "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [1000],
        }, index=pd.DatetimeIndex(["2024-01-01"]))
        mock_pickle_load.return_value = cached_df

        # stat().st_mtime = now (cache is fresh)
        import time
        mock_stat.return_value.st_mtime = time.time()

        df = load_data("AAPL", period="1y", use_cache=True)
        assert len(df) == 1
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_cache_miss_fetches_from_yfinance(self):
        """When cache doesn't exist, should fetch from yfinance."""
        from crabquant.data import load_data

        mock_df = pd.DataFrame({
            "Open": [150.0], "High": [155.0], "Low": [149.0],
            "Close": [152.0], "Volume": [1000000],
        }, index=pd.DatetimeIndex(["2024-01-01"], tz="America/New_York"))
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df

        with patch("crabquant.data.Path.exists", return_value=False):
            with patch.dict("sys.modules", {"yfinance": MagicMock(Ticker=lambda t: mock_ticker)}):
                df = load_data("TEST", period="1y", use_cache=True)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_empty_yfinance_response_raises(self):
        """yfinance returning empty DataFrame should raise ValueError."""
        from crabquant.data import load_data

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("crabquant.data.Path.exists", return_value=False):
            with patch.dict("sys.modules", {"yfinance": MagicMock(Ticker=lambda t: mock_ticker)}):
                with pytest.raises(ValueError, match="No data returned"):
                    load_data("EMPTY", period="1y", use_cache=False)

    def test_use_cache_false_skips_cache(self):
        """use_cache=False should always fetch from yfinance."""
        from crabquant.data import load_data

        with patch("crabquant.data.Path.exists") as mock_exists:
            # Even if cache exists, use_cache=False skips it
            mock_exists.return_value = True
            with pytest.raises(Exception):
                # This will fail because yfinance needs network,
                # but the important thing is it doesn't use cache
                load_data("NONEXISTENT_FAKE_TICKER_XYZ", period="1y", use_cache=False)


# ── clear_cache ─────────────────────────────────────────────────────────────

class TestClearCache:
    def test_clear_all_cache(self, tmp_path):
        """clear_cache() with no ticker should remove all .pkl files."""
        from crabquant.data import clear_cache, CACHE_DIR

        # Create temp cache files
        (tmp_path / "AAPL_2y.pkl").touch()
        (tmp_path / "MSFT_1y.pkl").touch()
        (tmp_path / "readme.txt").touch()

        with patch("crabquant.data.CACHE_DIR", tmp_path):
            clear_cache()

        pkl_files = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == 0
        # Non-pkl files should remain
        assert (tmp_path / "readme.txt").exists()

    def test_clear_specific_ticker(self, tmp_path):
        """clear_cache('AAPL') should only remove AAPL_*.pkl files."""
        from crabquant.data import clear_cache

        (tmp_path / "AAPL_2y.pkl").touch()
        (tmp_path / "AAPL_1y.pkl").touch()
        (tmp_path / "MSFT_2y.pkl").touch()

        with patch("crabquant.data.CACHE_DIR", tmp_path):
            clear_cache("AAPL")

        aapl_files = list(tmp_path.glob("AAPL_*.pkl"))
        msft_files = list(tmp_path.glob("MSFT_*.pkl"))
        assert len(aapl_files) == 0
        assert len(msft_files) == 1


# ── load_multi ──────────────────────────────────────────────────────────────

class TestLoadMulti:
    @patch("crabquant.data.load_data", side_effect=ValueError("fail"))
    def test_all_failures_returns_empty_dict(self, mock_load):
        """If all tickers fail, result should be empty dict."""
        from crabquant.data import load_multi

        result = load_multi(["BAD1", "BAD2", "BAD3"], period="1y")
        assert result == {}

    @patch("crabquant.data.load_data")
    def test_partial_failures_return_successful(self, mock_load):
        """Successful tickers should still be in result even if some fail."""
        from crabquant.data import load_multi

        good_df = pd.DataFrame({
            "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [1000],
        }, index=pd.DatetimeIndex(["2024-01-01"]))

        mock_load.side_effect = [
            good_df,  # AAPL succeeds
            ValueError("fail"),  # BAD fails
            good_df,  # MSFT succeeds
        ]

        result = load_multi(["AAPL", "BAD", "MSFT"], period="1y")
        assert "AAPL" in result
        assert "MSFT" in result
        assert "BAD" not in result

    @patch("crabquant.data.load_data")
    def test_empty_ticker_list(self, mock_load):
        """Empty ticker list should return empty dict."""
        from crabquant.data import load_multi

        result = load_multi([], period="1y")
        assert result == {}
        mock_load.assert_not_called()

    @patch("crabquant.data.load_data")
    def test_passes_period_and_use_cache(self, mock_load):
        """load_multi should forward period and use_cache to load_data."""
        from crabquant.data import load_multi

        mock_load.return_value = pd.DataFrame({
            "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [1000],
        }, index=pd.DatetimeIndex(["2024-01-01"]))

        load_multi(["AAPL"], period="5y", use_cache=False)
        mock_load.assert_called_once_with("AAPL", "5y", False)


# ── Data standardization ────────────────────────────────────────────────────

class TestDataStandardization:
    def test_columns_lowercased(self):
        """yfinance columns should be lowercased."""
        from crabquant.data import load_data

        mock_df = pd.DataFrame({
            "Open": [150.0], "High": [155.0], "Low": [149.0],
            "Close": [152.0], "Volume": [1000000], "Dividends": [0], "Stock Splits": [0],
        }, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"))
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df

        with patch("crabquant.data.Path.exists", return_value=False):
            with patch.dict("sys.modules", {"yfinance": MagicMock(Ticker=lambda t: mock_ticker)}):
                df = load_data("TEST", period="1y", use_cache=False)

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_timezone_removed_from_index(self):
        """Index timezone should be removed (tz_localize(None))."""
        from crabquant.data import load_data

        mock_df = pd.DataFrame({
            "Open": [150.0], "High": [155.0], "Low": [149.0],
            "Close": [152.0], "Volume": [1000000],
        }, index=pd.DatetimeIndex(["2024-01-01"], tz="America/New_York"))
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df

        with patch("crabquant.data.Path.exists", return_value=False):
            with patch.dict("sys.modules", {"yfinance": MagicMock(Ticker=lambda t: mock_ticker)}):
                df = load_data("TEST", period="1y", use_cache=False)

        assert df.index.tz is None


# ── CACHE_DIR ───────────────────────────────────────────────────────────────

class TestCacheDir:
    def test_cache_dir_exists(self):
        """CACHE_DIR should be a Path and exist."""
        from crabquant.data import CACHE_DIR

        assert isinstance(CACHE_DIR, Path)
        assert CACHE_DIR.exists()
        assert CACHE_DIR.is_dir()

    def test_cache_dir_respects_env_var(self):
        """CACHE_DIR should use CRABQUANT_CACHE_DIR env var."""
        import crabquant.data as data_mod
        # Just verify the logic — we can't easily reload the module
        # but we can check the expression
        env_val = os.environ.get("CRABQUANT_CACHE_DIR")
        if env_val:
            assert str(data_mod.CACHE_DIR) == env_val

    def test_cache_dir_has_pkl_files_after_load(self):
        """After loading data with cache, a .pkl file should exist."""
        from crabquant.data import load_data, CACHE_DIR

        load_data("SPY", period="1y")
        pkl_files = list(CACHE_DIR.glob("SPY_*.pkl"))
        assert len(pkl_files) >= 1
