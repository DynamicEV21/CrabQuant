"""Tests for CrabQuant data loader."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path


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
