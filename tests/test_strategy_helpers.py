"""Tests for crabquant.strategies.strategy_helpers — Enhancement 13."""

import numpy as np
import pandas as pd
import pytest

from crabquant.strategies.strategy_helpers import atr, adx, obv, vwap, supertrend


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv(n: int = 200):
    """Generate synthetic OHLCV data with realistic bar ranges."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    # Generate realistic high/low with some variation
    intrabar_range = close * rng.uniform(0.005, 0.03, n)
    high = close + intrabar_range * rng.uniform(0.3, 1.0, n)
    low = close - intrabar_range * rng.uniform(0.3, 1.0, n)
    # Ensure high >= close >= low
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    volume = rng.integers(500_000, 2_000_000, size=n).astype(float)
    df = pd.DataFrame(
        {"high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


# ── ATR ────────────────────────────────────────────────────────────────────

class TestATR:
    def test_returns_correct_length(self, ohlcv):
        result = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result) == len(ohlcv)

    def test_first_values_are_nan(self, ohlcv):
        result = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        assert result.iloc[:13].isna().all()
        assert not result.iloc[13:].isna().all()

    def test_always_positive(self, ohlcv):
        result = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        valid = result.dropna()
        assert (valid > 0).all(), "ATR should always be positive where not NaN"


# ── ADX ────────────────────────────────────────────────────────────────────

class TestADX:
    def test_returns_correct_length(self, ohlcv):
        result = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result) == len(ohlcv)

    def test_between_0_and_100(self, ohlcv):
        result = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "ADX must be in [0, 100]"

    def test_warmup_period(self, ohlcv):
        result = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        warmup = 2 * 14 - 1  # 27
        assert result.iloc[:warmup].isna().all()


# ── OBV ────────────────────────────────────────────────────────────────────

class TestOBV:
    def test_returns_correct_length(self, ohlcv):
        result = obv(ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv)

    def test_cumulative_sum_matches(self, ohlcv):
        result = obv(ohlcv["close"], ohlcv["volume"])
        # OBV is the cumulative sum of (volume * sign(close_diff))
        direction = ohlcv["close"].diff().fillna(0)
        signed_vol = ohlcv["volume"] * np.sign(direction)
        expected = signed_vol.cumsum()
        pd.testing.assert_series_equal(result, expected)

    def test_first_value(self, ohlcv):
        result = obv(ohlcv["close"], ohlcv["volume"])
        # First bar: diff is NaN → filled with 0 → sign(0) = 0 → OBV = 0
        assert result.iloc[0] == 0.0


# ── VWAP ───────────────────────────────────────────────────────────────────

class TestVWAP:
    def test_returns_correct_length(self, ohlcv):
        result = vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv)

    def test_first_value_is_nan(self, ohlcv):
        result = vwap(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert pd.isna(result.iloc[0])

    def test_simple_data(self):
        """VWAP with hand-computable values."""
        h = pd.Series([12.0, 14.0])
        l = pd.Series([10.0, 12.0])
        c = pd.Series([11.0, 13.0])
        v = pd.Series([100.0, 200.0])
        result = vwap(h, l, c, v)
        # Bar 0: TP=11, VWAP = (11*100)/(100) = 11.0 → NaN by design
        assert pd.isna(result.iloc[0])
        # Bar 1: TP=(14+12+13)/3 = 13.0
        #   cum_tp_vol = 11*100 + 13*200 = 1100 + 2600 = 3700
        #   cum_vol = 100 + 200 = 300
        #   VWAP = 3700/300 ≈ 12.3333
        assert abs(result.iloc[1] - (3700.0 / 300.0)) < 1e-8

    def test_vwap_near_typical_price_with_uniform_volume(self):
        """With uniform volume, VWAP ≈ simple average of typical prices."""
        n = 50
        h = pd.Series(100.0 + np.arange(n) * 0.1)
        l = pd.Series(99.0 + np.arange(n) * 0.1)
        c = pd.Series(99.5 + np.arange(n) * 0.1)
        v = pd.Series(1000.0, index=range(n))
        result = vwap(h, l, c, v).dropna()
        # Cumulative VWAP includes all bars from 0 onward
        tp = (h + l + c) / 3.0
        # VWAP at bar i (i>=1) = sum(TP[0:i+1]*V[0:i+1]) / sum(V[0:i+1])
        expected = ((tp * v).cumsum() / v.cumsum()).iloc[1:]
        pd.testing.assert_series_equal(result, expected)


# ── Supertrend ─────────────────────────────────────────────────────────────

class TestSupertrend:
    def test_returns_correct_length(self, ohlcv):
        st, direction = supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(st) == len(ohlcv)
        assert len(direction) == len(ohlcv)

    def test_warmup_period(self, ohlcv):
        st, direction = supertrend(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], period=10
        )
        # First `period` bars should be NaN
        assert st.iloc[:10].isna().all()
        assert direction.iloc[:10].isna().all()

    def test_direction_values(self, ohlcv):
        _, direction = supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid = direction.dropna()
        assert set(valid.unique()).issubset({1.0, -1.0})

    def test_supertrend_line_below_close_in_bullish(self, ohlcv):
        """In bullish mode, supertrend line should be below close price."""
        st, direction = supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        # Use numpy arrays to avoid index-alignment pitfalls
        st_vals, d_vals, c_vals = st.values, direction.values, ohlcv["close"].values
        bullish_mask = d_vals == 1.0
        if bullish_mask.any():
            assert (st_vals[bullish_mask] <= c_vals[bullish_mask]).all(), (
                "Supertrend line should be ≤ close in bullish mode"
            )

    def test_supertrend_line_above_close_in_bearish(self, ohlcv):
        """In bearish mode, supertrend line should be above close price."""
        st, direction = supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        st_vals, d_vals, c_vals = st.values, direction.values, ohlcv["close"].values
        bearish_mask = d_vals == -1.0
        if bearish_mask.any():
            assert (st_vals[bearish_mask] >= c_vals[bearish_mask]).all(), (
                "Supertrend line should be ≥ close in bearish mode"
            )
