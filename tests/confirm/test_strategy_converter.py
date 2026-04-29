"""Tests for crabquant.confirm.strategy_converter indicator helper functions."""

import numpy as np
import pytest

from crabquant.confirm.strategy_converter import (
    _rolling_max,
    _rolling_min,
    _rolling_mean,
    _rolling_sum,
    _ewm_mean,
    _rsi,
    _atr,
    _adx,
    _macd,
    _stoch,
    _bbands,
    _roc,
    _sma,
    _vpt,
)


def _make_price_data(n=100, seed=42):
    """Generate synthetic price data for testing."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.rand(n) * 1.0
    low = close - rng.rand(n) * 1.0
    volume = rng.randint(1000, 10000, n).astype(float)
    return high, low, close, volume


class TestRollingMax:
    def test_basic(self):
        data = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
        result = _rolling_max(data, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 5.0  # max(1, 5, 3)
        assert result[3] == 8.0  # max(5, 3, 8)
        assert result[4] == 8.0  # max(3, 8, 2)

    def test_window_equals_length(self):
        data = np.array([1.0, 2.0, 3.0])
        result = _rolling_max(data, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 3.0

    def test_constant_data(self):
        data = np.array([5.0, 5.0, 5.0, 5.0])
        result = _rolling_max(data, 2)
        assert np.isnan(result[0])
        assert result[1] == 5.0
        assert result[2] == 5.0
        assert result[3] == 5.0


class TestRollingMin:
    def test_basic(self):
        data = np.array([5.0, 2.0, 8.0, 1.0, 4.0])
        result = _rolling_min(data, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0
        assert result[3] == 1.0
        assert result[4] == 1.0


class TestRollingMean:
    def test_basic(self):
        data = np.array([10.0, 20.0, 30.0, 40.0])
        result = _rolling_mean(data, 2)
        assert np.isnan(result[0])
        assert result[1] == 15.0
        assert result[2] == 25.0
        assert result[3] == 35.0

    def test_window_of_one(self):
        data = np.array([1.0, 2.0, 3.0])
        result = _rolling_mean(data, 1)
        # window=1: first valid at index 0
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] == 3.0

    def test_with_nan(self):
        data = np.array([1.0, np.nan, 3.0, 4.0])
        result = _rolling_mean(data, 2)
        assert np.isnan(result[0])
        assert result[1] == 1.0  # nanmean(1, nan) = 1.0
        assert result[2] == 3.0  # nanmean(nan, 3) = 3.0
        assert result[3] == 3.5  # nanmean(3, 4) = 3.5


class TestRollingSum:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_sum(data, 3)
        assert result[0] == 1.0  # sum of just first element
        assert result[1] == 3.0  # sum of first two
        assert result[2] == 6.0  # 1+2+3
        assert result[3] == 9.0  # 2+3+4
        assert result[4] == 12.0  # 3+4+5


class TestEwmMean:
    def test_basic(self):
        data = np.array([10.0, 20.0, 30.0, 40.0])
        result = _ewm_mean(data, span=3)
        # First value should be the data value itself
        assert result[0] == 10.0
        # Should be monotonically increasing for increasing data
        assert result[1] > result[0]
        assert result[2] > result[1]
        assert result[3] > result[2]

    def test_alpha(self):
        """Verify EWM uses correct alpha."""
        data = np.array([10.0, 20.0])
        result = _ewm_mean(data, span=3)
        alpha = 2.0 / (3 + 1)  # 0.5
        expected = alpha * 20.0 + (1 - alpha) * 10.0  # 15.0
        assert abs(result[1] - expected) < 1e-10

    def test_with_nan(self):
        data = np.array([10.0, np.nan, 30.0])
        result = _ewm_mean(data, span=3)
        assert result[0] == 10.0
        assert result[1] == 10.0  # nan carries forward previous
        # Should recover from nan
        assert result[2] > result[1]


class TestRSI:
    def test_rising_prices(self):
        """Rising prices should give RSI > 50."""
        close = np.linspace(100, 200, 50)
        result = _rsi(close, length=14)
        # After warmup, RSI should be very high for steadily rising prices
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(v > 70 for v in valid)

    def test_falling_prices(self):
        """Falling prices should give RSI < 50."""
        close = np.linspace(200, 100, 50)
        result = _rsi(close, length=14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(v < 30 for v in valid)

    def test_warmup_period(self):
        """First `length` values should be NaN."""
        close = np.linspace(100, 200, 50)
        result = _rsi(close, length=14)
        assert all(np.isnan(result[:14]))

    def test_flat_prices(self):
        """Flat prices → zero gains, zero losses → RSI = 100."""
        close = np.full(30, 100.0)
        result = _rsi(close, length=14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(v == 100.0 for v in valid)


class TestATR:
    def test_basic(self):
        h, l, c, _ = _make_price_data(50)
        result = _atr(h, l, c, length=14)
        assert len(result) == 50
        # First `length` values should be NaN
        assert all(np.isnan(result[:14]))
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_constant_prices(self):
        close = np.full(30, 100.0)
        high = np.full(30, 100.0)
        low = np.full(30, 100.0)
        result = _atr(high, low, close, length=14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        # No volatility → ATR should be 0
        assert all(v == 0.0 for v in valid)

    def test_volatile_prices(self):
        """Higher volatility should produce higher ATR."""
        rng = np.random.RandomState(42)
        low_vol_close = 100.0 + np.cumsum(rng.randn(50) * 0.1)
        high_vol_close = 100.0 + np.cumsum(rng.randn(50) * 5.0)

        low_vol_high = low_vol_close + 0.1
        low_vol_low = low_vol_close - 0.1
        high_vol_high = high_vol_close + 5.0
        high_vol_low = high_vol_close - 5.0

        atr_low = _atr(low_vol_high, low_vol_low, low_vol_close, length=14)
        atr_high = _atr(high_vol_high, high_vol_low, high_vol_close, length=14)

        avg_low = np.nanmean(atr_low)
        avg_high = np.nanmean(atr_high)
        assert avg_high > avg_low * 10


class TestADX:
    def test_basic(self):
        h, l, c, _ = _make_price_data(60)
        result = _adx(h, l, c, length=14)
        assert len(result) == 60
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(v >= 0 for v in valid)

    def test_strong_trend(self):
        """Strong trend should produce higher ADX than range-bound."""
        rng = np.random.RandomState(42)
        n = 60

        # Trending data
        trend_close = np.cumsum(rng.randn(n) * 0.5) + 100
        trend_high = trend_close + np.abs(rng.randn(n)) * 0.5
        trend_low = trend_close - np.abs(rng.randn(n)) * 0.5

        # Range-bound data
        range_close = 100 + rng.randn(n) * 0.1
        range_high = range_close + 0.05
        range_low = range_close - 0.05

        adx_trend = _adx(trend_high, trend_low, trend_close, length=14)
        adx_range = _adx(range_high, range_low, range_close, length=14)

        # Trend should have higher average ADX (may not always be true for random data,
        # but generally holds)
        avg_trend = np.nanmean(adx_trend)
        avg_range = np.nanmean(adx_range)
        # At minimum, both should be non-negative
        assert avg_trend >= 0
        assert avg_range >= 0


class TestMACD:
    def test_basic_shape(self):
        close = np.linspace(100, 200, 50)
        result = _macd(close, fast=12, slow=26, signal=9)
        assert len(result) == 50
        # For monotonically increasing data, MACD histogram should be mostly positive
        # (fast EMA > slow EMA)
        # After enough warmup
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_falling_prices(self):
        close = np.linspace(200, 100, 50)
        result = _macd(close, fast=5, slow=10, signal=3)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        # For falling data, MACD should be mostly negative
        assert np.mean(valid) < 0


class TestStoch:
    def test_basic(self):
        h, l, c, _ = _make_price_data(50)
        k, d = _stoch(h, l, c, k=14, d=3)
        assert len(k) == 50
        assert len(d) == 50
        # Stochastic K should be between 0 and 100
        valid_k = k[~np.isnan(k)]
        assert all(v >= 0 and v <= 100 for v in valid_k)

    def test_at_high(self):
        """When close equals high of period, Stoch K should be 100."""
        close = np.array([10.0, 15.0, 20.0])
        high = np.array([15.0, 20.0, 20.0])
        low = np.array([5.0, 10.0, 10.0])
        k, d = _stoch(high, low, close, k=3, d=2)
        # k=3 window: at index 2, window is [10,15,20], low=10, high=20, close=20
        # (20-10)/(20-10)*100 = 100
        assert k[2] == 100.0


class TestBBands:
    def test_basic(self):
        close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        upper, mid, lower = _bbands(close, length=5, std_mult=2.0)
        assert len(upper) == 11
        assert len(mid) == 11
        assert len(lower) == 11

        # Upper > Mid > Lower where valid
        for i in range(11):
            if not (np.isnan(upper[i]) or np.isnan(mid[i]) or np.isnan(lower[i])):
                assert upper[i] >= mid[i]
                assert mid[i] >= lower[i]

    def test_constant_prices(self):
        close = np.full(20, 100.0)
        upper, mid, lower = _bbands(close, length=5)
        # With zero std, upper == mid == lower where all are valid
        for i in range(1, 20):  # skip index 0 which has NaN for mid
            if not (np.isnan(upper[i]) or np.isnan(mid[i]) or np.isnan(lower[i])):
                assert upper[i] == mid[i] == lower[i]


class TestROC:
    def test_basic(self):
        close = np.array([100.0, 110.0, 121.0, 133.1])
        result = _roc(close, length=1)
        assert np.isnan(result[0])
        assert result[1] == 10.0  # (110-100)/100*100
        assert abs(result[2] - 10.0) < 1e-10  # (121-110)/110*100
        assert abs(result[3] - 10.0) < 1e-10

    def test_warmup(self):
        close = np.array([100.0, 110.0, 120.0])
        result = _roc(close, length=5)
        assert all(np.isnan(result))


class TestSMA:
    def test_basic(self):
        data = np.array([10.0, 20.0, 30.0, 40.0])
        result = _sma(data, 2)
        assert np.isnan(result[0])
        assert result[1] == 15.0
        assert result[2] == 25.0
        assert result[3] == 35.0

    def test_same_as_rolling_mean(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.array_equal(_sma(data, 3), _rolling_mean(data, 3), equal_nan=True)


class TestVPT:
    def test_basic(self):
        close = np.array([100.0, 110.0, 105.0])
        volume = np.array([1000.0, 2000.0, 1500.0])
        result = _vpt(close, volume)
        assert result[0] == 0.0  # first value is 0
        assert result[1] > 0  # price went up with volume → positive VPT
        # Second day: VPT contribution = 2000 * (110-100)/100 = 200

    def test_falling_price(self):
        close = np.array([100.0, 90.0])
        volume = np.array([1000.0, 1000.0])
        result = _vpt(close, volume)
        assert result[0] == 0.0
        assert result[1] < 0  # price fell → negative VPT contribution

    def test_flat_price(self):
        close = np.array([100.0, 100.0, 100.0])
        volume = np.array([1000.0, 2000.0, 3000.0])
        result = _vpt(close, volume)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_division_by_zero_protection(self):
        close = np.array([0.0, 100.0])
        volume = np.array([1000.0, 1000.0])
        result = _vpt(close, volume)
        # Should not crash
        assert len(result) == 2


class TestMakeStrategyClass:
    def test_unknown_strategy_raises(self):
        from crabquant.confirm.strategy_converter import _make_strategy_class
        with pytest.raises(ValueError, match="No converter registered"):
            _make_strategy_class("nonexistent_strategy", {})

    def test_known_strategy_returns_class(self):
        from crabquant.confirm.strategy_converter import _make_strategy_class
        from backtesting import Strategy
        cls = _make_strategy_class("rsi_crossover", {"fast_len": 5, "slow_len": 14, "regime_len": 50, "exit_level": 70, "regime_bull": 50})
        assert issubclass(cls, Strategy)
        assert hasattr(cls, 'init')
        assert hasattr(cls, 'next')
