"""Tests for the CrabQuant confirmation module."""

import os
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from crabquant.confirm import ConfirmationResult, CONFIRM_THRESHOLDS, confirm_strategy
from crabquant.confirm.strategy_converter import (
    convert_strategy,
    _rsi, _atr, _ewm_mean, _sma, _macd, _roc, _adx, _bbands, _stoch, _vpt,
    _rolling_max, _rolling_min, _rolling_mean,
    CrabQuantBacktest,
)
from crabquant.confirm.runner import run_confirmation
from crabquant.confirm.batch import batch_confirm, _aggregate_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n=300, seed=42):
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range(end="2025-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)


def _make_trending_ohlcv(n=300, seed=42):
    """Generate trending OHLCV data (uptrend with pullbacks)."""
    np.random.seed(seed)
    dates = pd.date_range(end="2025-01-01", periods=n, freq="B")
    # Strong uptrend
    trend = np.linspace(80, 150, n)
    noise = np.random.randn(n) * 1.5
    close = trend + noise
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)


SAMPLE_DF = _make_ohlcv()
TRENDING_DF = _make_trending_ohlcv()


# ---------------------------------------------------------------------------
# Indicator helper tests
# ---------------------------------------------------------------------------

class TestIndicatorHelpers:
    """Test that our pure-numpy indicator helpers work correctly."""

    def test_rolling_max(self):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = _rolling_max(arr, 3)
        assert np.isnan(result[0])  # Partial window
        assert np.isnan(result[1])  # Partial window
        assert result[2] == 3.0  # Max of [1, 3, 2]
        assert result[3] == 5.0  # Max of [3, 2, 5]
        assert result[4] == 5.0  # Max of [2, 5, 4]

    def test_rolling_min(self):
        arr = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
        result = _rolling_min(arr, 3)
        assert np.isnan(result[0])  # Partial window
        assert np.isnan(result[1])  # Partial window
        assert result[2] == 3.0  # Min of [5, 3, 4]
        assert result[3] == 1.0  # Min of [3, 4, 1]
        assert result[4] == 1.0  # Min of [4, 1, 2]

    def test_rolling_mean(self):
        arr = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = _rolling_mean(arr, 3)
        assert np.isnan(result[0])  # Partial window
        assert np.isnan(result[1])  # Partial window
        assert result[2] == 4.0  # Mean of [2, 4, 6]
        assert result[3] == 6.0  # Mean of [4, 6, 8]
        assert result[4] == 8.0  # Mean of [6, 8, 10]

    def test_ewm_mean(self):
        arr = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = _ewm_mean(arr, 3)
        # First value should be the input
        assert result[0] == 10.0
        # Should be monotonically increasing
        assert result[4] > result[3] > result[2] > result[1]

    def test_rsi(self):
        arr = np.linspace(100, 110, 100)  # Steadily rising
        result = _rsi(arr, 14)
        # RSI of steadily rising series should be high (> 50)
        assert not np.isnan(result[-1])
        assert result[-1] > 50

    def test_rsi_falling(self):
        arr = np.linspace(110, 100, 100)  # Steadily falling
        result = _rsi(arr, 14)
        assert not np.isnan(result[-1])
        assert result[-1] < 50

    def test_atr(self):
        n = 50
        close = np.linspace(100, 110, n)
        high = close + 1.0
        low = close - 1.0
        result = _atr(high, low, close, 14)
        assert not np.isnan(result[-1])
        assert result[-1] > 0

    def test_macd(self):
        close = np.linspace(100, 120, 100)
        result = _macd(close, 12, 26, 9)
        # For steadily rising price, histogram should be positive (eventually)
        assert not np.isnan(result[-1])

    def test_roc(self):
        arr = np.array([100, 105, 110, 115, 120], dtype=float)
        result = _roc(arr, 1)
        assert np.isnan(result[0])
        assert result[1] == 5.0  # 5% change

    def test_bbands(self):
        close = np.random.randn(100).cumsum() + 100
        upper, mid, lower = _bbands(close, 20, 2.0)
        assert not np.isnan(upper[-1])
        assert not np.isnan(mid[-1])
        assert not np.isnan(lower[-1])
        assert upper[-1] > mid[-1] > lower[-1]

    def test_stoch(self):
        n = 50
        close = np.linspace(100, 110, n)
        high = close + 1.0
        low = close - 1.0
        k, d = _stoch(high, low, close, 14, 3)
        assert not np.isnan(k[-1])
        assert 0 <= k[-1] <= 100

    def test_vpt(self):
        close = np.array([100, 105, 103, 108, 110], dtype=float)
        volume = np.array([1000, 2000, 1500, 3000, 2500], dtype=float)
        result = _vpt(close, volume)
        assert result[0] == 0
        assert result[1] > 0  # Price went up with volume


# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------

class TestStrategyConverter:
    """Test that the converter produces valid backtesting.py Strategy classes."""

    # All 17 strategy names from the registry
    ALL_STRATEGIES = [
        "rsi_crossover", "macd_momentum", "adx_pullback",
        "atr_channel_breakout", "volume_breakout", "multi_rsi_confluence",
        "ema_ribbon_reversal", "bollinger_squeeze", "ichimoku_trend",
        "invented_momentum_rsi_atr", "invented_momentum_rsi_stoch",
        "vpt_crossover", "roc_ema_volume", "bb_stoch_macd",
        "rsi_regime_dip", "ema_crossover", "injected_momentum_atr_volume",
    ]

    def test_all_strategies_have_converters(self):
        """Every registry strategy should have a converter."""
        for name in self.ALL_STRATEGIES:
            cls = convert_strategy(name, {})
            assert issubclass(cls, CrabQuantBacktest), f"{name} converter should produce CrabQuantBacktest subclass"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="No converter"):
            convert_strategy("nonexistent_strategy", {})

    def test_converter_sets_params(self):
        cls = convert_strategy("ema_crossover", {"fast_len": 9, "slow_len": 21})
        assert cls._cq_params == {"fast_len": 9, "slow_len": 21}
        assert cls._cq_position_pct == 0.95
        assert cls._cq_slippage_pct == 0.001

    def test_converter_custom_position_size(self):
        cls = convert_strategy("ema_crossover", {"fast_len": 9, "slow_len": 21},
                               position_pct=0.8, slippage_pct=0.002)
        assert cls._cq_position_pct == 0.8
        assert cls._cq_slippage_pct == 0.002


# ---------------------------------------------------------------------------
# Backtest execution tests (using backtesting.py)
# ---------------------------------------------------------------------------

class TestRunner:
    """Test the confirmation runner with synthetic data."""

    def _prep_df(self, df):
        """Prepare DataFrame for backtesting.py."""
        bt_df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }).copy()
        if bt_df.index.tz is not None:
            bt_df.index = bt_df.index.tz_localize(None)
        return bt_df

    def test_run_returns_confirmation_result(self):
        """Runner should return a ConfirmationResult."""
        result = run_confirmation(
            "ema_crossover", "TEST",
            {"fast_len": 9, "slow_len": 21},
            df=SAMPLE_DF.copy(),
        )
        assert isinstance(result, ConfirmationResult)
        assert isinstance(result.sharpe, float)
        assert isinstance(result.total_return, float)
        assert isinstance(result.max_dd, float)
        assert isinstance(result.trades, int)
        assert isinstance(result.passed, bool)
        assert isinstance(result.notes, list)

    def test_run_with_insufficient_data(self):
        """Runner should handle insufficient data gracefully."""
        tiny_df = SAMPLE_DF.head(10)
        result = run_confirmation(
            "ema_crossover", "TEST",
            {"fast_len": 9, "slow_len": 21},
            df=tiny_df,
        )
        assert isinstance(result, ConfirmationResult)
        assert result.passed is False

    def test_run_unknown_strategy(self):
        """Runner should handle unknown strategy gracefully."""
        result = run_confirmation(
            "nonexistent", "TEST", {},
            df=SAMPLE_DF.copy(),
        )
        assert isinstance(result, ConfirmationResult)
        assert result.passed is False
        assert any("No converter" in n for n in result.notes)

    @pytest.mark.parametrize("strategy,params", [
        ("ema_crossover", {"fast_len": 9, "slow_len": 21}),
        ("rsi_crossover", {"fast_len": 7, "slow_len": 21, "regime_len": 50, "regime_bull": 55, "exit_level": 40}),
        ("rsi_regime_dip", {"regime_len": 50, "timing_len": 14, "dip_level": 40, "recovery_level": 60, "regime_bull": 50}),
        ("invented_momentum_rsi_stoch", {"rsi_len": 14, "rsi_oversold": 35, "volume_window": 20, "volume_mult": 1.2}),
        ("roc_ema_volume", {"roc_len": 10, "ema_len": 20, "vol_sma_len": 20, "atr_len": 14, "atr_mult": 2.0, "trailing_len": 20}),
        ("bb_stoch_macd", {"bb_len": 20, "bb_std": 2.0, "stoch_k": 14, "stoch_d": 3, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}),
        ("macd_momentum", {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "exit_hist": -0.5, "sma_len": 50, "volume_window": 20, "volume_mult": 1.2}),
        ("adx_pullback", {"adx_len": 14, "adx_threshold": 25, "ema_len": 20, "take_atr": 3}),
        ("atr_channel_breakout", {"ema_len": 20, "atr_len": 10, "mult": 2.0, "vol_mult": 1.2}),
        ("volume_breakout", {"dc_len": 20, "atr_len": 14, "vol_len": 20, "vol_mult": 1.5}),
        ("multi_rsi_confluence", {"rsi1": 7, "rsi2": 14, "rsi3": 28, "thresh": 35, "vol_mult": 1.0, "exit_thresh": 65}),
        ("ema_ribbon_reversal", {"dip_level": 40}),
        ("bollinger_squeeze", {"bb_len": 20, "bb_std": 2.0, "squeeze_len": 50, "squeeze_mult": 0.8, "vol_mult": 1.2}),
        ("ichimoku_trend", {}),
        ("invented_momentum_rsi_atr", {"rsi_len": 14, "rsi_pullback": 45, "rsi_overbought": 78, "roc_len": 14, "roc_threshold": 0, "ema_len": 50, "atr_len": 14, "atr_exit_mult": 3.0}),
        ("vpt_crossover", {"vpt_signal_len": 20, "rsi_len": 14, "vol_sma_len": 20, "rsi_entry": 40, "rsi_exit": 80}),
        ("injected_momentum_atr_volume", {"roc_len": 5, "roc_threshold": 0.5, "vol_sma_len": 10, "vol_threshold": 1.2, "rsi_len": 10, "rsi_min_uptrend": 25, "rsi_max_downtrend": 75, "ema_short_len": 10, "ema_long_len": 20, "atr_len": 10, "atr_mult": 1.5}),
    ])
    def test_all_strategies_run_without_error(self, strategy, params):
        """Every strategy should run without crashing on sample data."""
        result = run_confirmation(
            strategy, "TEST", params,
            df=SAMPLE_DF.copy(),
        )
        assert isinstance(result, ConfirmationResult)

    def test_zero_trades_strategy(self):
        """Strategy with very restrictive params might produce 0 trades."""
        # Use extreme params that are unlikely to trigger
        result = run_confirmation(
            "rsi_crossover", "TEST",
            {"fast_len": 3, "slow_len": 5, "regime_len": 200, "regime_bull": 99, "exit_level": 1},
            df=SAMPLE_DF.copy(),
        )
        assert isinstance(result, ConfirmationResult)
        # 0 trades should fail
        if result.trades == 0:
            assert result.passed is False


# ---------------------------------------------------------------------------
# Batch confirmation tests
# ---------------------------------------------------------------------------

class TestBatchConfirmation:
    """Test batch confirmation logic."""

    def test_aggregate_robust(self):
        """All pass → ROBUST."""
        results = [
            ConfirmationResult(passed=True),   # 2y, 0% slip
            ConfirmationResult(passed=True),   # 2y, 0.1% slip
            ConfirmationResult(passed=True),   # 2y, 0.2% slip
            ConfirmationResult(passed=True),   # 1y, 0% slip
            ConfirmationResult(passed=True),   # 1y, 0.1% slip
            ConfirmationResult(passed=True),   # 1y, 0.2% slip
            ConfirmationResult(passed=True),   # 6mo, 0% slip
            ConfirmationResult(passed=True),   # 6mo, 0.1% slip
            ConfirmationResult(passed=True),   # 6mo, 0.2% slip
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is True
        assert fragile is False

    def test_aggregate_fragile(self):
        """Pass at 0% slip, fail at 0.2% → FRAGILE."""
        results = [
            ConfirmationResult(passed=True),    # 2y, 0% slip
            ConfirmationResult(passed=True),    # 2y, 0.1% slip
            ConfirmationResult(passed=False),   # 2y, 0.2% slip
            ConfirmationResult(passed=True),    # 1y, 0% slip
            ConfirmationResult(passed=True),    # 1y, 0.1% slip
            ConfirmationResult(passed=False),   # 1y, 0.2% slip
            ConfirmationResult(passed=True),    # 6mo, 0% slip
            ConfirmationResult(passed=True),    # 6mo, 0.1% slip
            ConfirmationResult(passed=False),   # 6mo, 0.2% slip
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is True
        assert robust is False
        assert fragile is True

    def test_aggregate_failed(self):
        """Fail at 0% slip → FAILED."""
        results = [
            ConfirmationResult(passed=False),   # 2y, 0% slip
            ConfirmationResult(passed=False),   # 2y, 0.1% slip
            ConfirmationResult(passed=False),   # 2y, 0.2% slip
        ]
        base, robust, fragile = _aggregate_results(results)
        assert base is False
        assert robust is False
        assert fragile is False

    def test_aggregate_empty(self):
        """No results → all False."""
        base, robust, fragile = _aggregate_results([])
        assert base is False
        assert robust is False
        assert fragile is False

    def test_batch_confirm_with_mocked_data(self):
        """Batch confirm with mocked load_data."""
        winner = {
            "strategy": "ema_crossover",
            "ticker": "MOCK",
            "params": {"fast_len": 9, "slow_len": 21},
        }
        with patch("crabquant.confirm.batch.load_data", return_value=SAMPLE_DF.copy()):
            result = batch_confirm(winner, n_periods=1)
        assert isinstance(result, ConfirmationResult)
        assert result.verdict in ("ROBUST", "FRAGILE", "FAILED")
        assert len(result.notes) > 0

    def test_batch_confirm_verdict_types(self):
        """Verdict should be one of ROBUST, FRAGILE, FAILED."""
        for verdict in ["ROBUST", "FRAGILE", "FAILED"]:
            r = ConfirmationResult(verdict=verdict)
            assert r.verdict == verdict


# ---------------------------------------------------------------------------
# Confirmation thresholds tests
# ---------------------------------------------------------------------------

class TestThresholds:
    """Test that thresholds are correctly configured."""

    def test_thresholds_values(self):
        t = CONFIRM_THRESHOLDS
        assert t["sharpe"] == 1.0
        assert t["max_drawdown"] == 0.30
        assert t["total_return"] == 0.05
        assert t["min_trades"] == 5
        assert t["expectancy"] == 0.0

    def test_confirmation_result_to_dict(self):
        r = ConfirmationResult(
            sharpe=1.5, total_return=0.1, max_dd=-0.05,
            trades=10, win_rate=0.6, profit_factor=2.0,
            expectancy=50.0, passed=True, verdict="ROBUST",
            notes=["test"],
        )
        d = r.to_dict()
        assert d["sharpe"] == 1.5
        assert d["passed"] is True
        assert d["verdict"] == "ROBUST"
        assert isinstance(d["notes"], list)


# ---------------------------------------------------------------------------
# Integration: confirm_strategy convenience function
# ---------------------------------------------------------------------------

class TestConfirmStrategy:
    """Test the main confirm_strategy convenience function."""

    def test_confirm_strategy_with_df(self):
        result = confirm_strategy(
            "ema_crossover", "TEST",
            {"fast_len": 9, "slow_len": 21},
            df=SAMPLE_DF.copy(),
        )
        assert isinstance(result, ConfirmationResult)

    def test_confirm_strategy_no_df_uses_load_data(self):
        """When no df provided, should try to load data."""
        with patch("crabquant.confirm.runner.load_data", return_value=SAMPLE_DF.copy()) as mock_load:
            result = confirm_strategy("ema_crossover", "MOCK", {"fast_len": 9, "slow_len": 21})
        mock_load.assert_called_once()
        assert isinstance(result, ConfirmationResult)
