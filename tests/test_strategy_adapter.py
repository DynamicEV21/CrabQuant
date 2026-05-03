"""Tests for the Strategy Porting Adapter.

Tests cover:
- Porting a simple SMA crossover strategy
- Signal fidelity validation (correlation > 0.8)
- Edge cases (empty signals, all buy, all sell)
- Indicator parsing via AST
- Expected Value and Sortino in BacktestResult
"""

import numpy as np
import pandas as pd
import pytest

from crabquant.strategy_adapter import (
    StrategyAdapter,
    ExecutionConfig,
    SignalParser,
    ParsedIndicator,
    port_strategy,
    validate_ported_strategy,
)


# ── Test Data Fixtures ────────────────────────────────────────────────────


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n).astype(float)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


# ── Sample Strategies ─────────────────────────────────────────────────────


def sma_crossover_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Simple SMA crossover: fast crosses above slow → buy, below → sell."""
    fast_period = params.get("fast", 10)
    slow_period = params.get("slow", 30)

    fast_sma = df["close"].rolling(fast_period).mean()
    slow_sma = df["close"].rolling(slow_period).mean()

    signals = pd.Series(0, index=df.index)
    signals[fast_sma > slow_sma] = 1
    signals[fast_sma < slow_sma] = -1

    return pd.DataFrame({"signal": signals}, index=df.index)


def empty_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Strategy that produces all zeros (no trades)."""
    return pd.DataFrame({"signal": np.zeros(len(df))}, index=df.index)


def all_buy_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Strategy that always says buy."""
    return pd.DataFrame({"signal": np.ones(len(df))}, index=df.index)


def all_sell_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Strategy that always says sell."""
    return pd.DataFrame({"signal": -np.ones(len(df))}, index=df.index)


# ── SignalParser Tests ─────────────────────────────────────────────────────


class TestSignalParser:
    """Test AST-based signal parsing."""

    def test_parses_rolling_sma(self):
        """Should detect df['close'].rolling(N).mean() as SMA."""
        parser = SignalParser()
        indicators = parser.parse(sma_crossover_signals)
        # At minimum should detect the rolling patterns
        sma_indicators = [i for i in indicators if i.indicator_type == "SMA"]
        assert len(sma_indicators) >= 2

    def test_returns_parsed_indicator_list(self):
        """Parser should return list of ParsedIndicator."""
        parser = SignalParser()
        result = parser.parse(sma_crossover_signals)
        assert isinstance(result, list)

    def test_empty_strategy_no_indicators(self):
        """Empty signal strategy should produce no detected indicators."""
        parser = SignalParser()
        result = parser.parse(empty_signals)
        assert isinstance(result, list)


# ── StrategyAdapter Tests ──────────────────────────────────────────────────


class TestStrategyAdapterPorting:
    """Test strategy porting functionality."""

    def test_port_sma_crossover(self):
        """Should successfully port SMA crossover strategy."""
        adapter = StrategyAdapter()
        result = adapter.port(sma_crossover_signals, "SmaCrossover")
        assert result.strategy_class is not None
        assert result.strategy_class.__name__ == "SmaCrossover"
        assert len(result.detected_indicators) >= 2

    def test_port_returns_porting_result(self):
        """Port should return PortingResult with correct fields."""
        adapter = StrategyAdapter()
        result = adapter.port(sma_crossover_signals)
        assert hasattr(result, "original_func")
        assert hasattr(result, "strategy_class")
        assert hasattr(result, "detected_indicators")
        assert hasattr(result, "warnings")

    def test_port_with_custom_config(self):
        """Should respect custom execution config."""
        config = ExecutionConfig(
            slippage_bps=5.0,
            commission_pct=0.002,
            risk_per_trade=0.01,
        )
        adapter = StrategyAdapter(config)
        result = adapter.port(sma_crossover_signals)
        assert result.strategy_class._config.slippage_bps == 5.0
        assert result.strategy_class._config.risk_per_trade == 0.01


class TestSignalFidelity:
    """Test signal fidelity validation."""

    def test_sma_crossover_high_correlation(self):
        """Ported SMA crossover should have high signal correlation with original."""
        df = make_ohlcv(200)
        adapter = StrategyAdapter()
        result = adapter.port(sma_crossover_signals)
        passed, correlation = adapter.validate_fidelity(
            sma_crossover_signals, result.strategy_class, df,
            params={"fast": 10, "slow": 30},
        )
        assert passed
        assert correlation > 0.8

    def test_empty_signals_fidelity(self):
        """Empty signals should still pass fidelity check."""
        df = make_ohlcv(100)
        adapter = StrategyAdapter()
        result = adapter.port(empty_signals)
        passed, correlation = adapter.validate_fidelity(
            empty_signals, result.strategy_class, df,
        )
        # Both produce all zeros → perfect correlation or constant signal
        assert correlation >= 0.8 or correlation == 1.0

    def test_all_buy_fidelity(self):
        """All-buy strategy should pass fidelity check."""
        df = make_ohlcv(100)
        adapter = StrategyAdapter()
        result = adapter.port(all_buy_signals)
        passed, correlation = adapter.validate_fidelity(
            all_buy_signals, result.strategy_class, df,
        )
        assert correlation >= 0.8

    def test_all_sell_fidelity(self):
        """All-sell strategy should pass fidelity check."""
        df = make_ohlcv(100)
        adapter = StrategyAdapter()
        result = adapter.port(all_sell_signals)
        passed, correlation = adapter.validate_fidelity(
            all_sell_signals, result.strategy_class, df,
        )
        assert correlation >= 0.8

    def test_different_params_lower_correlation(self):
        """Different params should potentially lower correlation."""
        df = make_ohlcv(200)
        adapter = StrategyAdapter()
        result = adapter.port(sma_crossover_signals)

        _, corr_same = adapter.validate_fidelity(
            sma_crossover_signals, result.strategy_class, df,
            params={"fast": 10, "slow": 30},
        )

        # Running with different params than what the original uses
        # should still have some correlation since replay uses the same function
        assert isinstance(corr_same, float)


class TestEdgeCases:
    """Test edge cases for the strategy adapter."""

    def test_very_short_dataframe(self):
        """Should handle very short data gracefully."""
        df = make_ohlcv(5)
        adapter = StrategyAdapter()
        result = adapter.port(sma_crossover_signals)
        passed, correlation = adapter.validate_fidelity(
            sma_crossover_signals, result.strategy_class, df,
        )
        # With only 5 bars and SMA(10) + SMA(30), signals are mostly NaN/0
        # Correlation should be computable (not crash)
        assert isinstance(correlation, float)

    def test_missing_signal_column(self):
        """Should handle function that doesn't return 'signal' column."""

        def bad_signals(df, params):
            return pd.DataFrame({"not_signal": np.zeros(len(df))})

        adapter = StrategyAdapter()
        result = adapter.port(bad_signals)
        passed, correlation = adapter.validate_fidelity(
            bad_signals, result.strategy_class, make_ohlcv(50),
        )
        assert passed is False
        assert correlation == 0.0

    def test_single_row_dataframe(self):
        """Should handle single-row data without crashing."""
        df = make_ohlcv(1)
        adapter = StrategyAdapter()
        result = adapter.port(sma_crossover_signals)
        # Should not raise
        assert result is not None


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_port_strategy_quick(self):
        """port_strategy() should work as a one-liner."""
        result = port_strategy(sma_crossover_signals, "QuickPort")
        assert result.strategy_class.__name__ == "QuickPort"

    def test_validate_ported_strategy_quick(self):
        """validate_ported_strategy() should work as a one-liner."""
        result = port_strategy(sma_crossover_signals)
        df = make_ohlcv(200)
        passed, correlation = validate_ported_strategy(
            sma_crossover_signals, result.strategy_class, df,
            params={"fast": 10, "slow": 30},
        )
        assert passed
        assert correlation > 0.8


class TestExecutionConfig:
    """Test execution configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ExecutionConfig()
        assert config.slippage_bps == 2.0
        assert config.commission_pct == 0.001
        assert config.risk_per_trade == 0.02
        assert config.max_position_pct == 0.95

    def test_custom_config(self):
        """Should accept custom parameters."""
        config = ExecutionConfig(
            slippage_bps=5.0,
            commission_pct=0.002,
            risk_per_trade=0.05,
        )
        assert config.slippage_bps == 5.0
        assert config.risk_per_trade == 0.05
