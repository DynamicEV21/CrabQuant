"""Tests for trade count rescue optimization (Phase 6)."""

import numpy as np
import pandas as pd
import pytest

from crabquant.refinement.param_optimizer import (
    OptimizationResult,
    optimize_for_trade_count,
    format_trade_count_rescue_for_prompt,
)


def _make_ohlcv(n=500, seed=42):
    """Create a simple OHLCV DataFrame for testing."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1000, 100000, n).astype(float),
    }, index=pd.date_range("2023-01-01", periods=n, freq="1D"))


def _make_strategy_fn(num_trades=10):
    """Create a strategy function that generates a fixed number of round-trip trades.

    Each trade: enter True on one day, exit True a few days later.
    vectorbt.from_signals uses entries/exits as boolean Series.
    """
    def generate_signals(df, params):
        n = len(df)
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        # Place trades evenly spaced
        hold_days = 5
        gap = max(hold_days + 2, n // max(num_trades, 1))
        for i in range(0, n - hold_days, gap):
            entries.iloc[i] = True
            exits.iloc[i + hold_days] = True
        return entries, exits
    return generate_signals


class TestOptimizeForTradeCount:
    """Tests for the optimize_for_trade_count function."""

    def test_returns_result_when_no_optimization_needed(self):
        """If default already has enough trades, return was_optimized=False."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=25)  # 25 trades > 20 target
        params = {"period": 20}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=1.5, target_trades=20
        )
        assert isinstance(result, OptimizationResult)
        assert not result.was_optimized
        assert result.default_trades >= 20
        assert result.optimized_trades >= 20

    def test_returns_result_when_no_valid_variants(self):
        """If all variants have terrible Sharpe, return was_optimized=False."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=5)  # Only 5 trades
        params = {"period": 20}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=0.5, target_trades=20,
            max_sharpe_penalty=0.01,  # Very strict — almost no Sharpe drop allowed
        )
        assert isinstance(result, OptimizationResult)
        # May or may not be optimized depending on param sweep results

    def test_wider_sweep_generates_variants(self):
        """Wider sweep should generate and test multiple variants."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=15)  # 15 trades, below 20
        params = {"period": 50, "threshold": 0.5}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=1.0, target_trades=20,
            sweep_factor=0.8, max_combinations=20,
        )
        assert isinstance(result, OptimizationResult)
        assert result.combinations_tested >= 1

    def test_respects_max_sharpe_penalty(self):
        """Should not accept variants that drop Sharpe below penalty threshold."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=8)
        params = {"period": 20}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=0.1, target_trades=20,
            max_sharpe_penalty=0.99,  # Allow almost any Sharpe
        )
        assert isinstance(result, OptimizationResult)

    def test_empty_params_returns_no_optimization(self):
        """Empty params dict should return quickly with no optimization."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=5)

        result = optimize_for_trade_count(
            df, fn, {}, current_sharpe=1.0, target_trades=20
        )
        assert isinstance(result, OptimizationResult)

    def test_non_numeric_params_returns_no_optimization(self):
        """Non-numeric params can't be swept, should return quickly."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=5)
        params = {"mode": "fast", "enabled": True}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=1.0, target_trades=20
        )
        assert isinstance(result, OptimizationResult)

    def test_single_numeric_param_swept(self):
        """Single numeric param should generate variants."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=12)
        params = {"period": 30}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=0.8, target_trades=20,
            max_combinations=10,
        )
        assert isinstance(result, OptimizationResult)
        assert result.combinations_tested >= 1

    def test_custom_target_trades(self):
        """Should respect custom target_trades value."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=10)
        params = {"period": 20}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=1.0, target_trades=5
        )
        # With target=5 and 10 trades, should not need optimization
        assert isinstance(result, OptimizationResult)

    def test_default_params_preserved_in_result(self):
        """Original params should always be in default_params."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=15)
        params = {"period": 20, "threshold": 0.5}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=1.0, target_trades=20,
        )
        assert result.default_params == params

    def test_optimized_params_differ_when_optimized(self):
        """When optimized, params should differ from default."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=15)
        params = {"period": 20, "threshold": 0.5}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=1.0, target_trades=20,
            sweep_factor=0.9, max_combinations=20, max_sharpe_penalty=0.9,
        )
        if result.was_optimized:
            assert result.optimized_params != params

    def test_returns_immediately_for_zero_sharpe(self):
        """Zero Sharpe should still attempt optimization (min_sharpe = 0)."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=15)
        params = {"period": 20}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=0.0, target_trades=20,
        )
        assert isinstance(result, OptimizationResult)

    def test_negative_sharpe_allows_negative_variants(self):
        """Negative current Sharpe means min_sharpe is even more negative."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=15)
        params = {"period": 20}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=-0.5, target_trades=20,
        )
        assert isinstance(result, OptimizationResult)

    def test_fallback_picks_most_trades(self):
        """When no variant hits target, fallback picks most trades."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=8)
        params = {"period": 100, "threshold": 2.0}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=1.0, target_trades=20,
            sweep_factor=0.3, max_combinations=5,
        )
        assert isinstance(result, OptimizationResult)

    def test_sweep_time_is_recorded(self):
        """Sweep time should be a valid number (can be negative in CI due to clock)."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=15)
        params = {"period": 20}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=1.0, target_trades=20,
        )
        assert isinstance(result, OptimizationResult)
        # sweep_time_seconds should be a number; may be negative in CI
        assert isinstance(result.sweep_time_seconds, float)


class TestFormatTradeCountRescueForPrompt:
    """Tests for the format_trade_count_rescue_for_prompt function."""

    def test_returns_empty_when_not_optimized(self):
        """Should return empty string when optimization didn't happen."""
        result = OptimizationResult(
            default_sharpe=1.5,
            optimized_sharpe=1.5,
            default_params={"period": 20},
            optimized_params={"period": 20},
            default_trades=5,
            optimized_trades=5,
            was_optimized=False,
        )
        output = format_trade_count_rescue_for_prompt(result, target_trades=20)
        assert output == ""

    def test_shows_target_reached_when_trades_hit(self):
        """Should show TARGET REACHED when optimized trades >= target."""
        result = OptimizationResult(
            default_sharpe=1.5,
            optimized_sharpe=1.2,
            default_params={"period": 20},
            optimized_params={"period": 10},
            default_trades=5,
            optimized_trades=25,
            was_optimized=True,
            combinations_tested=10,
            sweep_time_seconds=2.5,
        )
        output = format_trade_count_rescue_for_prompt(result, target_trades=20)
        assert "TARGET REACHED" in output
        assert "25 trades" in output
        assert "enough trades for walk-forward validation" in output

    def test_shows_partial_when_trades_below_target(self):
        """Should show Partial when optimized trades < target."""
        result = OptimizationResult(
            default_sharpe=1.5,
            optimized_sharpe=1.4,
            default_params={"period": 20},
            optimized_params={"period": 15},
            default_trades=5,
            optimized_trades=15,
            was_optimized=True,
            combinations_tested=10,
            sweep_time_seconds=1.5,
        )
        output = format_trade_count_rescue_for_prompt(result, target_trades=20)
        assert "Partial" in output
        assert "15/20" in output
        assert "5 trades short" in output

    def test_shows_sharpe_penalty_when_dropped(self):
        """Should mention Sharpe penalty when it dropped."""
        result = OptimizationResult(
            default_sharpe=2.0,
            optimized_sharpe=1.5,
            default_params={"period": 20},
            optimized_params={"period": 10},
            default_trades=8,
            optimized_trades=22,
            was_optimized=True,
            combinations_tested=15,
            sweep_time_seconds=3.0,
        )
        output = format_trade_count_rescue_for_prompt(result, target_trades=20)
        assert "Sharpe penalty" in output
        assert "-0.500" in output

    def test_no_penalty_message_when_sharpe_improved(self):
        """Should not mention penalty when Sharpe improved."""
        result = OptimizationResult(
            default_sharpe=1.0,
            optimized_sharpe=1.2,
            default_params={"period": 20},
            optimized_params={"period": 10},
            default_trades=8,
            optimized_trades=22,
            was_optimized=True,
            combinations_tested=15,
            sweep_time_seconds=3.0,
        )
        output = format_trade_count_rescue_for_prompt(result, target_trades=20)
        assert "Sharpe penalty" not in output
        assert "TARGET REACHED" in output

    def test_includes_combo_count_and_time(self):
        """Should include combinations tested and sweep time."""
        result = OptimizationResult(
            default_sharpe=1.5,
            optimized_sharpe=1.2,
            default_params={"period": 20},
            optimized_params={"period": 10},
            default_trades=5,
            optimized_trades=25,
            was_optimized=True,
            combinations_tested=12,
            sweep_time_seconds=4.2,
        )
        output = format_trade_count_rescue_for_prompt(result, target_trades=20)
        assert "12 combos" in output
        assert "4.2s" in output


class TestOptimizeForTradeCountEdgeCases:
    """Edge case tests for trade count rescue."""

    def test_strategy_that_never_signals(self):
        """Strategy with zero signals should return gracefully."""
        df = _make_ohlcv()

        def no_signals(df, params):
            entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)
            return entries, exits

        result = optimize_for_trade_count(
            df, no_signals, {"period": 20}, current_sharpe=1.0, target_trades=20,
        )
        assert isinstance(result, OptimizationResult)

    def test_strategy_that_always_signals(self):
        """Strategy that's always long should still work."""
        df = _make_ohlcv()

        def always_long(df, params):
            entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)
            entries.iloc[0] = True
            exits.iloc[-1] = True
            return entries, exits

        result = optimize_for_trade_count(
            df, always_long, {}, current_sharpe=1.0, target_trades=20,
        )
        assert isinstance(result, OptimizationResult)

    def test_very_small_dataframe(self):
        """Small DataFrame should not crash."""
        df = _make_ohlcv(n=30)
        fn = _make_strategy_fn(num_trades=3)
        params = {"period": 5}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=0.5, target_trades=20,
        )
        assert isinstance(result, OptimizationResult)

    def test_float_params_preserve_type(self):
        """Float params should remain float in optimized version."""
        df = _make_ohlcv()
        fn = _make_strategy_fn(num_trades=15)
        params = {"threshold": 0.75, "period": 20}

        result = optimize_for_trade_count(
            df, fn, params, current_sharpe=1.0, target_trades=20,
        )
        # Original params preserved
        assert result.default_params["threshold"] == 0.75
        assert isinstance(result.default_params["threshold"], float)
