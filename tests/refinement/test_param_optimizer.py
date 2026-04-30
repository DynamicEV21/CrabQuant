"""Tests for crabquant.refinement.param_optimizer module."""

import itertools
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from crabquant.refinement.param_optimizer import (
    OptimizationResult,
    _generate_param_variants,
    _run_param_backtest,
    format_optimization_for_context,
    format_optimization_for_prompt,
    optimize_parameters,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a realistic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 252
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


@pytest.fixture
def simple_strategy_fn():
    """Strategy that uses 'period' param to compute SMA crossover."""
    def generate_signals(df, params):
        period = params.get("period", 20)
        sma = df["close"].rolling(window=int(period)).mean()
        sma_prev = sma.shift(1)
        entries = (df["close"] > sma) & (df["close"].shift(1) <= sma_prev)
        exits = (df["close"] < sma) & (df["close"].shift(1) >= sma_prev)
        return entries.fillna(False), exits.fillna(False)
    return generate_signals


@pytest.fixture
def multi_param_strategy_fn():
    """Strategy with multiple params."""
    def generate_signals(df, params):
        fast = int(params.get("fast_period", 10))
        slow = int(params.get("slow_period", 30))
        threshold = params.get("threshold", 0.02)

        fast_ma = df["close"].rolling(window=fast).mean()
        slow_ma = df["close"].rolling(window=slow).mean()
        spread = (fast_ma - slow_ma) / slow_ma

        entries = (spread > threshold) & (spread.shift(1) <= threshold)
        exits = (spread < -threshold) & (spread.shift(1) >= -threshold)
        return entries.fillna(False), exits.fillna(False)
    return generate_signals


# ── _generate_param_variants ──────────────────────────────────────────────

class TestGenerateParamVariants:

    def test_empty_params(self):
        variants = _generate_param_variants({})
        assert variants == [{}]

    def test_single_numeric_param(self):
        base = {"period": 20}
        variants = _generate_param_variants(base, sweep_factor=0.5)
        # Should have 3 variants: [10, 20, 30]
        assert len(variants) == 3
        periods = sorted(set(v["period"] for v in variants))
        assert periods[0] == 10
        assert periods[1] == 20
        assert periods[2] == 30

    def test_preserves_int_type(self):
        base = {"period": 20, "threshold": 0.5}
        variants = _generate_param_variants(base)
        for v in variants:
            assert isinstance(v["period"], int)

    def test_float_param_preserved(self):
        base = {"threshold": 0.5}
        variants = _generate_param_variants(base)
        for v in variants:
            assert isinstance(v["threshold"], float)

    def test_non_numeric_params_unchanged(self):
        base = {"period": 20, "use_short": True, "name": "test"}
        variants = _generate_param_variants(base)
        for v in variants:
            assert v["use_short"] is True
            assert v["name"] == "test"

    def test_bool_not_treated_as_numeric(self):
        base = {"enabled": True, "period": 20}
        variants = _generate_param_variants(base)
        # 'enabled' is bool, should not be swept
        enabled_values = set(v["enabled"] for v in variants)
        assert enabled_values == {True}

    def test_max_combinations_cap(self):
        # With 3 params × 3 values = 27 combos, cap at 20
        base = {"a": 10, "b": 20, "c": 30}
        variants = _generate_param_variants(base, max_combinations=20)
        assert len(variants) <= 20

    def test_max_combinations_single_param(self):
        base = {"period": 20}
        variants = _generate_param_variants(base, max_combinations=2)
        assert len(variants) <= 2

    def test_original_always_included(self):
        base = {"period": 20, "threshold": 0.5}
        variants = _generate_param_variants(base)
        found_original = False
        for v in variants:
            if v["period"] == 20 and v["threshold"] == 0.5:
                found_original = True
                break
        assert found_original

    def test_sweep_factor_zero(self):
        base = {"period": 20}
        variants = _generate_param_variants(base, sweep_factor=0.0)
        # All variants should be the original
        assert all(v["period"] == 20 for v in variants)

    def test_large_sweep_factor(self):
        base = {"period": 20}
        variants = _generate_param_variants(base, sweep_factor=0.9)
        periods = sorted(set(v["period"] for v in variants))
        assert periods[0] >= 1  # Clamped to positive
        assert periods[-1] >= 30

    def test_positive_clamp_for_period_params(self):
        base = {"period": 2}
        variants = _generate_param_variants(base, sweep_factor=0.8)
        for v in variants:
            assert v["period"] >= 1

    def test_negative_base_value(self):
        base = {"offset": -5}
        variants = _generate_param_variants(base, sweep_factor=0.5)
        offsets = sorted(set(v["offset"] for v in variants))
        assert len(offsets) == 3
        assert offsets[1] == -5  # Original preserved

    def test_zero_base_value(self):
        base = {"offset": 0}
        variants = _generate_param_variants(base, sweep_factor=0.5)
        # 0 * (1 - 0.5) = 0, 0 * (1 + 0.5) = 0 → all same
        assert all(v["offset"] == 0 for v in variants)


# ── _run_param_backtest ───────────────────────────────────────────────────

class TestRunParamBacktest:

    def test_basic_backtest(self, sample_df, simple_strategy_fn):
        result = _run_param_backtest(sample_df, simple_strategy_fn, {"period": 20})
        assert result is not None
        assert "sharpe" in result
        assert "num_trades" in result
        assert "params" in result

    def test_none_entries_returns_none(self, sample_df):
        def bad_fn(df, params):
            return None, None
        result = _run_param_backtest(sample_df, bad_fn, {"period": 20})
        assert result is None

    def test_all_false_signals_returns_none(self, sample_df):
        def no_signals(df, params):
            entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)
            return entries, exits
        result = _run_param_backtest(sample_df, no_signals, {"period": 20})
        assert result is None

    def test_crashing_strategy_returns_none(self, sample_df):
        def crash_fn(df, params):
            raise ValueError("test crash")
        result = _run_param_backtest(sample_df, crash_fn, {"period": 20})
        assert result is None

    def test_params_passed_through(self, sample_df):
        captured_params = {}
        def capture_fn(df, params):
            captured_params.update(params)
            entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)
            return entries, exits
        _run_param_backtest(sample_df, capture_fn, {"period": 15, "custom": 42})
        assert captured_params.get("period") == 15
        assert captured_params.get("custom") == 42


# ── OptimizationResult ────────────────────────────────────────────────────

class TestOptimizationResult:

    def test_default_values(self):
        result = OptimizationResult()
        assert result.default_sharpe == 0.0
        assert result.optimized_sharpe == 0.0
        assert result.was_optimized is False
        assert result.combinations_tested == 0

    def test_summary_not_optimized(self):
        result = OptimizationResult(
            default_sharpe=0.5,
            combinations_tested=10,
            sweep_time_seconds=1.5,
        )
        summary = result.summary()
        assert "no improvement" in summary
        assert "0.500" in summary

    def test_summary_optimized(self):
        result = OptimizationResult(
            default_sharpe=0.5,
            optimized_sharpe=1.0,
            default_trades=10,
            optimized_trades=15,
            combinations_tested=20,
            improvement_pct=100.0,
            was_optimized=True,
            sweep_time_seconds=2.0,
        )
        summary = result.summary()
        assert "IMPROVED" in summary
        assert "0.500 → 1.000" in summary
        assert "+100.0%" in summary


# ── optimize_parameters ───────────────────────────────────────────────────

class TestOptimizeParameters:

    def test_basic_optimization(self, sample_df, simple_strategy_fn):
        base_params = {"period": 20}
        result = optimize_parameters(
            sample_df, simple_strategy_fn, base_params,
            max_combinations=5,
        )
        assert result.default_sharpe is not None
        assert result.combinations_tested >= 1
        assert result.sweep_time_seconds >= 0
        assert isinstance(result.was_optimized, bool)

    def test_returns_default_when_no_improvement(self, sample_df):
        """When all variants perform the same, should not be 'optimized'."""
        def constant_fn(df, params):
            # Always returns the same signals regardless of params
            entries = pd.Series([True, False] * (len(df) // 2), index=df.index[:len([True, False] * (len(df) // 2))])
            exits = pd.Series([False, True] * (len(df) // 2), index=df.index[:len([False, True] * (len(df) // 2))])
            return entries, exits

        result = optimize_parameters(
            sample_df, constant_fn, {"period": 20},
            max_combinations=3,
        )
        assert not result.was_optimized

    def test_min_improvement_threshold(self, sample_df, simple_strategy_fn):
        """Should not report optimization if improvement < min_improvement."""
        result = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            min_improvement=100.0,  # Impossible threshold
            max_combinations=3,
        )
        assert not result.was_optimized

    def test_min_trades_filter(self, sample_df):
        """Should not report optimization if best has < min_trades."""
        def rare_signals(df, params):
            # Only fires once regardless of params
            entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)
            entries.iloc[50] = True
            exits.iloc[55] = True
            return entries, exits

        result = optimize_parameters(
            sample_df, rare_signals, {"period": 20},
            min_trades=5,
            max_combinations=3,
        )
        assert not result.was_optimized

    def test_multi_param_sweep(self, sample_df):
        """Test with multiple numeric parameters."""
        def multi_fn(df, params):
            fast = int(params.get("fast", 10))
            slow = int(params.get("slow", 30))
            fast_ma = df["close"].rolling(window=max(1, fast)).mean()
            slow_ma = df["close"].rolling(window=max(1, slow)).mean()
            entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
            return entries.fillna(False), exits.fillna(False)

        base_params = {"fast": 10, "slow": 30}
        result = optimize_parameters(
            sample_df, multi_fn, base_params,
            max_combinations=9,
        )
        assert result.combinations_tested >= 1
        assert result.default_params == base_params

    def test_empty_params(self, sample_df, simple_strategy_fn):
        """Empty params dict should still work (no sweep)."""
        result = optimize_parameters(
            sample_df, simple_strategy_fn, {},
            max_combinations=5,
        )
        assert result.default_params == {}
        assert result.optimized_params == {}

    def test_crashing_strategy_returns_defaults(self, sample_df):
        """Strategy that crashes should return default result with zeros."""
        def crash_fn(df, params):
            raise ValueError("crash")

        result = optimize_parameters(
            sample_df, crash_fn, {"period": 20},
        )
        assert result.default_sharpe == 0.0
        assert not result.was_optimized

    def test_sweep_factor_affects_variants(self, sample_df, simple_strategy_fn):
        """Different sweep factors should produce different variant counts."""
        r1 = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            sweep_factor=0.0,  # No sweep
        )
        r2 = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            sweep_factor=0.5,
        )
        # With factor=0, only 1 variant (original)
        assert r1.combinations_tested <= r2.combinations_tested


# ── format_optimization_for_prompt ────────────────────────────────────────

class TestFormatOptimizationForPrompt:

    def test_not_optimized_returns_empty(self):
        result = OptimizationResult(was_optimized=False)
        assert format_optimization_for_prompt(result) == ""

    def test_optimized_includes_sharpe_change(self):
        result = OptimizationResult(
            default_sharpe=0.5,
            optimized_sharpe=1.2,
            default_trades=8,
            optimized_trades=12,
            default_params={"period": 20},
            optimized_params={"period": 15},
            was_optimized=True,
            combinations_tested=9,
            improvement_pct=140.0,
            sweep_time_seconds=1.5,
        )
        text = format_optimization_for_prompt(result)
        assert "Parameter Optimization Applied" in text
        assert "0.500" in text
        assert "1.200" in text
        assert "period" in text
        assert "20 → 15" in text
        assert "STRUCTURAL changes" in text

    def test_multiple_param_changes(self):
        result = OptimizationResult(
            default_sharpe=0.3,
            optimized_sharpe=0.8,
            default_params={"fast": 10, "slow": 30, "threshold": 0.02},
            optimized_params={"fast": 5, "slow": 50, "threshold": 0.01},
            was_optimized=True,
            combinations_tested=27,
            improvement_pct=166.7,
            sweep_time_seconds=3.0,
        )
        text = format_optimization_for_prompt(result)
        # Check all param changes are listed
        assert "fast" in text
        assert "slow" in text
        assert "threshold" in text

    def test_unchanged_params_not_listed(self):
        result = OptimizationResult(
            default_sharpe=0.5,
            optimized_sharpe=1.0,
            default_params={"period": 20, "unchanged": True},
            optimized_params={"period": 15, "unchanged": True},
            was_optimized=True,
        )
        text = format_optimization_for_prompt(result)
        assert "period" in text
        assert "unchanged" not in text


# ── format_optimization_for_context ───────────────────────────────────────

class TestFormatOptimizationForContext:

    def test_basic_fields(self):
        result = OptimizationResult(
            default_sharpe=0.5,
            optimized_sharpe=1.0,
            was_optimized=True,
            combinations_tested=9,
            improvement_pct=100.0,
            sweep_time_seconds=2.0,
        )
        ctx = format_optimization_for_context(result)
        assert ctx["param_optimization_applied"] is True
        assert ctx["default_sharpe"] == 0.5
        assert ctx["optimized_sharpe"] == 1.0
        assert ctx["improvement_pct"] == 100.0

    def test_not_optimized(self):
        result = OptimizationResult(was_optimized=False)
        ctx = format_optimization_for_context(result)
        assert ctx["param_optimization_applied"] is False

    def test_all_keys_present(self):
        result = OptimizationResult(was_optimized=True)
        ctx = format_optimization_for_context(result)
        expected_keys = {
            "param_optimization_applied",
            "default_sharpe",
            "optimized_sharpe",
            "improvement_pct",
            "combinations_tested",
            "sweep_time_seconds",
        }
        assert expected_keys == set(ctx.keys())


# ── Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_very_short_df(self, simple_strategy_fn):
        """Very short DataFrame should not crash."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="B"))

        result = optimize_parameters(
            df, simple_strategy_fn, {"period": 2},
            max_combinations=3,
        )
        assert isinstance(result, OptimizationResult)

    def test_single_param_with_list_value(self):
        """Params with list values should not crash."""
        variants = _generate_param_variants({"periods": [10, 20, 30]})
        # Lists are not numeric, should be kept as-is
        assert all(v["periods"] == [10, 20, 30] for v in variants)

    def test_string_param_not_swept(self):
        variants = _generate_param_variants({"mode": "fast", "period": 20})
        modes = set(v["mode"] for v in variants)
        assert modes == {"fast"}

    def test_optimization_with_zero_default_sharpe(self, sample_df, simple_strategy_fn):
        """When default Sharpe is 0, improvement_pct should handle division."""
        result = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 200},  # Very long period, likely ~0 Sharpe
            max_combinations=3,
        )
        # Should not crash due to division by zero
        assert isinstance(result, OptimizationResult)
        if result.default_sharpe == 0 and result.was_optimized:
            assert result.improvement_pct >= 0

    # ── Sharpe target (gap rescue) tests ──────────────────────────

    def test_sharpe_target_none_does_not_affect_was_optimized(self, sample_df, simple_strategy_fn):
        """When sharpe_target is None (default), behavior is unchanged."""
        result_default = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            max_combinations=5, min_improvement=0.1,
        )
        result_none = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            max_combinations=5, min_improvement=0.1, sharpe_target=None,
        )
        # Both should have the same was_optimized status
        assert result_default.was_optimized == result_none.was_optimized

    def test_sharpe_target_marks_optimized_when_target_reached(self, sample_df, simple_strategy_fn):
        """When best Sharpe >= sharpe_target, was_optimized should be True
        even if improvement is below min_improvement."""
        # First, find what Sharpe the strategy achieves
        probe = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            max_combinations=5, min_improvement=0.0,
        )
        best_sharpe = max(probe.default_sharpe, probe.optimized_sharpe)

        # Set target slightly below the best achievable Sharpe
        # so that some variant will reach it
        target = best_sharpe - 0.01
        if target <= 0:
            return  # Skip if strategy has no positive Sharpe

        result = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            max_combinations=5,
            min_improvement=999.0,  # Impossible to reach via improvement
            sharpe_target=target,
        )
        # Even though min_improvement is impossible, reaching the target
        # should make was_optimized True
        if result.optimized_sharpe >= target and result.optimized_trades >= 5:
            assert result.was_optimized, (
                f"Should be optimized when target reached: "
                f"optimized_sharpe={result.optimized_sharpe:.3f} >= "
                f"target={target:.3f}"
            )

    def test_sharpe_target_too_high_does_not_force_optimization(self, sample_df, simple_strategy_fn):
        """When sharpe_target is impossibly high, was_optimized should
        follow the normal min_improvement logic."""
        result = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            max_combinations=5,
            min_improvement=0.1,
            sharpe_target=999.0,  # Impossible to reach
        )
        # Should behave exactly as without sharpe_target
        result_baseline = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            max_combinations=5,
            min_improvement=0.1,
        )
        assert result.was_optimized == result_baseline.was_optimized

    def test_sharpe_target_with_insufficient_trades(self, sample_df, simple_strategy_fn):
        """Even when target is reached, was_optimized should be False
        if trades are below min_trades."""
        result = optimize_parameters(
            sample_df, simple_strategy_fn, {"period": 20},
            max_combinations=5,
            min_improvement=0.0,
            min_trades=99999,  # Impossible trade count
            sharpe_target=0.0,  # Always reachable
        )
        # Should NOT be optimized because trade count is too low
        assert not result.was_optimized
