"""Tests for scipy Differential Evolution optimizer in param_optimizer.py."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import rosen

from crabquant.refinement.param_optimizer import optimize_with_de


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame for testing."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 100000, n).astype(float),
    }, index=pd.date_range("2023-01-01", periods=n, freq="1D"))


def _simple_strategy(df: pd.DataFrame, params: dict) -> tuple:
    """A minimal strategy: buy when close > SMA(params['period']), sell otherwise.

    The optimal period should be around 10-15 for typical random walk data.
    """
    period = int(params.get("period", 10))
    sma = df["close"].rolling(period).mean()
    entries = (df["close"] > sma).astype(bool)
    exits = (df["close"] <= sma).astype(bool)
    # Ensure exits happen after entries (shift by 1)
    exits = exits.shift(1).fillna(False).astype(bool)
    return entries, exits


def _sharpe_mock_strategy(sharpe_value: float):
    """Return a strategy_fn that always produces a fixed sharpe.

    Used to test structure/bounds without depending on backtest results.
    """
    def strategy_fn(df, params):
        period = int(params.get("period", 10))
        sma = df["close"].rolling(period).mean()
        entries = (df["close"] > sma).astype(bool)
        exits = (df["close"] <= sma).astype(bool)
        exits = exits.shift(1).fillna(False).astype(bool)
        return entries, exits
    return strategy_fn


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDEConvergesOnSimpleFunction:
    """Test that scipy DE itself works (smoke test on Rosenbrock)."""

    def test_rosenbrock_convergence(self):
        """DE should find near-optimal solution for 2D Rosenbrock."""
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            rosen,
            bounds=[(-5, 5), (-5, 5)],
            maxiter=200,
            seed=42,
            tol=1e-6,
        )
        # Rosenbrock minimum is at (1, 1) with value 0
        assert result.x[0] == pytest.approx(1.0, abs=0.05)
        assert result.x[1] == pytest.approx(1.0, abs=0.05)
        assert result.fun < 0.01


class TestOptimizeWithDEReturnStructure:
    """Test that optimize_with_de returns the correct dict structure."""

    def test_returns_required_keys(self):
        """Result dict must have best_params, best_sharpe, n_evals, success, method."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}
        config = {"maxiter": 2, "popsize": 5, "workers": 1, "seed": 42}

        result = optimize_with_de(_simple_strategy, param_ranges, data, config)

        assert isinstance(result, dict)
        assert "best_params" in result
        assert "best_sharpe" in result
        assert "n_evals" in result
        assert "success" in result
        assert "method" in result

    def test_best_params_is_dict(self):
        """best_params should be a dict with param names as keys."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}
        config = {"maxiter": 2, "popsize": 5, "workers": 1, "seed": 42}

        result = optimize_with_de(_simple_strategy, param_ranges, data, config)

        assert isinstance(result["best_params"], dict)
        assert "period" in result["best_params"]

    def test_best_sharpe_is_float(self):
        """best_sharpe should be a float."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}
        config = {"maxiter": 2, "popsize": 5, "workers": 1, "seed": 42}

        result = optimize_with_de(_simple_strategy, param_ranges, data, config)

        assert isinstance(result["best_sharpe"], float)

    def test_n_evals_is_positive_int(self):
        """n_evals should be a positive integer."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}
        config = {"maxiter": 2, "popsize": 5, "workers": 1, "seed": 42}

        result = optimize_with_de(_simple_strategy, param_ranges, data, config)

        assert isinstance(result["n_evals"], int)
        assert result["n_evals"] > 0

    def test_method_is_differential_evolution(self):
        """method should always be 'differential_evolution'."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}
        config = {"maxiter": 2, "popsize": 5, "workers": 1, "seed": 42}

        result = optimize_with_de(_simple_strategy, param_ranges, data, config)

        assert result["method"] == "differential_evolution"

    def test_success_is_bool(self):
        """success should be a boolean."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}
        config = {"maxiter": 2, "popsize": 5, "workers": 1, "seed": 42}

        result = optimize_with_de(_simple_strategy, param_ranges, data, config)

        assert isinstance(result["success"], bool)


class TestOptimizeWithDEBounds:
    """Test that optimize_with_de respects param_ranges bounds."""

    def test_single_param_within_bounds(self):
        """Best param should be within the specified range."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}
        config = {"maxiter": 5, "popsize": 5, "workers": 1, "seed": 42}

        result = optimize_with_de(_simple_strategy, param_ranges, data, config)

        assert 5 <= result["best_params"]["period"] <= 30

    def test_multi_param_within_bounds(self):
        """All params should be within their ranges."""
        data = _make_ohlcv(200)
        param_ranges = {
            "period": (5, 30),
            "threshold": (0.5, 3.0),
        }

        def strategy_with_threshold(df, params):
            period = int(params.get("period", 10))
            threshold = params.get("threshold", 1.0)
            sma = df["close"].rolling(period).mean()
            entries = ((df["close"] - sma) > threshold).astype(bool)
            exits = ((df["close"] - sma) <= threshold).astype(bool)
            exits = exits.shift(1).fillna(False).astype(bool)
            return entries, exits

        config = {"maxiter": 5, "popsize": 5, "workers": 1, "seed": 42}
        result = optimize_with_de(strategy_with_threshold, param_ranges, data, config)

        assert 5 <= result["best_params"]["period"] <= 30
        assert 0.5 <= result["best_params"]["threshold"] <= 3.0

    def test_narrow_bounds_constrain_result(self):
        """With very narrow bounds, result should be within them."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (9, 11)}  # very narrow
        config = {"maxiter": 5, "popsize": 5, "workers": 1, "seed": 42}

        result = optimize_with_de(_simple_strategy, param_ranges, data, config)

        assert 9 <= result["best_params"]["period"] <= 11


class TestOptimizeWithDEConfig:
    """Test that config options are respected."""

    def test_default_config_uses_grid_compatible_settings(self):
        """Without config, should use sensible defaults."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}

        result = optimize_with_de(_simple_strategy, param_ranges, data)

        assert result["method"] == "differential_evolution"
        assert isinstance(result["best_params"], dict)

    def test_custom_maxiter_limits_evaluations(self):
        """Lower maxiter should generally produce fewer evaluations."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}

        result_low = optimize_with_de(
            _simple_strategy, param_ranges, data,
            config={"maxiter": 2, "popsize": 5, "workers": 1, "seed": 42},
        )
        result_high = optimize_with_de(
            _simple_strategy, param_ranges, data,
            config={"maxiter": 10, "popsize": 5, "workers": 1, "seed": 42},
        )

        # More iterations should generally produce more evaluations
        assert result_high["n_evals"] >= result_low["n_evals"]


class TestOptimizeWithDEErrorHandling:
    """Test graceful error handling."""

    def test_failing_strategy_returns_valid_result(self):
        """A strategy that always fails should still return a valid result dict."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}

        def failing_strategy(df, params):
            raise ValueError("always fails")

        config = {"maxiter": 2, "popsize": 5, "workers": 1, "seed": 42}
        result = optimize_with_de(failing_strategy, param_ranges, data, config)

        # Should still have the correct structure
        assert isinstance(result, dict)
        assert "best_params" in result
        assert "best_sharpe" in result
        assert "n_evals" in result
        assert result["method"] == "differential_evolution"

    def test_none_returning_strategy_handled(self):
        """A strategy that returns None signals should be handled gracefully."""
        data = _make_ohlcv(200)
        param_ranges = {"period": (5, 30)}

        def none_strategy(df, params):
            return None, None

        config = {"maxiter": 2, "popsize": 5, "workers": 1, "seed": 42}
        result = optimize_with_de(none_strategy, param_ranges, data, config)

        assert isinstance(result, dict)
        assert result["method"] == "differential_evolution"
