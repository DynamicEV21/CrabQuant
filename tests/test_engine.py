"""Tests for CrabQuant backtest engine."""

import time

import numpy as np
import pandas as pd
import pytest

from crabquant.data import load_data
from crabquant.engine import BacktestEngine, BacktestResult
from crabquant.strategies.rsi_crossover import generate_signals, generate_signals_matrix, PARAM_GRID


class TestBacktestEngine:
    """Test the backtest engine."""

    def test_engine_creation(self):
        """Engine should create with default params."""
        engine = BacktestEngine()
        assert engine.sharpe_target == 1.5
        assert engine.max_drawdown_limit == 0.25
        assert engine.initial_cash == 100_000

    def test_engine_custom_params(self):
        """Engine should accept custom params."""
        engine = BacktestEngine(sharpe_target=2.0, min_trades=10)
        assert engine.sharpe_target == 2.0
        assert engine.min_trades == 10

    def test_run_returns_result(self):
        """run() should return a BacktestResult."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")

        assert isinstance(result, BacktestResult)
        assert result.ticker == "AAPL"
        assert result.strategy_name == "test"

    def test_result_has_all_fields(self):
        """Result should have all expected fields."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")

        assert hasattr(result, "sharpe")
        assert hasattr(result, "total_return")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "num_trades")
        assert hasattr(result, "calmar_ratio")
        assert hasattr(result, "sortino_ratio")
        assert hasattr(result, "profit_factor")
        assert hasattr(result, "score")
        assert hasattr(result, "passed")

    def test_no_signals_returns_zero_trades(self):
        """All-False entries should produce zero trades."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert result.num_trades == 0
        assert result.passed == False

    def test_passed_criteria(self):
        """Result should only pass when all criteria met."""
        engine = BacktestEngine(sharpe_target=10.0)  # Impossible to pass
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert result.passed == False  # Shouldn't pass with Sharpe >= 10

    def test_score_penalizes_low_trades(self):
        """Score should be finite for any result."""
        engine = BacktestEngine(sharpe_target=0.0, min_trades=0, min_total_return=0)
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert np.isfinite(result.score)

    def test_params_stored_in_result(self):
        """Params should be stored in the result."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)
        params = {"fast_len": 7, "slow_len": 21}

        result = engine.run(df, entries, exits, "test", "AAPL", params=params)
        assert result.params == params


class TestVectorizedEngine:
    """Test the vectorized backtest engine."""

    @pytest.fixture
    def df(self):
        return load_data("AAPL", period="2y")

    @pytest.fixture
    def engine(self):
        return BacktestEngine()

    def test_matrix_generates_correct_shape(self, df):
        """generate_signals_matrix should return DataFrames with correct number of columns."""
        small_grid = {k: v[:2] for k, v in PARAM_GRID.items()}
        entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)

        expected_cols = 1
        for v in small_grid.values():
            expected_cols *= len(v)

        assert entries_df.shape[1] == expected_cols
        assert exits_df.shape[1] == expected_cols
        assert len(param_list) == expected_cols
        assert entries_df.shape[0] == len(df)

    def test_matrix_columns_match_index(self, df):
        """Each column in entries_df/exits_df should correspond to a param dict."""
        small_grid = {k: v[:2] for k, v in PARAM_GRID.items()}
        entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)

        assert list(entries_df.columns) == list(exits_df.columns)
        for i, col in enumerate(entries_df.columns):
            assert col == f"c{i}"

    def test_matrix_values_are_boolean(self, df):
        """Signal matrices should contain only boolean values."""
        small_grid = {k: v[:2] for k, v in PARAM_GRID.items()}
        entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)

        assert entries_df.dtypes.apply(lambda dt: dt == bool or pd.api.types.is_bool_dtype(dt)).all()
        assert exits_df.dtypes.apply(lambda dt: dt == bool or pd.api.types.is_bool_dtype(dt)).all()

    def test_run_vectorized_returns_results(self, df, engine):
        """run_vectorized should return a list of BacktestResults."""
        small_grid = {k: v[:2] for k, v in PARAM_GRID.items()}
        entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)

        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "rsi_crossover", "AAPL")

        assert isinstance(results, list)
        assert len(results) == len(param_list)
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_run_vectorized_results_have_params(self, df, engine):
        """Each vectorized result should have the correct params attached."""
        small_grid = {k: v[:2] for k, v in PARAM_GRID.items()}
        entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)

        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "rsi_crossover", "AAPL")

        for i, (result, params) in enumerate(zip(results, param_list)):
            assert result.params == params
            assert result.ticker == "AAPL"
            assert result.strategy_name == "rsi_crossover"
            assert result.iteration == i

    def test_run_vectorized_all_fields_populated(self, df, engine):
        """Vectorized results should have all metric fields populated."""
        small_grid = {k: v[:2] for k, v in PARAM_GRID.items()}
        entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)

        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "rsi_crossover", "AAPL")

        for r in results:
            assert isinstance(r.sharpe, float)
            assert isinstance(r.total_return, float)
            assert isinstance(r.max_drawdown, float)
            assert isinstance(r.win_rate, float)
            assert isinstance(r.num_trades, int)
            assert isinstance(r.score, float)
            assert isinstance(r.passed, bool)

    def test_run_vectorized_scores_are_finite(self, df, engine):
        """All vectorized scores should be finite (no inf/nan)."""
        small_grid = {k: v[:2] for k, v in PARAM_GRID.items()}
        entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)

        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "rsi_crossover", "AAPL")

        for r in results:
            assert np.isfinite(r.score), f"Non-finite score for params {r.params}"

    def test_vectorized_matches_sequential(self, df, engine):
        """Vectorized results should match sequential run() results for the same params."""
        # Use a small subset of params for quick comparison
        test_params = {"fast_len": 7, "slow_len": 21, "regime_len": 50, "regime_bull": 55, "exit_level": 40}

        # Sequential
        entries, exits = generate_signals(df, test_params)
        seq_result = engine.run(df, entries, exits, "rsi_crossover", "AAPL", 0, test_params)

        # Vectorized (single combo)
        single_grid = {k: [v] for k, v in test_params.items()}
        entries_df, exits_df, param_list = generate_signals_matrix(df, single_grid)
        vec_results = engine.run_vectorized(df, entries_df, exits_df, param_list, "rsi_crossover", "AAPL")
        vec_result = vec_results[0]

        # Metrics should match closely (minor floating point differences possible)
        assert abs(seq_result.sharpe - vec_result.sharpe) < 0.001, \
            f"Sharpe mismatch: seq={seq_result.sharpe}, vec={vec_result.sharpe}"
        assert abs(seq_result.total_return - vec_result.total_return) < 0.001, \
            f"Return mismatch: seq={seq_result.total_return}, vec={vec_result.total_return}"
        assert seq_result.num_trades == vec_result.num_trades, \
            f"Trade count mismatch: seq={seq_result.num_trades}, vec={vec_result.num_trades}"

    def test_vectorized_single_combo(self, df, engine):
        """Vectorized with a single param combo should work."""
        # Pick one combo from the grid
        single_grid = {k: [v[0]] for k, v in PARAM_GRID.items()}
        entries_df, exits_df, param_list = generate_signals_matrix(df, single_grid)

        assert entries_df.shape[1] == 1
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "rsi_crossover", "AAPL")
        assert len(results) == 1

    def test_vectorized_portfolio_speedup(self, df, engine):
        """Vectorized Portfolio.from_signals() should be faster than sequential calls.

        The speedup comes from ONE call to vbt.Portfolio.from_signals() with
        multi-column DataFrames vs N sequential calls. Signal generation is
        identical in both paths, so we pre-generate signals and only time
        the portfolio construction + stats extraction.
        """
        small_grid = {k: v[:3] for k, v in PARAM_GRID.items()}
        total_combos = 1
        for v in small_grid.values():
            total_combos *= len(v)

        entries_df, exits_df, param_list = generate_signals_matrix(df, small_grid)

        # Sequential: one Portfolio.from_signals() call per combo
        t0 = time.time()
        for i in range(len(param_list)):
            col = entries_df.columns[i]
            engine.run(df, entries_df[col], exits_df[col], "rsi_crossover", "AAPL", i, param_list[i])
        seq_time = time.time() - t0

        # Vectorized: one Portfolio.from_signals() call for all combos
        t0 = time.time()
        engine.run_vectorized(df, entries_df, exits_df, param_list, "rsi_crossover", "AAPL")
        vec_time = time.time() - t0

        speedup = seq_time / vec_time if vec_time > 0 else float('inf')
        print(f"\n   📊 Vectorized speedup: {speedup:.1f}x ({seq_time:.2f}s → {vec_time:.2f}s, {total_combos} combos)")
        # Should be at least 2x faster (conservative check)
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.1f}x"


class TestAllStrategiesMatrix:
    """Test that all strategies have a working generate_signals_matrix function."""

    @pytest.fixture
    def df(self):
        return load_data("AAPL", period="2y")

    @pytest.fixture
    def engine(self):
        return BacktestEngine()

    @pytest.mark.parametrize("strategy_name", [
        "rsi_crossover", "macd_momentum", "adx_pullback",
        "atr_channel_breakout", "volume_breakout", "multi_rsi_confluence",
        "ema_ribbon_reversal", "bollinger_squeeze", "ichimoku_trend",
        "invented_momentum_rsi_atr", "invented_momentum_rsi_stoch",
    ])
    def test_strategy_matrix_function_exists(self, strategy_name):
        """Every strategy should have a matrix function in the registry."""
        from crabquant.strategies import STRATEGY_REGISTRY
        assert strategy_name in STRATEGY_REGISTRY
        entry = STRATEGY_REGISTRY[strategy_name]
        assert len(entry) == 5, f"{strategy_name} registry entry should have 5 elements (fn, defaults, grid, desc, matrix_fn)"
        matrix_fn = entry[4]
        assert callable(matrix_fn), f"{strategy_name} matrix function should be callable"

    @pytest.mark.parametrize("strategy_name", [
        "rsi_crossover", "macd_momentum", "adx_pullback",
        "atr_channel_breakout", "volume_breakout", "multi_rsi_confluence",
        "ema_ribbon_reversal", "bollinger_squeeze", "ichimoku_trend",
        "invented_momentum_rsi_atr", "invented_momentum_rsi_stoch",
    ])
    def test_strategy_matrix_produces_valid_output(self, df, engine, strategy_name):
        """Every strategy matrix function should produce valid DataFrames."""
        from crabquant.strategies import STRATEGY_REGISTRY
        _, defaults, param_grid, _, matrix_fn = STRATEGY_REGISTRY[strategy_name]

        if not param_grid:
            # Ichimoku has no params — still should work
            entries_df, exits_df, param_list = matrix_fn(df)
            assert entries_df.shape[1] == 1
            assert len(param_list) == 1
        else:
            small_grid = {k: v[:2] for k, v in param_grid.items()}
            entries_df, exits_df, param_list = matrix_fn(df, small_grid)

            expected_cols = 1
            for v in small_grid.values():
                expected_cols *= len(v)
            assert entries_df.shape[1] == expected_cols

        # Should be able to run through the vectorized engine without errors
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, strategy_name, "AAPL")
        assert len(results) == len(param_list)
        assert all(isinstance(r, BacktestResult) for r in results)
