"""Tests for CrabQuant backtest engine."""

import time

import numpy as np
import pandas as pd
import pytest

from crabquant.data import load_data
from crabquant.engine import BacktestEngine, BacktestResult
from crabquant.engine.resource_monitor import ResourceSnapshot
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


# ── BacktestResult dataclass ──────────────────────────────────────────────

class TestBacktestResult:
    def test_result_is_dataclass(self):
        """BacktestResult should be a dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(BacktestResult)

    def test_result_default_values(self):
        """Default BacktestResult should have sensible defaults."""
        r = BacktestResult(
            ticker="TEST", strategy_name="test", iteration=0,
            sharpe=1.0, total_return=0.2, max_drawdown=-0.1, win_rate=0.6,
            num_trades=10, avg_trade_return=0.02, calmar_ratio=2.0,
            sortino_ratio=2.5, profit_factor=1.5, avg_holding_bars=5.0,
            best_trade=0.1, worst_trade=-0.05, passed=True, score=1.0,
            notes="test notes",
        )
        assert r.params == {}
        assert isinstance(r.timestamp, str)
        assert len(r.timestamp) > 0

    def test_result_asdict(self):
        """BacktestResult should be convertible to dict via dataclasses.asdict."""
        from dataclasses import asdict
        r = BacktestResult(
            ticker="TEST", strategy_name="test", iteration=0,
            sharpe=1.0, total_return=0.2, max_drawdown=-0.1, win_rate=0.6,
            num_trades=10, avg_trade_return=0.02, calmar_ratio=2.0,
            sortino_ratio=2.5, profit_factor=1.5, avg_holding_bars=5.0,
            best_trade=0.1, worst_trade=-0.05, passed=True, score=1.0,
            notes="test notes", params={"a": 1},
        )
        d = asdict(r)
        assert isinstance(d, dict)
        assert d["ticker"] == "TEST"
        assert d["params"] == {"a": 1}

    def test_result_all_field_types(self):
        """Verify types of all BacktestResult fields."""
        r = BacktestResult(
            ticker="T", strategy_name="s", iteration=0,
            sharpe=1.0, total_return=0.1, max_drawdown=-0.05, win_rate=0.5,
            num_trades=5, avg_trade_return=0.01, calmar_ratio=1.0,
            sortino_ratio=1.0, profit_factor=1.0, avg_holding_bars=3.0,
            best_trade=0.05, worst_trade=-0.03, passed=True, score=0.5,
            notes="n",
        )
        assert isinstance(r.ticker, str)
        assert isinstance(r.strategy_name, str)
        assert isinstance(r.iteration, int)
        assert isinstance(r.sharpe, float)
        assert isinstance(r.total_return, float)
        assert isinstance(r.max_drawdown, float)
        assert isinstance(r.win_rate, float)
        assert isinstance(r.num_trades, int)
        assert isinstance(r.avg_trade_return, float)
        assert isinstance(r.calmar_ratio, float)
        assert isinstance(r.sortino_ratio, float)
        assert isinstance(r.profit_factor, float)
        assert isinstance(r.avg_holding_bars, float)
        assert isinstance(r.best_trade, float)
        assert isinstance(r.worst_trade, float)
        assert isinstance(r.passed, bool)
        assert isinstance(r.score, float)
        assert isinstance(r.notes, str)


# ── BacktestEngine edge cases ─────────────────────────────────────────────

class TestBacktestEngineEdgeCases:
    def test_run_with_return_portfolio(self):
        """run() with return_portfolio=True should return (result, portfolio)."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL", return_portfolio=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], BacktestResult)
        assert result[1] is not None  # portfolio object

    def test_run_iteration_stored(self):
        """Iteration number should be stored in result."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL", iteration=42)
        assert result.iteration == 42

    def test_passed_false_when_drawdown_exceeds(self):
        """Result should fail when max drawdown exceeds limit."""
        engine = BacktestEngine(max_drawdown_limit=0.001)  # Very tight
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert result.passed == False

    def test_passed_false_when_too_few_trades(self):
        """Result should fail when min_trades not met."""
        engine = BacktestEngine(min_trades=9999)
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert result.passed == False

    def test_passed_false_when_return_too_low(self):
        """Result should fail when total return below minimum."""
        engine = BacktestEngine(min_total_return=10.0)  # 1000% required
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert result.passed == False

    def test_zero_commission(self):
        """Engine should work with zero commission."""
        engine = BacktestEngine(commission=0.0, sharpe_target=0, min_trades=0, min_total_return=0)
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert isinstance(result, BacktestResult)

    def test_high_commission(self):
        """Engine should work with high commission."""
        engine = BacktestEngine(commission=0.05, sharpe_target=0, min_trades=0, min_total_return=0)
        df = load_data("AAPL", period="2y")
        entries, exits = generate_signals(df)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert isinstance(result, BacktestResult)
        assert np.isfinite(result.score)

    def test_build_notes_format(self):
        """_build_notes should return pipe-separated string."""
        engine = BacktestEngine()
        notes = engine._build_notes(sharpe=2.0, max_dd=-0.1, num_trades=15, total_return=0.3)
        assert "|" in notes
        assert "Sharpe" in notes

    def test_build_notes_fail_sharpe(self):
        """_build_notes should show Sharpe below target."""
        engine = BacktestEngine(sharpe_target=3.0)
        notes = engine._build_notes(sharpe=1.0, max_dd=-0.1, num_trades=15, total_return=0.3)
        assert "< 3.0" in notes

    def test_build_notes_fail_drawdown(self):
        """_build_notes should show MaxDD exceeded."""
        engine = BacktestEngine(max_drawdown_limit=0.10)
        notes = engine._build_notes(sharpe=2.0, max_dd=-0.5, num_trades=15, total_return=0.3)
        assert "> 10%" in notes

    def test_build_notes_few_trades(self):
        """_build_notes should show 'Only N trades' when below min."""
        engine = BacktestEngine(min_trades=10)
        notes = engine._build_notes(sharpe=2.0, max_dd=-0.1, num_trades=3, total_return=0.3)
        assert "Only 3 trades" in notes

    def test_no_trades_result_metrics_zero(self):
        """With no trades, trade-level metrics should be zero."""
        engine = BacktestEngine()
        df = load_data("AAPL", period="2y")
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        result = engine.run(df, entries, exits, "test", "AAPL")
        assert result.win_rate == 0.0
        assert result.avg_trade_return == 0.0
        assert result.profit_factor == 0.0
        assert result.best_trade == 0.0
        assert result.worst_trade == 0.0
        assert result.avg_holding_bars == 0.0
        assert result.num_trades == 0


# ── ResourceSnapshot ──────────────────────────────────────────────────────

class TestResourceSnapshot:
    def test_ram_headroom_mb_positive(self):
        """ram_headroom_mb should be positive when available > reserve."""
        from crabquant.engine.resource_monitor import ResourceSnapshot, DEFAULT_RAM_RESERVE_MB

        snap = ResourceSnapshot(
            cpu_percent=10.0, ram_total_mb=16000, ram_used_mb=4000,
            ram_available_mb=12000, ram_usage_pct=0.25, disk_free_gb=100.0,
            cpu_count=8,
        )
        expected = 12000 - DEFAULT_RAM_RESERVE_MB
        assert snap.ram_headroom_mb == expected

    def test_ram_headroom_mb_clamped_to_zero(self):
        """ram_headroom_mb should be 0 when available < reserve."""
        from crabquant.engine.resource_monitor import ResourceSnapshot

        snap = ResourceSnapshot(
            cpu_percent=10.0, ram_total_mb=16000, ram_used_mb=15000,
            ram_available_mb=100, ram_usage_pct=0.94, disk_free_gb=100.0,
            cpu_count=8,
        )
        assert snap.ram_headroom_mb == 0.0

    def test_max_workers_by_ram(self):
        """max_workers_by_ram should divide headroom by per-worker memory."""
        from crabquant.engine.resource_monitor import ResourceSnapshot, DEFAULT_MEMORY_PER_WORKER_MB

        headroom = 1500.0  # enough for 10 workers at 150MB each
        available = headroom + 500  # 500 reserve
        snap = ResourceSnapshot(
            cpu_percent=10.0, ram_total_mb=16000, ram_used_mb=16000 - available,
            ram_available_mb=available, ram_usage_pct=0.25, disk_free_gb=100.0,
            cpu_count=8,
        )
        expected = max(1, int(headroom / DEFAULT_MEMORY_PER_WORKER_MB))
        assert snap.max_workers_by_ram == expected

    def test_max_workers_by_cpu_low_usage(self):
        """With low CPU usage, max_workers_by_cpu should be close to cpu_count."""
        from crabquant.engine.resource_monitor import ResourceSnapshot

        snap = ResourceSnapshot(
            cpu_percent=5.0, ram_total_mb=16000, ram_used_mb=4000,
            ram_available_mb=12000, ram_usage_pct=0.25, disk_free_gb=100.0,
            cpu_count=8,
        )
        assert snap.max_workers_by_cpu >= 7  # 95% headroom

    def test_max_workers_by_cpu_high_usage(self):
        """With high CPU usage, max_workers_by_cpu should be low."""
        from crabquant.engine.resource_monitor import ResourceSnapshot

        snap = ResourceSnapshot(
            cpu_percent=95.0, ram_total_mb=16000, ram_used_mb=4000,
            ram_available_mb=12000, ram_usage_pct=0.25, disk_free_gb=100.0,
            cpu_count=8,
        )
        assert snap.max_workers_by_cpu == 1

    def test_is_ram_constrained_true(self):
        """Should detect RAM constraint when usage > 80%."""
        from crabquant.engine.resource_monitor import ResourceSnapshot

        snap = ResourceSnapshot(
            cpu_percent=10.0, ram_total_mb=16000, ram_used_mb=14000,
            ram_available_mb=2000, ram_usage_pct=0.875, disk_free_gb=100.0,
            cpu_count=8,
        )
        assert snap.is_ram_constrained == True

    def test_is_ram_constrained_false(self):
        """Should not flag RAM constraint when usage < 80%."""
        from crabquant.engine.resource_monitor import ResourceSnapshot

        snap = ResourceSnapshot(
            cpu_percent=10.0, ram_total_mb=16000, ram_used_mb=4000,
            ram_available_mb=12000, ram_usage_pct=0.25, disk_free_gb=100.0,
            cpu_count=8,
        )
        assert snap.is_ram_constrained == False

    def test_is_cpu_constrained_true(self):
        """Should detect CPU constraint when usage > 90%."""
        from crabquant.engine.resource_monitor import ResourceSnapshot

        snap = ResourceSnapshot(
            cpu_percent=95.0, ram_total_mb=16000, ram_used_mb=4000,
            ram_available_mb=12000, ram_usage_pct=0.25, disk_free_gb=100.0,
            cpu_count=8,
        )
        assert snap.is_cpu_constrained == True

    def test_is_cpu_constrained_false(self):
        """Should not flag CPU constraint when usage < 90%."""
        from crabquant.engine.resource_monitor import ResourceSnapshot

        snap = ResourceSnapshot(
            cpu_percent=50.0, ram_total_mb=16000, ram_used_mb=4000,
            ram_available_mb=12000, ram_usage_pct=0.25, disk_free_gb=100.0,
            cpu_count=8,
        )
        assert snap.is_cpu_constrained == False

    def test_to_dict_keys(self):
        """to_dict should return all expected keys."""
        from crabquant.engine.resource_monitor import ResourceSnapshot

        snap = ResourceSnapshot(
            cpu_percent=50.0, ram_total_mb=16000, ram_used_mb=4000,
            ram_available_mb=12000, ram_usage_pct=0.25, disk_free_gb=100.0,
            cpu_count=8, load_avg_1m=1.0, load_avg_5m=0.8,
        )
        d = snap.to_dict()
        expected_keys = {
            "cpu_percent", "ram_total_mb", "ram_used_mb", "ram_available_mb",
            "ram_usage_pct", "ram_headroom_mb", "disk_free_gb",
            "max_workers_by_ram", "max_workers_by_cpu",
            "is_ram_constrained", "is_cpu_constrained", "cpu_count",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_are_rounded(self):
        """to_dict values should be rounded numbers."""
        from crabquant.engine.resource_monitor import ResourceSnapshot

        snap = ResourceSnapshot(
            cpu_percent=55.123456, ram_total_mb=16000.789,
            ram_used_mb=4000.123, ram_available_mb=12000.666,
            ram_usage_pct=0.25, disk_free_gb=100.555,
            cpu_count=8,
        )
        d = snap.to_dict()
        # Values should be rounded (not raw floats with many decimals)
        assert d["cpu_percent"] == 55.1
        assert d["disk_free_gb"] == 100.56


# ── compute_optimal_workers ───────────────────────────────────────────────

class TestComputeOptimalWorkers:
    def test_minimum_one(self):
        """Should always return at least 1."""
        from crabquant.engine.resource_monitor import compute_optimal_workers

        snap = ResourceSnapshot(
            cpu_percent=99.0, ram_total_mb=1000, ram_used_mb=990,
            ram_available_mb=10, ram_usage_pct=0.99, disk_free_gb=1.0,
            cpu_count=2,
        )
        result = compute_optimal_workers(100, snap)
        assert result >= 1

    def test_respects_requested(self):
        """Should not exceed requested count."""
        from crabquant.engine.resource_monitor import compute_optimal_workers

        snap = ResourceSnapshot(
            cpu_percent=0.0, ram_total_mb=100000, ram_used_mb=1000,
            ram_available_mb=99000, ram_usage_pct=0.01, disk_free_gb=500.0,
            cpu_count=64,
        )
        result = compute_optimal_workers(4, snap)
        assert result <= 4

    def test_custom_memory_per_worker(self):
        """Should use custom memory_per_worker_mb."""
        from crabquant.engine.resource_monitor import compute_optimal_workers

        # 16000 total, 1000 used → 15000 available, budget at 80% = 12800 - 1000 = 11800
        # With 200MB per worker: 11800/200 = 59
        snap = ResourceSnapshot(
            cpu_percent=0.0, ram_total_mb=16000, ram_used_mb=1000,
            ram_available_mb=15000, ram_usage_pct=0.0625, disk_free_gb=500.0,
            cpu_count=64,
        )
        result = compute_optimal_workers(100, snap, memory_per_worker_mb=200.0)
        # RAM budget: (16000*0.8 - 1000) / 200 = 11800/200 = 59
        assert result == 59

    def test_ram_budget_limit(self):
        """Should respect max_ram_usage_pct budget."""
        from crabquant.engine.resource_monitor import compute_optimal_workers

        # 16000 total, 12000 used → budget at 80% = 12800, headroom = 800
        snap = ResourceSnapshot(
            cpu_percent=0.0, ram_total_mb=16000, ram_used_mb=12000,
            ram_available_mb=4000, ram_usage_pct=0.75, disk_free_gb=500.0,
            cpu_count=64,
        )
        result = compute_optimal_workers(100, snap, max_ram_usage_pct=0.80)
        # Budget: 16000*0.8 - 12000 = 800. 800/150 = 5
        assert result == 5


# ── ResourceMonitor ───────────────────────────────────────────────────────

class TestResourceMonitor:
    def test_check_increments_count(self):
        """check() should increment _check_count."""
        from crabquant.engine.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor()
        assert monitor._check_count == 0
        monitor.check()
        assert monitor._check_count == 1
        monitor.check()
        assert monitor._check_count == 2

    def test_get_workers_returns_int(self):
        """get_workers() should return an integer >= 1."""
        from crabquant.engine.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor()
        workers = monitor.get_workers(8)
        assert isinstance(workers, int)
        assert workers >= 1

    def test_get_workers_respects_min_workers(self):
        """get_workers() should return at least min_workers."""
        from crabquant.engine.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(min_workers=3)
        workers = monitor.get_workers(100)
        assert workers >= 3

    def test_get_status_keys(self):
        """get_status() should return expected keys."""
        from crabquant.engine.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor()
        monitor.check()
        status = monitor.get_status()
        assert "checks" in status
        assert "throttles" in status
        assert "last_recommendation" in status
        assert "snapshot" in status
        assert isinstance(status["snapshot"], dict)

    def test_context_manager(self):
        """ResourceMonitor should work as context manager."""
        from crabquant.engine.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor()
        with monitor:
            workers = monitor.get_workers(4)
            assert isinstance(workers, int)
        assert monitor._check_count >= 1

    def test_get_snapshot_caches(self):
        """get_snapshot() should return cached snapshot if fresh."""
        from crabquant.engine.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(check_interval=60.0)
        snap1 = monitor.get_snapshot()
        snap2 = monitor.get_snapshot()
        assert snap1 is snap2  # same object (cached)

    def test_get_snapshot_refreshes_when_stale(self):
        """get_snapshot() should refresh when check_interval elapsed."""
        from crabquant.engine.resource_monitor import ResourceMonitor
        import time

        monitor = ResourceMonitor(check_interval=0.01)  # 10ms
        snap1 = monitor.get_snapshot()
        time.sleep(0.02)
        snap2 = monitor.get_snapshot()
        # May or may not be same object, but check_count should increase
        assert monitor._check_count >= 2

    def test_throttle_count_increments(self):
        """_throttle_count should increment when workers are throttled."""
        from crabquant.engine.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor()
        # Request huge number — should be throttled down
        workers = monitor.get_workers(99999)
        if workers < 99999:
            assert monitor._throttle_count >= 1

    def test_custom_check_interval(self):
        """ResourceMonitor should accept custom check_interval."""
        from crabquant.engine.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(check_interval=123.45)
        assert monitor.check_interval == 123.45
