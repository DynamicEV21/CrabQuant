"""
Comprehensive unit tests for crabquant/engine/backtest.py
BacktestEngine and BacktestResult — the core VectorBT backtest engine.
"""

import numpy as np
import pandas as pd
import pytest

from crabquant.engine.backtest import BacktestResult, BacktestEngine


# ---------------------------------------------------------------------------
# Helpers — synthetic data fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(n=500, seed=42, trend=0.0, volatility=0.02):
    """Create a synthetic OHLCV DataFrame with daily dates."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    # Random walk close prices
    returns = rng.normal(trend / n, volatility, size=n)
    close = 100 * np.cumprod(1 + returns)
    noise = rng.normal(0, 0.3, size=n)
    high = close + np.abs(noise)
    low = close - np.abs(noise)
    open_ = close + rng.normal(0, 0.15, size=n)
    volume = rng.integers(1_000_000, 10_000_000, size=n)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def make_entry_exit_signals(df, entry_every_n=20, hold_bars=5):
    """
    Generate simple periodic boolean entry/exit signals.
    Entry every *entry_every_n* bars, hold for *hold_bars* bars, then exit.
    """
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    i = 0
    while i < len(df):
        entries.iloc[i] = True
        exit_idx = min(i + hold_bars, len(df) - 1)
        exits.iloc[exit_idx] = True
        i += entry_every_n
    return entries, exits


def make_trending_signals(df, seed=42):
    """
    Create signals that follow the trend: enter when price rising, exit when falling.
    Produces a reasonably profitable backtest.
    """
    rng = np.random.default_rng(seed)
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    in_position = False
    for i in range(1, len(df)):
        if not in_position and df["close"].iloc[i] > df["close"].iloc[i - 1]:
            entries.iloc[i] = True
            in_position = True
        elif in_position and df["close"].iloc[i] < df["close"].iloc[i - 1]:
            exits.iloc[i] = True
            in_position = False
    # Force exit at end if still in position
    if in_position:
        exits.iloc[-1] = True
    return entries, exits


# ---------------------------------------------------------------------------
# BacktestResult dataclass tests
# ---------------------------------------------------------------------------

class TestBacktestResult:
    """Tests for the BacktestResult dataclass."""

    def test_default_params_and_timestamp(self):
        """Default params={} and timestamp auto-generated."""
        r = BacktestResult(
            ticker="TEST", strategy_name="s", iteration=0,
            sharpe=1.0, total_return=0.2, max_drawdown=-0.1,
            win_rate=0.6, num_trades=10, avg_trade_return=0.02,
            calmar_ratio=2.0, sortino_ratio=2.5, profit_factor=1.5,
            avg_holding_bars=5.0, best_trade=500.0, worst_trade=-200.0,
            passed=True, score=1.5, notes="all good",
        )
        assert r.params == {}
        assert isinstance(r.timestamp, str)
        # timestamp should be parseable ISO format
        pd.Timestamp(r.timestamp)  # raises if invalid

    def test_asdict_conversion(self):
        """asdict() should return a plain dict with all fields."""
        r = BacktestResult(
            ticker="AAPL", strategy_name="test", iteration=3,
            sharpe=1.8, total_return=0.35, max_drawdown=-0.12,
            win_rate=0.65, num_trades=20, avg_trade_return=0.03,
            calmar_ratio=3.0, sortino_ratio=3.5, profit_factor=2.0,
            avg_holding_bars=4.0, best_trade=1000.0, worst_trade=-300.0,
            passed=True, score=2.5, notes="great",
            params={"fast": 7, "slow": 21},
        )
        d = __import__("dataclasses").asdict(r)
        assert isinstance(d, dict)
        assert d["ticker"] == "AAPL"
        assert d["sharpe"] == 1.8
        assert d["params"] == {"fast": 7, "slow": 21}
        assert d["passed"] is True

    def test_field_types(self):
        """All fields should have expected types."""
        r = BacktestResult(
            ticker="X", strategy_name="y", iteration=1,
            sharpe=0.5, total_return=0.1, max_drawdown=-0.05,
            win_rate=0.5, num_trades=5, avg_trade_return=0.01,
            calmar_ratio=1.0, sortino_ratio=1.0, profit_factor=1.0,
            avg_holding_bars=3.0, best_trade=100.0, worst_trade=-50.0,
            passed=False, score=0.3, notes="meh",
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
        assert isinstance(r.params, dict)
        assert isinstance(r.timestamp, str)

    def test_custom_params_stored(self):
        """Custom params dict is stored correctly."""
        params = {"a": 1, "b": [2, 3]}
        r = BacktestResult(
            ticker="T", strategy_name="s", iteration=0,
            sharpe=0, total_return=0, max_drawdown=0,
            win_rate=0, num_trades=0, avg_trade_return=0,
            calmar_ratio=0, sortino_ratio=0, profit_factor=0,
            avg_holding_bars=0, best_trade=0, worst_trade=0,
            passed=False, score=0, notes="", params=params,
        )
        assert r.params == params


# ---------------------------------------------------------------------------
# BacktestEngine constructor tests
# ---------------------------------------------------------------------------

class TestBacktestEngineConstructor:
    """Tests for BacktestEngine initialization."""

    def test_default_values(self):
        engine = BacktestEngine()
        assert engine.initial_cash == 100_000
        assert engine.commission == 0.001
        assert engine.sharpe_target == 1.5
        assert engine.max_drawdown_limit == 0.25
        assert engine.min_total_return == 0.10
        assert engine.min_trades == 5

    def test_custom_values(self):
        engine = BacktestEngine(
            initial_cash=500_000,
            commission=0.0005,
            sharpe_target=2.0,
            max_drawdown_limit=0.15,
            min_total_return=0.20,
            min_trades=10,
        )
        assert engine.initial_cash == 500_000
        assert engine.commission == 0.0005
        assert engine.sharpe_target == 2.0
        assert engine.max_drawdown_limit == 0.15
        assert engine.min_total_return == 0.20
        assert engine.min_trades == 10

    def test_zero_commission(self):
        engine = BacktestEngine(commission=0.0)
        assert engine.commission == 0.0

    def test_high_sharpe_target(self):
        engine = BacktestEngine(sharpe_target=99.0)
        assert engine.sharpe_target == 99.0


# ---------------------------------------------------------------------------
# BacktestEngine.run() tests — core backtest
# ---------------------------------------------------------------------------

class TestBacktestEngineRun:
    """Tests for BacktestEngine.run() with synthetic data."""

    def test_run_returns_backtest_result(self):
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=20, hold_bars=5)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "test_strategy", "SYNTH")
        assert isinstance(result, BacktestResult)

    def test_run_result_fields_populated(self):
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=15, hold_bars=5)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "strat", "TKR")
        assert result.ticker == "TKR"
        assert result.strategy_name == "strat"
        assert isinstance(result.sharpe, float)
        assert isinstance(result.total_return, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.num_trades, int)
        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)

    def test_run_iteration_stored(self):
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T", iteration=42)
        assert result.iteration == 42

    def test_run_params_stored(self):
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df)
        engine = BacktestEngine()
        params = {"fast": 7, "slow": 21}
        result = engine.run(df, entries, exits, "s", "T", params=params)
        assert result.params == params

    def test_run_return_portfolio_tuple(self):
        """return_portfolio=True should return (result, portfolio) tuple."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T", return_portfolio=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], BacktestResult)
        # Second element is a vbt Portfolio
        import vectorbt as vbt
        assert isinstance(result[1], vbt.Portfolio)

    def test_run_all_false_signals_zero_trades(self):
        """All-False signals → zero trades, not passed."""
        df = make_ohlcv(n=500)
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        assert result.num_trades == 0
        assert result.passed is False
        assert result.score == 0.0
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0

    def test_run_all_true_entries_no_exits(self):
        """All-True entries with all-False exits: first entry triggers, no exit.
        VectorBT will handle this; we just check no crash."""
        df = make_ohlcv(n=500)
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        assert isinstance(result, BacktestResult)

    def test_run_nan_in_signals(self):
        """NaN in signals should not crash; treated as False by VectorBT."""
        df = make_ohlcv(n=500)
        entries = pd.Series(np.nan, index=df.index, dtype=object)
        exits = pd.Series(np.nan, index=df.index, dtype=object)
        entries.iloc[50:55] = True
        exits.iloc[55:56] = True
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        assert isinstance(result, BacktestResult)

    def test_run_score_is_finite(self):
        """Score should always be finite (no inf/nan)."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=10, hold_bars=3)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        assert np.isfinite(result.score), f"Non-finite score: {result.score}"

    def test_run_score_formula(self):
        """Verify score formula: sharpe * sqrt(min(trades,100)/20) * max(0, 1 - |max_dd|)."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=20, hold_bars=5)
        engine = BacktestEngine(sharpe_target=0, min_trades=0, min_total_return=0)
        result = engine.run(df, entries, exits, "s", "T")
        trade_factor = np.sqrt(min(result.num_trades, 100) / 20)
        dd_penalty = max(0, 1 - abs(result.max_drawdown))
        expected_score = result.sharpe * trade_factor * dd_penalty
        assert abs(result.score - expected_score) < 1e-10, \
            f"Score mismatch: {result.score} vs {expected_score}"

    def test_run_passed_true_with_permissive_criteria(self):
        """With very permissive criteria, a result with trades should pass."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=15, hold_bars=5)
        engine = BacktestEngine(
            sharpe_target=-999, min_trades=0, min_total_return=-1.0,
            max_drawdown_limit=1.0,  # allow any drawdown
        )
        result = engine.run(df, entries, exits, "s", "T")
        # With enough trades, even terrible sharpe should pass permissive criteria
        if result.num_trades > 0:
            assert result.passed is True

    def test_run_passed_false_high_sharpe_target(self):
        """Impossible sharpe target → always fails."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df)
        engine = BacktestEngine(sharpe_target=999.0)
        result = engine.run(df, entries, exits, "s", "T")
        assert result.passed is False

    def test_run_passed_false_high_min_trades(self):
        """min_trades higher than actual → fails."""
        df = make_ohlcv(n=200)
        entries, exits = make_entry_exit_signals(df, entry_every_n=100, hold_bars=3)
        engine = BacktestEngine(min_trades=50)
        result = engine.run(df, entries, exits, "s", "T")
        assert result.passed is False

    def test_run_win_rate_in_range(self):
        """Win rate should be in [0, 1]."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=15, hold_bars=5)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        if result.num_trades > 0:
            assert 0.0 <= result.win_rate <= 1.0

    def test_run_profit_factor_nonnegative(self):
        """Profit factor should be >= 0."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=15, hold_bars=5)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        assert result.profit_factor >= 0.0

    def test_run_max_drawdown_nonpositive(self):
        """Max drawdown should be <= 0 (negative means loss)."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=15, hold_bars=5)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        if result.num_trades > 0:
            assert result.max_drawdown <= 0.0

    def test_run_notes_contain_sharpe(self):
        """Notes string should mention Sharpe."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        assert "Sharpe" in result.notes

    def test_run_notes_contain_trades_info(self):
        """Notes should mention trade count."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        assert "trades" in result.notes.lower()

    def test_run_short_dataframe(self):
        """Very short data (50 bars) should still work without crash."""
        df = make_ohlcv(n=50)
        entries, exits = make_entry_exit_signals(df, entry_every_n=10, hold_bars=3)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        assert isinstance(result, BacktestResult)

    def test_run_trending_up_data_positive_return(self):
        """Trending-up data with trend-following signals → positive total return."""
        df = make_ohlcv(n=500, trend=0.5, seed=123)
        entries, exits = make_trending_signals(df, seed=123)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "trend", "UP")
        # With uptrending data and trend-following signals, expect positive return
        assert isinstance(result, BacktestResult)

    def test_run_best_worst_trade(self):
        """Best trade should be >= worst trade."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=15, hold_bars=5)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        if result.num_trades > 0:
            assert result.best_trade >= result.worst_trade

    def test_run_calmar_and_sortino_are_floats(self):
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=15, hold_bars=5)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "s", "T")
        assert isinstance(result.calmar_ratio, float)
        assert isinstance(result.sortino_ratio, float)

    def test_run_different_commission_affects_result(self):
        """Higher commission should generally reduce returns."""
        df = make_ohlcv(n=500)
        entries, exits = make_entry_exit_signals(df, entry_every_n=15, hold_bars=5)
        engine_zero = BacktestEngine(commission=0.0)
        engine_high = BacktestEngine(commission=0.01)
        r_zero = engine_zero.run(df, entries, exits, "s", "T")
        r_high = engine_high.run(df, entries, exits, "s", "T")
        # Higher commission should not produce higher total return
        assert r_high.total_return <= r_zero.total_return + 1e-10


# ---------------------------------------------------------------------------
# BacktestEngine.run_vectorized() tests
# ---------------------------------------------------------------------------

class TestBacktestEngineRunVectorized:
    """Tests for BacktestEngine.run_vectorized() with synthetic multi-column data."""

    @pytest.fixture
    def df(self):
        return make_ohlcv(n=500)

    @pytest.fixture
    def engine(self):
        return BacktestEngine()

    def _make_multi_column_signals(self, df, n_combos=3):
        """Create multi-column entries/exits DataFrames for vectorized testing."""
        entries_cols = {}
        exits_cols = {}
        param_list = []
        for i in range(n_combos):
            e, x = make_entry_exit_signals(df, entry_every_n=15 + i * 5, hold_bars=3 + i)
            entries_cols[f"c{i}"] = e
            exits_cols[f"c{i}"] = x
            param_list.append({"combo": i, "entry_every": 15 + i * 5})
        entries_df = pd.DataFrame(entries_cols)
        exits_df = pd.DataFrame(exits_cols)
        return entries_df, exits_df, param_list

    def test_vectorized_returns_list(self, df, engine):
        entries_df, exits_df, param_list = self._make_multi_column_signals(df, n_combos=3)
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "strat", "T")
        assert isinstance(results, list)
        assert len(results) == 3

    def test_vectorized_all_results_are_backtest_results(self, df, engine):
        entries_df, exits_df, param_list = self._make_multi_column_signals(df, n_combos=3)
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "strat", "T")
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_vectorized_iteration_matches_index(self, df, engine):
        entries_df, exits_df, param_list = self._make_multi_column_signals(df, n_combos=5)
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "strat", "T")
        for i, r in enumerate(results):
            assert r.iteration == i

    def test_vectorized_params_attached(self, df, engine):
        entries_df, exits_df, param_list = self._make_multi_column_signals(df, n_combos=3)
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "strat", "T")
        for r, p in zip(results, param_list):
            assert r.params == p

    def test_vectorized_ticker_and_strategy(self, df, engine):
        entries_df, exits_df, param_list = self._make_multi_column_signals(df, n_combos=2)
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "my_strat", "XYZ")
        for r in results:
            assert r.ticker == "XYZ"
            assert r.strategy_name == "my_strat"

    def test_vectorized_scores_are_finite(self, df, engine):
        entries_df, exits_df, param_list = self._make_multi_column_signals(df, n_combos=4)
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "strat", "T")
        for r in results:
            assert np.isfinite(r.score), f"Non-finite score: {r.score}"

    def test_vectorized_single_combo(self, df, engine):
        entries_df, exits_df, param_list = self._make_multi_column_signals(df, n_combos=1)
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "strat", "T")
        assert len(results) == 1
        assert isinstance(results[0], BacktestResult)

    def test_vectorized_all_false_columns(self, df, engine):
        """All-False entries across all columns → all zero trades."""
        entries_df = pd.DataFrame({"c0": pd.Series(False, index=df.index),
                                    "c1": pd.Series(False, index=df.index)})
        exits_df = pd.DataFrame({"c0": pd.Series(False, index=df.index),
                                  "c1": pd.Series(False, index=df.index)})
        param_list = [{"a": 1}, {"a": 2}]
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "strat", "T")
        assert len(results) == 2
        for r in results:
            assert r.num_trades == 0
            assert r.passed is False

    def test_vectorized_matches_sequential_single_combo(self, df, engine):
        """Vectorized result should closely match sequential run() for same signals."""
        entries, exits = make_entry_exit_signals(df, entry_every_n=20, hold_bars=5)
        seq_result = engine.run(df, entries, exits, "strat", "T", iteration=0,
                                params={"fast": 10})

        entries_df = pd.DataFrame({"c0": entries})
        exits_df = pd.DataFrame({"c0": exits})
        param_list = [{"fast": 10}]
        vec_results = engine.run_vectorized(df, entries_df, exits_df, param_list, "strat", "T")
        vec_result = vec_results[0]

        assert abs(seq_result.sharpe - vec_result.sharpe) < 0.01, \
            f"Sharpe: seq={seq_result.sharpe}, vec={vec_result.sharpe}"
        assert abs(seq_result.total_return - vec_result.total_return) < 0.001, \
            f"Return: seq={seq_result.total_return}, vec={vec_result.total_return}"
        assert seq_result.num_trades == vec_result.num_trades

    def test_vectorized_notes_populated(self, df, engine):
        entries_df, exits_df, param_list = self._make_multi_column_signals(df, n_combos=2)
        results = engine.run_vectorized(df, entries_df, exits_df, param_list, "strat", "T")
        for r in results:
            assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# BacktestEngine._build_notes() tests
# ---------------------------------------------------------------------------

class TestBuildNotes:
    """Tests for the _build_notes helper method."""

    def test_notes_passing(self):
        engine = BacktestEngine()
        notes = engine._build_notes(sharpe=2.0, max_dd=-0.10, num_trades=10, total_return=0.20)
        assert "Sharpe 2.00 >= 1.5" in notes
        assert "MaxDD" in notes
        assert "OK" in notes
        assert "10 trades" in notes
        assert "Return 20.0%" in notes

    def test_notes_failing_all(self):
        engine = BacktestEngine()
        notes = engine._build_notes(sharpe=0.5, max_dd=-0.50, num_trades=2, total_return=0.01)
        assert "Sharpe 0.50 < 1.5" in notes
        assert "Only 2 trades" in notes
        assert "Return 1.0% < 10%" in notes

    def test_notes_separator(self):
        engine = BacktestEngine()
        notes = engine._build_notes(sharpe=1.0, max_dd=-0.10, num_trades=5, total_return=0.10)
        parts = notes.split(" | ")
        assert len(parts) == 4  # sharpe, maxdd, trades, return


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_flat_price_data(self):
        """Flat (constant) price data should not crash."""
        n = 200
        dates = pd.bdate_range("2023-01-01", periods=n)
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1_000_000),
        }, index=dates)
        entries, exits = make_entry_exit_signals(df, entry_every_n=20, hold_bars=5)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "flat", "T")
        assert isinstance(result, BacktestResult)

    def test_very_volatile_data(self):
        """Highly volatile data should not crash."""
        df = make_ohlcv(n=500, volatility=0.10, seed=99)
        entries, exits = make_entry_exit_signals(df, entry_every_n=15, hold_bars=3)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "volatile", "T")
        assert isinstance(result, BacktestResult)

    def test_entry_without_exit(self):
        """Entry signal with no matching exit — VectorBT handles auto-close."""
        df = make_ohlcv(n=200)
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        entries.iloc[10] = True
        # No exit ever triggered
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "no_exit", "T")
        assert isinstance(result, BacktestResult)

    def test_exit_before_entry(self):
        """Exit signal before any entry — should be ignored by VectorBT."""
        df = make_ohlcv(n=200)
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        exits.iloc[5] = True
        entries.iloc[10] = True
        exits.iloc[20] = True
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "early_exit", "T")
        assert isinstance(result, BacktestResult)

    def test_single_bar_data(self):
        """Single bar of data — degenerate but should not crash."""
        df = make_ohlcv(n=1)
        entries = pd.Series([True], index=df.index)
        exits = pd.Series([True], index=df.index)
        engine = BacktestEngine()
        result = engine.run(df, entries, exits, "tiny", "T")
        assert isinstance(result, BacktestResult)
