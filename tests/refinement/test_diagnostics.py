"""Tests for crabquant.refinement.diagnostics."""

import numpy as np
import pandas as pd
import pytest
from types import ModuleType
from unittest.mock import MagicMock, patch, call

from crabquant.engine.backtest import BacktestResult
from crabquant.refinement.diagnostics import (
    compute_sharpe_by_year,
    compute_strategy_hash,
    run_backtest_safely,
)


# ── Factories ─────────────────────────────────────────────────────────────────

def make_result(**overrides) -> BacktestResult:
    defaults = dict(
        ticker="AAPL", strategy_name="test", iteration=0,
        sharpe=1.5, total_return=0.15, max_drawdown=-0.10,
        win_rate=0.55, num_trades=30, avg_trade_return=0.02,
        calmar_ratio=1.5, sortino_ratio=2.0, profit_factor=1.5,
        avg_holding_bars=5.0, best_trade=500.0, worst_trade=-200.0,
        passed=True, score=1.2, notes="ok", params={},
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def make_sample_df(n: int = 252, start: str = "2022-01-03") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({
        "open": np.full(n, 100.0),
        "high": np.full(n, 105.0),
        "low":  np.full(n, 95.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 1_000_000.0),
    }, index=idx)


def make_strategy_module(signals_fn=None, params=None) -> MagicMock:
    module = MagicMock(spec=["generate_signals", "DEFAULT_PARAMS", "DESCRIPTION"])
    module.DEFAULT_PARAMS = params or {"period": 14}
    module.DESCRIPTION = "Test strategy for unit tests"

    if signals_fn is not None:
        module.generate_signals = signals_fn
    else:
        def _default(df, p):
            entries = pd.Series(False, index=df.index)
            exits = pd.Series(False, index=df.index)
            entries.iloc[::20] = True
            exits.iloc[10::20] = True
            return entries, exits
        module.generate_signals = _default

    return module


def make_portfolio_mock(years: list[int] = None, n_per_year: int = 252, seed: int = 42) -> MagicMock:
    """Mock vbt.Portfolio with a .returns() pd.Series spanning given years."""
    rng = np.random.default_rng(seed)
    if years is None:
        years = [2022, 2023, 2024]

    idx_parts = []
    for yr in years:
        idx_parts.append(pd.date_range(f"{yr}-01-03", periods=n_per_year, freq="B"))
    idx = idx_parts[0]
    for p in idx_parts[1:]:
        idx = idx.append(p)

    returns = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
    pf = MagicMock()
    pf.returns.return_value = returns
    return pf


# ── run_backtest_safely ───────────────────────────────────────────────────────

class TestRunBacktestSafely:

    @patch("crabquant.refinement.diagnostics.BacktestEngine")
    @patch("crabquant.refinement.diagnostics.load_data")
    def test_returns_result_df_none_when_no_portfolio(self, mock_load, mock_engine_cls):
        """return_portfolio=False → (result, df, None) without calling engine with the kwarg."""
        df = make_sample_df()
        mock_load.return_value = df
        result = make_result()
        mock_engine = mock_engine_cls.return_value
        mock_engine.run.return_value = result

        module = make_strategy_module()
        r, ret_df, portfolio = run_backtest_safely(module, "AAPL", "2y", return_portfolio=False)

        assert r is result
        assert ret_df is df
        assert portfolio is None

        # Engine must NOT have been called with return_portfolio kwarg
        _, call_kwargs = mock_engine.run.call_args
        assert "return_portfolio" not in call_kwargs

    @patch("crabquant.refinement.diagnostics.BacktestEngine")
    @patch("crabquant.refinement.diagnostics.load_data")
    def test_returns_portfolio_when_engine_supports_it(self, mock_load, mock_engine_cls):
        """return_portfolio=True + engine returns (result, pf) → (result, df, pf)."""
        df = make_sample_df()
        mock_load.return_value = df
        result = make_result()
        mock_pf = MagicMock(name="portfolio")
        mock_engine = mock_engine_cls.return_value
        mock_engine.run.return_value = (result, mock_pf)

        module = make_strategy_module()
        r, ret_df, pf = run_backtest_safely(module, "AAPL", "2y", return_portfolio=True)

        assert r is result
        assert ret_df is df
        assert pf is mock_pf

    @patch("crabquant.refinement.diagnostics.BacktestEngine")
    @patch("crabquant.refinement.diagnostics.load_data")
    def test_falls_back_when_engine_raises_type_error(self, mock_load, mock_engine_cls):
        """return_portfolio=True + engine TypeError → retry without kwarg → portfolio=None."""
        df = make_sample_df()
        mock_load.return_value = df
        result = make_result()
        mock_engine = mock_engine_cls.return_value
        # First call raises TypeError (unsupported kwarg); second call succeeds
        mock_engine.run.side_effect = [
            TypeError("unexpected keyword argument 'return_portfolio'"),
            result,
        ]

        module = make_strategy_module()
        r, ret_df, pf = run_backtest_safely(module, "AAPL", "2y", return_portfolio=True)

        assert r is result
        assert ret_df is df
        assert pf is None
        assert mock_engine.run.call_count == 2

    @patch("crabquant.refinement.diagnostics.load_data")
    def test_value_error_from_load_data_returns_none_tuple(self, mock_load):
        """ValueError from load_data → (None, None, None)."""
        mock_load.side_effect = ValueError("No data returned from yfinance for 'FAKE'")
        module = make_strategy_module()
        assert run_backtest_safely(module, "FAKE", "2y") == (None, None, None)

    @patch("crabquant.refinement.diagnostics.load_data")
    def test_empty_dataframe_returns_none_tuple(self, mock_load):
        """Empty DataFrame from load_data → (None, None, None)."""
        mock_load.return_value = pd.DataFrame()
        module = make_strategy_module()
        assert run_backtest_safely(module, "FAKE", "2y") == (None, None, None)

    @patch("crabquant.refinement.diagnostics.BacktestEngine")
    @patch("crabquant.refinement.diagnostics.load_data")
    def test_exception_from_generate_signals_returns_none_tuple(self, mock_load, mock_engine_cls):
        """generate_signals raises → (None, None, None)."""
        mock_load.return_value = make_sample_df()

        def bad_signals(df, p):
            raise RuntimeError("indicator blew up")

        module = make_strategy_module(signals_fn=bad_signals)
        assert run_backtest_safely(module, "AAPL", "2y") == (None, None, None)

    @patch("crabquant.refinement.diagnostics.BacktestEngine")
    @patch("crabquant.refinement.diagnostics.load_data")
    def test_passes_correct_ticker_period_and_params(self, mock_load, mock_engine_cls):
        """load_data called with correct ticker+period; generate_signals gets DEFAULT_PARAMS."""
        df = make_sample_df()
        mock_load.return_value = df
        mock_engine = mock_engine_cls.return_value
        mock_engine.run.return_value = make_result()

        captured_params: list = []

        def capturing_signals(df, p):
            captured_params.append(p)
            return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

        custom_params = {"window": 20, "threshold": 0.5}
        module = make_strategy_module(signals_fn=capturing_signals, params=custom_params)

        run_backtest_safely(module, "MSFT", "1y")

        mock_load.assert_called_once_with("MSFT", "1y")
        assert captured_params == [custom_params]

    @patch("crabquant.refinement.diagnostics.BacktestEngine")
    @patch("crabquant.refinement.diagnostics.load_data")
    def test_default_period_is_2y(self, mock_load, mock_engine_cls):
        """Period defaults to '2y' when not specified."""
        df = make_sample_df()
        mock_load.return_value = df
        mock_engine_cls.return_value.run.return_value = make_result()

        module = make_strategy_module()
        run_backtest_safely(module, "AAPL")

        mock_load.assert_called_once_with("AAPL", "2y")

    @patch("crabquant.refinement.diagnostics.BacktestEngine")
    @patch("crabquant.refinement.diagnostics.load_data")
    def test_module_without_description_attribute(self, mock_load, mock_engine_cls):
        """Modules lacking DESCRIPTION don't crash; a default name is used."""
        df = make_sample_df()
        mock_load.return_value = df
        mock_engine = mock_engine_cls.return_value
        mock_engine.run.return_value = make_result()

        module = MagicMock(spec=["generate_signals", "DEFAULT_PARAMS"])
        module.DEFAULT_PARAMS = {}
        module.generate_signals = lambda df, p: (
            pd.Series(False, index=df.index),
            pd.Series(False, index=df.index),
        )

        r, ret_df, pf = run_backtest_safely(module, "AAPL")
        assert r is not None


# ── compute_sharpe_by_year ────────────────────────────────────────────────────

class TestComputeSharpeByYear:

    def test_returns_dict_with_string_year_keys(self):
        pf = make_portfolio_mock(years=[2022, 2023, 2024])
        result = compute_sharpe_by_year(pf)
        assert isinstance(result, dict)
        assert len(result) > 0
        for key in result:
            assert key.isdigit() and len(key) == 4

    def test_values_are_rounded_floats(self):
        pf = make_portfolio_mock()
        result = compute_sharpe_by_year(pf)
        for v in result.values():
            assert isinstance(v, float)
            # Rounded to 4 decimal places — repr has at most 4 sig figs after dot
            assert v == round(v, 4)

    def test_returns_empty_dict_for_none_portfolio(self):
        assert compute_sharpe_by_year(None) == {}

    def test_returns_empty_dict_on_exception(self):
        pf = MagicMock()
        pf.returns.side_effect = RuntimeError("portfolio crashed")
        assert compute_sharpe_by_year(pf) == {}

    def test_skips_years_with_fewer_than_10_points(self):
        """A year represented by < 10 bars should be omitted."""
        idx_tiny = pd.date_range("2021-12-27", periods=3, freq="B")  # 3 bars in 2021
        idx_full = pd.date_range("2022-01-03", periods=252, freq="B")
        idx = idx_tiny.append(idx_full)
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
        pf = MagicMock()
        pf.returns.return_value = returns
        result = compute_sharpe_by_year(pf)
        assert "2021" not in result
        assert "2022" in result

    def test_zero_std_returns_zero(self):
        """Constant daily returns → std = 0 → Sharpe reported as 0.0."""
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        returns = pd.Series(0.001, index=idx)
        pf = MagicMock()
        pf.returns.return_value = returns
        result = compute_sharpe_by_year(pf)
        assert result.get("2023") == 0.0

    def test_annualization_factor_is_sqrt_252(self):
        """Verify the exact formula: mean/std * sqrt(252), rounded to 4dp."""
        np.random.seed(99)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        rets = pd.Series(np.random.randn(252) * 0.01, index=idx)
        pf = MagicMock()
        pf.returns.return_value = rets
        result = compute_sharpe_by_year(pf)
        expected = round(rets.mean() / rets.std() * (252 ** 0.5), 4)
        assert result.get("2023") == expected

    def test_multiple_years_all_present(self):
        pf = make_portfolio_mock(years=[2022, 2023, 2024], n_per_year=252)
        result = compute_sharpe_by_year(pf)
        assert "2022" in result
        assert "2023" in result
        assert "2024" in result


# ── compute_strategy_hash ─────────────────────────────────────────────────────

class TestComputeStrategyHash:

    def test_returns_12_character_string(self):
        assert len(compute_strategy_hash("def foo(): pass")) == 12

    def test_returns_hex_string(self):
        result = compute_strategy_hash("some code")
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        code = "def generate_signals(df, params):\n    return df, df"
        assert compute_strategy_hash(code) == compute_strategy_hash(code)

    def test_different_code_produces_different_hash(self):
        h1 = compute_strategy_hash("def foo(): return 1")
        h2 = compute_strategy_hash("def foo(): return 2")
        assert h1 != h2

    def test_blank_lines_are_ignored(self):
        code_compact = "def foo():\n    pass"
        code_spaced = "def foo():\n\n    pass\n\n"
        assert compute_strategy_hash(code_compact) == compute_strategy_hash(code_spaced)

    def test_leading_trailing_whitespace_per_line_ignored(self):
        code_indented = "  def foo():\n    return 1"
        code_stripped = "def foo():\nreturn 1"
        assert compute_strategy_hash(code_indented) == compute_strategy_hash(code_stripped)

    def test_empty_string_does_not_crash(self):
        result = compute_strategy_hash("")
        assert isinstance(result, str)
        assert len(result) == 12

    def test_whitespace_only_same_as_empty(self):
        assert compute_strategy_hash("   \n\n  ") == compute_strategy_hash("")
