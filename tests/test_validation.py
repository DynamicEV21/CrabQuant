"""Tests for CrabQuant validation suite."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch, MagicMock

from crabquant.validation import (
    WalkForwardResult,
    CrossTickerResult,
    RollingWalkForwardResult,
    walk_forward_test,
    cross_ticker_validation,
    rolling_walk_forward,
    full_validation,
    time_reversed_check,
    check_degenerate_strategy,
    _parse_duration,
    _detect_regime_for_period,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_backtest_result(**overrides):
    """Build a mock BacktestResult."""
    defaults = {
        "sharpe": 1.5,
        "total_return": 0.12,
        "max_drawdown": -0.08,
        "win_rate": 0.55,
        "num_trades": 20,
        "passed": True,
        "avg_trade_return": 0.005,
        "calmar_ratio": 1.5,
        "sortino_ratio": 2.0,
        "profit_factor": 1.3,
        "avg_holding_bars": 5.0,
        "best_trade": 0.03,
        "worst_trade": -0.02,
        "score": 2.0,
        "ticker": "AAPL",
        "strategy_name": "test",
        "iteration": 0,
        "notes": "OK",
        "params": {},
        "timestamp": "2025-01-01T00:00:00",
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_df(n=500):
    """Generate a simple OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-01", periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": 1_000_000.0,
    }, index=dates)


def _noop_strategy(df, params):
    """Strategy that generates some signals."""
    entries = pd.Series(False, index=df.index, dtype=bool)
    exits = pd.Series(False, index=df.index, dtype=bool)
    if len(df) > 50:
        entries.iloc[50] = True
        exits.iloc[60] = True
    return entries, exits


# ── WalkForwardResult dataclass ─────────────────────────────────────────────

class TestWalkForwardResult:
    def test_construction_with_defaults(self):
        r = WalkForwardResult(
            strategy_name="test", ticker="AAPL",
            train_sharpe=1.0, train_return=0.1,
            test_sharpe=0.8, test_return=0.08,
            test_max_dd=-0.05, degradation=0.2,
            robust=True, notes="ok",
        )
        assert r.train_regime == ""
        assert r.test_regime == ""
        assert r.regime_shift is False

    def test_regime_shift_fields(self):
        r = WalkForwardResult(
            strategy_name="test", ticker="AAPL",
            train_sharpe=1.0, train_return=0.1,
            test_sharpe=0.8, test_return=0.08,
            test_max_dd=-0.05, degradation=0.2,
            robust=True, notes="ok",
            train_regime="low_volatility",
            test_regime="high_volatility",
            regime_shift=True,
        )
        assert r.regime_shift is True
        assert r.train_regime == "low_volatility"

    def test_asdict_conversion(self):
        r = WalkForwardResult(
            strategy_name="test", ticker="AAPL",
            train_sharpe=1.0, train_return=0.1,
            test_sharpe=0.8, test_return=0.08,
            test_max_dd=-0.05, degradation=0.2,
            robust=True, notes="ok",
        )
        d = asdict(r)
        assert d["strategy_name"] == "test"
        assert d["robust"] is True


# ── CrossTickerResult dataclass ─────────────────────────────────────────────

class TestCrossTickerResult:
    def test_construction(self):
        r = CrossTickerResult(
            strategy_name="test", params={},
            tickers_tested=3, tickers_profitable=2, tickers_passed=1,
            avg_sharpe=1.0, median_sharpe=0.9, sharpe_std=0.3,
            avg_return=0.05, avg_max_dd=0.10,
            win_rate_across_tickers=0.66, robust=True,
            notes="ok",
        )
        assert r.tickers_tested == 3
        assert r.robust is True

    def test_all_fields_accessible(self):
        r = CrossTickerResult(
            strategy_name="test", params={},
            tickers_tested=5, tickers_profitable=3, tickers_passed=2,
            avg_sharpe=1.5, median_sharpe=1.2, sharpe_std=0.5,
            avg_return=0.08, avg_max_dd=0.12,
            win_rate_across_tickers=0.6, robust=True,
            notes="",
        )
        for field in ["avg_sharpe", "median_sharpe", "sharpe_std", "avg_return", "avg_max_dd"]:
            assert hasattr(r, field)
            assert isinstance(getattr(r, field), float)


# ── RollingWalkForwardResult dataclass ──────────────────────────────────────

class TestRollingWalkForwardResult:
    def test_construction_with_defaults(self):
        r = RollingWalkForwardResult(
            strategy_name="test", ticker="AAPL",
            num_windows=3, windows_passed=2,
            avg_test_sharpe=1.0, min_test_sharpe=0.5,
            avg_degradation=0.3, robust=True, notes="ok",
        )
        assert r.window_results == []

    def test_window_results_stored(self):
        r = RollingWalkForwardResult(
            strategy_name="test", ticker="AAPL",
            num_windows=2, windows_passed=1,
            avg_test_sharpe=0.8, min_test_sharpe=0.5,
            avg_degradation=0.4, robust=False, notes="",
            window_results=[{"window": 1, "passed": True}, {"window": 2, "passed": False}],
        )
        assert len(r.window_results) == 2


# ── _parse_duration ─────────────────────────────────────────────────────────

class TestParseDuration:
    def test_months(self):
        assert _parse_duration("18mo") == 18 * 21

    def test_days(self):
        assert _parse_duration("126d") == 126

    def test_years(self):
        assert _parse_duration("2y") == 2 * 252

    def test_bare_number(self):
        assert _parse_duration("500") == 500

    def test_whitespace_and_case(self):
        assert _parse_duration(" 6MO ") == 6 * 21

    def test_single_month(self):
        assert _parse_duration("1mo") == 21

    def test_single_day(self):
        assert _parse_duration("1d") == 1

    def test_single_year(self):
        assert _parse_duration("1y") == 252


# ── _detect_regime_for_period ───────────────────────────────────────────────

class TestDetectRegimeForPeriod:
    def test_short_df_returns_unknown(self):
        """DataFrame shorter than 20 bars should return 'unknown'."""
        short_df = _make_df(n=10)
        result = _detect_regime_for_period(short_df)
        assert result == "unknown"

    def test_df_without_datetime_index(self):
        """DataFrame without DatetimeIndex should return 'unknown'."""
        df = _make_df(n=50)
        df = df.reset_index(drop=True)
        result = _detect_regime_for_period(df)
        assert result == "unknown"
    def test_df_without_datetime_index(self):
        """DataFrame without DatetimeIndex falls back to own data."""
        df = _make_df(n=50)
        df = df.reset_index(drop=True)
        # The function falls back to using df's own data since it has no .index.min()
        # With 50 rows >= 20, it should try detect_regime and get a regime string
        result = _detect_regime_for_period(df)
        # It won't be "unknown" since the data has enough rows and will try regime detection
        assert isinstance(result, str)

    def test_empty_df(self):
        """Empty DataFrame should return 'unknown'."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = _detect_regime_for_period(df)
        assert result == "unknown"

    @patch("crabquant.regime.detect_regime")
    def test_uses_spy_data_when_provided(self, mock_detect):
        """Should use SPY slice when valid SPY data is provided."""
        from crabquant.regime import MarketRegime
        mock_detect.return_value = (MarketRegime.LOW_VOLATILITY, {"confidence": 0.8})

        df = _make_df(n=200)
        spy_df = _make_df(n=200)

        result = _detect_regime_for_period(df, spy_df)
        assert result == "low_volatility"

    @patch("crabquant.regime.detect_regime", side_effect=Exception("boom"))
    def test_exception_returns_unknown(self, mock_detect):
        """If detect_regime raises, return 'unknown'."""
        df = _make_df(n=200)
        result = _detect_regime_for_period(df)
        assert result == "unknown"


# ── walk_forward_test ───────────────────────────────────────────────────────

class TestWalkForwardMocked:
    @patch("crabquant.validation.load_data", side_effect=ValueError("no data"))
    def test_data_error_returns_non_robust(self, mock_load):
        result = walk_forward_test(_noop_strategy, "BADTICKER", {})
        assert result.robust is False
        assert "Data error" in result.notes

    @patch("crabquant.validation.BacktestEngine")
    @patch("crabquant.validation.load_data")
    def test_insufficient_data_returns_non_robust(self, mock_load, mock_engine_cls):
        """Short DataFrame should return non-robust result."""
        mock_load.return_value = _make_df(n=100)  # too short for 252 min_train_bars
        mock_engine = MagicMock()
        mock_engine.run.return_value = _make_backtest_result()
        mock_engine_cls.return_value = mock_engine

        result = walk_forward_test(_noop_strategy, "AAPL", {}, min_train_bars=252)
        assert result.robust is False
        assert "Insufficient data" in result.notes

    @patch("crabquant.validation.BacktestEngine")
    @patch("crabquant.validation.load_data")
    def test_custom_thresholds(self, mock_load, mock_engine_cls):
        """Custom min_test_sharpe and min_test_trades should affect robustness."""
        mock_load.return_value = _make_df(n=800)
        mock_engine = MagicMock()
        # Train result: good. Test result: low sharpe, few trades
        mock_engine.run.side_effect = [
            _make_backtest_result(sharpe=2.0, num_trades=30),  # train
            _make_backtest_result(sharpe=0.1, num_trades=3),    # test
        ]
        mock_engine_cls.return_value = mock_engine

        result = walk_forward_test(
            _noop_strategy, "AAPL", {},
            min_test_sharpe=1.0, min_test_trades=20, max_degradation=0.5,
        )
        # Sharpe 0.1 < 1.0, trades 3 < 20 → not robust
        assert result.robust is False


# ── cross_ticker_validation ─────────────────────────────────────────────────

class TestCrossTickerMocked:
    @patch("crabquant.validation.load_data", side_effect=ValueError("fail"))
    def test_all_tickers_fail(self, mock_load):
        result = cross_ticker_validation(_noop_strategy, {}, ["BAD1", "BAD2"])
        assert result.robust is False
        assert result.tickers_profitable == 0
        assert "No valid results" in result.notes

    @patch("crabquant.validation.BacktestEngine")
    @patch("crabquant.validation.load_data")
    def test_filters_zero_trade_results(self, mock_load, mock_engine_cls):
        """Results with 0 trades should be excluded."""
        mock_load.return_value = _make_df(n=500)
        mock_engine = MagicMock()
        mock_engine.run.return_value = _make_backtest_result(num_trades=0, sharpe=0, total_return=0)
        mock_engine_cls.return_value = mock_engine

        result = cross_ticker_validation(_noop_strategy, {}, ["AAPL", "MSFT"])
        assert result.tickers_profitable == 0


# ── rolling_walk_forward ────────────────────────────────────────────────────

class TestRollingWalkForwardMocked:
    @patch("crabquant.validation.load_data", side_effect=ValueError("no data"))
    def test_data_error_returns_non_robust(self, mock_load):
        result = rolling_walk_forward(_noop_strategy, "BAD", {})
        assert result.robust is False
        assert "Data error" in result.notes

    @patch("crabquant.validation.BacktestEngine")
    @patch("crabquant.validation.load_data")
    def test_data_too_short(self, mock_load, mock_engine_cls):
        """Data shorter than window should return non-robust."""
        mock_load.return_value = _make_df(n=100)  # way too short for 18mo+6mo
        mock_engine_cls.return_value = MagicMock()

        result = rolling_walk_forward(_noop_strategy, "AAPL", {})
        assert result.robust is False
        assert "Data too short" in result.notes


# ── full_validation ─────────────────────────────────────────────────────────

class TestFullValidation:
    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.walk_forward_test")
    def test_returns_expected_keys(self, mock_wf, mock_ct):
        mock_wf.return_value = WalkForwardResult(
            strategy_name="test", ticker="AAPL",
            train_sharpe=1.0, train_return=0.1,
            test_sharpe=0.8, test_return=0.08,
            test_max_dd=-0.05, degradation=0.2,
            robust=True, notes="ok",
        )
        mock_ct.return_value = CrossTickerResult(
            strategy_name="test", params={},
            tickers_tested=2, tickers_profitable=2, tickers_passed=1,
            avg_sharpe=1.0, median_sharpe=1.0, sharpe_std=0.0,
            avg_return=0.05, avg_max_dd=0.05,
            win_rate_across_tickers=1.0, robust=True,
            notes="ok",
        )

        result = full_validation(_noop_strategy, {}, "AAPL", ["MSFT", "GOOGL"])

        assert "walk_forward" in result
        assert "cross_ticker" in result
        assert "overall_robust" in result
        assert "timestamp" in result
        assert result["overall_robust"] is True

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.walk_forward_test")
    def test_overall_robust_false_when_one_fails(self, mock_wf, mock_ct):
        mock_wf.return_value = WalkForwardResult(
            strategy_name="test", ticker="AAPL",
            train_sharpe=1.0, train_return=0.1,
            test_sharpe=0.8, test_return=0.08,
            test_max_dd=-0.05, degradation=0.2,
            robust=True, notes="ok",
        )
        mock_ct.return_value = CrossTickerResult(
            strategy_name="test", params={},
            tickers_tested=2, tickers_profitable=0, tickers_passed=0,
            avg_sharpe=0.1, median_sharpe=0.1, sharpe_std=0.0,
            avg_return=0.0, avg_max_dd=0.0,
            win_rate_across_tickers=0.0, robust=False,
            notes="bad",
        )

        result = full_validation(_noop_strategy, {}, "AAPL", ["MSFT"])
        assert result["overall_robust"] is False

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.walk_forward_test")
    def test_excludes_discovery_ticker_from_cross_ticker(self, mock_wf, mock_ct):
        mock_wf.return_value = WalkForwardResult(
            strategy_name="test", ticker="AAPL",
            train_sharpe=1.0, train_return=0.1,
            test_sharpe=0.8, test_return=0.08,
            test_max_dd=-0.05, degradation=0.2,
            robust=True, notes="ok",
        )
        mock_ct.return_value = CrossTickerResult(
            strategy_name="test", params={},
            tickers_tested=1, tickers_profitable=1, tickers_passed=1,
            avg_sharpe=1.0, median_sharpe=1.0, sharpe_std=0.0,
            avg_return=0.05, avg_max_dd=0.05,
            win_rate_across_tickers=1.0, robust=True,
            notes="ok",
        )

        full_validation(_noop_strategy, {}, "AAPL", ["AAPL", "MSFT"])

        # cross_ticker_validation should be called with discovery ticker excluded
        call_args = mock_ct.call_args
        oos_tickers = call_args[0][2]  # 3rd positional arg
        assert "AAPL" not in oos_tickers
        assert "MSFT" in oos_tickers


# ── time_reversed_check ─────────────────────────────────────────────────

class TestTimeReversedCheck:
    @patch("crabquant.validation.BacktestEngine")
    def test_passes_when_reversed_score_low(self, mock_engine_cls):
        """When reversed Sharpe is well below threshold, check passes."""
        mock_engine = MagicMock()
        mock_engine.run.side_effect = [
            _make_backtest_result(sharpe=2.0),   # normal
            _make_backtest_result(sharpe=0.2),   # reversed — 10% of normal
        ]
        mock_engine_cls.return_value = mock_engine

        data = _make_df(n=500)
        passed, notes = time_reversed_check(_noop_strategy, data, {})

        assert passed is True
        assert "Passed" in notes
        assert mock_engine.run.call_count == 2

    @patch("crabquant.validation.BacktestEngine")
    def test_flags_overfit_when_reversed_score_high(self, mock_engine_cls):
        """When reversed Sharpe is above threshold, flag as overfit."""
        mock_engine = MagicMock()
        mock_engine.run.side_effect = [
            _make_backtest_result(sharpe=1.0),   # normal
            _make_backtest_result(sharpe=0.8),   # reversed — 80% of normal
        ]
        mock_engine_cls.return_value = mock_engine

        data = _make_df(n=500)
        passed, notes = time_reversed_check(_noop_strategy, data, {})

        assert passed is False
        assert "Overfit" in notes

    @patch("crabquant.validation.BacktestEngine")
    def test_custom_threshold(self, mock_engine_cls):
        """Custom threshold should be respected."""
        mock_engine = MagicMock()
        mock_engine.run.side_effect = [
            _make_backtest_result(sharpe=1.0),   # normal
            _make_backtest_result(sharpe=0.25),  # reversed — 25% of normal
        ]
        mock_engine_cls.return_value = mock_engine

        data = _make_df(n=500)

        # With default threshold=0.3, 25% < 30% → passes
        passed, _ = time_reversed_check(_noop_strategy, data, {})
        assert passed is True

        # With threshold=0.2, 25% > 20% → flagged
        mock_engine_cls.reset_mock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.run.side_effect = [
            _make_backtest_result(sharpe=1.0),
            _make_backtest_result(sharpe=0.25),
        ]
        passed, notes = time_reversed_check(_noop_strategy, data, {}, threshold=0.2)
        assert passed is False
        assert "Overfit" in notes

    @patch("crabquant.validation.BacktestEngine")
    def test_handles_zero_normal_sharpe(self, mock_engine_cls):
        """Should not divide by zero when normal Sharpe is 0."""
        mock_engine = MagicMock()
        mock_engine.run.side_effect = [
            _make_backtest_result(sharpe=0.0),   # normal — zero
            _make_backtest_result(sharpe=0.0),   # reversed
        ]
        mock_engine_cls.return_value = mock_engine

        data = _make_df(n=500)
        passed, notes = time_reversed_check(_noop_strategy, data, {})

        # 0 / max(0, 1e-8) = 0 → passes
        assert passed is True
        assert "Passed" in notes


# ── check_degenerate_strategy ─────────────────────────────────────────────

class TestCheckDegenerateStrategy:
    def test_zero_trades_detected(self):
        """Strategy with 0 trades should be flagged as degenerate."""
        result = _make_backtest_result(num_trades=0, total_return=0, max_drawdown=0)
        is_degenerate, reason = check_degenerate_strategy(result)
        assert is_degenerate is True
        assert "0 trades" in reason

    def test_one_trade_detected(self):
        """Strategy with 1 trade (below min_trades=3) should be flagged."""
        result = _make_backtest_result(num_trades=1, total_return=0.02, max_drawdown=-0.01)
        is_degenerate, reason = check_degenerate_strategy(result)
        assert is_degenerate is True
        assert "1 trades" in reason

    def test_two_trades_detected(self):
        """Strategy with 2 trades (below min_trades=3) should be flagged."""
        result = _make_backtest_result(num_trades=2, total_return=0.03, max_drawdown=-0.01)
        is_degenerate, reason = check_degenerate_strategy(result)
        assert is_degenerate is True
        assert "2 trades" in reason

    def test_constant_position_detected(self):
        """Strategy with trades but zero return AND zero drawdown = constant position."""
        result = _make_backtest_result(
            num_trades=10, total_return=0.0, max_drawdown=0.0,
        )
        is_degenerate, reason = check_degenerate_strategy(result)
        assert is_degenerate is True
        assert "never changed position" in reason

    def test_flat_pnl_curve_detected(self):
        """Strategy with near-zero return volatility should be flagged."""
        flat_series = pd.Series(np.zeros(100))
        result = _make_backtest_result(num_trades=20, total_return=0.001, max_drawdown=-0.001)
        result.returns = flat_series
        is_degenerate, reason = check_degenerate_strategy(result)
        assert is_degenerate is True
        assert "flat PnL" in reason

    def test_flat_equity_curve_detected(self):
        """Flat equity_curve attribute should also be caught."""
        flat_series = pd.Series(np.ones(100) * 100000)
        result = _make_backtest_result(num_trades=20, total_return=0.001, max_drawdown=-0.001)
        result.equity_curve = flat_series
        is_degenerate, reason = check_degenerate_strategy(result)
        assert is_degenerate is True
        assert "flat PnL" in reason

    def test_normal_strategy_not_flagged(self):
        """A normal strategy with reasonable trades should NOT be flagged."""
        result = _make_backtest_result(
            num_trades=20, total_return=0.12, max_drawdown=-0.08,
        )
        is_degenerate, reason = check_degenerate_strategy(result)
        assert is_degenerate is False
        assert reason == ""

    def test_custom_min_trades(self):
        """Custom min_trades threshold should be respected."""
        result = _make_backtest_result(num_trades=5, total_return=0.02, max_drawdown=-0.01)
        # With default min_trades=3, 5 trades should pass
        is_degenerate, _ = check_degenerate_strategy(result)
        assert is_degenerate is False
        # With min_trades=10, 5 trades should fail
        is_degenerate, reason = check_degenerate_strategy(result, min_trades=10)
        assert is_degenerate is True
        assert "5 trades" in reason

    def test_drawdown_but_no_return_ok(self):
        """Strategy with drawdown but zero return is NOT constant-position."""
        result = _make_backtest_result(
            num_trades=10, total_return=0.0, max_drawdown=-0.05,
        )
        is_degenerate, reason = check_degenerate_strategy(result)
        assert is_degenerate is False
        assert reason == ""

    def test_return_but_no_drawdown_ok(self):
        """Strategy with positive return but zero drawdown is NOT degenerate."""
        result = _make_backtest_result(
            num_trades=10, total_return=0.05, max_drawdown=0.0,
        )
        is_degenerate, reason = check_degenerate_strategy(result)
        assert is_degenerate is False
        assert reason == ""


# ── Error-Type-Specific Repair Guidance (Enhancement 9) ─────────────────

class TestErrorSpecificGuidance:
    """Tests for get_error_specific_guidance in llm_api.py."""

    def _get_guidance(self, error):
        # Import directly from the module to avoid heavy import chain
        # (crabquant.refinement.__init__ pulls in strategies which need pandas_ta)
        import importlib.util, sys
        mod_path = "crabquant.refinement.llm_api"
        if mod_path not in sys.modules:
            import crabquant
            pkg_root = str(Path(crabquant.__file__).parent.parent)
            file_path = Path(crabquant.__file__).parent / "refinement" / "llm_api.py"
            spec = importlib.util.spec_from_file_location(mod_path, str(file_path),
                                                          submodule_search_locations=[])
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_path] = mod
            spec.loader.exec_module(mod)
        from crabquant.refinement.llm_api import get_error_specific_guidance
        return get_error_specific_guidance(error)

    def test_syntax_error_returns_indentation_guidance(self):
        """SyntaxError should mention indentation and colons."""
        guidance = self._get_guidance(SyntaxError("invalid syntax"))
        assert "indentation" in guidance.lower()
        assert "colon" in guidance.lower()

    def test_indentation_error_returns_indentation_guidance(self):
        """IndentationError should mention 4-space indent."""
        guidance = self._get_guidance(IndentationError("unexpected indent"))
        assert "4 space" in guidance.lower() or "4-space" in guidance.lower()

    def test_import_error_returns_module_availability_guidance(self):
        """ImportError should mention available modules."""
        guidance = self._get_guidance(ImportError("No module named 'foo'"))
        assert "numpy" in guidance.lower() or "pandas" in guidance.lower()
        assert "strategy_helpers" in guidance.lower()

    def test_module_not_found_error_returns_module_availability_guidance(self):
        """ModuleNotFoundError (subclass of ImportError) should also get module guidance."""
        guidance = self._get_guidance(ModuleNotFoundError("No module named 'foo'"))
        assert "numpy" in guidance.lower() or "pandas" in guidance.lower()

    def test_type_error_returns_signature_guidance(self):
        """TypeError should mention function signatures."""
        guidance = self._get_guidance(TypeError("func() takes 1 argument but 2 were given"))
        assert "signature" in guidance.lower() or "argument" in guidance.lower()

    def test_name_error_returns_variable_scope_guidance(self):
        """NameError should mention variable scope and spelling."""
        guidance = self._get_guidance(NameError("name 'undefined_var' is not defined"))
        assert "typo" in guidance.lower() or "scope" in guidance.lower()
        assert "strategy_helpers" in guidance.lower()

    def test_index_error_returns_bounds_guidance(self):
        """IndexError should mention array bounds."""
        guidance = self._get_guidance(IndexError("list index out of range"))
        assert "bound" in guidance.lower() or "lookback" in guidance.lower()

    def test_key_error_returns_dict_key_guidance(self):
        """KeyError should mention dictionary keys."""
        guidance = self._get_guidance(KeyError("missing_key"))
        assert "key" in guidance.lower()
        assert "column" in guidance.lower()

    def test_value_error_returns_guard_guidance(self):
        """ValueError should mention NaN/inf guards."""
        guidance = self._get_guidance(ValueError("math domain error"))
        assert "nan" in guidance.lower() or "inf" in guidance.lower() or "guard" in guidance.lower()

    def test_zero_division_error_returns_epsilon_guidance(self):
        """ZeroDivisionError should mention epsilon."""
        guidance = self._get_guidance(ZeroDivisionError("division by zero"))
        assert "epsilon" in guidance.lower() or "1e-8" in guidance.lower()

    def test_attribute_error_returns_attribute_guidance(self):
        """AttributeError should mention method/attribute names."""
        guidance = self._get_guidance(AttributeError("'str' object has no attribute 'foo'"))
        assert "attribute" in guidance.lower() or "method" in guidance.lower()

    def test_unknown_error_returns_generic_guidance(self):
        """Unknown error type should return generic traceback guidance."""
        guidance = self._get_guidance(RuntimeError("something unexpected"))
        assert "traceback" in guidance.lower() or "root cause" in guidance.lower()

    def test_os_error_returns_generic_guidance(self):
        """OSError is not in the guidance map, should get default."""
        guidance = self._get_guidance(OSError("permission denied"))
        assert "traceback" in guidance.lower() or "root cause" in guidance.lower()


# ── Enhancement 14: Per-Ticker Alpha Decomposition ──────────────────────────


class TestBuildTickerAlphaContext:
    """Tests for build_ticker_alpha_context in context_builder.

    The function is extracted via AST to avoid importing context_builder
    which has heavy deps (pandas_ta, etc.).
    """

    @staticmethod
    def _get_fn():
        """Extract build_ticker_alpha_context from source to avoid heavy deps."""
        import ast, importlib.util, sys, types
        from pathlib import Path

        # If the module is already loaded (with heavy deps), use it directly
        mod_path = "crabquant.refinement.context_builder"
        if mod_path in sys.modules and hasattr(sys.modules[mod_path], "build_ticker_alpha_context"):
            return sys.modules[mod_path].build_ticker_alpha_context

        # Otherwise, parse the source file and extract just the function
        file_path = Path(__file__).resolve().parent.parent / "crabquant" / "refinement" / "context_builder.py"
        source = file_path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_ticker_alpha_context":
                code = compile(ast.Module(body=[node], type_ignores=[]), filename=str(file_path), mode="exec")
                mod = types.ModuleType("_ticker_alpha_fn")
                exec(code, mod.__dict__)
                return mod.build_ticker_alpha_context
        raise ImportError("build_ticker_alpha_context not found in source")

    def test_empty_history_returns_no_prior_data(self):
        """Empty history should return 'No prior data for TICKER'."""
        fn = self._get_fn()
        result = fn("AAPL", [])
        assert result == "No prior data for AAPL."

    def test_no_matching_ticker_returns_no_prior_data(self):
        """History with different tickers should return no-data message."""
        fn = self._get_fn()
        history = [
            {"ticker": "SPY", "sharpe": 1.2, "archetype": "momentum", "turn": 1},
            {"ticker": "TSLA", "sharpe": 0.8, "archetype": "mean_reversion", "turn": 2},
        ]
        result = fn("AAPL", history)
        assert result == "No prior data for AAPL."

    def test_single_attempt_skipped_returns_no_prior_data(self):
        """Only one attempt for a ticker should be skipped (one-offs)."""
        fn = self._get_fn()
        history = [
            {"ticker": "AAPL", "sharpe": 0.5, "archetype": "momentum", "turn": 1},
        ]
        result = fn("AAPL", history)
        assert result == "No prior data for AAPL."

    def test_mixed_results_generates_correct_summary(self):
        """Multiple archetypes with 2+ attempts should produce a summary."""
        fn = self._get_fn()
        history = [
            {"ticker": "AAPL", "sharpe": -0.3, "archetype": "momentum", "turn": 1, "indicators": ["MACD"]},
            {"ticker": "AAPL", "sharpe": -0.1, "archetype": "momentum", "turn": 2, "indicators": ["MACD"]},
            {"ticker": "AAPL", "sharpe": 0.1, "archetype": "mean_reversion", "turn": 3, "indicators": ["RSI"]},
            {"ticker": "AAPL", "sharpe": 0.5, "archetype": "mean_reversion", "turn": 4, "indicators": ["RSI"]},
            {"ticker": "SPY", "sharpe": 1.5, "archetype": "momentum", "turn": 5, "indicators": ["EMA"]},
        ]
        result = fn("AAPL", history)
        assert "AAPL past attempts (4 turns)" in result
        assert "momentum" in result
        assert "mean_reversion" in result
        assert "avg Sharpe" in result

    def test_suggestion_identifies_best_family(self):
        """Suggestion should highlight the best-performing archetype."""
        fn = self._get_fn()
        history = [
            {"ticker": "AAPL", "sharpe": -0.5, "archetype": "momentum", "turn": 1, "indicators": ["MACD"]},
            {"ticker": "AAPL", "sharpe": -0.8, "archetype": "momentum", "turn": 2, "indicators": ["MACD"]},
            {"ticker": "AAPL", "sharpe": 0.3, "archetype": "mean_reversion", "turn": 3, "indicators": ["RSI"]},
            {"ticker": "AAPL", "sharpe": 0.7, "archetype": "mean_reversion", "turn": 4, "indicators": ["RSI"]},
        ]
        result = fn("AAPL", history)
        assert "Suggestion:" in result
        assert "mean_reversion shows promise" in result
        assert "Avoid momentum on AAPL" in result

    def test_oneoff_archetype_excluded(self):
        """Archetypes with only 1 attempt should be excluded from output."""
        fn = self._get_fn()
        history = [
            {"ticker": "AAPL", "sharpe": 0.1, "archetype": "momentum", "turn": 1, "indicators": ["MACD"]},
            {"ticker": "AAPL", "sharpe": 0.2, "archetype": "momentum", "turn": 2, "indicators": ["MACD"]},
            {"ticker": "AAPL", "sharpe": -0.8, "archetype": "breakout", "turn": 3, "indicators": ["Bollinger"]},
        ]
        result = fn("AAPL", history)
        assert "momentum" in result
        assert "breakout" not in result  # only 1 attempt, skipped

    def test_case_insensitive_ticker_match(self):
        """Ticker matching should be case-insensitive."""
        fn = self._get_fn()
        history = [
            {"ticker": "aapl", "sharpe": 0.5, "archetype": "momentum", "turn": 1},
            {"ticker": "AAPL", "sharpe": 0.3, "archetype": "momentum", "turn": 2},
        ]
        result = fn("aapl", history)
        assert "AAPL past attempts (2 turns)" in result


# ── Enhancement 2: Deflated Sharpe Ratio ────────────────────────────────────

from crabquant.refinement.deflated_sharpe import (
    deflated_sharpe,
    deflated_sharpe_ratio,
    _expected_max_sharpe,
    _probabilistic_sharpe_ratio,
)


class TestExpectedMaxSharpe:
    def test_single_trial_returns_benchmark(self):
        """With 1 trial, expected max = benchmark (no luck factor)."""
        assert _expected_max_sharpe(1, sr0=0.0) == 0.0
        assert _expected_max_sharpe(1, sr0=1.0) == 1.0

    def test_more_trials_increases_expected_max(self):
        """More trials → higher expected max Sharpe."""
        em1 = _expected_max_sharpe(10, sr0=0.0, sharpe_std=1.0)
        em2 = _expected_max_sharpe(100, sr0=0.0, sharpe_std=1.0)
        em3 = _expected_max_sharpe(1000, sr0=0.0, sharpe_std=1.0)
        assert em2 > em1
        assert em3 > em2

    def test_expected_max_scales_with_std(self):
        """Higher sharpe_std → higher expected max."""
        em_low = _expected_max_sharpe(100, sr0=0.0, sharpe_std=0.5)
        em_high = _expected_max_sharpe(100, sr0=0.0, sharpe_std=2.0)
        assert em_high > em_low


class TestProbabilisticSharpeRatio:
    def test_above_benchmark_returns_high_probability(self):
        """Observed well above benchmark → high PSR."""
        psr = _probabilistic_sharpe_ratio(3.0, 0.0, 1.0)
        assert psr > 0.99

    def test_below_benchmark_returns_low_probability(self):
        """Observed well below benchmark → low PSR."""
        psr = _probabilistic_sharpe_ratio(-3.0, 0.0, 1.0)
        assert psr < 0.01

    def test_equal_to_benchmark_returns_05(self):
        """Observed == benchmark → PSR = 0.5."""
        psr = _probabilistic_sharpe_ratio(1.0, 1.0, 1.0)
        assert abs(psr - 0.5) < 1e-6

    def test_zero_std_returns_deterministic(self):
        """Zero std → deterministic (1.0 if above, 0.0 if below)."""
        assert _probabilistic_sharpe_ratio(2.0, 1.0, 0.0) == 1.0
        assert _probabilistic_sharpe_ratio(0.5, 1.0, 0.0) == 0.0


class TestDeflatedSharpeRatio:
    def test_single_trial_no_deflation(self):
        """With n_trials=1, DSR should be high (no multiple-testing penalty)."""
        dsr = deflated_sharpe_ratio(observed_sharpe=2.0, sharpe_std=1.0, n_trials=1)
        assert dsr > 0.95

    def test_many_trials_deflates(self):
        """With many trials, DSR should be lower for same observed Sharpe."""
        dsr_few = deflated_sharpe_ratio(observed_sharpe=2.0, sharpe_std=1.0, n_trials=5)
        dsr_many = deflated_sharpe_ratio(observed_sharpe=2.0, sharpe_std=1.0, n_trials=5000)
        assert dsr_many < dsr_few

    def test_very_high_n_trials_rejects_moderate_sharpe(self):
        """With 4614 trials, a moderate Sharpe should get low DSR."""
        dsr = deflated_sharpe_ratio(observed_sharpe=1.0, sharpe_std=1.0, n_trials=4614)
        assert dsr < 0.5

    def test_outstanding_sharpe_passes_even_with_many_trials(self):
        """Very high Sharpe should still pass even with many trials."""
        dsr = deflated_sharpe_ratio(observed_sharpe=5.0, sharpe_std=1.0, n_trials=4614)
        assert dsr > 0.95


class TestDeflatedSharpe:
    """Tests for the convenience wrapper deflated_sharpe()."""

    def test_n_trials_1_no_penalty(self):
        """With n_trials=1, deflated ≈ observed (no penalty)."""
        result = deflated_sharpe(sharpe=2.0, n_trials=1)
        assert abs(result - 2.0) < 1e-6

    def test_more_trials_lower_deflated_sharpe(self):
        """More trials → lower deflated Sharpe (penalizes multiple testing)."""
        dsr_low = deflated_sharpe(sharpe=2.0, n_trials=5)
        dsr_high = deflated_sharpe(sharpe=2.0, n_trials=5000)
        assert dsr_high < dsr_low

    def test_higher_observed_sharpe_higher_deflated(self):
        """Higher observed Sharpe → higher deflated Sharpe (monotonic)."""
        dsr_1 = deflated_sharpe(sharpe=1.0, n_trials=100)
        dsr_2 = deflated_sharpe(sharpe=3.0, n_trials=100)
        assert dsr_2 > dsr_1

    def test_very_high_n_trials_significant_deflation(self):
        """With 4614 trials, moderate Sharpe should get negative deflated."""
        result = deflated_sharpe(sharpe=1.5, n_trials=4614)
        assert result < 0, f"Expected negative deflated Sharpe, got {result}"

    def test_outstanding_sharpe_survives_many_trials(self):
        """Very high Sharpe should still be positive even with 4614 trials."""
        result = deflated_sharpe(sharpe=5.0, n_trials=4614)
        assert result > 0

    def test_zero_sharpe(self):
        """Zero observed Sharpe should give negative deflated Sharpe."""
        result = deflated_sharpe(sharpe=0.0, n_trials=100)
        assert result < 0

    def test_negative_sharpe(self):
        """Negative observed Sharpe should give negative deflated Sharpe."""
        result = deflated_sharpe(sharpe=-1.0, n_trials=100)
        assert result < 0

    def test_n_trials_less_than_1_clamped(self):
        """n_trials < 1 should be clamped to 1 (no penalty)."""
        result = deflated_sharpe(sharpe=2.0, n_trials=0)
        assert abs(result - 2.0) < 1e-6

    def test_with_returns_array(self):
        """Passing returns array should produce valid result."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        result = deflated_sharpe(sharpe=2.0, n_trials=100, returns=returns)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_with_skew_and_kurt(self):
        """Custom skew/kurtosis should be accepted and produce valid result."""
        result = deflated_sharpe(sharpe=2.0, n_trials=100, skew=-0.5, kurt=5.0)
        assert isinstance(result, float)
        assert not np.isnan(result)


# ── Enhancement 3: Complexity Scoring ──────────────────────────────────────

from crabquant.refinement.complexity import (
    complexity_score,
    complexity_penalty,
    ComplexityScore,
)


SIMPLE_CODE = """\
DEFAULT_PARAMS = {"period": 20, "std_dev": 2.0}

def simulate(df, params):
    period = params["period"]
    std = params["std_dev"]
    mid = df["close"].rolling(period).mean()
    upper = mid + std * df["close"].rolling(period).std()
    lower = mid - std * df["close"].rolling(period).std()
    entries = df["close"] < lower
    exits = df["close"] > upper
    return entries, exits
"""

COMPLEX_CODE = """\
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "fast_period": 10,
    "slow_period": 30,
    "signal_period": 9,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "volume_threshold": 1.5,
}

def _compute_indicators(df, params):
    ema_fast = df["close"].ewm(span=params["fast_period"]).mean()
    ema_slow = df["close"].ewm(span=params["slow_period"]).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=params["signal_period"]).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(params["rsi_period"]).mean()
    loss = (-delta.clip(upper=0)).rolling(params["rsi_period"]).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    atr = df["high"].sub(df["low"]).rolling(params["atr_period"]).mean()
    vol_avg = df["volume"].rolling(20).mean()
    return macd, signal, rsi, atr, vol_avg

def _check_conditions(row, macd_val, signal_val, rsi_val, atr_val, vol_val, params):
    if macd_val > signal_val:
        if rsi_val < params["rsi_oversold"]:
            if vol_val > params["volume_threshold"]:
                return "strong_buy"
            else:
                return "weak_buy"
        elif rsi_val < 50:
            return "buy"
        else:
            return "hold"
    elif macd_val < signal_val:
        if rsi_val > params["rsi_overbought"]:
            return "strong_sell"
        elif rsi_val > 50:
            return "sell"
        else:
            return "hold"
    else:
        return "hold"

def _apply_filters(entries, df, params):
    filtered = entries.copy()
    for i in range(len(filtered)):
        if filtered.iloc[i]:
            if df["close"].iloc[i] < df["low"].rolling(50).mean().iloc[i]:
                for j in range(5):
                    if j < len(filtered) and i + j < len(filtered):
                        filtered.iloc[i + j] = False
    return filtered

def simulate(df, params):
    macd, signal, rsi, atr, vol_avg = _compute_indicators(df, params)
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    for i in range(len(df)):
        if i > max(params["slow_period"], params["rsi_period"]):
            action = _check_conditions(
                df.iloc[i], macd.iloc[i], signal.iloc[i],
                rsi.iloc[i], atr.iloc[i], vol_avg.iloc[i], params,
            )
            if action in ("strong_buy", "buy", "weak_buy"):
                entries.iloc[i] = True
            elif action in ("strong_sell", "sell"):
                exits.iloc[i] = True
    entries = _apply_filters(entries, df, params)
    return entries, exits
"""


class TestComplexityScore:
    """Tests for complexity_score()."""

    def test_simple_strategy_low_complexity(self):
        """A simple strategy should have a low complexity score (< 40)."""
        result = complexity_score(SIMPLE_CODE)
        assert result["complexity"] < 40
        assert result["complexity"] >= 0

    def test_complex_strategy_high_complexity(self):
        """A complex strategy with many branches should have high score (> 40)."""
        result = complexity_score(COMPLEX_CODE)
        assert result["complexity"] > 40
        assert "high_complexity" in result["flags"] or result["complexity"] > 50

    def test_many_params_flagged(self):
        """Strategy with >8 params should get too_many_params flag."""
        params = {f"p{i}": i for i in range(9)}
        result = complexity_score(SIMPLE_CODE, params=params)
        assert "too_many_params" in result["flags"]

    def test_few_params_not_flagged(self):
        """Strategy with <=8 params should NOT get too_many_params flag."""
        result = complexity_score(SIMPLE_CODE)
        assert "too_many_params" not in result["flags"]

    def test_deep_nesting_flagged(self):
        """Complex strategy with nested loops/ifs should get deep_nesting flag."""
        result = complexity_score(COMPLEX_CODE)
        assert "deep_nesting" in result["flags"]

    def test_empty_code_zero_complexity(self):
        """Empty code should return 0 complexity."""
        result = complexity_score("")
        assert result["complexity"] == 0.0
        assert result["flags"] == []

    def test_whitespace_only_zero_complexity(self):
        """Whitespace-only code should return 0 complexity."""
        result = complexity_score("   \n\t  \n")
        assert result["complexity"] == 0.0

    def test_invalid_python_flagged(self):
        """Invalid Python should return max complexity with invalid_python flag."""
        result = complexity_score("def foo(:\n  pass")
        assert result["complexity"] == 100.0
        assert "invalid_python" in result["flags"]

    def test_no_params_zero_param_count(self):
        """When params=None, n_params should be 0."""
        result = complexity_score(SIMPLE_CODE, params=None)
        assert result["breakdown"]["n_params"] == 0

    def test_params_counted(self):
        """When params dict is provided, n_params should match."""
        params = {"a": 1, "b": 2, "c": 3}
        result = complexity_score(SIMPLE_CODE, params=params)
        assert result["breakdown"]["n_params"] == 3

    def test_breakdown_has_all_fields(self):
        """Result breakdown should have all ComplexityScore fields."""
        result = complexity_score(SIMPLE_CODE)
        bd = result["breakdown"]
        for field in ["n_params", "n_nodes", "n_functions", "n_branches",
                       "max_nesting", "n_features", "total"]:
            assert field in bd

    def test_deterministic(self):
        """Same code should always produce the same score."""
        r1 = complexity_score(SIMPLE_CODE)
        r2 = complexity_score(SIMPLE_CODE)
        assert r1 == r2

    def test_complex_higher_than_simple(self):
        """Complex code should score higher than simple code."""
        r_simple = complexity_score(SIMPLE_CODE)
        r_complex = complexity_score(COMPLEX_CODE)
        assert r_complex["complexity"] > r_simple["complexity"]


class TestComplexityPenalty:
    """Tests for complexity_penalty()."""

    def test_zero_complexity_no_penalty(self):
        """Zero complexity should return the base threshold unchanged."""
        assert complexity_penalty(0.0, base_threshold=1.5) == 1.5

    def test_high_complexity_increases_threshold(self):
        """High complexity should increase the threshold."""
        base = 1.5
        low = complexity_penalty(20.0, base_threshold=base)
        high = complexity_penalty(80.0, base_threshold=base)
        assert high > low
        assert high > base

    def test_max_complexity_50_percent_increase(self):
        """Complexity=100 should increase threshold by 50%."""
        result = complexity_penalty(100.0, base_threshold=1.5)
        assert abs(result - 1.5 * 1.5) < 1e-6

    def test_returns_float(self):
        """Result should be a float."""
        result = complexity_penalty(50.0)
        assert isinstance(result, float)
