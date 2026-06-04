"""Tests for crabquant.confirm.runner — confirmation backtest runner."""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from crabquant.confirm import ConfirmationResult, CONFIRM_THRESHOLDS
from crabquant.confirm.runner import (
    _slippage_commission,
    _compute_profit_factor,
    _compute_expectancy,
    run_confirmation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trades_df(pnl_col="PnL", values=None):
    """Build a minimal trades DataFrame with a PnL column."""
    if values is None:
        values = [100.0, -50.0, 200.0, -30.0]
    return pd.DataFrame({pnl_col: values})


def _make_stats_mock(
    sharpe=1.5,
    total_return=15.0,
    max_dd=-5.0,
    num_trades=30,
    win_rate=60.0,
    trades_df=None,
):
    """Build a MagicMock that behaves like a backtesting.py stats dict."""
    stats = MagicMock()
    data = {
        "Sharpe Ratio": sharpe,
        "Return [%]": total_return,
        "Max. Drawdown [%]": max_dd,
        "# Trades": num_trades,
        "Win Rate [%]": win_rate,
        "_trades": trades_df,
    }
    stats.get.side_effect = lambda k, d=None: data.get(k, d)
    return stats


def _make_ohlcv_df(n=200, seed=42):
    """Create a lowercase OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close + rng.randn(n) * 0.3,
            "high": close + abs(rng.randn(n)),
            "low": close - abs(rng.randn(n)),
            "close": close,
            "volume": rng.randint(1_000, 100_000, n).astype(float),
        },
        index=dates,
    )


# ===========================================================================
# _slippage_commission
# ===========================================================================


class TestSlippageCommission:
    """Tests for _slippage_commission(size, price, _slippage_pct)."""

    def test_entry_with_slippage(self):
        """Entry (size > 0) should return base + slippage cost."""
        result = _slippage_commission(size=10, price=100.0, _slippage_pct=0.001)
        base = 10 * 100.0 * 0.001  # 1.0
        slip = 10 * 100.0 * 0.001  # 1.0
        assert result == pytest.approx(base + slip)

    def test_exit_no_slippage(self):
        """Exit (size < 0) should return base commission only."""
        result = _slippage_commission(size=-10, price=100.0, _slippage_pct=0.001)
        base = 10 * 100.0 * 0.001  # 1.0
        assert result == pytest.approx(base)

    def test_zero_size(self):
        """Zero size should return zero commission."""
        result = _slippage_commission(size=0, price=100.0, _slippage_pct=0.001)
        assert result == 0.0

    def test_zero_slippage(self):
        """Zero slippage should give base commission only on entry."""
        result = _slippage_commission(size=10, price=50.0, _slippage_pct=0.0)
        base = 10 * 50.0 * 0.001  # 0.5
        assert result == pytest.approx(base)

    def test_large_size(self):
        """Large position size scales linearly."""
        result = _slippage_commission(size=1000, price=200.0, _slippage_pct=0.0005)
        base = 1000 * 200.0 * 0.001  # 200
        slip = 1000 * 200.0 * 0.0005  # 100
        assert result == pytest.approx(base + slip)

    def test_high_slippage(self):
        """High slippage environment."""
        result = _slippage_commission(size=5, price=50.0, _slippage_pct=0.01)
        base = 5 * 50.0 * 0.001  # 0.25
        slip = 5 * 50.0 * 0.01  # 2.5
        assert result == pytest.approx(base + slip)

    def test_exit_negative_size_uses_abs(self):
        """Exit with negative size should use abs(size) for base."""
        result = _slippage_commission(size=-5, price=100.0, _slippage_pct=0.002)
        base = 5 * 100.0 * 0.001  # 0.5
        assert result == pytest.approx(base)

    def test_entry_fractional_size(self):
        """Fractional shares on entry."""
        result = _slippage_commission(size=0.5, price=1000.0, _slippage_pct=0.001)
        base = 0.5 * 1000.0 * 0.001  # 0.5
        slip = 0.5 * 1000.0 * 0.001  # 0.5
        assert result == pytest.approx(base + slip)

    def test_zero_price(self):
        """Zero price returns zero commission."""
        result = _slippage_commission(size=10, price=0.0, _slippage_pct=0.001)
        assert result == 0.0


# ===========================================================================
# _compute_profit_factor
# ===========================================================================


class TestComputeProfitFactor:
    """Tests for _compute_profit_factor(trades_df)."""

    def test_mixed_wins_losses_pnl(self):
        """Mixed winning and losing trades with 'PnL' column."""
        df = _make_trades_df("PnL", [100, -50, 200, -30])
        # wins = 300, losses = 80, pf = 3.75
        assert _compute_profit_factor(df) == pytest.approx(300.0 / 80.0)

    def test_pnl_lowercase(self):
        """Recognize 'pnl' (lowercase) column."""
        df = _make_trades_df("pnl", [100, -50])
        assert _compute_profit_factor(df) == pytest.approx(100.0 / 50.0)

    def test_profit_column(self):
        """Recognize 'Profit' column."""
        df = _make_trades_df("Profit", [80, -20])
        assert _compute_profit_factor(df) == pytest.approx(80.0 / 20.0)

    def test_all_wins(self):
        """All winning trades → infinity."""
        df = _make_trades_df("PnL", [50, 100, 30])
        assert _compute_profit_factor(df) == float("inf")

    def test_all_losses(self):
        """All losing trades → 0."""
        df = _make_trades_df("PnL", [-10, -20, -30])
        assert _compute_profit_factor(df) == 0.0

    def test_none_input(self):
        """None input → 0.0."""
        assert _compute_profit_factor(None) == 0.0

    def test_empty_dataframe(self):
        """Empty DataFrame → 0.0."""
        df = pd.DataFrame(columns=["PnL"])
        assert _compute_profit_factor(df) == 0.0

    def test_missing_pnl_column(self):
        """DataFrame without any PnL column → 0.0."""
        df = pd.DataFrame({"foo": [1, 2, 3]})
        assert _compute_profit_factor(df) == 0.0

    def test_single_zero_trade(self):
        """Single trade with PnL = 0 → wins=0, losses=0 → 0.0."""
        df = _make_trades_df("PnL", [0])
        assert _compute_profit_factor(df) == 0.0

    def test_wins_with_zero_loss_trades(self):
        """Wins and zero PnL trades → inf if total losses == 0."""
        df = _make_trades_df("PnL", [50, 0, 30])
        assert _compute_profit_factor(df) == float("inf")

    def test_equal_wins_and_losses(self):
        """Equal total wins and losses → 1.0."""
        df = _make_trades_df("PnL", [100, -100])
        assert _compute_profit_factor(df) == pytest.approx(1.0)

    def test_column_priority_pnl_over_pnl_lowercase(self):
        """'PnL' takes priority over 'pnl' when both present."""
        df = pd.DataFrame({"PnL": [100, -50], "pnl": [10, -5]})
        # Should use 'PnL': 100/50 = 2.0
        assert _compute_profit_factor(df) == pytest.approx(2.0)

    def test_large_numbers(self):
        """Large PnL values to test numeric stability."""
        df = _make_trades_df("PnL", [1e8, -5e7])
        assert _compute_profit_factor(df) == pytest.approx(2.0)


# ===========================================================================
# _compute_expectancy
# ===========================================================================


class TestComputeExpectancy:
    """Tests for _compute_expectancy(trades_df)."""

    def test_basic_average(self):
        """Simple average of PnL values."""
        df = _make_trades_df("PnL", [100, -50, 200, -30])
        assert _compute_expectancy(df) == pytest.approx((100 - 50 + 200 - 30) / 4)

    def test_pnl_lowercase(self):
        """Recognize 'pnl' column."""
        df = _make_trades_df("pnl", [10, 20])
        assert _compute_expectancy(df) == pytest.approx(15.0)

    def test_profit_column(self):
        """Recognize 'Profit' column."""
        df = _make_trades_df("Profit", [10, 20, 30])
        assert _compute_expectancy(df) == pytest.approx(20.0)

    def test_none_input(self):
        """None → 0.0."""
        assert _compute_expectancy(None) == 0.0

    def test_empty_dataframe(self):
        """Empty DataFrame → 0.0."""
        df = pd.DataFrame(columns=["PnL"])
        assert _compute_expectancy(df) == 0.0

    def test_missing_pnl_column(self):
        """No PnL column → 0.0."""
        df = pd.DataFrame({"foo": [1, 2, 3]})
        assert _compute_expectancy(df) == 0.0

    def test_single_trade(self):
        """Single trade returns its own value."""
        df = _make_trades_df("PnL", [42.0])
        assert _compute_expectancy(df) == pytest.approx(42.0)

    def test_all_negative(self):
        """All negative PnL → negative expectancy."""
        df = _make_trades_df("PnL", [-10, -20, -30])
        assert _compute_expectancy(df) == pytest.approx(-20.0)

    def test_column_priority(self):
        """'PnL' priority over 'pnl'."""
        df = pd.DataFrame({"PnL": [100, 200], "pnl": [1, 2]})
        assert _compute_expectancy(df) == pytest.approx(150.0)

    def test_large_dataframe(self):
        """Many trades."""
        values = list(range(1, 101))  # 1..100
        df = _make_trades_df("PnL", values)
        assert _compute_expectancy(df) == pytest.approx(50.5)


# ===========================================================================
# run_confirmation
# ===========================================================================


class TestRunConfirmationLoadData:
    """Tests for run_confirmation — data loading failures."""

    @patch("crabquant.confirm.runner.load_data")
    def test_load_data_failure(self, mock_load):
        """Failed load_data returns failed ConfirmationResult."""
        mock_load.side_effect = Exception("Network error")
        result = run_confirmation("sma_cross", "AAPL", {})
        assert result.passed is False
        assert any("Failed to load data" in n for n in result.notes)

    @patch("crabquant.confirm.runner.load_data")
    def test_load_data_returns_none(self, mock_load):
        """load_data returns None — code will raise TypeError on len(None)."""
        mock_load.return_value = None
        # The code does len(df) < 100 which raises TypeError for None;
        # it's not caught by the try/except (only load_data call is in try),
        # so we expect an unhandled exception
        with pytest.raises(TypeError):
            run_confirmation("sma_cross", "AAPL", {})

    @patch("crabquant.confirm.runner.load_data")
    def test_load_data_insufficient_bars(self, mock_load):
        """DataFrame with < 100 rows returns insufficient data note."""
        mock_load.return_value = _make_ohlcv_df(n=50)
        result = run_confirmation("sma_cross", "AAPL", {})
        assert result.passed is False
        assert any("Insufficient data" in n for n in result.notes)
        assert "50 bars" in str(result.notes)

    @patch("crabquant.confirm.runner.load_data")
    def test_load_data_exactly_100_bars(self, mock_load):
        """Exactly 100 bars should pass the data check (not insufficient)."""
        mock_load.return_value = _make_ohlcv_df(n=100)
        # This will proceed past the data check; may fail at convert_strategy
        result = run_confirmation("sma_cross", "AAPL", {})
        # Should NOT have insufficient data note
        assert not any("Insufficient data" in n for n in result.notes)


class TestRunConfirmationConvertStrategy:
    """Tests for run_confirmation — strategy conversion."""

    @patch("crabquant.confirm.runner.convert_strategy")
    def test_convert_strategy_value_error(self, mock_convert):
        """convert_strategy raises ValueError returns failed result."""
        mock_convert.side_effect = ValueError("Unknown strategy: foo_bar")
        result = run_confirmation("foo_bar", "AAPL", {}, df=_make_ohlcv_df(n=200))
        assert result.passed is False
        assert any("Unknown strategy" in n for n in result.notes)


class TestRunConfirmationBacktestException:
    """Tests for run_confirmation — backtest execution failure."""

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_backtest_raises_exception(self, mock_convert, mock_bt_cls):
        """Exception during backtest returns failed result."""
        mock_convert.return_value = MagicMock()
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.side_effect = RuntimeError("Backtest engine error")
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        assert result.passed is False
        assert any("Backtest failed" in n for n in result.notes)


class TestRunConfirmationSuccess:
    """Tests for run_confirmation — successful backtest."""

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_successful_backtest_passes(self, mock_convert, mock_bt_cls):
        """Successful backtest with good stats → passed=True."""
        mock_convert.return_value = MagicMock()
        trades_df = _make_trades_df("PnL", [100, -50, 200, -30])
        stats = _make_stats_mock(
            sharpe=1.5,
            total_return=15.0,
            max_dd=-5.0,
            num_trades=30,
            win_rate=60.0,
            trades_df=trades_df,
        )

        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = stats
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        assert result.passed is True
        assert result.verdict == "PASSED"
        assert result.sharpe == pytest.approx(1.5, abs=0.01)
        assert result.total_return == pytest.approx(0.15, abs=0.01)
        assert result.max_dd == pytest.approx(-0.05, abs=0.01)
        assert result.trades == 30
        assert result.win_rate == pytest.approx(0.60, abs=0.01)

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_backtest_fails_thresholds(self, mock_convert, mock_bt_cls):
        """Backtest with poor stats → passed=False."""
        mock_convert.return_value = MagicMock()
        trades_df = _make_trades_df("PnL", [-100, -200])
        stats = _make_stats_mock(
            sharpe=0.3,
            total_return=-10.0,
            max_dd=-50.0,
            num_trades=2,
            win_rate=10.0,
            trades_df=trades_df,
        )

        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = stats
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        assert result.passed is False
        assert result.verdict == "FAILED"

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_profit_factor_and_expectancy_extracted(self, mock_convert, mock_bt_cls):
        """Profit factor and expectancy are computed from trades."""
        mock_convert.return_value = MagicMock()
        trades_df = _make_trades_df("PnL", [100, -50, 200, -30])
        stats = _make_stats_mock(trades_df=trades_df)

        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = stats
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        # wins=300, losses=80, pf=3.75
        assert result.profit_factor == pytest.approx(3.75, abs=0.01)
        # expectancy = (100-50+200-30)/4 = 55
        assert result.expectancy == pytest.approx(55.0, abs=0.01)

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_no_trades_dataframe(self, mock_convert, mock_bt_cls):
        """When stats has no _trades, pf=0 and exp=0."""
        mock_convert.return_value = MagicMock()
        stats = _make_stats_mock(trades_df=None)

        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = stats
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        assert result.profit_factor == 0.0
        assert result.expectancy == 0.0

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_backtest_called_with_correct_args(self, mock_convert, mock_bt_cls):
        """Verify Backtest is instantiated with expected parameters."""
        mock_convert.return_value = MagicMock()
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = _make_stats_mock()
        mock_bt_cls.return_value = mock_bt_instance

        df = _make_ohlcv_df(n=200)
        run_confirmation(
            "sma_cross", "AAPL", {"period": 20},
            df=df, cash=5000, slippage_pct=0.002, position_pct=0.8,
        )

        # Check Backtest was called once
        mock_bt_cls.assert_called_once()
        call_args, call_kwargs = mock_bt_cls.call_args
        assert call_kwargs["cash"] == 5000
        assert call_kwargs["exclusive_orders"] is True
        assert call_kwargs["hedging"] is False
        assert call_kwargs["finalize_trades"] is True
        assert call_args[1] is mock_convert.return_value  # strategy_class is 2nd positional

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_df_columns_renamed_to_capitalized(self, mock_convert, mock_bt_cls):
        """Input lowercase columns should be renamed to Backtest format."""
        mock_convert.return_value = MagicMock()
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = _make_stats_mock()
        mock_bt_cls.return_value = mock_bt_instance

        df = _make_ohlcv_df(n=200)
        run_confirmation("sma_cross", "AAPL", {}, df=df)

        # The DataFrame passed to Backtest should have capitalized columns
        call_args = mock_bt_cls.call_args[0]
        bt_df = call_args[0]
        assert "Open" in bt_df.columns
        assert "High" in bt_df.columns
        assert "Low" in bt_df.columns
        assert "Close" in bt_df.columns
        assert "Volume" in bt_df.columns

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_notes_contain_check_marks(self, mock_convert, mock_bt_cls):
        """Notes should contain ✓/✗ for each threshold check."""
        mock_convert.return_value = MagicMock()
        stats = _make_stats_mock(
            sharpe=1.5, total_return=15.0, max_dd=-5.0,
            num_trades=30, win_rate=60.0,
            trades_df=_make_trades_df("PnL", [100, -50]),
        )
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = stats
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        check_notes = [n for n in result.notes if n.startswith("✓") or n.startswith("✗")]
        assert len(check_notes) == 5  # sharpe, max_dd, return, trades, expectancy

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_preloaded_df_not_overridden(self, mock_convert, mock_bt_cls):
        """When df is provided, load_data should NOT be called."""
        mock_convert.return_value = MagicMock()
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = _make_stats_mock()
        mock_bt_cls.return_value = mock_bt_instance

        with patch("crabquant.confirm.runner.load_data") as mock_load:
            df = _make_ohlcv_df(n=200)
            run_confirmation("sma_cross", "AAPL", {}, df=df)
            mock_load.assert_not_called()


class TestRunConfirmationEdgeCases:
    """Edge cases for run_confirmation."""

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_stats_with_none_values(self, mock_convert, mock_bt_cls):
        """Stats returning None for some keys should not crash."""
        mock_convert.return_value = MagicMock()

        stats = MagicMock()
        stats.get.side_effect = lambda k, d=None: {
            "Sharpe Ratio": None,
            "Return [%]": None,
            "Max. Drawdown [%]": None,
            "# Trades": None,
            "Win Rate [%]": None,
            "_trades": None,
        }.get(k, d)
        # Also handle 'or 0.0' — None or 0.0 = 0.0

        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = stats
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        assert result.sharpe == 0.0
        assert result.total_return == 0.0
        assert result.max_dd == 0.0
        assert result.trades == 0
        assert result.win_rate == 0.0

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_convert_strategy_called_with_params(self, mock_convert, mock_bt_cls):
        """Verify convert_strategy receives correct arguments."""
        mock_convert.return_value = MagicMock()
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = _make_stats_mock()
        mock_bt_cls.return_value = mock_bt_instance

        run_confirmation(
            "sma_cross", "AAPL", {"fast": 10, "slow": 30},
            df=_make_ohlcv_df(n=200),
            position_pct=0.8,
            slippage_pct=0.003,
        )

        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["position_pct"] == 0.8
        assert call_kwargs["slippage_pct"] == 0.003

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_tz_aware_index_stripped(self, mock_convert, mock_bt_cls):
        """Timezone-aware DatetimeIndex should have tz stripped."""
        mock_convert.return_value = MagicMock()
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = _make_stats_mock()
        mock_bt_cls.return_value = mock_bt_instance

        df = _make_ohlcv_df(n=200)
        df.index = df.index.tz_localize("UTC")

        run_confirmation("sma_cross", "AAPL", {}, df=df)

        bt_df = mock_bt_cls.call_args[0][0]
        assert bt_df.index.tz is None

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_return_value_is_confirmation_result(self, mock_convert, mock_bt_cls):
        """Result should be a ConfirmationResult dataclass."""
        mock_convert.return_value = MagicMock()
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = _make_stats_mock()
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        assert isinstance(result, ConfirmationResult)

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_result_to_dict_works(self, mock_convert, mock_bt_cls):
        """Result.to_dict() should produce a valid dict."""
        mock_convert.return_value = MagicMock()
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = _make_stats_mock()
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "sharpe" in d
        assert "passed" in d
        assert "notes" in d

    @patch("crabquant.confirm.runner.load_data")
    def test_default_period_used(self, mock_load):
        """When df=None and period not specified, default '2y' is used."""
        mock_load.return_value = _make_ohlcv_df(n=50)
        run_confirmation("sma_cross", "AAPL", {})
        mock_load.assert_called_once_with("AAPL", period="2y")

    @patch("crabquant.confirm.runner.load_data")
    def test_custom_period_used(self, mock_load):
        """Custom period is passed to load_data."""
        mock_load.return_value = _make_ohlcv_df(n=50)
        run_confirmation("sma_cross", "AAPL", {}, period="1y")
        mock_load.assert_called_once_with("AAPL", period="1y")

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_trades_int_is_int_type(self, mock_convert, mock_bt_cls):
        """Result.trades should be int even if stats returns float."""
        mock_convert.return_value = MagicMock()
        stats = _make_stats_mock(num_trades=30.0)
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = stats
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        assert isinstance(result.trades, int)

    @patch("crabquant.confirm.runner.Backtest")
    @patch("crabquant.confirm.runner.convert_strategy")
    def test_inf_profit_factor_rounded(self, mock_convert, mock_bt_cls):
        """Infinite profit factor should be handled (all wins)."""
        mock_convert.return_value = MagicMock()
        trades_df = _make_trades_df("PnL", [100, 200, 50])
        stats = _make_stats_mock(
            trades_df=trades_df,
        )
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = stats
        mock_bt_cls.return_value = mock_bt_instance

        result = run_confirmation("sma_cross", "AAPL", {}, df=_make_ohlcv_df(n=200))
        assert result.profit_factor == float("inf")
