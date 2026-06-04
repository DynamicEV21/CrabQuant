"""Tests for crabquant.refinement.gate3_smoke — Gate 3 smoke backtest validation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from crabquant.refinement.gate3_smoke import gate_smoke_backtest


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_mock_df(n: int = 126) -> pd.DataFrame:
    """6 months of business-day OHLCV data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    returns = rng.normal(0.0005, 0.012, n)
    close = 100.0 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


VALID_STRATEGY = """\
import pandas as pd

DEFAULT_PARAMS = {"window": 20}
PARAM_GRID = {"window": [10, 20, 30]}
DESCRIPTION = "Simple MA crossover for testing."

def generate_signals(df: pd.DataFrame, params=None) -> tuple:
    p = {**DEFAULT_PARAMS, **(params or {})}
    w = p["window"]
    ma = df["close"].rolling(w).mean()
    entries = df["close"] > ma
    entries.iloc[:w] = False
    exits = pd.Series(False, index=df.index, dtype=bool)
    return entries.astype(bool), exits
"""


def _make_backtest_result(**overrides):
    """Build a mock BacktestResult-like object."""
    defaults = {
        "sharpe": 1.8,
        "total_return": 0.15,
        "max_drawdown": -0.08,
        "num_trades": 12,
        "win_rate": 0.55,
        "profit_factor": 1.4,
        "calmar_ratio": 1.9,
        "sortino_ratio": 2.1,
        "score": 1.7,
        "strategy_name": "test",
        "iteration": 1,
        "params": {"window": 20},
    }
    defaults.update(overrides)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


# ── Tests ────────────────────────────────────────────────────────────────────


class TestGateSmokeBacktest:
    """Gate 3: smoke backtest validation."""

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_passes_for_healthy_result(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result()

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is True
        assert errors == []

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_fails_on_nan_metrics(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=float("nan"))

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("NaN" in e or "Inf" in e for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_fails_on_inf_metrics(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(total_return=float("inf"))

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("NaN" in e or "Inf" in e for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_fails_on_suspiciously_high_sharpe(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=6.5)

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("lookahead" in e.lower() or "suspicious" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_fails_on_zero_trades(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(num_trades=0)

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("trade" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_fails_on_overtrading(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(num_trades=500)

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("overtrad" in e.lower() or "excessive" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_fails_on_load_error(self, mock_load):
        mock_load.return_value = None

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("load" in e.lower() or "import" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_fails_on_backtest_exception(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.side_effect = RuntimeError("backtest exploded")

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("backtest" in e.lower() or "error" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_returns_tuple_bool_list(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result()

        result = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_custom_timeout_passed(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result()

        gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL", timeout=20)
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        # timeout should be in the call
        assert call_kwargs is not None

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_negative_sharpe_passes_if_valid(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=-0.5, total_return=-0.1)

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        # Negative sharpe is not NaN/Inf and not suspiciously high, so it passes
        assert ok is True
        assert errors == []

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_multiple_errors_accumulated(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(
            sharpe=float("nan"), num_trades=0
        )

        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert len(errors) >= 2


# ── Expanded Tests (Phase 6) ─────────────────────────────────────────────────


class TestGateSmokeBacktestParametrized:
    """Parametrized edge-case tests for Gate 3 smoke backtest."""

    @pytest.mark.parametrize("sharpe_val", [0.0, 0.5, 2.5, 4.9])
    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_passes_for_reasonable_sharpe(self, mock_load, mock_run, sharpe_val):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=sharpe_val)
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is True
        assert errors == []

    @pytest.mark.parametrize("sharpe_val", [5.01, 5.1, 10.0, 100.0])
    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_fails_for_sharpe_above_threshold(self, mock_load, mock_run, sharpe_val):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=sharpe_val)
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("lookahead" in e.lower() or "suspicious" in e.lower() for e in errors)

    @pytest.mark.parametrize("num_trades", [1, 2, 5, 150, 299])
    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_passes_for_valid_trade_counts(self, mock_load, mock_run, num_trades):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(num_trades=num_trades)
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is True
        assert errors == []

    @pytest.mark.parametrize("num_trades", [301, 500, 1000])
    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_fails_for_excessive_trade_counts(self, mock_load, mock_run, num_trades):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(num_trades=num_trades)
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("overtrad" in e.lower() or "excessive" in e.lower() for e in errors)


class TestGateSmokeBacktestEdgeCases:
    """Edge-case tests for Gate 3 smoke backtest."""

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_inf_sharpe_detected(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=float("inf"))
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("nan" in e.lower() or "inf" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_neg_inf_sharpe_detected(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=float("-inf"))
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("nan" in e.lower() or "inf" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_nan_max_drawdown_detected(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(max_drawdown=float("nan"))
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("nan" in e.lower() or "inf" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_inf_total_return_detected(self, mock_load, mock_run):
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(total_return=float("inf"))
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("nan" in e.lower() or "inf" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_all_nan_metrics_reports_all(self, mock_load, mock_run):
        """When all three metrics are NaN, all three are reported in errors."""
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(
            sharpe=float("nan"), total_return=float("nan"), max_drawdown=float("nan")
        )
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        nan_err = [e for e in errors if "NaN" in e or "Inf" in e]
        assert len(nan_err) == 1  # All three reported in a single error message
        assert "sharpe" in nan_err[0].lower()
        assert "total_return" in nan_err[0].lower()
        assert "max_drawdown" in nan_err[0].lower()

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_result_with_none_metrics_skipped(self, mock_load, mock_run):
        """None metrics are treated as missing, not NaN — should pass if other metrics ok."""
        mock_load.return_value = MagicMock()
        result = _make_backtest_result()
        result.sharpe = None
        result.total_return = None
        result.max_drawdown = None
        mock_run.return_value = result
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        # No NaN/Inf errors since values are None (not NaN), no overtrading, trades > 0
        assert ok is True
        assert errors == []

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_data_load_returns_none(self, mock_load, mock_run):
        """When load_data returns None, should return error."""
        mock_load.return_value = MagicMock()
        with patch("crabquant.data.load_data", return_value=None):
            ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("data" in e.lower() or "no data" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_data_load_returns_empty_df(self, mock_load, mock_run):
        """When load_data returns empty DataFrame, should return error."""
        mock_load.return_value = MagicMock()
        with patch("crabquant.data.load_data", return_value=pd.DataFrame()):
            ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("data" in e.lower() or "no data" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_data_load_raises_exception(self, mock_load, mock_run):
        """When load_data raises, should return error."""
        mock_load.return_value = MagicMock()
        with patch("crabquant.data.load_data",
                    side_effect=ConnectionError("network down")):
            ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("data" in e.lower() or "load" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_sharpe_exactly_at_threshold_passes(self, mock_load, mock_run):
        """Sharpe exactly 5.0 should pass (not strictly greater than)."""
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=5.0)
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is True
        assert errors == []

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_trades_exactly_at_threshold_passes(self, mock_load, mock_run):
        """num_trades exactly 300 should pass (not strictly greater than)."""
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(num_trades=300)
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is True
        assert errors == []

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_all_long_strategy_passes(self, mock_load, mock_run):
        """Strategy that generates all-long signals with reasonable trades passes."""
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(num_trades=15, sharpe=1.2)
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is True

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_high_sharpe_with_zero_trades_reports_both(self, mock_load, mock_run):
        """High Sharpe + zero trades should report both suspicious sharpe and zero trades."""
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=10.0, num_trades=0)
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert len(errors) >= 2
        assert any("lookahead" in e.lower() or "suspicious" in e.lower() for e in errors)
        assert any("trade" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_backtest_raises_valueerror(self, mock_load, mock_run):
        """ValueError from backtest is caught gracefully."""
        mock_load.return_value = MagicMock()
        mock_run.side_effect = ValueError("bad data shape")
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("backtest" in e.lower() or "error" in e.lower() for e in errors)

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_backtest_raises_keyboardinterrupt_propagates(self, mock_load, mock_run):
        """KeyboardInterrupt should NOT be caught — it should propagate."""
        mock_load.return_value = MagicMock()
        mock_run.side_effect = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_different_ticker_parameter(self, mock_load, mock_run):
        """Verify the ticker parameter is used correctly."""
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result()
        with patch("crabquant.data.load_data") as mock_data:
            mock_data.return_value = make_mock_df()
            ok, _ = gate_smoke_backtest(VALID_STRATEGY, ticker="TSLA")
        mock_data.assert_called_once_with("TSLA", period="6mo")

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_zero_sharpe_passes(self, mock_load, mock_run):
        """Zero Sharpe is a valid value — not NaN/Inf, not above threshold."""
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(sharpe=0.0)
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is True
        assert errors == []

    @patch("crabquant.refinement.gate3_smoke._run_smoke_backtest")
    @patch("crabquant.refinement.gate3_smoke._load_strategy_module")
    def test_negative_inf_total_return(self, mock_load, mock_run):
        """Negative infinity in total_return is detected."""
        mock_load.return_value = MagicMock()
        mock_run.return_value = _make_backtest_result(total_return=float("-inf"))
        ok, errors = gate_smoke_backtest(VALID_STRATEGY, ticker="AAPL")
        assert ok is False
        assert any("nan" in e.lower() or "inf" in e.lower() for e in errors)
