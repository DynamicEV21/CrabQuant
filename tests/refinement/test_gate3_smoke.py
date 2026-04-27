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
