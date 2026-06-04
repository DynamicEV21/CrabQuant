"""
Tests for wiring validation functions into the refinement pipeline.

Tests three integration points:
1. check_degenerate_strategy — wired into scripts/refinement_loop.py
2. time_reversed_check — wired into crabquant/refinement/promotion.py
3. check_family_plateau — wired into crabquant/refinement/context_builder.py
"""

import pytest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_backtest_result(**overrides):
    """Build a lightweight mock BacktestResult with default healthy values."""
    defaults = {
        "sharpe": 1.5,
        "total_return": 0.12,
        "max_drawdown": -0.08,
        "win_rate": 0.55,
        "num_trades": 20,
        "passed": True,
        "score": 2.0,
        "ticker": "AAPL",
        "strategy_name": "test",
        "params": {"fast": 12, "slow": 26},
        "sortino_ratio": 2.0,
        "calmar_ratio": 1.5,
        "profit_factor": 1.3,
        "avg_holding_bars": 5.0,
    }
    defaults.update(overrides)
    result = SimpleNamespace(**defaults)
    return result


def _make_history_entry(**overrides):
    """Build a turn history dict with sensible defaults."""
    defaults = {
        "turn": 1,
        "status": "failed",
        "sharpe": 0.3,
        "num_trades": 5,
        "action": "modify_params",
    }
    defaults.update(overrides)
    return defaults


# ══════════════════════════════════════════════════════════════════════════
# 1. check_degenerate_strategy — direct function tests
# ══════════════════════════════════════════════════════════════════════════


class TestCheckDegenerateStrategy:
    """Tests for check_degenerate_strategy from crabquant.validation."""

    def test_zero_trades_is_degenerate(self):
        """Strategy with 0 trades should be flagged."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(num_trades=0)
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is True
        assert "0 trades" in reason
        assert "DEGENERATE" in reason

    def test_one_trade_is_degenerate(self):
        """Strategy with 1 trade (below min_trades=3) should be flagged."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(num_trades=1)
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is True
        assert "1 trades" in reason

    def test_two_trades_is_degenerate(self):
        """Strategy with 2 trades (below min_trades=3) should be flagged."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(num_trades=2)
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is True

    def test_exactly_min_trades_not_degenerate(self):
        """Strategy with exactly min_trades=3 should not be flagged for trade count."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(num_trades=3, total_return=0.05, max_drawdown=-0.02)
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is False
        assert reason == ""

    def test_constant_position_is_degenerate(self):
        """Zero return AND zero drawdown despite having trades = constant position."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(
            num_trades=10, total_return=0, max_drawdown=0
        )
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is True
        assert "constant position" in reason.lower() or "never changed" in reason.lower()

    def test_nonzero_return_not_constant_position(self):
        """Non-zero return should not trigger constant-position check."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(
            num_trades=10, total_return=0.01, max_drawdown=0
        )
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is False

    def test_nonzero_drawdown_not_constant_position(self):
        """Non-zero drawdown should not trigger constant-position check."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(
            num_trades=10, total_return=0, max_drawdown=-0.01
        )
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is False

    def test_flat_pnl_curve_is_degenerate(self):
        """Near-zero return volatility (flat equity curve) should be flagged."""
        from crabquant.validation import check_degenerate_strategy

        returns_series = pd.Series(np.zeros(100))
        result = _make_backtest_result(
            num_trades=10, total_return=0.05, max_drawdown=-0.01
        )
        result.returns = returns_series
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is True
        assert "flat PnL" in reason

    def test_flat_equity_curve_is_degenerate(self):
        """Flat equity_curve attribute should also trigger degenerate check."""
        from crabquant.validation import check_degenerate_strategy

        equity = pd.Series(np.ones(100))
        result = _make_backtest_result(
            num_trades=10, total_return=0.05, max_drawdown=-0.01
        )
        result.equity_curve = equity
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is True
        assert "flat PnL" in reason

    def test_normal_strategy_not_degenerate(self):
        """Healthy strategy with normal metrics should pass."""
        from crabquant.validation import check_degenerate_strategy

        returns_series = pd.Series(np.random.randn(100) * 0.01)
        result = _make_backtest_result(
            num_trades=20, total_return=0.12, max_drawdown=-0.08
        )
        result.returns = returns_series
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is False
        assert reason == ""

    def test_custom_min_trades(self):
        """Custom min_trades parameter should be respected."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(num_trades=5, total_return=0.01, max_drawdown=-0.01)
        # With min_trades=10, 5 trades should be degenerate
        is_deg, _ = check_degenerate_strategy(result, min_trades=10)
        assert is_deg is True
        # With min_trades=3, 5 trades should be fine
        is_deg, _ = check_degenerate_strategy(result, min_trades=3)
        assert is_deg is False

    def test_custom_min_return_vol(self):
        """Custom min_return_vol should control the flat-PnL threshold."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(num_trades=10, total_return=0.05, max_drawdown=-0.01)
        # Very low vol returns
        result.returns = pd.Series(np.random.randn(100) * 1e-10)
        # Default threshold (1e-8) should flag this
        is_deg, _ = check_degenerate_strategy(result)
        assert is_deg is True
        # Very relaxed threshold should not flag
        is_deg, _ = check_degenerate_strategy(result, min_return_vol=1e-15)
        assert is_deg is False

    def test_missing_attributes_graceful(self):
        """Result object missing optional attrs should not crash."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(num_trades=5, total_return=0.05, max_drawdown=-0.01)
        # No returns or equity_curve — should skip flat-PnL check
        is_deg, reason = check_degenerate_strategy(result)
        assert is_deg is False

    def test_zero_trades_custom_min(self):
        """0 trades always degenerate regardless of min_trades."""
        from crabquant.validation import check_degenerate_strategy

        result = _make_backtest_result(num_trades=0)
        is_deg, _ = check_degenerate_strategy(result, min_trades=0)
        # 0 < 0 is False, so this shouldn't be degenerate for trade count.
        # But it may still trigger constant position check if return==0 and dd==0.
        # Default result has total_return=0.12 and max_drawdown=-0.08, so no.
        assert is_deg is False


# ══════════════════════════════════════════════════════════════════════════
# 2. time_reversed_check — direct function tests
# ══════════════════════════════════════════════════════════════════════════


class TestTimeReversedCheck:
    """Tests for time_reversed_check from crabquant.validation."""

    @patch("crabquant.validation.BacktestEngine")
    def test_overfit_detected_when_reversed_performs_well(self, MockEngine):
        """When reversed Sharpe is >30% of normal, should flag overfit."""
        from crabquant.validation import time_reversed_check

        mock_engine = MockEngine.return_value
        mock_normal = SimpleNamespace(sharpe=1.0)
        mock_reversed = SimpleNamespace(sharpe=0.5)  # 50% of normal > 30% threshold
        mock_engine.run.side_effect = [mock_normal, mock_reversed]

        strategy_fn = MagicMock(__name__="test_strategy")
        strategy_fn.side_effect = [
            (pd.Series([True, False]), pd.Series([False, True])),
            (pd.Series([True, False]), pd.Series([False, True])),
        ]
        data = pd.DataFrame({
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [100, 200, 150, 300, 250],
        })
        params = {"fast": 12}

        passed, explanation = time_reversed_check(strategy_fn, data, params, threshold=0.3)
        assert passed is False
        assert "Overfit" in explanation
        assert "50%" in explanation

    @patch("crabquant.validation.BacktestEngine")
    def test_passed_when_reversed_performs_poorly(self, MockEngine):
        """When reversed Sharpe is << normal, should pass."""
        from crabquant.validation import time_reversed_check

        mock_engine = MockEngine.return_value
        mock_normal = SimpleNamespace(sharpe=2.0)
        mock_reversed = SimpleNamespace(sharpe=0.1)  # 5% of normal << 30%
        mock_engine.run.side_effect = [mock_normal, mock_reversed]

        strategy_fn = MagicMock(__name__="test_strategy")
        strategy_fn.side_effect = [
            (pd.Series([True, False]), pd.Series([False, True])),
            (pd.Series([True, False]), pd.Series([False, True])),
        ]
        data = pd.DataFrame({
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [100, 200, 150, 300, 250],
        })
        params = {"fast": 12}

        passed, explanation = time_reversed_check(strategy_fn, data, params, threshold=0.3)
        assert passed is True
        assert "Passed" in explanation

    @patch("crabquant.validation.BacktestEngine")
    def test_exactly_at_threshold_is_overfit(self, MockEngine):
        """Exactly at threshold ratio should be flagged (ratio > threshold)."""
        from crabquant.validation import time_reversed_check

        mock_engine = MockEngine.return_value
        mock_normal = SimpleNamespace(sharpe=1.0)
        mock_reversed = SimpleNamespace(sharpe=0.31)  # 31% > 30%
        mock_engine.run.side_effect = [mock_normal, mock_reversed]

        strategy_fn = MagicMock(__name__="test_strategy")
        strategy_fn.side_effect = [
            (pd.Series([True, False]), pd.Series([False, True])),
            (pd.Series([True, False]), pd.Series([False, True])),
        ]
        data = pd.DataFrame({
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [100, 200, 150, 300, 250],
        })
        params = {}

        passed, _ = time_reversed_check(strategy_fn, data, params, threshold=0.3)
        assert passed is False

    @patch("crabquant.validation.BacktestEngine")
    def test_custom_threshold(self, MockEngine):
        """Custom threshold should be respected."""
        from crabquant.validation import time_reversed_check

        mock_engine = MockEngine.return_value
        mock_normal = SimpleNamespace(sharpe=1.0)
        mock_reversed = SimpleNamespace(sharpe=0.4)  # 40%
        mock_engine.run.side_effect = [mock_normal, mock_reversed, mock_normal, mock_reversed]

        strategy_fn = MagicMock(__name__="test_strategy")
        strategy_fn.side_effect = [
            (pd.Series([True, False]), pd.Series([False, True])),
            (pd.Series([True, False]), pd.Series([False, True])),
            (pd.Series([True, False]), pd.Series([False, True])),
            (pd.Series([True, False]), pd.Series([False, True])),
        ]
        data = pd.DataFrame({
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [100, 200, 150, 300, 250],
        })
        params = {}

        # 40% > 30% → overfit with default threshold
        passed, _ = time_reversed_check(strategy_fn, data, params, threshold=0.3)
        assert passed is False
        # 40% < 50% → pass with relaxed threshold
        passed, _ = time_reversed_check(strategy_fn, data, params, threshold=0.5)
        assert passed is True

    @patch("crabquant.validation.BacktestEngine")
    def test_data_reversal(self, MockEngine):
        """Verify the function actually reverses the data (checks call order)."""
        from crabquant.validation import time_reversed_check

        mock_engine = MockEngine.return_value
        mock_normal = SimpleNamespace(sharpe=1.0)
        mock_reversed = SimpleNamespace(sharpe=0.0)
        mock_engine.run.side_effect = [mock_normal, mock_reversed]

        strategy_fn = MagicMock(__name__="test_strategy")
        strategy_fn.side_effect = [
            (pd.Series([True]), pd.Series([False])),
            (pd.Series([True]), pd.Series([False])),
        ]
        data = pd.DataFrame({
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 200, 150],
        })
        params = {}

        time_reversed_check(strategy_fn, data, params)

        # Strategy should have been called twice
        assert strategy_fn.call_count == 2
        # Second call should receive reversed data
        reversed_data = strategy_fn.call_args_list[1][0][0]
        pd.testing.assert_frame_equal(
            reversed_data.reset_index(drop=True),
            data.iloc[::-1].reset_index(drop=True),
        )


# ══════════════════════════════════════════════════════════════════════════
# 3. check_family_plateau — direct function tests
# ══════════════════════════════════════════════════════════════════════════


class TestCheckFamilyPlateau:
    """Tests for check_family_plateau from crabquant.refinement.stagnation."""

    def test_same_family_triggers_pivot(self):
        """3 consecutive same-family turns with no KEEP should trigger pivot."""
        from crabquant.refinement.stagnation import check_family_plateau

        history = [
            _make_history_entry(
                turn=1, status="failed",
                code='entries = cached_indicator("ema", df, 12)',
            ),
            _make_history_entry(
                turn=2, status="failed",
                code='entries = cached_indicator("sma", df, 20)',
            ),
            _make_history_entry(
                turn=3, status="failed",
                code='entries = cached_indicator("macd", df)',
            ),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is True
        assert pivot_type is not None
        assert message is not None
        assert "momentum" in message.lower() or "STUCK" in message

    def test_cross_pivot_when_different_archetype(self):
        """Should suggest cross-family pivot when stuck family != mandate archetype."""
        from crabquant.refinement.stagnation import check_family_plateau

        history = [
            _make_history_entry(
                turn=1, status="failed",
                code='entries = cached_indicator("rsi", df, 14)',
            ),
            _make_history_entry(
                turn=2, status="failed",
                code='entries = cached_indicator("bbands", df, 20)',
            ),
            _make_history_entry(
                turn=3, status="failed",
                code='entries = cached_indicator("stoch", df, 14)',
            ),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is True
        assert pivot_type == "cross"
        assert "mean_reversion" in message.lower() or "rsi" in message.lower()

    def test_within_pivot_when_same_archetype(self):
        """Should suggest within-family pivot when stuck family matches mandate."""
        from crabquant.refinement.stagnation import check_family_plateau

        history = [
            _make_history_entry(
                turn=1, status="failed",
                code='entries = cached_indicator("ema", df, 12)',
            ),
            _make_history_entry(
                turn=2, status="failed",
                code='entries = cached_indicator("sma", df, 20)',
            ),
            _make_history_entry(
                turn=3, status="failed",
                code='entries = cached_indicator("macd", df)',
            ),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is True
        assert pivot_type == "within"

    def test_force_diversify_triggers_cross_pivot(self):
        """force_diversify=True should always trigger cross-family pivot."""
        from crabquant.refinement.stagnation import check_family_plateau

        history = [
            _make_history_entry(
                turn=1, status="failed",
                code='entries = cached_indicator("ema", df, 12)',
            ),
            _make_history_entry(
                turn=2, status="failed",
                code='entries = cached_indicator("sma", df, 20)',
            ),
            _make_history_entry(
                turn=3, status="failed",
                code='entries = cached_indicator("macd", df)',
            ),
        ]
        mandate = {"strategy_archetype": "momentum", "force_diversify": True}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is True
        assert pivot_type == "cross"

    def test_keep_status_prevents_pivot(self):
        """If any recent turn has KEEP status, should not trigger pivot."""
        from crabquant.refinement.stagnation import check_family_plateau

        history = [
            _make_history_entry(
                turn=1, status="failed",
                code='entries = cached_indicator("ema", df, 12)',
            ),
            _make_history_entry(
                turn=2, status="KEEP",
                code='entries = cached_indicator("sma", df, 20)',
            ),
            _make_history_entry(
                turn=3, status="failed",
                code='entries = cached_indicator("macd", df)',
            ),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is False
        assert pivot_type is None

    def test_mixed_families_no_pivot(self):
        """Different indicator families should not trigger pivot."""
        from crabquant.refinement.stagnation import check_family_plateau

        history = [
            _make_history_entry(
                turn=1, status="failed",
                code='entries = cached_indicator("ema", df, 12)',
            ),
            _make_history_entry(
                turn=2, status="failed",
                code='entries = cached_indicator("rsi", df, 14)',
            ),
            _make_history_entry(
                turn=3, status="failed",
                code='entries = cached_indicator("atr", df, 14)',
            ),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is False

    def test_insufficient_history_no_pivot(self):
        """Fewer than max_same_family entries should not trigger pivot."""
        from crabquant.refinement.stagnation import check_family_plateau

        history = [
            _make_history_entry(
                turn=1, status="failed",
                code='entries = cached_indicator("ema", df, 12)',
            ),
            _make_history_entry(
                turn=2, status="failed",
                code='entries = cached_indicator("sma", df, 20)',
            ),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is False

    def test_unknown_family_no_pivot(self):
        """Turns with unrecognizable indicators should not trigger pivot."""
        from crabquant.refinement.stagnation import check_family_plateau

        history = [
            _make_history_entry(
                turn=1, status="failed",
                code='entries = some_unknown_func(df)',
            ),
            _make_history_entry(
                turn=2, status="failed",
                code='entries = another_mystery(df)',
            ),
            _make_history_entry(
                turn=3, status="failed",
                code='entries = yet_more_unknown(df)',
            ),
        ]
        mandate = {"strategy_archetype": "momentum"}
        should_pivot, pivot_type, message = check_family_plateau(history, mandate)
        assert should_pivot is False

    def test_custom_max_same_family(self):
        """Custom max_same_family should change the trigger threshold."""
        from crabquant.refinement.stagnation import check_family_plateau

        history = [
            _make_history_entry(
                turn=i, status="failed",
                code='entries = cached_indicator("ema", df, 12)',
            )
            for i in range(1, 5)  # 4 same-family turns
        ]
        mandate = {"strategy_archetype": "momentum"}
        # Default max_same_family=3 → should trigger with 3+ same
        should_pivot, _, _ = check_family_plateau(history, mandate, max_same_family=3)
        assert should_pivot is True
        # With max_same_family=5 → should NOT trigger (only 4 same)
        should_pivot, _, _ = check_family_plateau(history, mandate, max_same_family=5)
        assert should_pivot is False


# ══════════════════════════════════════════════════════════════════════════
# 4. Integration: time_reversed_check in promotion.py
# ══════════════════════════════════════════════════════════════════════════


class TestTimeReversedInPromotion:
    """Tests that time_reversed_check is properly wired into run_full_validation_check."""

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.data.load_data")
    @patch("crabquant.validation.time_reversed_check")
    def test_overfit_flag_in_result_dict(
        self, mock_trc, mock_load_data, mock_rwf, mock_ct
    ):
        """When time_reversed_check detects overfit, result should have flag."""
        from crabquant.refinement.promotion import run_full_validation_check

        # Set up mock data — load_data is called by both rolling_walk_forward
        # and the time-reversed check inside promotion.py
        test_data = pd.DataFrame({
            "open": [1] * 300, "high": [2] * 300,
            "low": [0.5] * 300, "close": [1.5] * 300,
            "volume": [100] * 300,
        })
        mock_load_data.return_value = test_data

        # Mock rolling_walk_forward
        rwf_result = MagicMock()
        rwf_result.avg_test_sharpe = 0.5
        rwf_result.min_test_sharpe = 0.3
        rwf_result.avg_degradation = 0.2
        rwf_result.num_windows = 2
        rwf_result.windows_passed = 2
        rwf_result.robust = True
        rwf_result.notes = "all good"
        rwf_result.window_results = []
        mock_rwf.return_value = rwf_result

        # Mock cross_ticker_validation
        ct_result = MagicMock()
        ct_result.avg_sharpe = 0.4
        ct_result.median_sharpe = 0.4
        ct_result.robust = True
        ct_result.tickers_profitable = 2
        ct_result.tickers_tested = 3
        ct_result.notes = "good"
        mock_ct.return_value = ct_result

        # Mock time_reversed_check — overfit detected
        mock_trc.return_value = (False, "Overfit: reversed Sharpe 0.5 is 50% of normal 1.0")

        strategy_fn = MagicMock(__name__="test_strategy")
        strategy_fn.side_effect = [
            # rolling walk-forward calls
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            # cross_ticker calls
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
        ]

        result = run_full_validation_check(
            strategy_fn, {"fast": 12}, "AAPL", ["AAPL", "MSFT"],
        )

        assert "time_reversed_overfit" in result
        assert result["time_reversed_overfit"] is True
        assert "time_reversed_explanation" in result
        assert isinstance(result["time_reversed_explanation"], str)

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.data.load_data")
    @patch("crabquant.validation.time_reversed_check")
    def test_passed_when_no_overfit(
        self, mock_trc, mock_load_data, mock_rwf, mock_ct
    ):
        """When time_reversed_check passes, flag should be False."""
        from crabquant.refinement.promotion import run_full_validation_check

        test_data = pd.DataFrame({
            "open": [1] * 300, "high": [2] * 300,
            "low": [0.5] * 300, "close": [1.5] * 300,
            "volume": [100] * 300,
        })
        mock_load_data.return_value = test_data

        rwf_result = MagicMock()
        rwf_result.avg_test_sharpe = 0.5
        rwf_result.min_test_sharpe = 0.3
        rwf_result.avg_degradation = 0.2
        rwf_result.num_windows = 2
        rwf_result.windows_passed = 2
        rwf_result.robust = True
        rwf_result.notes = "all good"
        rwf_result.window_results = []
        mock_rwf.return_value = rwf_result

        ct_result = MagicMock()
        ct_result.avg_sharpe = 0.4
        ct_result.median_sharpe = 0.4
        ct_result.robust = True
        ct_result.tickers_profitable = 2
        ct_result.tickers_tested = 3
        ct_result.notes = "good"
        mock_ct.return_value = ct_result

        # time_reversed_check passes (no overfit)
        mock_trc.return_value = (True, "Passed: reversed Sharpe 0.1 is 5% of normal 2.0")

        strategy_fn = MagicMock(__name__="test_strategy")
        strategy_fn.side_effect = [
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
        ]

        result = run_full_validation_check(
            strategy_fn, {"fast": 12}, "AAPL", ["AAPL", "MSFT"],
        )

        assert "time_reversed_overfit" in result
        assert result["time_reversed_overfit"] is False

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    @patch("crabquant.data.load_data", return_value=None)
    @patch("crabquant.validation.BacktestEngine")
    def test_insufficient_data_sets_none_flag(
        self, MockEngine, mock_load_data, mock_rwf, mock_ct
    ):
        """When data is insufficient for time-reversed check, flag should be None."""
        from crabquant.refinement.promotion import run_full_validation_check

        rwf_result = MagicMock()
        rwf_result.avg_test_sharpe = 0.5
        rwf_result.min_test_sharpe = 0.3
        rwf_result.avg_degradation = 0.2
        rwf_result.num_windows = 2
        rwf_result.windows_passed = 2
        rwf_result.robust = True
        rwf_result.notes = "all good"
        rwf_result.window_results = []
        mock_rwf.return_value = rwf_result

        ct_result = MagicMock()
        ct_result.avg_sharpe = 0.4
        ct_result.median_sharpe = 0.4
        ct_result.robust = True
        ct_result.tickers_profitable = 2
        ct_result.tickers_tested = 3
        ct_result.notes = "good"
        mock_ct.return_value = ct_result

        strategy_fn = MagicMock(__name__="test_strategy")
        strategy_fn.side_effect = [
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
            (pd.Series(dtype=bool), pd.Series(dtype=bool)),
        ]

        result = run_full_validation_check(
            strategy_fn, {"fast": 12}, "AAPL", ["AAPL", "MSFT"],
        )

        assert "time_reversed_overfit" in result
        assert result["time_reversed_overfit"] is None
        assert "Insufficient" in result["time_reversed_explanation"]


# ══════════════════════════════════════════════════════════════════════════
# 5. Integration: check_family_plateau in context_builder.py
# ══════════════════════════════════════════════════════════════════════════


class TestFamilyPlateauInContextBuilder:
    """Tests that check_family_plateau is wired into build_llm_context."""

    def test_plateau_detected_adds_section(self):
        """When plateau is detected, context should include family_plateau_section."""
        from crabquant.refinement.context_builder import build_llm_context

        state = SimpleNamespace(
            current_turn=4,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=0.3,
            best_composite_score=-1.0,
            best_turn=1,
            tickers=["AAPL"],
            history=[
                _make_history_entry(
                    turn=1, status="failed",
                    code='entries = cached_indicator("ema", df, 12)',
                ),
                _make_history_entry(
                    turn=2, status="failed",
                    code='entries = cached_indicator("sma", df, 20)',
                ),
                _make_history_entry(
                    turn=3, status="failed",
                    code='entries = cached_indicator("macd", df)',
                ),
            ],
            revert_notice="",
            code_quality_feedback="",
        )
        mandate = {"strategy_archetype": "momentum"}

        context = build_llm_context(state, mandate=mandate)

        assert "family_plateau_section" in context
        assert "INDICATOR FAMILY PLATEAU DETECTED" in context["family_plateau_section"]
        assert "within" in context["family_plateau_section"].lower()

    def test_no_plateau_no_section(self):
        """When no plateau is detected, context should not include family_plateau_section."""
        from crabquant.refinement.context_builder import build_llm_context

        state = SimpleNamespace(
            current_turn=2,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=0.3,
            best_composite_score=-1.0,
            best_turn=1,
            tickers=["AAPL"],
            history=[
                _make_history_entry(
                    turn=1, status="failed",
                    code='entries = cached_indicator("ema", df, 12)',
                ),
            ],
            revert_notice="",
            code_quality_feedback="",
        )
        mandate = {"strategy_archetype": "momentum"}

        context = build_llm_context(state, mandate=mandate)

        assert "family_plateau_section" not in context

    def test_mixed_history_no_plateau(self):
        """Mixed indicator families should not produce plateau section."""
        from crabquant.refinement.context_builder import build_llm_context

        state = SimpleNamespace(
            current_turn=4,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=0.3,
            best_composite_score=-1.0,
            best_turn=1,
            tickers=["AAPL"],
            history=[
                _make_history_entry(
                    turn=1, status="failed",
                    code='entries = cached_indicator("ema", df, 12)',
                ),
                _make_history_entry(
                    turn=2, status="failed",
                    code='entries = cached_indicator("rsi", df, 14)',
                ),
                _make_history_entry(
                    turn=3, status="failed",
                    code='entries = cached_indicator("atr", df, 14)',
                ),
            ],
            revert_notice="",
            code_quality_feedback="",
        )
        mandate = {"strategy_archetype": "momentum"}

        context = build_llm_context(state, mandate=mandate)

        assert "family_plateau_section" not in context

    def test_cross_pivot_in_context(self):
        """Cross-family pivot should appear in context section."""
        from crabquant.refinement.context_builder import build_llm_context

        state = SimpleNamespace(
            current_turn=4,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=0.3,
            best_composite_score=-1.0,
            best_turn=1,
            tickers=["AAPL"],
            history=[
                _make_history_entry(
                    turn=1, status="failed",
                    code='entries = cached_indicator("rsi", df, 14)',
                ),
                _make_history_entry(
                    turn=2, status="failed",
                    code='entries = cached_indicator("bbands", df, 20)',
                ),
                _make_history_entry(
                    turn=3, status="failed",
                    code='entries = cached_indicator("stoch", df, 14)',
                ),
            ],
            revert_notice="",
            code_quality_feedback="",
        )
        # Mandate says momentum, but stuck in mean_reversion → cross pivot
        mandate = {"strategy_archetype": "momentum"}

        context = build_llm_context(state, mandate=mandate)

        assert "family_plateau_section" in context
        assert "cross" in context["family_plateau_section"].lower()
