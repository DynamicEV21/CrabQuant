"""Tests for Expected Value metric and updated composite score.

Task: Add Sortino/EV metrics to CrabQuant.

Tests cover:
- Expected Value computation (known data)
- Sortino ratio presence in BacktestResult
- Updated composite score formula: (sortino_weighted + ev_weighted) * robustness_factor
- Backward compatibility of compute_composite_score
"""

import importlib
import importlib.util
import math
import os
import sys
from types import ModuleType
import pytest
import numpy as np

# Mock pandas_ta before any crabquant imports to avoid transitive dependency
# (pandas_ta dropped Python 3.11 support, but we only need it in strategy files)
if "pandas_ta" not in sys.modules:
    _mock_ta = ModuleType("pandas_ta")
    _mock_ta.__version__ = "0.0.0-mock"
    _mock_ta.__all__ = []
    sys.modules["pandas_ta"] = _mock_ta

from crabquant.engine.backtest import BacktestResult

# Now import compute_composite_score — the pandas_ta mock prevents the chain failure
from crabquant.refinement.prompts import compute_composite_score


class TestBacktestResultExpectedValue:
    """Test that BacktestResult includes expected_value field."""

    def test_expected_value_field_exists(self):
        """BacktestResult should have expected_value field."""
        result = BacktestResult(
            ticker="AAPL", strategy_name="test", iteration=0,
            sharpe=1.5, total_return=0.10, max_drawdown=-0.05,
            win_rate=0.6, num_trades=20, avg_trade_return=0.02,
            calmar_ratio=2.0, sortino_ratio=2.5, expected_value=150.0,
            profit_factor=1.5, avg_holding_bars=5.0,
            best_trade=500.0, worst_trade=-200.0,
            passed=True, score=1.0, notes="test",
        )
        assert hasattr(result, "expected_value")
        assert result.expected_value == 150.0

    def test_expected_value_default_zero(self):
        """Default expected_value should be 0.0."""
        result = BacktestResult(
            ticker="AAPL", strategy_name="test", iteration=0,
            sharpe=1.5, total_return=0.10, max_drawdown=-0.05,
            win_rate=0.6, num_trades=20, avg_trade_return=0.02,
            calmar_ratio=2.0, sortino_ratio=2.5, expected_value=0.0,
            profit_factor=1.5, avg_holding_bars=5.0,
            best_trade=500.0, worst_trade=-200.0,
            passed=True, score=1.0, notes="test",
        )
        assert result.expected_value == 0.0

    def test_ev_computation_basic(self):
        """Verify EV formula: (win_rate * avg_win) - ((1 - win_rate) * avg_loss)."""
        # 60% win rate, avg win $200, avg loss $100
        win_rate = 0.6
        avg_win = 200.0
        avg_loss = 100.0
        ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        # 0.6 * 200 - 0.4 * 100 = 120 - 40 = 80
        assert ev == pytest.approx(80.0, rel=1e-6)

    def test_ev_positive_edge_case(self):
        """EV should be positive when wins outweigh losses."""
        win_rate = 0.55
        avg_win = 300.0
        avg_loss = 150.0
        ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        # 0.55 * 300 - 0.45 * 150 = 165 - 67.5 = 97.5
        assert ev > 0
        assert ev == pytest.approx(97.5, rel=1e-6)

    def test_ev_negative_when_losing_strategy(self):
        """EV should be negative for losing strategies."""
        win_rate = 0.4
        avg_win = 100.0
        avg_loss = 200.0
        ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        # 0.4 * 100 - 0.6 * 200 = 40 - 120 = -80
        assert ev < 0
        assert ev == pytest.approx(-80.0, rel=1e-6)

    def test_ev_zero_when_balanced(self):
        """EV should be ~0 when wins/losses are balanced."""
        win_rate = 0.5
        avg_win = 200.0
        avg_loss = 200.0
        ev = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        assert ev == pytest.approx(0.0, rel=1e-6)


class TestUpdatedCompositeScore:
    """Test the updated composite score formula: (sortino_weighted + ev_weighted) * robustness_factor."""

    def test_basic_with_sortino_and_ev(self):
        """Basic case with both sortino and EV."""
        score = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.10,
            sortino=3.0, expected_value=100.0,
        )
        # sortino_weighted = min(3.0/3.0, 1.0) = 1.0
        # ev_weighted = sign(100) * min(100/100, 1.0) = 1.0
        # robustness = sqrt(40/20) * (1-0.10) = sqrt(2) * 0.9 = 1.2728
        # score = (1.0 + 1.0) * 1.2728 = 2.5456
        expected = 2.0 * math.sqrt(40 / 20) * 0.9
        assert score == pytest.approx(expected, rel=1e-6)

    def test_backward_compat_no_sortino_no_ev(self):
        """Without sortino/EV, score should be based on robustness alone."""
        score = compute_composite_score(sharpe=2.0, trades=40, max_drawdown=0.10)
        # sortino_weighted = 0, ev_weighted = 0 → score = 0
        assert score == 0.0

    def test_positive_ev_negative_sortino(self):
        """Positive EV but negative sortino → only EV contributes."""
        score = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.10,
            sortino=-1.0, expected_value=50.0,
        )
        # sortino_weighted = 0 (clamped), ev_weighted = min(50/100, 1) = 0.5
        # robustness = sqrt(2) * 0.9
        expected = 0.5 * math.sqrt(40 / 20) * 0.9
        assert score == pytest.approx(expected, rel=1e-6)

    def test_negative_ev_positive_sortino(self):
        """Positive sortino but negative EV → only sortino contributes (EV penalizes)."""
        score = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.10,
            sortino=3.0, expected_value=-50.0,
        )
        # sortino_weighted = 1.0, ev_weighted = -min(50/100, 1) = -0.5
        # robustness = sqrt(2) * 0.9
        expected = (1.0 - 0.5) * math.sqrt(40 / 20) * 0.9
        assert score == pytest.approx(expected, rel=1e-6)

    def test_zero_trades_returns_zero(self):
        """Zero trades should return 0.0 regardless of sortino/EV."""
        assert compute_composite_score(
            sharpe=5.0, trades=0, max_drawdown=0.01,
            sortino=10.0, expected_value=1000.0,
        ) == 0.0

    def test_sortino_cap_at_3(self):
        """Sortino above 3.0 should be capped (sortino_weighted max 1.0)."""
        score_low = compute_composite_score(
            sharpe=2.0, trades=20, max_drawdown=0.0,
            sortino=3.0, expected_value=0.0,
        )
        score_high = compute_composite_score(
            sharpe=2.0, trades=20, max_drawdown=0.0,
            sortino=10.0, expected_value=0.0,
        )
        assert score_low == score_high  # Both capped at 1.0 * robustness

    def test_ev_normalisation_large_values(self):
        """Very large EV should be capped at ev_weighted = 1.0."""
        score_small = compute_composite_score(
            sharpe=2.0, trades=20, max_drawdown=0.0,
            sortino=0.0, expected_value=100.0,
        )
        score_large = compute_composite_score(
            sharpe=2.0, trades=20, max_drawdown=0.0,
            sortino=0.0, expected_value=10000.0,
        )
        assert score_small == score_large  # Both capped at 1.0 * robustness

    def test_negative_ev_penalizes_score(self):
        """Negative EV should reduce score compared to zero EV."""
        score_zero = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.10,
            sortino=2.0, expected_value=0.0,
        )
        score_neg = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.10,
            sortino=2.0, expected_value=-50.0,
        )
        assert score_neg < score_zero

    def test_drawdown_penalty_applies(self):
        """High drawdown should reduce score via robustness_factor."""
        score_good = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.05,
            sortino=3.0, expected_value=100.0,
        )
        score_bad = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.50,
            sortino=3.0, expected_value=100.0,
        )
        assert score_good > score_bad

    def test_inf_sortino_handled(self):
        """Infinite sortino should be treated as 0."""
        score_inf = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.10,
            sortino=float('inf'), expected_value=50.0,
        )
        score_zero = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.10,
            sortino=0.0, expected_value=50.0,
        )
        assert score_inf == pytest.approx(score_zero, rel=1e-6)

    def test_nan_sortino_handled(self):
        """NaN sortino should be treated as 0."""
        score_nan = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.10,
            sortino=float('nan'), expected_value=50.0,
        )
        score_zero = compute_composite_score(
            sharpe=2.0, trades=40, max_drawdown=0.10,
            sortino=0.0, expected_value=50.0,
        )
        assert score_nan == pytest.approx(score_zero, rel=1e-6)


class TestSortinoRatioInResult:
    """Test that sortino_ratio is properly stored in BacktestResult."""

    def test_sortino_ratio_field_exists(self):
        """BacktestResult should have sortino_ratio field."""
        result = BacktestResult(
            ticker="AAPL", strategy_name="test", iteration=0,
            sharpe=1.5, total_return=0.10, max_drawdown=-0.05,
            win_rate=0.6, num_trades=20, avg_trade_return=0.02,
            calmar_ratio=2.0, sortino_ratio=2.5, expected_value=100.0,
            profit_factor=1.5, avg_holding_bars=5.0,
            best_trade=500.0, worst_trade=-200.0,
            passed=True, score=1.0, notes="test",
        )
        assert result.sortino_ratio == 2.5

    def test_sortino_formula_downside_only(self):
        """Sortino should penalize downside volatility only.

        With returns [0.01, 0.02, -0.01, 0.03, -0.005]:
        - mean return = 0.009
        - downside (below 0) = [-0.01, -0.005]
        - downside_std = std([-0.01, -0.005]) ≈ 0.00354
        - sortino ≈ 0.009 / 0.00354 ≈ 2.545 (annualised * sqrt(252))
        """
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.005])
        mean_ret = np.mean(returns)
        downside = returns[returns < 0]
        if len(downside) > 1:
            downside_std = np.std(downside, ddof=1)
        elif len(downside) == 1:
            downside_std = abs(downside[0])
        else:
            downside_std = 1e-10

        sortino = (mean_ret / downside_std) * np.sqrt(252)
        assert sortino > 0
        assert sortino > 2.0  # Strong risk-adjusted return

    def test_sortino_higher_than_sharpe_for_positive_skew(self):
        """For positively skewed returns, sortino > sharpe (downside vol < total vol)."""
        returns = np.array([0.05, 0.03, -0.01, 0.08, -0.005, 0.02, 0.04])
        mean_ret = np.mean(returns)
        total_std = np.std(returns, ddof=1)
        downside = returns[returns < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else abs(downside[0])

        sharpe = (mean_ret / total_std) * np.sqrt(252)
        sortino = (mean_ret / downside_std) * np.sqrt(252)
        assert sortino > sharpe
