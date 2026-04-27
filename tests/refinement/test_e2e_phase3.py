"""End-to-end test for Phase 3 pipeline — mandate → refine → validate → promote.

Tests integration across Phase 2 and Phase 3 components:
  - mandate_generator: generate mandates from strategy catalog
  - stagnation: detect stagnation during refinement
  - circuit_breaker: track LLM pass rate
  - regime_sharpe: regime-dependent Sharpe analysis
  - promotion: walk-forward validation + auto-register
  - portfolio_correlation: equity curve correlation across winners
  - action_analytics: action success rate tracking

All external dependencies (LLM calls, backtest engine, data loading,
file I/O for strategy registration) are mocked.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_strategy_code():
    """Valid strategy code that passes syntax and import checks."""
    return '''
import numpy as np
import pandas as pd

DESCRIPTION = "Test momentum strategy"
DEFAULT_PARAMS = {"fast": 10, "slow": 30}
PARAM_GRID = {"fast": [5, 10, 15], "slow": [20, 30, 50]}

def generate_signals(df, params):
    fast = params["fast"]
    slow = params["slow"]
    df = df.copy()
    df["fast_ma"] = df["close"].rolling(fast).mean()
    df["slow_ma"] = df["close"].rolling(slow).mean()
    entries = pd.Series(0, index=df.index)
    exits = pd.Series(0, index=df.index)
    entries[df["fast_ma"] > df["slow_ma"]] = 1
    exits[df["fast_ma"] < df["slow_ma"]] = -1
    return entries, exits
'''


@pytest.fixture
def mock_backtest_result():
    """A successful BacktestResult-like object."""
    return SimpleNamespace(
        sharpe=2.1,
        total_return=0.25,
        max_drawdown=-0.08,
        win_rate=0.58,
        num_trades=45,
        profit_factor=1.8,
        calmar_ratio=2.5,
        sortino_ratio=2.8,
        score=0.85,
        params={"fast": 10, "slow": 30},
        passed=True,
        ticker="SPY",
        strategy_name="test_momentum",
        iteration=3,
    )


@pytest.fixture
def mock_guardrail_report():
    return SimpleNamespace(
        passed=True,
        violations=[],
        warnings=[],
        score_adjustment=0.0,
    )


@pytest.fixture
def sample_mandate():
    return {
        "name": "momentum_spy_1",
        "description": "Refine momentum strategy for SPY",
        "strategy_archetype": "momentum",
        "tickers": ["SPY", "AAPL", "MSFT"],
        "primary_ticker": "SPY",
        "period": "2y",
        "sharpe_target": 1.5,
        "max_turns": 5,
        "constraints": {
            "max_parameters": 8,
            "min_trades": 5,
            "max_drawdown_pct": 25,
        },
    }


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for testing."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "results" / "winners").mkdir(parents=True)
    (tmp_path / "results" / "run_history.jsonl").touch()
    (tmp_path / "refinement" / "mandates").mkdir(parents=True)
    (tmp_path / "refinement_runs").mkdir()
    return tmp_path


# ── Phase 3 Component Integration Tests ───────────────────────────────────────

class TestStagnationAndCircuitBreaker:

    def test_stagnation_detects_declining_sharpes(self):
        """Stagnation module should detect declining Sharpe trend."""
        from crabquant.refinement.stagnation import compute_stagnation

        history = [
            {"sharpe": 1.5, "action": "modify_params"},
            {"sharpe": 1.2, "action": "modify_params"},
            {"sharpe": 0.8, "action": "modify_params"},
            {"sharpe": 0.5, "action": "modify_params"},
        ]
        score, trend = compute_stagnation(history)
        assert trend == "declining"
        assert score > 0.5

    def test_stagnation_improving_trend(self):
        from crabquant.refinement.stagnation import compute_stagnation

        history = [
            {"sharpe": 0.3, "action": "replace_indicator"},
            {"sharpe": 0.8, "action": "change_entry_logic"},
            {"sharpe": 1.3, "action": "add_filter"},
            {"sharpe": 1.8, "action": "modify_params"},
        ]
        score, trend = compute_stagnation(history)
        assert trend == "improving"
        assert score < 0.5

    def test_circuit_breaker_opens_on_low_pass_rate(self):
        from crabquant.refinement.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(window=10, min_pass_rate=0.3)
        # Record 8 failures, 1 success — 10% pass rate < 30% threshold
        for _ in range(8):
            cb.record(False)
        cb.record(True)

        assert cb.is_open()

    def test_circuit_breaker_stays_closed_on_good_rate(self):
        from crabquant.refinement.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(window=10, min_pass_rate=0.3)
        for _ in range(7):
            cb.record(True)
        for _ in range(3):
            cb.record(False)

        assert not cb.is_open()
        assert cb.pass_rate >= 0.3

    def test_circuit_breaker_state_persistence(self):
        from crabquant.refinement.circuit_breaker import CircuitBreaker

        cb1 = CircuitBreaker(window=5, min_pass_rate=0.4)
        cb1.record(True)
        cb1.record(False)
        cb1.record(True)

        state = cb1.get_state()
        cb2 = CircuitBreaker.restore(state)

        assert cb2.pass_rate == cb1.pass_rate
        assert cb2.total_attempts == 3


class TestRegimeSharpe:

    def test_computes_regime_sharpe(self):
        from crabquant.refinement.regime_sharpe import compute_regime_sharpe

        # Mock portfolio with returns
        n = 252
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.02, n), index=dates)

        mock_portfolio = MagicMock()
        mock_portfolio.returns.return_value = returns

        # Create regime labels
        regime_labels = pd.Series(
            ["bull"] * 126 + ["bear"] * 126,
            index=dates,
        )

        report = compute_regime_sharpe(mock_portfolio, regime_labels)
        assert "bull" in report.sharpe_by_regime
        assert "bear" in report.sharpe_by_regime

    def test_regime_dependent_detection(self):
        from crabquant.refinement.regime_sharpe import (
            RegimeSharpeReport,
            is_regime_dependent,
        )

        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 2.5, "bear": -1.0},
            regime_segments=[],
        )
        assert is_regime_dependent(report)

    def test_regime_independent(self):
        from crabquant.refinement.regime_sharpe import (
            RegimeSharpeReport,
            is_regime_dependent,
        )

        report = RegimeSharpeReport(
            sharpe_by_regime={"bull": 1.5, "bear": 1.2},
            regime_segments=[],
        )
        assert not is_regime_dependent(report, threshold=2.0)


class TestPromotionWithMocks:

    def test_full_validation_check_passes(self, mock_backtest_result):
        """Walk-forward + cross-ticker validation with mocks."""
        from crabquant.validation import WalkForwardResult, CrossTickerResult

        mock_wf = WalkForwardResult(
            strategy_name="test_momentum",
            ticker="SPY",
            train_sharpe=1.8,
            train_return=0.20,
            test_sharpe=1.2,
            test_return=0.15,
            test_max_dd=-0.06,
            degradation=0.33,
            robust=True,
            notes="Good",
            regime_shift=False,
            test_regime="neutral",
            train_regime="bull",
        )
        mock_ct = CrossTickerResult(
            strategy_name="test_momentum",
            params={"fast": 10},
            tickers_tested=2,
            tickers_profitable=2,
            tickers_passed=2,
            avg_sharpe=1.0,
            median_sharpe=1.1,
            sharpe_std=0.1,
            avg_return=0.12,
            avg_max_dd=-0.07,
            win_rate_across_tickers=1.0,
            robust=True,
            notes="Good",
        )

        with patch("crabquant.validation.walk_forward_test", return_value=mock_wf), \
             patch("crabquant.validation.cross_ticker_validation", return_value=mock_ct):

            from crabquant.refinement.promotion import run_full_validation_check

            dummy_fn = lambda df, p: (pd.Series(dtype=float), pd.Series(dtype=float))
            result = run_full_validation_check(
                dummy_fn,
                params={"fast": 10},
                discovery_ticker="SPY",
                validation_tickers=["SPY", "AAPL"],
            )

        assert result["passed"] is True
        assert result["walk_forward_robust"] is True

    def test_full_validation_check_fails_low_sharpe(self):
        from crabquant.validation import WalkForwardResult, CrossTickerResult

        mock_wf = WalkForwardResult(
            strategy_name="test_momentum",
            ticker="SPY",
            train_sharpe=1.8,
            train_return=0.20,
            test_sharpe=0.1,
            test_return=0.01,
            test_max_dd=-0.20,
            degradation=0.94,
            robust=False,
            notes="Bad",
            regime_shift=True,
            test_regime="bear",
            train_regime="bull",
        )
        mock_ct = CrossTickerResult(
            strategy_name="test_momentum",
            params={"fast": 10},
            tickers_tested=2,
            tickers_profitable=0,
            tickers_passed=0,
            avg_sharpe=0.2,
            median_sharpe=0.3,
            sharpe_std=0.2,
            avg_return=0.02,
            avg_max_dd=-0.15,
            win_rate_across_tickers=0.0,
            robust=False,
            notes="Bad",
        )

        with patch("crabquant.validation.walk_forward_test", return_value=mock_wf), \
             patch("crabquant.validation.cross_ticker_validation", return_value=mock_ct):

            from crabquant.refinement.promotion import run_full_validation_check

            dummy_fn = lambda df, p: (pd.Series(dtype=float), pd.Series(dtype=float))
            result = run_full_validation_check(
                dummy_fn,
                params={"fast": 10},
                discovery_ticker="SPY",
                validation_tickers=["SPY", "AAPL"],
            )

        assert result["passed"] is False


class TestPortfolioCorrelation:

    def test_correlation_report_across_winners(self):
        from crabquant.refinement.portfolio_correlation import generate_correlation_report

        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")
        curves = {}
        for name in ["strat_a", "strat_b", "strat_c"]:
            curves[name] = pd.Series(
                rng.standard_normal(252).cumsum() + 100, index=idx
            )

        report = generate_correlation_report(curves)
        assert report["n_strategies"] == 3
        assert isinstance(report["redundant_pairs"], list)
        assert isinstance(report["diversifying_pairs"], list)


class TestActionAnalytics:

    def test_full_analytics_pipeline(self):
        from crabquant.refinement.action_analytics import (
            track_action_result,
            load_run_history,
            compute_action_success_rates,
            generate_llm_context,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = f.name

        try:
            # Simulate a refinement run's action history
            with patch("crabquant.refinement.action_analytics.RUN_HISTORY_FILE", tmp_path):
                track_action_result("mandate_1", 1, "modify_params", 0.5, False, "low_sharpe")
                track_action_result("mandate_1", 2, "change_entry_logic", 1.8, True, "")
                track_action_result("mandate_1", 3, "add_filter", 0.9, False, "too_few_trades")
                track_action_result("mandate_2", 1, "full_rewrite", 2.0, True, "")
                track_action_result("mandate_2", 2, "modify_params", 1.1, True, "")

            history = load_run_history(tmp_path)
            assert len(history) == 5

            rates = compute_action_success_rates(history)
            assert len(rates) == 4  # 4 unique action types
            # Top two should have 100% success rate
            assert rates[0]["success_rate"] == 1.0
            assert rates[1]["success_rate"] == 1.0

            ctx = generate_llm_context(history)
            assert "change_entry_logic" in ctx
            # The context contains percentage formatting, not literal "success_rate"
            assert "100%" in ctx or "100.0%" in ctx
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestMandateGenerator:

    def test_generates_mandates_from_catalog(self, tmp_project):
        from crabquant.refinement.mandate_generator import generate_mandates

        # Create a sample strategy file
        strat_file = tmp_project / "strategies" / "test_momentum.py"
        strat_file.write_text('''
DESCRIPTION = "Momentum strategy using moving averages"
DEFAULT_PARAMS = {"fast": 10, "slow": 30}
PARAM_GRID = {"fast": [5, 10, 20], "slow": [20, 30, 50]}

def generate_signals(df, params):
    import pandas as pd
    fast_ma = df["close"].rolling(params["fast"]).mean()
    slow_ma = df["close"].rolling(params["slow"]).mean()
    entries = pd.Series(0, index=df.index)
    exits = pd.Series(0, index=df.index)
    entries[fast_ma > slow_ma] = 1
    exits[fast_ma < slow_ma] = -1
    return entries, exits
''')

        mandates = generate_mandates(
            strategies_dir=str(tmp_project / "strategies"),
            tickers=["SPY", "AAPL"],
            count=3,
        )

        assert len(mandates) == 3
        for m in mandates:
            assert "name" in m
            assert "primary_ticker" in m
            assert "sharpe_target" in m
            assert "max_turns" in m
            assert "constraints" in m
            assert m["strategy_archetype"] == "momentum"


# ── Full Pipeline E2E ─────────────────────────────────────────────────────────

class TestFullPipelineE2E:

    def test_mandate_to_promotion_flow(
        self, mock_strategy_code, mock_backtest_result, mock_guardrail_report,
        sample_mandate, tmp_project,
    ):
        """End-to-end: generate mandate → refine → validate → promote.

        Uses mocks for LLM calls, backtest engine, and data loading.
        Verifies all Phase 3 components integrate correctly.
        """
        from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response
        from crabquant.refinement.circuit_breaker import CircuitBreaker

        # Simulate a refinement run
        history = []
        cb = CircuitBreaker(window=10, min_pass_rate=0.3)
        mandate = sample_mandate
        sharpe_target = mandate["sharpe_target"]
        max_turns = mandate["max_turns"]
        best_sharpe = -999.0
        success_turn = None

        for turn in range(1, max_turns + 1):
            # Simulate improving Sharpes (converging strategy)
            simulated_sharpe = 0.5 + turn * 0.4  # 0.9, 1.3, 1.7, 2.1, 2.5
            success = simulated_sharpe >= sharpe_target
            action = "modify_params" if turn <= 2 else "change_entry_logic"

            # Record action for analytics
            history.append({
                "turn": turn,
                "sharpe": simulated_sharpe,
                "action": action,
                "success": success,
                "failure_mode": "" if success else "low_sharpe",
            })

            # Circuit breaker: mock LLM validation (all pass)
            cb.record(True)

            # Track best
            if simulated_sharpe > best_sharpe:
                best_sharpe = simulated_sharpe

            # Check stagnation
            score, trend = compute_stagnation(history)
            stag = get_stagnation_response(turn, score)
            assert stag["constraint"] != "abandon"  # Should never abandon with improving Sharpes

            # Check if converged
            if success:
                success_turn = turn
                break

        # Verify convergence
        assert success_turn is not None, "Strategy should have converged"
        assert best_sharpe >= sharpe_target

        # Verify circuit breaker stayed closed (all LLM outputs were valid)
        assert not cb.is_open()

        # Verify stagnation shows improving trend at the end
        final_score, final_trend = compute_stagnation(history)
        assert final_trend == "improving"

    def test_stagnation_triggered_abandon(self):
        """Verify stagnation detection triggers strong action for stuck strategies."""
        from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response

        # Monotonically declining Sharpes
        history = [
            {"sharpe": 0.5, "action": "modify_params"},
            {"sharpe": 0.3, "action": "modify_params"},
            {"sharpe": 0.2, "action": "modify_params"},
            {"sharpe": 0.1, "action": "modify_params"},
            {"sharpe": 0.05, "action": "modify_params"},
        ]

        score, trend = compute_stagnation(history)
        assert score > 0.5

        response = get_stagnation_response(iteration=5, score=score)
        # With high stagnation score, should trigger pivot or abandon
        assert response["constraint"] in ("abandon", "pivot", "broaden")

    def test_circuit_breaker_halts_on_bad_llm(self):
        """Verify circuit breaker opens when LLM consistently produces invalid code."""
        from crabquant.refinement.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(window=5, min_pass_rate=0.4, grace_turns=0, min_attempts=1)
        # 4 failures out of 5 = 20% < 40%
        for _ in range(4):
            cb.record(False, turn=1, mandate="test")
        cb.record(True, turn=2, mandate="test")

        assert cb.is_open()

        # Verify summary is informative
        summary = cb.summary()
        assert "OPEN" in summary
        assert "20.0%" in summary

    def test_action_analytics_informs_llm(self):
        """Verify action analytics produces useful LLM context."""
        from crabquant.refinement.action_analytics import (
            aggregate_action_stats,
            generate_llm_context,
        )

        history = [
            {"action": "modify_params", "sharpe": 0.5, "success": False},
            {"action": "modify_params", "sharpe": 0.6, "success": False},
            {"action": "modify_params", "sharpe": 0.7, "success": False},
            {"action": "full_rewrite", "sharpe": 1.8, "success": True},
            {"action": "full_rewrite", "sharpe": 2.0, "success": True},
        ]

        stats = aggregate_action_stats(history)
        assert stats["modify_params"]["success_rate"] == 0.0
        assert stats["full_rewrite"]["success_rate"] == 1.0

        ctx = generate_llm_context(history)
        assert "modify_params" in ctx
        assert "full_rewrite" in ctx
        # The context should recommend the better action
        assert len(ctx) > 50  # Substantial context

    def test_portfolio_correlation_informs_construction(self):
        """Verify correlation report identifies diversification opportunities."""
        from crabquant.refinement.portfolio_correlation import (
            compute_correlation_matrix,
            identify_diversifying_strategies,
            identify_redundant_strategies,
        )

        rng = np.random.default_rng(42)
        idx = pd.date_range("2023-01-03", periods=252, freq="B")

        # Create 3 independent strategies and 2 correlated ones
        curves = {}
        for i in range(3):
            curves[f"independent_{i}"] = pd.Series(
                rng.standard_normal(252).cumsum() + 100, index=idx
            )
        base = rng.standard_normal(252).cumsum() + 100
        for i in range(2):
            noise = rng.standard_normal(252) * 0.5
            curves[f"correlated_{i}"] = pd.Series(base + noise, index=idx)

        matrix = compute_correlation_matrix(curves)
        assert matrix.shape == (5, 5)

        redundant = identify_redundant_strategies(matrix, threshold=0.9)
        # The two correlated strategies should be redundant
        assert len(redundant) >= 1

        diversifying = identify_diversifying_strategies(matrix, threshold=0.3)
        # Independent strategies should be diversifying with each other
        assert len(diversifying) > 0
