"""Tests for crabquant.production.strategy_decay"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import numpy as np
import pandas as pd
import pytest

from crabquant.production.strategy_decay import (
    DecayCheckResult,
    check_strategy_decay,
    check_all_strategies_decay,
    load_decay_state,
    save_decay_state,
    format_decay_report,
    _load_registry,
    _detect_regime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backtest_result(sharpe: float = 1.5) -> MagicMock:
    """Create a mock BacktestResult with the given Sharpe."""
    result = MagicMock()
    result.sharpe = sharpe
    result.total_return = 0.10
    result.max_drawdown = -0.05
    result.num_trades = 20
    result.passed = sharpe >= 1.5
    result.notes = ""
    return result


def _make_signals_df(n: int = 126) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    dates = pd.bdate_range("2025-01-01", periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        },
        index=dates,
    )


def _make_entries_exits(n: int = 126) -> tuple[pd.Series, pd.Series]:
    """Create minimal boolean entry/exit series."""
    dates = pd.bdate_range("2025-01-01", periods=n)
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)
    # A few trades
    entries.iloc[10] = True
    exits.iloc[20] = True
    entries.iloc[40] = True
    exits.iloc[50] = True
    return entries, exits


# ---------------------------------------------------------------------------
# load_decay_state / save_decay_state
# ---------------------------------------------------------------------------

class TestLoadDecayState:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_decay_state(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_valid_file(self, tmp_path):
        state = {"strat_a": {"consecutive_decayed": 2}}
        p = tmp_path / "state.json"
        p.write_text(json.dumps(state))
        result = load_decay_state(str(p))
        assert result == state

    def test_invalid_json_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json {{{")
        result = load_decay_state(str(p))
        assert result == {}

    def test_non_dict_returns_empty(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text(json.dumps([1, 2, 3]))
        result = load_decay_state(str(p))
        assert result == {}


class TestSaveDecayState:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "sub" / "state.json"
        save_decay_state(str(p), {"a": 1})
        assert p.exists()
        data = json.loads(p.read_text())
        assert data == {"a": 1}

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "state.json"
        save_decay_state(str(p), {"x": 10})
        assert p.parent.exists()

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "state.json"
        save_decay_state(str(p), {"v": 1})
        save_decay_state(str(p), {"v": 2})
        data = json.loads(p.read_text())
        assert data["v"] == 2

    def test_roundtrip(self, tmp_path):
        p = tmp_path / "state.json"
        original = {
            "strat_a": {"consecutive_decayed": 3, "current_sharpe": 0.5},
            "strat_b": {"consecutive_decayed": 0, "current_sharpe": 2.0},
        }
        save_decay_state(str(p), original)
        loaded = load_decay_state(str(p))
        assert loaded == original


# ---------------------------------------------------------------------------
# check_strategy_decay — no decay (good performance)
# ---------------------------------------------------------------------------

class TestNoDecayGoodPerformance:
    @patch("crabquant.production.strategy_decay._detect_regime", return_value="TRENDING")
    def test_current_sharpe_equals_promotion(self, mock_regime):
        """If current Sharpe matches promotion, no decay."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=2.0)

        with patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "test_strat": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result):
            result = check_strategy_decay("test_strat", promotion_sharpe=2.0)

        assert result.strategy_name == "test_strat"
        assert result.current_sharpe == 2.0
        assert result.sharpe_decline_pct == 0.0
        assert result.is_decayed is False
        assert result.should_retire is False
        assert result.current_regime == "TRENDING"

    @patch("crabquant.production.strategy_decay._detect_regime", return_value="LOW_VOLATILITY")
    def test_current_sharpe_better_than_promotion(self, mock_regime):
        """If current Sharpe exceeds promotion, no decay."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=2.5)

        with patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "good_strat": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result):
            result = check_strategy_decay("good_strat", promotion_sharpe=2.0)

        assert result.current_sharpe == 2.5
        # decline_pct should be 0 (or negative, clamped to 0 effectively)
        assert result.is_decayed is False

    def test_strategy_not_in_registry_returns_not_decayed(self):
        """Unknown strategy should return is_decayed=False gracefully."""
        with patch("crabquant.strategies.STRATEGY_REGISTRY", {}):
            result = check_strategy_decay("nonexistent", promotion_sharpe=2.0)

        assert result.strategy_name == "nonexistent"
        assert result.is_decayed is False
        assert result.current_sharpe == 0.0

    def test_backtest_error_returns_not_decayed(self):
        """If data loading fails, return not-decayed rather than crashing."""
        with patch("crabquant.data.load_data", side_effect=ConnectionError("no network")), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "any": (MagicMock(), {}, {}, "", MagicMock()),
             }):
            result = check_strategy_decay("any", promotion_sharpe=2.0)

        assert result.is_decayed is False
        assert result.current_sharpe == 0.0


# ---------------------------------------------------------------------------
# check_strategy_decay — decay detected (Sharpe drop)
# ---------------------------------------------------------------------------

class TestDecayDetectedSharpeDrop:
    @patch("crabquant.production.strategy_decay._detect_regime", return_value="VOLATILE")
    def test_sharpe_drops_below_threshold(self, mock_regime):
        """50% Sharpe drop with 30% threshold should be decayed."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=1.0)

        with patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "decaying": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result):
            result = check_strategy_decay("decaying", promotion_sharpe=2.0, decay_threshold=0.30)

        assert result.is_decayed is True
        assert result.sharpe_decline_pct == pytest.approx(0.50, rel=1e-3)
        assert result.current_regime == "VOLATILE"

    def test_sharpe_drop_below_threshold_not_decayed(self):
        """A 10% drop with 30% threshold should NOT be decayed."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=1.8)

        with patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "slight_drop": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result):
            result = check_strategy_decay("slight_drop", promotion_sharpe=2.0, decay_threshold=0.30)

        assert result.is_decayed is False
        assert result.sharpe_decline_pct == pytest.approx(0.10, rel=1e-3)


# ---------------------------------------------------------------------------
# Decay threshold edge case (exactly at threshold)
# ---------------------------------------------------------------------------

class TestDecayThresholdEdgeCase:
    @patch("crabquant.production.strategy_decay._detect_regime", return_value="TRENDING")
    def test_exactly_at_threshold_is_decayed(self, mock_regime):
        """Exactly 30% decline with 30% threshold should be decayed (>=)."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        # promotion_sharpe=2.0, 30% decline => current_sharpe=1.4
        mock_result = _make_backtest_result(sharpe=1.4)

        with patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "edge": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result):
            result = check_strategy_decay("edge", promotion_sharpe=2.0, decay_threshold=0.30)

        assert result.is_decayed is True
        assert result.sharpe_decline_pct == pytest.approx(0.30, rel=1e-3)


# ---------------------------------------------------------------------------
# Consecutive checks and retirement
# ---------------------------------------------------------------------------

class TestConsecutiveRequiredBeforeRetire:
    def test_single_decay_does_not_retire(self, tmp_path):
        """One decayed check with consecutive_required=3 should not retire."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=0.5)

        state_file = tmp_path / "state.json"
        registry = [
            {"strategy_name": "some_strat", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "active"},
        ]

        with patch("crabquant.production.strategy_decay._load_registry", return_value=registry), \
             patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "some_strat": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result), \
             patch("crabquant.production.strategy_decay._detect_regime", return_value="TRENDING"):
            results = check_all_strategies_decay(
                decay_state_file=str(state_file),
                decay_threshold=0.30,
                consecutive_required=3,
            )

        assert len(results) == 1
        assert results[0].is_decayed is True
        assert results[0].consecutive_decayed_checks == 1
        assert results[0].should_retire is False

    def test_exactly_consecutive_required_retires(self, tmp_path):
        """Three consecutive decayed checks with consecutive_required=3 should retire."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=0.5)

        state_file = tmp_path / "state.json"
        registry = [
            {"strategy_name": "retiring_strat", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "active"},
        ]

        # Pre-seed state with 2 consecutive decayed checks
        initial_state = {
            "retiring_strat": {"consecutive_decayed": 2},
        }
        save_decay_state(str(state_file), initial_state)

        with patch("crabquant.production.strategy_decay._load_registry", return_value=registry), \
             patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "retiring_strat": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result), \
             patch("crabquant.production.strategy_decay._detect_regime", return_value="TRENDING"):
            results = check_all_strategies_decay(
                decay_state_file=str(state_file),
                decay_threshold=0.30,
                consecutive_required=3,
            )

        assert len(results) == 1
        assert results[0].consecutive_decayed_checks == 3
        assert results[0].should_retire is True


# ---------------------------------------------------------------------------
# Consecutive reset on recovery
# ---------------------------------------------------------------------------

class TestConsecutiveResetOnRecovery:
    def test_recovery_resets_counter(self, tmp_path):
        """If a previously-decaying strategy recovers, counter resets to 0."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=2.0)  # back to normal

        state_file = tmp_path / "state.json"
        registry = [
            {"strategy_name": "recovered", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "active"},
        ]

        # Pre-seed with 2 consecutive decayed checks
        initial_state = {
            "recovered": {"consecutive_decayed": 2},
        }
        save_decay_state(str(state_file), initial_state)

        with patch("crabquant.production.strategy_decay._load_registry", return_value=registry), \
             patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "recovered": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result), \
             patch("crabquant.production.strategy_decay._detect_regime", return_value="TRENDING"):
            results = check_all_strategies_decay(
                decay_state_file=str(state_file),
                decay_threshold=0.30,
                consecutive_required=3,
            )

        assert len(results) == 1
        assert results[0].is_decayed is False
        assert results[0].consecutive_decayed_checks == 0
        assert results[0].should_retire is False

        # Verify state was persisted
        state = load_decay_state(str(state_file))
        assert state["recovered"]["consecutive_decayed"] == 0


# ---------------------------------------------------------------------------
# Decay state persistence
# ---------------------------------------------------------------------------

class TestDecayStatePersistence:
    def test_state_saved_after_check(self, tmp_path):
        """Decay state file should be created/updated after batch check."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=1.8)

        state_file = tmp_path / "state.json"
        registry = [
            {"strategy_name": "persist_test", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "active"},
        ]

        with patch("crabquant.production.strategy_decay._load_registry", return_value=registry), \
             patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "persist_test": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result), \
             patch("crabquant.production.strategy_decay._detect_regime", return_value="TRENDING"):
            check_all_strategies_decay(
                decay_state_file=str(state_file),
                decay_threshold=0.30,
                consecutive_required=3,
            )

        assert state_file.exists()
        state = load_decay_state(str(state_file))
        assert "persist_test" in state
        assert "consecutive_decayed" in state["persist_test"]
        assert "last_check" in state["persist_test"]

    def test_state_accumulates_across_runs(self, tmp_path):
        """Consecutive count should accumulate across multiple check runs."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=0.5)  # always decayed

        state_file = tmp_path / "state.json"
        registry = [
            {"strategy_name": "accumulate", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "active"},
        ]

        for expected_count in range(1, 4):
            with patch("crabquant.production.strategy_decay._load_registry", return_value=registry), \
                 patch("crabquant.data.load_data", return_value=df), \
                 patch("crabquant.strategies.STRATEGY_REGISTRY", {
                     "accumulate": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
                 }), \
                 patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result), \
                 patch("crabquant.production.strategy_decay._detect_regime", return_value="TRENDING"):
                results = check_all_strategies_decay(
                    decay_state_file=str(state_file),
                    decay_threshold=0.30,
                    consecutive_required=3,
                )

            assert results[0].consecutive_decayed_checks == expected_count

        # After 3 runs, should retire
        state = load_decay_state(str(state_file))
        assert state["accumulate"]["should_retire"] is True


# ---------------------------------------------------------------------------
# All strategies check
# ---------------------------------------------------------------------------

class TestAllStrategiesCheck:
    def test_multiple_strategies_sorted_by_severity(self, tmp_path):
        """Results should be sorted by sharpe_decline_pct descending."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()

        registry = [
            {"strategy_name": "mild", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "active"},
            {"strategy_name": "severe", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "active"},
            {"strategy_name": "healthy", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "active"},
        ]

        mock_results = {
            "mild": _make_backtest_result(sharpe=1.5),     # 25% decline
            "severe": _make_backtest_result(sharpe=0.8),    # 60% decline
            "healthy": _make_backtest_result(sharpe=2.2),   # -10% decline (better)
        }

        def mock_signal_fn(d, p):
            return entries, exits

        state_file = tmp_path / "state.json"

        with patch("crabquant.production.strategy_decay._load_registry", return_value=registry), \
             patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 name: (mock_signal_fn, {}, {}, "", MagicMock()) for name in ["mild", "severe", "healthy"]
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", side_effect=lambda *a, **kw: mock_results[kw.get("strategy_name", a[3] if len(a) > 3 else "")]), \
             patch("crabquant.production.strategy_decay._detect_regime", return_value="TRENDING"):
            results = check_all_strategies_decay(
                decay_state_file=str(state_file),
                decay_threshold=0.30,
                consecutive_required=3,
            )

        assert len(results) == 3
        # Most decayed first
        assert results[0].strategy_name == "severe"
        assert results[1].strategy_name == "mild"
        assert results[2].strategy_name == "healthy"


# ---------------------------------------------------------------------------
# Empty registry
# ---------------------------------------------------------------------------

class TestEmptyRegistryGraceful:
    def test_empty_registry_returns_empty_list(self, tmp_path):
        """Empty registry should return empty results without errors."""
        state_file = tmp_path / "state.json"

        with patch("crabquant.production.strategy_decay._load_registry", return_value=[]):
            results = check_all_strategies_decay(
                decay_state_file=str(state_file),
            )

        assert results == []

    def test_all_retired_strategies_skipped(self, tmp_path):
        """Registry with only retired strategies should return empty results."""
        state_file = tmp_path / "state.json"
        registry = [
            {"strategy_name": "old_a", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "retired"},
            {"strategy_name": "old_b", "promotion_sharpe": 1.8, "ticker": "SPY", "params": {}, "status": "inactive"},
        ]

        with patch("crabquant.production.strategy_decay._load_registry", return_value=registry):
            results = check_all_strategies_decay(
                decay_state_file=str(state_file),
            )

        assert results == []


# ---------------------------------------------------------------------------
# Retirement marks inactive
# ---------------------------------------------------------------------------

class TestRetirementMarksInactive:
    def test_should_retire_updates_status_in_state(self, tmp_path):
        """When should_retire is True, state should record status as retired."""
        df = _make_signals_df()
        entries, exits = _make_entries_exits()
        mock_result = _make_backtest_result(sharpe=0.5)

        state_file = tmp_path / "state.json"
        registry = [
            {"strategy_name": "doomed", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "active"},
        ]

        # Pre-seed with 2 consecutive decayed checks so this one triggers retirement
        initial_state = {"doomed": {"consecutive_decayed": 2}}
        save_decay_state(str(state_file), initial_state)

        with patch("crabquant.production.strategy_decay._load_registry", return_value=registry), \
             patch("crabquant.data.load_data", return_value=df), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "doomed": (MagicMock(return_value=(entries, exits)), {}, {}, "", MagicMock()),
             }), \
             patch("crabquant.engine.backtest.BacktestEngine.run", return_value=mock_result), \
             patch("crabquant.production.strategy_decay._detect_regime", return_value="VOLATILE"):
            results = check_all_strategies_decay(
                decay_state_file=str(state_file),
                decay_threshold=0.30,
                consecutive_required=3,
            )

        assert results[0].should_retire is True

        # Verify state marks as retired
        state = load_decay_state(str(state_file))
        assert state["doomed"]["status"] == "retired"
        assert state["doomed"]["should_retire"] is True

    def test_retired_strategy_not_rechecked(self, tmp_path):
        """Strategy marked as retired in registry should be skipped."""
        state_file = tmp_path / "state.json"
        registry = [
            {"strategy_name": "already_dead", "promotion_sharpe": 2.0, "ticker": "SPY", "params": {}, "status": "retired"},
        ]

        with patch("crabquant.production.strategy_decay._load_registry", return_value=registry):
            results = check_all_strategies_decay(
                decay_state_file=str(state_file),
            )

        assert results == []


# ---------------------------------------------------------------------------
# format_decay_report
# ---------------------------------------------------------------------------

class TestFormatDecayReport:
    def test_empty_results(self):
        report = format_decay_report([])
        assert "No strategies checked" in report

    def test_all_healthy(self):
        results = [
            DecayCheckResult(
                strategy_name="good_a",
                promotion_sharpe=2.0,
                current_sharpe=2.2,
                sharpe_decline_pct=-0.10,
                current_regime="TRENDING",
                is_decayed=False,
                consecutive_decayed_checks=0,
                should_retire=False,
            ),
        ]
        report = format_decay_report(results)
        assert "HEALTHY" in report
        assert "good_a" in report
        assert "No action required" in report

    def test_decayed_with_retirement(self):
        results = [
            DecayCheckResult(
                strategy_name="bad_strat",
                promotion_sharpe=2.0,
                current_sharpe=0.8,
                sharpe_decline_pct=0.60,
                current_regime="VOLATILE",
                is_decayed=True,
                consecutive_decayed_checks=3,
                should_retire=True,
            ),
        ]
        report = format_decay_report(results)
        assert "DECAYED" in report
        assert "bad_strat" in report
        assert "RETIRE" in report
        assert "ACTION REQUIRED" in report

    def test_decayed_watching_not_retired(self):
        results = [
            DecayCheckResult(
                strategy_name="watching",
                promotion_sharpe=2.0,
                current_sharpe=1.2,
                sharpe_decline_pct=0.40,
                current_regime="TRENDING",
                is_decayed=True,
                consecutive_decayed_checks=1,
                should_retire=False,
            ),
        ]
        report = format_decay_report(results)
        assert "WATCH" in report
        assert "RETIRE" not in report
        assert "ATTENTION" in report

    def test_mixed_results(self):
        results = [
            DecayCheckResult(
                strategy_name="healthy", promotion_sharpe=2.0, current_sharpe=2.5,
                sharpe_decline_pct=-0.25, current_regime="TRENDING",
                is_decayed=False, consecutive_decayed_checks=0, should_retire=False,
            ),
            DecayCheckResult(
                strategy_name="decayed", promotion_sharpe=2.0, current_sharpe=1.0,
                sharpe_decline_pct=0.50, current_regime="VOLATILE",
                is_decayed=True, consecutive_decayed_checks=2, should_retire=False,
            ),
            DecayCheckResult(
                strategy_name="retire_me", promotion_sharpe=2.0, current_sharpe=0.5,
                sharpe_decline_pct=0.75, current_regime="VOLATILE",
                is_decayed=True, consecutive_decayed_checks=3, should_retire=True,
            ),
        ]
        report = format_decay_report(results)
        assert "Checked: 3 strategies" in report
        assert "DECAYED (2)" in report
        assert "HEALTHY (1)" in report
        assert "healthy" in report
        assert "decayed" in report
        assert "retire_me" in report

    def test_report_contains_sharpe_values(self):
        results = [
            DecayCheckResult(
                strategy_name="test", promotion_sharpe=1.8, current_sharpe=0.9,
                sharpe_decline_pct=0.50, current_regime="MEAN_REV",
                is_decayed=True, consecutive_decayed_checks=1, should_retire=False,
            ),
        ]
        report = format_decay_report(results)
        assert "1.80" in report or "1.8" in report
        assert "0.90" in report or "0.9" in report
        assert "50.0%" in report or "50%" in report


# ---------------------------------------------------------------------------
# _detect_regime
# ---------------------------------------------------------------------------

class TestDetectRegime:
    def test_returns_unknown_on_failure(self):
        with patch("crabquant.regime.detect_regime", side_effect=Exception("fail")):
            result = _detect_regime()
            assert result == "unknown"

    def test_returns_regime_name_on_success(self):
        mock_regime = MagicMock()
        mock_regime.name = "TRENDING"
        with patch("crabquant.data.load_data", return_value=_make_signals_df()), \
             patch("crabquant.regime.detect_regime", return_value=(mock_regime, {"confidence": 0.9})):
            result = _detect_regime()
            assert result == "TRENDING"


# ---------------------------------------------------------------------------
# _load_registry
# ---------------------------------------------------------------------------

class TestLoadRegistry:
    def test_results_registry_json(self, tmp_path):
        """Loads from results/STRATEGY_REGISTRY.json."""
        registry_data = [
            {"strategy_name": "a", "promotion_sharpe": 2.0},
        ]
        reg_file = tmp_path / "results" / "STRATEGY_REGISTRY.json"
        reg_file.parent.mkdir(parents=True)
        reg_file.write_text(json.dumps(registry_data))

        with patch("crabquant.production.strategy_decay.BASE_DIR", tmp_path):
            result = _load_registry()

        assert len(result) == 1
        assert result[0]["strategy_name"] == "a"

    def test_production_registry_json(self, tmp_path):
        """Falls back to strategies/production/registry.json."""
        prod_file = tmp_path / "strategies" / "production" / "registry.json"
        prod_file.parent.mkdir(parents=True)
        prod_file.write_text(json.dumps([
            {"strategy_name": "b", "ticker": "QQQ", "params": {"fast": 10}},
        ]))

        # No results/STRATEGY_REGISTRY.json exists
        with patch("crabquant.production.strategy_decay.BASE_DIR", tmp_path):
            result = _load_registry()

        assert len(result) == 1
        assert result[0]["strategy_name"] == "b"
        assert result[0]["ticker"] == "QQQ"

    def test_fallback_to_code_registry(self, tmp_path):
        """If no JSON files, falls back to in-code STRATEGY_REGISTRY."""
        # No JSON files exist at all
        with patch("crabquant.production.strategy_decay.BASE_DIR", tmp_path), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {
                 "rsi_crossover": (MagicMock(), {}, {}, "", MagicMock()),
                 "macd_momentum": (MagicMock(), {}, {}, "", MagicMock()),
             }):
            result = _load_registry()

        assert len(result) == 2
        names = {r["strategy_name"] for r in result}
        assert "rsi_crossover" in names
        assert "macd_momentum" in names

    def test_empty_when_nothing_available(self, tmp_path):
        """Returns empty list when nothing is available."""
        with patch("crabquant.production.strategy_decay.BASE_DIR", tmp_path), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {}):
            result = _load_registry()

        assert result == []
