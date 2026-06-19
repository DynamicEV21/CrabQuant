"""Tests for Phase 5.6.3 — Soft-Promote Tier.

Covers:
- soft_promote() function in promotion.py
- Candidate file creation and format
- Threshold enforcement (Sharpe, windows)
- Regime-specific lower threshold
- Edge cases (validation passed, no walk-forward data, etc.)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crabquant.refinement.promotion import soft_promote, _get_candidates_dir


# ─── Test Fixtures ─────────────────────────────────────────────────────────

@dataclass
class FakeBacktestResult:
    """Minimal BacktestResult stand-in."""
    sharpe: float = 1.5
    total_return: float = 0.2
    max_drawdown: float = 0.1
    num_trades: int = 40
    ticker: str = "SPY"
    params: dict = field(default_factory=lambda: {"fast": 10, "slow": 30})


@dataclass
class FakeRunState:
    """Minimal RunState stand-in."""
    mandate_name: str = "test_mandate"
    run_id: str = "test_run_001"
    current_turn: int = 3


@pytest.fixture
def strategy_code():
    return (
        "DESCRIPTION = 'Test strategy'\n"
        "DEFAULT_PARAMS = {'fast': 10}\n"
        "def generate_signals(df, params):\n"
        "    import pandas as pd\n"
        "    return pd.Series(False, index=df.index), pd.Series(False, index=df.index)\n"
    )


@pytest.fixture
def strategy_module(strategy_code):
    """Create a mock strategy module."""
    import types
    mod = types.ModuleType("fake_strategy")
    exec(strategy_code, mod.__dict__)
    mod.generate_signals = mod.generate_signals
    mod.DEFAULT_PARAMS = {"fast": 10}
    mod.DESCRIPTION = "Test strategy"
    return mod


@pytest.fixture
def result():
    return FakeBacktestResult()


@pytest.fixture
def state():
    return FakeRunState()


@pytest.fixture
def passing_validation():
    """Validation that passed — soft_promote should skip."""
    return {"passed": True, "walk_forward": {"avg_test_sharpe": 1.0, "windows_passed": 4}}


@pytest.fixture
def failing_validation_good_metrics():
    """Validation that failed but has decent walk-forward metrics."""
    return {
        "passed": False,
        "walk_forward_robust": False,
        "cross_ticker_robust": False,
        "walk_forward": {
            "avg_test_sharpe": 0.7,
            "min_test_sharpe": 0.3,
            "avg_degradation": 0.2,
            "num_windows": 6,
            "windows_passed": 3,
            "robust": False,
            "notes": [],
            "window_results": [],
        },
    }


@pytest.fixture
def failing_validation_poor_metrics():
    """Validation that failed with poor metrics."""
    return {
        "passed": False,
        "walk_forward": {
            "avg_test_sharpe": 0.1,
            "windows_passed": 0,
            "num_windows": 6,
        },
    }


@pytest.fixture
def failing_validation_no_wf():
    """Validation with no walk-forward data."""
    return {"passed": False, "error": "no data"}


# ─── soft_promote: basic behavior ──────────────────────────────────────────

class TestSoftPromoteBasic:
    """Tests for basic soft_promote behavior."""

    def test_skip_when_validation_passed(self, strategy_code, strategy_module, result, state, passing_validation):
        """Should not soft-promote when full validation already passed."""
        with patch("crabquant.refinement.promotion._get_candidates_dir") as mock_dir:
            mock_dir.return_value = Path("/tmp/nonexistent_candidates")
            resp = soft_promote(
                strategy_code, strategy_module, result, passing_validation, state,
            )
        assert resp["promoted"] is False
        assert "auto_promote" in resp["reason"]

    def test_skip_when_no_walk_forward_data(self, strategy_code, strategy_module, result, state, failing_validation_no_wf):
        resp = soft_promote(
            strategy_code, strategy_module, result, failing_validation_no_wf, state,
        )
        assert resp["promoted"] is False
        assert "walk-forward" in resp["reason"].lower()

    def test_skip_when_sharpe_too_low(self, strategy_code, strategy_module, result, state, failing_validation_poor_metrics):
        resp = soft_promote(
            strategy_code, strategy_module, result, failing_validation_poor_metrics, state,
        )
        assert resp["promoted"] is False
        assert "below threshold" in resp["reason"]

    def test_promotes_when_criteria_met(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        """Should create a candidate file when Sharpe and windows thresholds are met."""
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, failing_validation_good_metrics, state,
            )
        assert resp["promoted"] is True
        assert resp["candidate_file"] is not None
        assert resp["avg_test_sharpe"] == 0.7
        assert resp["windows_passed"] == 3

    def test_candidate_file_exists(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, failing_validation_good_metrics, state,
            )
        candidate_path = Path(resp["candidate_file"])
        assert candidate_path.exists()

    def test_candidate_file_is_valid_json(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, failing_validation_good_metrics, state,
            )
        candidate_path = Path(resp["candidate_file"])
        data = json.loads(candidate_path.read_text())
        assert data["name"] == "refined_test_mandate"
        assert data["needs_ongoing_validation"] is True

    def test_candidate_file_contains_required_fields(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        required = {"name", "timestamp", "needs_ongoing_validation", "avg_test_sharpe",
                     "windows_passed", "total_windows", "strategy_code", "discovery_ticker",
                     "backtest_sharpe", "backtest_trades", "backtest_max_drawdown",
                     "refinement_run", "validation_summary"}
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, failing_validation_good_metrics, state,
            )
        data = json.loads(Path(resp["candidate_file"]).read_text())
        for field in required:
            assert field in data, f"Missing required field: {field}"


# ─── soft_promote: threshold enforcement ───────────────────────────────────

class TestSoftPromoteThresholds:
    """Tests for Sharpe and windows threshold enforcement."""

    def test_min_sharpe_threshold(self, strategy_code, strategy_module, result, state, tmp_path):
        """Strategy with Sharpe just below threshold should not be promoted."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 0.49, "windows_passed": 3, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, validation, state,
                min_sharpe=0.5,
            )
        assert resp["promoted"] is False
        assert "0.49" in resp["reason"]

    def test_min_sharpe_exactly_at_threshold(self, strategy_code, strategy_module, result, state, tmp_path):
        """Strategy with Sharpe exactly at threshold should be promoted."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 0.5, "windows_passed": 3, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, validation, state,
                min_sharpe=0.5,
            )
        assert resp["promoted"] is True

    def test_min_windows_threshold(self, strategy_code, strategy_module, result, state, tmp_path):
        """Strategy with too few windows passed should not be promoted."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 1.0, "windows_passed": 1, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, validation, state,
                min_windows=2,
            )
        assert resp["promoted"] is False
        assert "1 below minimum 2" in resp["reason"]

    def test_min_windows_exactly_at_threshold(self, strategy_code, strategy_module, result, state, tmp_path):
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 1.0, "windows_passed": 2, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, validation, state,
                min_windows=2,
            )
        assert resp["promoted"] is True


# ─── soft_promote: regime-specific ─────────────────────────────────────────

class TestSoftPromoteRegimeSpecific:
    """Tests for regime-specific lower threshold behavior."""

    def test_regime_specific_uses_lower_threshold(self, strategy_code, strategy_module, result, state, tmp_path):
        """Regime-specific strategy should be promoted at lower Sharpe."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 0.35, "windows_passed": 3, "num_windows": 6},
        }
        # Non-regime-specific: 0.35 < 0.5 → should NOT promote
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp_normal = soft_promote(
                strategy_code, strategy_module, result, validation, state,
                min_sharpe=0.5, is_regime_specific=False,
            )
        assert resp_normal["promoted"] is False

        # Regime-specific: 0.35 >= 0.3 (soft floor) → should promote
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp_regime = soft_promote(
                strategy_code, strategy_module, result, validation, state,
                min_sharpe=0.5, is_regime_specific=True,
            )
        assert resp_regime["promoted"] is True

    def test_regime_specific_still_requires_min_windows(self, strategy_code, strategy_module, result, state, tmp_path):
        """Even regime-specific strategies need minimum windows."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 0.5, "windows_passed": 0, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, validation, state,
                min_sharpe=0.5, min_windows=2, is_regime_specific=True,
            )
        assert resp["promoted"] is False


# ─── soft_promote: response format ─────────────────────────────────────────

class TestSoftPromoteResponse:
    """Tests for response dict format."""

    def test_response_has_all_keys(self, strategy_code, strategy_module, result, state):
        validation = {"passed": False}
        resp = soft_promote(strategy_code, strategy_module, result, validation, state)
        expected_keys = {"promoted", "candidate_file", "avg_test_sharpe", "windows_passed", "reason", "error"}
        assert set(resp.keys()) == expected_keys

    def test_not_promoted_has_none_file(self, strategy_code, strategy_module, result, state):
        validation = {"passed": False}
        resp = soft_promote(strategy_code, strategy_module, result, validation, state)
        assert resp["candidate_file"] is None

    def test_promoted_response_has_metrics(self, strategy_code, strategy_module, result, state, tmp_path):
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 0.8, "windows_passed": 4, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(strategy_code, strategy_module, result, validation, state)
        assert resp["avg_test_sharpe"] == 0.8
        assert resp["windows_passed"] == 4
