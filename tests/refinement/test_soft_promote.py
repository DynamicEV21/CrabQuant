"""Tests for Phase 5.6.3 — Soft-Promote Tier.

Covers:
- soft_promote() function in promotion.py
- Candidate file creation and format
- Threshold enforcement (Sharpe, windows)
- Regime-specific lower threshold
- Edge cases (validation passed, no walk-forward data, etc.)
- run_full_validation_check() variations
- _update_winner_status and _update_winner_regime_tags
- promote_to_winner
- is_already_registered and register_strategy
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crabquant.refinement.promotion import (
    _get_candidates_dir,
    _get_winners_path,
    _update_winner_regime_tags,
    _update_winner_status,
    is_already_registered,
    promote_to_winner,
    register_strategy,
    run_full_validation_check,
    soft_promote,
)


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
        for field_name in required:
            assert field_name in data, f"Missing required field: {field_name}"


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

    def test_regime_specific_below_soft_floor_rejected(self, strategy_code, strategy_module, result, state, tmp_path):
        """Regime-specific strategy below soft_promote_test_sharpe floor is rejected."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 0.1, "windows_passed": 3, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, validation, state,
                min_sharpe=0.5, is_regime_specific=True,
            )
        assert resp["promoted"] is False
        assert "below threshold" in resp["reason"]


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


# ─── soft_promote: additional edge cases ───────────────────────────────────

class TestSoftPromoteEdgeCases:
    """Additional edge cases for soft_promote."""

    def test_zero_sharpe_rejected(self, strategy_code, strategy_module, result, state, tmp_path):
        """Strategy with zero Sharpe should be rejected."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 0.0, "windows_passed": 5, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(strategy_code, strategy_module, result, validation, state)
        assert resp["promoted"] is False

    def test_negative_sharpe_rejected(self, strategy_code, strategy_module, result, state, tmp_path):
        """Strategy with negative Sharpe should be rejected."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": -0.5, "windows_passed": 5, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(strategy_code, strategy_module, result, validation, state)
        assert resp["promoted"] is False

    def test_very_high_sharpe_promoted(self, strategy_code, strategy_module, result, state, tmp_path):
        """Strategy with very high Sharpe should be promoted."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 5.0, "windows_passed": 6, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(strategy_code, strategy_module, result, validation, state)
        assert resp["promoted"] is True

    def test_candidate_file_includes_strategy_code(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        """Candidate file should include the full strategy source."""
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, failing_validation_good_metrics, state,
            )
        data = json.loads(Path(resp["candidate_file"]).read_text())
        assert data["strategy_code"] == strategy_code

    def test_candidate_file_includes_discovery_ticker(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        """Candidate file should include the discovery ticker."""
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, failing_validation_good_metrics, state,
            )
        data = json.loads(Path(resp["candidate_file"]).read_text())
        assert data["discovery_ticker"] == "SPY"

    def test_candidate_file_includes_is_regime_specific(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        """Candidate file should flag regime-specific strategies."""
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, failing_validation_good_metrics, state,
                is_regime_specific=True,
            )
        data = json.loads(Path(resp["candidate_file"]).read_text())
        assert data["is_regime_specific"] is True

    def test_candidate_file_has_backtest_metrics(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        """Candidate file should include backtest Sharpe, trades, drawdown."""
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, failing_validation_good_metrics, state,
            )
        data = json.loads(Path(resp["candidate_file"]).read_text())
        assert data["backtest_sharpe"] == 1.5
        assert data["backtest_trades"] == 40
        assert data["backtest_max_drawdown"] == 0.1

    def test_candidate_file_validation_summary(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        """Candidate file should include validation summary."""
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result, failing_validation_good_metrics, state,
            )
        data = json.loads(Path(resp["candidate_file"]).read_text())
        vs = data["validation_summary"]
        assert vs["walk_forward_robust"] is False
        assert vs["cross_ticker_robust"] is False
        assert "avg_degradation" in vs

    def test_walk_forward_missing_avg_test_sharpe_defaults_to_zero(self, strategy_code, strategy_module, result, state, tmp_path):
        """If avg_test_sharpe is missing from walk_forward, defaults to 0."""
        validation = {
            "passed": False,
            "walk_forward": {"windows_passed": 3, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(strategy_code, strategy_module, result, validation, state)
        assert resp["promoted"] is False  # 0.0 < 0.3

    def test_walk_forward_missing_windows_passed_defaults_to_zero(self, strategy_code, strategy_module, result, state, tmp_path):
        """If windows_passed is missing, defaults to 0."""
        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 1.0, "num_windows": 6},
        }
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(strategy_code, strategy_module, result, validation, state)
        assert resp["promoted"] is False  # 0 < 2

    def test_null_params_handled(self, strategy_code, strategy_module, result, state, failing_validation_good_metrics, tmp_path):
        """Strategy with None params should not crash."""
        result_none_params = FakeBacktestResult(params=None)
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=tmp_path):
            resp = soft_promote(
                strategy_code, strategy_module, result_none_params, failing_validation_good_metrics, state,
            )
        assert resp["promoted"] is True


# ─── run_full_validation_check ─────────────────────────────────────────────

class TestRunFullValidationCheck:
    """Tests for run_full_validation_check()."""

    def test_result_structure(self):
        """Result dict should have expected keys."""
        from unittest.mock import patch as p
        with p("crabquant.validation.rolling_walk_forward") as mock_rwf, \
             p("crabquant.validation.cross_ticker_validation") as mock_ct:
            mock_rwf.return_value = MagicMock(
                avg_test_sharpe=0.8, min_test_sharpe=0.5, avg_degradation=0.3,
                num_windows=4, windows_passed=3, robust=True, notes=[], window_results=[],
            )
            mock_ct.return_value = MagicMock(
                avg_sharpe=0.6, median_sharpe=0.5, robust=True,
                tickers_profitable=2, tickers_tested=3, notes=[],
            )
            result = run_full_validation_check(
                MagicMock(), {"fast": 10}, "SPY", ["SPY", "AAPL"],
                min_walk_forward_sharpe=0.5, min_cross_ticker_sharpe=0.5,
            )
        expected_keys = {"passed", "walk_forward_robust", "cross_ticker_robust",
                         "walk_forward", "cross_ticker", "error",
                         "validation_method", "is_regime_specific",
                         "deflated_sharpe", "complexity_score",
                         "time_reversed_overfit", "time_reversed_explanation"}
        assert set(result.keys()) == expected_keys

    def test_passed_when_both_pass(self):
        """Both walk-forward and cross-ticker pass → overall passed."""
        with patch("crabquant.validation.rolling_walk_forward") as mock_rwf, \
             patch("crabquant.validation.cross_ticker_validation") as mock_ct:
            mock_rwf.return_value = MagicMock(
                avg_test_sharpe=1.0, min_test_sharpe=0.6, avg_degradation=0.2,
                num_windows=4, windows_passed=4, robust=True, notes=[], window_results=[],
            )
            mock_ct.return_value = MagicMock(
                avg_sharpe=0.8, median_sharpe=0.7, robust=True,
                tickers_profitable=3, tickers_tested=3, notes=[],
            )
            result = run_full_validation_check(
                MagicMock(), {}, "SPY", ["SPY", "AAPL"],
            )
        assert result["passed"] is True

    def test_failed_when_wf_fails(self):
        """Walk-forward fails → overall failed."""
        with patch("crabquant.validation.rolling_walk_forward") as mock_rwf, \
             patch("crabquant.validation.cross_ticker_validation") as mock_ct:
            mock_rwf.return_value = MagicMock(
                avg_test_sharpe=0.2, min_test_sharpe=0.1, avg_degradation=0.5,
                num_windows=4, windows_passed=1, robust=False, notes=[], window_results=[],
            )
            mock_ct.return_value = MagicMock(
                avg_sharpe=0.8, median_sharpe=0.7, robust=True,
                tickers_profitable=3, tickers_tested=3, notes=[],
            )
            result = run_full_validation_check(
                MagicMock(), {}, "SPY", ["SPY", "AAPL"],
            )
        assert result["passed"] is False

    def test_validation_method_rolling_by_default(self):
        with patch("crabquant.validation.rolling_walk_forward") as mock_rwf, \
             patch("crabquant.validation.cross_ticker_validation"):
            mock_rwf.return_value = MagicMock(
                avg_test_sharpe=0.8, min_test_sharpe=0.5, avg_degradation=0.3,
                num_windows=4, windows_passed=3, robust=True, notes=[], window_results=[],
            )
            result = run_full_validation_check(
                MagicMock(), {}, "SPY", ["SPY"],
            )
        assert result["validation_method"] == "rolling"

    def test_validation_method_single_split(self):
        with patch("crabquant.validation.walk_forward_test") as mock_wf, \
             patch("crabquant.validation.cross_ticker_validation"):
            mock_wf.return_value = MagicMock(
                test_sharpe=0.8, train_sharpe=1.0, degradation=0.2,
                robust=True, notes=[], regime_shift=False,
                test_regime="bull", train_regime="bull",
            )
            result = run_full_validation_check(
                MagicMock(), {}, "SPY", ["SPY"], use_rolling=False,
            )
        assert result["validation_method"] == "single_split"

    def test_no_oos_tickers_passes_ct(self):
        """When only discovery ticker is in validation_tickers, ct passes by default."""
        with patch("crabquant.validation.rolling_walk_forward") as mock_rwf:
            mock_rwf.return_value = MagicMock(
                avg_test_sharpe=1.0, min_test_sharpe=0.5, avg_degradation=0.2,
                num_windows=4, windows_passed=4, robust=True, notes=[], window_results=[],
            )
            result = run_full_validation_check(
                MagicMock(), {}, "SPY", ["SPY"],
            )
        assert result["cross_ticker_robust"] is True
        assert result["cross_ticker"] is None

    def test_error_handled_gracefully(self):
        """If validation throws, error is captured in result."""
        with patch("crabquant.validation.rolling_walk_forward") as mock_rwf:
            mock_rwf.side_effect = RuntimeError("backtest crash")
            result = run_full_validation_check(
                MagicMock(), {}, "SPY", ["SPY"],
            )
        assert result["passed"] is False
        assert result["error"] == "backtest crash"

    def test_is_regime_specific_flag(self):
        with patch("crabquant.validation.rolling_walk_forward") as mock_rwf, \
             patch("crabquant.validation.cross_ticker_validation"):
            mock_rwf.return_value = MagicMock(
                avg_test_sharpe=1.0, min_test_sharpe=0.5, avg_degradation=0.2,
                num_windows=4, windows_passed=4, robust=True, notes=[], window_results=[],
            )
            result = run_full_validation_check(
                MagicMock(), {}, "SPY", ["SPY"], is_regime_specific=True,
            )
        assert result["is_regime_specific"] is True


# ─── _get_candidates_dir ──────────────────────────────────────────────────

class TestGetCandidatesDir:
    def test_returns_path(self):
        d = _get_candidates_dir()
        assert isinstance(d, Path)
        assert str(d).endswith("candidates")

    def test_path_matches_expected(self):
        assert str(_get_candidates_dir()) == "results/candidates"


# ─── _get_winners_path ────────────────────────────────────────────────────

class TestGetWinnersPath:
    def test_returns_path(self):
        p = _get_winners_path()
        assert isinstance(p, Path)
        assert str(p).endswith("winners.json")

    def test_path_matches_expected(self):
        assert str(_get_winners_path()) == "results/winners/winners.json"


# ─── _update_winner_status ────────────────────────────────────────────────

class TestUpdateWinnerStatus:
    def test_updates_existing_winner(self, tmp_path):
        winners = [
            {"strategy": "refined_test", "sharpe": 1.0, "validation_status": "backtest_only"},
            {"strategy": "other", "sharpe": 0.5, "validation_status": "backtest_only"},
        ]
        winners_path = tmp_path / "winners.json"
        winners_path.write_text(json.dumps(winners))

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_path):
            _update_winner_status("refined_test", "promoted")

        updated = json.loads(winners_path.read_text())
        assert updated[0]["validation_status"] == "promoted"
        assert updated[1]["validation_status"] == "backtest_only"

    def test_no_file_no_error(self):
        with patch("crabquant.refinement.promotion._get_winners_path", return_value=Path("/nonexistent/w.json")):
            _update_winner_status("test", "promoted")  # should not raise

    def test_invalid_json_no_error(self, tmp_path):
        winners_path = tmp_path / "bad.json"
        winners_path.write_text("not json")

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_path):
            _update_winner_status("test", "promoted")  # should not raise

    def test_strategy_not_found_no_change(self, tmp_path):
        winners = [{"strategy": "other", "sharpe": 1.0}]
        winners_path = tmp_path / "winners.json"
        winners_path.write_text(json.dumps(winners))

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_path):
            _update_winner_status("nonexistent", "promoted")

        data = json.loads(winners_path.read_text())
        assert data[0].get("validation_status") is None


# ─── _update_winner_regime_tags ───────────────────────────────────────────

class TestUpdateWinnerRegimeTags:
    def test_adds_regime_tags_to_winner(self, tmp_path):
        winners = [{"strategy": "refined_test", "sharpe": 1.0}]
        winners_path = tmp_path / "winners.json"
        winners_path.write_text(json.dumps(winners))

        tags = {
            "preferred_regimes": ["bull"],
            "acceptable_regimes": ["neutral"],
            "weak_regimes": ["bear"],
            "is_regime_specific": True,
        }
        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_path):
            _update_winner_regime_tags("refined_test", tags)

        data = json.loads(winners_path.read_text())
        assert "regime_tags" in data[0]
        assert data[0]["regime_tags"]["preferred_regimes"] == ["bull"]
        assert data[0]["regime_tags"]["is_regime_specific"] is True

    def test_no_file_no_error(self):
        with patch("crabquant.refinement.promotion._get_winners_path", return_value=Path("/nonexistent/w.json")):
            _update_winner_regime_tags("test", {})  # should not raise

    def test_updates_most_recent_entry(self, tmp_path):
        """When there are duplicate entries, the most recent one is updated."""
        winners = [
            {"strategy": "refined_test", "sharpe": 0.5},
            {"strategy": "refined_test", "sharpe": 1.0},
        ]
        winners_path = tmp_path / "winners.json"
        winners_path.write_text(json.dumps(winners))

        tags = {"preferred_regimes": ["bull"], "acceptable_regimes": [], "weak_regimes": [], "is_regime_specific": False}
        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_path):
            _update_winner_regime_tags("refined_test", tags)

        data = json.loads(winners_path.read_text())
        # Only the second (most recent) entry should be updated
        assert "regime_tags" in data[1]
        assert data[1]["regime_tags"]["preferred_regimes"] == ["bull"]


# ─── promote_to_winner ────────────────────────────────────────────────────

class TestPromoteToWinner:
    def test_creates_winners_file(self, tmp_path, strategy_code, state):
        result = FakeBacktestResult()
        validation = {"passed": True}

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=tmp_path / "winners.json"):
            resp = promote_to_winner(strategy_code, result, validation, state)

        assert "strategy_name" in resp
        assert resp["strategy_name"] == "refined_test_mandate"

    def test_validation_status_passed(self, tmp_path, strategy_code, state):
        result = FakeBacktestResult()
        validation = {"passed": True}

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=tmp_path / "winners.json"):
            promote_to_winner(strategy_code, result, validation, state)

        winners = json.loads((tmp_path / "winners.json").read_text())
        assert winners[0]["validation_status"] == "walk_forward_passed"

    def test_validation_status_backtest_only(self, tmp_path, strategy_code, state):
        result = FakeBacktestResult()
        validation = {"passed": False}

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=tmp_path / "winners.json"):
            promote_to_winner(strategy_code, result, validation, state)

        winners = json.loads((tmp_path / "winners.json").read_text())
        assert winners[0]["validation_status"] == "backtest_only"

    def test_appends_to_existing_winners(self, tmp_path, strategy_code, state):
        """Should append to existing winners.json, not overwrite."""
        existing = [{"strategy": "existing_winner", "sharpe": 2.0}]
        winners_path = tmp_path / "winners.json"
        winners_path.write_text(json.dumps(existing))

        result = FakeBacktestResult()
        validation = {"passed": True}

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_path):
            promote_to_winner(strategy_code, result, validation, state)

        winners = json.loads(winners_path.read_text())
        assert len(winners) == 2
        assert winners[0]["strategy"] == "existing_winner"

    def test_handles_corrupt_winners_file(self, tmp_path, strategy_code, state):
        """Should start fresh if winners.json is corrupt."""
        winners_path = tmp_path / "winners.json"
        winners_path.write_text("not json{{{")

        result = FakeBacktestResult()
        validation = {"passed": True}

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_path):
            promote_to_winner(strategy_code, result, validation, state)

        winners = json.loads(winners_path.read_text())
        assert len(winners) == 1
