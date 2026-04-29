"""Tests for regime-aware promotion thresholds in run_full_validation_check."""

import sys
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from crabquant.refinement.promotion import run_full_validation_check
from crabquant.refinement.config import VALIDATION_CONFIG


def _make_wf_result(avg_sharpe=1.0, min_sharpe=0.5, passed=3, total=4, robust=True):
    """Create a RollingWalkForwardResult-like object."""
    result = MagicMock()
    result.avg_test_sharpe = avg_sharpe
    result.min_test_sharpe = min_sharpe
    result.avg_degradation = 0.1
    result.windows_passed = passed
    result.num_windows = total
    result.robust = robust
    result.notes = ""
    result.window_results = []
    return result


def _make_single_wf_result(test_sharpe=1.0, train_sharpe=1.5, robust=True):
    """Create a single-split WalkForwardResult-like object."""
    result = MagicMock()
    result.test_sharpe = test_sharpe
    result.train_sharpe = train_sharpe
    result.degradation = 0.2
    result.robust = robust
    result.notes = ""
    result.regime_shift = False
    result.test_regime = "trending"
    result.train_regime = "trending"
    return result


class TestRegimeAwareThresholds:

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_regime_specific_gets_relaxed_thresholds(self, mock_wf, mock_ct):
        """Regime-specific strategies should pass with lower Sharpe."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.35, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.6, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            is_regime_specific=True,
        )

        # Relaxed threshold: 0.5 * 0.6 = 0.3 — 0.35 > 0.3 so should pass
        assert result["passed"] is True
        assert result["is_regime_specific"] is True

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_non_regime_specific_uses_normal_thresholds(self, mock_wf, mock_ct):
        """Non-regime-specific strategies should require full Sharpe threshold."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.35, robust=False)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.6, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            is_regime_specific=False,
        )

        # Normal threshold: 0.5 — 0.35 < 0.5 so should fail
        assert result["passed"] is False

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_regime_specific_still_fails_very_low_sharpe(self, mock_wf, mock_ct):
        """Even regime-specific strategies should fail if Sharpe is way too low."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.15, robust=False)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.4, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            is_regime_specific=True,
        )

        # Even relaxed: max(0.5 * 0.5, 0.3) = 0.3 — 0.15 < 0.3
        assert result["passed"] is False

    def test_config_has_regime_threshold_keys(self):
        """Verify VALIDATION_CONFIG has the expected regime-specific keys."""
        assert "regime_specific_wf_sharpe_factor" in VALIDATION_CONFIG
        assert "regime_specific_ct_sharpe_factor" in VALIDATION_CONFIG
        assert isinstance(VALIDATION_CONFIG["regime_specific_wf_sharpe_factor"], float)
        assert VALIDATION_CONFIG["regime_specific_wf_sharpe_factor"] == 0.5

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_result_includes_regime_specific_flag(self, mock_wf, mock_ct):
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.35, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.6, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            is_regime_specific=True,
        )

        assert "is_regime_specific" in result
        assert result["is_regime_specific"] is True

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_result_default_is_not_regime_specific(self, mock_wf, mock_ct):
        """Default (no flag) should treat as non-regime-specific."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.8, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.7, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["is_regime_specific"] is False

    # ── New tests ──────────────────────────────────────────────────────

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_rolling_result_structure(self, mock_wf, mock_ct):
        """Result dict should have all expected keys for rolling mode."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        expected_keys = [
            "passed", "walk_forward_robust", "cross_ticker_robust",
            "walk_forward", "cross_ticker", "error",
            "validation_method", "is_regime_specific",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_rolling_walk_forward_details_populated(self, mock_wf, mock_ct):
        """Rolling walk-forward result should include all window details."""
        wf = _make_wf_result(avg_sharpe=1.2, min_sharpe=0.8, passed=3, total=4)
        wf.window_results = [{"sharpe": 1.0}, {"sharpe": 1.4}]
        mock_wf.return_value = wf
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.9, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        wf_data = result["walk_forward"]
        assert wf_data["avg_test_sharpe"] == 1.2
        assert wf_data["min_test_sharpe"] == 0.8
        assert wf_data["num_windows"] == 4
        assert wf_data["windows_passed"] == 3
        assert wf_data["robust"] is True
        assert wf_data["window_results"] == [{"sharpe": 1.0}, {"sharpe": 1.4}]

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.walk_forward_test")
    def test_single_split_mode(self, mock_wf, mock_ct):
        """use_rolling=False should use single-split walk_forward_test."""
        mock_wf.return_value = _make_single_wf_result(test_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            use_rolling=False,
        )

        assert result["validation_method"] == "single_split"
        assert result["passed"] is True
        wf_data = result["walk_forward"]
        assert "test_sharpe" in wf_data
        assert "train_sharpe" in wf_data
        assert "regime_shift" in wf_data
        assert wf_data["test_sharpe"] == 1.0
        assert wf_data["train_regime"] == "trending"

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_rolling_mode_label(self, mock_wf, mock_ct):
        """Default rolling mode should label validation_method as 'rolling'."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["validation_method"] == "rolling"

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_no_oos_tickers_passes_by_default(self, mock_wf, mock_ct):
        """If no OOS tickers available, cross_ticker should pass by default."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        # No cross_ticker_validation call should happen

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["SPY"],  # Only discovery ticker — no OOS
        )

        assert result["cross_ticker_robust"] is True
        assert result["cross_ticker"] is None
        mock_ct.assert_not_called()

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_empty_validation_tickers_list(self, mock_wf, mock_ct):
        """Empty validation_tickers list — no OOS tickers, pass cross-ticker by default."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=[],
        )

        assert result["cross_ticker_robust"] is True
        assert result["passed"] is True

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_cross_ticker_details_populated(self, mock_wf, mock_ct):
        """Cross-ticker result should have all expected fields."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(
            robust=True, avg_sharpe=0.8, median_sharpe=0.7,
            tickers_profitable=3, tickers_tested=4, notes="Good",
        )

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ", "IWM"],
        )

        ct_data = result["cross_ticker"]
        assert ct_data["avg_sharpe"] == 0.8
        assert ct_data["median_sharpe"] == 0.7
        assert ct_data["robust"] is True
        assert ct_data["tickers_profitable"] == 3
        assert ct_data["tickers_tested"] == 4

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_walk_forward_fails_still_populates_result(self, mock_wf, mock_ct):
        """Even when walk-forward fails, result should have data."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.2, robust=False)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["passed"] is False
        assert result["walk_forward_robust"] is False
        assert result["walk_forward"] is not None
        assert result["walk_forward"]["avg_test_sharpe"] == 0.2

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_cross_ticker_fail_prevents_pass(self, mock_wf, mock_ct):
        """Walk-forward pass but cross-ticker fail → overall fail."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=False, avg_sharpe=0.2, passed=False)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["walk_forward_robust"] is True
        assert result["cross_ticker_robust"] is False
        assert result["passed"] is False

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_both_pass(self, mock_wf, mock_ct):
        """Both walk-forward and cross-ticker pass → overall pass."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["walk_forward_robust"] is True
        assert result["cross_ticker_robust"] is True
        assert result["passed"] is True

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_robust_false_but_high_sharpe_still_fails_wf(self, mock_wf, mock_ct):
        """If robust=False, even high Sharpe should fail walk-forward."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=2.0, robust=False)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=1.0, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["walk_forward_robust"] is False
        assert result["passed"] is False

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_regime_specific_cross_ticker_relaxed(self, mock_wf, mock_ct):
        """Regime-specific strategies get relaxed cross-ticker threshold too."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        # Cross-ticker avg_sharpe = 0.35; normal threshold = 0.5, relaxed = max(0.5*0.7, 0.3) = 0.35
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.35, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            is_regime_specific=True,
        )

        assert result["cross_ticker_robust"] is True
        assert result["passed"] is True

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_error_handling_catches_exception(self, mock_wf, mock_ct):
        """Exceptions in validation should be caught, error field populated."""
        mock_wf.side_effect = RuntimeError("boom")

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["passed"] is False
        assert result["error"] is not None
        assert "boom" in result["error"]

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_custom_rolling_config_override(self, mock_wf, mock_ct):
        """rolling_config should override defaults from VALIDATION_CONFIG."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        custom_config = {
            "train_window": "24mo",
            "test_window": "3mo",
            "step": "3mo",
            "min_avg_test_sharpe": 0.3,
            "min_windows_passed": 1,
        }
        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            rolling_config=custom_config,
        )

        # Should have been called with custom config values
        call_kwargs = mock_wf.call_args[1]
        assert call_kwargs["train_window"] == "24mo"
        assert call_kwargs["test_window"] == "3mo"
        assert call_kwargs["step"] == "3mo"
        assert call_kwargs["min_avg_test_sharpe"] == 0.3
        assert call_kwargs["min_windows_passed"] == 1

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_custom_min_sharpe_thresholds(self, mock_wf, mock_ct):
        """Custom min_walk_forward_sharpe and min_cross_ticker_sharpe should be respected."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.7, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.7, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            min_walk_forward_sharpe=0.8,
            min_cross_ticker_sharpe=0.8,
        )

        # 0.7 < 0.8 for both → should fail
        assert result["walk_forward_robust"] is False
        assert result["cross_ticker_robust"] is False
        assert result["passed"] is False

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_soft_floor_applies_to_regime_specific(self, mock_wf, mock_ct):
        """Soft floor should prevent thresholds from going too low for regime-specific."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.29, robust=True)
        # 0.29 < 0.3 (soft floor) → should fail
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            is_regime_specific=True,
            min_walk_forward_sharpe=0.1,  # Very low, but soft floor is 0.3
        )

        # soft floor = 0.3, 0.29 < 0.3 → fail
        assert result["walk_forward_robust"] is False

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_sharpe_exactly_at_threshold_passes(self, mock_wf, mock_ct):
        """Sharpe exactly at threshold should pass (>= comparison)."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.5, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.5, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            min_walk_forward_sharpe=0.5,
            min_cross_ticker_sharpe=0.5,
        )

        assert result["passed"] is True

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_sharpe_just_below_threshold_fails(self, mock_wf, mock_ct):
        """Sharpe just below threshold should fail."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.499, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.5, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            min_walk_forward_sharpe=0.5,
        )

        assert result["walk_forward_robust"] is False

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_multiple_oos_tickers_all_pass(self, mock_wf, mock_ct):
        """Multiple OOS tickers should all be included in cross-ticker validation."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.9, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["SPY", "QQQ", "IWM", "DIA"],
        )

        # OOS tickers should be QQQ, IWM, DIA (excluding SPY)
        mock_ct.assert_called_once()
        call_args = mock_ct.call_args
        assert set(call_args[0][2]) == {"QQQ", "IWM", "DIA"}

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_rolling_config_none_uses_defaults(self, mock_wf, mock_ct):
        """rolling_config=None should use defaults from VALIDATION_CONFIG."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            rolling_config=None,
        )

        call_kwargs = mock_wf.call_args[1]
        # Should use VALIDATION_CONFIG defaults (18mo/6mo/6mo)
        assert call_kwargs["train_window"] == "18mo"
        assert call_kwargs["test_window"] == "6mo"
        assert call_kwargs["step"] == "6mo"

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.walk_forward_test")
    def test_single_split_cross_ticker_interaction(self, mock_wf, mock_ct):
        """Single-split mode should still do cross-ticker validation."""
        mock_wf.return_value = _make_single_wf_result(test_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            use_rolling=False,
        )

        mock_ct.assert_called_once()
        assert result["passed"] is True

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_regime_specific_true_in_result_when_flag_set(self, mock_wf, mock_ct):
        """is_regime_specific flag should be True in result when set."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            is_regime_specific=True,
        )

        assert result["is_regime_specific"] is True

    def test_config_soft_promote_sharpe_is_float(self):
        """Verify soft_promote_test_sharpe is a float in VALIDATION_CONFIG."""
        assert "soft_promote_test_sharpe" in VALIDATION_CONFIG
        assert isinstance(VALIDATION_CONFIG["soft_promote_test_sharpe"], float)
        assert VALIDATION_CONFIG["soft_promote_test_sharpe"] == 0.3

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_error_field_none_on_success(self, mock_wf, mock_ct):
        """Error field should be None when validation succeeds."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["error"] is None

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_passed_false_in_initial_result(self, mock_wf, mock_ct):
        """Before any validation, passed should default to False."""
        # Make validation raise to check initial state
        mock_wf.side_effect = ValueError("test error")

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["passed"] is False
        assert result["walk_forward_robust"] is False
        assert result["cross_ticker_robust"] is False

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_regime_specific_ct_sharpe_factor(self, mock_wf, mock_ct):
        """Test that regime_specific_ct_sharpe_factor is applied correctly."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=1.0, robust=True)
        # 0.5 * 0.7 = 0.35 → 0.35 exactly at threshold should pass
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.35, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
            is_regime_specific=True,
        )

        assert result["cross_ticker_robust"] is True

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_degradation_in_rolling_result(self, mock_wf, mock_ct):
        """avg_degradation should be in the rolling walk-forward result."""
        wf = _make_wf_result(avg_sharpe=1.0, robust=True)
        wf.avg_degradation = 0.25
        mock_wf.return_value = wf
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.8, passed=True)

        import importlib
        import crabquant.refinement.promotion as promo_mod
        importlib.reload(promo_mod)

        result = promo_mod.run_full_validation_check(
            strategy_fn=MagicMock(),
            params={},
            discovery_ticker="SPY",
            validation_tickers=["QQQ"],
        )

        assert result["walk_forward"]["avg_degradation"] == 0.25
