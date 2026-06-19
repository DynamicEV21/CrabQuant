"""Tests for regime-aware promotion thresholds in run_full_validation_check."""

import sys
import pytest
from unittest.mock import patch, MagicMock

from crabquant.refinement.promotion import run_full_validation_check
from crabquant.refinement.config import VALIDATION_CONFIG


def _make_wf_result(avg_sharpe=1.0, min_sharpe=0.5, passed=3, total=4, robust=True):
    """Create a RollingWalkForwardResult-like object."""
    result = MagicMock()
    result.avg_test_sharpe = avg_sharpe
    result.min_test_sharpe = min_sharpe
    result.windows_passed = passed
    result.num_windows = total
    result.robust = robust
    result.notes = ""
    return result


class TestRegimeAwareThresholds:

    @patch("crabquant.validation.cross_ticker_validation")
    @patch("crabquant.validation.rolling_walk_forward")
    def test_regime_specific_gets_relaxed_thresholds(self, mock_wf, mock_ct):
        """Regime-specific strategies should pass with lower Sharpe."""
        mock_wf.return_value = _make_wf_result(avg_sharpe=0.35, robust=True)
        mock_ct.return_value = MagicMock(robust=True, avg_sharpe=0.6, passed=True)

        # Must reload promotion to pick up fresh imports
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

        # Even relaxed: max(0.5 * 0.6, 0.3) = 0.3 — 0.15 < 0.3
        assert result["passed"] is False

    def test_config_has_regime_threshold_keys(self):
        """Verify VALIDATION_CONFIG has the expected regime-specific keys."""
        assert "regime_specific_wf_sharpe_factor" in VALIDATION_CONFIG
        assert "regime_specific_ct_sharpe_factor" in VALIDATION_CONFIG
        assert isinstance(VALIDATION_CONFIG["regime_specific_wf_sharpe_factor"], float)
        assert VALIDATION_CONFIG["regime_specific_wf_sharpe_factor"] == 0.6

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
