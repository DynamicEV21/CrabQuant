"""Tests for promotion.py — full validation on success + auto-promotion.

Component 7: Full validation (walk-forward + cross-ticker) before promoting.
Component 9: Auto-promotion — register validated strategies in STRATEGY_REGISTRY.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_strategy_fn():
    """Mock strategy function that returns valid signals."""
    def fn(df, params):
        import pandas as pd
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        return entries, exits
    fn.__name__ = "test_strategy"
    return fn


@pytest.fixture
def mock_strategy_module():
    """Mock strategy module with all required attributes."""
    import pandas as pd

    def generate_signals(df, params=None):
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        return entries, exits

    def generate_signals_matrix(df, param_grid):
        return pd.DataFrame(), pd.DataFrame(), []

    module = MagicMock()
    module.generate_signals = generate_signals
    module.generate_signals_matrix = generate_signals_matrix
    module.DEFAULT_PARAMS = {"fast": 12, "slow": 26}
    module.PARAM_GRID = {"fast": [8, 12, 16]}
    module.DESCRIPTION = "Test strategy for validation"
    return module


@pytest.fixture
def mock_backtest_result():
    """Mock BacktestResult with passing metrics."""
    result = MagicMock()
    result.sharpe = 2.0
    result.total_return = 0.25
    result.max_drawdown = -0.10
    result.num_trades = 30
    result.win_rate = 0.6
    result.profit_factor = 1.8
    result.calmar_ratio = 2.5
    result.sortino_ratio = 3.0
    result.score = 3.5
    result.passed = True
    result.params = {"fast": 12, "slow": 26}
    result.ticker = "SPY"
    result.strategy_name = "test_strategy"
    result.iteration = 3
    return result


@pytest.fixture
def mock_run_state():
    """Mock RunState for testing."""
    from crabquant.refinement.schemas import RunState
    return RunState(
        run_id="2026-04-26_140000_a3f2",
        mandate_name="momentum_spy_1",
        created_at="2026-04-26T14:00:00",
        current_turn=3,
        status="success",
        best_sharpe=2.0,
        best_turn=3,
        tickers=["SPY", "AAPL", "NVDA"],
    )


# ---------------------------------------------------------------------------
# Component 7: TestFullValidation
# ---------------------------------------------------------------------------

class TestFullValidation:
    """Test full validation — rolling walk-forward + cross-ticker before promotion."""

    def test_run_full_validation_passes_both(self, mock_strategy_fn):
        """Strategy passes both rolling walk-forward and cross-ticker validation."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf_result = MagicMock()
        mock_wf_result.avg_test_sharpe = 1.5
        mock_wf_result.avg_train_sharpe = 2.0
        mock_wf_result.windows_passed = 4
        mock_wf_result.windows_total = 4
        mock_wf_result.robust = True
        mock_wf_result.notes = "All windows profitable"
        mock_wf_result.regime_shifts = []

        mock_ct_result = MagicMock()
        mock_ct_result.avg_sharpe = 1.2
        mock_ct_result.tickers_profitable = 3
        mock_ct_result.tickers_tested = 3
        mock_ct_result.robust = True
        mock_ct_result.notes = "All tickers profitable"

        with patch("crabquant.validation.rolling_walk_forward", return_value=mock_wf_result), \
             patch("crabquant.validation.cross_ticker_validation", return_value=mock_ct_result):
            result = run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={"fast": 12, "slow": 26},
                discovery_ticker="SPY",
                validation_tickers=["SPY", "AAPL", "NVDA"],
                min_walk_forward_sharpe=0.5,
                min_cross_ticker_sharpe=0.5,
            )

        assert result["passed"] is True
        assert result["walk_forward_robust"] is True
        assert result["cross_ticker_robust"] is True

    def test_full_validation_fails_walk_forward(self, mock_strategy_fn):
        """Strategy fails rolling walk-forward — should not promote."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf_result = MagicMock()
        mock_wf_result.avg_test_sharpe = 0.1
        mock_wf_result.avg_train_sharpe = 2.0
        mock_wf_result.windows_passed = 1
        mock_wf_result.windows_total = 4
        mock_wf_result.robust = False
        mock_wf_result.notes = "Only 1/4 windows profitable"
        mock_wf_result.regime_shifts = ["window_2"]

        mock_ct_result = MagicMock()
        mock_ct_result.avg_sharpe = 1.5
        mock_ct_result.robust = True
        mock_ct_result.notes = "Good cross-ticker"

        with patch("crabquant.validation.rolling_walk_forward", return_value=mock_wf_result), \
             patch("crabquant.validation.cross_ticker_validation", return_value=mock_ct_result):
            result = run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={"fast": 12, "slow": 26},
                discovery_ticker="SPY",
                validation_tickers=["SPY", "AAPL"],
                min_walk_forward_sharpe=0.5,
            )

        assert result["passed"] is False
        assert result["walk_forward_robust"] is False

    def test_full_validation_fails_cross_ticker(self, mock_strategy_fn):
        """Strategy fails cross-ticker — should not promote."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf_result = MagicMock()
        mock_wf_result.avg_test_sharpe = 1.5
        mock_wf_result.avg_train_sharpe = 2.0
        mock_wf_result.windows_passed = 3
        mock_wf_result.windows_total = 4
        mock_wf_result.robust = True
        mock_wf_result.notes = "Good"
        mock_wf_result.regime_shifts = []

        mock_ct_result = MagicMock()
        mock_ct_result.avg_sharpe = -0.5
        mock_ct_result.robust = False
        mock_ct_result.notes = "Loses money on other tickers"

        with patch("crabquant.validation.rolling_walk_forward", return_value=mock_wf_result), \
             patch("crabquant.validation.cross_ticker_validation", return_value=mock_ct_result):
            result = run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={"fast": 12},
                discovery_ticker="SPY",
                validation_tickers=["SPY", "AAPL", "TSLA"],
                min_cross_ticker_sharpe=0.3,
            )

        assert result["passed"] is False
        assert result["cross_ticker_robust"] is False

    def test_full_validation_returns_detailed_results(self, mock_strategy_fn):
        """Result includes detailed rolling walk-forward and cross-ticker data."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf_result = MagicMock()
        mock_wf_result.avg_test_sharpe = 1.5
        mock_wf_result.avg_train_sharpe = 2.0
        mock_wf_result.windows_passed = 3
        mock_wf_result.windows_total = 4
        mock_wf_result.robust = True
        mock_wf_result.notes = "3/4 windows passed"
        mock_wf_result.regime_shifts = []
        mock_wf_result.window_results = [
            MagicMock(test_sharpe=1.8, train_sharpe=2.1),
            MagicMock(test_sharpe=1.2, train_sharpe=2.0),
            MagicMock(test_sharpe=1.6, train_sharpe=1.9),
            MagicMock(test_sharpe=0.3, train_sharpe=1.5),
        ]

        mock_ct_result = MagicMock()
        mock_ct_result.avg_sharpe = 1.2
        mock_ct_result.median_sharpe = 1.1
        mock_ct_result.robust = True
        mock_ct_result.notes = "3/3 profitable"
        mock_ct_result.tickers_profitable = 3
        mock_ct_result.tickers_tested = 3

        with patch("crabquant.validation.rolling_walk_forward", return_value=mock_wf_result), \
             patch("crabquant.validation.cross_ticker_validation", return_value=mock_ct_result):
            result = run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={},
                discovery_ticker="SPY",
                validation_tickers=["SPY", "AAPL"],
            )

        assert "walk_forward" in result
        assert "cross_ticker" in result
        assert result["walk_forward"]["avg_test_sharpe"] == 1.5
        assert result["cross_ticker"]["avg_sharpe"] == 1.2

    def test_full_validation_handles_exception(self, mock_strategy_fn):
        """Graceful handling when backtest throws an exception."""
        from crabquant.refinement.promotion import run_full_validation_check

        with patch("crabquant.validation.rolling_walk_forward", side_effect=Exception("Data error")):
            result = run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={},
                discovery_ticker="BADTICKER",
                validation_tickers=["BADTICKER"],
            )

        assert result["passed"] is False
        assert "error" in result

    def test_promote_to_winner_writes_file(self, mock_strategy_module, mock_backtest_result, mock_run_state, tmp_path):
        """Promotion writes strategy code and metadata to winners file."""
        from crabquant.refinement.promotion import promote_to_winner

        strategy_code = "# my great strategy\npass"
        validation = {
            "passed": True,
            "walk_forward_robust": True,
            "cross_ticker_robust": True,
        }

        with patch("crabquant.refinement.promotion.register_strategy", return_value=True), \
             patch("crabquant.refinement.promotion._get_winners_path", return_value=tmp_path / "winners.json"):
            promote_to_winner(
                strategy_code=strategy_code,
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
                strategy_module=mock_strategy_module,
            )

        winners_file = tmp_path / "winners.json"
        assert winners_file.exists()
        winners = json.loads(winners_file.read_text())
        assert len(winners) >= 1
        assert winners[0]["sharpe"] == 2.0
        assert winners[0]["validation"]["passed"] is True


# ---------------------------------------------------------------------------
# Component 9: TestAutoPromotion
# ---------------------------------------------------------------------------

class TestAutoPromotion:
    """Test auto-promotion — register validated strategies in STRATEGY_REGISTRY."""

    def test_register_strategy_adds_to_registry(self, mock_strategy_module, tmp_path):
        """Strategy gets registered in STRATEGY_REGISTRY."""
        from crabquant.refinement.promotion import register_strategy

        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()

        with patch("crabquant.refinement.promotion._get_strategies_dir", return_value=strategies_dir), \
             patch("crabquant.strategies.STRATEGY_REGISTRY", {}):
            from crabquant.strategies import STRATEGY_REGISTRY
            # Need to re-import to get the patched registry
            success = register_strategy("refined_test", mock_strategy_module)

        assert success is True

    def test_register_strategy_writes_file(self, mock_strategy_module, tmp_path):
        """Strategy file is written to strategies directory."""
        from crabquant.refinement.promotion import register_strategy

        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()
        strategy_code = "# refined strategy\ndef generate_signals(): pass"

        with patch("crabquant.refinement.promotion._get_strategies_dir", return_value=strategies_dir), \
             patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
            from crabquant.strategies import STRATEGY_REGISTRY
            success = register_strategy(
                "refined_momentum",
                mock_strategy_module,
                strategy_code=strategy_code,
            )

        assert success is True
        assert (strategies_dir / "refined_momentum.py").exists()

    def test_prevent_duplicate_registration(self, mock_strategy_module, tmp_path):
        """Duplicate registration is prevented."""
        from crabquant.refinement.promotion import register_strategy, is_already_registered

        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()

        with patch("crabquant.refinement.promotion._get_strategies_dir", return_value=strategies_dir), \
             patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
            from crabquant.strategies import STRATEGY_REGISTRY

            # First registration succeeds
            result1 = register_strategy("refined_dup", mock_strategy_module)
            # Second registration should be prevented
            result2 = register_strategy("refined_dup", mock_strategy_module)

        assert result1 is True
        assert result2 is False

    def test_is_already_registered_true(self, tmp_path):
        """Returns True when strategy is already in registry."""
        from crabquant.refinement.promotion import is_already_registered

        with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {"refined_exists": "dummy"}):
            assert is_already_registered("refined_exists") is True

    def test_is_already_registered_false(self, tmp_path):
        """Returns False when strategy is not in registry."""
        from crabquant.refinement.promotion import is_already_registered

        with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}):
            assert is_already_registered("refined_new") is False

    def test_auto_promote_registers_on_pass(self, mock_strategy_module, mock_backtest_result, mock_run_state, tmp_path):
        """Auto-promote registers strategy when full validation passes."""
        from crabquant.refinement.promotion import auto_promote

        validation = {
            "passed": True,
            "walk_forward_robust": True,
            "cross_ticker_robust": True,
        }
        strategy_code = "# auto-promoted strategy\npass"

        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()

        mock_regime_tags = {
            "preferred_regimes": ["trending_up"],
            "acceptable_regimes": [],
            "weak_regimes": ["high_volatility"],
            "regime_sharpes": {"trending_up": 1.2, "high_volatility": -0.5},
            "is_regime_specific": True,
        }

        with patch("crabquant.refinement.promotion._get_strategies_dir", return_value=strategies_dir), \
             patch("crabquant.refinement.promotion._get_winners_path", return_value=tmp_path / "winners.json"), \
             patch("crabquant.refinement.regime_tagger.compute_strategy_regime_tags", return_value=mock_regime_tags), \
             patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
            from crabquant.strategies import STRATEGY_REGISTRY
            result = auto_promote(
                strategy_code=strategy_code,
                strategy_module=mock_strategy_module,
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        assert result["registered"] is True
        assert result["strategy_name"].startswith("refined_")
        assert result["regime_tags"] is not None
        assert result["regime_tags"]["preferred_regimes"] == ["trending_up"]

    def test_auto_promote_skips_on_fail(self, mock_strategy_module, mock_backtest_result, mock_run_state):
        """Auto-promote does not register when validation fails."""
        from crabquant.refinement.promotion import auto_promote

        validation = {
            "passed": False,
            "walk_forward_robust": False,
            "cross_ticker_robust": True,
        }
        strategy_code = "# not promoted\npass"

        with patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}):
            result = auto_promote(
                strategy_code=strategy_code,
                strategy_module=mock_strategy_module,
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        assert result["registered"] is False

    def test_save_strategy_metadata(self, mock_strategy_module, mock_backtest_result, mock_run_state, tmp_path):
        """Auto-promote saves metadata (Sharpe, regime, date) alongside strategy."""
        from crabquant.refinement.promotion import auto_promote

        validation = {
            "passed": True,
            "walk_forward_robust": True,
            "cross_ticker_robust": True,
            "walk_forward": {"avg_test_sharpe": 1.5, "windows_passed": 3},
        }
        strategy_code = "# metadata strategy\npass"

        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()

        mock_regime_tags = {
            "preferred_regimes": ["trending_up", "mean_reverting"],
            "acceptable_regimes": [],
            "weak_regimes": [],
            "regime_sharpes": {"trending_up": 1.0, "mean_reverting": 0.9},
            "is_regime_specific": False,
        }

        with patch("crabquant.refinement.promotion._get_strategies_dir", return_value=strategies_dir), \
             patch("crabquant.refinement.promotion._get_winners_path", return_value=tmp_path / "winners.json"), \
             patch("crabquant.refinement.regime_tagger.compute_strategy_regime_tags", return_value=mock_regime_tags), \
             patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
            result = auto_promote(
                strategy_code=strategy_code,
                strategy_module=mock_strategy_module,
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        assert result["registered"] is True

        # Check winners.json has metadata
        winners_file = tmp_path / "winners.json"
        assert winners_file.exists()
        winners = json.loads(winners_file.read_text())
        winner = winners[-1]
        assert winner["sharpe"] == 2.0
        assert "validation" in winner
        assert "refinement_run" in winner
        # Check regime tags saved to winner
        assert "regime_tags" in winner
        assert winner["regime_tags"]["preferred_regimes"] == ["trending_up", "mean_reverting"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Test internal path helper functions."""

    def test_get_strategies_dir_returns_path(self):
        from crabquant.refinement.promotion import _get_strategies_dir
        result = _get_strategies_dir()
        assert isinstance(result, Path)

    def test_get_winners_path_returns_path(self):
        from crabquant.refinement.promotion import _get_winners_path
        result = _get_winners_path()
        assert isinstance(result, Path)
        assert str(result).endswith("winners.json")

    def test_get_candidates_dir_returns_path(self):
        from crabquant.refinement.promotion import _get_candidates_dir
        result = _get_candidates_dir()
        assert isinstance(result, Path)
        assert "candidates" in str(result)


# ---------------------------------------------------------------------------
# Component 7: Additional full validation tests
# ---------------------------------------------------------------------------


class TestFullValidationEdgeCases:
    """Additional edge cases for full validation check."""

    def test_single_split_mode(self, mock_strategy_fn):
        """Single-split walk-forward (use_rolling=False) works correctly."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf = MagicMock()
        mock_wf.test_sharpe = 1.2
        mock_wf.train_sharpe = 2.0
        mock_wf.degradation = 0.4
        mock_wf.robust = True
        mock_wf.notes = "Good"
        mock_wf.regime_shift = False
        mock_wf.test_regime = "trending"
        mock_wf.train_regime = "trending"

        with patch("crabquant.validation.walk_forward_test", return_value=mock_wf), \
             patch("crabquant.validation.cross_ticker_validation") as mock_ct:
            mock_ct_result = MagicMock()
            mock_ct_result.avg_sharpe = 1.0
            mock_ct_result.robust = True
            mock_ct_result.notes = "OK"
            mock_ct.return_value = mock_ct_result

            result = run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={},
                discovery_ticker="SPY",
                validation_tickers=["SPY", "AAPL"],
                use_rolling=False,
            )

        assert result["validation_method"] == "single_split"
        assert result["passed"] is True
        assert result["walk_forward"]["test_sharpe"] == 1.2

    def test_no_oos_tickers_passes_by_default(self, mock_strategy_fn):
        """When validation_tickers only has discovery ticker, ct passes."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf = MagicMock()
        mock_wf.avg_test_sharpe = 1.5
        mock_wf.avg_train_sharpe = 2.0
        mock_wf.windows_passed = 3
        mock_wf.windows_total = 4
        mock_wf.robust = True
        mock_wf.notes = "OK"
        mock_wf.regime_shifts = []
        mock_wf.window_results = []

        with patch("crabquant.validation.rolling_walk_forward", return_value=mock_wf):
            result = run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={},
                discovery_ticker="SPY",
                validation_tickers=["SPY"],  # only discovery ticker
            )

        assert result["cross_ticker_robust"] is True
        assert result["cross_ticker"] is None  # no cross-ticker run

    def test_regime_specific_relaxed_thresholds(self, mock_strategy_fn):
        """Regime-specific strategies get relaxed Sharpe thresholds."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf = MagicMock()
        mock_wf.avg_test_sharpe = 0.35  # below normal 0.5 but above soft floor
        mock_wf.avg_train_sharpe = 1.0
        mock_wf.windows_passed = 2
        mock_wf.windows_total = 3
        mock_wf.robust = True
        mock_wf.notes = "OK"
        mock_wf.regime_shifts = []
        mock_wf.window_results = []

        with patch("crabquant.validation.rolling_walk_forward", return_value=mock_wf), \
             patch("crabquant.refinement.config.VALIDATION_CONFIG", {
                 "regime_specific_wf_sharpe_factor": 0.6,
                 "regime_specific_ct_sharpe_factor": 0.7,
                 "soft_promote_test_sharpe": 0.3,
             }):
            result = run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={},
                discovery_ticker="SPY",
                validation_tickers=["SPY"],
                is_regime_specific=True,
                min_walk_forward_sharpe=0.5,
            )

        assert result["is_regime_specific"] is True
        assert result["walk_forward_robust"] is True

    def test_custom_rolling_config_override(self, mock_strategy_fn):
        """Custom rolling_config overrides defaults."""
        from crabquant.refinement.promotion import run_full_validation_check

        mock_wf = MagicMock()
        mock_wf.avg_test_sharpe = 1.5
        mock_wf.windows_passed = 2
        mock_wf.windows_total = 3
        mock_wf.robust = True
        mock_wf.notes = "OK"
        mock_wf.regime_shifts = []
        mock_wf.window_results = []

        with patch("crabquant.validation.rolling_walk_forward", return_value=mock_wf) as mock_rwf:
            run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={},
                discovery_ticker="SPY",
                validation_tickers=["SPY"],
                rolling_config={"train_window": "24mo", "test_window": "3mo", "step": "3mo"},
            )

            # Verify the custom config was passed through
            call_kwargs = mock_rwf.call_args[1]
            assert call_kwargs["train_window"] == "24mo"
            assert call_kwargs["test_window"] == "3mo"
            assert call_kwargs["step"] == "3mo"

    def test_result_structure_keys(self, mock_strategy_fn):
        """Result dict always has expected keys, even on error."""
        from crabquant.refinement.promotion import run_full_validation_check

        with patch("crabquant.validation.rolling_walk_forward", side_effect=ValueError("boom")):
            result = run_full_validation_check(
                strategy_fn=mock_strategy_fn,
                params={},
                discovery_ticker="SPY",
                validation_tickers=["SPY"],
            )

        expected_keys = {"passed", "walk_forward_robust", "cross_ticker_robust",
                         "walk_forward", "cross_ticker", "error",
                         "validation_method", "is_regime_specific"}
        assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# promote_to_winner edge cases
# ---------------------------------------------------------------------------


class TestPromoteToWinnerEdgeCases:
    """Edge cases for promote_to_winner."""

    def test_backtest_only_validation_status(self, mock_backtest_result, mock_run_state, tmp_path):
        """When validation didn't pass, status is 'backtest_only'."""
        from crabquant.refinement.promotion import promote_to_winner

        validation = {"passed": False, "walk_forward_robust": False}

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=tmp_path / "winners.json"):
            promote_to_winner(
                strategy_code="code",
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        winners = json.loads((tmp_path / "winners.json").read_text())
        assert winners[0]["validation_status"] == "backtest_only"

    def test_walk_forward_passed_status(self, mock_backtest_result, mock_run_state, tmp_path):
        """When validation passed, status is 'walk_forward_passed'."""
        from crabquant.refinement.promotion import promote_to_winner

        validation = {"passed": True, "walk_forward_robust": True}

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=tmp_path / "winners.json"):
            promote_to_winner(
                strategy_code="code",
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        winners = json.loads((tmp_path / "winners.json").read_text())
        assert winners[0]["validation_status"] == "walk_forward_passed"

    def test_corrupted_winners_json_handled(self, mock_backtest_result, mock_run_state, tmp_path):
        """Corrupted winners.json is replaced, not crashed."""
        from crabquant.refinement.promotion import promote_to_winner

        winners_file = tmp_path / "winners.json"
        winners_file.write_text("NOT JSON{{{")

        validation = {"passed": True}

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_file):
            promote_to_winner(
                strategy_code="code",
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        winners = json.loads(winners_file.read_text())
        assert len(winners) == 1

    def test_appends_to_existing_winners(self, mock_backtest_result, mock_run_state, tmp_path):
        """New winner is appended to existing winners list."""
        from crabquant.refinement.promotion import promote_to_winner

        winners_file = tmp_path / "winners.json"
        winners_file.write_text(json.dumps([{"strategy": "old_winner", "sharpe": 1.0}]))

        validation = {"passed": True}

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_file):
            promote_to_winner(
                strategy_code="code",
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        winners = json.loads(winners_file.read_text())
        assert len(winners) == 2
        assert winners[0]["strategy"] == "old_winner"
        assert winners[1]["sharpe"] == 2.0


# ---------------------------------------------------------------------------
# _update_winner_status / _update_winner_regime_tags
# ---------------------------------------------------------------------------


class TestUpdateWinnerHelpers:
    """Test internal winner file update helpers."""

    def test_update_winner_status(self, tmp_path):
        """Updates validation_status of the most recent matching entry."""
        from crabquant.refinement.promotion import _update_winner_status

        winners_file = tmp_path / "winners.json"
        winners = [
            {"strategy": "refined_test", "validation_status": "backtest_only"},
            {"strategy": "other", "validation_status": "backtest_only"},
            {"strategy": "refined_test", "validation_status": "backtest_only"},
        ]
        winners_file.write_text(json.dumps(winners))

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_file):
            _update_winner_status("refined_test", "promoted")

        updated = json.loads(winners_file.read_text())
        # Last occurrence should be updated
        assert updated[0]["validation_status"] == "backtest_only"
        assert updated[2]["validation_status"] == "promoted"
        assert updated[1]["validation_status"] == "backtest_only"

    def test_update_winner_status_missing_file(self, tmp_path):
        """No error when winners.json doesn't exist."""
        from crabquant.refinement.promotion import _update_winner_status

        missing = tmp_path / "nonexistent.json"
        with patch("crabquant.refinement.promotion._get_winners_path", return_value=missing):
            _update_winner_status("refined_test", "promoted")
        # No exception raised

    def test_update_winner_regime_tags(self, tmp_path):
        """Adds regime_tags to the most recent matching entry."""
        from crabquant.refinement.promotion import _update_winner_regime_tags

        winners_file = tmp_path / "winners.json"
        winners = [{"strategy": "refined_test", "sharpe": 2.0}]
        winners_file.write_text(json.dumps(winners))

        regime_tags = {
            "preferred_regimes": ["trending"],
            "acceptable_regimes": ["mean_rev"],
            "weak_regimes": ["volatile"],
            "is_regime_specific": True,
        }

        with patch("crabquant.refinement.promotion._get_winners_path", return_value=winners_file):
            _update_winner_regime_tags("refined_test", regime_tags)

        updated = json.loads(winners_file.read_text())
        assert updated[0]["regime_tags"]["preferred_regimes"] == ["trending"]
        assert updated[0]["regime_tags"]["is_regime_specific"] is True


# ---------------------------------------------------------------------------
# soft_promote
# ---------------------------------------------------------------------------


class TestSoftPromote:
    """Test soft-promotion of near-miss strategies."""

    def test_soft_promote_skips_when_validation_passed(self, mock_strategy_module, mock_backtest_result, mock_run_state, tmp_path):
        """Soft-promote skips if full validation already passed."""
        from crabquant.refinement.promotion import soft_promote

        validation = {"passed": True}

        result = soft_promote(
            strategy_code="code",
            strategy_module=mock_strategy_module,
            result=mock_backtest_result,
            validation=validation,
            state=mock_run_state,
        )

        assert result["promoted"] is False
        assert "auto_promote" in result["reason"]

    def test_soft_promote_skips_no_walk_forward(self, mock_strategy_module, mock_backtest_result, mock_run_state):
        """Soft-promote skips when no walk-forward data available."""
        from crabquant.refinement.promotion import soft_promote

        validation = {"passed": False, "walk_forward": None}

        result = soft_promote(
            strategy_code="code",
            strategy_module=mock_strategy_module,
            result=mock_backtest_result,
            validation=validation,
            state=mock_run_state,
        )

        assert result["promoted"] is False
        assert "No walk-forward data" in result["reason"]

    def test_soft_promote_skips_low_sharpe(self, mock_strategy_module, mock_backtest_result, mock_run_state):
        """Soft-promote rejects when avg test Sharpe is too low."""
        from crabquant.refinement.promotion import soft_promote

        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 0.1, "windows_passed": 2},
        }

        result = soft_promote(
            strategy_code="code",
            strategy_module=mock_strategy_module,
            result=mock_backtest_result,
            validation=validation,
            state=mock_run_state,
        )

        assert result["promoted"] is False
        assert "below threshold" in result["reason"]

    def test_soft_promote_skips_few_windows(self, mock_strategy_module, mock_backtest_result, mock_run_state):
        """Soft-promote rejects when too few windows passed."""
        from crabquant.refinement.promotion import soft_promote

        validation = {
            "passed": False,
            "walk_forward": {"avg_test_sharpe": 0.8, "windows_passed": 1},
        }

        result = soft_promote(
            strategy_code="code",
            strategy_module=mock_strategy_module,
            result=mock_backtest_result,
            validation=validation,
            state=mock_run_state,
            min_windows=2,
        )

        assert result["promoted"] is False
        assert "Windows passed" in result["reason"]

    def test_soft_promote_succeeds(self, mock_strategy_module, mock_backtest_result, mock_run_state, tmp_path):
        """Soft-promote writes candidate file when criteria met."""
        from crabquant.refinement.promotion import soft_promote

        validation = {
            "passed": False,
            "walk_forward": {
                "avg_test_sharpe": 0.6,
                "windows_passed": 2,
                "num_windows": 3,
                "avg_degradation": 0.2,
            },
            "walk_forward_robust": False,
            "cross_ticker_robust": True,
        }

        candidates_dir = tmp_path / "candidates"
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=candidates_dir), \
             patch("crabquant.refinement.regime_tagger.compute_strategy_regime_tags", return_value=None):
            result = soft_promote(
                strategy_code="def generate_signals(df, p): pass",
                strategy_module=mock_strategy_module,
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        assert result["promoted"] is True
        assert result["candidate_file"] is not None
        assert result["avg_test_sharpe"] == 0.6
        assert result["windows_passed"] == 2

        # Verify candidate file content
        candidate_file = Path(result["candidate_file"])
        assert candidate_file.exists()
        candidate = json.loads(candidate_file.read_text())
        assert candidate["needs_ongoing_validation"] is True
        assert candidate["strategy_code"] == "def generate_signals(df, p): pass"

    def test_soft_promote_regime_specific_lower_threshold(self, mock_strategy_module, mock_backtest_result, mock_run_state, tmp_path):
        """Regime-specific strategies get lower soft-promote threshold."""
        from crabquant.refinement.promotion import soft_promote

        validation = {
            "passed": False,
            "walk_forward": {
                "avg_test_sharpe": 0.35,  # above 0.3 floor, below normal 0.5
                "windows_passed": 2,
                "num_windows": 3,
                "avg_degradation": 0.1,
            },
            "walk_forward_robust": False,
            "cross_ticker_robust": True,
        }

        candidates_dir = tmp_path / "candidates"
        with patch("crabquant.refinement.promotion._get_candidates_dir", return_value=candidates_dir), \
             patch("crabquant.refinement.config.VALIDATION_CONFIG", {"soft_promote_test_sharpe": 0.3}), \
             patch("crabquant.refinement.regime_tagger.compute_strategy_regime_tags", return_value=None):
            result = soft_promote(
                strategy_code="code",
                strategy_module=mock_strategy_module,
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
                is_regime_specific=True,
            )

        assert result["promoted"] is True


# ---------------------------------------------------------------------------
# auto_promote additional edge cases
# ---------------------------------------------------------------------------


class TestAutoPromoteEdgeCases:
    """Additional edge cases for auto-promotion."""

    def test_auto_promote_skips_already_registered(self, mock_strategy_module, mock_backtest_result, mock_run_state):
        """Auto-promote skips when strategy is already in registry."""
        from crabquant.refinement.promotion import auto_promote

        validation = {"passed": True}

        with patch("crabquant.refinement.promotion.is_already_registered", return_value=True):
            result = auto_promote(
                strategy_code="code",
                strategy_module=mock_strategy_module,
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        assert result["registered"] is False
        assert "already registered" in result["error"]

    def test_auto_promote_handles_regime_tag_failure(self, mock_strategy_module, mock_backtest_result, mock_run_state, tmp_path):
        """Auto-promote still registers even when regime tagging fails."""
        from crabquant.refinement.promotion import auto_promote

        validation = {"passed": True}
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()

        with patch("crabquant.refinement.promotion._get_strategies_dir", return_value=strategies_dir), \
             patch("crabquant.refinement.promotion._get_winners_path", return_value=tmp_path / "winners.json"), \
             patch("crabquant.refinement.regime_tagger.compute_strategy_regime_tags", side_effect=Exception("tag fail")), \
             patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
            result = auto_promote(
                strategy_code="code",
                strategy_module=mock_strategy_module,
                result=mock_backtest_result,
                validation=validation,
                state=mock_run_state,
            )

        assert result["registered"] is True
        assert result["regime_tags"] is None  # regime tagging failed

    def test_register_strategy_without_regime_tags(self, mock_strategy_module, tmp_path):
        """Registering without regime tags sets defaults."""
        from crabquant.refinement.promotion import register_strategy

        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()

        with patch("crabquant.refinement.promotion._get_strategies_dir", return_value=strategies_dir), \
             patch.dict("crabquant.strategies.STRATEGY_REGISTRY", {}, clear=True):
            from crabquant.strategies import STRATEGY_REGISTRY
            register_strategy("refined_notags", mock_strategy_module)

            entry = STRATEGY_REGISTRY["refined_notags"]
            assert entry["preferred_regimes"] == []
            assert entry["is_regime_specific"] is False
            assert entry["regime_sharpes"] == {}
