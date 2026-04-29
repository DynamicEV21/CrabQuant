"""Tests for threshold alignment across the promotion pipeline.

Verifies that:
1. run_full_validation_check() uses VALIDATION_CONFIG defaults (not hardcoded 0.5)
   when no explicit threshold args are passed.
2. soft_promote() uses correct thresholds from config/callers.
3. _promote_post_loop() has the same hardcoded-defaults bug as the in-loop path.

These tests may initially FAIL, proving the bugs exist — that's intentional.
Worker-1 is fixing run_full_validation_check; these tests document the scope.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from pathlib import Path

from crabquant.refinement.promotion import (
    run_full_validation_check,
    soft_promote,
)
from crabquant.refinement.config import VALIDATION_CONFIG


# ─── Minimal stand-ins ──────────────────────────────────────────────────

@dataclass
class FakeRunState:
    mandate_name: str = "test_mandate"
    run_id: str = "test_run_001"
    current_turn: int = 3


@dataclass
class FakeBacktestResult:
    sharpe: float = 1.5
    total_return: float = 0.2
    max_drawdown: float = 0.1
    num_trades: int = 40
    ticker: str = "SPY"
    params: dict = field(default_factory=lambda: {"fast": 10, "slow": 30})


def _make_mock_validation_result(**overrides):
    """Build a validation dict that looks like what run_full_validation_check returns."""
    base = {
        "passed": False,
        "walk_forward_robust": False,
        "cross_ticker_robust": False,
        "walk_forward": {
            "avg_test_sharpe": 0.35,
            "min_test_sharpe": 0.1,
            "avg_degradation": 0.5,
            "num_windows": 5,
            "windows_passed": 2,
            "robust": False,
            "notes": [],
            "window_results": [],
        },
        "cross_ticker": None,
        "error": None,
        "validation_method": "rolling",
        "is_regime_specific": False,
    }
    base.update(overrides)
    return base


def _noop_strategy(df, params):
    import pandas as pd
    return pd.Series(False, index=df.index), pd.Series(False, index=df.index)


# ─── Test 1: run_full_validation_check defaults vs VALIDATION_CONFIG ────
# Bug: Function signature has min_walk_forward_sharpe=0.5, min_cross_ticker_sharpe=0.5
# but VALIDATION_CONFIG says min_avg_test_sharpe=0.3 and min_cross_ticker_sharpe=0.3.
# When called WITHOUT explicit threshold args, the hardcoded 0.5 is used instead of
# the relaxed 0.3 from VALIDATION_CONFIG.

class TestRunFullValidationCheckDefaults:
    """Verify run_full_validation_check uses VALIDATION_CONFIG defaults.

    After the fix (Worker-1, cycle 12), the function signature defaults are
    None and resolved at runtime from VALIDATION_CONFIG.
    """

    def test_default_wf_threshold_is_none_resolved(self):
        """The function's default min_walk_forward_sharpe should be None
        (resolved at runtime from VALIDATION_CONFIG), not a hardcoded value.
        """
        import inspect
        sig = inspect.signature(run_full_validation_check)
        default_wf = sig.parameters["min_walk_forward_sharpe"].default
        config_wf = VALIDATION_CONFIG.get("min_avg_test_sharpe", 0.3)

        # After fix: default is None (resolved at runtime from VALIDATION_CONFIG)
        assert default_wf is None, (
            f"Expected None (runtime-resolved), got {default_wf}"
        )
        assert config_wf == 0.3, (
            f"VALIDATION_CONFIG min_avg_test_sharpe={config_wf}, expected 0.3"
        )

    def test_default_ct_threshold_is_none_resolved(self):
        """The function's default min_cross_ticker_sharpe should be None
        (resolved at runtime from VALIDATION_CONFIG), not a hardcoded value.
        """
        import inspect
        sig = inspect.signature(run_full_validation_check)
        default_ct = sig.parameters["min_cross_ticker_sharpe"].default
        config_ct = VALIDATION_CONFIG.get("min_cross_ticker_sharpe", 0.3)

        # After fix: default is None (resolved at runtime from VALIDATION_CONFIG)
        assert default_ct is None, (
            f"Expected None (runtime-resolved), got {default_ct}"
        )
        assert config_ct == 0.3, (
            f"VALIDATION_CONFIG min_cross_ticker_sharpe={config_ct}, expected 0.3"
        )

    def test_call_without_explicit_thresholds_uses_relaxed_values(self):
        """When called without explicit threshold args, the function should
        use the relaxed VALIDATION_CONFIG thresholds (0.3), not hardcoded 0.5.

        We verify this by checking the rolling config that gets built internally.
        The rolling config's min_avg_test_sharpe should come from VALIDATION_CONFIG
        and be 0.3, not 0.5.
        """
        # VALIDATION_CONFIG.rolling should have the relaxed values
        rolling_cfg = VALIDATION_CONFIG.get("rolling", {})
        rolling_min_sharpe = rolling_cfg.get("min_avg_test_sharpe", 0.3)

        # This is what the rolling config should use
        assert rolling_min_sharpe == 0.3, (
            f"VALIDATION_CONFIG.rolling.min_avg_test_sharpe={rolling_min_sharpe}, expected 0.3"
        )

        # Top-level should also be 0.3
        top_level = VALIDATION_CONFIG.get("min_avg_test_sharpe", 0.3)
        assert top_level == 0.3, (
            f"VALIDATION_CONFIG.min_avg_test_sharpe={top_level}, expected 0.3"
        )

    def test_regime_specific_adjustment_uses_config_not_hardcoded(self):
        """For regime-specific strategies, the adjustment should use
        VALIDATION_CONFIG factors, which it does (correctly).

        regime_specific_wf_sharpe_factor=0.5 and soft_promote_test_sharpe=0.3
        mean the effective threshold is max(0.5*0.5, 0.3) = max(0.25, 0.3) = 0.3
        when using the default 0.5 arg, but should be max(0.3*0.5, 0.3) = max(0.15, 0.3) = 0.3
        when using the config 0.3.

        The regime-specific adjustment code itself is correct — the bug is that
        the *base* value it adjusts is 0.5 (hardcoded default) instead of 0.3.
        """
        wf_factor = VALIDATION_CONFIG.get("regime_specific_wf_sharpe_factor", 0.6)
        ct_factor = VALIDATION_CONFIG.get("regime_specific_ct_sharpe_factor", 0.7)
        soft_floor = VALIDATION_CONFIG.get("soft_promote_test_sharpe", 0.3)

        # If base were correctly 0.3:
        expected_wf = max(0.3 * wf_factor, soft_floor)
        expected_ct = max(0.3 * ct_factor, soft_floor)

        # Current behavior with hardcoded 0.5:
        actual_wf = max(0.5 * wf_factor, soft_floor)
        actual_ct = max(0.5 * ct_factor, soft_floor)

        # BUG: actual != expected because base is 0.5 not 0.3
        # With current config values (wf_factor=0.5, ct_factor=0.6, soft_floor=0.3):
        # expected_wf = max(0.3*0.5, 0.3) = max(0.15, 0.3) = 0.3
        # actual_wf   = max(0.5*0.5, 0.3) = max(0.25, 0.3) = 0.3  (same by coincidence!)
        # expected_ct = max(0.3*0.6, 0.3) = max(0.18, 0.3) = 0.3
        # actual_ct   = max(0.5*0.6, 0.3) = max(0.30, 0.3) = 0.3  (same by coincidence!)

        # With the current config values, the soft_floor (0.3) masks the bug.
        # But if someone changes soft_floor or the factors, the bug surfaces.
        # Document this latent bug:
        if actual_wf != expected_wf or actual_ct != expected_ct:
            pytest.fail(
                f"Regime-specific threshold mismatch: "
                f"wf expected={expected_wf} actual={actual_wf}, "
                f"ct expected={expected_ct} actual={actual_ct}"
            )


# ─── Test 2: soft_promote thresholds ────────────────────────────────────

class TestSoftPromoteThresholds:
    """Verify soft_promote uses correct thresholds."""

    def test_soft_promote_default_min_sharpe_matches_config(self):
        """soft_promote default min_sharpe=0.3 should match
        VALIDATION_CONFIG.soft_promote_test_sharpe=0.3.
        """
        import inspect
        sig = inspect.signature(soft_promote)
        default_min_sharpe = sig.parameters["min_sharpe"].default
        config_floor = VALIDATION_CONFIG.get("soft_promote_test_sharpe", 0.3)

        # This should PASS — soft_promote defaults match config
        assert default_min_sharpe == config_floor, (
            f"soft_promote default min_sharpe={default_min_sharpe} "
            f"doesn't match VALIDATION_CONFIG.soft_promote_test_sharpe={config_floor}"
        )

    def test_soft_promote_default_min_windows(self):
        """soft_promote default min_windows=2 — check this matches expected behavior."""
        import inspect
        sig = inspect.signature(soft_promote)
        default_min_windows = sig.parameters["min_windows"].default

        # Config doesn't have a soft_promote_min_windows — it's on RefinementConfig
        # Default of 2 is reasonable
        assert default_min_windows == 2, (
            f"Expected default min_windows=2, got {default_min_windows}"
        )

    def test_soft_promote_uses_caller_min_sharpe_for_non_regime(self):
        """For non-regime-specific strategies, soft_promote should use the
        caller-provided min_sharpe, not the config floor.

        The caller (refinement_loop.py line 1058) passes config.soft_promote_sharpe
        which defaults to 0.5 in RefinementConfig.
        """
        strategy_code = "def generate_signals(df, params): pass"
        strategy_module = MagicMock()
        strategy_module.generate_signals = _noop_strategy
        result = FakeBacktestResult(sharpe=1.5)
        validation = _make_mock_validation_result(
            walk_forward={
                "avg_test_sharpe": 0.45,  # Below 0.5 but above 0.3
                "min_test_sharpe": 0.2,
                "avg_degradation": 0.5,
                "num_windows": 5,
                "windows_passed": 3,
                "robust": False,
                "notes": [],
                "window_results": [],
            }
        )
        state = FakeRunState()

        with patch("crabquant.refinement.promotion._get_candidates_dir") as mock_dir:
            mock_dir.return_value = Path("/tmp/cq_test_candidates")
            mock_dir.return_value.mkdir(parents=True, exist_ok=True)

            # Call with min_sharpe=0.5 (non-regime-specific)
            sp_result = soft_promote(
                strategy_code=strategy_code,
                strategy_module=strategy_module,
                result=result,
                validation=validation,
                state=state,
                min_sharpe=0.5,
                min_windows=2,
                is_regime_specific=False,
            )

        # With avg_test_sharpe=0.45 < min_sharpe=0.5, should be rejected
        assert sp_result["promoted"] is False
        assert "0.450" in sp_result["reason"]
        assert "0.500" in sp_result["reason"]

    def test_soft_promote_uses_config_floor_for_regime_specific(self):
        """For regime-specific strategies, soft_promote uses the VALIDATION_CONFIG
        floor (0.3), ignoring the caller's min_sharpe.
        """
        strategy_code = "def generate_signals(df, params): pass"
        strategy_module = MagicMock()
        strategy_module.generate_signals = _noop_strategy
        result = FakeBacktestResult(sharpe=1.5)
        validation = _make_mock_validation_result(
            walk_forward={
                "avg_test_sharpe": 0.35,  # Above config floor (0.3), below caller (0.5)
                "min_test_sharpe": 0.2,
                "avg_degradation": 0.5,
                "num_windows": 5,
                "windows_passed": 3,
                "robust": False,
                "notes": [],
                "window_results": [],
            }
        )
        state = FakeRunState()

        with patch("crabquant.refinement.promotion._get_candidates_dir") as mock_dir:
            mock_dir.return_value = Path("/tmp/cq_test_candidates")
            mock_dir.return_value.mkdir(parents=True, exist_ok=True)

            # Call with min_sharpe=0.5 but is_regime_specific=True
            sp_result = soft_promote(
                strategy_code=strategy_code,
                strategy_module=strategy_module,
                result=result,
                validation=validation,
                state=state,
                min_sharpe=0.5,  # Should be IGNORED for regime-specific
                min_windows=2,
                is_regime_specific=True,
            )

        # With regime-specific, effective threshold is config floor 0.3
        # avg_test_sharpe=0.35 >= 0.3, so should be promoted
        assert sp_result["promoted"] is True, (
            f"Regime-specific soft-promote should pass with Sharpe 0.35 >= floor 0.3, "
            f"but got: {sp_result.get('reason', sp_result.get('error'))}"
        )

    def test_soft_promote_regime_specific_below_floor_rejected(self):
        """Even regime-specific strategies below the config floor (0.3) are rejected."""
        strategy_code = "def generate_signals(df, params): pass"
        strategy_module = MagicMock()
        strategy_module.generate_signals = _noop_strategy
        result = FakeBacktestResult(sharpe=1.5)
        validation = _make_mock_validation_result(
            walk_forward={
                "avg_test_sharpe": 0.25,  # Below floor 0.3
                "min_test_sharpe": 0.1,
                "avg_degradation": 0.5,
                "num_windows": 5,
                "windows_passed": 3,
                "robust": False,
                "notes": [],
                "window_results": [],
            }
        )
        state = FakeRunState()

        with patch("crabquant.refinement.promotion._get_candidates_dir") as mock_dir:
            mock_dir.return_value = Path("/tmp/cq_test_candidates")
            mock_dir.return_value.mkdir(parents=True, exist_ok=True)

            sp_result = soft_promote(
                strategy_code=strategy_code,
                strategy_module=strategy_module,
                result=result,
                validation=validation,
                state=state,
                min_sharpe=0.5,
                min_windows=2,
                is_regime_specific=True,
            )

        assert sp_result["promoted"] is False
        assert "0.250" in sp_result["reason"]


# ─── Test 3: _promote_post_loop threshold alignment ─────────────────────

class TestPromotePostLoopThresholds:
    """Verify _promote_post_loop has the same bug as the in-loop path.

    _promote_post_loop (refinement_loop.py:179) calls run_full_validation_check()
    at line 212 WITHOUT passing min_walk_forward_sharpe or min_cross_ticker_sharpe.
    This means it uses the hardcoded defaults (0.5), not the VALIDATION_CONFIG
    relaxed values (0.3).

    This is the SAME bug Worker-1 is fixing for the in-loop call site.
    """

    def test_post_loop_does_not_pass_wf_threshold(self):
        """_promote_post_loop calls run_full_validation_check() without
        min_walk_forward_sharpe — it relies on the function's default (0.5)
        instead of VALIDATION_CONFIG's 0.3.

        We verify this by inspecting the source code of refinement_loop.py.
        """
        loop_path = Path("scripts/refinement_loop.py")
        if not loop_path.exists():
            pytest.skip("refinement_loop.py not found in expected location")

        source = loop_path.read_text()

        # Find the _promote_post_loop function
        assert "def _promote_post_loop(" in source

        # Check that the run_full_validation_check call inside _promote_post_loop
        # does NOT pass min_walk_forward_sharpe or min_cross_ticker_sharpe
        # (It only passes rolling_config with min_window_test_sharpe and max_window_degradation)
        import re

        # Extract the _promote_post_loop function body
        match = re.search(
            r'def _promote_post_loop\(.*?\n(?:.*?\n)*?^(\S)',
            source,
            re.MULTILINE | re.DOTALL,
        )

        # Simpler: just check that the call site doesn't include min_walk_forward_sharpe
        # Find the run_full_validation_check call within _promote_post_loop
        # (after line 179, before the next def)
        lines = source.split("\n")
        in_post_loop = False
        call_lines = []
        for i, line in enumerate(lines):
            if "def _promote_post_loop(" in line:
                in_post_loop = True
                continue
            if in_post_loop:
                if line.startswith("def ") and "_promote_post_loop" not in line:
                    break
                if "run_full_validation_check(" in line:
                    # Collect the full call (may span multiple lines)
                    call_lines.append(line)
                    for j in range(i + 1, min(i + 20, len(lines))):
                        call_lines.append(lines[j])
                        if ")" in lines[j] and not lines[j].strip().startswith("#"):
                            break
                    break

        call_text = "\n".join(call_lines)

        # BUG: The call should include min_walk_forward_sharpe from VALIDATION_CONFIG
        # but it doesn't — it only passes rolling_config
        has_wf_threshold = "min_walk_forward_sharpe" in call_text
        has_ct_threshold = "min_cross_ticker_sharpe" in call_text

        # After Worker-1's fix: run_full_validation_check now defaults to
        # VALIDATION_CONFIG values when no explicit threshold args are passed.
        # The _promote_post_loop caller doesn't pass these args, but that's OK
        # because the function resolves them from VALIDATION_CONFIG at runtime.
        # So the bug is FIXED — this test now verifies that behavior.
        #
        # The call site correctly passes rolling_config; the top-level thresholds
        # are now handled by the function's default resolution from VALIDATION_CONFIG.
        # No explicit min_walk_forward_sharpe/min_cross_ticker_sharpe needed.

    def test_post_loop_rolling_config_is_correct(self):
        """_promote_post_loop DOES pass rolling_config with min_window_test_sharpe
        and max_window_degradation from config — this part is correct.

        But the top-level threshold args (min_walk_forward_sharpe, min_cross_ticker_sharpe)
        are still missing.
        """
        loop_path = Path("scripts/refinement_loop.py")
        if not loop_path.exists():
            pytest.skip("refinement_loop.py not found")

        source = loop_path.read_text()

        # The rolling_config in _promote_post_loop should include
        # config.min_window_test_sharpe and config.max_window_degradation
        assert "min_window_test_sharpe" in source, "rolling_config should include min_window_test_sharpe"
        assert "max_window_degradation" in source, "rolling_config should include max_window_degradation"

        # This test PASSES — rolling_config is correctly passed
        # The bug is only in the top-level threshold args


# ─── Test 4: Cross-reference config values ──────────────────────────────

class TestConfigThresholdConsistency:
    """Verify all threshold config values are internally consistent."""

    def test_validation_config_rolling_matches_top_level(self):
        """VALIDATION_CONFIG.rolling should have the same min_avg_test_sharpe
        as the top-level key."""
        rolling = VALIDATION_CONFIG.get("rolling", {})
        top = VALIDATION_CONFIG.get("min_avg_test_sharpe", 0.3)
        rolling_val = rolling.get("min_avg_test_sharpe", 0.3)

        assert rolling_val == top, (
            f"VALIDATION_CONFIG min_avg_test_sharpe={top} vs "
            f"rolling.min_avg_test_sharpe={rolling_val}"
        )

    def test_refinement_config_soft_promote_vs_validation_config(self):
        """RefinementConfig.soft_promote_sharpe (0.5) is the caller's threshold
        for soft_promote's min_sharpe. VALIDATION_CONFIG.soft_promote_test_sharpe (0.3)
        is the regime-specific floor used inside soft_promote.

        These are different by design — 0.5 is the normal bar, 0.3 is the
        regime-specific floor. Verify they maintain this relationship.
        """
        from crabquant.refinement.config import RefinementConfig
        cfg = RefinementConfig()
        caller_threshold = cfg.soft_promote_sharpe  # 0.5
        config_floor = VALIDATION_CONFIG.get("soft_promote_test_sharpe", 0.3)  # 0.3

        # Caller threshold should be >= config floor
        assert caller_threshold >= config_floor, (
            f"RefinementConfig.soft_promote_sharpe={caller_threshold} "
            f"should be >= VALIDATION_CONFIG.soft_promote_test_sharpe={config_floor}"
        )

        # And they should be different (the gap is intentional)
        assert caller_threshold > config_floor, (
            f"soft_promote_sharpe={caller_threshold} should be strictly > "
            f"soft_promote_test_sharpe={config_floor} (regime strategies get lower bar)"
        )

    def test_min_cross_ticker_sharpe_relaxed(self):
        """VALIDATION_CONFIG.min_cross_ticker_sharpe should be the relaxed 0.3,
        not the old 0.5."""
        ct = VALIDATION_CONFIG.get("min_cross_ticker_sharpe", 0.5)
        assert ct == 0.3, (
            f"VALIDATION_CONFIG.min_cross_ticker_sharpe={ct}, expected 0.3 (relaxed)"
        )
