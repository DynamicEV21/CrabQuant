"""Tests for adaptive Sharpe targeting.

Covers:
- compute_effective_target() pure function
- Per-turn ramp progression with various configs
- Disabled mode passthrough
- Final promotion gate always uses original target
- Edge cases (turn 0, negative targets, extreme factors)
"""

import math
import pytest

from crabquant.refinement.config import (
    RefinementConfig,
    compute_effective_target,
)


# ── compute_effective_target: disabled mode ────────────────────────────

class TestAdaptiveDisabled:
    """When adaptive_sharpe_target=False, effective_target == sharpe_target."""

    @pytest.mark.parametrize("turn", [1, 2, 3, 5, 7, 10, 100])
    @pytest.mark.parametrize("sharpe", [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_returns_original_when_disabled(self, sharpe, turn):
        result = compute_effective_target(
            sharpe_target=sharpe,
            turn=turn,
            adaptive_sharpe_target=False,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        assert result == sharpe

    def test_disabled_with_default_config(self):
        cfg = RefinementConfig()
        assert cfg.adaptive_sharpe_target is False
        result = compute_effective_target(
            sharpe_target=1.5,
            turn=1,
            adaptive_sharpe_target=cfg.adaptive_sharpe_target,
            adaptive_start_factor=cfg.adaptive_start_factor,
            adaptive_ramp_turns=cfg.adaptive_ramp_turns,
        )
        assert result == 1.5


# ── compute_effective_target: basic ramp ───────────────────────────────

class TestAdaptiveRamp:
    """Linear ramp from start_factor*target to target over ramp_turns."""

    def test_turn1_at_start_factor(self):
        """Turn 1 should be exactly sharpe_target * start_factor."""
        result = compute_effective_target(
            sharpe_target=2.0,
            turn=1,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        # progress = (1-1)/3 = 0 → target = 2.0 * (0.5 + 0.5*0) = 1.0
        assert result == pytest.approx(1.0)

    def test_last_ramp_turn_at_full_target(self):
        """Turn == ramp_turns should be close to but not yet the full target.
        
        progress = (ramp-1)/ramp = 2/3 for ramp=3.
        target = 2.0 * (0.5 + 0.5 * 2/3) = 2.0 * 0.8333 = 1.6667
        """
        result = compute_effective_target(
            sharpe_target=2.0,
            turn=3,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        assert result == pytest.approx(2.0 * (0.5 + 0.5 * (2 / 3)))

    def test_turn_after_ramp_equals_original(self):
        """Any turn > ramp_turns should return the original sharpe_target."""
        result = compute_effective_target(
            sharpe_target=2.0,
            turn=4,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        assert result == pytest.approx(2.0)

    def test_turn_far_after_ramp_equals_original(self):
        result = compute_effective_target(
            sharpe_target=1.5,
            turn=100,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        assert result == pytest.approx(1.5)

    def test_ramp_is_monotonically_increasing(self):
        """Target should increase (or stay flat) turn over turn."""
        targets = [
            compute_effective_target(
                sharpe_target=2.0,
                turn=t,
                adaptive_sharpe_target=True,
                adaptive_start_factor=0.5,
                adaptive_ramp_turns=5,
            )
            for t in range(1, 11)
        ]
        for i in range(1, len(targets)):
            assert targets[i] >= targets[i - 1], (
                f"Target decreased: turn {i}={targets[i]:.4f} < "
                f"turn {i}={targets[i-1]:.4f}"
            )


# ── compute_effective_target: specific turn-by-turn values ─────────────

class TestRampProgression:
    """Verify exact expected values at each turn of the ramp."""

    @pytest.fixture
    def default_params(self):
        return dict(
            sharpe_target=1.5,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )

    def test_turn1_value(self, default_params):
        # progress=0, target=1.5*0.5=0.75
        result = compute_effective_target(turn=1, **default_params)
        assert result == pytest.approx(0.75)

    def test_turn2_value(self, default_params):
        # progress=1/3, target=1.5*(0.5+0.5/3)=1.5*0.6667=1.0
        result = compute_effective_target(turn=2, **default_params)
        assert result == pytest.approx(1.0)

    def test_turn3_value(self, default_params):
        # progress=2/3, target=1.5*(0.5+1.0/3)=1.5*0.8333=1.25
        result = compute_effective_target(turn=3, **default_params)
        assert result == pytest.approx(1.25)

    def test_turn4_value(self, default_params):
        # past ramp → original
        result = compute_effective_target(turn=4, **default_params)
        assert result == pytest.approx(1.5)

    def test_turn5_value(self, default_params):
        result = compute_effective_target(turn=5, **default_params)
        assert result == pytest.approx(1.5)

    def test_turn6_value(self, default_params):
        result = compute_effective_target(turn=6, **default_params)
        assert result == pytest.approx(1.5)

    def test_turn7_value(self, default_params):
        result = compute_effective_target(turn=7, **default_params)
        assert result == pytest.approx(1.5)


# ── Different start_factors ────────────────────────────────────────────

class TestDifferentStartFactors:
    @pytest.mark.parametrize("factor", [0.1, 0.25, 0.3, 0.5, 0.7, 0.9])
    def test_turn1_uses_factor(self, factor):
        result = compute_effective_target(
            sharpe_target=2.0,
            turn=1,
            adaptive_sharpe_target=True,
            adaptive_start_factor=factor,
            adaptive_ramp_turns=5,
        )
        assert result == pytest.approx(2.0 * factor)

    def test_factor_0_1_very_low_start(self):
        """With factor=0.1, turn 1 target is 10% of full target."""
        result = compute_effective_target(
            sharpe_target=1.5,
            turn=1,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.1,
            adaptive_ramp_turns=4,
        )
        assert result == pytest.approx(0.15)

    def test_factor_1_0_passthrough(self):
        """With factor=1.0, target is always the full target."""
        for turn in range(1, 8):
            result = compute_effective_target(
                sharpe_target=1.5,
                turn=turn,
                adaptive_sharpe_target=True,
                adaptive_start_factor=1.0,
                adaptive_ramp_turns=5,
            )
            assert result == pytest.approx(1.5)


# ── Different ramp_turns ──────────────────────────────────────────────

class TestDifferentRampTurns:
    def test_ramp_1_turn(self):
        """ramp_turns=1: turn 1 starts at factor, turn 2+ is full."""
        t1 = compute_effective_target(
            sharpe_target=2.0, turn=1,
            adaptive_sharpe_target=True, adaptive_start_factor=0.5,
            adaptive_ramp_turns=1,
        )
        t2 = compute_effective_target(
            sharpe_target=2.0, turn=2,
            adaptive_sharpe_target=True, adaptive_start_factor=0.5,
            adaptive_ramp_turns=1,
        )
        # progress = 0/1 = 0 → target = 2.0 * 0.5 = 1.0
        assert t1 == pytest.approx(1.0)
        assert t2 == pytest.approx(2.0)

    def test_ramp_7_turns(self):
        """ramp_turns=7: gradual ramp across all 7 turns."""
        targets = [
            compute_effective_target(
                sharpe_target=1.5, turn=t,
                adaptive_sharpe_target=True, adaptive_start_factor=0.5,
                adaptive_ramp_turns=7,
            )
            for t in range(1, 8)
        ]
        # First should be 0.75, last should still be ramping (not yet full)
        assert targets[0] == pytest.approx(0.75)
        assert targets[-1] < 1.5  # turn 7 with ramp=7: progress=6/7 < 1
        # Monotonic
        for i in range(1, len(targets)):
            assert targets[i] > targets[i - 1]

    def test_ramp_10_turns(self):
        """ramp_turns larger than max_turns — target never reaches full."""
        for turn in range(1, 8):
            result = compute_effective_target(
                sharpe_target=1.5, turn=turn,
                adaptive_sharpe_target=True, adaptive_start_factor=0.5,
                adaptive_ramp_turns=10,
            )
            assert result < 1.5, f"Turn {turn} should not reach full target with ramp=10"


# ── Edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_sharpe_target(self):
        """Zero target should remain zero regardless of adaptive."""
        result = compute_effective_target(
            sharpe_target=0.0,
            turn=1,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        assert result == pytest.approx(0.0)

    def test_negative_sharpe_target(self):
        """Negative targets should still ramp correctly (math works)."""
        result = compute_effective_target(
            sharpe_target=-1.0,
            turn=1,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        assert result == pytest.approx(-0.5)

    def test_very_large_sharpe_target(self):
        result = compute_effective_target(
            sharpe_target=100.0,
            turn=1,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        assert result == pytest.approx(50.0)

    def test_zero_ramp_turns(self):
        """ramp_turns=0 would cause division by zero — should return original."""
        # With ramp_turns=0, turn > 0 > 0 is always True, so returns original
        result = compute_effective_target(
            sharpe_target=1.5,
            turn=1,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=0,
        )
        assert result == pytest.approx(1.5)


# ── RefinementConfig integration ───────────────────────────────────────

class TestRefinementConfigIntegration:
    def test_default_adaptive_disabled(self):
        cfg = RefinementConfig()
        assert cfg.adaptive_sharpe_target is False
        assert cfg.adaptive_start_factor == 0.5
        assert cfg.adaptive_ramp_turns == 3

    def test_config_serialization_roundtrip(self):
        """Adaptive fields survive JSON roundtrip."""
        cfg = RefinementConfig(
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.3,
            adaptive_ramp_turns=4,
        )
        json_str = cfg.to_json()
        cfg2 = RefinementConfig.from_json(json_str)
        assert cfg2.adaptive_sharpe_target is True
        assert cfg2.adaptive_start_factor == 0.3
        assert cfg2.adaptive_ramp_turns == 4

    def test_config_dict_roundtrip(self):
        cfg = RefinementConfig(
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.7,
            adaptive_ramp_turns=2,
        )
        d = cfg.to_dict()
        cfg2 = RefinementConfig.from_dict(d)
        assert cfg2.adaptive_sharpe_target is True
        assert cfg2.adaptive_start_factor == 0.7
        assert cfg2.adaptive_ramp_turns == 2

    def test_compute_with_config_fields(self):
        """Use config fields directly with compute_effective_target."""
        cfg = RefinementConfig(
            sharpe_target=2.0,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        result = compute_effective_target(
            sharpe_target=cfg.sharpe_target,
            turn=1,
            adaptive_sharpe_target=cfg.adaptive_sharpe_target,
            adaptive_start_factor=cfg.adaptive_start_factor,
            adaptive_ramp_turns=cfg.adaptive_ramp_turns,
        )
        assert result == pytest.approx(1.0)


# ── Final promotion gate semantics ─────────────────────────────────────

class TestFinalPromotionGate:
    """The final promotion gate should ALWAYS use the original sharpe_target,
    never the effective_target. These tests verify the semantic contract:
    the caller is responsible for using sharpe_target (not effective_target)
    at the promotion gate."""

    def test_effective_below_original_but_above_ramped(self):
        """A strategy with Sharpe=1.0 passes turn 1 (effective=0.75)
        but fails final promotion (original=1.5)."""
        sharpe_target = 1.5
        effective = compute_effective_target(
            sharpe_target=sharpe_target,
            turn=1,
            adaptive_sharpe_target=True,
            adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        strategy_sharpe = 1.0
        
        # Per-turn success: passes (1.0 >= 0.75)
        assert strategy_sharpe >= effective
        
        # Final promotion: fails (1.0 < 1.5)
        assert strategy_sharpe < sharpe_target

    def test_strategy_meets_original_always_promotes(self):
        """If strategy meets original target, it passes both gates."""
        sharpe_target = 1.5
        for turn in range(1, 8):
            effective = compute_effective_target(
                sharpe_target=sharpe_target,
                turn=turn,
                adaptive_sharpe_target=True,
                adaptive_start_factor=0.5,
                adaptive_ramp_turns=3,
            )
            strategy_sharpe = 1.6
            assert strategy_sharpe >= effective
            assert strategy_sharpe >= sharpe_target

    def test_strategy_between_effective_and_original(self):
        """Strategy that passes early turns but not final target."""
        sharpe_target = 2.0
        strategy_sharpe = 1.2
        
        # Turn 1: effective = 1.0 → passes
        t1_eff = compute_effective_target(
            sharpe_target=sharpe_target, turn=1,
            adaptive_sharpe_target=True, adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        assert strategy_sharpe >= t1_eff
        
        # Turn 3: effective = 1.667 → fails
        t3_eff = compute_effective_target(
            sharpe_target=sharpe_target, turn=3,
            adaptive_sharpe_target=True, adaptive_start_factor=0.5,
            adaptive_ramp_turns=3,
        )
        assert strategy_sharpe < t3_eff
        
        # Final promotion: fails (1.2 < 2.0)
        assert strategy_sharpe < sharpe_target


# ── Prompt display formatting (via prompts.py) ────────────────────────

class TestPromptDisplay:
    """Test that the prompt builders correctly handle adaptive target display."""

    def test_turn1_prompt_with_adaptive_target(self):
        """build_turn1_prompt should show ramping info when effective != target."""
        from crabquant.refinement.prompts import build_turn1_prompt
        
        mandate = {
            "name": "test",
            "strategy_archetype": "any",
            "sharpe_target": 1.5,
            "tickers": ["AAPL"],
            "period": "1y",
        }
        
        # Without adaptive target — should just show the number
        prompt_no_adaptive = build_turn1_prompt(
            mandate=mandate,
            current_turn=1,
            max_turns=7,
        )
        assert "1.50" in prompt_no_adaptive
        assert "ramping" not in prompt_no_adaptive
        
        # With adaptive target — should show ramping info
        prompt_adaptive = build_turn1_prompt(
            mandate=mandate,
            current_turn=1,
            max_turns=7,
            effective_target=0.75,
        )
        assert "0.75" in prompt_adaptive
        assert "ramping" in prompt_adaptive
        assert "1.50" in prompt_adaptive

    def test_turn1_prompt_equal_targets_no_ramp_text(self):
        """When effective == sharpe_target, no ramping text should appear."""
        from crabquant.refinement.prompts import build_turn1_prompt
        
        mandate = {
            "name": "test",
            "strategy_archetype": "any",
            "sharpe_target": 1.5,
            "tickers": ["AAPL"],
            "period": "1y",
        }
        
        prompt = build_turn1_prompt(
            mandate=mandate,
            current_turn=1,
            max_turns=7,
            effective_target=1.5,
        )
        assert "1.50" in prompt
        assert "ramping" not in prompt

    def test_refinement_prompt_with_adaptive_target(self):
        """build_refinement_prompt should show ramping info when effective != target."""
        from crabquant.refinement.prompts import build_refinement_prompt
        
        tier1 = {
            "sharpe_ratio": 0.8,
            "total_return_pct": 5.0,
            "max_drawdown_pct": 10.0,
            "total_trades": 30,
            "win_rate": 0.55,
            "profit_factor": 1.2,
            "sortino_ratio": 1.0,
            "calmar_ratio": 0.8,
            "failure_mode": "low_sharpe",
            "failure_details": "Below target",
            "sharpe_target": 1.5,
            "current_strategy_code": "# test",
            "current_params": {},
        }
        
        prompt = build_refinement_prompt(
            tier1_report=tier1,
            current_turn=2,
            max_turns=7,
            sharpe_target=1.5,
            effective_target=1.0,
            best_sharpe=0.8,
            best_turn=1,
        )
        assert "1.00" in prompt
        assert "ramping" in prompt
        assert "1.50" in prompt

    def test_refinement_prompt_no_adaptive(self):
        """Without effective_target, should just show sharpe_target."""
        from crabquant.refinement.prompts import build_refinement_prompt
        
        tier1 = {
            "sharpe_ratio": 0.8,
            "total_return_pct": 5.0,
            "max_drawdown_pct": 10.0,
            "total_trades": 30,
            "win_rate": 0.55,
            "profit_factor": 1.2,
            "sortino_ratio": 1.0,
            "calmar_ratio": 0.8,
            "failure_mode": "low_sharpe",
            "failure_details": "Below target",
            "sharpe_target": 1.5,
            "current_strategy_code": "# test",
            "current_params": {},
        }
        
        prompt = build_refinement_prompt(
            tier1_report=tier1,
            current_turn=2,
            max_turns=7,
            sharpe_target=1.5,
            best_sharpe=0.8,
            best_turn=1,
        )
        assert "1.50" in prompt
        assert "ramping" not in prompt


# ── context_builder integration ────────────────────────────────────────

class TestContextBuilderIntegration:
    """Test that build_llm_context passes effective_target correctly."""

    def test_context_includes_effective_target(self):
        from crabquant.refinement.context_builder import build_llm_context
        
        class FakeState:
            sharpe_target = 1.5
            current_turn = 0
            max_turns = 7
            tickers = ["AAPL"]
            period = "1y"
            mandate_name = "test"
            history = []
        
        state = FakeState()
        
        # Without effective_target → effective_target == sharpe_target
        ctx = build_llm_context(state)
        assert ctx["sharpe_target"] == 1.5
        assert ctx["effective_target"] == 1.5
        
        # With effective_target
        ctx2 = build_llm_context(state, effective_target=0.75)
        assert ctx2["sharpe_target"] == 1.5
        assert ctx2["effective_target"] == 0.75
