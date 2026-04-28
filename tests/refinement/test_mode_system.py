"""Tests for Phase 5.6 — Mode System Integration.

Covers:
- apply_mode() for all presets (conservative/fast/explorer/balanced/custom)
- from_mandate() with mode field
- Individual toggle overrides after mode
- Default values when no mode specified
"""

from __future__ import annotations

import pytest

from crabquant.refinement.config import RefinementConfig


class TestApplyModePresets:
    """Test that apply_mode() correctly sets toggles for each preset."""

    def test_conervative_mode(self) -> None:
        config = RefinementConfig()
        config.cross_run_learning = True  # Start with True
        config.parallel_invention = True
        config.soft_promote = True
        config.apply_mode("conservative")
        assert config.cross_run_learning is False
        assert config.parallel_invention is False
        assert config.soft_promote is False

    def test_fast_mode(self) -> None:
        config = RefinementConfig()
        config.apply_mode("fast")
        assert config.cross_run_learning is True
        assert config.parallel_invention is False
        assert config.soft_promote is False

    def test_explorer_mode(self) -> None:
        config = RefinementConfig()
        config.apply_mode("explorer")
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is True

    def test_balanced_mode(self) -> None:
        config = RefinementConfig()
        config.apply_mode("balanced")
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is False

    def test_custom_mode_preserves_existing(self) -> None:
        """Custom mode should leave toggles unchanged."""
        config = RefinementConfig()
        config.cross_run_learning = True
        config.parallel_invention = False
        config.soft_promote = True
        config.apply_mode("custom")
        assert config.cross_run_learning is True
        assert config.parallel_invention is False
        assert config.soft_promote is True

    def test_unknown_mode_preserves_existing(self) -> None:
        """Unknown mode should leave toggles unchanged."""
        config = RefinementConfig()
        config.cross_run_learning = True
        config.parallel_invention = False
        config.apply_mode("nonexistent_mode")
        assert config.cross_run_learning is True
        assert config.parallel_invention is False

    def test_case_insensitive(self) -> None:
        config = RefinementConfig()
        config.apply_mode("EXPLORER")
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is True

    def test_whitespace_stripped(self) -> None:
        config = RefinementConfig()
        config.apply_mode("  fast  ")
        assert config.cross_run_learning is True
        assert config.parallel_invention is False

    def test_returns_self_for_chaining(self) -> None:
        config = RefinementConfig()
        result = config.apply_mode("explorer")
        assert result is config


class TestFromMandateWithMode:
    """Test that from_mandate() respects the mode field."""

    def test_from_mandate_explorer_mode(self) -> None:
        mandate = {
            "name": "test",
            "mode": "explorer",
        }
        config = RefinementConfig.from_mandate(mandate)
        # from_mandate doesn't apply mode — refinement_loop does that separately
        # So we verify apply_mode works when called after from_mandate
        config.apply_mode(mandate["mode"])
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is True

    def test_from_mandate_no_mode(self) -> None:
        mandate = {"name": "test"}
        config = RefinementConfig.from_mandate(mandate)
        # No mode → defaults unchanged (cross_run_learning defaults to True)
        assert config.cross_run_learning is True
        assert config.parallel_invention is False
        assert config.soft_promote is False


class TestToggleOverrides:
    """Test that individual toggles override mode presets."""

    def test_override_after_mode(self) -> None:
        """After apply_mode, individual attribute assignment should override."""
        config = RefinementConfig()
        config.apply_mode("explorer")  # all True
        config.soft_promote = False  # override
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is False

    def test_from_mandate_with_sharpe_target(self) -> None:
        mandate = {
            "name": "test",
            "sharpe_target": 2.5,
            "max_turns": 10,
        }
        config = RefinementConfig.from_mandate(mandate)
        assert config.sharpe_target == 2.5
        assert config.max_turns == 10

    def test_from_mandate_with_constraints(self) -> None:
        mandate = {
            "name": "test",
            "constraints": {
                "min_trades": 30,
                "max_drawdown_pct": 15,
            },
        }
        config = RefinementConfig.from_mandate(mandate)
        assert config.min_trades == 30
        assert config.max_drawdown_pct == 15.0


class TestModeTransitions:
    """Test switching between modes."""

    def test_switch_from_conservative_to_explorer(self) -> None:
        config = RefinementConfig()
        config.apply_mode("conservative")
        assert config.cross_run_learning is False
        config.apply_mode("explorer")
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is True

    def test_switch_from_explorer_to_conservative(self) -> None:
        config = RefinementConfig()
        config.apply_mode("explorer")
        config.apply_mode("conservative")
        assert config.cross_run_learning is False
        assert config.parallel_invention is False
        assert config.soft_promote is False
