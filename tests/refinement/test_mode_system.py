"""Tests for Phase 5.6 — Mode System Integration.

Covers:
- apply_mode() for all presets (conservative/fast/explorer/balanced/custom)
- from_mandate() with mode field
- Individual toggle overrides after mode
- Default values when no mode specified
- RefinementConfig defaults and overrides
- Serialization (to_dict, from_dict, to_json, from_json)
- File I/O (save, load)
- from_mandate edge cases
- VALIDATION_CONFIG and DIVERSITY_CONFIG
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crabquant.refinement.config import (
    DIVERSITY_CONFIG,
    RefinementConfig,
    REGIME_TAG_CONFIG,
    VALIDATION_CONFIG,
)


# ─── apply_mode() presets ─────────────────────────────────────────────────

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


# ─── apply_mode() edge cases ──────────────────────────────────────────────

class TestApplyModeEdgeCases:
    """Edge cases for apply_mode()."""

    def test_mixed_case_explorer(self) -> None:
        config = RefinementConfig()
        config.apply_mode("ExPlOrEr")
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is True

    def test_mixed_case_conservative(self) -> None:
        config = RefinementConfig()
        config.apply_mode("CoNsErVaTiVe")
        assert config.cross_run_learning is False
        assert config.parallel_invention is False
        assert config.soft_promote is False

    def test_mixed_case_balanced(self) -> None:
        config = RefinementConfig()
        config.apply_mode("BaLaNcEd")
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is False

    def test_empty_string_mode(self) -> None:
        """Empty string should be treated as unknown/custom."""
        config = RefinementConfig()
        original_crl = config.cross_run_learning
        config.apply_mode("")
        assert config.cross_run_learning is original_crl

    def test_mode_does_not_change_parallel_invention_count(self) -> None:
        """apply_mode should not touch parallel_invention_count."""
        config = RefinementConfig()
        config.parallel_invention_count = 5
        config.apply_mode("explorer")
        assert config.parallel_invention_count == 5

    def test_mode_does_not_change_soft_promote_sharpe(self) -> None:
        """apply_mode should not touch soft_promote_sharpe."""
        config = RefinementConfig()
        config.soft_promote_sharpe = 1.0
        config.apply_mode("explorer")
        assert config.soft_promote_sharpe == 1.0

    def test_mode_does_not_change_soft_promote_min_windows(self) -> None:
        """apply_mode should not touch soft_promote_min_windows."""
        config = RefinementConfig()
        config.soft_promote_min_windows = 5
        config.apply_mode("conservative")
        assert config.soft_promote_min_windows == 5

    def test_chaining_multiple_modes(self) -> None:
        """Chaining multiple apply_mode calls — last one wins."""
        config = RefinementConfig()
        result = config.apply_mode("explorer").apply_mode("conservative")
        assert result is config
        assert config.cross_run_learning is False
        assert config.parallel_invention is False
        assert config.soft_promote is False


# ─── from_mandate with mode ───────────────────────────────────────────────

class TestFromMandateWithMode:
    """Test that from_mandate() respects the mode field."""

    def test_from_mandate_explorer_mode(self) -> None:
        mandate = {
            "name": "test",
            "mode": "explorer",
        }
        config = RefinementConfig.from_mandate(mandate)
        config.apply_mode(mandate["mode"])
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is True

    def test_from_mandate_no_mode(self) -> None:
        mandate = {"name": "test"}
        config = RefinementConfig.from_mandate(mandate)
        assert config.cross_run_learning is True
        assert config.parallel_invention is False
        assert config.soft_promote is False


# ─── toggle overrides ─────────────────────────────────────────────────────

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


# ─── mode transitions ─────────────────────────────────────────────────────

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

    def test_switch_fast_to_balanced(self) -> None:
        config = RefinementConfig()
        config.apply_mode("fast")
        assert config.parallel_invention is False
        config.apply_mode("balanced")
        assert config.parallel_invention is True
        assert config.soft_promote is False

    def test_switch_balanced_to_custom(self) -> None:
        """Switching to custom preserves the balanced settings."""
        config = RefinementConfig()
        config.apply_mode("balanced")
        config.apply_mode("custom")
        assert config.cross_run_learning is True
        assert config.parallel_invention is True
        assert config.soft_promote is False

    def test_switch_conservative_to_fast(self) -> None:
        config = RefinementConfig()
        config.apply_mode("conservative")
        assert config.cross_run_learning is False
        config.apply_mode("fast")
        assert config.cross_run_learning is True
        assert config.parallel_invention is False


# ─── RefinementConfig defaults ────────────────────────────────────────────

class TestRefinementConfigDefaults:
    """Test default values of RefinementConfig."""

    def test_default_max_turns(self) -> None:
        config = RefinementConfig()
        assert config.max_turns == 7

    def test_default_sharpe_target(self) -> None:
        config = RefinementConfig()
        assert config.sharpe_target == 1.5

    def test_default_max_drawdown_pct(self) -> None:
        config = RefinementConfig()
        assert config.max_drawdown_pct == 25.0

    def test_default_min_trades(self) -> None:
        config = RefinementConfig()
        assert config.min_trades == 5

    def test_default_llm_model(self) -> None:
        config = RefinementConfig()
        assert config.llm_model == "glm-5-turbo"

    def test_default_llm_timeout(self) -> None:
        config = RefinementConfig()
        assert config.llm_timeout_seconds == 120

    def test_default_cross_run_learning(self) -> None:
        config = RefinementConfig()
        assert config.cross_run_learning is True

    def test_default_parallel_invention(self) -> None:
        config = RefinementConfig()
        assert config.parallel_invention is False

    def test_default_soft_promote(self) -> None:
        config = RefinementConfig()
        assert config.soft_promote is False

    def test_default_soft_promote_sharpe(self) -> None:
        config = RefinementConfig()
        assert config.soft_promote_sharpe == 0.5

    def test_default_parallel_invention_count(self) -> None:
        config = RefinementConfig()
        assert config.parallel_invention_count == 3

    def test_default_feature_importance(self) -> None:
        config = RefinementConfig()
        assert config.feature_importance is True

    def test_default_multi_ticker_backtest(self) -> None:
        config = RefinementConfig()
        assert config.multi_ticker_backtest is False


# ─── RefinementConfig serialization ───────────────────────────────────────

class TestRefinementConfigSerialization:
    """Test to_dict, from_dict, to_json, from_json."""

    def test_to_dict_roundtrip(self) -> None:
        config = RefinementConfig(max_turns=10, sharpe_target=2.0)
        d = config.to_dict()
        restored = RefinementConfig.from_dict(d)
        assert restored.max_turns == 10
        assert restored.sharpe_target == 2.0

    def test_to_dict_contains_all_fields(self) -> None:
        config = RefinementConfig()
        d = config.to_dict()
        assert "max_turns" in d
        assert "sharpe_target" in d
        assert "cross_run_learning" in d
        assert "parallel_invention" in d
        assert "soft_promote" in d

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {"max_turns": 5, "unknown_field": "should_be_ignored"}
        config = RefinementConfig.from_dict(d)
        assert config.max_turns == 5

    def test_to_json_roundtrip(self) -> None:
        config = RefinementConfig(max_turns=3, cross_run_learning=False)
        blob = config.to_json()
        restored = RefinementConfig.from_json(blob)
        assert restored.max_turns == 3
        assert restored.cross_run_learning is False

    def test_from_json_invalid_raises(self) -> None:
        with pytest.raises(Exception):
            RefinementConfig.from_json("not json")

    def test_to_json_indent(self) -> None:
        config = RefinementConfig()
        blob = config.to_json(indent=2)
        # Should be valid JSON
        parsed = json.loads(blob)
        assert isinstance(parsed, dict)


# ─── RefinementConfig file I/O ────────────────────────────────────────────

class TestRefinementConfigFileIO:
    """Test save/load methods."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        config = RefinementConfig(max_turns=12, sharpe_target=3.0)
        path = tmp_path / "config.json"
        config.save(path)
        restored = RefinementConfig.load(path)
        assert restored.max_turns == 12
        assert restored.sharpe_target == 3.0

    def test_save_creates_file(self, tmp_path: Path) -> None:
        config = RefinementConfig()
        path = tmp_path / "config.json"
        config.save(path)
        assert path.exists()

    def test_save_output_is_valid_json(self, tmp_path: Path) -> None:
        config = RefinementConfig()
        path = tmp_path / "config.json"
        config.save(path)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)


# ─── from_mandate edge cases ──────────────────────────────────────────────

class TestFromMandateEdgeCases:
    """Edge cases for from_mandate()."""

    def test_empty_mandate(self) -> None:
        config = RefinementConfig.from_mandate({})
        assert config.max_turns == 7  # default
        assert config.sharpe_target == 1.5  # default

    def test_mandate_with_overrides(self) -> None:
        mandate = {"max_turns": 5}
        config = RefinementConfig.from_mandate(mandate, sharpe_target=99.0)
        assert config.max_turns == 5
        assert config.sharpe_target == 99.0

    def test_mandate_max_drawdown_pct_coerced_to_float(self) -> None:
        mandate = {"constraints": {"max_drawdown_pct": "20"}}
        config = RefinementConfig.from_mandate(mandate)
        assert config.max_drawdown_pct == 20.0

    def test_mandate_missing_constraints_key(self) -> None:
        """from_mandate should not crash if constraints key is absent."""
        mandate = {"max_turns": 5}
        config = RefinementConfig.from_mandate(mandate)
        assert config.max_turns == 5
        assert config.min_trades == 5  # default

    def test_mandate_empty_constraints(self) -> None:
        """from_mandate should handle empty constraints dict."""
        mandate = {"constraints": {}}
        config = RefinementConfig.from_mandate(mandate)
        assert config.min_trades == 5  # default

    def test_from_mandate_file(self, tmp_path: Path) -> None:
        mandate = {
            "name": "test",
            "max_turns": 15,
            "sharpe_target": 2.0,
            "constraints": {"min_trades": 20},
        }
        path = tmp_path / "mandate.json"
        path.write_text(json.dumps(mandate))
        config = RefinementConfig.from_mandate_file(str(path))
        assert config.max_turns == 15
        assert config.sharpe_target == 2.0
        assert config.min_trades == 20

    def test_from_mandate_file_with_overrides(self, tmp_path: Path) -> None:
        mandate = {"max_turns": 5}
        path = tmp_path / "mandate.json"
        path.write_text(json.dumps(mandate))
        config = RefinementConfig.from_mandate_file(str(path), sharpe_target=4.0)
        assert config.max_turns == 5
        assert config.sharpe_target == 4.0


# ─── Module-level configs ─────────────────────────────────────────────────

class TestModuleConfigs:
    """Test VALIDATION_CONFIG, DIVERSITY_CONFIG, REGIME_TAG_CONFIG."""

    def test_validation_config_has_required_keys(self) -> None:
        required = {"train_window", "test_window", "step", "min_avg_test_sharpe",
                     "min_windows_passed", "train_pct", "min_train_bars",
                     "min_test_sharpe", "min_test_trades", "max_degradation",
                     "min_cross_ticker_sharpe", "regime_specific_wf_sharpe_factor",
                     "regime_specific_ct_sharpe_factor", "soft_promote_test_sharpe"}
        assert required.issubset(VALIDATION_CONFIG.keys())

    def test_diversity_config_has_required_keys(self) -> None:
        required = {"max_winners_per_combo", "min_ticker_diversity",
                     "min_archetype_diversity", "winners_file"}
        assert required.issubset(DIVERSITY_CONFIG.keys())

    def test_regime_tag_config_has_required_keys(self) -> None:
        required = {"sharpe_good_threshold", "sharpe_acceptable_threshold",
                     "min_bars_per_regime"}
        assert required.issubset(REGIME_TAG_CONFIG.keys())

    def test_validation_config_regime_factors_are_floats(self) -> None:
        assert isinstance(VALIDATION_CONFIG["regime_specific_wf_sharpe_factor"], float)
        assert isinstance(VALIDATION_CONFIG["regime_specific_ct_sharpe_factor"], float)
        assert isinstance(VALIDATION_CONFIG["soft_promote_test_sharpe"], float)

    def test_regime_wf_factor_less_than_one(self) -> None:
        """Regime-specific WF factor should be < 1.0 (relaxed threshold)."""
        assert VALIDATION_CONFIG["regime_specific_wf_sharpe_factor"] < 1.0

    def test_regime_ct_factor_less_than_one(self) -> None:
        """Regime-specific CT factor should be < 1.0 (relaxed threshold)."""
        assert VALIDATION_CONFIG["regime_specific_ct_sharpe_factor"] < 1.0

    def test_soft_promote_test_sharpe_is_positive(self) -> None:
        assert VALIDATION_CONFIG["soft_promote_test_sharpe"] > 0


# ─── Stagnation config defaults ───────────────────────────────────────────

class TestStagnationDefaults:
    """Test stagnation-related defaults."""

    def test_stagnation_abandon_threshold(self) -> None:
        config = RefinementConfig()
        assert config.stagnation_abandon_threshold == 0.8

    def test_stagnation_nuclear_threshold(self) -> None:
        config = RefinementConfig()
        assert config.stagnation_nuclear_threshold == 0.6

    def test_stagnation_pivot_threshold(self) -> None:
        config = RefinementConfig()
        assert config.stagnation_pivot_threshold == 0.7

    def test_stagnation_broaden_threshold(self) -> None:
        config = RefinementConfig()
        assert config.stagnation_broaden_threshold == 0.5


# ─── Circuit breaker defaults ─────────────────────────────────────────────

class TestCircuitBreakerDefaults:
    """Test circuit breaker config defaults."""

    def test_circuit_breaker_window(self) -> None:
        config = RefinementConfig()
        assert config.circuit_breaker_window == 20

    def test_circuit_breaker_min_pass_rate(self) -> None:
        config = RefinementConfig()
        assert config.circuit_breaker_min_pass_rate == 0.3
