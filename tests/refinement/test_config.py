"""Tests for crabquant.refinement.config — TDD first pass."""

import json
import tempfile
from pathlib import Path

import pytest

from crabquant.refinement.config import RefinementConfig


# ─── Defaults ────────────────────────────────────────────────────────────────

class TestDefaults:
    def test_loop_defaults(self):
        cfg = RefinementConfig()
        assert cfg.max_turns == 7
        assert cfg.sharpe_target == 1.5
        assert cfg.max_drawdown_pct == 25.0
        assert cfg.min_trades == 5

    def test_llm_defaults(self):
        cfg = RefinementConfig()
        assert cfg.llm_model == "glm-5-turbo"
        assert cfg.llm_base_url == "https://api.z.ai/api/coding/paas/v4"
        assert cfg.llm_api_key_env == ""
        assert cfg.llm_timeout_seconds == 120
        assert cfg.max_code_repair_attempts == 3
        assert cfg.max_llm_output_tokens == 4096

    def test_validation_defaults(self):
        cfg = RefinementConfig()
        assert cfg.smoke_backtest_timeout == 10
        assert cfg.signal_sanity_ticker == "AAPL"
        assert cfg.signal_sanity_period == "1y"

    def test_stagnation_defaults(self):
        cfg = RefinementConfig()
        assert cfg.stagnation_abandon_threshold == 0.8
        assert cfg.stagnation_nuclear_threshold == 0.6
        assert cfg.stagnation_pivot_threshold == 0.7
        assert cfg.stagnation_broaden_threshold == 0.5

    def test_tier2_defaults(self):
        cfg = RefinementConfig()
        assert cfg.tier2_stagnation_trigger == 0.4
        assert cfg.tier2_iteration_trigger == 3

    def test_circuit_breaker_defaults(self):
        cfg = RefinementConfig()
        assert cfg.circuit_breaker_window == 20
        assert cfg.circuit_breaker_min_pass_rate == 0.3

    def test_timeout_defaults(self):
        cfg = RefinementConfig()
        assert cfg.per_strategy_timeout_minutes == 15
        assert cfg.backtest_timeout_seconds == 60
        assert cfg.lock_staleness_minutes == 20

    def test_path_defaults(self):
        cfg = RefinementConfig()
        assert cfg.mandates_dir == "refinement/mandates"
        assert cfg.runs_dir == "refinement/runs"
        assert cfg.winners_file == "results/winners/winners.json"


# ─── Serialisation ───────────────────────────────────────────────────────────

class TestSerialisation:
    def test_to_dict_returns_dict(self):
        cfg = RefinementConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_all_defaults(self):
        cfg = RefinementConfig()
        d = cfg.to_dict()
        assert d["max_turns"] == 7
        assert d["sharpe_target"] == 1.5
        assert d["llm_model"] == "glm-5-turbo"
        assert d["mandates_dir"] == "refinement/mandates"

    def test_from_dict_roundtrip(self):
        cfg = RefinementConfig(max_turns=10, sharpe_target=2.0, llm_model="gpt-4o")
        d = cfg.to_dict()
        cfg2 = RefinementConfig.from_dict(d)
        assert cfg2.max_turns == 10
        assert cfg2.sharpe_target == 2.0
        assert cfg2.llm_model == "gpt-4o"

    def test_from_dict_unknown_keys_ignored(self):
        d = RefinementConfig().to_dict()
        d["unknown_future_field"] = "should_be_ignored"
        cfg = RefinementConfig.from_dict(d)
        assert cfg.max_turns == 7

    def test_to_json_returns_string(self):
        cfg = RefinementConfig()
        blob = cfg.to_json()
        assert isinstance(blob, str)

    def test_to_json_is_valid_json(self):
        cfg = RefinementConfig()
        blob = cfg.to_json()
        parsed = json.loads(blob)
        assert parsed["max_turns"] == 7
        assert parsed["llm_model"] == "glm-5-turbo"

    def test_from_json_roundtrip(self):
        cfg = RefinementConfig(max_turns=5, signal_sanity_ticker="SPY")
        blob = cfg.to_json()
        cfg2 = RefinementConfig.from_json(blob)
        assert cfg2.max_turns == 5
        assert cfg2.signal_sanity_ticker == "SPY"

    def test_json_roundtrip_preserves_floats(self):
        cfg = RefinementConfig(sharpe_target=1.75, stagnation_abandon_threshold=0.85)
        cfg2 = RefinementConfig.from_json(cfg.to_json())
        assert cfg2.sharpe_target == 1.75
        assert cfg2.stagnation_abandon_threshold == 0.85

    def test_json_roundtrip_preserves_ints(self):
        cfg = RefinementConfig(max_turns=12, backtest_timeout_seconds=90)
        cfg2 = RefinementConfig.from_json(cfg.to_json())
        assert cfg2.max_turns == 12
        assert cfg2.backtest_timeout_seconds == 90


# ─── File I/O ────────────────────────────────────────────────────────────────

class TestFileIO:
    def test_save_creates_file(self, tmp_path):
        cfg = RefinementConfig()
        p = tmp_path / "config.json"
        cfg.save(p)
        assert p.exists()

    def test_save_writes_valid_json(self, tmp_path):
        cfg = RefinementConfig(max_turns=9)
        p = tmp_path / "config.json"
        cfg.save(p)
        parsed = json.loads(p.read_text())
        assert parsed["max_turns"] == 9

    def test_load_roundtrip(self, tmp_path):
        cfg = RefinementConfig(max_turns=4, llm_model="claude-3", runs_dir="custom/runs")
        p = tmp_path / "config.json"
        cfg.save(p)
        cfg2 = RefinementConfig.load(p)
        assert cfg2.max_turns == 4
        assert cfg2.llm_model == "claude-3"
        assert cfg2.runs_dir == "custom/runs"

    def test_save_accepts_string_path(self, tmp_path):
        cfg = RefinementConfig()
        p = str(tmp_path / "config.json")
        cfg.save(p)
        assert Path(p).exists()

    def test_load_accepts_string_path(self, tmp_path):
        cfg = RefinementConfig(max_turns=3)
        p = tmp_path / "config.json"
        cfg.save(p)
        cfg2 = RefinementConfig.load(str(p))
        assert cfg2.max_turns == 3


# ─── from_mandate ─────────────────────────────────────────────────────────────

class TestFromMandate:
    def _sample_mandate(self, **overrides) -> dict:
        base = {
            "name": "momentum_spy_1",
            "description": "Momentum strategy",
            "strategy_archetype": "momentum",
            "tickers": ["SPY", "AAPL"],
            "primary_ticker": "SPY",
            "period": "2y",
            "sharpe_target": 1.8,
            "max_turns": 5,
            "constraints": {
                "min_trades": 10,
                "max_drawdown_pct": 20,
            },
        }
        base.update(overrides)
        return base

    def test_from_mandate_overrides_sharpe_target(self):
        mandate = self._sample_mandate(sharpe_target=2.0)
        cfg = RefinementConfig.from_mandate(mandate)
        assert cfg.sharpe_target == 2.0

    def test_from_mandate_overrides_max_turns(self):
        mandate = self._sample_mandate(max_turns=3)
        cfg = RefinementConfig.from_mandate(mandate)
        assert cfg.max_turns == 3

    def test_from_mandate_overrides_min_trades(self):
        mandate = self._sample_mandate()
        mandate["constraints"]["min_trades"] = 15
        cfg = RefinementConfig.from_mandate(mandate)
        assert cfg.min_trades == 15

    def test_from_mandate_overrides_max_drawdown_pct(self):
        mandate = self._sample_mandate()
        mandate["constraints"]["max_drawdown_pct"] = 30
        cfg = RefinementConfig.from_mandate(mandate)
        assert cfg.max_drawdown_pct == 30.0

    def test_from_mandate_preserves_llm_defaults(self):
        cfg = RefinementConfig.from_mandate(self._sample_mandate())
        assert cfg.llm_model == "glm-5-turbo"
        assert cfg.llm_timeout_seconds == 120

    def test_from_mandate_preserves_path_defaults(self):
        cfg = RefinementConfig.from_mandate(self._sample_mandate())
        assert cfg.mandates_dir == "refinement/mandates"
        assert cfg.runs_dir == "refinement/runs"

    def test_from_mandate_no_constraints_uses_defaults(self):
        mandate = {"name": "bare", "sharpe_target": 1.5, "max_turns": 7}
        cfg = RefinementConfig.from_mandate(mandate)
        assert cfg.min_trades == 5
        assert cfg.max_drawdown_pct == 25.0

    def test_from_mandate_partial_constraints(self):
        mandate = {"name": "partial", "constraints": {"min_trades": 8}}
        cfg = RefinementConfig.from_mandate(mandate)
        assert cfg.min_trades == 8
        assert cfg.max_drawdown_pct == 25.0

    def test_from_mandate_extra_kwargs_override(self):
        mandate = self._sample_mandate(max_turns=5)
        cfg = RefinementConfig.from_mandate(mandate, max_turns=9, llm_model="custom")
        assert cfg.max_turns == 9
        assert cfg.llm_model == "custom"

    def test_from_mandate_missing_optional_fields_uses_defaults(self):
        mandate = {}
        cfg = RefinementConfig.from_mandate(mandate)
        assert cfg.max_turns == 7
        assert cfg.sharpe_target == 1.5

    def test_from_mandate_file(self, tmp_path):
        mandate = self._sample_mandate(sharpe_target=1.9, max_turns=6)
        p = tmp_path / "mandate.json"
        p.write_text(json.dumps(mandate))
        cfg = RefinementConfig.from_mandate_file(p)
        assert cfg.sharpe_target == 1.9
        assert cfg.max_turns == 6

    def test_from_mandate_file_string_path(self, tmp_path):
        mandate = self._sample_mandate(max_turns=4)
        p = tmp_path / "mandate.json"
        p.write_text(json.dumps(mandate))
        cfg = RefinementConfig.from_mandate_file(str(p))
        assert cfg.max_turns == 4


# ─── Type correctness ────────────────────────────────────────────────────────

class TestTypes:
    def test_int_fields_are_int(self):
        cfg = RefinementConfig()
        assert isinstance(cfg.max_turns, int)
        assert isinstance(cfg.min_trades, int)
        assert isinstance(cfg.llm_timeout_seconds, int)
        assert isinstance(cfg.max_code_repair_attempts, int)
        assert isinstance(cfg.max_llm_output_tokens, int)
        assert isinstance(cfg.smoke_backtest_timeout, int)
        assert isinstance(cfg.tier2_iteration_trigger, int)
        assert isinstance(cfg.circuit_breaker_window, int)
        assert isinstance(cfg.per_strategy_timeout_minutes, int)
        assert isinstance(cfg.backtest_timeout_seconds, int)
        assert isinstance(cfg.lock_staleness_minutes, int)

    def test_float_fields_are_float(self):
        cfg = RefinementConfig()
        assert isinstance(cfg.sharpe_target, float)
        assert isinstance(cfg.max_drawdown_pct, float)
        assert isinstance(cfg.stagnation_abandon_threshold, float)
        assert isinstance(cfg.stagnation_nuclear_threshold, float)
        assert isinstance(cfg.stagnation_pivot_threshold, float)
        assert isinstance(cfg.stagnation_broaden_threshold, float)
        assert isinstance(cfg.tier2_stagnation_trigger, float)
        assert isinstance(cfg.circuit_breaker_min_pass_rate, float)

    def test_str_fields_are_str(self):
        cfg = RefinementConfig()
        assert isinstance(cfg.llm_model, str)
        assert isinstance(cfg.llm_base_url, str)
        assert isinstance(cfg.llm_api_key_env, str)
        assert isinstance(cfg.signal_sanity_ticker, str)
        assert isinstance(cfg.signal_sanity_period, str)
        assert isinstance(cfg.mandates_dir, str)
        assert isinstance(cfg.runs_dir, str)
        assert isinstance(cfg.winners_file, str)


# ─── Per-window walk-forward thresholds (Phase 6) ───────────────────────────

class TestPerWindowThresholds:
    def test_defaults_are_relaxed(self):
        cfg = RefinementConfig()
        assert cfg.min_window_test_sharpe == 0.0
        assert cfg.max_window_degradation == 1.0

    def test_from_dict_custom_values(self):
        cfg = RefinementConfig.from_dict({
            "min_window_test_sharpe": 0.2,
            "max_window_degradation": 0.5,
        })
        assert cfg.min_window_test_sharpe == 0.2
        assert cfg.max_window_degradation == 0.5

    def test_to_dict_includes_new_fields(self):
        cfg = RefinementConfig()
        d = cfg.to_dict()
        assert "min_window_test_sharpe" in d
        assert "max_window_degradation" in d
        assert d["min_window_test_sharpe"] == 0.0
        assert d["max_window_degradation"] == 1.0

    def test_json_roundtrip_preserves_values(self):
        cfg = RefinementConfig(min_window_test_sharpe=0.15, max_window_degradation=0.6)
        cfg2 = RefinementConfig.from_json(cfg.to_json())
        assert cfg2.min_window_test_sharpe == 0.15
        assert cfg2.max_window_degradation == 0.6

    def test_fields_are_float(self):
        cfg = RefinementConfig()
        assert isinstance(cfg.min_window_test_sharpe, float)
        assert isinstance(cfg.max_window_degradation, float)
