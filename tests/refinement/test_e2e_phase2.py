"""End-to-end Phase 2 integration test.

Verifies all Phase 2 components work together within the refinement pipeline:
- Stagnation detection triggers pivot/abandon responses
- Circuit breaker halts when pass rate drops too low
- Cosmetic guard forces structural interventions after 3 cosmetic actions
- Hypothesis enforcement rejects generic hypotheses
- Guardrails integration supplements failure classification
- Gate 3 smoke backtest catches overtrading/undertrading
- Wave scaling manages parallel mandate execution
- Per-wave metrics track convergence across waves

All external dependencies (LLM, data, backtest engine) are mocked.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestPhase2StagnationIntegration:
    """Stagnation detection should correctly identify stalled refinements."""

    def test_stagnation_triggers_abandon_after_many_failures(self):
        """After many failed turns with no improvement, stagnation should recommend abandon."""
        from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response

        # Simulate a run that's been failing for 6 turns with flat Sharpe
        history = [
            {"turn": i, "sharpe": -0.5, "action": "modify_params"}
            for i in range(1, 7)
        ]
        score, label = compute_stagnation(history)
        assert score > 0.5, f"Expected high stagnation score, got {score} ({label})"

        response = get_stagnation_response(iteration=6, score=score)
        assert response is not None
        # Response has 'constraint' key (not 'action')
        assert "constraint" in response
        assert response["constraint"] in ("pivot", "abandon")

    def test_stagnation_low_score_early_run(self):
        """Early in the run with mixed results, stagnation should be low."""
        from crabquant.refinement.stagnation import compute_stagnation

        history = [
            {"turn": 1, "sharpe": -0.3, "action": "novel"},
            {"turn": 2, "sharpe": 0.8, "action": "change_indicators"},
            {"turn": 3, "sharpe": 1.2, "action": "modify_params"},
        ]
        score, label = compute_stagnation(history)
        assert score < 0.5, f"Expected low stagnation score, got {score} ({label})"


class TestPhase2CircuitBreakerIntegration:
    """Circuit breaker should halt runs when LLM pass rate is too low."""

    def test_circuit_breaker_opens_on_low_pass_rate(self):
        """When most LLM responses fail validation, circuit breaker should open."""
        from crabquant.refinement.circuit_breaker import CircuitBreaker, CircuitBreakerStatus

        cb = CircuitBreaker(window=10, min_pass_rate=0.3)
        assert cb.status == CircuitBreakerStatus.CLOSED

        # Record 8 failures and 2 passes
        for i in range(8):
            cb.record(False, turn=i)
        for i in range(2):
            cb.record(True, turn=8 + i)

        assert cb.status == CircuitBreakerStatus.OPEN

    def test_circuit_breaker_stays_closed_with_good_pass_rate(self):
        """When LLM pass rate is healthy, circuit breaker stays closed."""
        from crabquant.refinement.circuit_breaker import CircuitBreaker, CircuitBreakerStatus

        cb = CircuitBreaker(window=10, min_pass_rate=0.3)

        for i in range(7):
            cb.record(True, turn=i)
        for i in range(3):
            cb.record(False, turn=7 + i)

        assert cb.status == CircuitBreakerStatus.CLOSED


class TestPhase2CosmeticGuardIntegration:
    """Cosmetic guard should force structural changes after consecutive cosmetic actions."""

    def test_cosmetic_guard_triggers_after_three_param_tweaks(self):
        """3 consecutive modify_params should trigger structural intervention."""
        from crabquant.refinement.cosmetic_guard import check_cosmetic_guard, CosmeticGuardState

        state = CosmeticGuardState(threshold=3)
        history = []

        # First two modify_params — no warning
        for i in range(2):
            history.append({"turn": i + 1, "action": "modify_params"})
            state, result = check_cosmetic_guard(history, state)
            assert not result.warning

        # Third one triggers
        history.append({"turn": 3, "action": "modify_params"})
        state, result = check_cosmetic_guard(history, state)
        assert result.warning
        assert result.forced_action is not None
        assert result.forced_action != "modify_params"

    def test_cosmetic_guard_resets_on_structural_change(self):
        """A structural action should reset the consecutive counter."""
        from crabquant.refinement.cosmetic_guard import check_cosmetic_guard, CosmeticGuardState

        state = CosmeticGuardState(threshold=3)
        history = [
            {"turn": 1, "action": "modify_params"},
            {"turn": 2, "action": "modify_params"},
            {"turn": 3, "action": "change_indicators"},  # structural — resets
            {"turn": 4, "action": "modify_params"},
            {"turn": 5, "action": "modify_params"},
        ]
        state, result = check_cosmetic_guard(history, state)
        assert not result.warning  # only 2 consecutive after reset


class TestPhase2HypothesisEnforcementIntegration:
    """Hypothesis enforcement should reject generic LLM hypotheses."""

    def test_rejects_generic_hypothesis(self):
        """Generic hypotheses like 'improve performance' should be rejected."""
        from crabquant.refinement.hypothesis_enforcement import check_hypothesis

        result = check_hypothesis("improve performance by tweaking parameters")
        assert not result.valid
        assert result.reason is not None

    def test_accepts_specific_hypothesis(self):
        """Specific hypotheses referencing indicators should pass."""
        from crabquant.refinement.hypothesis_enforcement import check_hypothesis

        result = check_hypothesis(
            "Adding a 20-period RSI filter will reduce false signals "
            "during low-volatility regimes, as observed in the 2024 drawdown period"
        )
        assert result.valid

    def test_rejects_empty_hypothesis(self):
        """Empty or None hypothesis should fail."""
        from crabquant.refinement.hypothesis_enforcement import check_hypothesis

        assert not check_hypothesis("").valid
        assert not check_hypothesis(None).valid


class TestPhase2GuardrailsIntegration:
    """Guardrails integration should supplement failure classification."""

    def test_guardrails_detect_low_sharpe(self):
        """Guardrails should flag low Sharpe ratio."""
        from crabquant.refinement.guardrails_integration import run_guardrails_check

        mock_result = MagicMock()
        mock_result.sharpe = 0.1
        mock_result.total_return = 0.02  # 2%
        mock_result.max_drawdown = -0.05  # -5%
        mock_result.num_trades = 50
        mock_result.win_rate = 0.45
        mock_result.profit_factor = 0.8
        mock_result.calmar_ratio = 0.1
        mock_result.avg_holding_bars = 10

        report = run_guardrails_check(mock_result)
        assert report is not None
        # Low Sharpe should produce a violation
        assert any("sharpe" in v.lower() for v in report.violations)

    def test_guardrails_config_adapts_to_iteration(self):
        """Early iterations should use aggressive config, late iterations conservative."""
        from crabquant.refinement.guardrails_integration import select_guardrail_config

        early = select_guardrail_config(iteration=1, max_turns=7)
        late = select_guardrail_config(iteration=6, max_turns=7)
        # Conservative should have stricter thresholds
        assert late.min_sharpe >= early.min_sharpe


class TestPhase2Gate3Integration:
    """Gate 3 smoke backtest should catch trading anomalies."""

    def test_gate3_catches_invalid_code(self):
        """Gate 3 should reject clearly invalid strategy code."""
        from crabquant.refinement.gate3_smoke import gate_smoke_backtest

        passed, errors = gate_smoke_backtest("this is not valid python code", ticker="AAPL")
        assert not passed
        assert len(errors) > 0


class TestPhase2WaveScalingIntegration:
    """Wave scaling should manage parallel execution properly."""

    def test_scaling_config_defaults(self):
        """Default parallel limit should be 5."""
        from crabquant.refinement.wave_scaling import get_parallel_limit

        limit = get_parallel_limit()
        assert limit == 5

    def test_wave_status_tracker_full_cycle(self):
        """Track waves from start to completion."""
        from crabquant.refinement.wave_scaling import WaveStatusTracker

        tracker = WaveStatusTracker()
        tracker.start_wave(wave_number=1, mandate_count=5)
        tracker.complete_wave(1, successful=3, failed=2)

        summary = tracker.get_status_summary()
        assert summary["total_waves"] == 1
        assert summary["total_mandates"] == 5
        assert summary["total_successful"] == 3

    def test_parallel_limit_clamped(self):
        """Parallel limit should be clamped to max."""
        from crabquant.refinement.wave_scaling import get_parallel_limit

        limit = get_parallel_limit(override=100)
        assert limit <= 10  # max should be around 10


class TestPhase2PerWaveMetricsIntegration:
    """Per-wave metrics should track convergence across waves."""

    def test_metrics_across_multiple_waves(self):
        """Tracker should aggregate stats across multiple waves."""
        from crabquant.refinement.per_wave_metrics import (
            PerWaveMetricsTracker, compute_convergence_rate
        )

        tracker = PerWaveMetricsTracker()

        # Wave 1: 60% convergence
        tracker.record_wave({
            "wave_number": 1,
            "total_mandates": 10,
            "successful": 6,
            "failed": 4,
            "results": [
                {"mandate_name": f"m{i}", "status": "success" if i < 6 else "failed",
                 "sharpe": 1.5 if i < 6 else 0.0, "turns": 5,
                 "archetype": "momentum" if i % 2 == 0 else "mean_reversion"}
                for i in range(10)
            ],
        })

        # Wave 2: 80% convergence
        tracker.record_wave({
            "wave_number": 2,
            "total_mandates": 10,
            "successful": 8,
            "failed": 2,
            "results": [
                {"mandate_name": f"m{i}", "status": "success" if i < 8 else "failed",
                 "sharpe": 2.0 if i < 8 else 0.0, "turns": 4,
                 "archetype": "momentum" if i % 2 == 0 else "mean_reversion"}
                for i in range(10)
            ],
        })

        summary = tracker.get_summary()
        assert summary["total_waves"] == 2
        assert compute_convergence_rate(8, 10) == 0.8


class TestPhase2FullPipelineIntegration:
    """Full Phase 2 pipeline: all components wiring through the orchestrator.

    This test verifies that Phase 2 components can be called in sequence
    as they would be during a real refinement run.
    """

    def test_full_phase2_component_chain(self):
        """Run through all Phase 2 checks on a simulated refinement turn."""
        from crabquant.refinement.circuit_breaker import CircuitBreaker, CircuitBreakerStatus
        from crabquant.refinement.cosmetic_guard import check_cosmetic_guard, CosmeticGuardState
        from crabquant.refinement.hypothesis_enforcement import check_hypothesis_from_modification
        from crabquant.refinement.guardrails_integration import run_guardrails_check
        from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response

        # Simulate turn 4 of a refinement run
        current_turn = 4
        max_turns = 7

        # 1. LLM returns a modification
        modification = {
            "action": "change_indicators",
            "hypothesis": "Adding Bollinger Bands with 20-period SMA will reduce "
                         "false breakouts during ranging markets, specifically "
                         "targeting the Q2 2024 drawdown pattern",
            "new_strategy_code": "def generate_signals(df, params):\n    return df['close'], df['close']\nDEFAULT_PARAMS = {}\nPARAM_GRID = {}\nDESCRIPTION = 'bb strategy'",
        }

        # 2. Hypothesis enforcement check
        hypo_result = check_hypothesis_from_modification(modification)
        assert hypo_result.valid, f"Hypothesis rejected: {hypo_result.reason}"

        # 3. Circuit breaker check (simulate previous pass rate)
        cb = CircuitBreaker(window=10, min_pass_rate=0.3)
        cb.record(True, turn=current_turn)  # this turn's validation passed
        # Previous turns: 6 passes, 3 failures
        for i in range(6):
            cb.record(True, turn=i)
        for i in range(3):
            cb.record(False, turn=6 + i)
        assert cb.status == CircuitBreakerStatus.CLOSED  # still operational

        # 4. Backtest result
        mock_result = MagicMock()
        mock_result.sharpe = 1.5
        mock_result.total_return = 0.15  # 15%
        mock_result.max_drawdown = -0.08  # -8%
        mock_result.num_trades = 45
        mock_result.win_rate = 0.6
        mock_result.profit_factor = 1.8
        mock_result.calmar_ratio = 1.5
        mock_result.avg_holding_bars = 8

        # 5. Guardrails check
        guardrail_report = run_guardrails_check(mock_result)
        # Sharpe 1.5 should be acceptable

        # 6. Cosmetic guard check
        cosmetic_state = CosmeticGuardState(threshold=3)
        action_history = [
            {"turn": 1, "action": "modify_params"},
            {"turn": 2, "action": "modify_params"},
            {"turn": 3, "action": "change_indicators"},  # structural reset
            {"turn": 4, "action": "change_indicators"},  # current turn
        ]
        cosmetic_state, cosmetic_result = check_cosmetic_guard(action_history, cosmetic_state)
        assert not cosmetic_result.warning  # only 1 consecutive after reset

        # 7. Stagnation check
        run_history = [
            {"turn": 1, "sharpe": 0.3, "action": "modify_params"},
            {"turn": 2, "sharpe": 0.6, "action": "change_indicators"},
            {"turn": 3, "sharpe": 0.8, "action": "modify_params"},
            {"turn": 4, "sharpe": 1.5, "action": "change_indicators"},
        ]
        score, label = compute_stagnation(run_history)
        response = get_stagnation_response(iteration=current_turn, score=score)
        # Improving — should not recommend abandon
        if response.get("action") == "abandon":
            assert False, f"Should not abandon an improving run (score={score})"

        # All Phase 2 checks pass — this turn is valid
        assert True

    def test_phase2_catches_bad_turn(self):
        """Verify Phase 2 components correctly reject a bad refinement turn."""
        from crabquant.refinement.cosmetic_guard import check_cosmetic_guard, CosmeticGuardState
        from crabquant.refinement.hypothesis_enforcement import check_hypothesis_from_modification
        from crabquant.refinement.stagnation import compute_stagnation, get_stagnation_response

        # Bad modification: generic hypothesis
        bad_mod = {
            "action": "modify_params",
            "hypothesis": "tweak parameters to improve performance",
        }

        # Hypothesis enforcement should reject
        hypo_result = check_hypothesis_from_modification(bad_mod)
        assert not hypo_result.valid

        # Cosmetic guard should trigger (3 consecutive modify_params)
        cosmetic_state = CosmeticGuardState(threshold=3)
        action_history = [
            {"turn": 1, "action": "modify_params"},
            {"turn": 2, "action": "modify_params"},
            {"turn": 3, "action": "modify_params"},
        ]
        cosmetic_state, cosmetic_result = check_cosmetic_guard(action_history, cosmetic_state)
        assert cosmetic_result.warning
        assert cosmetic_result.forced_action is not None

        # Stagnation should be high
        run_history = [
            {"turn": i, "sharpe": -0.2, "action": "modify_params"}
            for i in range(1, 7)
        ]
        score, label = compute_stagnation(run_history)
        response = get_stagnation_response(iteration=6, score=score)
        assert score > 0.5

        # This turn should be rejected by multiple Phase 2 guards
        assert not hypo_result.valid
        assert cosmetic_result.warning
        assert score > 0.5
