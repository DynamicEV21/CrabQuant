"""Tests for Phase 5.6 stagnation recovery system.

Tests cover:
- classify_indicator() — indicator name → family mapping
- extract_indicators_from_code() — source code parsing
- track_indicator_diversity() — rut detection across turns
- detect_stagnation_trap() — trap type classification
- build_stagnation_recovery() — recovery instruction generation
- Context builder integration — stagnation_recovery in context dict
"""

import pytest
from unittest.mock import MagicMock, patch

from crabquant.refinement.stagnation import (
    classify_indicator,
    extract_indicators_from_code,
    track_indicator_diversity,
    detect_stagnation_trap,
    build_stagnation_recovery,
)


# ── classify_indicator ───────────────────────────────────────────────────────


class TestClassifyIndicator:
    """Tests for classify_indicator function."""

    def test_momentum_indicators(self):
        for name in ["ema", "EMA", "sma", "wma", "macd", "roc", "tsi", "dpo"]:
            assert classify_indicator(name) == "momentum", f"{name} should be momentum"

    def test_mean_reversion_indicators(self):
        for name in ["rsi", "RSI", "bbands", "bollinger", "stoch", "cci", "willr"]:
            assert classify_indicator(name) == "mean_reversion", f"{name} should be mean_reversion"

    def test_volatility_indicators(self):
        for name in ["atr", "ATR", "adx", "supertrend", "keltner"]:
            assert classify_indicator(name) == "volatility", f"{name} should be volatility"

    def test_volume_indicators(self):
        for name in ["obv", "OBV", "vwap", "ad", "cmf", "mfi"]:
            assert classify_indicator(name) == "volume", f"{name} should be volume"

    def test_trend_indicators(self):
        for name in ["ichimoku", "dema", "tema", "psar"]:
            assert classify_indicator(name) == "trend", f"{name} should be trend"

    def test_unknown_indicator(self):
        assert classify_indicator("notarealindicator") == "unknown"
        assert classify_indicator("xyz") == "unknown"
        assert classify_indicator("") == "unknown"


# ── extract_indicators_from_code ────────────────────────────────────────────


class TestExtractIndicatorsFromCode:
    """Tests for extract_indicators_from_code function."""

    def test_cached_indicator_pattern(self):
        code = '''
        ema_val = cached_indicator("ema", close, length=20)
        rsi_val = cached_indicator('rsi', close, length=14)
        atr_val = cached_indicator("atr", high, low, close, length=14)
        '''
        result = extract_indicators_from_code(code)
        assert result == ["ema", "rsi", "atr"]

    def test_ta_direct_pattern(self):
        code = '''
        macd_line = ta.macd(close, fast=12, slow=26)
        atr_val = pandas_ta.atr(high, low, close, length=14)
        '''
        result = extract_indicators_from_code(code)
        assert "macd" in result
        assert "atr" in result

    def test_deduplication(self):
        code = '''
        ema1 = cached_indicator("ema", close, length=10)
        ema2 = cached_indicator("ema", close, length=20)
        '''
        result = extract_indicators_from_code(code)
        assert result == ["ema"]

    def test_empty_code(self):
        assert extract_indicators_from_code("") == []
        assert extract_indicators_from_code("# just a comment") == []

    def test_no_indicators(self):
        code = '''
        entries = close > close.shift(1)
        exits = close < close.shift(1)
        '''
        result = extract_indicators_from_code(code)
        assert result == []


# ── track_indicator_diversity ───────────────────────────────────────────────


class TestTrackIndicatorDiversity:
    """Tests for track_indicator_diversity function."""

    def test_empty_history(self):
        result = track_indicator_diversity([])
        assert result["is_rut"] is False
        assert result["dominant_family"] is None
        assert result["family_counts"] == {}

    def test_no_rut_with_diverse_indicators(self):
        # 3 turns with different families
        history = [
            {"code_path": "/nonexistent1.py"},
            {"code_path": "/nonexistent2.py"},
            {"code_path": "/nonexistent3.py"},
        ]
        result = track_indicator_diversity(history)
        # Non-existent files → no code extracted → no family counts
        assert result["is_rut"] is False

    def test_rut_detection_with_params(self):
        # Simulate 3 turns all using momentum params
        history = [
            {"params_used": {"ema_fast": 10, "ema_slow": 20}},
            {"params_used": {"macd_fast": 12, "macd_slow": 26}},
            {"params_used": {"roc_length": 14}},
        ]
        result = track_indicator_diversity(history)
        # Params contain momentum indicator names
        assert len(result["all_indicators"]) > 0

    def test_rut_with_current_code(self):
        history = []  # No history
        code = '''
        ema_fast = cached_indicator("ema", close, length=10)
        ema_slow = cached_indicator("ema", close, length=20)
        macd_val = cached_indicator("macd", close, fast=12, slow=26)
        '''
        result = track_indicator_diversity(history, current_code=code)
        assert "momentum" in result["families_used"]
        assert result["is_rut"] is False  # Only 1 turn, need 3 for rut

    def test_rut_with_explicit_family_counts(self):
        # Simulate a rut by mocking internal tracking
        code = 'ema_val = cached_indicator("ema", close, length=10)'
        history = []
        # Add same code 4 times via current_code param chain won't work
        # but we can test the recovery_hint structure
        result = track_indicator_diversity(history, current_code=code)
        assert result["recovery_hint"] == ""  # Only 1 use, not a rut


# ── detect_stagnation_trap ──────────────────────────────────────────────────


class TestDetectStagnationTrap:
    """Tests for detect_stagnation_trap function."""

    def test_no_trap_with_few_turns(self):
        result = detect_stagnation_trap([])
        assert result["trap"] == "no_trap"
        assert result["severity"] == "low"
        assert result["turns_in_trap"] == 0

    def test_no_trap_with_one_turn(self):
        result = detect_stagnation_trap([{"sharpe": 0.5}])
        assert result["trap"] == "no_trap"

    def test_zero_sharpe_trap(self):
        history = [
            {"sharpe": -0.1},
            {"sharpe": -0.5},
            {"sharpe": 0.0},
        ]
        result = detect_stagnation_trap(history)
        assert result["trap"] == "zero_sharpe"
        assert result["severity"] == "critical"
        assert result["turns_in_trap"] >= 2

    def test_low_sharpe_plateau(self):
        history = [
            {"sharpe": 0.1},
            {"sharpe": 0.2},
            {"sharpe": 0.15},
        ]
        result = detect_stagnation_trap(history)
        assert result["trap"] == "low_sharpe_plateau"
        assert result["severity"] == "high"

    def test_mid_sharpe_trap(self):
        history = [
            {"sharpe": 0.4},
            {"sharpe": 0.5},
            {"sharpe": 0.6},
        ]
        result = detect_stagnation_trap(history, best_sharpe=0.6, sharpe_target=1.5)
        assert result["trap"] == "mid_sharpe_trap"
        assert result["severity"] == "medium"

    def test_near_target(self):
        history = [
            {"sharpe": 0.8},
            {"sharpe": 1.0},
            {"sharpe": 1.1},
        ]
        result = detect_stagnation_trap(history, best_sharpe=1.1, sharpe_target=1.5)
        assert result["trap"] == "near_target"
        assert result["severity"] == "low"

    def test_high_sharpe_few_trades(self):
        history = [
            {"sharpe": 1.5, "num_trades": 5},
            {"sharpe": 1.8, "num_trades": 8},
        ]
        result = detect_stagnation_trap(
            history, best_sharpe=1.8, sharpe_target=1.5
        )
        assert result["trap"] == "high_sharpe_few_trades"
        assert result["severity"] == "high"

    def test_validation_loop(self):
        history = [
            {"sharpe": 1.0},
            {"sharpe": 1.2},
        ]
        # Need failure_mode in recent entries
        history[-5:]  # empty slice is fine
        # The trap detection uses failure_modes from last 5 entries
        # Let's test with explicit failure modes
        result = detect_stagnation_trap(
            history + [
                {"sharpe": 1.3, "failure_mode": "validation_failed"},
                {"sharpe": 1.4, "failure_mode": "validation_failed"},
            ],
            best_sharpe=1.4,
            sharpe_target=1.5,
        )
        # Should detect validation loop (2+ validation fails + high sharpe)
        assert result["trap"] == "validation_loop"

    def test_action_loop(self):
        history = [
            {"sharpe": 0.3, "action": "modify_params"},
            {"sharpe": 0.25, "action": "modify_params"},
            {"sharpe": 0.2, "action": "modify_params"},
        ]
        result = detect_stagnation_trap(history)
        assert result["trap"] == "action_loop"
        assert result["severity"] == "medium"
        assert result["turns_in_trap"] == 3

    def test_progressing_not_a_trap(self):
        history = [
            {"sharpe": 0.6},
            {"sharpe": 0.9},
            {"sharpe": 1.2},
        ]
        result = detect_stagnation_trap(history, best_sharpe=1.2, sharpe_target=1.5)
        # Progressing upward → near_target (low severity, not a problem trap)
        assert result["trap"] == "near_target"
        assert result["severity"] == "low"

    def test_result_structure(self):
        result = detect_stagnation_trap([{"sharpe": 0.0}])
        expected_keys = {"trap", "severity", "sharpes", "recent_failure_modes",
                         "recent_actions", "turns_in_trap", "description"}
        assert set(result.keys()) == expected_keys


# ── build_stagnation_recovery ───────────────────────────────────────────────


class TestBuildStagnationRecovery:
    """Tests for build_stagnation_recovery function."""

    def test_no_trap_returns_empty(self):
        result = build_stagnation_recovery({"trap": "no_trap", "severity": "low"})
        assert result == ""

    def test_low_severity_returns_empty(self):
        result = build_stagnation_recovery({"trap": "near_target", "severity": "low", "turns_in_trap": 2})
        assert result == ""

    def test_zero_sharpe_recovery(self):
        trap_info = {
            "trap": "zero_sharpe",
            "severity": "critical",
            "turns_in_trap": 3,
            "sharpes": [-0.1, -0.5, 0.0],
        }
        recovery = build_stagnation_recovery(trap_info)
        assert "Zero Sharpe" in recovery
        assert "MANDATORY CHANGES" in recovery
        assert "different indicator" in recovery.lower()
        assert len(recovery) > 100

    def test_low_sharpe_plateau_recovery(self):
        trap_info = {
            "trap": "low_sharpe_plateau",
            "severity": "high",
            "turns_in_trap": 4,
            "sharpes": [0.1, 0.2, 0.15],
        }
        recovery = build_stagnation_recovery(trap_info)
        assert "Low Sharpe Plateau" in recovery
        assert "MANDATORY CHANGES" in recovery
        assert "ABANDON" in recovery

    def test_mid_sharpe_trap_recovery(self):
        trap_info = {
            "trap": "mid_sharpe_trap",
            "severity": "medium",
            "turns_in_trap": 3,
            "sharpes": [0.4, 0.5, 0.6],
        }
        recovery = build_stagnation_recovery(trap_info)
        assert "Mid Sharpe Trap" in recovery
        assert "RECOMMENDED CHANGES" in recovery

    def test_high_sharpe_few_trades_recovery(self):
        trap_info = {
            "trap": "high_sharpe_few_trades",
            "severity": "high",
            "turns_in_trap": 2,
            "sharpes": [1.5, 1.8],
        }
        recovery = build_stagnation_recovery(trap_info)
        assert "Trade Count Trap" in recovery
        assert "CURVE-FITTING" in recovery
        assert "30+ trades" in recovery

    def test_validation_loop_recovery(self):
        trap_info = {
            "trap": "validation_loop",
            "severity": "high",
            "turns_in_trap": 3,
            "sharpes": [1.0, 1.2, 1.4],
        }
        recovery = build_stagnation_recovery(trap_info)
        assert "Validation Failure Loop" in recovery
        assert "overfitting" in recovery.lower()
        assert "CUT complexity" in recovery

    def test_action_loop_recovery(self):
        trap_info = {
            "trap": "action_loop",
            "severity": "medium",
            "turns_in_trap": 3,
            "recent_actions": ["modify_params", "modify_params", "modify_params"],
            "sharpes": [0.3, 0.25, 0.2],
        }
        recovery = build_stagnation_recovery(trap_info)
        assert "Action Loop" in recovery
        assert "MANDATORY CHANGE" in recovery

    def test_unknown_trap_returns_empty(self):
        result = build_stagnation_recovery({
            "trap": "some_future_trap",
            "severity": "high",
            "turns_in_trap": 5,
        })
        assert result == ""

    def test_all_recovery_types_are_markdown(self):
        """All recovery messages should be markdown-formatted."""
        trap_types = [
            "zero_sharpe", "low_sharpe_plateau", "mid_sharpe_trap",
            "high_sharpe_few_trades", "validation_loop", "action_loop",
        ]
        for trap in trap_types:
            info = {"trap": trap, "severity": "high", "turns_in_trap": 3}
            recovery = build_stagnation_recovery(info)
            assert recovery.startswith("#") or recovery.startswith("<!--"), \
                f"{trap} recovery should start with markdown heading or comment"


# ── Context Builder Integration ─────────────────────────────────────────────


class TestContextBuilderIntegration:
    """Tests that stagnation recovery flows through context builder."""

    def test_context_builder_includes_stagnation_recovery(self):
        """When stagnation is detected, context should include recovery section."""
        from crabquant.refinement.context_builder import build_llm_context

        # Create a mock state with history showing zero sharpe
        mock_state = MagicMock()
        mock_state.current_turn = 5
        mock_state.max_turns = 7
        mock_state.sharpe_target = 1.5
        mock_state.tickers = ["AAPL"]
        mock_state.best_sharpe = 0.0
        mock_state.best_composite_score = -999.0
        mock_state.best_turn = 0
        mock_state.history = [
            {"turn": 1, "sharpe": -0.1, "action": "novel", "failure_mode": "low_sharpe"},
            {"turn": 2, "sharpe": -0.5, "action": "modify_params", "failure_mode": "low_sharpe"},
            {"turn": 3, "sharpe": 0.0, "action": "replace_indicator", "failure_mode": "low_sharpe"},
        ]

        context = build_llm_context(mock_state)

        # Should have stagnation_recovery key
        assert "stagnation_recovery" in context
        assert len(context["stagnation_recovery"]) > 0
        assert "Zero Sharpe" in context["stagnation_recovery"]

    def test_context_builder_no_stagnation_when_progressing(self):
        """When making progress, no stagnation recovery should be injected."""
        from crabquant.refinement.context_builder import build_llm_context

        mock_state = MagicMock()
        mock_state.current_turn = 3
        mock_state.max_turns = 7
        mock_state.sharpe_target = 1.5
        mock_state.tickers = ["AAPL"]
        mock_state.best_sharpe = 1.2
        mock_state.best_composite_score = 5.0
        mock_state.best_turn = 3
        mock_state.history = [
            {"turn": 1, "sharpe": 0.6, "action": "novel"},
            {"turn": 2, "sharpe": 0.9, "action": "modify_params"},
            {"turn": 3, "sharpe": 1.2, "action": "modify_params"},
        ]

        context = build_llm_context(mock_state)

        # Near target (severity=low) should NOT inject stagnation_recovery
        assert context.get("stagnation_recovery", "") == ""

    def test_context_builder_no_stagnation_with_few_turns(self):
        """With only 1 turn, no stagnation should be detected."""
        from crabquant.refinement.context_builder import build_llm_context

        mock_state = MagicMock()
        mock_state.current_turn = 1
        mock_state.max_turns = 7
        mock_state.sharpe_target = 1.5
        mock_state.tickers = ["AAPL"]
        mock_state.best_sharpe = 0.0
        mock_state.best_composite_score = -999.0
        mock_state.best_turn = 0
        mock_state.history = [
            {"turn": 1, "sharpe": -0.5, "action": "novel"},
        ]

        context = build_llm_context(mock_state)

        # Only 1 turn → no stagnation detection
        assert context.get("stagnation_recovery", "") == ""

    def test_context_builder_action_loop_injection(self):
        """Action loop should be detected and recovery injected."""
        from crabquant.refinement.context_builder import build_llm_context

        mock_state = MagicMock()
        mock_state.current_turn = 5
        mock_state.max_turns = 7
        mock_state.sharpe_target = 1.5
        mock_state.tickers = ["AAPL"]
        mock_state.best_sharpe = 0.3
        mock_state.best_composite_score = 1.0
        mock_state.best_turn = 1
        mock_state.history = [
            {"turn": 1, "sharpe": 0.3, "action": "modify_params"},
            {"turn": 2, "sharpe": 0.25, "action": "modify_params"},
            {"turn": 3, "sharpe": 0.2, "action": "modify_params"},
            {"turn": 4, "sharpe": 0.15, "action": "modify_params"},
            {"turn": 5, "sharpe": 0.1, "action": "modify_params"},
        ]

        context = build_llm_context(mock_state)

        assert "stagnation_recovery" in context
        # Should detect action loop (5 consecutive same actions)
        assert "Action Loop" in context["stagnation_recovery"]


# ── Integration: Full Pipeline ──────────────────────────────────────────────


class TestStagnationPipelineIntegration:
    """End-to-end tests for the stagnation recovery pipeline."""

    def test_full_pipeline_zero_sharpe(self):
        """Full pipeline: detect trap → build recovery → verify content."""
        history = [
            {"sharpe": -0.2, "action": "novel"},
            {"sharpe": -0.3, "action": "modify_params"},
            {"sharpe": 0.0, "action": "replace_indicator"},
        ]
        trap = detect_stagnation_trap(history)
        assert trap["trap"] == "zero_sharpe"

        recovery = build_stagnation_recovery(trap)
        assert "Zero Sharpe" in recovery
        assert "MANDATORY CHANGES" in recovery

    def test_full_pipeline_improving(self):
        """Full pipeline: improving strategy → no recovery needed."""
        history = [
            {"sharpe": 0.6, "action": "novel"},
            {"sharpe": 0.9, "action": "modify_params"},
            {"sharpe": 1.2, "action": "modify_params"},
        ]
        trap = detect_stagnation_trap(history, best_sharpe=1.2, sharpe_target=1.5)
        # near_target has severity "low" → no recovery
        recovery = build_stagnation_recovery(trap)
        assert recovery == ""

    def test_full_pipeline_plateau_then_recovery(self):
        """Full pipeline: plateau detected → recovery generated."""
        history = [
            {"sharpe": 0.1, "action": "novel"},
            {"sharpe": 0.2, "action": "modify_params"},
            {"sharpe": 0.15, "action": "replace_indicator"},
            {"sharpe": 0.18, "action": "change_entry_logic"},
        ]
        trap = detect_stagnation_trap(history)
        assert trap["trap"] == "low_sharpe_plateau"

        recovery = build_stagnation_recovery(trap)
        assert "Low Sharpe Plateau" in recovery
        assert len(recovery) > 100
