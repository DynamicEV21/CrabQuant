"""Tests for strategy archetype templates (Phase 5.6)."""

from __future__ import annotations

import pytest

from crabquant.refinement.archetypes import (
    ARCHETYPE_REGISTRY,
    Archetype,
    format_archetype_for_prompt,
    get_archetype,
    list_archetypes,
)


# ── Registry Tests ──────────────────────────────────────────────────────────


class TestArchetypeRegistry:
    """Verify the archetype registry contains all expected archetypes."""

    def test_all_four_archetypes_exist(self) -> None:
        """Registry must have the original 4 archetypes."""
        expected = {"mean_reversion", "momentum", "breakout", "volatility"}
        actual = set(ARCHETYPE_REGISTRY.keys())
        assert expected <= actual, f"Missing archetypes: {expected - actual}"

    def test_extended_archetypes_exist(self) -> None:
        """Registry should also have the extended archetypes."""
        extended = {"statistical_arb", "multi_signal_ensemble", "volatility_breakout"}
        actual = set(ARCHETYPE_REGISTRY.keys())
        assert extended <= actual, f"Missing extended archetypes: {extended - actual}"

    def test_list_archetypes_returns_all(self) -> None:
        """list_archetypes() should return at least the 7 archetypes."""
        names = list_archetypes()
        assert len(names) >= 7
        for name in ["mean_reversion", "momentum", "breakout", "volatility",
                      "statistical_arb", "multi_signal_ensemble", "volatility_breakout"]:
            assert name in names


# ── Per-Archetype Validation ───────────────────────────────────────────────


class TestArchetypeStructure:
    """Every archetype must have all required fields with valid content."""

    @pytest.fixture(params=[
        "mean_reversion", "momentum", "breakout", "volatility",
        "statistical_arb", "multi_signal_ensemble", "volatility_breakout",
    ])
    def archetype(self, request: pytest.FixtureRequest) -> Archetype:
        return ARCHETYPE_REGISTRY[request.param]

    def test_has_name(self, archetype: Archetype) -> None:
        assert isinstance(archetype["name"], str)
        assert len(archetype["name"]) > 3

    def test_has_description(self, archetype: Archetype) -> None:
        assert isinstance(archetype["description"], str)
        assert len(archetype["description"]) > 20

    def test_has_skeleton_code(self, archetype: Archetype) -> None:
        code = archetype["skeleton_code"]
        assert isinstance(code, str)
        assert len(code) > 100

    def test_skeleton_has_generate_signals(self, archetype: Archetype) -> None:
        """Skeleton code MUST define generate_signals function."""
        assert "def generate_signals" in archetype["skeleton_code"]

    def test_skeleton_has_default_params(self, archetype: Archetype) -> None:
        """Skeleton code MUST define DEFAULT_PARAMS."""
        assert "DEFAULT_PARAMS" in archetype["skeleton_code"]

    def test_skeleton_has_description(self, archetype: Archetype) -> None:
        """Skeleton code MUST define DESCRIPTION."""
        assert "DESCRIPTION" in archetype["skeleton_code"]

    def test_skeleton_uses_cached_indicator(self, archetype: Archetype) -> None:
        """Skeleton code should use cached_indicator for indicators."""
        assert "cached_indicator" in archetype["skeleton_code"]

    def test_skeleton_returns_entries_exits(self, archetype: Archetype) -> None:
        """Skeleton code must return (entries, exits) tuple."""
        assert "return entries, exits" in archetype["skeleton_code"]

    def test_skeleton_imports_pandas(self, archetype: Archetype) -> None:
        """Skeleton code must import pandas."""
        assert "import pandas" in archetype["skeleton_code"]

    def test_has_default_params_dict(self, archetype: Archetype) -> None:
        """default_params must be a non-empty dict."""
        params = archetype["default_params"]
        assert isinstance(params, dict)
        assert len(params) >= 2

    def test_has_typical_indicators(self, archetype: Archetype) -> None:
        """typical_indicators must be a non-empty list of strings."""
        indicators = archetype["typical_indicators"]
        assert isinstance(indicators, list)
        assert len(indicators) >= 2
        for ind in indicators:
            assert isinstance(ind, str)

    def test_has_trade_frequency_expectation(self, archetype: Archetype) -> None:
        """trade_frequency_expectation must describe expected trade count."""
        freq = archetype["trade_frequency_expectation"]
        assert isinstance(freq, str)
        assert len(freq) > 10

    def test_has_regime_affinity(self, archetype: Archetype) -> None:
        """regime_affinity must be a valid regime type."""
        affinity = archetype["regime_affinity"]
        assert affinity in {"trending", "ranging", "volatile", "any"}

    def test_has_anti_patterns(self, archetype: Archetype) -> None:
        """anti_patterns must be a non-empty list of strings."""
        patterns = archetype["anti_patterns"]
        assert isinstance(patterns, list)
        assert len(patterns) >= 1
        for p in patterns:
            assert isinstance(p, str)
            assert len(p) > 20


# ── Archetype-Specific Tests ───────────────────────────────────────────────


class TestMeanReversionArchetype:
    """Mean reversion archetype specific tests."""

    def test_uses_rsi(self) -> None:
        archetype = ARCHETYPE_REGISTRY["mean_reversion"]
        assert "rsi" in archetype["typical_indicators"]
        assert "rsi" in archetype["skeleton_code"].lower()

    def test_uses_bbands(self) -> None:
        archetype = ARCHETYPE_REGISTRY["mean_reversion"]
        assert "bbands" in archetype["typical_indicators"]

    def test_regime_affinity_ranging(self) -> None:
        archetype = ARCHETYPE_REGISTRY["mean_reversion"]
        assert archetype["regime_affinity"] == "ranging"

    def test_moderate_trade_frequency(self) -> None:
        archetype = ARCHETYPE_REGISTRY["mean_reversion"]
        assert "30" in archetype["trade_frequency_expectation"]


class TestMomentumArchetype:
    """Momentum archetype specific tests."""

    def test_uses_ema(self) -> None:
        archetype = ARCHETYPE_REGISTRY["momentum"]
        assert "ema" in archetype["typical_indicators"]
        assert "ema" in archetype["skeleton_code"].lower()

    def test_uses_roc(self) -> None:
        archetype = ARCHETYPE_REGISTRY["momentum"]
        assert "roc" in archetype["typical_indicators"]

    def test_regime_affinity_trending(self) -> None:
        archetype = ARCHETYPE_REGISTRY["momentum"]
        assert archetype["regime_affinity"] == "trending"


class TestBreakoutArchetype:
    """Breakout archetype specific tests."""

    def test_uses_atr(self) -> None:
        archetype = ARCHETYPE_REGISTRY["breakout"]
        assert "atr" in archetype["typical_indicators"]
        assert "atr" in archetype["skeleton_code"].lower()

    def test_uses_range_breakout_logic(self) -> None:
        archetype = ARCHETYPE_REGISTRY["breakout"]
        code = archetype["skeleton_code"].lower()
        assert "highest" in code or "rolling" in code or "donchian" in code

    def test_regime_affinity_volatile(self) -> None:
        archetype = ARCHETYPE_REGISTRY["breakout"]
        assert archetype["regime_affinity"] == "volatile"


class TestVolatilityArchetype:
    """Volatility archetype specific tests."""

    def test_uses_atr_ratio(self) -> None:
        archetype = ARCHETYPE_REGISTRY["volatility"]
        assert "atr" in archetype["typical_indicators"]
        code = archetype["skeleton_code"].lower()
        assert "atr_ratio" in code or "atr_short" in code

    def test_regime_affinity_volatile(self) -> None:
        archetype = ARCHETYPE_REGISTRY["volatility"]
        assert archetype["regime_affinity"] == "volatile"


# ── get_archetype() Tests ──────────────────────────────────────────────────


class TestGetArchetype:
    """Test the get_archetype lookup function."""

    def test_exact_name(self) -> None:
        result = get_archetype("mean_reversion")
        assert result is not None
        assert result["name"] == "Mean Reversion"

    def test_case_insensitive(self) -> None:
        result = get_archetype("MOMENTUM")
        assert result is not None
        assert result["name"] == "Momentum / Trend Following"

    def test_whitespace_stripped(self) -> None:
        result = get_archetype("  breakout  ")
        assert result is not None
        assert result["name"] == "Breakout / Range Expansion"

    def test_unknown_returns_none(self) -> None:
        result = get_archetype("nonexistent_strategy")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = get_archetype("")
        assert result is None


# ── format_archetype_for_prompt() Tests ────────────────────────────────────


class TestFormatArchetypeForPrompt:
    """Test the prompt formatting function."""

    def test_includes_name(self) -> None:
        archetype = ARCHETYPE_REGISTRY["mean_reversion"]
        result = format_archetype_for_prompt(archetype)
        assert "Mean Reversion" in result

    def test_includes_description(self) -> None:
        archetype = ARCHETYPE_REGISTRY["momentum"]
        result = format_archetype_for_prompt(archetype)
        assert archetype["description"] in result

    def test_includes_skeleton_code(self) -> None:
        archetype = ARCHETYPE_REGISTRY["breakout"]
        result = format_archetype_for_prompt(archetype)
        assert "```python" in result
        assert "generate_signals" in result

    def test_includes_regime_affinity(self) -> None:
        archetype = ARCHETYPE_REGISTRY["volatility"]
        result = format_archetype_for_prompt(archetype)
        assert "volatile" in result

    def test_includes_trade_frequency(self) -> None:
        archetype = ARCHETYPE_REGISTRY["mean_reversion"]
        result = format_archetype_for_prompt(archetype)
        assert archetype["trade_frequency_expectation"] in result

    def test_includes_typical_indicators(self) -> None:
        archetype = ARCHETYPE_REGISTRY["momentum"]
        result = format_archetype_for_prompt(archetype)
        for ind in archetype["typical_indicators"]:
            assert ind in result

    def test_includes_anti_patterns(self) -> None:
        archetype = ARCHETYPE_REGISTRY["mean_reversion"]
        result = format_archetype_for_prompt(archetype)
        assert "Common mistakes" in result or "AVOID" in result
        for ap in archetype["anti_patterns"]:
            assert ap in result

    def test_code_block_properly_closed(self) -> None:
        archetype = ARCHETYPE_REGISTRY["breakout"]
        result = format_archetype_for_prompt(archetype)
        # Count ``` occurrences — should be even (open + close)
        count = result.count("```")
        assert count % 2 == 0, "Code blocks not properly closed"


# ── Prompt Integration Tests ───────────────────────────────────────────────


class TestArchetypePromptIntegration:
    """Test that archetypes are properly wired into the prompt system."""

    def test_turn1_prompt_injects_archetype(self) -> None:
        """When mandate has strategy_archetype, build_turn1_prompt should inject template."""
        from crabquant.refinement.prompts import build_turn1_prompt

        mandate = {
            "name": "test_mandate",
            "strategy_archetype": "mean_reversion",
            "sharpe_target": 1.5,
            "tickers": ["SPY"],
            "period": "2y",
        }
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=5)

        # The archetype section should be present in the prompt
        assert "Strategy Archetype Template" in prompt
        assert "Mean Reversion" in prompt
        assert "generate_signals" in prompt
        assert "RSI" in prompt or "rsi" in prompt

    def test_turn1_prompt_no_archetype_when_any(self) -> None:
        """When archetype is 'any', no template should be injected."""
        from crabquant.refinement.prompts import build_turn1_prompt

        mandate = {
            "name": "test_mandate",
            "strategy_archetype": "any",
            "sharpe_target": 1.5,
            "tickers": ["SPY"],
            "period": "2y",
        }
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=5)

        # No archetype section
        assert "Strategy Archetype Template" not in prompt

    def test_turn1_prompt_no_archetype_when_missing(self) -> None:
        """When archetype is not in mandate, no template should be injected."""
        from crabquant.refinement.prompts import build_turn1_prompt

        mandate = {
            "name": "test_mandate",
            "sharpe_target": 1.5,
            "tickers": ["SPY"],
            "period": "2y",
        }
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=5)

        assert "Strategy Archetype Template" not in prompt

    def test_turn1_prompt_unknown_archetype_falls_back(self) -> None:
        """Unknown archetype name should not inject any template."""
        from crabquant.refinement.prompts import build_turn1_prompt

        mandate = {
            "name": "test_mandate",
            "strategy_archetype": "nonexistent_type",
            "sharpe_target": 1.5,
            "tickers": ["SPY"],
            "period": "2y",
        }
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=5)

        assert "Strategy Archetype Template" not in prompt

    def test_turn1_prompt_momentum_archetype(self) -> None:
        """Momentum archetype should inject EMA crossover template."""
        from crabquant.refinement.prompts import build_turn1_prompt

        mandate = {
            "name": "test_mandate",
            "strategy_archetype": "momentum",
            "sharpe_target": 1.5,
            "tickers": ["SPY"],
            "period": "2y",
        }
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=5)

        assert "Strategy Archetype Template" in prompt
        assert "Momentum" in prompt

    def test_turn1_prompt_breakout_archetype(self) -> None:
        """Breakout archetype should inject ATR-based template."""
        from crabquant.refinement.prompts import build_turn1_prompt

        mandate = {
            "name": "test_mandate",
            "strategy_archetype": "breakout",
            "sharpe_target": 1.5,
            "tickers": ["SPY"],
            "period": "2y",
        }
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=5)

        assert "Strategy Archetype Template" in prompt
        assert "Breakout" in prompt
        assert "atr" in prompt.lower()

    def test_turn1_prompt_volatility_archetype(self) -> None:
        """Volatility archetype should inject ATR ratio template."""
        from crabquant.refinement.prompts import build_turn1_prompt

        mandate = {
            "name": "test_mandate",
            "strategy_archetype": "volatility",
            "sharpe_target": 1.5,
            "tickers": ["SPY"],
            "period": "2y",
        }
        prompt = build_turn1_prompt(mandate=mandate, current_turn=1, max_turns=5)

        assert "Strategy Archetype Template" in prompt
        assert "Volatility" in prompt

    def test_explicit_archetype_section_overrides(self) -> None:
        """If archetype_section is explicitly provided, use it instead of lookup."""
        from crabquant.refinement.prompts import build_turn1_prompt

        mandate = {
            "name": "test_mandate",
            "strategy_archetype": "mean_reversion",
            "sharpe_target": 1.5,
            "tickers": ["SPY"],
            "period": "2y",
        }
        custom_section = "## Custom Archetype\nThis is a custom template."
        prompt = build_turn1_prompt(
            mandate=mandate,
            current_turn=1,
            max_turns=5,
            archetype_section=custom_section,
        )

        assert "Custom Archetype" in prompt
        assert "Mean Reversion" not in prompt or "Custom" in prompt
