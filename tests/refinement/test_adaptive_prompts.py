"""Tests for crabquant.refinement.adaptive_prompts — Phase 6 adaptive invention prompts.

Covers:
- build_adaptive_invention_prompt (regime hints, portfolio gaps, control group, rate)
- get_regime_indicators (all regimes + unknown)
- get_portfolio_gap_nudges (low/good coverage)
- format_adaptive_section
- Token/char budget enforcement
- Graceful handling of empty stats
"""

import pytest

from crabquant.refinement.adaptive_prompts import (
    MAX_ADAPTIVE_CHARS,
    build_adaptive_invention_prompt,
    format_adaptive_section,
    get_portfolio_gap_nudges,
    get_regime_indicators,
)

BASE_PROMPT = "Invent a new trading strategy. Output JSON."


# ─── get_regime_indicators ────────────────────────────────────────────────────


class TestGetRegimeIndicators:
    """Verify regime → indicator family mapping."""

    def test_high_volatility_suggests_volatility_indicators(self):
        indicators = get_regime_indicators("HIGH_VOLATILITY")
        text = " ".join(indicators).lower()
        assert "atr" in text
        assert "bollinger" in text
        assert "vix" in text

    def test_low_volatility_suggests_trend_indicators(self):
        indicators = get_regime_indicators("LOW_VOLATILITY")
        text = " ".join(indicators).lower()
        assert "ema" in text
        assert "macd" in text
        assert "momentum" in text

    def test_trending_suggests_trend_indicators(self):
        indicators = get_regime_indicators("TRENDING")
        text = " ".join(indicators).lower()
        assert "ema" in text
        assert "moving average" in text
        assert "adx" in text

    def test_ranging_suggests_range_indicators(self):
        indicators = get_regime_indicators("RANGING")
        text = " ".join(indicators).lower()
        assert "rsi" in text
        assert "bollinger" in text
        assert "stochastic" in text

    def test_unknown_regime_no_indicators(self):
        indicators = get_regime_indicators("UNKNOWN")
        assert indicators == []

    def test_unrecognized_regime_no_indicators(self):
        indicators = get_regime_indicators("FOOBAR")
        assert indicators == []


# ─── get_portfolio_gap_nudges ────────────────────────────────────────────────


class TestPortfolioGapNudges:
    """Verify portfolio gap nudge generation."""

    def test_portfolio_gap_nudges_low_coverage(self):
        gaps = {
            "momentum": 0.10,
            "mean_reversion": 0.05,
            "breakout": 0.80,
            "volatility": 0.50,
        }
        nudges = get_portfolio_gap_nudges(gaps)
        assert len(nudges) == 2
        # Lowest coverage archetypes should be nudged first.
        text = " ".join(nudges)
        assert "mean_reversion" in text
        assert "momentum" in text
        assert "breakout" not in text

    def test_portfolio_gap_nudges_good_coverage(self):
        gaps = {
            "momentum": 0.90,
            "mean_reversion": 0.85,
        }
        nudges = get_portfolio_gap_nudges(gaps)
        assert nudges == []

    def test_portfolio_gap_nudges_max_two(self):
        gaps = {
            "momentum": 0.05,
            "mean_reversion": 0.10,
            "breakout": 0.15,
            "volatility": 0.20,
        }
        nudges = get_portfolio_gap_nudges(gaps)
        assert len(nudges) <= 2

    def test_portfolio_gap_nudges_empty_dict(self):
        nudges = get_portfolio_gap_nudges({})
        assert nudges == []


# ─── format_adaptive_section ─────────────────────────────────────────────────


class TestFormatAdaptiveSection:
    """Verify adaptive section formatting."""

    def test_format_adaptive_section(self):
        result = format_adaptive_section(
            regime="HIGH_VOLATILITY",
            indicators=["ATR", "Bollinger Bands"],
            gap_nudges=["Consider a momentum strategy (coverage: 10%)."],
            action_recommendations=["novel (best: 30% for low_sharpe)"],
        )
        assert "### Adaptive Context Hints" in result
        assert "HIGH_VOLATILITY" in result
        assert "ATR" in result
        assert "Bollinger Bands" in result
        assert "momentum" in result
        assert "novel" in result

    def test_format_empty_inputs(self):
        result = format_adaptive_section(
            regime="UNKNOWN",
            indicators=[],
            gap_nudges=[],
            action_recommendations=[],
        )
        assert "### Adaptive Context Hints" in result
        assert "UNKNOWN" in result


# ─── build_adaptive_invention_prompt ─────────────────────────────────────────


class TestBuildAdaptiveInventionPrompt:
    """Verify the main prompt-building function."""

    def test_adaptive_prompt_includes_regime(self):
        """When adaptation is applied, the result should mention the regime."""
        # Use rate=1.0 to guarantee adaptation.
        result = build_adaptive_invention_prompt(
            BASE_PROMPT,
            regime="HIGH_VOLATILITY",
            portfolio_gaps={"momentum": 0.1},
            adaptation_rate=1.0,
        )
        assert "HIGH_VOLATILITY" in result
        assert result.startswith(BASE_PROMPT)

    def test_adaptive_prompt_includes_portfolio_gaps(self):
        """Portfolio gaps below threshold should produce nudges."""
        result = build_adaptive_invention_prompt(
            BASE_PROMPT,
            regime="TRENDING",
            portfolio_gaps={"breakout": 0.05, "momentum": 0.90},
            adaptation_rate=1.0,
        )
        assert "breakout" in result
        assert "momentum" not in result  # 0.90 > 0.3 threshold

    def test_control_group_returns_base_prompt(self):
        """With adaptation_rate=0.0, always return base_prompt."""
        for _ in range(50):
            result = build_adaptive_invention_prompt(
                BASE_PROMPT,
                regime="HIGH_VOLATILITY",
                portfolio_gaps={"momentum": 0.0},
                adaptation_rate=0.0,
            )
            assert result == BASE_PROMPT

    def test_adaptation_rate_respected(self):
        """With rate=1.0, always get adaptation; with rate=0.0, never."""
        gaps = {"momentum": 0.0}

        # rate=1.0 → always adapted
        for _ in range(20):
            result = build_adaptive_invention_prompt(
                BASE_PROMPT, "HIGH_VOLATILITY", gaps, adaptation_rate=1.0
            )
            assert result != BASE_PROMPT
            assert "Adaptive" in result

        # rate=0.0 → never adapted
        for _ in range(20):
            result = build_adaptive_invention_prompt(
                BASE_PROMPT, "HIGH_VOLATILITY", gaps, adaptation_rate=0.0
            )
            assert result == BASE_PROMPT

    def test_token_limit_not_exceeded(self):
        """Adaptive additions must not exceed MAX_ADAPTIVE_CHARS."""
        # Create extreme inputs that would produce a very long section.
        huge_gaps = {f"archetype_{i}": 0.01 for i in range(100)}
        action_stats = {
            "by_failure_mode": {
                f"failure_{i}": {
                    "ranked_actions": [
                        (f"action_{j}", 0.5 + j * 0.01, 10, 5)
                        for j in range(50)
                    ],
                }
                for i in range(20)
            }
        }
        result = build_adaptive_invention_prompt(
            BASE_PROMPT,
            regime="HIGH_VOLATILITY",
            portfolio_gaps=huge_gaps,
            action_stats=action_stats,
            adaptation_rate=1.0,
        )
        # The addition is everything after the base prompt.
        addition = result[len(BASE_PROMPT.rstrip()):]
        assert len(addition) <= MAX_ADAPTIVE_CHARS + 100  # small margin for whitespace

    def test_empty_stats_graceful(self):
        """None or empty action_stats should not cause errors."""
        result = build_adaptive_invention_prompt(
            BASE_PROMPT,
            regime="RANGING",
            portfolio_gaps={"mean_reversion": 0.1},
            action_stats=None,
            adaptation_rate=1.0,
        )
        assert "RANGING" in result
        assert "mean_reversion" in result
        assert "Historically effective" not in result

        # Also test with empty dict
        result2 = build_adaptive_invention_prompt(
            BASE_PROMPT,
            regime="RANGING",
            portfolio_gaps={"mean_reversion": 0.1},
            action_stats={},
            adaptation_rate=1.0,
        )
        assert "RANGING" in result2

    def test_action_stats_with_recommendations(self):
        """Valid action_stats should produce recommendation lines."""
        action_stats = {
            "by_failure_mode": {
                "low_sharpe": {
                    "ranked_actions": [
                        ("novel", 0.50, 10, 5),
                        ("replace_indicator", 0.20, 5, 1),
                    ],
                },
            }
        }
        result = build_adaptive_invention_prompt(
            BASE_PROMPT,
            regime="TRENDING",
            portfolio_gaps={},
            action_stats=action_stats,
            adaptation_rate=1.0,
        )
        assert "Historically effective actions" in result
        assert "novel" in result
