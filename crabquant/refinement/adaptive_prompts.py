"""Adaptive Invention Prompts — Phase 6 regime-aware prompt augmentation.

Dynamically adjusts Turn 1 invention prompts based on:
- Current market regime (volatility, trend, range)
- Portfolio archetype coverage gaps
- Historically successful action types from action_effectiveness

A control-group mechanism (adaptation_rate < 1) ensures A/B testability.
"""

from __future__ import annotations

import random
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum character budget for adaptive additions (~500 tokens).
MAX_ADAPTIVE_CHARS = 2000

# Coverage threshold below which an archetype is considered underrepresented.
GAP_THRESHOLD = 0.3

# Maximum number of portfolio gap nudges to include.
MAX_GAP_NUDGES = 2

# ── Regime → Indicator Family Mapping ─────────────────────────────────────────

REGIME_INDICATORS: dict[str, list[str]] = {
    "HIGH_VOLATILITY": [
        "ATR (Average True Range)",
        "Bollinger Bands",
        "VIX-related indicators",
    ],
    "LOW_VOLATILITY": [
        "EMA (Exponential Moving Average)",
        "MACD",
        "Momentum indicators (ROC, CCI)",
    ],
    "TRENDING": [
        "EMA crossover",
        "Moving averages (SMA, EMA)",
        "ADX (Average Directional Index)",
    ],
    "RANGING": [
        "RSI (Relative Strength Index)",
        "Bollinger Bands",
        "Stochastic oscillator",
    ],
    "UNKNOWN": [],
}


# ── Public API ────────────────────────────────────────────────────────────────


def get_regime_indicators(regime: str) -> list[str]:
    """Return suggested indicator families for a market regime.

    Args:
        regime: One of HIGH_VOLATILITY, LOW_VOLATILITY, TRENDING, RANGING, UNKNOWN.

    Returns:
        List of indicator family names. Empty for UNKNOWN or unrecognized regimes.
    """
    return list(REGIME_INDICATORS.get(regime, []))


def get_portfolio_gap_nudges(portfolio_gaps: dict[str, float]) -> list[str]:
    """Generate nudges for underrepresented strategy archetypes.

    Args:
        portfolio_gaps: Dict mapping archetype name to coverage score (0–1).

    Returns:
        List of nudge strings (max 2) for archetypes with coverage < 0.3.
    """
    # Sort by coverage ascending so the worst-covered archetypes are nudged first.
    underrepresented = sorted(
        [
            (archetype, coverage)
            for archetype, coverage in portfolio_gaps.items()
            if coverage < GAP_THRESHOLD
        ],
        key=lambda x: x[1],
    )

    nudges = []
    for archetype, coverage in underrepresented[:MAX_GAP_NUDGES]:
        pct = coverage * 100
        nudges.append(
            f"Consider a {archetype} strategy (current portfolio coverage: {pct:.0f}%)."
        )
    return nudges


def format_adaptive_section(
    regime: str,
    indicators: list[str],
    gap_nudges: list[str],
    action_recommendations: list[str],
) -> str:
    """Format the adaptive additions section for prompt injection.

    Args:
        regime: Current market regime string.
        indicators: Suggested indicator families from get_regime_indicators().
        gap_nudges: Portfolio gap nudges from get_portfolio_gap_nudges().
        action_recommendations: Historically successful action suggestions.

    Returns:
        Formatted string ready to append to a base prompt.
    """
    lines: list[str] = []

    lines.append("### Adaptive Context Hints")
    lines.append(f"**Current regime:** {regime}")

    if indicators:
        indicator_list = ", ".join(indicators)
        lines.append(f"**Suggested indicators for this regime:** {indicator_list}")

    if gap_nudges:
        lines.append("**Portfolio diversification nudges:**")
        for nudge in gap_nudges:
            lines.append(f"- {nudge}")

    if action_recommendations:
        lines.append("**Historically effective actions:**")
        for rec in action_recommendations:
            lines.append(f"- {rec}")

    return "\n".join(lines)


def _extract_action_recommendations(
    action_stats: dict[str, Any] | None,
) -> list[str]:
    """Extract top historically successful action recommendations.

    Args:
        action_stats: Optional dict from analyze_action_effectiveness().

    Returns:
        List of recommendation strings (max 3). Empty if no data.
    """
    if not action_stats:
        return []

    by_failure_mode = action_stats.get("by_failure_mode", {})
    if not by_failure_mode:
        return []

    # Collect the top-ranked action across all failure modes.
    all_ranked: list[tuple[str, str, float]] = []
    for failure_mode, mode_data in by_failure_mode.items():
        ranked = mode_data.get("ranked_actions", [])
        for entry in ranked:
            # entry is (action_name, success_rate, total, successes)
            if len(entry) >= 2 and entry[1] > 0:
                all_ranked.append((failure_mode, entry[0], entry[1]))

    # Sort by success rate descending, deduplicate by action name.
    seen: set[str] = set()
    recommendations: list[str] = []
    for failure_mode, action, rate in sorted(
        all_ranked, key=lambda x: x[2], reverse=True
    ):
        if action not in seen:
            seen.add(action)
            recommendations.append(
                f"{action} (best success rate: {rate:.0%} for {failure_mode})"
            )
        if len(recommendations) >= 3:
            break

    return recommendations


def build_adaptive_invention_prompt(
    base_prompt: str,
    regime: str,
    portfolio_gaps: dict[str, float],
    action_stats: dict[str, Any] | None = None,
    adaptation_rate: float = 0.80,
) -> str:
    """Build an invention prompt adapted to current market conditions.

    With probability (1 - adaptation_rate), returns base_prompt unchanged
    (control group for A/B testing). Otherwise, appends regime-specific hints,
    portfolio gap nudges, and historically successful indicator suggestions.

    Args:
        base_prompt: Full Turn 1 invention prompt string.
        regime: Market regime string (HIGH_VOLATILITY, LOW_VOLATILITY, etc.).
        portfolio_gaps: Dict mapping archetype name to coverage score (0–1).
        action_stats: Optional dict from analyze_action_effectiveness().
        adaptation_rate: Probability of applying adaptive augmentation (0–1).

    Returns:
        Augmented prompt string, or base_prompt for the control group.
    """
    # Control group: return base prompt unchanged.
    if random.random() > adaptation_rate:
        return base_prompt

    indicators = get_regime_indicators(regime)
    gap_nudges = get_portfolio_gap_nudges(portfolio_gaps)
    action_recommendations = _extract_action_recommendations(action_stats)

    adaptive_section = format_adaptive_section(
        regime, indicators, gap_nudges, action_recommendations
    )

    # Enforce token/character budget.
    if len(adaptive_section) > MAX_ADAPTIVE_CHARS:
        adaptive_section = adaptive_section[:MAX_ADAPTIVE_CHARS].rsplit("\n", 1)[0]

    if not adaptive_section.strip():
        return base_prompt

    return base_prompt.rstrip() + "\n\n" + adaptive_section + "\n"
