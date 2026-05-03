"""Failure Pattern Analysis — Phase 6.

Mine run_history.jsonl to compute failure mode distributions, detect dominant
failure patterns, and generate automatic adjustments for the LLM refinement loop.

This complements action_effectiveness.py by focusing on *what* fails rather than
*which actions work*.  When a single failure mode dominates (>40%), the module
produces actionable recommendations and auto-adjustments to steer the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from crabquant.refinement.action_analytics import load_run_history
from crabquant.refinement.action_effectiveness import SKIP_MANDATES

logger = logging.getLogger(__name__)

# Threshold for a failure mode to be considered "dominant"
DOMINANT_THRESHOLD = 0.40

# Known failure modes with their recommendation sets
_RECOMMENDATIONS: dict[str, list[str]] = {
    "too_few_trades": [
        "Lower the minimum trade threshold (e.g. accept 8+ trades instead of 20+).",
        "Use shorter lookback windows or faster signal frequencies to increase entries.",
        "Add complementary entry signals so the strategy triggers more often.",
        "Consider momentum or volatility breakout overlays to boost trade count.",
    ],
    "low_sharpe": [
        "Focus on quality over quantity — remove weak signals even if it cuts trade count.",
        "Simplify the strategy: fewer parameters often generalize better.",
        "Check for look-ahead bias or overly fitted thresholds.",
        "Add a trend filter to avoid mean-reversion signals in trending markets.",
    ],
    "regime_fragility": [
        "Add explicit regime detection (trend/mean-reversion/volatile) and adjust parameters per regime.",
        "Use adaptive lookback windows that shorten in volatile markets.",
        "Consider multi-regime strategies that switch behaviour based on market state.",
        "Add a volatility regime filter to avoid trading in unfavourable conditions.",
    ],
    "excessive_drawdown": [
        "Add a stop-loss or trailing stop to cap individual trade losses.",
        "Reduce position sizing during high-volatility periods.",
        "Implement a drawdown circuit-breaker that pauses trading after X% drop.",
        "Add a maximum portfolio heat constraint (aggregate position risk).",
    ],
    "backtest_crash": [
        "Check for division-by-zero or missing-data issues in indicator calculations.",
        "Add defensive guards around price access (e.g. .dropna(), bounds checks).",
        "Ensure the strategy handles edge cases like empty DataFrames after filtering.",
    ],
    "module_load_failed": [
        "Verify all imports are available in the target environment.",
        "Avoid dynamic imports or optional dependencies that may not be installed.",
    ],
    "timeout": [
        "Reduce computational complexity — vectorize loops, precompute indicators.",
        "Limit the backtest date range during iteration to speed up feedback.",
    ],
}

# Fill in a generic recommendation for any mode not explicitly listed
_GENERIC_RECOMMENDATIONS = [
    "Review recent changes for regressions.",
    "Try a full_rewrite to start from a cleaner base.",
    "Check logs for additional diagnostic information.",
]


def _recommendations_for_mode(mode: str) -> list[str]:
    """Return recommendation list for a given failure mode."""
    return _RECOMMENDATIONS.get(mode, _GENERIC_RECOMMENDATIONS.copy())


def analyze_failure_patterns(
    history: list[dict],
    window: int = 100,
) -> dict[str, Any]:
    """Analyse failure mode distribution and detect patterns from run history.

    Filters out test/smoke mandates, computes failure-mode percentages,
    detects a dominant mode (>40 %), generates recommendations, and
    compares recent vs older trends.

    Args:
        history: List of history entry dicts (from ``load_run_history``).
        window:  Maximum number of recent entries to consider (default 100).

    Returns:
        Dict with keys:
        - distribution:        {failure_mode: percentage, ...}
        - dominant_mode:       str | None  — the mode exceeding 40 %, if any
        - dominant_pct:        float       — percentage of the dominant mode
        - total_failures:      int         — number of failure entries analysed
        - recommendations:     list[str]   — actionable suggestions
        - recent_trend:        dict        — {"direction": "increasing"|"decreasing"|"stable", "detail": ...}
    """
    # Filter out test/smoke mandates and successful entries
    filtered = [
        entry for entry in history
        if entry.get("mandate", "") not in SKIP_MANDATES
        and entry.get("failure_mode", "")
    ]

    # Apply window
    filtered = filtered[-window:]

    result: dict[str, Any] = {
        "distribution": {},
        "dominant_mode": None,
        "dominant_pct": 0.0,
        "total_failures": 0,
        "recommendations": [],
        "recent_trend": {"direction": "stable", "detail": "No data"},
    }

    total = len(filtered)
    if total == 0:
        return result

    # Count failures per mode
    mode_counts: dict[str, int] = {}
    for entry in filtered:
        mode = entry.get("failure_mode", "")
        if mode:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    # Compute distribution
    distribution: dict[str, float] = {}
    for mode, count in mode_counts.items():
        distribution[mode] = count / total

    result["distribution"] = distribution
    result["total_failures"] = total

    # Detect dominant mode
    dominant_mode = None
    dominant_pct = 0.0
    for mode, pct in distribution.items():
        if pct > dominant_pct:
            dominant_pct = pct
            dominant_mode = mode

    if dominant_pct > DOMINANT_THRESHOLD:
        result["dominant_mode"] = dominant_mode
        result["dominant_pct"] = round(dominant_pct, 4)
        result["recommendations"] = _recommendations_for_mode(dominant_mode)
    else:
        # No single dominant mode — give generic advice
        result["dominant_pct"] = round(dominant_pct, 4)
        result["recommendations"] = _GENERIC_RECOMMENDATIONS.copy()

    # Recent trend: compare last 50 vs previous 50
    half = min(50, total // 2)
    if half >= 2:
        older = filtered[-(2 * half):-half]
        newer = filtered[-half:]

        # Compare dominant mode frequency
        if dominant_mode:
            older_rate = (
                sum(1 for e in older if e.get("failure_mode") == dominant_mode) / len(older)
            )
            newer_rate = (
                sum(1 for e in newer if e.get("failure_mode") == dominant_mode) / len(newer)
            )
            delta = newer_rate - older_rate

            if delta > 0.10:
                direction = "increasing"
                detail = (
                    f"Dominant mode '{dominant_mode}' rose from "
                    f"{older_rate:.0%} to {newer_rate:.0%} (+{delta:.0%})"
                )
            elif delta < -0.10:
                direction = "decreasing"
                detail = (
                    f"Dominant mode '{dominant_mode}' fell from "
                    f"{older_rate:.0%} to {newer_rate:.0%} ({delta:.0%})"
                )
            else:
                direction = "stable"
                detail = (
                    f"Dominant mode '{dominant_mode}' stable at ~{newer_rate:.0%} "
                    f"(was {older_rate:.0%})"
                )
        else:
            direction = "stable"
            detail = "No dominant failure mode detected"

        result["recent_trend"] = {"direction": direction, "detail": detail}

    return result


def get_auto_adjustments(pattern_analysis: dict[str, Any]) -> dict[str, Any]:
    """Generate automatic adjustments based on failure pattern analysis.

    Args:
        pattern_analysis: Output from ``analyze_failure_patterns``.

    Returns:
        Dict with keys:
        - prompt_hints:            list[str]  — hints to inject into the LLM prompt
        - threshold_adjustments:   dict       — parameter tweaks
        - priority_actions:        list[str]  — suggested high-priority actions
    """
    result: dict[str, Any] = {
        "prompt_hints": [],
        "threshold_adjustments": {},
        "priority_actions": [],
    }

    dominant = pattern_analysis.get("dominant_mode")
    if not dominant:
        return result

    if dominant == "too_few_trades":
        result["prompt_hints"] = [
            "Focus on increasing trade frequency — use shorter lookbacks and more sensitive entry signals.",
            "Accept a lower minimum trade count as a starting point; optimise quality later.",
        ]
        result["threshold_adjustments"] = {
            "min_trades_hint": 8,
            "entry_sensitivity": "high",
        }
        result["priority_actions"] = [
            "change_entry_logic",
            "add_complementary_signals",
        ]

    elif dominant == "low_sharpe":
        result["prompt_hints"] = [
            "Prioritise signal quality over trade count — remove noisy entries.",
            "Try simpler strategies with fewer parameters to reduce overfitting.",
        ]
        result["threshold_adjustments"] = {
            "strategy_complexity": "low",
            "quality_over_quantity": True,
        }
        result["priority_actions"] = [
            "simplify_strategy",
            "remove_weak_signals",
        ]

    elif dominant == "regime_fragility":
        result["prompt_hints"] = [
            "Add explicit regime detection to adapt behaviour to market conditions.",
            "Consider a multi-regime strategy that uses different logic per regime.",
        ]
        result["threshold_adjustments"] = {
            "regime_aware": True,
            "adaptive_lookback": True,
        }
        result["priority_actions"] = [
            "add_regime_filter",
            "add_regime_switching",
        ]

    elif dominant == "excessive_drawdown":
        result["prompt_hints"] = [
            "Implement risk management controls — stop-loss, position sizing, drawdown circuit-breaker.",
            "Reduce exposure during volatile periods.",
        ]
        result["threshold_adjustments"] = {
            "max_drawdown_pct": 15.0,
            "stop_loss_pct": 3.0,
        }
        result["priority_actions"] = [
            "add_stop_loss",
            "add_drawdown_circuit_breaker",
        ]

    else:
        # Generic fallback for unknown dominant modes
        result["prompt_hints"] = [
            f"The dominant failure mode is '{dominant}'. Review recent changes for regressions.",
            "Consider a full_rewrite to start from a cleaner base.",
        ]
        result["priority_actions"] = ["full_rewrite"]

    return result


def format_failure_patterns_for_prompt(pattern_analysis: dict[str, Any]) -> str:
    """Format failure pattern analysis for LLM context injection.

    Args:
        pattern_analysis: Output from ``analyze_failure_patterns``.

    Returns:
        Formatted string suitable for inclusion in an LLM prompt.
        Returns empty string if there are no failures to report.
    """
    total = pattern_analysis.get("total_failures", 0)
    if total == 0:
        return ""

    lines: list[str] = [
        "### Failure Pattern Analysis",
        f"Analysed {total} failure entries:",
    ]

    # Distribution
    dist = pattern_analysis.get("distribution", {})
    if dist:
        sorted_modes = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        for mode, pct in sorted_modes:
            bar_len = int(pct * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"- {mode}: {pct:.1%} {bar}")

    # Dominant mode
    dominant = pattern_analysis.get("dominant_mode")
    if dominant:
        dom_pct = pattern_analysis.get("dominant_pct", 0)
        lines.append(f"\nDominant failure mode: **{dominant}** ({dom_pct:.1%})")

    # Recommendations
    recs = pattern_analysis.get("recommendations", [])
    if recs:
        lines.append("\nRecommendations:")
        for rec in recs:
            lines.append(f"- {rec}")

    # Trend
    trend = pattern_analysis.get("recent_trend", {})
    if trend.get("direction") and trend.get("direction") != "stable":
        lines.append(f"\nTrend: {trend['detail']}")

    return "\n".join(lines)
