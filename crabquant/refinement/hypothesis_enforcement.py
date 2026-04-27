"""
Hypothesis enforcement — validate LLM output contains a non-trivial causal hypothesis.

Rejects:
- Missing/empty hypothesis
- Too-short hypotheses (< 20 chars)
- Generic patterns like "improve performance", "adjust parameters", "optimize"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Generic patterns that indicate a lazy/non-causal hypothesis ─────────────

GENERIC_PATTERNS: list[str] = [
    "improve performance",
    "improve the strategy",
    "adjust parameter",
    "optimize the strategy",
    "optimize strategy",
    "make it better",
    "increase returns",
    "reduce losses",
    "tweak settings",
    "fine-tune",
    "improve results",
    "get better results",
    "improve sharpe",
    "improve profitability",
]

# Pre-compile for efficiency
_GENERIC_RE = re.compile(
    "|".join(re.escape(p) for p in GENERIC_PATTERNS),
    re.IGNORECASE,
)

# Minimum hypothesis length (characters)
_MIN_LENGTH = 20


@dataclass
class HypothesisCheckResult:
    """Result of hypothesis validation."""

    valid: bool
    reason: str  # Empty when valid=True; explanation when valid=False


def check_hypothesis(
    hypothesis: str | None,
    *,
    min_length: int = _MIN_LENGTH,
    extra_generic_patterns: list[str] | None = None,
) -> HypothesisCheckResult:
    """Validate that a hypothesis is non-trivial and causal.

    Args:
        hypothesis: The hypothesis string from LLM output. Can be None.
        min_length: Minimum character length for the hypothesis.
        extra_generic_patterns: Additional generic patterns to reject.

    Returns:
        HypothesisCheckResult with valid/reason.
    """
    # Missing / empty
    if not hypothesis or not hypothesis.strip():
        return HypothesisCheckResult(
            valid=False,
            reason="Hypothesis is missing or empty",
        )

    hypothesis = hypothesis.strip()

    # Too short
    if len(hypothesis) < min_length:
        return HypothesisCheckResult(
            valid=False,
            reason=f"Hypothesis too short ({len(hypothesis)} chars, minimum {min_length})",
        )

    # Check generic patterns
    pattern = _GENERIC_RE
    if extra_generic_patterns:
        extras = re.compile(
            "|".join(re.escape(p) for p in extra_generic_patterns),
            re.IGNORECASE,
        )
        if extras.search(hypothesis):
            return HypothesisCheckResult(
                valid=False,
                reason="Hypothesis is too generic — must state a specific causal mechanism",
            )

    if pattern.search(hypothesis):
        return HypothesisCheckResult(
            valid=False,
            reason="Hypothesis is too generic — must state a specific causal mechanism",
        )

    return HypothesisCheckResult(valid=True, reason="")


def check_hypothesis_from_modification(
    modification: dict,
    **kwargs,
) -> HypothesisCheckResult:
    """Convenience: extract hypothesis from a StrategyModification-like dict and validate.

    Args:
        modification: Dict with at least a "hypothesis" key.
        **kwargs: Forwarded to check_hypothesis().

    Returns:
        HypothesisCheckResult.
    """
    hypothesis = modification.get("hypothesis") if modification else None
    return check_hypothesis(hypothesis, **kwargs)
